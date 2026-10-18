"""pyGSM double-ended growing string on the harness potential.

The pin is ZimmermanGroup/pyGSM commit 1b4d5a55. The import version
string ends in ``dirty`` because versioningit rewrites ``_version.py``
while building, and the repository has no release tag. The level of
theory is the harness kernel. Endpoints stay the archived pair.

LJ38 has no covalent graph. The coordinate system is the union of
endpoint contacts inside 1.35, plus translation and rotation. That
cutoff sits in the flat first-neighbour shell. Angles and dihedrals
are left out: a covalent-radius topology builds tens of thousands of
primitives. Delocalized internals of the contact set span the
Cartesian space, which is what eigenvector following requires.

``nnodes`` is the problem's interior image count plus the two ends.
``DMAX`` is the problem ``max_move``. The Cartesian force norm and the
call cap are the problem stop. ``ADD_NODE_TOL`` and the three
optimizer steps per cycle are the DE-GSM command-line defaults.
``CONV_TOL`` is the force norm divided by ``sqrt(3N)``, so the
delocalized RMS target matches that Cartesian norm. Numpy 2 removed
``ndarray.tostring`` and rejects a generator passed to ``np.sum``.
The rotation eigensolve can also fail to converge on these clusters.
The shims below restore those calls. They run in the worker.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import shutil
import statistics
import tempfile
import textwrap
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .counter import ForceCallCap, HarnessCounter
from .failures import classify_failure
from .pes import evaluate, load_pairs
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance, write_starts

_CONTACT_CUTOFF = 1.35
_ADD_NODE_TOL = 0.01
_MAX_OPT_STEPS = 3
_EIGEN_TOL = 1e-6
_FD_STEP = 1e-5
_SYMBOL = "Ar"
_PATCHED = False
_NP_SUM = np.sum
_NP_DOT = np.dot


def _threads_one() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _dot_scalar(left, right, *args, **kwargs):
    result = _NP_DOT(left, right, *args, **kwargs)
    array = np.asarray(result)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return result


def _sum_generators(values, *args, **kwargs):
    if not inspect.isgenerator(values):
        return _NP_SUM(values, *args, **kwargs)
    items = list(values)
    if not items:
        return 0
    stacked = np.stack([np.asarray(item) for item in items], axis=0)
    if args or kwargs:
        return _NP_SUM(stacked, *args, **kwargs)
    return _NP_SUM(stacked, axis=0)


def _patch_wilson(cls) -> None:
    original = cls.wilsonB
    source = textwrap.dedent(inspect.getsource(original)).replace(".tostring()", ".tobytes()")
    namespace: dict = {}
    exec(source, original.__globals__, namespace)
    cls.wilsonB = namespace["wilsonB"]


def _stable_sorted_eigh(original):
    def sorted_eigh(mat, b=None, asc=False):
        matrix = np.array(mat, dtype=np.float64, copy=True)
        matrix = 0.5 * (matrix + matrix.T)
        try:
            return original(matrix, b=b, asc=asc)
        except np.linalg.LinAlgError:
            matrix = matrix + np.eye(matrix.shape[0]) * 1e-10
            return original(matrix, b=b, asc=asc)

    return sorted_eigh


def ensure_runtime() -> None:
    """Import pyGSM and apply the Numpy 2 shims once in this process."""
    global _PATCHED
    if _PATCHED:
        return
    from pyGSM.coordinate_systems import PrimitiveInternalCoordinates, rotate
    from pyGSM.coordinate_systems.internal_coordinates import InternalCoordinates

    _patch_wilson(PrimitiveInternalCoordinates)
    _patch_wilson(InternalCoordinates)
    rotate.sorted_eigh = _stable_sorted_eigh(rotate.sorted_eigh)
    np.dot = _dot_scalar
    np.sum = _sum_generators
    _PATCHED = True


def contact_pairs(coords: np.ndarray, cutoff: float) -> list[tuple[int, int]]:
    """Undirected pairs whose distance is below ``cutoff``."""
    points = np.asarray(coords, dtype=np.float64)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    left, right = np.triu_indices(len(points), k=1)
    return [
        (int(i), int(j))
        for i, j, gap in zip(left, right, dist[left, right])
        if gap < cutoff
    ]


def _geom(coords: np.ndarray, symbol: str):
    return [(symbol, float(row[0]), float(row[1]), float(row[2])) for row in coords]


def _distance_tric(reactant: np.ndarray, product: np.ndarray, atoms, cutoff: float):
    from pyGSM.coordinate_systems import DelocalizedInternalCoordinates, PrimitiveInternalCoordinates
    from pyGSM.coordinate_systems.slots import (
        Distance,
        RotationA,
        RotationB,
        RotationC,
        TranslationX,
        TranslationY,
        TranslationZ,
    )
    from pyGSM.coordinate_systems.topology import MyG

    bonds = sorted(set(contact_pairs(reactant, cutoff)) | set(contact_pairs(product, cutoff)))
    graph = MyG()
    for index in range(len(reactant)):
        graph.add_node(index)
    for left, right in bonds:
        graph.add_edge(left, right)
    prim = PrimitiveInternalCoordinates.from_options(
        xyz=reactant,
        atoms=atoms,
        topology=graph,
        connect=False,
        addtr=True,
        addcart=False,
        form_topology=False,
    )
    nodes = list(range(len(reactant)))
    weight = np.ones(len(nodes)) / len(nodes)
    prim.Internals = [
        TranslationX(nodes, weight),
        TranslationY(nodes, weight),
        TranslationZ(nodes, weight),
    ]
    flat = np.asarray(reactant, dtype=np.float64).reshape(-1)
    centered = reactant - reactant.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    prim.Internals.extend(
        [
            RotationA(nodes, flat, prim.Rotators, w=radius),
            RotationB(nodes, flat, prim.Rotators, w=radius),
            RotationC(nodes, flat, prim.Rotators, w=radius),
        ]
    )
    prim.Internals.extend(Distance(left, right) for left, right in bonds)
    count = len(prim.Internals)
    prim.block_info = [(0, len(reactant), 0, count, "reg")]
    prim.prim_only_block_info = list(prim.block_info)
    prim.fragments = [graph]
    return DelocalizedInternalCoordinates.from_options(
        xyz=reactant,
        atoms=atoms,
        connect=False,
        addtr=True,
        addcart=False,
        primitives=prim,
    )


def _lot_class():
    from pyGSM.level_of_theories.base_lot import Lot
    from pyGSM.utilities import manage_xyz, units

    class HarnessLot(Lot):
        def __init__(self, options, evaluate_fn, counter: HarnessCounter):
            super().__init__(options)
            self._evaluate = evaluate_fn
            self.counter = counter
            self.last_coords = None
            self.last_energy = None
            self.last_gradient = None

        @classmethod
        def from_options(cls, evaluate_fn, counter, **kwargs):
            return cls(cls.default_options().set_values(kwargs), evaluate_fn, counter)

        @classmethod
        def copy(cls, lot, options=None, copy_wavefunction=True):
            return cls(lot.options.copy().set_values(options or {}), lot._evaluate, lot.counter)

        def run(self, geom, mult, ad_idx, runtype="gradient"):
            coords = np.asarray(manage_xyz.xyz_to_np(geom), dtype=np.float64)
            flat = coords.reshape(-1)
            key = tuple(float(value) for value in flat)
            fresh, cached = self.counter.charge(key)
            if not fresh and cached is not None:
                energy, gradient = cached
            else:
                energy, grad = self._evaluate(coords)
                gradient = tuple(float(value) for value in np.asarray(grad, dtype=np.float64).reshape(-1))
                energy = float(energy)
                self.counter.remember(key, (energy, gradient))
            grad = np.asarray(gradient, dtype=np.float64).reshape(coords.shape)
            self.last_coords = np.array(coords, dtype=np.float64, copy=True)
            self.last_energy = float(energy)
            self.last_gradient = np.array(grad, dtype=np.float64, copy=True)
            # pyGSM reports energy in kcal/mol and builds the quadratic
            # model as if the gradient were Hartree per angstrom. Storing
            # the harness values as Hartree, with the Bohr factor removed
            # from the gradient, makes that conversion apply to both.
            self._Energies[(mult, ad_idx)] = self.Energy(energy, "Hartree")
            scale = 1.0 / float(units.ANGSTROM_TO_AU)
            self._Gradients[(mult, ad_idx)] = self.Gradient(grad * scale, "Hartree/Bohr")
            self.hasRanForCurrentCoords = True

    return HarnessLot


def _build(reactant, product, counter, evaluate_fn, *, nnodes: int, force_norm: float, max_move: float):
    from pyGSM.growing_string_methods import DE_GSM
    from pyGSM.molecule import Molecule
    from pyGSM.optimizers import eigenvector_follow
    from pyGSM.potential_energy_surfaces import PES
    from pyGSM.utilities import elements

    lot_cls = _lot_class()
    lot = lot_cls.from_options(
        evaluate_fn,
        counter,
        geom=_geom(reactant, _SYMBOL),
        states=[(1, 0)],
        ID=0,
    )
    pes = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)
    table = elements.ElementData()
    atoms = [table.from_symbol(_SYMBOL) for _ in range(len(reactant))]
    coord = _distance_tric(reactant, product, atoms, _CONTACT_CUTOFF)
    molecule_r = Molecule.from_options(
        geom=_geom(reactant, _SYMBOL),
        PES=pes,
        coord_obj=coord,
        Form_Hessian=True,
        node_id=0,
    )
    molecule_p = Molecule.copy_from_options(
        molecule_r,
        xyz=product,
        new_node_id=nnodes - 1,
        copy_wavefunction=False,
    )
    optimizer = eigenvector_follow.from_options(
        print_level=0,
        Linesearch="NoLineSearch",
        update_hess_in_bg=True,
        conv_Ediff=100.0,
        conv_dE=1.0,
        conv_gmax=100.0,
        DMAX=float(max_move),
    )
    return DE_GSM.from_options(
        reactant=molecule_r,
        product=molecule_p,
        nnodes=int(nnodes),
        CONV_TOL=float(force_norm) / np.sqrt(3 * len(reactant)),
        CONV_gmax=100.0,
        CONV_Ediff=100.0,
        CONV_dE=1.0,
        ADD_NODE_TOL=_ADD_NODE_TOL,
        growth_direction=0,
        optimizer=optimizer,
        ID=0,
        print_level=0,
        mp_cores=1,
        interp_method="DLC",
    )


def _stored(node) -> tuple[float, np.ndarray, np.ndarray] | None:
    lot = node.PES.lot
    coords = getattr(lot, "last_coords", None)
    grad = getattr(lot, "last_gradient", None)
    energy = getattr(lot, "last_energy", None)
    if coords is None or grad is None or energy is None:
        return None
    xyz = np.asarray(node.xyz, dtype=np.float64)
    if xyz.shape != np.asarray(coords).shape:
        return None
    if not np.allclose(xyz, coords, rtol=0.0, atol=1e-12):
        return None
    return float(energy), np.asarray(grad, dtype=np.float64), np.asarray(coords, dtype=np.float64)


def _candidate(gsm) -> dict | None:
    best = None
    nodes = list(gsm.nodes)
    for index, node in enumerate(nodes):
        if node is None or index == 0 or index == len(nodes) - 1:
            continue
        stored = _stored(node)
        if stored is None:
            continue
        energy, grad, coords = stored
        if best is None or energy > best["energy"]:
            best = {
                "index": index,
                "energy": energy,
                "gradient": grad,
                "coords": coords,
            }
    return best


def _judge(
    name: str,
    counter: HarnessCounter,
    candidate: dict | None,
    *,
    force_norm: float,
    cap: int,
    project_rigid: bool,
    crashed: str,
    potential: str,
) -> dict:
    reasons: list[str] = []
    if crashed:
        reasons.append("crash")
    if counter.calls >= cap:
        reasons.append("cap")
    energy = float("nan")
    force = float("nan")
    coords = None
    if candidate is not None:
        energy = float(candidate["energy"])
        coords = np.asarray(candidate["coords"], dtype=np.float64)
        force = float(np.linalg.norm(np.asarray(candidate["gradient"], dtype=np.float64).reshape(-1)))
    if not np.isfinite(force) or force > force_norm:
        reasons.append("force_norm")
    hessian_index = None
    if "crash" not in reasons and "cap" not in reasons and "force_norm" not in reasons and coords is not None:
        shape = coords.shape

        def gradient(point):
            flat = np.asarray(point, dtype=np.float64).reshape(-1)
            key = tuple(float(value) for value in flat)
            fresh, cached = counter.charge(key)
            if not fresh and cached is not None:
                return cached[1]
            energy, grad = evaluate(potential, flat.reshape(shape), None)
            stored = tuple(float(value) for value in np.asarray(grad, dtype=np.float64).reshape(-1))
            counter.remember(key, (float(energy), stored))
            return stored

        with counter.verification():
            hessian = numerical_hessian(gradient, coords.reshape(-1).tolist(), step=_FD_STEP)
        hessian_index = vibrational_index(
            np.asarray(hessian, dtype=np.float64),
            coords.reshape((-1, 3)),
            project_rigid,
        )
        if hessian_index != 1:
            reasons.append("hessian_index")
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(counter.calls),
        "verify_calls": int(counter.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if not np.isfinite(energy) else energy,
        "force_norm": None if not np.isfinite(force) else force,
        "hessian_index": hessian_index,
        "converged": failure is None,
        "error": crashed or None,
        "grown": candidate is not None,
    }


def search_pair(
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    *,
    force_norm: float,
    cap: int,
    max_move: float,
    n_interior: int,
    project_rigid: bool,
    potential: str = "lj",
) -> dict:
    """One double-ended growing string. The call cap is the problem cap."""
    _threads_one()
    ensure_runtime()
    reactant = np.asarray(reactant, dtype=np.float64)
    product = np.asarray(product, dtype=np.float64)
    counter = HarnessCounter(cache=True, cap=int(cap))

    def evaluate_fn(coords, _box=None):
        return evaluate(potential, np.asarray(coords, dtype=np.float64), None)

    crashed = ""
    gsm = None
    work = Path(tempfile.mkdtemp(prefix="pygsm-"))
    previous = Path.cwd()
    try:
        os.chdir(work)
        with Path(os.devnull).open("w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            import logging

            for handler in logging.getLogger("NiftyLogger").handlers:
                handler.stream = sink
            try:
                gsm = _build(
                    reactant,
                    product,
                    counter,
                    evaluate_fn,
                    nnodes=int(n_interior) + 2,
                    force_norm=float(force_norm),
                    max_move=float(max_move),
                )
                gsm.set_V0()
                gsm.add_GSM_nodes(2)
                gsm.grow_string(max_iters=1_000_000, max_opt_steps=_MAX_OPT_STEPS)
                if gsm.nodes and all(node is not None for node in gsm.nodes):
                    gsm.done_growing = True
                    gsm.reparameterize()
                    gsm.optimize_string(max_iter=1_000_000, opt_steps=_MAX_OPT_STEPS, rtype=2)
            except ForceCallCap:
                pass
            except Exception as exc:
                crashed = type(exc).__name__
                traceback.print_exc()
    finally:
        os.chdir(previous)
        shutil.rmtree(work, ignore_errors=True)
    candidate = _candidate(gsm) if gsm is not None else None
    record = _judge(
        name,
        counter,
        candidate,
        force_norm=float(force_norm),
        cap=int(cap),
        project_rigid=project_rigid,
        crashed=crashed,
        potential=potential,
    )
    return record


def _registry(root: Path) -> dict:
    import tomllib

    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == "pygsm":
            return row
    raise ValueError("codes/registry.toml has no pygsm entry")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the pygsm registry")
    band = problem["band"]
    convergence = problem["convergence"]
    return {
        "potential": band["potential"],
        "n_interior": int(band["interior_images"]),
        "max_move": float(band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": problem["geometry"]["kind"],
        "contact_cutoff": _CONTACT_CUTOFF,
        "add_node_tol": _ADD_NODE_TOL,
        "max_opt_steps": _MAX_OPT_STEPS,
        "coordinate": "distance-tric",
    }


def _job(pair, settings: dict) -> dict:
    free = np.asarray(pair.free, dtype=bool)
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64),
        "product": np.asarray(pair.product, dtype=np.float64),
        "project_rigid": bool(np.all(free) and int(np.count_nonzero(free)) >= 2 and settings["potential"] == "lj"),
        "settings": settings,
    }


def _execute(job: dict) -> dict:
    settings = job["settings"]
    return search_pair(
        job["name"],
        job["reactant"],
        job["product"],
        force_norm=settings["force_norm"],
        cap=settings["cap"],
        max_move=settings["max_move"],
        n_interior=settings["n_interior"],
        project_rigid=bool(job["project_rigid"]),
        potential=str(settings["potential"]),
    )


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, problem: dict, score: dict) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness",
        "# force_calls count raw potential evaluations and are not divided by the image count",
        "# the mean is round-half-even over successes; a failure is outside that mean",
        f"n {len(records)}",
        f"successes {len(calls)}",
        f"nfailed {len(records) - len(calls)}",
    ]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        images = int(settings["n_interior"])
        if images > 0:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
            lines.append("call_unit total")
    lines.append(f"par2_mean {score['par2_mean']:.10f}")
    lines.append(f"par2_median {score['median']:.10f}")
    lines.append(f"wilson_low {score['wilson_low']:.10f}")
    lines.append(f"wilson_high {score['wilson_high']:.10f}")
    for name in ("cap", "wrong_stationary", "divergence", "crash"):
        lines.append(f"n_{name} {score[f'n_{name}']}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append(f"problem {problem['id']}")
    lines.append(f"images {settings['n_interior']}")
    lines.append(
        "note pyGSM DE-GSM. The harness Lennard-Jones kernel is the level of theory. "
        f"{settings['n_interior'] + 2} nodes. Distance-plus-rigid coordinates, contact cutoff "
        f"{settings['contact_cutoff']}. Endpoints fixed. DMAX {settings['max_move']}. "
        f"Call cap {settings['cap']}."
    )
    return "\n".join(lines) + "\n"


def write_outputs(output: Path, records: list[dict], template: dict, settings: dict, problem: dict) -> dict:
    """Write the dat, the per-start file, and the v2 score."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored = [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records]
    score = score_v2(scored, settings["cap"])
    output.write_text(_dat(records, template, settings, problem, score), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    parameters = dict(settings)
    parameters["counter"] = "harness"
    parameters["seed"] = "0"
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score,
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": score.get("success_only_mean"),
        "nfailed": score["nfailed"],
        "starts": score["n"],
        "successes": score["n"] - score["nfailed"],
        "par2_mean": score["par2_mean"],
        "output": str(output),
    }


_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    key = (str(Path(root).resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        loaded = load_pairs(
            geometry_archive(root, problem),
            settings["geometry"],
            settings["pairs"],
        )
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def search_named(root: Path, problem_name: str, template_name: str, start: str, seed: str = "0") -> dict:
    """One archived pair. The seed is recorded. The string does not draw random numbers."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(pairs[start], settings))
    record["seed"] = str(seed)
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def _load_resume(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    resume: bool = False,
    limit: int = 0,
) -> dict:
    """Run the archived pairs and write the dat, starts, and v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != "pygsm-band":
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem, settings).values())
    if limit > 0:
        pairs = pairs[: int(limit)]
    jobs = [_job(pair, settings) for pair in pairs]
    done: list[dict] = []
    if resume:
        done = _load_resume(Path(output).with_suffix(".starts.jsonl"))
    have = {str(row["start"]) for row in done}
    pending = [job for job in jobs if job["name"] not in have]
    order = {job["name"]: index for index, job in enumerate(jobs)}

    def publish(records: list[dict]) -> dict:
        ranked = [row for row in records if row["start"] in order]
        ranked.sort(key=lambda row: order[row["start"]])
        for record in ranked:
            record["seed"] = "0"
            record["problem"] = problem["id"]
            record["template"] = template["id"]
            record["code"] = template["code"]
            record["code_version"] = template["code_version"]
            record["commit"] = template["commit"]
            record["algorithm"] = template["algorithm"]
            record["tuning"] = template["tuning"]
        return write_outputs(output, ranked, template, settings, problem)

    if not pending and done:
        return publish(done)
    _threads_one()
    records = list(done)
    if workers <= 1:
        for job in pending:
            records.append(_execute(job))
            publish(records)
    else:
        import multiprocessing
        import sys

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_execute, job): job["name"] for job in pending}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                summary = publish(records)
                print(
                    f"{record['start']} force_calls={record['force_calls']} "
                    f"failure={record['failure']} n={summary['starts']}",
                    file=sys.stderr,
                    flush=True,
                )
    return publish(records)
