"""pyKMC event search through ``partn_search``.

The pin is hugomoison/pyKMC, not SourceForge lkmc. One search is
``pykmc.enginemanager.lmpi.lammps_operations.partn_search`` in the
pyKMC virtualenv. LAMMPS holds the free atoms. ``pair_style zero``
contributes nothing. ``fix external pf/callback`` evaluates the spec
potential and ``HarnessCounter`` counts those calls.

``partn_search`` leaves ``converge_property`` unset, so pARTn would
test the maximum force component. The worker sets ``norm`` and
``forc_thr`` to the problem force stop when that parameter is written.
``ninit`` is 0 on a transition-state guess. A search that starts at a
minimum uses the published k-ART push: 0.4 Å on the atom and its
neighbors inside 3 Å, with the Pt(111) example's step sizes.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from optbench.connection import link_tolerance
from optbench.pykmc_worker import (
    ARTN_COMMIT,
    IRA_COMMIT,
    LAMMPS_COMMIT,
    PYKMC_COMMIT,
    library_path,
    python_path,
    stack_status,
)

from .common import judge_saddle, pair_seed, registry_pin, require_pin, write_campaign
from .geometries import load_starts

ENGINE = "pykmc"
CODE = "pykmc"
REPOSITORY = "https://github.com/hugomoison/pyKMC"
RELEASE = "0.0.1"
_SEED_BASE = 20261015
# On these slabs a one-atom hop is about 2.7 A, and the mean over every
# free atom stays under 0.5. The protruding atoms sit more than 1.5 A
# above the next layer: the Pt heptamer, or the Al adatom.
_TOP_WINDOW = 1.5
_STATUS: str | None = None
_STATUS_READY = False
_EAM_LOCK = threading.Lock()


def available() -> str | None:
    global _STATUS, _STATUS_READY
    if not _STATUS_READY:
        _STATUS = stack_status()
        _STATUS_READY = True
    return _STATUS


def _section(template: dict) -> dict:
    """pARTn knobs. A template value wins.

    The defaults are the published minimum-search profile. Béland,
    Mousseau, and coauthors, Comput. Mater. Sci. (2015),
    doi:10.1016/j.commatsci.2014.11.047, deform one atom and its
    surroundings by 0.2–0.5 Å and then take one to three perpendicular
    steps. Lanczos there costs about 12–15 force evaluations. The
    pinned artn-plugin Pt(111) heptamer, ``examples/LAMMPS.Pt111.d/artn.in``,
    uses ninit 1, nsmooth 2, nnewchance 2, eigen_step_size 0.25,
    eigval_thr -0.05, lanczos_disp 1e-4, and push_over 1.0, with
    ``dmax`` 5 in ``lammps.in``. Its forc_thr is 0.01. The harness stop
    stays 0.001. That example pushes along a file, so a catalogue stays
    on ``push_mode rad`` with ``push_dist_thr`` 3, which reaches the Pt
    neighbor at r0 2.897. pyKMC's own default of 1.0 does not.
    """
    section = dict(template.get("pykmc") or {})
    defaults = {
        "ninit": 1,
        "push_mode": "rad",
        "push_step_size": 0.4,
        "push_dist_thr": 3.0,
        "eigen_step_size": 0.25,
        "eigval_thr": -0.05,
        "nsmooth": 2,
        "nnewchance": 2,
        "neigen": 1,
        "nperp": 3,
        "lanczos_min_size": 12,
        "lanczos_max_size": 15,
        "lanczos_disp": 1.0e-4,
        "lanczos_eval_conv_thr": 1.0e-2,
        "push_over": 1.0,
        "dmax": 5.0,
        "forc_thr": 0.001,
        "delr_thr": 1.0e9,
        "converge_property": "norm",
        "evalf_max": 10000,
        "rcut": 6.5,
        "rnei": 3.0,
        "timeout": 1800,
    }
    for key, value in defaults.items():
        section.setdefault(key, value)
    return section


def _env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = library_path() + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYKMC_STACK"] = str(Path(os.environ.get("PYKMC_STACK", "/opt/pykmc-stack")))
    return env


def _invoke(root: Path, job: dict) -> dict:
    python = python_path()
    with tempfile.TemporaryDirectory(prefix="pykmc-ipc-") as folder:
        spec = Path(folder) / "job.json"
        dest = Path(folder) / "result.json"
        spec.write_text(json.dumps(job), encoding="utf-8")
        try:
            proc = subprocess.run(
                [str(python), "-m", "optbench.pykmc_worker", str(spec), str(dest)],
                cwd=folder,
                env=_env(root),
                capture_output=True,
                text=True,
                timeout=int(job.get("timeout") or 1800),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "calls": 0,
                "capped": False,
                "crashed": True,
                "diverged": False,
                "ok": False,
                "message": f"timeout: {exc}",
                "positions": None,
                "artn_energy": None,
            }
        if dest.is_file():
            payload = json.loads(dest.read_text(encoding="utf-8"))
        else:
            tail = (proc.stderr or proc.stdout or "worker wrote no result").strip().splitlines()
            payload = {
                "calls": 0,
                "capped": False,
                "crashed": True,
                "diverged": False,
                "ok": False,
                "message": tail[-1] if tail else "worker wrote no result",
                "positions": None,
                "artn_energy": None,
            }
        if proc.returncode != 0 and not payload.get("message"):
            payload["crashed"] = True
            payload["message"] = (proc.stderr or "worker failed").strip().splitlines()[-1]
        return payload


def _morse_surface(spec: dict, kind: str) -> dict:
    pbc = spec.get("pbc") or [True, True, False]
    return {
        "kind": kind,
        "element": "Pt",
        "pbc": [bool(pbc[0]), bool(pbc[1]), bool(pbc[2])],
        "depth": float(spec.get("D", 0.7102)),
        "alpha": float(spec.get("alpha", 1.6047)),
        "r0": float(spec.get("r0", 2.897)),
        "cutoff": float(spec.get("cutoff", 9.5)),
    }


def _surface(problem: dict, template: dict) -> dict:
    spec = problem.get("potential") or {}
    kernel = str(template.get("kernel") or spec.get("kind") or "")
    # An explicit template kernel wins. The shared Pt problem stays
    # smooth-linear so other codes keep that row.
    if kernel == "morse_shift":
        return _morse_surface(spec, "morse_shift")
    if kernel in {"morse", "morse_pt"} or spec.get("kind") == "morse":
        return _morse_surface(spec, "morse")
    if kernel == "eam":
        return {
            "kind": "eam",
            "element": "Al",
            "pbc": [True, True, True],
            "depth": 0.0,
            "alpha": 0.0,
            "r0": 0.0,
            "cutoff": 6.5,
        }
    raise ValueError(f"{template['id']} has no Morse or EAM kernel")


def kernel_matches(state: dict | None, kind: str) -> bool:
    """True when a checkpoint was produced with this potential.

    A snap written before the kernel was stored is smooth-linear Morse
    or the r2549 EAM. It does not resume a shifted-Morse run.
    """
    if not state:
        return False
    stored = state.get("kernel")
    if stored in (None, ""):
        return kind in {"morse", "eam"}
    return str(stored) == kind


def _diagonal_cell(lengths: np.ndarray) -> np.ndarray:
    cell = np.zeros(9, dtype=np.float64)
    cell[0], cell[4], cell[8] = (float(value) for value in lengths)
    return cell


def _evaluator(surface: dict, box: np.ndarray):
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3].copy()
    if surface["kind"] == "morse":
        from optbench.surfaces import morse_smooth_linear

        pbc = tuple(surface["pbc"])

        def evaluate(coords, _box):
            return morse_smooth_linear(
                np.asarray(coords, dtype=np.float64).reshape(-1, 3),
                depth=float(surface["depth"]),
                alpha=float(surface["alpha"]),
                r0=float(surface["r0"]),
                cutoff=float(surface["cutoff"]),
                cell=lengths,
                pbc=pbc,
            )

        return evaluate
    if surface["kind"] == "morse_shift":
        from optbench.eon_harness import morse_shift

        cell = _diagonal_cell(lengths)

        def evaluate(coords, _box):
            # r2549 Morse::force returns forces. The harness wants dE/dx.
            flat = np.asarray(coords, dtype=np.float64).reshape(-1)
            energy, forces = morse_shift(flat, None, cell)
            gradient = -np.asarray(forces, dtype=np.float64).reshape(-1, 3)
            return float(energy), gradient

        return evaluate
    from optbench.eon_harness import eam_al

    cell = _diagonal_cell(lengths)

    def evaluate(coords, _box):
        # The Fortran library is one global object. Parallel searches share it.
        with _EAM_LOCK:
            energy, forces = eam_al(np.asarray(coords, dtype=np.float64).reshape(-1, 3), None, cell)
            gradient = -np.asarray(forces, dtype=np.float64).reshape(-1, 3)
            return float(energy), gradient

    return evaluate


class _Calls:
    def __init__(self, calls: int):
        self.calls = int(calls)
        self.verify_calls = 0

    def verification(self):
        from contextlib import nullcontext

        return nullcontext()


def _stored_text(root: Path, archive_path: str) -> str:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return (root / row["stored"]).read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(archive_path)


def _read_geometry(root: Path, archive_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from optbench.pes import read_con

    positions, free, box = read_con(_stored_text(root, archive_path))
    if box is None:
        raise ValueError(f"{archive_path} has no orthorhombic box")
    return positions, free, np.asarray(box, dtype=np.float64).reshape(-1)[:3]


def _cap(problem: dict, template: dict) -> int:
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    if stated not in (None, ""):
        return int(stated)
    if template.get("failure_cap") not in (None, ""):
        return int(template["failure_cap"])
    raise ValueError(f"{problem['id']} has no force-call cap")


def _force(problem: dict, template: dict) -> float:
    convergence = problem.get("convergence") or {}
    stated = convergence.get("force_norm", convergence.get("gradient_norm"))
    if stated not in (None, ""):
        return float(stated)
    return float(_section(template)["forc_thr"])


def _base_job(name: str, positions, free, box, surface: dict, section: dict, *, cap: int, central: int, zseed: int) -> dict:
    return {
        "name": name,
        "positions": np.asarray(positions, dtype=np.float64).reshape(-1).tolist(),
        "free": np.asarray(free, dtype=bool).reshape(-1).tolist(),
        "box": np.asarray(box, dtype=np.float64).reshape(-1)[:3].tolist(),
        "pbc": list(surface["pbc"]),
        "kind": surface["kind"],
        "element": surface["element"],
        "depth": surface["depth"],
        "alpha": surface["alpha"],
        "r0": surface["r0"],
        "cutoff": surface["cutoff"],
        "cap": int(cap),
        "central": int(central),
        "zseed": int(zseed),
        "ninit": int(section["ninit"]),
        "push_mode": str(section["push_mode"]),
        "push_step_size": float(section["push_step_size"]),
        "push_dist_thr": float(section["push_dist_thr"]),
        "eigen_step_size": float(section["eigen_step_size"]),
        "eigval_thr": float(section["eigval_thr"]),
        "lanczos_max_size": int(section["lanczos_max_size"]),
        "lanczos_min_size": int(section["lanczos_min_size"]),
        "lanczos_disp": float(section["lanczos_disp"]),
        "lanczos_eval_conv_thr": float(section["lanczos_eval_conv_thr"]),
        "nsmooth": int(section["nsmooth"]),
        "nnewchance": int(section["nnewchance"]),
        "neigen": int(section["neigen"]),
        "nperp": int(section["nperp"]),
        "push_over": float(section["push_over"]),
        "dmax": float(section["dmax"]),
        "forc_thr": float(section["forc_thr"]),
        "delr_thr": float(section["delr_thr"]),
        "converge_property": str(section["converge_property"]),
        "evalf_max": int(min(int(section["evalf_max"]), int(cap))),
        "rcut": float(section["rcut"]),
        "rnei": float(section["rnei"]),
        "timeout": int(section["timeout"]),
        "engine_id": 0,
    }


def _prefix(payload: dict) -> list[str]:
    if payload.get("capped"):
        return ["cap"]
    if payload.get("diverged"):
        return ["divergence"]
    if payload.get("crashed"):
        return ["crash"]
    return []


def _judge(name: str, payload: dict, *, cap: int, force_norm: float, kind: str, free, box, evaluator, calls: int) -> dict:
    point = None
    if payload.get("positions"):
        point = np.asarray(payload["positions"], dtype=np.float64).reshape(-1, 3)
    gradient = None
    energy = None
    if point is not None and np.all(np.isfinite(point)):
        energy, gradient = evaluator(point, box)
        energy = float(energy)
        gradient = np.asarray(gradient, dtype=np.float64)
    record = judge_saddle(
        name,
        point,
        energy,
        gradient,
        counter=_Calls(calls),
        cap=cap,
        force_norm=force_norm,
        kind=kind,
        free=np.asarray(free, dtype=bool),
        box=box,
        prefix=_prefix(payload),
        keep_mode=True,
        evaluator=evaluator,
    )
    if point is not None:
        record["positions"] = np.asarray(point, dtype=np.float64)
    if payload.get("message"):
        record["error"] = str(payload["message"])
    record["artn_ok"] = bool(payload.get("ok"))
    return record


def _minimize_free(evaluator, positions, free, box, force_tol: float, budget: int):
    from optbench.rgsaddle_minmode import HarnessSurface, _lbfgs_minimize

    free = np.asarray(free, dtype=bool)

    def call(point):
        full = np.array(positions, dtype=np.float64, copy=True)
        full[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
        energy, grad = evaluator(full, box)
        return float(energy), np.asarray(grad, dtype=np.float64)[free].reshape(-1)

    surface = HarnessSurface(call)
    point, ok = _lbfgs_minimize(surface, positions[free].reshape(-1).copy(), force_tol, budget)
    out = np.array(positions, dtype=np.float64, copy=True)
    out[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    energy, _gradient = evaluator(out, box)
    return out, bool(ok), int(surface.calls), float(energy)


def _rmsd(moved: np.ndarray, reference: np.ndarray, lengths=None, pbc=None) -> float:
    delta = np.asarray(moved, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if lengths is not None and pbc is not None:
        cell = np.asarray(lengths, dtype=np.float64).reshape(-1)[:3]
        for axis, periodic in enumerate(pbc):
            length = float(cell[axis])
            if periodic and length > 0.0:
                delta[..., axis] -= np.round(delta[..., axis] / length) * length
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _event_mask(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Free atoms in the top window. The substrate stays in the frame."""
    free = np.asarray(free, dtype=bool)
    z = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[:, 2]
    top = float(z[free].max())
    mask = free & (z > top - _TOP_WINDOW)
    if not np.any(mask):
        return free
    return mask


def _placed(saddle: np.ndarray, free: np.ndarray, free_flat: np.ndarray) -> np.ndarray:
    out = np.array(saddle, dtype=np.float64, copy=True)
    out[free] = np.asarray(free_flat, dtype=np.float64).reshape(-1, 3)
    return out


def _xyz(points: np.ndarray) -> list:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3).tolist()


def _link(record: dict, *, evaluator, free, box, minimum, min_energy, force_norm: float, barrier_max: float, connect_rmsd: float, found: list[np.ndarray], pbc, tolerance: float | None = None) -> dict:
    """Judge the saddle.

    ``connect_rmsd`` is the saddle-novelty test, on the top-window atoms.
    When ``tolerance`` is set, the link is the eOn r2205 rule: exactly one
    endpoint has ``perAtomNorm`` strictly below that distance. Otherwise
    the link is the top-window RMSD test, which is what the stored hTST
    and aladatom campaigns used for process identity.
    """
    from optbench.connection import connected_endpoints, per_atom_max
    from optbench.failures import classify_outcome

    def unfinished(row: dict) -> dict:
        """No saddle was stored. The class comes from the shared classifier."""
        row.pop("positions", None)
        row.pop("lowest_mode", None)
        if row.get("success"):
            row["failure"] = classify_outcome(
                energy=row.get("energy"),
                force=row.get("force_norm"),
                calls=row.get("force_calls"),
                cap=None,
                force_tol=float(force_norm),
                unmet=True,
                dissociated=False,
            )
            row["success"] = row["failure"] is None
        return row

    record["endpoint_plus"] = None
    record["endpoint_minus"] = None
    record["eon_connected"] = False
    saddle = None if record.get("positions") is None else np.asarray(record["positions"], dtype=np.float64).reshape(-1, 3)
    record["reasons"] = list(record.get("reasons") or [])
    mode = record.get("lowest_mode")
    entered = bool(record.get("success"))
    if saddle is None or not mode:
        return unfinished(record)
    axis = np.asarray(mode, dtype=np.float64)
    scale = float(np.linalg.norm(axis))
    if scale < 1e-14:
        return unfinished(record)
    axis = axis / scale
    saddle_free = saddle[free].reshape(-1)
    plus, _plus_ok, plus_calls, _plus_energy = _minimize_free(
        evaluator, _placed(saddle, free, saddle_free + 0.1 * axis), free, box, force_norm, 2000
    )
    minus, _minus_ok, minus_calls, _minus_energy = _minimize_free(
        evaluator, _placed(saddle, free, saddle_free - 0.1 * axis), free, box, force_norm, 2000
    )
    record["endpoint_plus"] = _xyz(plus)
    record["endpoint_minus"] = _xyz(minus)
    mask = _event_mask(minimum, free)
    image = {"lengths": box, "pbc": pbc}
    plus_rmsd = _rmsd(plus[mask], minimum[mask], **image)
    minus_rmsd = _rmsd(minus[mask], minimum[mask], **image)
    record["per_atom_max_plus"] = per_atom_max(plus, minimum, box)
    record["per_atom_max_minus"] = per_atom_max(minus, minimum, box)
    if tolerance is None:
        links = (plus_rmsd <= connect_rmsd) ^ (minus_rmsd <= connect_rmsd)
    else:
        links = connected_endpoints(minimum, plus, minus, box, tolerance)
    record["eon_connected"] = bool(links)
    record["saddle_mask"] = _xyz(saddle[mask])
    record["barrier"] = float(record["energy"] - min_energy) if record.get("energy") is not None else None
    record["rmsd_plus"] = plus_rmsd
    record["rmsd_minus"] = minus_rmsd
    record["rmsd_plus_free"] = _rmsd(plus[free], minimum[free], **image)
    record["rmsd_minus_free"] = _rmsd(minus[free], minimum[free], **image)
    record["rmsd_atoms"] = int(np.count_nonzero(mask))
    record["endpoint_calls"] = int(plus_calls + minus_calls)
    record["force_calls"] = int(record["force_calls"]) + int(plus_calls + minus_calls)
    record["product"] = _xyz(minus if plus_rmsd <= minus_rmsd else plus)
    if not entered:
        record.pop("positions", None)
        record.pop("lowest_mode", None)
        if not record["eon_connected"]:
            record.pop("product", None)
        return record
    novel = all(_rmsd(saddle[mask], old, **image) > connect_rmsd for old in found)
    barrier = float(record["barrier"])
    record["positions"] = saddle.reshape(-1).tolist()
    if novel:
        found.append(saddle[mask].copy())
    # The catalogue still stops on a new unique saddle under the barrier
    # window. That count is not the per-search success.
    record["low_barrier"] = bool(links and novel and 0.0 < barrier < barrier_max)
    if links:
        record["success"] = True
        record["failure"] = None
        record["reasons"] = []
    else:
        record["reasons"] = ["not_connected"]
        record["success"] = False
        record["failure"] = classify_outcome(
            energy=record.get("energy"),
            geometry=saddle,
            force=record.get("force_norm"),
            calls=float(record["force_calls"]),
            cap=None,
            force_tol=float(force_norm),
            hessian_index=None if record.get("hessian_index") is None else int(record["hessian_index"]),
            expected_index=1,
            stationary_rejected=True,
        )
    record.pop("lowest_mode", None)
    return record


def _parameters(surface: dict, section: dict, cap: int, force_norm: float) -> dict:
    return {
        "kernel": surface["kind"],
        "element": surface["element"],
        "pbc": surface["pbc"],
        "cap": cap,
        "force_norm": force_norm,
        "counter": "harness",
        "pykmc": PYKMC_COMMIT,
        "lammps": LAMMPS_COMMIT,
        "artn": ARTN_COMMIT,
        "ira": IRA_COMMIT,
        "zseed_base": _SEED_BASE,
        **section,
    }


def _one_point(root: Path, job: dict, *, cap: int, force_norm: float, kind: str, free, box, evaluator) -> dict:
    payload = _invoke(root, job)
    calls = int(payload.get("calls") or 0)
    if payload.get("capped"):
        calls = max(calls, cap)
    record = _judge(
        job["name"],
        payload,
        cap=cap,
        force_norm=force_norm,
        kind=kind,
        free=free,
        box=box,
        evaluator=evaluator,
        calls=calls,
    )
    record["zseed"] = int(job["zseed"])
    record["central"] = int(job["central"])
    return record


def run_points(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    surface = _surface(problem, template)
    section = _section(template)
    cap = _cap(problem, template)
    force_norm = _force(problem, template)
    section["forc_thr"] = force_norm
    starts = load_starts(root, problem, output.parent / "work" / template["id"].replace("/", "-"))
    jobs = []
    evaluators = []
    metas = []
    for start in starts:
        positions = np.asarray(start["positions"], dtype=np.float64)
        free = np.asarray([not bool(flag) for flag in start["fixed"]], dtype=bool)
        box = np.asarray(start["box"], dtype=np.float64)
        central = int(np.flatnonzero(free)[0])
        zseed = pair_seed(f"{template['id']}:{start['name']}", _SEED_BASE)
        jobs.append(
            _base_job(
                start["name"],
                positions,
                free,
                box,
                surface,
                section,
                cap=cap,
                central=central,
                zseed=zseed,
            )
        )
        evaluators.append(_evaluator(surface, box))
        metas.append((free, box))
    ckpt = output.parent / "work" / template["id"].replace("/", "-") / "points.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if ckpt.is_file():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["start"])] = row
    write_lock = threading.Lock()

    def finish(index: int) -> dict:
        job = jobs[index]
        if job["name"] in done:
            return done[job["name"]]
        free, box = metas[index]
        print(f"{template['id']} {job['name']}", flush=True)
        record = _one_point(
            root,
            job,
            cap=cap,
            force_norm=force_norm,
            kind="morse_pt" if surface["kind"] == "morse" else surface["kind"],
            free=free,
            box=box,
            evaluator=evaluators[index],
        )
        record.pop("positions", None)
        record.pop("lowest_mode", None)
        with write_lock:
            with ckpt.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_json_default) + "\n")
        print(
            f"{template['id']} {job['name']} calls {record['force_calls']} "
            f"success {int(record['success'])} failure {record['failure']}",
            flush=True,
        )
        return record

    if workers <= 1 or len(jobs) <= 1:
        records = [finish(index) for index in range(len(jobs))]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(finish, range(len(jobs))))
    comment = (
        "# counter harness. One pyKMC partn_search from each transition-state guess.\n"
        "# ninit is 0. The free atoms are the LAMMPS system. Fixed atoms stay in the callback.\n"
        "# pair_style zero. Forces are the smooth-linear Morse gradient through fix external.\n"
        "# A success is under the cap, at the force stop, and Hessian index 1."
    )
    return write_campaign(
        output,
        records,
        template,
        problem,
        {"cap": cap, "force_norm": force_norm, "potential": surface["kind"]},
        _parameters(surface, section, cap, force_norm),
        images=None,
        comment=comment,
    )


def _truncate_endpoints(path: Path, count: int) -> None:
    """Drop endpoint lines that belong to a search the snapshot did not keep."""
    if not path.is_file():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= count:
        return
    body = "\n".join(lines[:count])
    path.write_text((body + "\n") if body else "", encoding="utf-8")


def _spill_endpoints(path: Path, record: dict, *, minimum, box, free, pbc, search: int, repetition: int) -> None:
    from optbench.pykmc_kmc import endpoint_record

    payload = endpoint_record(
        start=str(record["start"]),
        search=search,
        repetition=repetition,
        minimum=_xyz(minimum),
        box=np.asarray(box, dtype=np.float64).reshape(-1)[:3].tolist(),
        free=np.asarray(free, dtype=bool).reshape(-1).tolist(),
        pbc=[bool(flag) for flag in pbc],
        endpoint_plus=record.get("endpoint_plus"),
        endpoint_minus=record.get("endpoint_minus"),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
    record.pop("endpoint_plus", None)
    record.pop("endpoint_minus", None)
    record.pop("product", None)
    record.pop("saddle_mask", None)
    record.pop("positions", None)


def _judge_batch(root: Path, jobs: list[dict], *, workers: int, cap: int, force_norm: float, kind: str, free, box, evaluator) -> list[dict]:
    def one(job: dict) -> dict:
        return _one_point(
            root,
            job,
            cap=cap,
            force_norm=force_norm,
            kind=kind,
            free=free,
            box=box,
            evaluator=evaluator,
        )

    if len(jobs) <= 1 or workers <= 1:
        return [one(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(one, jobs))


def run_catalog(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    from optbench.pykmc_kmc import PROFILE, searches_to_launch
    from optbench.v2 import write_starts

    surface = _surface(problem, template)
    section = _section(template)
    force_norm = _force(problem, template)
    section["forc_thr"] = force_norm
    search_cap = int((template.get("search") or {}).get("search_budget", section["evalf_max"]))
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    search = dict(template.get("search") or {})
    repetitions = int(search.get("repetitions", 1))
    magnitude = float(search.get("displace_magnitude", 0.0))
    connect_rmsd = float(search.get("rmsd_connect", 0.5))
    max_searches = int(search.get("max_searches", 20000))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    failure_cap = int(template.get("failure_cap") or (search_cap + 4000))
    starts = load_starts(root, problem, output.parent / "work" / template["id"].replace("/", "-"))
    base = starts[0]
    positions = np.asarray(base["positions"], dtype=np.float64)
    free = np.asarray([not bool(flag) for flag in base["fixed"]], dtype=bool)
    box = np.asarray(base["box"], dtype=np.float64)
    evaluator = _evaluator(surface, box)
    kind = surface["kind"]
    work = output.parent / "work" / template["id"].replace("/", "-")
    snap_path = work / "catalog.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    endpoints_path = output.with_suffix(".endpoints.jsonl")
    records = []
    failed_reps = 0
    connected = 0
    searches = 0
    for rep in range(1, repetitions + 1):
        state = None
        if snap_path.is_file():
            state = json.loads(snap_path.read_text(encoding="utf-8"))
        resume = (
            state
            and state.get("profile") == PROFILE
            and kernel_matches(state, surface["kind"])
            and int(state.get("rep") or 0) == rep
            and state.get("minimum") is not None
        )
        if resume:
            minimum = np.asarray(state["minimum"], dtype=np.float64).reshape(-1, 3)
            min_energy = float(state["min_energy"])
            minimize_calls = int(state["minimize_calls"])
            min_ok = bool(state["min_ok"])
            connected = int(state["connected"])
            searches = int(state["searches"])
            search_calls = int(state["search_calls"])
            found = [np.asarray(item, dtype=np.float64).reshape(-1, 3) for item in state.get("found") or []]
            records = list(state.get("records") or [])
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng"]
            connected = sum(1 for saved in records if saved.get("low_barrier"))
            last_low = 0
            for index, saved in enumerate(records, start=1):
                if saved.get("low_barrier"):
                    last_low = index
            stall = searches - last_low
            _truncate_endpoints(endpoints_path, searches)
        else:
            if rep == 1:
                endpoints_path.unlink(missing_ok=True)
            minimum, min_ok, minimize_calls, min_energy = _minimize_free(
                evaluator, positions, free, box, force_norm, 5000
            )
            rng = np.random.default_rng(_SEED_BASE + rep)
            found = []
            connected = 0
            searches = 0
            search_calls = 0
            records = []
            stall = 0
        climb_cap = min(search_cap, failure_cap)

        def publish(partial: bool) -> None:
            total_calls = int(minimize_calls + search_calls)
            successes = sum(1 for row in records if row.get("success"))
            fraction = successes / searches if searches else 0.0
            eon_count = sum(1 for row in records if row.get("eon_connected"))
            eon_fraction = eon_count / searches if searches else 0.0
            lines = [
                f"# rep {rep} force_calls {total_calls} unique_saddles {len(found)} "
                f"fraction_connected {fraction} fraction_eon_connected {eon_fraction} "
                f"total_searches {searches} connected {successes} eon_connected {eon_count} "
                f"minimize_calls {minimize_calls} minimum_converged {int(min_ok)} seed {_SEED_BASE + rep}"
            ]
            last_at = 0
            last_barrier = None
            for index, saved in enumerate(records, start=1):
                if saved.get("low_barrier"):
                    last_at = index
                    last_barrier = saved.get("barrier")
            if last_at and searches > last_at:
                barrier_text = f"{float(last_barrier):.3f}" if last_barrier is not None else "unknown"
                lines.append(
                    "# fraction_connected counts index-1 saddles whose endpoints pass the eOn r2205 link. "
                    f"The catalogue goal is {goal} unique saddles under {barrier_max:g} eV. "
                    f"The last one is search {last_at}, barrier {barrier_text} eV. "
                    f"The next {searches - last_at} searches added none. "
                    "The runner exits after 160 searches with no new unique saddle under that window "
                    "once 200 searches have run."
                )
            potential = "morse_shift" if surface["kind"] == "morse_shift" else "morse/smooth/linear"
            lines.append(
                "# fraction_eon_connected is the eOn r2205 link: exactly one endpoint has "
                "perAtomNorm strictly below distance_difference 0.1 angstrom. "
                f"A connected index-1 saddle is a harness success. Both endpoint minima are in {endpoints_path.name}."
            )
            lines.append(
                f"# Potential is {potential}. The 2014 Pt archive used shifted Morse."
            )
            endpoint_rows = 0
            if endpoints_path.is_file():
                endpoint_rows = sum(1 for line in endpoints_path.read_text(encoding="utf-8").splitlines() if line.strip())
            text = "\n".join(lines) + "\n"
            if searches:
                text += f"force_calls {total_calls}\n"
                text += f"force_calls_min {total_calls}\n"
                text += f"force_calls_max {total_calls}\n"
            text += (
                f"code {template.get('code', CODE)}\n"
                f"code_version {template.get('code_version', '')}\n"
                f"commit {template.get('commit', '')}\n"
                f"algorithm {template.get('algorithm', '')}\n"
                f"tuning {template.get('tuning', '')}\n"
                f"template {template['id']}\n"
                f"problem {problem['id']}\n"
                f"counter harness\n"
                f"cap {failure_cap}\n"
                f"repetitions {repetitions}\n"
                f"nfailed {failed_reps}\n"
                f"n {len(records)}\n"
                f"goal {goal}\n"
                f"connected {successes}\n"
                f"eon_connected {eon_count}\n"
                f"fraction_connected {fraction}\n"
                f"fraction_eon_connected {eon_fraction}\n"
                f"total_searches {searches}\n"
                f"endpoints {endpoints_path.name}\n"
                f"endpoint_rows {endpoint_rows}\n"
                f"profile {PROFILE}\n"
            )
            if surface["kind"] != "morse_shift":
                text += "ranked false\n"
            text += f"potential {potential}\n"
            if partial:
                text += "partial true\n"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            from optbench.failures import apply_taxonomy
            from optbench.score import score_v2
            from optbench.v2 import collect_provenance

            stored = [apply_taxonomy(dict(row), failure_cap, force_norm) for row in records]
            write_starts(output.with_suffix(".starts.jsonl"), stored)
            scored = score_v2(
                [
                    {"force_calls": row["force_calls"], "success": row["success"], "failure": row.get("failure")}
                    for row in stored
                ],
                failure_cap,
            ) if stored else None
            if scored is not None:
                document = {
                    "schema": "optbench.score/v2",
                    "problem": problem["id"],
                    "template": template["id"],
                    "counter": "harness",
                    "provenance": collect_provenance(
                        code=template.get("code", CODE),
                        code_version=template.get("code_version", ""),
                        commit=template.get("commit", ""),
                        parameters=_parameters(surface, section, failure_cap, force_norm),
                    ),
                    "score": scored,
                    "note": (
                        "A harness success is an index-1 saddle at the force stop whose endpoints "
                        "pass the eOn r2205 link. The barrier window and the novelty test count "
                        f"unique saddles. Endpoint minima are in {endpoints_path.name}."
                    ),
                }
                output.with_suffix(".v2.json").write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

        while True:
            room = searches_to_launch(
                searches, stall, connected, goal, max_searches, max(1, int(workers))
            )
            if room == 0:
                break
            jobs = []
            for offset in range(room):
                number = searches + offset + 1
                atom = int(listed[int(rng.integers(0, len(listed)))])
                if not bool(free[atom]):
                    raise RuntimeError(f"displace atom {atom} is not free")
                point = np.array(minimum, copy=True)
                if magnitude > 0.0:
                    direction = rng.normal(size=3)
                    direction = direction / float(np.linalg.norm(direction))
                    point[atom] = point[atom] + magnitude * direction
                zseed = pair_seed(f"{template['id']}:rep{rep}:search{number}", _SEED_BASE)
                jobs.append(
                    _base_job(
                        f"rep{rep}-search{number}",
                        point,
                        free,
                        box,
                        surface,
                        section,
                        cap=climb_cap,
                        central=atom,
                        zseed=zseed,
                    )
                )
            print(
                f"{template['id']} rep {rep} search {searches + 1}-{searches + room} "
                f"connected {connected}",
                flush=True,
            )
            judged = _judge_batch(
                root,
                jobs,
                workers=workers,
                cap=climb_cap,
                force_norm=force_norm,
                kind=kind,
                free=free,
                box=box,
                evaluator=evaluator,
            )
            for record in judged:
                searches += 1
                before = int(record["force_calls"])
                record = _link(
                    record,
                    evaluator=evaluator,
                    free=free,
                    box=box,
                    minimum=minimum,
                    min_energy=min_energy,
                    force_norm=force_norm,
                    barrier_max=barrier_max,
                    connect_rmsd=connect_rmsd,
                    found=found,
                    pbc=surface["pbc"],
                    tolerance=link_tolerance(problem),
                )
                if record.get("low_barrier"):
                    connected += 1
                    stall = 0
                else:
                    stall += 1
                if int(record["force_calls"]) >= failure_cap and record["success"]:
                    from optbench.failures import classify_outcome

                    was_low = bool(record.get("low_barrier"))
                    record["reasons"] = ["cap"]
                    record["failure"] = classify_outcome(
                        energy=record.get("energy"),
                        force=record.get("force_norm"),
                        calls=float(record["force_calls"]),
                        cap=failure_cap,
                        force_tol=float(force_norm),
                    )
                    record["success"] = record["failure"] is None
                    record["low_barrier"] = False
                    if was_low and not record["success"]:
                        connected -= 1
                search_calls += int(record["force_calls"])
                _spill_endpoints(
                    endpoints_path,
                    record,
                    minimum=minimum,
                    box=box,
                    free=free,
                    pbc=surface["pbc"],
                    search=searches,
                    repetition=rep,
                )
                records.append(record)
                print(
                    f"{template['id']} search {searches} climb {before} total {record['force_calls']} "
                    f"success {int(record['success'])} eon {int(bool(record.get('eon_connected')))} "
                    f"connected {connected} barrier {record.get('barrier')} "
                    f"rmsd {record.get('rmsd_plus')} {record.get('rmsd_minus')}",
                    flush=True,
                )
                snap_path.write_text(
                    json.dumps(
                        {
                            "profile": PROFILE,
                            "kernel": surface["kind"],
                            "rep": rep,
                            "minimum": minimum.reshape(-1).tolist(),
                            "min_energy": min_energy,
                            "minimize_calls": minimize_calls,
                            "min_ok": min_ok,
                            "connected": connected,
                            "searches": searches,
                            "search_calls": search_calls,
                            "found": [item.reshape(-1).tolist() for item in found],
                            "records": records,
                            "rng": rng.bit_generator.state,
                        },
                        default=_json_default,
                    ),
                    encoding="utf-8",
                )
            publish(partial=True)
        if connected < goal:
            failed_reps += 1
        publish(partial=False)
    totals = []
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# rep "):
            continue
        parts = line.split()
        totals.append(int(parts[parts.index("force_calls") + 1]))
    return {
        "force_calls": statistics.fmean(totals) if totals else None,
        "nfailed": failed_reps,
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "connected": connected,
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    if problem["id"] == "tsopt/pt-island":
        return run_points(root, problem, template, output=output, workers=workers)
    if problem["id"] == "saddle-search/pt-island":
        return run_catalog(root, problem, template, output=output, workers=workers)
    if problem["id"] in {"hTST/Pt", "hTST/Al"}:
        from optbench.engines.pykmc_kmc_run import run_htst

        return run_htst(root, problem, template, output=output, workers=workers)
    if problem["id"] in {"accelerated-md/aladatom", "accelerated-md/standard_aladatom"}:
        from optbench.engines.pykmc_kmc_run import run_amd

        return run_amd(root, problem, template, output=output, workers=workers)
    raise ValueError(f"{template_name} does not score {problem['id']}")
