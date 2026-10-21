"""geomeTRIC transition-state searches on the harness potential.

The optimizer is geomeTRIC's ``--transition`` path. Coordinates are Bohr.
The potential and the force stop stay in the problem's units. A harness
Hessian is charged as the Cartesian finite-difference stencil it replaces.
Verification of the Hessian index is not part of the search cost.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .ase_sella import (
    CallCounter,
    _apply_pbc,
    _free_mask,
    _inner_calculator,
    _jsonable,
    _read_atoms,
    _reason_list,
    _verify_index,
    discover,
    format_metric,
    materialize,
)
from .campaign import load_template, resolve_problem
from .counter import ForceCallCap, attach_ase_counter
from .failures import classify_failure
from .potentials import _hessian_from_pairs, _pairs
from .score import score_v2, summarize_all

CONTRIBUTOR = "Rohit Goswami"


def lennard_jones_hessian(positions, epsilon: float, sigma: float, rc: float):
    """Cartesian Hessian of the truncated Lennard-Jones energy."""
    import numpy as np

    pos = np.asarray(positions, dtype=float)
    iu, ju, diff, r2 = _pairs(pos, float(rc), None)
    dim = 3 * pos.shape[0]
    if iu.size == 0:
        return np.zeros((dim, dim))
    inv_r2 = 1.0 / r2
    sr2 = (float(sigma) * float(sigma)) * inv_r2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    r = np.sqrt(r2)
    first = 24.0 * float(epsilon) * (-2.0 * sr12 + sr6) / r
    second = 24.0 * float(epsilon) * (26.0 * sr12 - 7.0 * sr6) * inv_r2
    return _hessian_from_pairs(pos.shape[0], iu, ju, diff, r2, first, second)


def smooth_morse_hessian(positions, cell, pbc, D: float, alpha: float, r0: float, rc: float):
    """Cartesian Hessian of the smooth-linear Morse energy.

    The pair geometry matches ``_morse_arrays``: minimum image on the
    periodic axes only, and a linear tail that makes the energy and the
    force continuous at the cutoff.
    """
    import numpy as np

    pos = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    periodic = np.asarray(pbc, dtype=bool)
    delta = pos[:, None, :] - pos[None, :, :]
    if bool(np.any(periodic)):
        frac = delta @ np.linalg.inv(cell)
        shift = np.round(frac)
        for axis, live in enumerate(periodic):
            if not live:
                shift[..., axis] = 0.0
        delta = (frac - shift) @ cell
    dist = np.linalg.norm(delta, axis=-1)
    mask = (dist > 1e-12) & (dist < float(rc))
    iu, ju = np.triu_indices(pos.shape[0], k=1)
    pair = mask[iu, ju]
    diff = delta[iu, ju][pair]
    r = dist[iu, ju][pair]
    x = np.exp(-float(alpha) * (r - float(r0)))
    x_cut = np.exp(-float(alpha) * (float(rc) - float(r0)))
    slope_cut = 2.0 * float(alpha) * float(D) * x_cut * (1.0 - x_cut)
    first = 2.0 * float(alpha) * float(D) * x * (1.0 - x) - slope_cut
    second = 2.0 * float(alpha) ** 2 * float(D) * x * (2.0 * x - 1.0)
    return _hessian_from_pairs(pos.shape[0], iu[pair], ju[pair], diff, r * r, first, second)


def _free_block(full, free):
    import numpy as np

    cols = [3 * int(atom) + axis for atom in free for axis in range(3)]
    index = np.asarray(cols, dtype=int)
    return np.array(full, dtype=float)[np.ix_(index, index)]


def _harness_hessian(atoms, potential: dict, free):
    kind = potential["kind"]
    if kind == "lennard-jones":
        full = lennard_jones_hessian(
            atoms.get_positions(),
            float(potential["epsilon"]),
            float(potential["sigma"]),
            float(potential["rc"]),
        )
    elif kind == "morse":
        if str(potential.get("form")) != "smooth":
            raise ValueError("the harness Morse Hessian is the smooth linear form")
        full = smooth_morse_hessian(
            atoms.get_positions(),
            atoms.cell.array,
            atoms.pbc,
            float(potential["D"]),
            float(potential["alpha"]),
            float(potential["r0"]),
            float(potential["rc"]),
        )
    else:
        raise ValueError(kind)
    return _free_block(full, free)


def entry_id(template: dict) -> str:
    rest = template["id"].split("/", 1)[1]
    return f"geometric-{template['code_version']}-{rest}"


def _cap_error(exc: BaseException) -> bool:
    return isinstance(exc, ForceCallCap) or (
        isinstance(exc, RuntimeError) and "force-call cap" in str(exc)
    )


def _charge_stencil(counter, count: int) -> None:
    for index in range(int(count)):
        counter.charge((float(counter.calls), float(index)))


def _write_dat(
    path: Path,
    template: dict,
    summary: dict,
    parameters: dict,
    *,
    starts: int,
    partial: bool,
    scored: dict | None,
) -> None:
    method = {
        "profile": template["profile"],
        "parameter_source": template.get("parameter_source", ""),
        "counter": "harness attach_ase_counter",
        "reducer": template["reducer"],
        "parameters": parameters,
        "potential": template.get("potential") or {},
        "stop": template.get("stop") or {},
    }
    note = template.get("note", "")
    fields = [
        ("force_calls", summary.get("force_calls")),
        ("force_calls_min", summary.get("force_calls_min")),
        ("force_calls_median", summary.get("force_calls_median")),
        ("force_calls_max", summary.get("force_calls_max")),
        ("nfailed", summary.get("nfailed")),
        ("starts", starts),
        ("code", template["code"]),
        ("code_version", template["code_version"]),
        ("commit", template.get("commit", "")),
        ("algorithm", template["algorithm"]),
        ("profile", template["profile"]),
        ("template", template["id"]),
        ("problem", template["problem"]),
        ("counter", "harness"),
        ("reducer", template["reducer"]),
        ("parameter_source", template.get("parameter_source", "")),
        ("method", json.dumps(_jsonable(method), sort_keys=True, separators=(",", ":"))),
        ("note", note),
        ("host", socket.gethostname()),
        ("date", date.today().isoformat()),
        ("contributor", CONTRIBUTOR),
    ]
    if scored:
        fields.extend(
            [
                ("scoring", scored.get("scoring")),
                ("par2_mean", scored.get("par2_mean")),
                ("par2_median", scored.get("median")),
                ("par2_median_low", scored.get("median_low")),
                ("par2_median_high", scored.get("median_high")),
                ("par2_iqr", scored.get("iqr")),
                ("success_rate", scored.get("success_rate")),
                ("wilson_low", scored.get("wilson_low")),
                ("wilson_high", scored.get("wilson_high")),
                ("par_applied", scored.get("par_applied")),
                ("par_factor", scored.get("par_factor")),
                ("bootstrap_resamples", scored.get("bootstrap_resamples")),
                ("bootstrap_seed", scored.get("bootstrap_seed")),
            ]
        )
    if partial:
        fields.append(("partial", "true"))
    lines = []
    for key, value in fields:
        text = format_metric(value)
        if text is None or text == "":
            continue
        lines.append(f"{key} {text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_start(payload: dict) -> dict:
    import logging

    import numpy as np

    logging.disable(logging.CRITICAL)
    from geometric.engine import Engine
    from geometric.internal import CartesianCoordinates, DelocalizedInternalCoordinates
    from geometric.molecule import Molecule
    from geometric.nifty import ang2bohr, bohr2ang
    from geometric.optimize import OPT_STATE, Optimizer
    from geometric.params import OptParams

    class HarnessEngine(Engine):
        def __init__(self, molecule, atoms, free):
            super().__init__(molecule)
            self.atoms = atoms
            self.free = np.asarray(free, dtype=int)
            self.base = np.array(atoms.get_positions(), dtype=float, copy=True)
            self.raw = {}
            self.samples = []

        def calc_new(self, coords, dirname):
            snapshot = np.array(coords, dtype=float, copy=True).ravel()
            pos = self.base.copy()
            pos[self.free] = snapshot.reshape(-1, 3) * bohr2ang
            self.atoms.set_positions(pos)
            energy = float(self.atoms.get_potential_energy())
            forces = np.asarray(self.atoms.get_forces(), dtype=float)[self.free].ravel()
            gradient = np.array(-forces * bohr2ang, dtype=float, copy=True)
            self.raw[_coord_key(snapshot)] = gradient
            self.samples.append((snapshot, gradient))
            return {"energy": energy, "gradient": gradient}

        def copy_scratch(self, src, dest):
            return

        def force_norm(self, coords) -> float:
            target = np.asarray(coords, dtype=float).ravel()
            gradient = self.raw.get(_coord_key(target))
            if gradient is None:
                best = None
                best_err = None
                for snapshot, grad in self.samples:
                    err = float(np.max(np.abs(snapshot - target)))
                    if best_err is None or err < best_err:
                        best = grad
                        best_err = err
                if best is None or best_err is None or best_err > 1e-6:
                    raise KeyError("no harness gradient at the current coordinates")
                gradient = best
            return float(np.linalg.norm(gradient) / bohr2ang)

    class FastCartesian(CartesianCoordinates):
        """Cartesian coordinates without the primitive-internal build.

        The library constructor builds bonds, angles, and dihedrals, then
        discards them. On the Pt island that build does not return.
        """

        def makePrimitives(self, molecule, connect, addcart):
            self.frags = [list(range(molecule.na))]

        def calcHess(self, xyz, gradx, hessx):
            return np.array(hessx, dtype=float, copy=True)

        def calcHessCart(self, xyz, gradq, hessq):
            return np.array(hessq, dtype=float, copy=True)

    bucket = CallCounter(int(payload["raw_cap"]))
    row = {
        "schema": "optbench.start/v2",
        "start": payload["start"],
        "seed": int(payload.get("seed") or 0),
        "force_calls": 0,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failed": True,
        "force_norm": None,
        "reference": "none",
        "expected_index": 1,
    }
    try:
        atoms = _apply_pbc(_read_atoms(Path(payload["paths"][0])), payload["potential"])
        free_mask = np.asarray(_free_mask(atoms), dtype=bool)
        free = np.flatnonzero(free_mask)
        if free.size == 0:
            raise RuntimeError("no free atoms")
        spec = payload["geometric"]
        coordsys = str(spec["coordsys"])
        if coordsys == "tric" and not bool(free_mask.all()):
            raise RuntimeError("TRIC keeps every atom free; this geometry has fixed atoms")
        calculator = _inner_calculator(payload["potential"])
        attach_ase_counter(
            calculator,
            cap=bucket[1],
            state=bucket.state,
            counter=bucket.harness,
        )
        atoms.calc = calculator
        with tempfile.TemporaryDirectory(prefix="geometric-") as scratch:
            scratch_path = Path(scratch)
            positions = np.array(atoms.get_positions(), dtype=float)[free]
            symbols = [atoms.symbols[int(index)] for index in free]
            xyz = scratch_path / "start.xyz"
            _write_xyz(xyz, symbols, positions)
            molecule = Molecule(str(xyz), build_topology=coordsys == "tric")
            if coordsys == "cart":
                ic = FastCartesian(molecule)
            elif coordsys == "tric":
                ic = DelocalizedInternalCoordinates(
                    molecule,
                    build=True,
                    connect=False,
                    addcart=False,
                    connect_isolated=True,
                )
            else:
                raise ValueError(coordsys)
            engine = HarnessEngine(molecule, atoms, free)
            hessian_mode = str(spec["hessian"])
            hess_data = None
            if hessian_mode == "harness":
                hess_ang = _harness_hessian(atoms, payload["potential"], free)
                hess_data = hess_ang * (bohr2ang ** 2)
                _charge_stencil(bucket.harness, 2 * hess_data.shape[0])
                bucket.state["calls"] = bucket.harness.calls
            elif hessian_mode != "finite-difference":
                raise ValueError(hessian_mode)
            params = OptParams(
                transition=True,
                frequency=False,
                hessian="first",
                maxiter=int(spec["maxiter"]),
                subfrctor=int(spec["subfrctor"]),
                convergence_gmax=float(spec["convergence_gmax"]),
                convergence_grms=float(spec["convergence_grms"]),
            )
            if hess_data is not None:
                params.hess_data = hess_data
            if payload.get("snapshot"):
                row["parameters"] = _parameters(params, spec, len(ic.Internals), len(atoms), int(free.size))
            coords = positions.ravel() * ang2bohr
            opt = Optimizer(
                coords,
                molecule,
                ic,
                engine,
                str(scratch_path / "run"),
                params,
                print_info=False,
            )
            norm = _drive(opt, engine, bucket, float(payload["tol"]))
            row["force_norm"] = norm
            row["iterations"] = int(opt.Iteration)
            calls = int(bucket[0])
            if norm is not None and norm <= float(payload["tol"]) and calls < bucket[1]:
                _place(atoms, engine.base, free, opt.X, bohr2ang)
                row["hessian_index"] = int(_verify_index(atoms, 1))
    except Exception as exc:
        if _cap_error(exc):
            row["error"] = "force-call cap"
        else:
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            row["failed"] = True
    row["force_calls"] = int(bucket[0])
    row["verify_calls"] = int(bucket.state["verify_calls"])
    reasons = _reason_list(row, bucket[1], float(payload["tol"]))
    row["reasons"] = reasons
    row["success"] = not reasons
    row["failed"] = not row["success"]
    row["converged"] = "force_norm" not in reasons and "cap" not in reasons and "error" not in reasons
    if row["success"]:
        row["failure"] = None
    else:
        row["failure"] = "crash" if row.get("error") not in (None, "force-call cap") else classify_failure(reasons)
    return row


def _coord_key(coords) -> tuple:
    import numpy as np

    return tuple(np.round(np.asarray(coords, dtype=float).ravel(), decimals=8))


def _write_xyz(path: Path, symbols: list[str], positions) -> None:
    lines = [str(len(symbols)), "optbench"]
    for symbol, xyz in zip(symbols, positions):
        lines.append(f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _place(atoms, base, free, coords_bohr, bohr2ang: float) -> None:
    import numpy as np

    pos = np.array(base, dtype=float, copy=True)
    pos[np.asarray(free, dtype=int)] = np.asarray(coords_bohr, dtype=float).reshape(-1, 3) * bohr2ang
    atoms.set_positions(pos)


def _parameters(params, spec: dict, n_coordinates: int, n_atoms: int, n_free: int) -> dict:
    return {
        "transition": True,
        "coordsys": spec["coordsys"],
        "hessian": spec["hessian"],
        "trust": float(params.trust),
        "tmax": float(params.tmax),
        "tmin": float(params.tmin),
        "maxiter": int(params.maxiter),
        "subfrctor": int(params.subfrctor),
        "convergence_energy": float(params.Convergence_energy),
        "convergence_grms": float(params.Convergence_grms),
        "convergence_gmax": float(params.Convergence_gmax),
        "convergence_drms": float(params.Convergence_drms),
        "convergence_dmax": float(params.Convergence_dmax),
        "frequency": False,
        "finite_difference_step_bohr": 0.001,
        "n_coordinates": int(n_coordinates),
        "n_atoms": int(n_atoms),
        "n_free": int(n_free),
    }


def _drive(opt, engine, bucket: CallCounter, tol: float) -> float | None:
    from geometric.optimize import OPT_STATE

    norm = None
    try:
        opt.calcEnergyForce()
        opt.prepareFirstStep()
        norm = engine.force_norm(opt.X)
        while opt.state not in (OPT_STATE.CONVERGED, OPT_STATE.FAILED):
            if bucket[0] >= bucket[1] or (norm is not None and norm <= tol):
                break
            opt.step()
            if opt.state == OPT_STATE.NEEDS_EVALUATION:
                opt.calcEnergyForce()
                opt.evaluateStep()
            if opt.recalcHess:
                opt.calcEnergyForce()
            norm = engine.force_norm(opt.X)
    except Exception as exc:
        if not _cap_error(exc):
            raise
    return norm


def _load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["start"])] = row
    return rows


def _summarize(rows: list[dict]) -> dict:
    return summarize_all(
        [int(row["force_calls"]) for row in rows],
        sum(1 for row in rows if row["failed"]),
    )


def _ordered_rows(jobs, by_start: dict[str, dict]) -> list[dict]:
    return [by_start[name] for name, _paths, _seed in jobs]


def run_template(
    root: Path,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
    cache: Path | None = None,
) -> dict:
    """Run one geomeTRIC template and write benchmark.dat plus per-start rows."""
    template = load_template(root, template_name)
    if template["engine"] != "geometric":
        raise ValueError(f"{template_name} is not a geomeTRIC template")
    problem = resolve_problem(root, template["problem"])
    cache = cache or Path("/tmp/optbench-geometry")
    material = materialize(root, template, problem, cache)
    jobs = discover(material, template)
    if limit > 0:
        jobs = jobs[:limit]
    if not jobs:
        raise FileNotFoundError(f"{template['id']} found no geometries")
    starts_path = output.with_name(output.stem + ".starts.jsonl")
    done = _load_rows(starts_path) if resume else {}
    if not resume and starts_path.exists():
        starts_path.unlink()
    cap = int(template["stop"]["max_force_calls"])
    tol = float(template["stop"]["force_norm"])
    payloads = []
    for name, paths, seed in jobs:
        if name in done:
            continue
        payloads.append(
            {
                "start": name,
                "paths": [str(path) for path in paths],
                "potential": template["potential"],
                "geometric": template["geometric"],
                "seed": 0 if seed is None else seed,
                "tol": tol,
                "raw_cap": cap,
                "snapshot": False,
            }
        )
    if payloads:
        payloads[0]["snapshot"] = True
    fresh: list[dict] = []
    if payloads:
        starts_path.parent.mkdir(parents=True, exist_ok=True)
        with starts_path.open("a", encoding="utf-8") as handle:
            if workers <= 1:
                iterator = (_run_start(payload) for payload in payloads)
                total = len(payloads)
                for index, row in enumerate(iterator, 1):
                    fresh.append(row)
                    handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                    handle.flush()
                    if index == 1 or index == total or index % 25 == 0:
                        print(f"{template['id']} {index}/{total}", flush=True)
            else:
                import multiprocessing as mp

                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    futures = [pool.submit(_run_start, payload) for payload in payloads]
                    total = len(futures)
                    for index, future in enumerate(as_completed(futures), 1):
                        row = future.result()
                        fresh.append(row)
                        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        handle.flush()
                        if index == 1 or index == total or index % 25 == 0:
                            print(f"{template['id']} {index}/{total}", flush=True)
    by_start = dict(done)
    for row in fresh:
        by_start[row["start"]] = row
    ordered = _ordered_rows(jobs, by_start)
    starts_path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    for row in ordered:
        if row.get("error"):
            print(f"error {template['id']} {row['start']}: {row['error']}", flush=True)
    summary = _summarize(ordered)
    records = [
        {
            "force_calls": float(row["force_calls"]),
            "success": bool(row.get("success")) and float(row["force_calls"]) < cap,
        }
        for row in ordered
    ]
    scored = score_v2(records, cap)
    summary["par2_mean"] = scored.get("par2_mean")
    summary["scoring"] = scored.get("scoring")
    parameters = dict(template.get("geometric") or {})
    for row in ordered:
        if row.get("parameters"):
            parameters = row["parameters"]
            break
    _write_dat(
        output,
        template,
        summary,
        parameters,
        starts=len(ordered),
        partial=limit > 0,
        scored=scored,
    )
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = template["problem"]
    summary["entry"] = entry_id(template)
    summary["output"] = str(output)
    return summary


def run_point(root: Path, template_name: str, start: str, seed: int, *, cache: Path | None = None) -> dict:
    """Run one start and return its v2 record."""
    import time

    template = load_template(root, template_name)
    problem = resolve_problem(root, template["problem"])
    cache = cache or Path("/tmp/optbench-geometry")
    material = materialize(root, template, problem, cache)
    jobs = discover(material, template)
    chosen = None
    for name, paths, job_seed in jobs:
        job_seed_i = 0 if job_seed is None else int(job_seed)
        if name == start and job_seed_i == int(seed):
            chosen = (name, paths, job_seed_i)
            break
    if chosen is None:
        raise KeyError(f"{template_name} has no start {start!r} seed {seed}")
    name, paths, job_seed_i = chosen
    payload = {
        "start": name,
        "paths": [str(path) for path in paths],
        "potential": template["potential"],
        "geometric": template["geometric"],
        "seed": job_seed_i,
        "tol": float(template["stop"]["force_norm"]),
        "raw_cap": int(template["stop"]["max_force_calls"]),
        "snapshot": True,
    }
    started = time.perf_counter()
    row = _run_start(payload)
    row["wall_seconds"] = time.perf_counter() - started
    row["template"] = template["id"]
    row["problem"] = template["problem"]
    return row
