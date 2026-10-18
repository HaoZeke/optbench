"""Minimize Lennard-Jones 38 with OptKing or PyBerny.

The potential is the harness Lennard-Jones kernel. Hydrogen labels are
only for the coordinate builder. Problem coordinates stay in angstrom.
The gradient those libraries see is that derivative times
``bohr2angstroms``. ``fix_com`` and ``fix_orientation`` stay on, so the
recorded start is the ``.con`` geometry.

One harness evaluation is one geometry. OptKing line search stays off,
and PyBerny interpolates its line search, so a step does not add a force
call. The search stops on the problem force norm. The library's own
thresholds do not end the run. A v2 success is under the cap, at that
norm, and vibrational Hessian index 0. Energy and Kabsch RMSD are not
applied.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, lennard_jones_hessian
from optbench.potserver import RgpotCounter
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance, write_starts

_CHECKPOINT = 25
_STALL_STEPS = 3
_START_ABS = 1e-6


class StartDisplaced(RuntimeError):
    """The first library geometry is not the ``.con`` start."""


def _close_optking_files() -> None:
    logger = logging.getLogger("optking")
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def _retarget_optking(directory: str) -> None:
    """Point OptKing's file log at ``directory`` and keep it off the console."""
    _close_optking_files()
    logger = logging.getLogger("optking")
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    path = os.path.join(directory, "opt_log.out")
    handler = logging.FileHandler(path, mode="w")
    handler.setLevel(logging.ERROR)
    logger.addHandler(handler)


def _quiet_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def library_version(engine: str) -> str:
    """Import the pinned library and return the version string it reports."""
    if engine == "pyberny":
        import berny  # noqa: F401
        from importlib.metadata import version

        return version("pyberny")
    if engine != "optking":
        raise ValueError(f"unknown geometry engine {engine}")
    cwd = os.getcwd()
    work = tempfile.mkdtemp(prefix="optking-ver-")
    try:
        os.chdir(work)
        import optking

        _retarget_optking(work)
        return str(optking.__version__)
    finally:
        os.chdir(cwd)
        _close_optking_files()
        shutil.rmtree(work, ignore_errors=True)


def _bohr_to_angstrom() -> float:
    import qcelemental as qcel

    return float(qcel.constants.bohr2angstroms)


def _symbol(options: dict, natom: int) -> list[str]:
    label = str(options.get("symbol") or "H")
    return [label] * natom


def _finite_pair(energy: float, gradient) -> bool:
    import numpy as np

    return bool(math.isfinite(energy) and np.all(np.isfinite(gradient)))


def _moved(previous, current) -> bool:
    import numpy as np

    return float(np.max(np.abs(np.asarray(current) - np.asarray(previous)))) > 1e-12


def drive_optking(coords, evaluate, should_stop, cap: int, options: dict, sink: dict) -> None:
    """Step OptKing. ``sink['point']`` is the last evaluated geometry."""
    import numpy as np
    import qcelemental as qcel

    _quiet_threads()
    spec0 = np.asarray(coords, dtype=float).reshape(-1)
    sink["point"] = [float(value) for value in spec0]
    sink["stopped"] = False
    if spec0.size % 3 != 0:
        raise ValueError("coordinates are not a set of atom positions")
    b2a = _bohr_to_angstrom()
    symbols = _symbol(options, spec0.size // 3)
    molecule = qcel.models.Molecule(
        symbols=symbols,
        geometry=(spec0 / b2a).tolist(),
        fix_com=True,
        fix_orientation=True,
        molecular_charge=0,
        molecular_multiplicity=1,
        nonphysical=True,
    )
    params = {
        "opt_type": "MIN",
        "step_type": str(options.get("step_type") or "RFO"),
        "opt_coordinates": str(options.get("coordinates") or "CARTESIAN"),
        "geom_maxiter": int(cap),
        "alg_geom_maxiter": int(cap),
        "dynamic_level": int(options.get("dynamic_level") or 0),
        "linesearch": False,
    }
    if options.get("intrafrag_trust") is not None:
        params["intrafrag_trust"] = float(options["intrafrag_trust"])
    if options.get("intrafrag_hess"):
        params["intrafrag_hess"] = str(options["intrafrag_hess"])
    cwd = os.getcwd()
    work = tempfile.mkdtemp(prefix="optking-")
    try:
        os.chdir(work)
        import optking

        _retarget_optking(work)
        opt = optking.CustomHelper(molecule, params, silent=True)
        current = np.asarray(opt.geom, dtype=float).reshape(-1) * b2a
        if float(np.max(np.abs(current - spec0))) > _START_ABS:
            raise StartDisplaced(
                f"OptKing moved the start by {float(np.max(np.abs(current - spec0))):.3e} angstrom"
            )
        point = current
        stall = 0
        for _ in range(int(cap)):
            energy, gradient = evaluate(point)
            sink["point"] = [float(value) for value in point]
            gradient = np.asarray(gradient, dtype=float).reshape(-1)
            energy = float(energy)
            if not _finite_pair(energy, gradient):
                return
            if should_stop():
                sink["stopped"] = True
                return
            opt.E = energy
            opt.gX = gradient * b2a
            opt.compute()
            opt.take_step()
            new = np.asarray(opt.geom, dtype=float).reshape(-1) * b2a
            if not _moved(point, new):
                stall += 1
                if stall >= _STALL_STEPS:
                    return
            else:
                stall = 0
            point = new
    finally:
        os.chdir(cwd)
        _close_optking_files()
        shutil.rmtree(work, ignore_errors=True)


def drive_pyberny(coords, evaluate, should_stop, cap: int, options: dict, sink: dict) -> None:
    """Step PyBerny. ``sink['point']`` is the last evaluated geometry."""
    import numpy as np
    from berny import Berny
    from berny.geomlib import Geometry

    _quiet_threads()
    logging.getLogger("berny").setLevel(logging.ERROR)
    spec0 = np.asarray(coords, dtype=float).reshape(-1)
    sink["point"] = [float(value) for value in spec0]
    sink["stopped"] = False
    if spec0.size % 3 != 0:
        raise ValueError("coordinates are not a set of atom positions")
    b2a = _bohr_to_angstrom()
    symbols = _symbol(options, spec0.size // 3)
    kwargs = {"maxsteps": int(cap), "symmetry": "nowarn"}
    if "dihedral" in options:
        kwargs["dihedral"] = bool(options["dihedral"])
    if "superweakdih" in options:
        kwargs["superweakdih"] = bool(options["superweakdih"])
    if options.get("trust") is not None:
        kwargs["trust"] = float(options["trust"])
    opt = Berny(Geometry(symbols, spec0.reshape(-1, 3)), **kwargs)
    current = np.asarray(next(opt).coords, dtype=float).reshape(-1)
    if float(np.max(np.abs(current - spec0))) > _START_ABS:
        raise StartDisplaced(
            f"PyBerny moved the start by {float(np.max(np.abs(current - spec0))):.3e} angstrom"
        )
    point = current
    stall = 0
    for _ in range(int(cap)):
        energy, gradient = evaluate(point)
        sink["point"] = [float(value) for value in point]
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        energy = float(energy)
        if not _finite_pair(energy, gradient):
            return
        if should_stop():
            sink["stopped"] = True
            return
        try:
            opt.send((energy, (gradient * b2a).reshape(-1, 3)))
            nxt = next(opt)
        except StopIteration:
            return
        new = np.asarray(nxt.coords, dtype=float).reshape(-1)
        if not _moved(point, new):
            stall += 1
            if stall >= _STALL_STEPS:
                return
        else:
            stall = 0
        point = new


def _options(template: dict) -> dict:
    engine = str(template.get("engine") or "")
    if engine == "optking":
        section = dict(template.get("optking") or {})
        if "coordinates" not in section:
            raise ValueError(f"{template['id']} needs optking.coordinates")
        return section
    if engine == "pyberny":
        return dict(template.get("pyberny") or {})
    raise ValueError(f"{template['id']} is not an OptKing or PyBerny template")


def _index_at(point: list[float], force_stop: float) -> tuple[int, list[float]]:
    import numpy as np

    matrix = lennard_jones_hessian(point)
    values = vibrational_eigenvalues(matrix, point, rotations=True)
    tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _minimize_job(job: dict) -> dict:
    import numpy as np

    _quiet_threads()
    calls = 0
    verify_calls = 0
    error = ""
    energy = None
    force_norm = None
    potential = None
    cap = int(job["cap"])
    gtol = float(job["gtol"])
    sink = {"point": [float(value) for value in job["coords"]], "stopped": False}
    try:
        counter = HarnessCounter(cache=True, cap=cap)
        potential = RgpotCounter(
            lambda positions, _numbers=None, _box=None: lennard_jones(positions),
            counter,
        )
        state = {"norm": None}

        def evaluate(coords):
            value, gradient = potential.calculate(np.asarray(coords, dtype=float))
            gradient = np.asarray(gradient, dtype=float).reshape(-1)
            state["norm"] = float(np.linalg.norm(gradient))
            return float(value), gradient

        def should_stop() -> bool:
            return state["norm"] is not None and state["norm"] <= gtol

        engine = str(job["engine"])
        if engine == "optking":
            drive_optking(job["coords"], evaluate, should_stop, cap, job["options"], sink)
        elif engine == "pyberny":
            drive_pyberny(job["coords"], evaluate, should_stop, cap, job["options"], sink)
        else:
            raise ValueError(f"unknown geometry engine {engine}")
    except (StartDisplaced, ImportError):
        raise
    except ForceCallCap as exc:
        error = f"ForceCallCap: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    point = [float(value) for value in sink["point"]]
    stopped = bool(sink["stopped"])
    if potential is not None:
        calls = int(potential.counter.calls)
        try:
            with potential.counter.verification():
                energy, gradient = potential.calculate(np.asarray(point, dtype=float))
            force_norm = float(np.linalg.norm(gradient))
            verify_calls = int(potential.counter.verify_calls)
            energy = float(energy)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    hessian_index = None
    min_eigenvalues: list[float] = []
    if point is not None and (not error or "ForceCallCap" in error or "force-call cap" in error):
        try:
            hessian_index, min_eigenvalues = _index_at(point, gtol)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    finite = (
        point is not None
        and all(math.isfinite(value) for value in point)
        and energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    at_force = (
        finite
        and stopped
        and force_norm is not None
        and force_norm <= gtol
        and calls < cap
        and not error
    )
    reasons: list[str] = []
    cap_hit = "ForceCallCap" in error or "force-call cap" in error
    if error and not cap_hit:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif calls >= cap or cap_hit:
        reasons.append("cap")
    elif not at_force:
        reasons.append("force_norm")
    elif hessian_index != 0:
        reasons.append("hessian_index")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": verify_calls,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": force_norm,
        "hessian_index": hessian_index,
        "expected_index": 0,
        "converged": at_force,
        "hessian_calls": 0,
        "min_eigenvalues": min_eigenvalues,
    }
    if error:
        record["error"] = error
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter harness. One force call is one RgpotCounter.calculate entry.",
        "# The library steps on the harness Lennard-Jones gradient.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
        "# LJ38 index removes the six rigid modes. A curvature is negative only below "
        "-max(force_norm, 1e-8*||H||_F, 1e-6).",
        "# energy_min and energy_max are the final potential values on the recorded points.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_done(path: Path) -> dict[str, dict]:
    found = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[str(row["start"])] = row
    return found


def _public_records(records: list[dict]) -> list[dict]:
    ordered = sorted(records, key=lambda row: int(row["index"]))
    public = []
    for record in ordered:
        row = dict(record)
        row.pop("index", None)
        public.append(row)
    return public


def _metrics_of(records: list[dict], cap: int) -> tuple[dict, dict]:
    public = _public_records(records)
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in public]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in public if record["converged"]]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in public if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in public
        ],
        cap,
    )
    summary["starts"] = len(public)
    return summary, metrics


def _publish(output: Path, template: dict, problem: dict, records: list[dict], cap: int, version: str, commit: str, passed: dict) -> dict:
    summary, metrics = _metrics_of(records, cap)
    public = _public_records(records)
    provenance = collect_provenance(
        code=str(template.get("code")),
        code_version=version,
        commit=commit,
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. The library steps on the harness Lennard-Jones gradient. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": passed,
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "force_calls": summary.get("force_calls"),
                    "force_calls_min": summary.get("force_calls_min"),
                    "force_calls_max": summary.get("force_calls_max"),
                    "nfailed": summary.get("nfailed"),
                    "median": summary.get("median"),
                },
                "starts": public,
            }
        ],
    }
    _atomic_write(output, _dat(summary, template, metrics, commit, version))
    _atomic_write(
        output.with_suffix(".starts.jsonl"),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in public),
    )
    _atomic_write(
        output.with_suffix(".v2.json"),
        json.dumps(document, indent=2, sort_keys=True) + "\n",
    )
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["v2_successes"] = metrics["successes"]
    return summary


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    engine = str(template.get("engine") or "")
    version = library_version(engine)
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"{engine} {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if not commit:
        raise RuntimeError(f"{template['id']} has no commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    if problem["id"] != "minimization/lj38":
        raise ValueError(f"{template['id']} does not run on {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    options = _options(template)
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    jobs = []
    for index, (name, coords, _box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": [float(value) for value in coords],
                "engine": engine,
                "options": options,
                "gtol": gtol,
                "cap": cap,
            }
        )
    starts_path = output.with_suffix(".starts.jsonl")
    done = _load_done(starts_path) if resume else {}
    order = {job["start"]: job["index"] for job in jobs}
    kept = []
    for start, record in done.items():
        if start not in order:
            continue
        row = dict(record)
        row["index"] = order[start]
        kept.append(row)
    pending = [job for job in jobs if job["start"] not in {row["start"] for row in kept}]
    passed = {"gtol": gtol, "cap": cap, **options}
    fresh: list[dict] = []

    def checkpoint(force: bool) -> None:
        combined = kept + fresh
        if not combined:
            return
        if not force and len(fresh) % _CHECKPOINT != 0:
            return
        _publish(output, template, problem, combined, cap, version, commit, passed)

    if pending:
        if workers <= 1:
            for done_n, job in enumerate(pending, 1):
                fresh.append(_minimize_job(job))
                checkpoint(done_n == len(pending))
                if done_n == len(pending) or done_n % _CHECKPOINT == 0:
                    print(f"{template['id']} {len(kept) + done_n}/{len(jobs)}", flush=True)
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                max_tasks_per_child=1,
            ) as pool:
                futures = [pool.submit(_minimize_job, job) for job in pending]
                total = len(futures)
                for done_n, future in enumerate(as_completed(futures), 1):
                    fresh.append(future.result())
                    checkpoint(done_n == total)
                    if done_n == total or done_n % _CHECKPOINT == 0:
                        print(f"{template['id']} {len(kept) + done_n}/{len(jobs)}", flush=True)
    records = kept + fresh
    if not records:
        raise ValueError(f"{template['id']} has no starts")
    return _publish(output, template, problem, records, cap, version, commit, passed)
