"""Minimize OptBench starts with NLopt and ``RgpotCounter``.

The objective callback is one ``RgpotCounter.calculate`` on a
``HarnessCounter``. A repeated coordinate is cached and is not a second
force call. The harness potential is the problem potential. Lennard-Jones
38 guards the first step, because an identity-Hessian step along the raw
gradient leaves the cutoff and the force falls to zero.

``LD_LBFGS``, ``LD_VAR2``, and ``LD_SLSQP`` multiply the objective by
``maxstep / ||g(x0)||`` when the initial force exceeds ``maxstep``. The
first identity-Hessian step then has length ``maxstep``. Luksan ``tolg``
is set to ``1e-8`` times that factor, which keeps the unscaled gradient
tolerance at the library default. ``LD_TNEWTON_PRECOND_RESTART`` is a
Newton step, so the same factor does not limit it. Each coordinate stays
in a box of half-width ``maxstep`` that recenters until the force stop.
``LD_MMA`` and every Morse row use the library defaults. The stop checked
by the callback is the unscaled problem force norm.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import (
    lennard_jones,
    lennard_jones_hessian,
    morse_bulk,
    morse_bulk_hessian,
    read_con,
)
from optbench.potserver import RgpotCounter
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance, write_starts

EIGENVALUE_TOL = 1e-6
DEFAULT_PREFIX = "/opt/nlopt"
MAXSTEP = 0.2
LUKSAN_TOLG = 1e-8
ALGORITHMS = (
    "LD_LBFGS",
    "LD_MMA",
    "LD_SLSQP",
    "LD_TNEWTON_PRECOND_RESTART",
    "LD_VAR2",
)
LUKSAN = frozenset({"LD_LBFGS", "LD_VAR2", "LD_TNEWTON_PRECOND_RESTART"})
SCALE_ALGORITHMS = frozenset({"LD_LBFGS", "LD_VAR2", "LD_SLSQP"})
BOX_ALGORITHMS = frozenset({"LD_TNEWTON_PRECOND_RESTART"})
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_RESULT_NAMES = {
    1: "SUCCESS",
    2: "STOPVAL_REACHED",
    3: "FTOL_REACHED",
    4: "XTOL_REACHED",
    5: "MAXEVAL_REACHED",
    6: "MAXTIME_REACHED",
    -1: "FAILURE",
    -2: "INVALID_ARGS",
    -3: "OUT_OF_MEMORY",
    -4: "ROUNDOFF_LIMITED",
    -5: "FORCED_STOP",
}


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def load_nlopt():
    """Import the source build installed under ``NLOPT_PREFIX``."""
    prefix = os.environ.get("NLOPT_PREFIX", DEFAULT_PREFIX)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site = os.path.join(prefix, "lib", f"python{version}", "site-packages")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    try:
        import nlopt
    except ImportError as exc:
        raise ImportError(f"nlopt is not importable from {site}") from exc
    commit_path = os.path.join(prefix, "COMMIT")
    built = ""
    if os.path.isfile(commit_path):
        built = Path(commit_path).read_text(encoding="utf-8").strip()
    return nlopt, built


def _kind(problem_id: str) -> str:
    if problem_id in {"minimization/lj38", "minimization/lj38-noise"}:
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no nlopt potential")


def require_step(problem_id: str, algorithm: str, step: str, maxstep: float | None) -> None:
    """The LJ38 step guard is part of the comparable row."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown nlopt algorithm {algorithm}")
    if step not in {"none", "scale", "box"}:
        raise ValueError(f"unknown nlopt step {step}")
    kind = _kind(problem_id)
    if kind == "lj":
        if algorithm in SCALE_ALGORITHMS:
            expected = "scale"
        elif algorithm in BOX_ALGORITHMS:
            expected = "box"
        else:
            expected = "none"
        if step != expected:
            raise ValueError(f"{problem_id} {algorithm} uses step {expected}")
        if expected == "none":
            if maxstep is not None:
                raise ValueError(f"{problem_id} {algorithm} has no maxstep")
            return
        if maxstep is None or abs(float(maxstep) - MAXSTEP) > 1e-12:
            raise ValueError(f"{problem_id} {algorithm} maxstep is {MAXSTEP}")
        return
    if step != "none" or maxstep is not None:
        raise ValueError(f"{problem_id} {algorithm} uses the library defaults")


def _options(template: dict) -> tuple[str, str, float | None]:
    section = template.get("nlopt") or {}
    algorithm = str(section.get("algorithm") or "")
    step = str(section.get("step") or "none")
    raw = section.get("maxstep")
    maxstep = None if raw is None else float(raw)
    require_step(str(template.get("problem") or ""), algorithm, step, maxstep)
    return algorithm, step, maxstep


def _index_at(kind: str, point: list[float], box, force_stop: float) -> tuple[int, list[float]]:
    import numpy as np

    if kind == "lj":
        matrix = lennard_jones_hessian(point)
        values = vibrational_eigenvalues(matrix, point, rotations=True)
        tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    else:
        matrix = morse_bulk_hessian(point, box)
        values = np.linalg.eigvalsh(matrix).tolist()
        tolerance = EIGENVALUE_TOL
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _result_name(nlopt, code: int | None) -> str:
    if code is None:
        return ""
    return _RESULT_NAMES.get(int(code), str(code))


def drive(
    x0,
    evaluate,
    *,
    algorithm: str,
    step: str,
    maxstep: float | None,
    stop,
    cap: int,
) -> dict:
    """Run one minimization.

    ``evaluate(coords)`` returns ``(energy, gradient)`` and charges the
    harness counter. ``stop(point, energy, gradient, norm)`` is the
    unscaled convergence test. ``ForceCallCap`` ends the search at the
    budget. The returned point is the first point that met ``stop``, or
    the last finite point when the search stops another way.
    """
    import numpy as np

    nlopt, _built = load_nlopt()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown nlopt algorithm {algorithm}")
    if not hasattr(nlopt, algorithm):
        raise ValueError(f"this nlopt build has no {algorithm}")
    x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    holder: dict = {"x": x.copy(), "met": None, "scale": 1.0, "rounds": 0}
    code: int | None = None
    error = ""

    def remember(point) -> None:
        if holder["met"] is None:
            holder["x"] = np.asarray(point, dtype=np.float64).copy()

    def one(start, scale: float, bounds) -> tuple[int | None, object]:
        opt = nlopt.opt(getattr(nlopt, algorithm), int(start.size))
        if algorithm in LUKSAN:
            opt.set_param("tolg", LUKSAN_TOLG * float(scale))
        if bounds is not None:
            lower, upper = bounds
            opt.set_lower_bounds([float(value) for value in lower])
            opt.set_upper_bounds([float(value) for value in upper])

        def objective(coords, grad):
            energy, gradient = evaluate(coords)
            g = np.asarray(gradient, dtype=np.float64).reshape(-1)
            point = np.asarray(coords, dtype=np.float64).reshape(-1).copy()
            if grad.size > 0:
                grad[:] = g * float(scale)
            remember(point)
            norm = float(np.linalg.norm(g)) if np.all(np.isfinite(g)) else math.inf
            if stop(point, float(energy), g, norm):
                holder["met"] = point
                holder["x"] = point
                opt.force_stop()
            if not math.isfinite(float(energy)):
                raise ValueError("non-finite energy")
            return float(energy) * float(scale)

        opt.set_min_objective(objective)
        try:
            out = opt.optimize(np.asarray(start, dtype=np.float64).copy())
            return int(opt.last_optimize_result()), out
        except ForceCallCap:
            raise
        except nlopt.ForcedStop:
            return int(nlopt.FORCED_STOP), holder["x"]
        except nlopt.RoundoffLimited:
            return int(nlopt.ROUNDOFF_LIMITED), holder["x"]

    try:
        scale = 1.0
        if step == "scale":
            if maxstep is None:
                raise ValueError("scale step needs maxstep")
            energy, gradient = evaluate(x)
            g = np.asarray(gradient, dtype=np.float64).reshape(-1)
            gnorm = float(np.linalg.norm(g)) if np.all(np.isfinite(g)) else math.inf
            scale = 1.0 if (not math.isfinite(gnorm) or gnorm <= float(maxstep)) else float(maxstep) / gnorm
            holder["scale"] = scale
            remember(x)
            if stop(x, float(energy), g, gnorm):
                holder["met"] = x.copy()
                holder["x"] = x.copy()
            else:
                code, out = one(x, scale, None)
                if holder["met"] is None and isinstance(out, np.ndarray) and np.all(np.isfinite(out)):
                    holder["x"] = np.asarray(out, dtype=np.float64).reshape(-1)
                holder["rounds"] = 1
        elif step == "box":
            if maxstep is None:
                raise ValueError("box step needs maxstep")
            delta = float(maxstep)
            while True:
                base = x.copy()
                code, out = one(x, 1.0, (x - delta, x + delta))
                holder["rounds"] += 1
                if holder["met"] is not None:
                    break
                if isinstance(out, np.ndarray) and np.all(np.isfinite(out)):
                    x = np.asarray(out, dtype=np.float64).reshape(-1)
                else:
                    x = np.asarray(holder["x"], dtype=np.float64).copy()
                moved = float(np.max(np.abs(x - base))) if x.size else 0.0
                if moved < 1e-14 or holder["rounds"] > cap:
                    break
        elif step == "none":
            code, out = one(x, 1.0, None)
            holder["rounds"] = 1
            if holder["met"] is None and isinstance(out, np.ndarray) and np.all(np.isfinite(out)):
                holder["x"] = np.asarray(out, dtype=np.float64).reshape(-1)
        else:
            raise ValueError(f"unknown nlopt step {step}")
    except ForceCallCap as exc:
        error = f"ForceCallCap: {exc}"
        code = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    point = holder["met"] if holder["met"] is not None else holder["x"]
    return {
        "point": [float(value) for value in np.asarray(point, dtype=np.float64).reshape(-1)],
        "stopped": holder["met"] is not None and not error,
        "code": _result_name(nlopt, code),
        "error": error,
        "rounds": int(holder["rounds"]),
        "scale": float(holder["scale"]),
    }


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    import numpy as np

    started = time.perf_counter()
    calls = 0
    verify_calls = 0
    error = ""
    energy = None
    force_norm = None
    point = None
    potential = None
    result = None
    try:
        x0 = [float(value) for value in job["coords"]]
        box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
        kind = job["kind"]
        cap = int(job["cap"])
        gtol = float(job["gtol"])

        def evaluate_potential(positions, _numbers=None, _box=None):
            if kind == "lj":
                return lennard_jones(positions)
            return morse_bulk(positions, box)

        counter = HarnessCounter(cache=True, cap=cap)
        potential = RgpotCounter(evaluate_potential, counter)

        def evaluate(coords):
            value, gradient = potential.calculate(coords)
            return float(value), np.asarray(gradient, dtype=np.float64)

        def stop(_point, _energy, _gradient, norm: float) -> bool:
            return math.isfinite(norm) and norm <= gtol

        result = drive(
            x0,
            evaluate,
            algorithm=str(job["algorithm"]),
            step=str(job["step"]),
            maxstep=job["maxstep"],
            stop=stop,
            cap=cap,
        )
        point = result["point"]
        error = str(result["error"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if "x0" in locals():
            point = [float(value) for value in x0]
    if potential is not None:
        calls = int(potential.counter.calls)
        if point is not None:
            try:
                with potential.counter.verification():
                    energy, gradient = potential.calculate(point)
                force_norm = float(np.linalg.norm(gradient))
                verify_calls = int(potential.counter.verify_calls)
            except Exception as exc:
                if not error:
                    error = f"{type(exc).__name__}: {exc}"
    gtol = float(job["gtol"])
    cap = int(job["cap"])
    hessian_index = None
    min_eigenvalues: list[float] = []
    cap_hit = "ForceCallCap" in error or "force-call cap" in error
    if point is not None and (not error or cap_hit):
        try:
            hessian_index, min_eigenvalues = _index_at(job["kind"], point, job["box"], gtol)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    stopped = bool(result and result["stopped"])
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
    if error and not cap_hit and calls < cap:
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
        "min_eigenvalues": min_eigenvalues,
        "seconds": time.perf_counter() - started,
        "nlopt_algorithm": job["algorithm"],
        "nlopt_step": job["step"],
        "nlopt_maxstep": job["maxstep"],
        "nlopt_result": None if result is None else result["code"],
        "nlopt_rounds": None if result is None else result["rounds"],
        "nlopt_scale": None if result is None else result["scale"],
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
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
        "# energy_min and energy_max are the final potential values on the recorded points.",
    ]
    if str(template.get("problem")) == "minimization/lj38":
        lines.append(
            "# LJ38 index removes the six rigid modes. A curvature is negative only below "
            "-max(force_norm, 1e-8*||H||_F, 1e-6)."
        )
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


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
    nlopt, built = load_nlopt()
    version = str(nlopt.__version__)
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"nlopt {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if not commit:
        raise RuntimeError(f"{template['id']} has no commit")
    if built and not (commit.startswith(built) or built.startswith(commit)):
        raise RuntimeError(f"nlopt build {built} does not match template {commit}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    algorithm, step, maxstep = _options(template)
    return _run(
        problem,
        template,
        kind=_kind(problem["id"]),
        algorithm=algorithm,
        step=step,
        maxstep=maxstep,
        gtol=gtol,
        cap=cap,
        geometries=geometries,
        output=output,
        workers=workers,
        work=work,
        version=version,
        commit=commit,
        root=root,
        limit=limit,
        resume=resume,
    )


def _load_geometries(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path):
    from optbench.runners.rgmin_min import _load_geometries as load

    return load(root, problem, geometries, work, output)


def _checkpoint(output: Path, records: list[dict], problem, template, summary_bits: dict) -> None:
    """Write the campaign files for the starts finished so far."""
    _emit(output, records, problem, template, summary_bits)


def _emit(output: Path, records: list[dict], problem, template, bits: dict) -> None:
    ordered = sorted(records, key=lambda row: int(row.get("index", 0)))
    stored = []
    for record in ordered:
        row = dict(record)
        row.pop("index", None)
        stored.append(row)
    cap = int(bits["cap"])
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in stored]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in stored if record["converged"]]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in stored if record.get("energy") is not None]
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
            for record in stored
        ],
        cap,
    )
    passed = {
        "algorithm": bits["algorithm"],
        "step": bits["step"],
        "maxstep": bits["maxstep"],
        "gtol": bits["gtol"],
        "cap": cap,
    }
    provenance = collect_provenance(
        code="nlopt",
        code_version=bits["version"],
        commit=bits["commit"],
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
            "Harness counter. The callback evaluates the problem potential. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "nlopt",
                "algorithm": template.get("algorithm", bits["algorithm"]),
                "code_version": bits["version"],
                "commit": bits["commit"],
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
                "starts": stored,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, bits["commit"], bits["version"]), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), stored)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(stored)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    return summary


def _resume_rows(output: Path, algorithm: str, step: str, maxstep) -> list[dict]:
    path = output.with_suffix(".starts.jsonl")
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("nlopt_algorithm") != algorithm or row.get("nlopt_step") != step:
            raise RuntimeError(f"{path} was produced with a different nlopt step")
        recorded = row.get("nlopt_maxstep")
        if maxstep is None:
            if recorded not in (None, ""):
                raise RuntimeError(f"{path} records a maxstep")
        elif recorded is None or abs(float(recorded) - float(maxstep)) > 1e-12:
            raise RuntimeError(f"{path} maxstep does not match the template")
        rows.append(row)
    return rows


def _run(
    problem,
    template,
    *,
    kind,
    algorithm,
    step,
    maxstep,
    gtol,
    cap,
    geometries,
    output,
    workers,
    work,
    version,
    commit,
    root,
    limit,
    resume,
) -> dict:
    _quiet_threads()
    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    if problem["id"] == "minimization/morse-bulk" and geometries is None and not limit and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    done: dict[str, dict] = {}
    if resume:
        for row in _resume_rows(output, algorithm, step, maxstep):
            done[str(row["start"])] = row
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        if name in done:
            done[name]["index"] = index
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if kind == "lj" or box is None else box.tolist(),
                "kind": kind,
                "algorithm": algorithm,
                "step": step,
                "maxstep": maxstep,
                "gtol": gtol,
                "cap": cap,
            }
        )
    bits = {
        "algorithm": algorithm,
        "step": step,
        "maxstep": maxstep,
        "gtol": gtol,
        "cap": cap,
        "version": version,
        "commit": commit,
    }
    finished = list(done.values())

    def on_batch(fresh: list[dict]) -> None:
        _checkpoint(output, finished + fresh, problem, template, bits)

    fresh = _map_jobs(template["id"], jobs, workers, on_batch=on_batch) if jobs else []
    records = finished + fresh
    if records:
        _checkpoint(output, records, problem, template, bits)
    return _emit(output, records, problem, template, bits)


def _map_jobs(template_id: str, jobs: list[dict], workers: int, on_batch=None) -> list[dict]:
    def flush(fresh: list[dict], done: int, total: int) -> None:
        if done == total or done % 50 == 0:
            print(f"{template_id} {done}/{total}", flush=True)
            if on_batch is not None:
                on_batch(list(fresh))

    if workers <= 1:
        fresh = []
        total = len(jobs)
        for done, job in enumerate(jobs, 1):
            fresh.append(_minimize_job(job))
            flush(fresh, done, total)
        return fresh
    import multiprocessing as mp

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_minimize_job, job) for job in jobs]
        fresh = []
        total = len(futures)
        for done, future in enumerate(as_completed(futures), 1):
            fresh.append(future.result())
            flush(fresh, done, total)
        return fresh
