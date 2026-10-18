"""Global optimization of Lennard-Jones 38 with NLopt.

The potential is the harness cluster: epsilon 1, sigma 1, cutoff 40, no
shift. One force call is one entry into ``HarnessCounter``. A repeated
coordinate is cached, so an energy read and a gradient read of the same
point are one evaluation.

The problem stop is the energy window. The template evaluation budget is
the force-call cap. NLopt is also given that budget as ``maxeval``. A
success is under the cap and within that window of the reference, on
either side. The reference energy is the harness potential on
``optbench/data/lj38_global_minimum.con``.

The harness pair loop drops a pair at distance zero. NLopt receives
that harness value. A repeated site is not a candidate minimum, so it
cannot be a success.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

MAXEVAL = 5
MAXTIME = 6
ROUNDOFF = -4
PENALTY = 1.0e6
_REFERENCE_NAME = "lj38_global_minimum.con"


def duplicate_sites(coords: np.ndarray) -> bool:
    """True when two atoms share one coordinate triple.

    Equality is bitwise. A pair at that distance is the hole the harness
    potential drops.
    """
    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    count = pos.shape[0]
    if count < 2:
        return False
    ordered = pos[np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))]
    return bool(np.any(np.all(np.diff(ordered, axis=0) == 0.0, axis=1)))


def objective_report(
    energy: float,
    gradient: np.ndarray,
    coords: np.ndarray,
) -> tuple[float, np.ndarray, bool]:
    """Value and gradient for NLopt, and whether the point can be kept.

    A non-finite value is a penalty. A repeated site keeps the harness
    value, which drops that pair, and is not a candidate.
    """
    grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if not math.isfinite(energy):
        return PENALTY, np.zeros(grad.size, dtype=np.float64), False
    return float(energy), grad, not duplicate_sites(coords)


def reference_minimum() -> tuple[float, float, np.ndarray]:
    """Energy, force norm, and coordinates of the Lennard-Jones 38 minimum."""
    path = Path(__file__).resolve().parents[1] / "data" / _REFERENCE_NAME
    pos, _box = read_con(path.read_text(encoding="utf-8"))
    energy, gradient = lennard_jones(pos)
    return float(energy), float(np.linalg.norm(gradient)), np.asarray(pos, dtype=np.float64).reshape(-1)


def search_bounds(coords: np.ndarray, half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Cube of the given half-width, centered on the geometry's centroid.

    The same limits apply to every atom, so a particle can move anywhere
    in the cube.
    """
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    center = pos.mean(axis=0)
    lower = np.tile(center - half_width, pos.shape[0])
    upper = np.tile(center + half_width, pos.shape[0])
    return lower, upper


def judge_global(
    *,
    calls: int,
    cap: int,
    energy: float | None,
    force_norm: float | None,
    finite: bool,
    nlopt_code: int | None,
    error: str,
    limit: float,
    floor: float,
    stationary_force: float,
) -> tuple[bool, str | None, list[str]]:
    """Success bit, primary failure class, and the reasons that produced it."""
    cap_error = "ForceCallCap" in error or "force-call cap" in error
    stopped_by_budget = calls >= cap or nlopt_code in (MAXEVAL, MAXTIME) or cap_error
    at_energy = (
        finite
        and energy is not None
        and float(floor) <= float(energy) <= float(limit)
        and calls < cap
        and not cap_error
    )
    if at_energy and not stopped_by_budget:
        return True, None, []
    reasons: list[str] = []
    raised = bool(error) and not cap_error and (nlopt_code is None or nlopt_code < 0)
    if raised and not stopped_by_budget:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif stopped_by_budget:
        reasons.append("cap")
    elif nlopt_code == ROUNDOFF:
        reasons.append("divergence")
    elif (
        force_norm is not None
        and math.isfinite(force_norm)
        and force_norm <= stationary_force
    ):
        reasons.append("energy")
    else:
        reasons.append("divergence")
    return False, classify_failure(reasons), reasons


def _nlopt_section(template: dict) -> dict:
    section = template.get("nlopt") or {}
    if "name" not in section:
        raise ValueError(f"{template['id']} needs nlopt.name")
    return section


def _settings(problem: dict, template: dict) -> dict:
    convergence = problem.get("convergence") or {}
    if "energy_above_minimum" not in convergence:
        raise ValueError(f"{problem['id']} has no energy window")
    window = float(convergence["energy_above_minimum"])
    stop = template.get("stop") or {}
    if "energy_above_minimum" in stop and abs(float(stop["energy_above_minimum"]) - window) > 1e-15:
        raise ValueError("template energy window is not the problem energy window")
    if "max_force_calls" not in stop:
        raise ValueError(f"{template['id']} needs stop.max_force_calls")
    cap = int(stop["max_force_calls"])
    if cap <= 0:
        raise ValueError("max_force_calls must be positive")
    domain = template.get("domain") or {}
    if "half_width" not in domain:
        raise ValueError(f"{template['id']} needs domain.half_width")
    half_width = float(domain["half_width"])
    classify = template.get("classify") or {}
    if "force_norm" not in classify:
        raise ValueError(f"{template['id']} needs classify.force_norm")
    stationary = float(classify["force_norm"])
    potential = template.get("potential") or {}
    if (
        potential.get("kind") != "lennard-jones"
        or float(potential.get("epsilon", -1)) != 1.0
        or float(potential.get("sigma", -1)) != 1.0
        or float(potential.get("rc", -1)) != 40.0
    ):
        raise ValueError(f"{template['id']} potential is not the harness Lennard-Jones cluster")
    return {
        "window": window,
        "cap": cap,
        "half_width": half_width,
        "stationary_force": stationary,
        "nlopt": _nlopt_section(template),
    }


def run_job(job: dict) -> dict:
    """One start. The NLopt seed is the start index."""
    import nlopt

    calls = 0
    verify_calls = 0
    error = ""
    code: int | None = None
    energy = None
    force_norm = None
    point = None
    seconds = 0.0
    cap = int(job["cap"])
    try:
        x0 = np.asarray(job["coords"], dtype=np.float64).reshape(-1).copy()
        lower, upper = search_bounds(x0, float(job["half_width"]))
        if np.any(x0 < lower - 1e-8) or np.any(x0 > upper + 1e-8):
            raise ValueError("start lies outside the search cube")
        counter = HarnessCounter(cache=True, cap=cap)
        incumbent = {"energy": math.inf, "x": x0.copy()}

        def consider(x, value: float, gradient: np.ndarray) -> tuple[float, np.ndarray]:
            reported, reported_grad, usable = objective_report(value, gradient, x)
            if usable and reported < incumbent["energy"]:
                incumbent["energy"] = float(reported)
                incumbent["x"] = np.asarray(x, dtype=np.float64).copy()
            return reported, reported_grad

        start_key = tuple(float(value) for value in x0)
        counter.charge(start_key)
        start_value, start_gradient = lennard_jones(x0)
        start_gradient = np.asarray(start_gradient, dtype=np.float64)
        counter.remember(start_key, (float(start_value), start_gradient))
        consider(x0, float(start_value), start_gradient)

        def objective(x, grad):
            key = tuple(float(value) for value in x)
            _fresh, cached = counter.charge(key)
            if cached is None:
                value, gradient = lennard_jones(x)
                gradient = np.asarray(gradient, dtype=np.float64)
                value = float(value)
                counter.remember(key, (value, gradient))
            else:
                value, gradient = cached
            reported, reported_grad = consider(x, float(value), np.asarray(gradient, dtype=np.float64))
            if grad.size:
                grad[:] = reported_grad
            return float(reported)

        nlopt.srand(int(job["seed"]))
        opt = nlopt.opt(getattr(nlopt, job["algorithm"]), x0.size)
        local_name = job.get("local")
        if local_name:
            local = nlopt.opt(getattr(nlopt, local_name), x0.size)
            if job.get("local_ftol_rel") is not None:
                local.set_ftol_rel(float(job["local_ftol_rel"]))
            if job.get("local_xtol_rel") is not None:
                local.set_xtol_rel(float(job["local_xtol_rel"]))
            opt.set_local_optimizer(local)
        if job.get("population"):
            opt.set_population(int(job["population"]))
        opt.set_min_objective(objective)
        opt.set_lower_bounds(lower.tolist())
        opt.set_upper_bounds(upper.tolist())
        opt.set_stopval(float(job["limit"]))
        opt.set_maxeval(cap)
        started = time.perf_counter()
        try:
            opt.optimize(x0.copy())
            code = int(opt.last_optimize_result())
        except ForceCallCap as exc:
            error = f"ForceCallCap: {exc}"
            try:
                code = int(opt.last_optimize_result())
            except Exception:
                code = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                code = int(opt.last_optimize_result())
            except Exception:
                code = None
        seconds = time.perf_counter() - started
        point = incumbent["x"]
        calls = int(counter.calls)
        with counter.verification():
            value, gradient = lennard_jones(point)
        verify_calls = int(counter.verify_calls)
        energy = float(value)
        force_norm = float(np.linalg.norm(gradient))
        if duplicate_sites(point):
            energy = None
            force_norm = None
    except Exception as exc:
        if not error:
            error = f"{type(exc).__name__}: {exc}"
        if point is None and "x0" in locals():
            point = np.asarray(x0, dtype=np.float64)
    finite = (
        energy is not None
        and force_norm is not None
        and math.isfinite(energy)
        and math.isfinite(force_norm)
        and point is not None
        and bool(np.all(np.isfinite(point)))
    )
    success, failure, reasons = judge_global(
        calls=calls,
        cap=cap,
        energy=energy,
        force_norm=force_norm,
        finite=finite,
        nlopt_code=code,
        error=error,
        limit=float(job["limit"]),
        floor=float(job["floor"]),
        stationary_force=float(job["stationary_force"]),
    )
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "seed": int(job["seed"]),
        "force_calls": calls,
        "verify_calls": verify_calls,
        "seconds": seconds,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None or not math.isfinite(energy) else float(energy),
        "force_norm": None if force_norm is None or not math.isfinite(force_norm) else float(force_norm),
        "converged": success,
        "nlopt_code": code,
        "nlopt_algorithm": job["algorithm"],
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


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, cap: int) -> str:
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
        "problem": template.get("problem"),
        "tuning": template.get("tuning", template.get("profile", "")),
        "counter": "harness",
        "cap": cap,
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "n_cap": metrics.get("n_cap"),
        "n_wrong_stationary": metrics.get("n_wrong_stationary"),
        "n_divergence": metrics.get("n_divergence"),
        "n_crash": metrics.get("n_crash"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter harness. One force call is one HarnessCounter entry.",
        "# The potential is Lennard-Jones, epsilon 1, sigma 1, cutoff 40, no shift.",
        "# A success is under the cap and within the problem energy window of the reference.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _load_geometries(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path):
    import tarfile

    from optbench.campaign import geometry_archive, iter_geometries

    if geometries is None:
        archive = geometry_archive(root, problem)
        dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".extracted"
        recorded = archive.name
        if marker.is_file():
            previous = marker.read_text(encoding="utf-8").strip()
            if previous != recorded:
                raise RuntimeError(f"{dest} was extracted from {previous}, not {recorded}")
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text(recorded, encoding="utf-8")
        cons = sorted(
            dest.rglob("*.con"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )
        if not cons:
            raise FileNotFoundError(f"{archive} has no .con files")
    else:
        cons = iter_geometries(geometries)
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected is not None and len(cons) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(cons)}")
    loaded = []
    for path in cons:
        pos, _box = read_con(path.read_text(encoding="utf-8", errors="replace"))
        loaded.append((path.stem, np.asarray(pos, dtype=np.float64).reshape(-1)))
    return loaded


def _partial_path(output: Path) -> Path:
    return output.parent / f".{output.stem}.partial.jsonl"


def _read_partial(path: Path) -> dict[int, dict]:
    found: dict[int, dict] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        found[int(record["index"])] = record
    return found


def _append_partial(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _job(index: int, name: str, coords: np.ndarray, settings: dict, limit: float, floor: float) -> dict:
    section = settings["nlopt"]
    return {
        "index": index,
        "start": name,
        "coords": coords.tolist(),
        "seed": index,
        "algorithm": str(section["name"]),
        "local": section.get("local"),
        "local_ftol_rel": section.get("local_ftol_rel"),
        "local_xtol_rel": section.get("local_xtol_rel"),
        "population": section.get("population"),
        "half_width": settings["half_width"],
        "cap": settings["cap"],
        "limit": limit,
        "floor": floor,
        "stationary_force": settings["stationary_force"],
    }


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
    import nlopt

    version = f"{nlopt.version_major()}.{nlopt.version_minor()}.{nlopt.version_bugfix()}"
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"nlopt {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if not commit:
        raise RuntimeError(f"{template['id']} has no commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    settings = _settings(problem, template)
    reference, _force, _coords = reference_minimum()
    energy_limit = reference + settings["window"]
    energy_floor = reference - settings["window"]
    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit > 0:
        loaded = loaded[:limit]
    jobs = [
        _job(index, name, coords, settings, energy_limit, energy_floor)
        for index, (name, coords) in enumerate(loaded)
    ]
    partial = _partial_path(output)
    done = _read_partial(partial) if resume else {}
    if not resume and partial.exists():
        partial.unlink()
        done = {}
    pending = [job for job in jobs if job["index"] not in done]
    records = [done[job["index"]] for job in jobs if job["index"] in done]
    if pending:
        if workers <= 1:
            fresh = []
            for job in pending:
                record = run_job(job)
                fresh.append(record)
                _append_partial(partial, record)
                print(
                    f"{template['id']} {record['start']} calls={record['force_calls']} "
                    f"failure={record['failure']} energy={record['energy']}",
                    flush=True,
                )
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            fresh = []
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(run_job, job): job for job in pending}
                total = len(jobs)
                finished = len(records)
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "schema": "optbench.start/v2",
                            "start": job["start"],
                            "index": job["index"],
                            "seed": int(job["seed"]),
                            "force_calls": 0,
                            "verify_calls": 0,
                            "seconds": 0.0,
                            "counter": "harness",
                            "comparable": True,
                            "success": False,
                            "failure": "crash",
                            "reasons": ["crash"],
                            "energy": None,
                            "force_norm": None,
                            "converged": False,
                            "nlopt_code": None,
                            "nlopt_algorithm": job["algorithm"],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    fresh.append(record)
                    _append_partial(partial, record)
                    finished += 1
                    print(
                        f"{template['id']} {finished}/{total} {record['start']} "
                        f"calls={record['force_calls']} failure={record['failure']} "
                        f"energy={record['energy']}",
                        flush=True,
                    )
        records.extend(fresh)
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    cap = settings["cap"]
    legacy_counts = [cap if not record["success"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["success"]]
    summary["median"] = float(np.median(successes)) if successes else None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
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
            for record in records
        ],
        cap,
    )
    section = settings["nlopt"]
    passed = {
        "algorithm": section["name"],
        "local": section.get("local"),
        "local_ftol_rel": section.get("local_ftol_rel"),
        "local_xtol_rel": section.get("local_xtol_rel"),
        "population": section.get("population"),
        "half_width": settings["half_width"],
        "max_force_calls": cap,
        "energy_limit": energy_limit,
        "energy_floor": energy_floor,
        "reference_energy": reference,
        "stationary_force": settings["stationary_force"],
        "seed": "start index",
    }
    passed = {key: value for key, value in passed.items() if value is not None}
    provenance = collect_provenance(
        code=str(template.get("code")),
        code_version=version,
        commit=commit,
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "campaign",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter on the Lennard-Jones cluster. "
            "A success is within the energy window of the reference and under the evaluation budget."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", template.get("profile", "")),
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
                "starts": records,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, cap), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if partial.exists() and len(records) == len(jobs):
        partial.unlink()
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["v2_successes"] = metrics["successes"]
    return summary


def write_asv_result(path: Path, commit: str, rows: dict[str, list[dict]], *, env_name: str, machine: dict) -> None:
    """Write one asv result file from campaign records.

    ``rows`` maps a benchmark class name to start records in start-index
    order. Track values and ``time_search`` are those records.
    """
    started = int(time.time() * 1000)
    results = {}
    for class_name, records in rows.items():
        if not records:
            raise ValueError(f"{class_name} has no starts")
        params = [["'0'"], [repr(str(index)) for index in range(len(records))]]
        calls = [float(record["force_calls"]) for record in records]
        seconds = [float(record["seconds"]) for record in records]
        success = [1.0 if record["success"] else 0.0 for record in records]
        failure = {
            name: [1.0 if record["failure"] == name else 0.0 for record in records]
            for name in ("cap", "wrong_stationary", "divergence", "crash")
        }
        prefix = f"bench_nloptglobal.{class_name}"
        version = "nloptglobal-1"
        duration = float(sum(seconds))
        tracks = {
            "track_force_calls": calls,
            "track_success": success,
            "track_failure_cap": failure["cap"],
            "track_failure_wrong_stationary": failure["wrong_stationary"],
            "track_failure_divergence": failure["divergence"],
            "track_failure_crash": failure["crash"],
        }
        for method, values in tracks.items():
            results[f"{prefix}.{method}"] = [values, params, version, started, duration]
        results[f"{prefix}.time_search"] = [seconds, params, version, started, duration]
    document = {
        "commit_hash": commit,
        "env_name": env_name,
        "date": started,
        "params": machine,
        "python": machine.get("python", "3.12"),
        "requirements": {},
        "env_vars": {},
        "result_columns": [
            "result",
            "params",
            "version",
            "started_at",
            "duration",
            "stats_ci_99_a",
            "stats_ci_99_b",
            "stats_q_25",
            "stats_q_75",
            "stats_number",
            "stats_repeat",
            "samples",
            "profile",
        ],
        "results": results,
        "durations": {},
        "version": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
