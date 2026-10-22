"""Minimize OptBench starts with pele on the harness potential.

LBFGS and MYLBFGS are ``pele.optimize.LBFGS`` and ``MYLBFGS``. FIRE is
``pele.optimize.Fire``. Steepest descent is the update in
``pele.optimize.steepest_descent``: a step ``-dx * gradient``, shortened
when ``maxstep`` is positive and the largest absolute gradient component
exceeds it.

The potential is the problem potential. One force call is one
``getEnergyGradient`` entry on a ``HarnessCounter``. A repeated
coordinate is cached. The stop is the Euclidean norm of the gradient
at the problem force norm. A v2 success also has Hessian index 0. On
LJ38 that index drops the six rigid modes. Energy and Kabsch RMSD are
not applied: the problem files do not give those tolerances or a
per-start reference.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
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
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance, write_starts

EIGENVALUE_TOL = 1e-6
PELE_COMMIT = "b7081b932a060be11342fbe8287ac35da8ada879"
PELE_VERSION = "0.1.2"

# Constructor values at that commit. A template key replaces one of them.
_DEFAULTS: dict[str, dict] = {
    "lbfgs": {"M": 4, "maxstep": 0.1, "maxErise": 1e-4, "H0": 0.1},
    "mylbfgs": {"M": 4, "maxstep": 0.1, "maxErise": 1e-4, "H0": 0.1},
    "fire": {
        "dt": 0.1,
        "maxstep": 0.5,
        "dtmax": 1.0,
        "Nmin": 5,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.1,
        "fa": 0.99,
        "a": 0.1,
    },
    "sd": {"dx": 1e-4, "maxstep": -1.0},
}
_INT_KEYS = frozenset({"M", "Nmin"})
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _quiet() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"
    logging.getLogger("pele").setLevel(logging.CRITICAL)


def merged_options(method: str, raw: dict) -> dict:
    """Constructor values with the template keys written over them."""
    if method not in _DEFAULTS:
        raise ValueError(f"unknown pele method {method}")
    unknown = set(raw) - set(_DEFAULTS[method])
    if unknown:
        raise ValueError(f"{method} has no option {sorted(unknown)}")
    merged = dict(_DEFAULTS[method])
    merged.update(raw)
    for key in _INT_KEYS:
        if key in merged:
            merged[key] = int(merged[key])
    for key, value in list(merged.items()):
        if key not in _INT_KEYS:
            merged[key] = float(value)
    return merged


class CountedPelePotential:
    """``getEnergyGradient`` on a raw ``(energy, gradient)`` function.

    The counter charges the entry. ``last`` is the last coordinates that
    were evaluated. A refused cap call leaves ``last`` where it was.
    """

    def __init__(self, evaluate, counter: HarnessCounter):
        self._evaluate = evaluate
        self.counter = counter
        self.last: list[float] | None = None

    def getEnergyGradient(self, coords):
        import numpy as np

        flat = np.asarray(coords, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in flat)
        fresh, cached = self.counter.charge(key)
        if not fresh and cached is not None:
            self.last = list(key)
            return cached[0], np.asarray(cached[1], dtype=np.float64)
        energy, gradient = self._evaluate(flat)
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        value = (float(energy), tuple(float(item) for item in grad))
        self.counter.remember(key, value)
        self.last = list(key)
        return value[0], grad

    def getEnergy(self, coords):
        energy, _gradient = self.getEnergyGradient(coords)
        return energy

    def getGradient(self, coords):
        _energy, gradient = self.getEnergyGradient(coords)
        return gradient


def _steepest(x0, potential, *, dx: float, maxstep: float, cap: int, stop) -> tuple[list[float], bool]:
    """One steepest-descent quench.

    The update is the body of ``pele.optimize.steepest_descent``. The
    library tests the RMS gradient. This loop tests ``stop``, which is
    the Euclidean norm from the problem file. The first evaluation is
    taken before that test, as in the library.
    """
    import numpy as np

    x = np.asarray(x0, dtype=np.float64).copy()
    _energy, gradient = potential.getEnergyGradient(x)
    stopped = False
    while potential.counter.calls < cap:
        step = -gradient * dx
        if maxstep > 0.0:
            largest = float(np.max(np.abs(gradient)))
            if largest > maxstep:
                step *= maxstep / largest
        x = x + step
        energy, gradient = potential.getEnergyGradient(x)
        if stop(energy=energy, gradient=gradient, tol=None, coords=x):
            stopped = True
            break
    return [float(value) for value in x], stopped


def drive_minimizer(method: str, coords, potential, *, stop, cap: int, options: dict) -> tuple[list[float] | None, bool, str]:
    """Run one quench. Returns the last point, whether ``stop`` held, and an error string."""
    _quiet()
    import numpy as np

    try:
        from pele.optimize import Fire, LBFGS, MYLBFGS
    except ImportError as exc:
        return None, False, f"ImportError: {exc}"
    point = list(potential.last) if potential.last is not None else None
    try:
        x = np.asarray(coords, dtype=np.float64).reshape(-1).copy()
        if method in {"lbfgs", "mylbfgs"}:
            klass = LBFGS if method == "lbfgs" else MYLBFGS
            optimizer = klass(
                x,
                potential,
                nsteps=int(cap),
                tol=1e-5,
                iprint=-1,
                alternate_stop_criterion=stop,
                M=int(options["M"]),
                maxstep=float(options["maxstep"]),
                maxErise=float(options["maxErise"]),
                H0=float(options["H0"]),
            )
            result = optimizer.run()
            point = [float(value) for value in np.asarray(result.coords, dtype=np.float64).reshape(-1)]
            return point, bool(result.success), ""
        if method == "fire":
            optimizer = Fire(
                x,
                potential,
                iprint=-1,
                alternate_stop_criterion=stop,
                dt=float(options["dt"]),
                maxstep=float(options["maxstep"]),
                dtmax=float(options["dtmax"]),
                Nmin=int(options["Nmin"]),
                finc=float(options["finc"]),
                fdec=float(options["fdec"]),
                astart=float(options["astart"]),
                fa=float(options["fa"]),
                a=float(options["a"]),
            )
            result = optimizer.run(fmax=1e-3, steps=int(cap))
            point = [float(value) for value in np.asarray(result.coords, dtype=np.float64).reshape(-1)]
            return point, bool(result.success), ""
        if method == "sd":
            point, stopped = _steepest(
                x,
                potential,
                dx=float(options["dx"]),
                maxstep=float(options["maxstep"]),
                cap=int(cap),
                stop=stop,
            )
            return point, stopped, ""
        return point, False, f"ValueError: unknown pele method {method}"
    except ForceCallCap as exc:
        if potential.last is not None:
            point = list(potential.last)
        return point, False, f"ForceCallCap: {exc}"
    except Exception as exc:
        if potential.last is not None:
            point = list(potential.last)
        return point, False, f"{type(exc).__name__}: {exc}"


def _kind(problem_id: str) -> str:
    if problem_id == "minimization/lj38":
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no pele potential")


def _section(template: dict) -> tuple[str, dict]:
    section = template.get("pele") or {}
    if "method" not in section:
        raise ValueError(f"{template['id']} needs pele.method")
    method = str(section["method"])
    raw = {key: value for key, value in section.items() if key != "method"}
    return method, merged_options(method, raw)


def _l2_stop(force_norm: float):
    def stop(*, energy, gradient, tol, coords=None) -> bool:
        import numpy as np

        return float(np.linalg.norm(gradient)) <= force_norm

    return stop


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


def _minimize_job(job: dict) -> dict:
    _quiet()
    import numpy as np

    started = time.perf_counter()
    calls = 0
    verify_calls = 0
    error = ""
    energy = None
    force_norm = None
    point = None
    potential = None
    stopped = False
    try:
        x0 = [float(value) for value in job["coords"]]
        box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
        kind = job["kind"]
        cap = int(job["cap"])
        gtol = float(job["gtol"])

        def evaluate(positions):
            if kind == "lj":
                return lennard_jones(positions)
            return morse_bulk(positions, box)

        counter = HarnessCounter(cache=True, cap=cap)
        potential = CountedPelePotential(evaluate, counter)
        point, stopped, error = drive_minimizer(
            str(job["method"]),
            x0,
            potential,
            stop=_l2_stop(gtol),
            cap=cap,
            options=job["options"],
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if potential is not None:
        calls = int(potential.counter.calls)
        if point is not None:
            try:
                with potential.counter.verification():
                    energy, gradient = potential.getEnergyGradient(point)
                force_norm = float(np.linalg.norm(gradient))
                verify_calls = int(potential.counter.verify_calls)
            except Exception as exc:
                if not error:
                    error = f"{type(exc).__name__}: {exc}"
    gtol = float(job["gtol"])
    cap = int(job["cap"])
    hessian_index = None
    min_eigenvalues: list[float] = []
    cap_hit = "force-call cap" in error or "ForceCallCap" in error
    if point is not None and (not error or cap_hit):
        try:
            hessian_index, min_eigenvalues = _index_at(job["kind"], point, job["box"], gtol)
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
    at_force = finite and stopped and force_norm is not None and force_norm <= gtol and calls < cap and not error
    reasons: list[str] = []
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
        "min_eigenvalues": min_eigenvalues,
        "seconds": time.perf_counter() - started,
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


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, partial: bool) -> str:
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
        "problem": template.get("problem"),
        "par2_mean": metrics.get("par2_mean"),
        "rate_low": metrics.get("rate_low"),
        "rate_high": metrics.get("rate_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one getEnergyGradient entry.",
        "# The potential is the problem potential. A repeated coordinate is cached.",
        "# The stop is the Euclidean norm of the gradient.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
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


def _installed_version() -> str:
    from importlib.metadata import version

    return version("pele")


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
) -> dict:
    version = _installed_version()
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"pele {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if commit != PELE_COMMIT:
        raise RuntimeError(f"{template['id']} commit {commit} is not the built pele revision {PELE_COMMIT}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    method, options = _section(template)
    return _run(
        problem,
        template,
        kind=_kind(problem["id"]),
        method=method,
        options=options,
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
    )


def _load_geometries(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path):
    from optbench.runners.rgmin_min import _load_geometries as load

    return load(root, problem, geometries, work, output)


def _run(
    problem,
    template,
    *,
    kind,
    method,
    options,
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
) -> dict:
    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    if problem["id"] == "minimization/morse-bulk" and geometries is None and not limit and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if kind == "lj" else box.tolist(),
                "kind": kind,
                "method": method,
                "options": options,
                "gtol": gtol,
                "cap": cap,
            }
        )
    if workers <= 1:
        records = [_minimize_job(job) for job in jobs]
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_minimize_job, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                if done == total or done % 50 == 0:
                    print(f"{template['id']} {done}/{total}", flush=True)
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    expected = (problem.get("sample") or {}).get("starts")
    if problem["id"] == "minimization/morse-bulk" and expected is None:
        expected = 100
    partial = bool(limit) or (expected is not None and len(records) < int(expected))
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["converged"]]
    summary["median"] = statistics.median(successes) if successes else None
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
    passed = {"method": method, "gtol": gtol, "maxiter": cap, **options}
    provenance = collect_provenance(
        code="pele",
        code_version=version,
        commit=commit,
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. The potential is the problem potential. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "pele",
                "algorithm": template.get("algorithm", method),
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
                "starts": records,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, partial), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary
