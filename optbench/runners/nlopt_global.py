"""Global optimization of Lennard-Jones clusters with NLopt.

The potential is the harness cluster: epsilon 1, sigma 1, cutoff 40, no
shift. One force call is one entry into ``HarnessCounter``. A repeated
coordinate is cached, so an energy read and a gradient read of the same
point are one evaluation.

The problem stop is the Cambridge Cluster Database rule: a finite energy
at or below the published minimum plus ``energy_above_minimum``. A lower
energy still succeeds. The template evaluation budget is the force-call
cap. A cube search gives NLopt that budget as ``maxeval`` and that
threshold as ``stopval``.

A coincident pair in the harness potential is a large finite energy.
The recorded geometry is the point the search returned, and the recorded
energy is the harness value of that point.

Basin hopping uses NLopt LD_LBFGS as the quench. Luksan's L-BFGS has no
step cap, and one long step dissociates the cluster, so each trial step
is a cube of half-width ``maxstep`` recentered on the current point.
Pairs inside ``core_radius`` use a quadratic stand-in matched to the
12-6 value and slope. Outside that radius the search value is the
harness potential.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.globalopt import energy_tolerance, judge_globalopt, reference_energy
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

MAXEVAL = 5
MAXTIME = 6
PENALTY = 1.0e6
_REFERENCE_NAME = "lj38_global_minimum.con"


def duplicate_sites(coords: np.ndarray) -> bool:
    """True when two atoms share one coordinate triple.

    Equality is bitwise. The harness potential gives that pair a large
    finite energy.
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
    """Value and gradient NLopt sees, and that the point stays a candidate.

    A non-finite value becomes a large finite penalty. A coincident pair
    already has a large finite harness energy, and that value is kept.
    """
    del coords
    grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if not math.isfinite(energy) or not bool(np.all(np.isfinite(grad))):
        return PENALTY, np.zeros(grad.size, dtype=np.float64), True
    return float(energy), grad, True


def reference_minimum() -> tuple[float, float, np.ndarray]:
    """Energy, force norm, and coordinates of the Lennard-Jones 38 minimum."""
    path = Path(__file__).resolve().parents[1] / "data" / _REFERENCE_NAME
    pos, _box = read_con(path.read_text(encoding="utf-8"))
    energy, gradient = lennard_jones(pos)
    return float(energy), float(np.linalg.norm(gradient)), np.asarray(pos, dtype=np.float64).reshape(-1)


def capped_lennard_jones(
    coords: np.ndarray,
    core_radius: float,
) -> tuple[float, np.ndarray, float]:
    """Harness 12-6, with a quadratic core inside ``core_radius``.

    The core matches the 12-6 value and ``dV/dr`` at the join, so the
    search gradient stays finite when two atoms overlap. The returned
    minimum pair distance is infinite when the cluster has one atom.
    """
    if core_radius <= 0.0:
        raise ValueError("core_radius must be positive")
    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    iu, ju = np.triu_indices(pos.shape[0], k=1)
    diff = pos[iu] - pos[ju]
    r2 = np.einsum("ij,ij->i", diff, diff)
    energy = 0.0
    slope = np.zeros(r2.shape, dtype=np.float64)
    core2 = core_radius * core_radius
    core = r2 < core2
    physical = (~core) & (r2 > 0.0) & (r2 < 40.0 * 40.0)
    if np.any(physical):
        rr = r2[physical]
        inv = 1.0 / rr
        sr6 = inv * inv * inv
        sr12 = sr6 * sr6
        energy += float(4.0 * np.sum(sr12 - sr6))
        slope[physical] = 24.0 * (sr6 - 2.0 * sr12) * inv
    if np.any(core):
        inv = 1.0 / core2
        sr6 = inv * inv * inv
        sr12 = sr6 * sr6
        value = 4.0 * (sr12 - sr6)
        derivative = 24.0 * (sr6 - 2.0 * sr12) * inv * core_radius
        curvature = derivative / (2.0 * core_radius)
        energy += float(np.sum(value + curvature * (r2[core] - core2)))
        slope[core] = derivative / core_radius
    grad = np.zeros_like(pos)
    piece = slope[:, None] * diff
    np.add.at(grad, iu, piece)
    np.add.at(grad, ju, -piece)
    minimum = float(np.sqrt(r2.min())) if r2.size else math.inf
    return float(energy), grad.reshape(-1), minimum


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
    finite: bool,
    nlopt_code: int | None,
    error: str,
    reference: float,
    tolerance: float,
) -> tuple[bool, str | None, list[str]]:
    """Cambridge energy rule, with a crash when the search raised under the cap."""
    cap_error = "ForceCallCap" in error or "force-call cap" in error
    stopped_by_budget = calls >= cap or nlopt_code in (MAXEVAL, MAXTIME) or cap_error
    raised = bool(error) and not cap_error and (nlopt_code is None or nlopt_code < 0)
    if raised and not stopped_by_budget:
        return False, "crash", ["crash"]
    judged = judge_globalopt(
        energy if finite else None,
        calls,
        reference=reference,
        tolerance=tolerance,
        cap=cap,
    )
    return bool(judged["success"]), judged["failure"], list(judged["reasons"])


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
    section = _nlopt_section(template)
    if "displacement" in section:
        for name in ("displacement", "temperature", "maxstep", "tolg", "core_radius"):
            if name not in section:
                raise ValueError(f"{template['id']} needs nlopt.{name}")
            if float(section[name]) <= 0.0:
                raise ValueError(f"{template['id']} nlopt.{name} must be positive")
        if float(section["core_radius"]) >= 1.0:
            raise ValueError(f"{template['id']} core would move the Lennard-Jones well")
        if "local_maxeval" not in section or int(section["local_maxeval"]) <= 0:
            raise ValueError(f"{template['id']} needs a positive nlopt.local_maxeval")
        half_width = float(domain["half_width"]) if "half_width" in domain else None
    else:
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
        "nlopt": section,
    }


def run_job(job: dict) -> dict:
    """One start. The NLopt seed is the start index."""
    if job.get("displacement") is not None:
        return _run_basin_hop(job)
    return _run_cube(job)


def _search_value(
    counter: HarnessCounter,
    coords: np.ndarray,
    core_radius: float,
) -> tuple[float, np.ndarray, float]:
    flat = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1)
    key = flat.tobytes()
    _fresh, cached = counter.charge(key)
    if cached is None:
        from optbench.fastpot import lj_capped_gradient

        native = lj_capped_gradient(flat, core_radius)
        if native is None:
            value, gradient, minimum = capped_lennard_jones(flat, core_radius)
            gradient = np.asarray(gradient, dtype=np.float64).copy()
        else:
            value, gradient, minimum = native
            gradient = np.asarray(gradient, dtype=np.float64)
        counter.remember(key, (float(value), gradient, float(minimum)))
    else:
        value, gradient, minimum = cached
    return float(value), np.asarray(gradient, dtype=np.float64), float(minimum)


def _harness_value(counter: HarnessCounter, coords: np.ndarray) -> tuple[float, np.ndarray]:
    """One harness evaluation that does not reuse a search-cache entry."""
    flat = np.asarray(coords, dtype=np.float64).reshape(-1)
    key = tuple(float(value) for value in flat)
    counter.remember((), (0.0, np.zeros(1, dtype=np.float64), 0.0))
    _fresh, cached = counter.charge(key)
    if cached is None:
        value, gradient = lennard_jones(flat)
        gradient = np.asarray(gradient, dtype=np.float64)
        counter.remember(key, (float(value), gradient, math.inf))
    else:
        value, gradient, _minimum = cached
    return float(value), np.asarray(gradient, dtype=np.float64)


def _quench(
    counter: HarnessCounter,
    origin: np.ndarray,
    *,
    cap: int,
    local_max: int,
    maxstep: float,
    tolg: float,
    core_radius: float,
) -> tuple[np.ndarray, float, int | None]:
    """Step-limited LD_LBFGS. The returned energy is the harness value."""
    import nlopt

    x = np.asarray(origin, dtype=np.float64).reshape(-1).copy()
    opened = counter.calls
    value, gradient, minimum = _search_value(counter, x, core_radius)
    best_e = value
    best_x = x.copy()
    best_r = minimum
    local_code: int | None = 1 if float(np.max(np.abs(gradient))) <= tolg else None
    stall = 0
    previous = value
    if local_code == 1:
        if best_r < core_radius:
            best_e, _gradient = _harness_value(counter, best_x)
        return best_x, float(best_e), local_code
    while counter.calls - opened < local_max and counter.calls < cap:
        box = x.copy()
        lower = box - maxstep
        upper = box + maxstep

        def objective(point, grad):
            reported, reported_grad, _minimum = _search_value(counter, point, core_radius)
            if grad.size:
                grad[:] = reported_grad
            return reported

        opt = nlopt.opt(nlopt.LD_LBFGS, x.size)
        opt.set_min_objective(objective)
        opt.set_lower_bounds(lower.tolist())
        opt.set_upper_bounds(upper.tolist())
        opt.set_xtol_rel(1e-14)
        opt.set_ftol_abs(1e-14)
        remain = min(local_max - (counter.calls - opened), cap - counter.calls)
        opt.set_maxeval(max(1, min(100, remain)))
        opt.set_param("tolg", tolg)
        before = counter.calls
        try:
            returned = np.asarray(opt.optimize(box.copy()), dtype=np.float64)
            local_code = int(opt.last_optimize_result())
        except ForceCallCap:
            raise
        except Exception as exc:
            if "ForceCallCap" in type(exc).__name__ or "force-call cap" in str(exc):
                raise ForceCallCap(str(exc)) from exc
            local_code = None
            break
        if returned.size != x.size or not bool(np.all(np.isfinite(returned))):
            break
        reported, reported_grad, minimum = _search_value(counter, returned, core_radius)
        if reported < best_e:
            best_e = reported
            best_x = returned.copy()
            best_r = minimum
        x = returned.copy()
        gmax = float(np.max(np.abs(reported_grad)))
        on_bound = bool(np.any((returned <= lower + 1e-8) | (returned >= upper - 1e-8)))
        if counter.calls == before:
            break
        if gmax <= tolg and not on_bound:
            break
        if abs(reported - previous) < 1e-10 and not on_bound:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
        previous = reported
    if not math.isfinite(best_e):
        return best_x, best_e, local_code
    if best_r < core_radius:
        best_e, _gradient = _harness_value(counter, best_x)
    return best_x, float(best_e), local_code


def _run_basin_hop(job: dict) -> dict:
    """Displace, quench with LD_LBFGS, and accept with the Metropolis rule."""
    calls = 0
    verify_calls = 0
    error = ""
    code: int | None = None
    energy = None
    force_norm = None
    nlopt_value = None
    seconds = 0.0
    cap = int(job["cap"])
    x0 = np.asarray(job["coords"], dtype=np.float64).reshape(-1).copy()
    point = x0.copy()
    try:
        counter = HarnessCounter(cache=True, cap=cap)
        limit = float(job["reference"]) + float(job["tolerance"])
        dr = float(job["displacement"])
        kT = float(job["temperature"])
        quench = {
            "cap": cap,
            "local_max": int(job["local_maxeval"]),
            "maxstep": float(job["maxstep"]),
            "tolg": float(job["tolg"]),
            "core_radius": float(job["core_radius"]),
        }
        rng = np.random.default_rng(int(job["seed"]))
        started = time.perf_counter()
        current, current_e, code = _quench(counter, x0, **quench)
        point = current.copy()
        best_e = current_e
        while counter.calls < cap and not (math.isfinite(best_e) and best_e <= limit):
            trial = current + rng.uniform(-dr, dr, size=current.size)
            trial = trial.reshape(-1, 3)
            trial -= trial.mean(axis=0)
            trial_x, trial_e, code = _quench(counter, trial.reshape(-1), **quench)
            if math.isfinite(trial_e) and trial_e < best_e:
                best_e = trial_e
                point = trial_x.copy()
            accept = math.isfinite(trial_e) and (
                trial_e <= current_e or rng.random() < math.exp(min(0.0, -(trial_e - current_e) / kT))
            )
            if accept:
                current = trial_x
                current_e = trial_e
        if math.isfinite(best_e) and best_e <= limit and counter.calls < cap:
            code = 2
        elif counter.calls >= cap:
            code = MAXEVAL
        seconds = time.perf_counter() - started
        calls = int(counter.calls)
        counter.remember((), (0.0, np.zeros(1, dtype=np.float64), 0.0))
        with counter.verification():
            key = tuple(float(value) for value in point)
            _fresh, cached = counter.charge(key)
            if cached is None:
                value, gradient = lennard_jones(point)
                gradient = np.asarray(gradient, dtype=np.float64)
                counter.remember(key, (float(value), gradient, math.inf))
            else:
                value, gradient, _minimum = cached
                gradient = np.asarray(gradient, dtype=np.float64)
        verify_calls = int(counter.verify_calls)
        energy = float(value)
        force_norm = float(np.linalg.norm(gradient))
        nlopt_value = energy
    except ForceCallCap as exc:
        error = f"ForceCallCap: {exc}"
        code = MAXEVAL
        if calls == 0 and "counter" in locals():
            calls = int(counter.calls)
        if seconds == 0.0 and "started" in locals():
            seconds = time.perf_counter() - started
    except Exception as exc:
        if not error:
            error = f"{type(exc).__name__}: {exc}"
        if code is None or (isinstance(code, int) and code >= 0 and "ForceCallCap" not in error):
            if "ForceCallCap" in error or "force-call cap" in error:
                code = MAXEVAL
            elif code is None:
                code = -1
        if calls == 0 and "counter" in locals():
            calls = int(counter.calls)
        if seconds == 0.0 and "started" in locals():
            seconds = time.perf_counter() - started
    if energy is None and point is not None:
        try:
            value, gradient = lennard_jones(point)
            if math.isfinite(float(value)):
                energy = float(value)
                force_norm = float(np.linalg.norm(gradient))
        except Exception:
            energy = None
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
        finite=finite,
        nlopt_code=code,
        error=error,
        reference=float(job["reference"]),
        tolerance=float(job["tolerance"]),
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
        "nlopt_value": None if nlopt_value is None or not math.isfinite(nlopt_value) else float(nlopt_value),
    }
    if error:
        record["error"] = error
    return record


def _run_cube(job: dict) -> dict:
    """One cube search. The NLopt seed is the start index."""
    import nlopt

    calls = 0
    verify_calls = 0
    error = ""
    code: int | None = None
    energy = None
    force_norm = None
    point = None
    nlopt_value = None
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
        opt.set_stopval(float(job["reference"]) + float(job["tolerance"]))
        opt.set_maxeval(cap)
        started = time.perf_counter()
        returned = None
        try:
            returned = np.asarray(opt.optimize(x0.copy()), dtype=np.float64).reshape(-1).copy()
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
        try:
            nlopt_value = float(opt.last_optimum_value())
        except Exception:
            nlopt_value = None
        if returned is not None and returned.size == x0.size and bool(np.all(np.isfinite(returned))):
            point = returned
        else:
            point = np.asarray(incumbent["x"], dtype=np.float64).reshape(-1)
        calls = int(counter.calls)
        with counter.verification():
            key = tuple(float(value) for value in np.asarray(point, dtype=np.float64).reshape(-1))
            _fresh, cached = counter.charge(key)
            if cached is None:
                value, gradient = lennard_jones(point)
                gradient = np.asarray(gradient, dtype=np.float64)
                counter.remember(key, (float(value), gradient))
            else:
                value, gradient = cached
        verify_calls = int(counter.verify_calls)
        energy = float(value)
        force_norm = float(np.linalg.norm(gradient))
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
        finite=finite,
        nlopt_code=code,
        error=error,
        reference=float(job["reference"]),
        tolerance=float(job["tolerance"]),
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
        "nlopt_value": None if nlopt_value is None or not math.isfinite(nlopt_value) else float(nlopt_value),
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
        "# A success is under the cap and at or below the Cambridge minimum plus 0.001.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _geometry_sort_key(path: Path) -> tuple:
    name = path.name
    if name.startswith("coords."):
        suffix = name.split(".", 1)[1]
        if suffix.isdigit():
            return (0, int(suffix))
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, name)


def _read_geometry(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".con":
        pos, _box = read_con(text)
        return np.asarray(pos, dtype=np.float64).reshape(-1)
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{path} has no coordinates")
    return np.asarray(rows, dtype=np.float64).reshape(-1)


def _collected_geometries(dest: Path) -> list[Path]:
    cons = [path for path in dest.rglob("*.con") if path.is_file()]
    if cons:
        return sorted(cons, key=_geometry_sort_key)
    coords = []
    for path in dest.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("coords.") and name.split(".", 1)[1].isdigit():
            coords.append(path)
    return sorted(coords, key=_geometry_sort_key)


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
        cons = _collected_geometries(dest)
        if not cons:
            raise FileNotFoundError(f"{archive} has no geometries")
    else:
        cons = iter_geometries(geometries)
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected is not None and len(cons) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(cons)}")
    loaded = []
    for path in cons:
        name = path.name if path.name.startswith("coords.") else path.stem
        loaded.append((name, _read_geometry(path)))
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


def _job(index: int, name: str, coords: np.ndarray, settings: dict, reference: float, tolerance: float) -> dict:
    section = settings["nlopt"]
    job = {
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
        "reference": reference,
        "tolerance": tolerance,
    }
    for name in ("displacement", "temperature", "maxstep", "tolg", "core_radius"):
        if section.get(name) is not None:
            job[name] = float(section[name])
    if section.get("local_maxeval") is not None:
        job["local_maxeval"] = int(section["local_maxeval"])
    return job


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
    reference = reference_energy(problem)
    tolerance = energy_tolerance(problem)
    if abs(tolerance - settings["window"]) > 1e-15:
        raise ValueError(f"{template['id']} energy window is not the problem tolerance")
    energy_limit = reference + tolerance
    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit > 0:
        loaded = loaded[:limit]
    jobs = [
        _job(index, name, coords, settings, reference, tolerance)
        for index, (name, coords) in enumerate(loaded)
    ]
    partial = _partial_path(output)
    done = _read_partial(partial) if resume else {}
    if not resume and partial.exists():
        partial.unlink()
        done = {}
    pending = [job for job in jobs if job["index"] not in done]
    records = [done[job["index"]] for job in jobs if job["index"] in done]
    if pending and settings["nlopt"].get("displacement") is not None:
        from optbench.fastpot import lj_capped_gradient

        lj_capped_gradient(np.zeros(6, dtype=np.float64), float(settings["nlopt"]["core_radius"]))
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
                            "nlopt_value": None,
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
        "reference_energy": reference,
        "seed": "start index",
    }
    for name in ("displacement", "temperature", "maxstep", "tolg", "local_maxeval", "core_radius"):
        if section.get(name) is not None:
            passed[name] = int(section[name]) if name == "local_maxeval" else float(section[name])
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
            "A success is at or below the Cambridge Cluster Database minimum plus 0.001 and under the evaluation budget."
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
