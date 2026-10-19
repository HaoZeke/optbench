"""Minimize OptBench starts with ``scipy.optimize.minimize``.

``fun`` and ``jac`` are the harness kernel. A second read of the same
coordinates is a cache hit on ``HarnessCounter`` and does not add a
force call. ``trust-krylov``, ``trust-ncg``, and ``Newton-CG`` take
Hessian-vector products from the analytic Hessian of that potential.
Each product counts as one force call. A penalty curvature, which does
not evaluate the potential, does not.

Lennard-Jones 38 uses a per-atom displacement cap of the same length as
the eOn max move. L-BFGS-B and the trust-region methods refuse a trial
past that cap. BFGS, CG, and Newton-CG map the trial back inside the
ball before the potential runs. The map is the identity through nine
tenths of the cap and smooth after that, so a line search does not see
a jump.
The problem Euclidean
force norm is the stop. Methods whose own test is an infinity norm use
that norm divided by ``sqrt(n)``, so a library stop is inside the
Euclidean stop. A callback stops when the Euclidean norm is met, which
keeps the search from solving a tighter problem than the file states.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.curvature import product_cost
from optbench.failures import classify_failure
from optbench.potentials import (
    lennard_jones,
    lennard_jones_hessian,
    morse_bulk,
    morse_bulk_hessian,
)
from optbench.potserver import RgpotCounter
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance, write_starts

COMMIT = "e4e854eaa8f18d807cd3496028e257e36caa93cc"
VERSION = "1.18.1"
REF = "v1.18.1"

METHODS = (
    "L-BFGS-B",
    "BFGS",
    "CG",
    "trust-krylov",
    "trust-ncg",
    "Newton-CG",
)
HESSIAN_METHODS = frozenset({"trust-krylov", "trust-ncg", "Newton-CG"})
_INF_NORM = frozenset({"L-BFGS-B", "BFGS", "CG"})
_TRUST = frozenset({"trust-krylov", "trust-ncg"})
# Wolfe line searches stop on a jump at the cap. Those methods see a
# smooth squash. L-BFGS-B and the trust methods keep a penalty that is
# strictly worse than the anchor, which their own step tests reject.
_SQUASH_METHODS = frozenset({"BFGS", "CG", "Newton-CG"})
# Identity through this fraction of the per-atom cap. The rest of the
# radius is a smooth squash onto the open ball.
SQUASH_FRACTION = 0.9
PENALTY_STIFFNESS = 1.0e6
_RESTARTS = 16
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _kind(problem_id: str) -> str:
    if problem_id == "minimization/lj38":
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no scipy potential")


def solver_options(
    method: str,
    dimension: int,
    force_tol: float,
    cap: int,
    maxstep: float | None,
    *,
    noise: bool,
) -> dict:
    """Options passed to ``minimize``. Values are JSON-safe.

    ``noise`` disables the library gradient stop. The noise-aware rule
    is applied by the callback, because a single force sample is not
    that rule.
    """
    if method not in METHODS:
        raise ValueError(f"unknown scipy method {method}")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if cap <= 0:
        raise ValueError("cap must be positive")
    # SciPy counts fun and jac separately, and one iteration can spend
    # several of each. The harness cap is the budget. This ceiling only
    # keeps the library from stopping first.
    ceiling = int(cap) * 20
    if method == "L-BFGS-B":
        gtol = 0.0 if noise else float(force_tol) / math.sqrt(dimension)
        # The default ftol, about 2e-9, is a relative energy test. On Morse
        # bulk it returns while the Euclidean force is still above the stop.
        # Machine epsilon keeps that test from winning over the force norm.
        return {
            "maxiter": ceiling,
            "maxfun": ceiling,
            "gtol": gtol,
            "ftol": sys.float_info.epsilon,
        }
    if method in _INF_NORM:
        gtol = 0.0 if noise else float(force_tol) / math.sqrt(dimension)
        return {"maxiter": ceiling, "gtol": gtol, "norm": "inf"}
    if method in _TRUST:
        options = {"maxiter": ceiling, "gtol": 0.0 if noise else float(force_tol)}
        if maxstep is not None:
            radius = float(maxstep) * math.sqrt(dimension // 3)
            options["max_trust_radius"] = radius
            initial = min(1.0, 0.5 * radius)
            if initial >= radius:
                initial = 0.5 * radius
            options["initial_trust_radius"] = initial
        return options
    return {"maxiter": ceiling, "xtol": 0.0}


def _call_options(options: dict) -> dict:
    import numpy as np

    called = dict(options)
    if called.get("norm") == "inf":
        called["norm"] = np.inf
    return called


def _flat(x) -> "object":
    import numpy as np

    return np.asarray(x, dtype=np.float64).reshape(-1)


def _squash_lengths(length: float, maxstep: float) -> tuple[float, float, float]:
    """Mapped length, first derivative, second derivative.

    The map is the identity on ``[0, fraction * maxstep]`` and
    ``tanh`` after that, so the image stays inside the cap.
    """
    limit = float(maxstep)
    start = SQUASH_FRACTION * limit
    if length <= start:
        return length, 1.0, 0.0
    width = limit - start
    z = (length - start) / width
    hyperbolic = math.tanh(z)
    sech2 = 1.0 - hyperbolic * hyperbolic
    mapped = start + width * hyperbolic
    first = sech2
    second = -2.0 * sech2 * hyperbolic / width
    return mapped, first, second


def _squash_image(x, anchor, maxstep: float):
    """Physical image of a trial and the per-atom map data."""
    import numpy as np

    delta = (_flat(x) - _flat(anchor)).reshape(-1, 3)
    norms = np.linalg.norm(delta, axis=1)
    disp = np.zeros_like(delta)
    meta = []
    for index, length in enumerate(norms):
        length = float(length)
        if length < 1e-15:
            meta.append(None)
            continue
        mapped, first, second = _squash_lengths(length, maxstep)
        disp[index] = delta[index] * (mapped / length)
        meta.append((length, mapped, first, second, delta[index] / length))
    image = _flat(anchor).reshape(-1, 3) + disp
    return image.reshape(-1), meta


def _squash_active(meta) -> bool:
    for item in meta:
        if item is None:
            continue
        length, mapped, _first, _second, _unit = item
        if mapped + 1e-8 < length:
            return True
    return False


def _squash_grad(grad_y, meta):
    import numpy as np

    gradient = np.asarray(grad_y, dtype=np.float64).reshape(-1, 3)
    out = np.zeros_like(gradient)
    for index, item in enumerate(meta):
        if item is None:
            out[index] = gradient[index]
            continue
        length, mapped, first, _second, unit = item
        scale = mapped / length
        extra = first - scale
        radial = float(np.dot(unit, gradient[index]))
        out[index] = scale * gradient[index] + extra * radial * unit
    return out.reshape(-1)


def _squash_hvp(hvp_y, vector, grad_y, meta):
    """Chain rule for a Hessian-vector product through the squash map."""
    import numpy as np

    step = _flat(vector).reshape(-1, 3)
    gradient = np.asarray(grad_y, dtype=np.float64).reshape(-1, 3)
    mapped_step = np.zeros_like(step)
    geometric = np.zeros_like(step)
    for index, item in enumerate(meta):
        if item is None:
            mapped_step[index] = step[index]
            continue
        length, mapped, first, second, unit = item
        scale = mapped / length
        extra = first - scale
        radial_step = float(np.dot(unit, step[index]))
        mapped_step[index] = scale * step[index] + extra * radial_step * unit
        dscale = extra / length
        d_extra = second - dscale
        unit_step = (step[index] - radial_step * unit) / length
        radial_gradient = float(np.dot(unit, gradient[index]))
        unit_gradient = float(np.dot(unit_step, gradient[index]))
        geometric[index] = (
            (dscale * radial_step) * gradient[index]
            + (d_extra * radial_step) * radial_gradient * unit
            + extra * (unit_gradient * unit + radial_gradient * unit_step)
        )
    pushed = np.asarray(hvp_y(mapped_step.reshape(-1)), dtype=np.float64).reshape(-1, 3)
    back = np.zeros_like(pushed)
    for index, item in enumerate(meta):
        if item is None:
            back[index] = pushed[index]
            continue
        length, mapped, first, _second, unit = item
        scale = mapped / length
        extra = first - scale
        radial = float(np.dot(unit, pushed[index]))
        back[index] = scale * pushed[index] + extra * radial * unit
    return (back + geometric).reshape(-1)


def _beyond(x, anchor, maxstep: float) -> bool:
    import numpy as np

    delta = (_flat(x) - _flat(anchor)).reshape(-1, 3)
    return float(np.max(np.linalg.norm(delta, axis=1))) > float(maxstep) + 1e-8


def _penalty(x, anchor, anchor_energy: float, maxstep: float):
    import numpy as np

    delta = (_flat(x) - _flat(anchor)).reshape(-1, 3)
    norms = np.linalg.norm(delta, axis=1)
    excess = np.maximum(norms - float(maxstep), 0.0)
    energy = float(anchor_energy) + 1.0 + 0.5 * PENALTY_STIFFNESS * float(np.dot(excess, excess))
    scale = np.zeros_like(norms)
    mask = excess > 0.0
    scale[mask] = PENALTY_STIFFNESS * excess[mask] / norms[mask]
    grad = (delta * scale[:, None]).reshape(-1)
    return energy, grad


def _penalty_hessp(x, anchor, maxstep: float, vector):
    import numpy as np

    delta = (_flat(x) - _flat(anchor)).reshape(-1, 3)
    step = _flat(vector).reshape(-1, 3)
    norms = np.linalg.norm(delta, axis=1)
    out = np.zeros_like(step)
    mask = norms > float(maxstep)
    out[mask] = PENALTY_STIFFNESS * step[mask]
    return out.reshape(-1)


def _cap_kind(method: str, maxstep: float | None) -> str:
    if maxstep is None:
        return "none"
    if method in _SQUASH_METHODS:
        return "squash"
    return "penalty"


def _template_method(template: dict) -> tuple[str, float | None]:
    section = template.get("scipy") or {}
    if "method" not in section:
        raise ValueError(f"{template.get('id')} needs scipy.method")
    method = str(section["method"])
    if method not in METHODS:
        raise ValueError(f"{template.get('id')} method {method} is not a scipy minimizer in this cell")
    unknown = sorted(key for key in section if key not in {"method", "maxstep"})
    if unknown:
        raise ValueError(f"{template.get('id')} has unknown scipy keys {unknown}")
    maxstep = section.get("maxstep")
    if maxstep is not None:
        maxstep = float(maxstep)
        if maxstep <= 0.0:
            raise ValueError(f"{template.get('id')} maxstep must be positive")
    if template.get("problem") == "minimization/lj38" and maxstep is None:
        raise ValueError(
            f"{template.get('id')} needs scipy.maxstep, the per-atom displacement cap. "
            "An uncapped trial on these starts leaves the Lennard-Jones cutoff."
        )
    return method, maxstep


def search(
    x0,
    method: str,
    evaluate,
    *,
    hessian=None,
    force_tol: float,
    cap: int,
    maxstep: float | None,
    stop=None,
    noise: bool = False,
    counter=None,
) -> dict:
    """Run one ``minimize`` and return the last feasible point.

    ``evaluate(coords) -> (energy, gradient)`` is the harness kernel,
    already wrapped with the counter. ``stop(energy, gradient)`` is the
    problem rule. The default stop is the Euclidean force norm.
    """
    import warnings

    import numpy as np
    from scipy.optimize import minimize

    _quiet_threads()
    if method in HESSIAN_METHODS and hessian is None:
        raise ValueError(f"{method} needs the analytic Hessian-vector product")
    x0 = _flat(x0).copy()
    if x0.size % 3 != 0:
        raise ValueError("coordinates must be a multiple of 3")
    options = solver_options(method, int(x0.size), float(force_tol), int(cap), maxstep, noise=noise)
    if stop is None:
        limit = float(force_tol)

        def stop(energy, grad, limit=limit):
            return float(np.linalg.norm(grad)) <= limit

    state = {
        "anchor": x0.copy(),
        "anchor_energy": 0.0,
        "stopped": False,
        "rejected": False,
        "cap_hit": False,
        "hessian_calls": 0,
        "hess": None,
        "hess_key": None,
        "refused": 0,
        "error": "",
        "scipy_success": None,
        "scipy_status": None,
        "scipy_message": "",
        "scipy_nit": None,
        "scipy_nfev": None,
    }

    def _sync_cap() -> None:
        if counter is not None:
            counter.cap = int(cap) - int(state["hessian_calls"])

    def _charge_product() -> None:
        gradients = 0 if counter is None else int(counter.calls)
        if gradients + int(state["hessian_calls"]) >= int(cap):
            state["cap_hit"] = True
            raise ForceCallCap(f"force-call cap {cap}")
        state["hessian_calls"] += product_cost()
        _sync_cap()

    def _cache_hessian(point):
        key = tuple(float(value) for value in _flat(point))
        if state["hess_key"] != key:
            state["hess"] = np.asarray(hessian(point), dtype=np.float64)
            state["hess_key"] = key

    _sync_cap()

    kind = _cap_kind(method, maxstep)

    def _count_refusal(point):
        key = tuple(float(value) for value in _flat(point))
        if state.get("refuse_key") != key:
            state["refused"] += 1
            state["refuse_key"] = key

    def _at(raw):
        point = _flat(raw).copy()
        if kind == "penalty" and _beyond(point, state["anchor"], float(maxstep)):
            _count_refusal(point)
            energy, grad = _penalty(point, state["anchor"], state["anchor_energy"], float(maxstep))
            return point, grad, float(energy), grad, "penalty"
        if kind == "squash":
            image, meta = _squash_image(point, state["anchor"], float(maxstep))
            if _squash_active(meta):
                _count_refusal(point)
        else:
            image, meta = point, None
        try:
            energy, grad_y = evaluate(image)
        except ForceCallCap:
            state["cap_hit"] = True
            raise
        grad_y = _flat(grad_y).copy()
        grad_x = grad_y if meta is None else _squash_grad(grad_y, meta)
        return image, grad_x, float(energy), grad_y, meta

    def fun(raw):
        _image, _grad_x, energy, _grad_y, _meta = _at(raw)
        return energy

    def jac(raw):
        _image, grad_x, _energy, _grad_y, _meta = _at(raw)
        return grad_x

    def hessp(raw, vector, *_args):
        image, _grad_x, _energy, grad_y, meta = _at(raw)
        if meta == "penalty":
            return _penalty_hessp(image, state["anchor"], float(maxstep), vector)
        _charge_product()
        _cache_hessian(image)

        def hvp_y(direction):
            return state["hess"] @ _flat(direction)

        if meta is None:
            return hvp_y(vector)
        return _squash_hvp(hvp_y, vector, grad_y, meta)

    def _accept(image, energy, grad_y, meta, raw):
        if meta == "penalty":
            state["rejected"] = True
            return False
        state["anchor"] = _flat(image).copy()
        state["anchor_energy"] = energy
        state["last_raw"] = _flat(raw).copy()
        if stop(energy, grad_y):
            state["stopped"] = True
        return True

    def callback(xk):
        image, _grad_x, energy, grad_y, meta = _at(xk)
        if not _accept(image, energy, grad_y, meta, xk):
            raise StopIteration
        if state["stopped"]:
            raise StopIteration

    try:
        image, _grad_x, energy, grad_y, _meta = _at(x0)
        state["anchor"] = _flat(image).copy()
        state["anchor_energy"] = energy
        if stop(energy, grad_y):
            state["stopped"] = True
        else:
            kwargs = {
                "fun": fun,
                "method": method,
                "jac": jac,
                "callback": callback,
                "options": _call_options(options),
            }
            if method in HESSIAN_METHODS:
                kwargs["hessp"] = hessp
            x = state["anchor"].copy()
            nit = 0
            nfev = 0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                for _attempt in range(_RESTARTS):
                    before = state["anchor"].copy()
                    state["last_raw"] = None
                    result = minimize(x0=x, **kwargs)
                    state["scipy_success"] = bool(result.success)
                    state["scipy_status"] = None if result.get("status") is None else int(result.status)
                    state["scipy_message"] = str(result.message or "")
                    nit += 0 if result.get("nit") is None else int(result.nit)
                    nfev += 0 if result.get("nfev") is None else int(result.nfev)
                    state["scipy_nit"] = nit
                    state["scipy_nfev"] = nfev
                    raw = _flat(result.x).copy()
                    if state.get("last_raw") is None or not np.allclose(state["last_raw"], raw):
                        image, _grad_x, energy, grad_y, meta = _at(raw)
                        _accept(image, energy, grad_y, meta, raw)
                    if state["stopped"] or state["cap_hit"] or state["rejected"]:
                        break
                    message = state["scipy_message"].lower()
                    moved = float(np.max(np.abs(state["anchor"] - before))) > 1e-10
                    if kind != "squash" or "precision" not in message or not moved:
                        break
                    x = state["anchor"].copy()
    except ForceCallCap as exc:
        state["cap_hit"] = True
        state["error"] = f"ForceCallCap: {exc}"
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "point": [float(value) for value in state["anchor"]],
        "stopped": bool(state["stopped"]),
        "hessian_calls": int(state["hessian_calls"]),
        "refused_trials": int(state["refused"]),
        "scipy_success": state["scipy_success"],
        "scipy_status": state["scipy_status"],
        "scipy_message": state["scipy_message"],
        "scipy_nit": state["scipy_nit"],
        "scipy_nfev": state["scipy_nfev"],
        "options": options,
        "error": state["error"],
        "cap_hit": bool(state["cap_hit"]),
        "rejected": bool(state["rejected"]),
    }


def _index_at(kind: str, point: list[float], box, force_stop: float) -> tuple[int, list[float]]:
    import numpy as np

    if kind == "lj":
        matrix = lennard_jones_hessian(point)
        values = vibrational_eigenvalues(matrix, point, rotations=True)
        tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    else:
        matrix = morse_bulk_hessian(point, box)
        values = np.linalg.eigvalsh(matrix).tolist()
        tolerance = 1e-6
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    import numpy as np

    calls = 0
    verify_calls = 0
    error = ""
    energy = None
    force_norm = None
    point = None
    potential = None
    info = {
        "stopped": False,
        "hessian_calls": 0,
        "refused_trials": 0,
        "scipy_success": None,
        "scipy_status": None,
        "scipy_message": "",
        "scipy_nit": None,
        "scipy_nfev": None,
        "options": {},
        "cap_hit": False,
        "rejected": False,
    }
    try:
        x0 = np.asarray(job["coords"], dtype=np.float64).reshape(-1).copy()
        box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
        kind = job["kind"]
        cap = int(job["cap"])
        method = str(job["method"])

        def evaluate_raw(positions, _numbers=None, _box=None):
            if kind == "lj":
                return lennard_jones(positions)
            return morse_bulk(positions, box)

        counter = HarnessCounter(cache=True, cap=cap)
        potential = RgpotCounter(evaluate_raw, counter)

        def evaluate(coords):
            value, gradient = potential.calculate(coords)
            return float(value), np.asarray(gradient, dtype=np.float64)

        def hessian(coords):
            if kind == "lj":
                return lennard_jones_hessian(coords)
            return morse_bulk_hessian(coords, box)

        info = search(
            x0,
            method,
            evaluate,
            hessian=hessian if method in HESSIAN_METHODS else None,
            force_tol=float(job["gtol"]),
            cap=cap,
            maxstep=job["maxstep"],
            noise=False,
            counter=counter,
        )
        point = info["point"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if info.get("error") and not error:
        error = str(info["error"])
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
    gradient_calls = calls
    hessian_calls = int(info.get("hessian_calls") or 0)
    if str(job["method"]) in HESSIAN_METHODS:
        calls = gradient_calls + hessian_calls
    gtol = float(job["gtol"])
    cap = int(job["cap"])
    hessian_index = None
    min_eigenvalues: list[float] = []
    cap_hit = bool(info.get("cap_hit")) or "force-call cap" in error
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
    stopped = bool(info.get("stopped")) or (
        finite and force_norm is not None and force_norm <= gtol and calls < cap and not error
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
        "gradient_calls": gradient_calls,
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
        "hessian_calls": hessian_calls,
        "hessian_force_calls": hessian_calls if str(job["method"]) in HESSIAN_METHODS else 0,
        "refused_trials": int(info.get("refused_trials") or 0),
        "min_eigenvalues": min_eigenvalues,
        "scipy_success": info.get("scipy_success"),
        "scipy_status": info.get("scipy_status"),
        "scipy_nit": info.get("scipy_nit"),
        "scipy_nfev": info.get("scipy_nfev"),
        "scipy_options": info.get("options") or {},
    }
    message = str(info.get("scipy_message") or "")
    if message:
        record["scipy_message"] = message
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
        "partial": True if partial else None,
    }
    lines = [
        "# counter harness. fun and jac are one potential evaluation when the coordinates match.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no energy_tol, rmsd_tol, or per-start reference.",
    ]
    if str((template.get("scipy") or {}).get("method") or "") in HESSIAN_METHODS:
        lines.append("# Each analytic Hessian-vector product counts as one force call.")
        lines.append(
            "# gradient_calls is the harness energy and gradient. "
            "hessian_calls counts products. force_calls is their sum."
        )
    if template.get("scipy", {}).get("maxstep") is not None:
        method_name = str((template.get("scipy") or {}).get("method") or "")
        if method_name in _SQUASH_METHODS:
            lines.append(
                "# A trial past the per-atom cap is mapped back inside that ball before the potential runs."
            )
        else:
            lines.append("# A trial past the per-atom cap is refused before the potential runs.")
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


def _load_previous(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[str(row["start"])] = row
    return found


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
    import scipy

    version = str(scipy.__version__)
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"scipy {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if commit != COMMIT:
        raise RuntimeError(f"{template['id']} commit is not the scipy pin {COMMIT}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    method, maxstep = _template_method(template)
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    if problem["id"] == "minimization/morse-bulk" and geometries is None and not limit and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    starts_path = output.with_suffix(".starts.jsonl")
    previous = _load_previous(starts_path) if resume else {}
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist() if hasattr(coords, "tolist") else list(coords),
                "box": None if _kind(problem["id"]) == "lj" or box is None else box.tolist(),
                "kind": _kind(problem["id"]),
                "method": method,
                "maxstep": maxstep,
                "gtol": gtol,
                "cap": cap,
            }
        )
    pending = [job for job in jobs if job["start"] not in previous]
    records_by_start = dict(previous)

    def checkpoint() -> None:
        ordered = [records_by_start[job["start"]] for job in jobs if job["start"] in records_by_start]
        _write_checkpoint(starts_path, ordered)

    if pending:
        total = len(pending)
        done = 0
        if workers <= 1:
            for job in pending:
                record = _minimize_job(job)
                records_by_start[str(record["start"])] = record
                done += 1
                if done == total or done % 25 == 0:
                    checkpoint()
                    print(f"{template['id']} {done}/{total}", flush=True)
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_minimize_job, job) for job in pending]
                for future in as_completed(futures):
                    record = future.result()
                    records_by_start[str(record["start"])] = record
                    done += 1
                    if done == total or done % 25 == 0:
                        checkpoint()
                        print(f"{template['id']} {done}/{total}", flush=True)
        checkpoint()
    records = []
    for job in jobs:
        row = records_by_start.get(job["start"])
        if row is None:
            continue
        row = dict(row)
        row["index"] = job["index"]
        records.append(row)
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    expected = (problem.get("sample") or {}).get("starts")
    if problem["id"] == "minimization/morse-bulk":
        expected = 100
    partial = expected is not None and len(records) < int(expected)
    return _finish(
        problem,
        template,
        records,
        output,
        version=version,
        commit=commit,
        gtol=gtol,
        cap=cap,
        method=method,
        maxstep=maxstep,
        partial=partial,
    )


def _write_checkpoint(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, sort_keys=True) for row in records)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(body + ("\n" if body else ""), encoding="utf-8")
    temporary.replace(path)


def _finish(
    problem,
    template,
    records,
    output,
    *,
    version,
    commit,
    gtol,
    cap,
    method,
    maxstep,
    partial,
) -> dict:
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
    passed = {"method": method, "gtol": gtol, "cap": cap}
    if maxstep is not None:
        passed["maxstep"] = maxstep
    if records and records[0].get("scipy_options"):
        passed["scipy_options"] = records[0]["scipy_options"]
    provenance = collect_provenance(
        code="SciPy",
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
            "Harness counter. "
            + (
                "Each analytic Hessian-vector product counts as one force call. "
                if method in HESSIAN_METHODS
                else "This method does not request a Hessian. "
            )
            + "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "SciPy",
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
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary
