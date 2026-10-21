"""Minimize OptBench starts with JAX MD, jaxopt, and optax.

FIRE and gradient descent are ``jax_md.minimize``. L-BFGS is
``jaxopt.LBFGS`` or ``optax.lbfgs``. The energy and gradient are the
harness potential. A JAX step reaches that potential through a host
callback on ``RgpotCounter.calculate``, so a repeated coordinate is one
call. Analytic Hessians used after the search are not force calls.

The library step ``dt * F`` on these Lennard-Jones starts leaves the
cutoff, and a dissociated cluster has a zero force. The shift, and the
L-BFGS trial check, keep the largest per-atom move at or below the
template maxstep. A rejected L-BFGS trial does not call the potential.

A success is under the cap, at the problem stop, and Hessian index 0.
On LJ38 the index removes the six rigid modes. Energy and Kabsch RMSD
are not applied: the problem files have no ``energy_tol`` on the clean
rows, no ``rmsd_tol``, and no per-start reference. The noise rows use
the noise-aware stop and the clean-energy window from the problem file.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.jaxmd_catalog import (
    JAXMD_COMMIT,
    JAXMD_VERSION,
    JAXOPT_COMMIT,
    JAXOPT_VERSION,
    OPTAX_COMMIT,
    OPTAX_VERSION,
)
from optbench.noise import KAPPA, converged_under_noise, stop_threshold
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

EIGENVALUE_TOL = 1e-6
_LBFGS = frozenset({"lbfgs", "optax-lbfgs"})
_SLOT: dict = {
    "evaluate": None,
    "guard": False,
    "maxstep": 0.2,
    "anchor": None,
    "e0": 0.0,
}
_BUILT: dict = {}


def _ensure_jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return jax, jnp


def _dist_commit(name: str) -> str:
    try:
        text = importlib.metadata.distribution(name).read_text("direct_url.json")
    except (importlib.metadata.PackageNotFoundError, FileNotFoundError, OSError):
        return ""
    if not text:
        return ""
    payload = json.loads(text)
    info = payload.get("vcs_info") or {}
    return str(info.get("commit_id") or "")


def _require_pins() -> tuple[str, str, str]:
    jaxmd = importlib.metadata.version("jax-md")
    jaxopt_version = importlib.metadata.version("jaxopt")
    optax_version = importlib.metadata.version("optax")
    if jaxmd != JAXMD_VERSION:
        raise RuntimeError(f"jax-md {jaxmd} is not {JAXMD_VERSION}")
    if jaxopt_version != JAXOPT_VERSION:
        raise RuntimeError(f"jaxopt {jaxopt_version} is not {JAXOPT_VERSION}")
    if optax_version != OPTAX_VERSION:
        raise RuntimeError(f"optax {optax_version} is not {OPTAX_VERSION}")
    for name, pinned in (
        ("jax-md", JAXMD_COMMIT),
        ("jaxopt", JAXOPT_COMMIT),
        ("optax", OPTAX_COMMIT),
    ):
        recorded = _dist_commit(name)
        if recorded and recorded != pinned:
            raise RuntimeError(f"{name} commit {recorded} is not {pinned}")
    return jaxmd, jaxopt_version, optax_version


def _host_energy_grad(flat):
    import numpy as np

    arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    if _SLOT["guard"] and _SLOT["anchor"] is not None:
        base = _SLOT["anchor"]
        move = float(np.linalg.norm((arr - base).reshape(-1, 3), axis=1).max())
        limit = float(_SLOT["maxstep"])
        if move > limit * (1.0 + 1e-12):
            e0 = float(_SLOT["e0"])
            return np.float64(e0 + abs(e0) + 1.0 + move), arr - base
    energy, grad = _SLOT["evaluate"](arr)
    return np.float64(energy), np.asarray(grad, dtype=np.float64).reshape(-1)


def _cb_force(flat):
    energy, grad = _host_energy_grad(flat)
    return energy, -grad


def _shift(maxstep: float):
    jax, jnp = _ensure_jax()

    def shift(R, dR, **kwargs):
        norms = jnp.linalg.norm(dR, axis=-1)
        largest = jnp.max(norms)
        scale = jnp.where(largest > maxstep, maxstep / jnp.maximum(largest, 1e-30), 1.0)
        return R + dR * scale

    return shift


def _force_fn():
    jax, jnp = _ensure_jax()

    def force(R, **kwargs):
        flat = jnp.reshape(R, (-1,))
        _energy, pushed = jax.pure_callback(
            _cb_force,
            (
                jax.ShapeDtypeStruct((), jnp.float64),
                jax.ShapeDtypeStruct(flat.shape, jnp.float64),
            ),
            flat,
        )
        return jnp.reshape(pushed, R.shape)

    return force


def _fire(maxstep: float):
    key = ("fire", float(maxstep))
    if key not in _BUILT:
        from jax_md import minimize

        init_fn, apply_fn = minimize.fire_descent(_force_fn(), _shift(maxstep))
        _BUILT[key] = (init_fn, apply_fn)
    return _BUILT[key]


def _gd(step_size: float, maxstep: float):
    jax, _jnp = _ensure_jax()
    key = ("gd", float(step_size), float(maxstep))
    if key not in _BUILT:
        from jax_md import minimize

        _init_fn, apply_fn = minimize.gradient_descent(_force_fn(), _shift(maxstep), float(step_size))
        _BUILT[key] = jax.jit(apply_fn)
    return _BUILT[key]


_VALUE_FN = None


def _value_function():
    """Scalar energy whose Jacobian-vector product is the harness gradient.

    ``jax.custom_vjp`` returns a new callable. Decorating the original
    function and then ignoring that return value leaves ``value_and_grad``
    on the raw callback, which differentiates through ``pure_callback`` and
    does not see the gradient the harness already computed.
    """
    global _VALUE_FN
    if _VALUE_FN is not None:
        return _VALUE_FN
    jax, jnp = _ensure_jax()

    @jax.custom_vjp
    def value_fn(params):
        energy, _grad = jax.pure_callback(
            _host_energy_grad,
            (
                jax.ShapeDtypeStruct((), jnp.float64),
                jax.ShapeDtypeStruct(params.shape, jnp.float64),
            ),
            params,
        )
        return energy

    def value_fwd(params):
        energy, grad = jax.pure_callback(
            _host_energy_grad,
            (
                jax.ShapeDtypeStruct((), jnp.float64),
                jax.ShapeDtypeStruct(params.shape, jnp.float64),
            ),
            params,
        )
        return energy, grad

    def value_bwd(grad, cotangent):
        return (cotangent * grad,)

    value_fn.defvjp(value_fwd, value_bwd)
    _VALUE_FN = value_fn
    return value_fn


def _fun_value_grad(params):
    jax, jnp = _ensure_jax()
    flat = jnp.reshape(params, (-1,))
    energy, grad = jax.pure_callback(
        _host_energy_grad,
        (
            jax.ShapeDtypeStruct((), jnp.float64),
            jax.ShapeDtypeStruct(flat.shape, jnp.float64),
        ),
        flat,
    )
    return energy, grad


def _jaxopt(history: int):
    key = ("jaxopt", int(history))
    if key not in _BUILT:
        import jaxopt

        _BUILT[key] = jaxopt.LBFGS(
            fun=_fun_value_grad,
            value_and_grad=True,
            tol=0.0,
            maxiter=10**9,
            jit=True,
            implicit_diff=False,
            history_size=int(history),
            verbose=False,
        )
    return _BUILT[key]


def _optax(memory: int):
    key = ("optax", int(memory))
    if key not in _BUILT:
        import optax

        jax, _jnp = _ensure_jax()
        value_fn = _value_function()
        solver = optax.lbfgs(memory_size=int(memory))

        def one(params, state, value, grad):
            updates, state = solver.update(
                grad,
                state,
                params,
                value=value,
                grad=grad,
                value_fn=value_fn,
            )
            return optax.apply_updates(params, updates), state

        _BUILT[key] = (solver, jax.jit(one), optax.value_and_grad_from_state(value_fn))
    return _BUILT[key]


def _flat(point) -> list[float]:
    import numpy as np

    return [float(value) for value in np.asarray(point, dtype=np.float64).reshape(-1)]


def _norm_of(flat) -> tuple[float, float]:
    import numpy as np

    energy, grad = _SLOT["evaluate"](np.asarray(flat, dtype=np.float64).reshape(-1))
    return float(np.linalg.norm(grad)), float(energy)


def _met(flat, job: dict, norm: float, observed) -> bool:
    if observed is None:
        return norm <= float(job["gtol"]) and math.isfinite(norm)
    return converged_under_noise(
        force_norm=norm,
        energies=observed.energies,
        clean_tol=float(job["clean_tol"]),
        sigma=float(job["sigma"]),
        components=len(flat),
        energy_tol=float(job["energy_tol"]),
        kappa=KAPPA,
    )


def _set_anchor(flat, energy: float) -> None:
    import numpy as np

    _SLOT["anchor"] = np.asarray(flat, dtype=np.float64).reshape(-1).copy()
    _SLOT["e0"] = float(energy)


def _drive_fire(job, x0, cap: int, observed) -> tuple[list[float], bool]:
    import numpy as np

    init_fn, apply_fn = _fire(float(job["maxstep"]))
    position = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    state = init_fn(position)
    point = _flat(state.position)
    norm, _energy = _norm_of(point)
    stopped = _met(point, job, norm, observed)
    stalls = 0
    while not stopped:
        if _calls(job) >= cap:
            break
        before = _calls(job)
        state = apply_fn(state)
        point = _flat(state.position)
        if not all(math.isfinite(value) for value in point):
            break
        norm, _energy = _norm_of(point)
        if not math.isfinite(norm):
            break
        stopped = _met(point, job, norm, observed)
        if _calls(job) == before:
            stalls += 1
            if stalls >= 2:
                break
        else:
            stalls = 0
    return point, stopped


def _drive_gd(job, x0, cap: int, observed) -> tuple[list[float], bool]:
    import numpy as np

    apply_fn = _gd(float(job["step_size"]), float(job["maxstep"]))
    position = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    point = _flat(position)
    norm, _energy = _norm_of(point)
    stopped = _met(point, job, norm, observed)
    stalls = 0
    while not stopped:
        if _calls(job) >= cap:
            break
        before = _calls(job)
        position = apply_fn(position)
        point = _flat(position)
        if not all(math.isfinite(value) for value in point):
            break
        norm, _energy = _norm_of(point)
        if not math.isfinite(norm):
            break
        stopped = _met(point, job, norm, observed)
        if _calls(job) == before:
            stalls += 1
            if stalls >= 2:
                break
        else:
            stalls = 0
    return point, stopped


def _calls(job) -> int:
    return int(job["counter"].calls)


def _as_bool(value) -> bool:
    import numpy as np

    return bool(np.asarray(value).item())


def _is_cap(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ForceCallCap):
            return True
        current = current.__cause__ or current.__context__
    return "ForceCallCap" in f"{type(exc).__name__}: {exc}"


def _drive_jaxopt(job, x0, cap: int, observed) -> tuple[list[float], bool]:
    import numpy as np

    _, jnp = _ensure_jax()
    solver = _jaxopt(int(job.get("history_size") or 10))
    params = jnp.asarray(np.asarray(x0, dtype=np.float64).reshape(-1))
    _set_anchor(_flat(params), 0.0)
    state = solver.init_state(params)
    _set_anchor(_flat(params), float(state.value))
    point = _flat(params)
    norm, _energy = _norm_of(point)
    stopped = _met(point, job, norm, observed)
    stalls = 0
    while not stopped:
        if _calls(job) >= cap:
            break
        before = _calls(job)
        previous = point
        _set_anchor(point, float(state.value))
        params, state = solver.update(params, state)
        point = _flat(params)
        if not all(math.isfinite(value) for value in point):
            point = previous
            break
        norm, _energy = _norm_of(point)
        if not math.isfinite(norm):
            point = previous
            break
        stopped = _met(point, job, norm, observed)
        failed = _as_bool(state.failed_linesearch)
        if _calls(job) == before or (failed and point == previous):
            stalls += 1
            if stalls >= 2:
                break
        else:
            stalls = 0
    return point, stopped


def _drive_optax(job, x0, cap: int, observed) -> tuple[list[float], bool]:
    import numpy as np

    _, jnp = _ensure_jax()
    solver, step, value_and_grad = _optax(int(job.get("memory_size") or 10))
    params = jnp.asarray(np.asarray(x0, dtype=np.float64).reshape(-1))
    state = solver.init(params)
    _set_anchor(_flat(params), 0.0)
    value, grad = value_and_grad(params, state=state)
    point = _flat(params)
    norm = float(jnp.linalg.norm(grad))
    stopped = _met(point, job, norm, observed)
    stalls = 0
    while not stopped:
        if _calls(job) >= cap:
            break
        before = _calls(job)
        previous = point
        _set_anchor(point, float(value))
        params, state = step(params, state, value, grad)
        value, grad = value_and_grad(params, state=state)
        point = _flat(params)
        norm = float(jnp.linalg.norm(grad))
        if not all(math.isfinite(value) for value in point) or not math.isfinite(norm):
            point = previous
            break
        stopped = _met(point, job, norm, observed)
        if _calls(job) == before:
            stalls += 1
            if stalls >= 2:
                break
        else:
            stalls = 0
    return point, stopped


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
    import numpy as np

    from optbench.runners.noise_min import _Observed

    _ensure_jax()
    started = time.perf_counter()
    calls = 0
    verify_calls = 0
    error = ""
    energy = None
    force_norm = None
    point = None
    potential = None
    observed = None
    stopped = False
    sigma = float(job.get("sigma") or 0.0)
    cap = int(job["cap"])
    try:
        x0 = [float(value) for value in job["coords"]]
        box = None if job.get("box") is None else np.asarray(job["box"], dtype=np.float64)
        kind = job["kind"]

        def raw(positions, _numbers=None, _box=None):
            if kind == "lj":
                return lennard_jones(positions)
            return morse_bulk(positions, box)

        counter = HarnessCounter(cache=True, cap=cap)
        job["counter"] = counter
        potential = RgpotCounter(raw, counter)
        if job.get("noisy"):
            observed = _Observed(
                potential,
                sigma,
                random.Random(int(job["seed"]) + int(job["index"])),
            )

            def evaluate(flat):
                value, grad = observed(np.asarray(flat, dtype=np.float64).reshape(-1))
                return float(value), np.asarray(grad, dtype=np.float64)
        else:

            def evaluate(flat):
                value, grad = potential.calculate(np.asarray(flat, dtype=np.float64).reshape(-1))
                return float(value), np.asarray(grad, dtype=np.float64)

        method = str(job["method"])
        _SLOT["evaluate"] = evaluate
        _SLOT["guard"] = method in _LBFGS
        _SLOT["maxstep"] = float(job["maxstep"])
        _SLOT["anchor"] = None
        _SLOT["e0"] = 0.0
        if method == "fire":
            point, stopped = _drive_fire(job, x0, cap, observed)
        elif method == "gd":
            point, stopped = _drive_gd(job, x0, cap, observed)
        elif method == "lbfgs":
            point, stopped = _drive_jaxopt(job, x0, cap, observed)
        elif method == "optax-lbfgs":
            point, stopped = _drive_optax(job, x0, cap, observed)
        else:
            raise ValueError(f"unknown method {method}")
    except Exception as exc:
        if _is_cap(exc):
            error = f"ForceCallCap: {exc}"
            if point is None and _SLOT.get("anchor") is not None:
                point = _flat(_SLOT["anchor"])
        else:
            error = f"{type(exc).__name__}: {exc}"
            if point is None and "x0" in locals():
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
    components = len(job["coords"])
    threshold = gtol if sigma == 0.0 else stop_threshold(gtol, sigma, components, KAPPA)
    hessian_index = None
    min_eigenvalues: list[float] = []
    cap_hit = "force-call cap" in error or "ForceCallCap" in error
    if point is not None and (not error or cap_hit):
        try:
            hessian_index, min_eigenvalues = _index_at(job["kind"], point, job.get("box"), threshold)
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
        and force_norm <= threshold
        and calls < cap
        and not error
    )
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
        "sigma": sigma,
        "seconds": time.perf_counter() - started,
    }
    if job.get("noisy"):
        record["seed"] = int(job["seed"]) + int(job["index"])
    if error:
        record["error"] = error
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, sigma: float | None) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "jaxopt_version": JAXOPT_VERSION,
        "jaxopt_commit": JAXOPT_COMMIT,
        "optax_version": OPTAX_VERSION,
        "optax_commit": OPTAX_COMMIT,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "problem": template.get("problem"),
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
    if sigma is not None:
        rows["sigma"] = sigma
    lines = [
        "# counter harness. One force call is one RgpotCounter.calculate entry.",
        "# FIRE and gradient descent are jax_md.minimize. L-BFGS is jaxopt.LBFGS or optax.lbfgs.",
        "# The potential is the harness potential. A rejected over-long L-BFGS trial is not a call.",
        "# v2 success is under the cap, at the problem stop, and Hessian index 0.",
        "# Energy and Kabsch RMSD are omitted: the problem file has no per-start reference.",
    ]
    if str(template.get("problem")) in {"minimization/lj38", "minimization/lj38-noise"}:
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


def _options(template: dict) -> dict:
    section = dict(template.get("jaxmd") or {})
    if "method" not in section:
        raise ValueError(f"{template['id']} needs jaxmd.method")
    return section


def _kind(problem_id: str) -> str:
    if problem_id in {"minimization/lj38", "minimization/lj38-noise"}:
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no jaxmd potential")


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
    version, _jaxopt_version, _optax_version = _require_pins()
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"jax-md {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if commit != JAXMD_COMMIT:
        raise RuntimeError(f"{template['id']} commit is not the jax-md pin")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    options = _options(template)
    method = str(options["method"])
    if method not in {"fire", "gd", "lbfgs", "optax-lbfgs"}:
        raise ValueError(f"{template['id']} method {method} is unknown")
    maxstep = options.get("maxstep")
    if maxstep is None or float(maxstep) <= 0.0:
        raise ValueError(f"{template['id']} needs jaxmd.maxstep, the per-atom move cap")
    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    noisy = problem["id"] == "minimization/lj38-noise"
    sigma = 0.0
    seed = 0
    energy_tol = gtol
    if noisy:
        noise = template.get("noise") or {}
        if "sigma" not in noise:
            raise ValueError(f"{template['id']} needs noise.sigma")
        sigma = float(noise["sigma"])
        seed = int(noise.get("seed", (problem.get("noise") or {})["seed"]))
        from optbench.noise import window_energy_tolerance

        energy_tol = window_energy_tolerance(
            float(convergence["energy_tol"]),
            sigma,
            float((problem.get("noise") or {}).get("kappa", KAPPA)),
        )
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    if problem["id"] == "minimization/morse-bulk" and geometries is None and not limit and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    done: dict[str, dict] = {}
    if resume:
        done = _read_starts(output.with_suffix(".starts.jsonl"))
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        if name in done:
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if _kind(problem["id"]) == "lj" or box is None else box.tolist(),
                "kind": _kind(problem["id"]),
                "method": method,
                "maxstep": float(maxstep),
                "step_size": float(options.get("step_size") or 0.0),
                "history_size": int(options.get("history_size") or 10),
                "memory_size": int(options.get("memory_size") or 10),
                "gtol": gtol,
                "clean_tol": gtol,
                "energy_tol": energy_tol,
                "cap": cap,
                "sigma": sigma,
                "seed": seed,
                "noisy": noisy,
            }
        )
    records = list(done.values())
    if jobs:
        if workers <= 1:
            fresh = []
            for done_n, job in enumerate(jobs, 1):
                fresh.append(_minimize_job(job))
                if done_n == len(jobs) or done_n % 25 == 0:
                    print(f"{template['id']} {done_n}/{len(jobs)}", flush=True)
                    _emit(output, template, problem, records + fresh, cap, commit, version, sigma if noisy else None)
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            fresh = []
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_minimize_job, job) for job in jobs]
                total = len(futures)
                for done_n, future in enumerate(as_completed(futures), 1):
                    fresh.append(future.result())
                    if done_n == total or done_n % 25 == 0:
                        print(f"{template['id']} {done_n}/{total}", flush=True)
                        _emit(
                            output,
                            template,
                            problem,
                            records + fresh,
                            cap,
                            commit,
                            version,
                            sigma if noisy else None,
                        )
        records.extend(fresh)
    records.sort(key=lambda row: int(row.get("index", 0)))
    return _emit(output, template, problem, records, cap, commit, version, sigma if noisy else None)


def _read_starts(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("start"):
            found[str(row["start"])] = row
    return found


def _emit(output, template, problem, records, cap, commit, version, sigma) -> dict:
    ordered = sorted(records, key=lambda row: int(row.get("index", 0)))
    public = [dict(record) for record in ordered]
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in public]
    summary = summarize_capped(legacy_counts, cap) if public else {
        "force_calls": None,
        "force_calls_min": None,
        "force_calls_max": None,
        "nfailed": 0,
    }
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
    ) if public else {}
    section = _options(template)
    passed = {
        "method": section["method"],
        "maxstep": float(section["maxstep"]),
        "jaxmd_commit": JAXMD_COMMIT,
        "jaxopt_commit": JAXOPT_COMMIT,
        "optax_commit": OPTAX_COMMIT,
    }
    if "step_size" in section:
        passed["step_size"] = float(section["step_size"])
    if "history_size" in section:
        passed["history_size"] = int(section["history_size"])
    if "memory_size" in section:
        passed["memory_size"] = int(section["memory_size"])
    if sigma is not None:
        passed["sigma"] = float(sigma)
    provenance = collect_provenance(
        code="jaxmd",
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
        "note": "Harness counter. The potential is the harness potential. Energy and Kabsch RMSD were not applied.",
        "runs": [
            {
                "entry": output.stem,
                "code": "jaxmd",
                "algorithm": template.get("algorithm", section["method"]),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".dat.tmp")
    temporary.write_text(_dat(summary, template, metrics, commit, version, sigma), encoding="utf-8")
    temporary.replace(output)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(public)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics.get("par2_mean")
    summary["v2_nfailed"] = metrics.get("nfailed")
    return summary
