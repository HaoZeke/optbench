"""JAX MD minimization templates for this cell.

FIRE and gradient descent are ``jax_md.minimize``. The L-BFGS rows are
jaxopt 0.8.5 and optax 0.2.8. Every row uses the harness potential, the
problem stop, and a 0.2 per-atom move cap.
"""

from __future__ import annotations

from optbench.noise_catalog import SIGMAS, sigma_slug

JAXMD_VERSION = "0.2.29"
JAXMD_COMMIT = "84ab10407420fd91ad1c25be4ecd16ce1f54a75c"
JAXOPT_VERSION = "0.8.5"
JAXOPT_COMMIT = "b12c2f3ddaa3ef71fac42d7e419adccb9dcc49a1"
OPTAX_VERSION = "0.2.8"
OPTAX_COMMIT = "107e9ad20ca62bcbee1a8c016daab71c690a678c"
MAXSTEP = 0.2
STEP_SIZE = 0.001
HISTORY = 10
NOISE_SEED = 20261016

_FIRE = (
    "JAX MD 0.2.29 fire_descent at the library defaults: dt_start 0.1, dt_max 0.4, "
    "n_min 5, f_inc 1.1, f_dec 0.5, alpha_start 0.1, f_alpha 0.99. "
    "The shift scales a trial so the largest per-atom move is 0.2, the ASE FIRE constructor maxstep. "
    "The potential is the harness potential for this problem. "
    "The stop is the problem force norm and the budget is the force-call cap."
)
_GD = (
    "JAX MD 0.2.29 gradient_descent. step_size is 0.001, the value in the JAX MD minimization notebooks. "
    "The library requires a step size. "
    "The shift scales a trial so the largest per-atom move is 0.2. "
    "The potential is the harness potential for this problem. "
    "The stop is the problem force norm and the budget is the force-call cap."
)
_JAXOPT = (
    "jaxopt 0.8.5 LBFGS, history 10, zoom line search, max_stepsize 1, with the line search compiled. "
    "A trial whose largest per-atom move exceeds 0.2 is turned away before the harness potential runs, and the line search shortens it. "
    "The gradient tolerance checked after each accepted step is the problem force norm. "
    "The potential is the harness potential for this problem. "
    f"jaxopt commit {JAXOPT_COMMIT}. optax {OPTAX_VERSION} commit {OPTAX_COMMIT} is the pinned neighbor."
)
_OPTAX = (
    "optax 0.2.8 lbfgs, memory 10, zoom line search. "
    "A trial whose largest per-atom move exceeds 0.2 is turned away before the harness potential runs, and the line search shortens it. "
    "The gradient tolerance checked after each accepted step is the problem force norm. "
    "The potential is the harness potential for this problem. "
    f"optax commit {OPTAX_COMMIT}. jaxopt {JAXOPT_VERSION} commit {JAXOPT_COMMIT} is the pinned neighbor."
)
_NOISE = (
    " Gaussian force noise uses the problem seed, kappa, and clean-energy window. "
    "The Hessian is the clean force."
)

_METHODS = (
    ("fire", "FIRE", "fire", _FIRE, {}),
    ("gd", "gradient descent", "gd", _GD, {"step_size": STEP_SIZE}),
    ("lbfgs", "LBFGS", "lbfgs", _JAXOPT, {"history_size": HISTORY}),
    ("optax-lbfgs", "Optax LBFGS", "optax-lbfgs", _OPTAX, {"memory_size": HISTORY}),
)


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0.0:
            return "0.0"
        return repr(value)
    return str(value)


def _solver(method: str, extra: dict) -> dict:
    solver = {"method": method, "maxstep": MAXSTEP}
    solver.update(extra)
    return solver


def specs() -> list[dict]:
    """One dict per template, clean problems first, then the noise grid."""
    found: list[dict] = []
    for problem, prefix in (
        ("minimization/lj38", "lj38"),
        ("minimization/morse-bulk", "morse"),
    ):
        for slug, algorithm, method, note, extra in _METHODS:
            found.append(
                {
                    "id": f"jaxmd/{prefix}-{slug}",
                    "engine": "jaxmd",
                    "problem": problem,
                    "code": "jaxmd",
                    "code_version": JAXMD_VERSION,
                    "commit": JAXMD_COMMIT,
                    "algorithm": algorithm,
                    "tuning": "default",
                    "note": note,
                    "jaxmd": _solver(method, extra),
                }
            )
    for slug, algorithm, method, note, extra in _METHODS:
        for sigma in SIGMAS:
            found.append(
                {
                    "id": f"jaxmd/noise-{slug}-{sigma_slug(sigma)}",
                    "engine": "jaxmd",
                    "problem": "minimization/lj38-noise",
                    "code": "jaxmd",
                    "code_version": JAXMD_VERSION,
                    "commit": JAXMD_COMMIT,
                    "algorithm": algorithm,
                    "tuning": "default",
                    "note": note + _NOISE,
                    "jaxmd": _solver(method, extra),
                    "noise": {"sigma": float(sigma), "seed": NOISE_SEED},
                }
            )
    return found


def render(spec: dict) -> str:
    lines = []
    for key in ("id", "engine", "problem", "code", "code_version", "commit", "algorithm", "tuning", "note"):
        lines.append(f"{key} = {_toml_value(spec[key])}")
    lines.append("")
    lines.append("[jaxmd]")
    for key, value in spec["jaxmd"].items():
        lines.append(f"{key} = {_toml_value(value)}")
    if "noise" in spec:
        lines.append("")
        lines.append("[noise]")
        lines.append(f"sigma = {_toml_value(spec['noise']['sigma'])}")
        lines.append(f"seed = {int(spec['noise']['seed'])}")
    lines.append("")
    return "\n".join(lines)
