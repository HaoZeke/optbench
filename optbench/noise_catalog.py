"""Templates for minimization/lj38-noise.

Newton and dogleg call the analytic Hessian of the clean force. Once
sigma is positive the stepper's force is not that gradient, so those
methods are not given a template. The other methods run at every
documented sigma.
"""

from __future__ import annotations

from pathlib import Path

SIGMAS = (0.0, 1e-4, 1e-3, 1e-2)
HESSIAN_METHODS = frozenset({"newton", "dogleg", "rfo"})
PROBLEM = "minimization/lj38-noise"
SEED = 20261016

RGMIN_COMMIT = "9a2e7ff11fa65826f9e847856bee10d3d70b7156"
RGMIN_VERSION = "0.4.1"
ASE_COMMIT = "f27c0005ae6a67ea419f996e728668865bfc1f86"
ASE_VERSION = "3.29.0"
SELLA_COMMIT = "6cfb60cdb100a146c3c6f0587d5308b247aa8010"
SELLA_VERSION = "2.6.0"
NLOPT_COMMIT = "673d2504f819bf554a7968e41de552ee102e7fb1"
NLOPT_VERSION = "2.11.0"

_SLUGS = ((0.0, "s0"), (1e-4, "s1em4"), (1e-3, "s1em3"), (1e-2, "s1em2"))


def sigma_slug(sigma: float) -> str:
    for value, slug in _SLUGS:
        if sigma == value:
            return slug
    raise ValueError(f"sigma {sigma} is not a documented level")


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


def _render(spec: dict) -> str:
    lines = []
    for key in ("id", "engine", "problem", "code", "code_version", "commit", "algorithm", "tuning", "note"):
        lines.append(f"{key} = {_toml_value(spec[key])}")
    lines.append("")
    lines.append("[solver]")
    solver = spec["solver"]
    for key, value in solver.items():
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    lines.append("[noise]")
    lines.append(f"sigma = {_toml_value(spec['noise']['sigma'])}")
    lines.append(f"seed = {int(spec['noise']['seed'])}")
    lines.append("")
    return "\n".join(lines)


def _base(ident: str, code: str, version: str, commit: str, algorithm: str, note: str, solver: dict, sigma: float) -> dict:
    return {
        "id": ident,
        "engine": "noise",
        "problem": PROBLEM,
        "code": code,
        "code_version": version,
        "commit": commit,
        "algorithm": algorithm,
        "tuning": "default",
        "note": note,
        "solver": solver,
        "noise": {"sigma": float(sigma), "seed": SEED},
    }


def specs() -> list[dict]:
    """One dict per campaign template, in stable order."""
    found: list[dict] = []
    force_note = (
        "Default profile. The stop is the noise-aware force norm, and at positive sigma "
        "the clean energy over the last window. The index is the clean Hessian with the rigid-mode rule."
    )
    hessian_note = (
        "Default profile at sigma 0. Newton and dogleg call the analytic Hessian of the clean force. "
        "Positive sigma has no template: the noisy force is not that gradient."
    )
    rgmin_force = (
        ("lbfgs-m100", "lbfgs", "LBFGS", {"memory": 100}),
        ("lbfgs", "lbfgs", "LBFGS", {}),
        ("fire", "fire", "FIRE", {}),
        ("cg", "cg", "CG", {}),
    )
    for slug, method, algorithm, extra in rgmin_force:
        for sigma in SIGMAS:
            solver = {"family": "rgmin", "method": method, "maxstep": 0.2, **extra}
            found.append(
                _base(
                    f"noise/rgmin-{slug}-{sigma_slug(sigma)}",
                    "rgmin",
                    RGMIN_VERSION,
                    RGMIN_COMMIT,
                    algorithm,
                    force_note,
                    solver,
                    sigma,
                )
            )
    for slug, method, algorithm in (("dogleg", "dogleg", "DOGLEG"), ("newton", "newton", "NEWTON")):
        solver = {"family": "rgmin", "method": method, "maxstep": 0.2}
        found.append(
            _base(
                f"noise/rgmin-{slug}-s0",
                "rgmin",
                RGMIN_VERSION,
                RGMIN_COMMIT,
                algorithm,
                hessian_note,
                solver,
                0.0,
            )
        )
    ase_rows = (
        ("bfgs", "BFGS", "BFGS", {"maxstep": 0.1}),
        ("lbfgs", "LBFGS", "LBFGS", {"maxstep": 0.1}),
        ("fire", "FIRE", "FIRE", {}),
    )
    ase_note = (
        "Default profile. BFGS and LBFGS use maxstep 0.1, the repaired LJ38 step. "
        "FIRE keeps the constructor defaults. The index is the clean Hessian with the rigid-mode rule."
    )
    for slug, name, algorithm, parameters in ase_rows:
        for sigma in SIGMAS:
            solver = {"family": "ase", "name": name, **parameters}
            found.append(
                _base(
                    f"noise/ase-{slug}-{sigma_slug(sigma)}",
                    "ASE",
                    ASE_VERSION,
                    ASE_COMMIT,
                    algorithm,
                    ase_note,
                    solver,
                    sigma,
                )
            )
    sella_note = (
        "Sella 2.6.0 minimum defaults, order 0. The internal model is built from forces. "
        "The reported index is still the clean analytic Hessian with the rigid-mode rule."
    )
    for sigma in SIGMAS:
        found.append(
            _base(
                f"noise/sella-min-{sigma_slug(sigma)}",
                "Sella",
                SELLA_VERSION,
                SELLA_COMMIT,
                "Sella",
                sella_note,
                {"family": "sella", "order": 0},
                sigma,
            )
        )
    nlopt_note = (
        "Default profile. The stop is the noise-aware force norm, and at positive sigma "
        "the clean energy over the last window. The index is the clean Hessian with the rigid-mode rule. "
        "LD_LBFGS, LD_VAR2, and LD_SLSQP scale the objective by 0.2 divided by the initial force norm. "
        "LD_TNEWTON_PRECOND_RESTART recenters a coordinate box of half-width 0.2. "
        "LD_MMA keeps the library defaults."
    )
    nlopt_rows = (
        ("lbfgs", "LD_LBFGS", "scale", 0.2),
        ("mma", "LD_MMA", "none", None),
        ("slsqp", "LD_SLSQP", "scale", 0.2),
        ("tnewton-precond-restart", "LD_TNEWTON_PRECOND_RESTART", "box", 0.2),
        ("var2", "LD_VAR2", "scale", 0.2),
    )
    for slug, method, step, maxstep in nlopt_rows:
        for sigma in SIGMAS:
            solver = {"family": "nlopt", "method": method, "step": step}
            if maxstep is not None:
                solver["maxstep"] = maxstep
            found.append(
                _base(
                    f"noise/nlopt-{slug}-{sigma_slug(sigma)}",
                    "nlopt",
                    NLOPT_VERSION,
                    NLOPT_COMMIT,
                    method,
                    nlopt_note,
                    solver,
                    sigma,
                )
            )
    return found


def write_templates(root: Path) -> list[Path]:
    """Write templates/noise and return the paths."""
    directory = root / "templates" / "noise"
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"{spec['id'].split('/', 1)[1]}.toml" for spec in specs()}
    for path in directory.glob("*.toml"):
        if path.name not in expected:
            path.unlink()
    written = []
    for spec in specs():
        name = spec["id"].split("/", 1)[1] + ".toml"
        path = directory / name
        path.write_text(_render(spec), encoding="utf-8")
        written.append(path)
    if len(written) != len(expected):
        raise RuntimeError("noise template count does not match the catalog")
    return written


def exact_hessian(method: str) -> bool:
    return str(method) in HESSIAN_METHODS


def allows_method(method: str, sigma: float) -> bool:
    """Exact Hessians are the clean force, so they stop at sigma 0."""
    return not (exact_hessian(method) and float(sigma) != 0.0)
