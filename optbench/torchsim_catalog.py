"""Templates for TorchSim FIRE and gradient descent on the minimization problems.

The values under ``[torchsim]`` are the constructor defaults of
``fire_init`` / ``fire_step`` and ``gradient_descent_step`` in TorchSim
0.6.2. FIRE's default flavor is the ASE flavor.
"""

from __future__ import annotations

from pathlib import Path

from optbench.noise_catalog import SIGMAS, sigma_slug

COMMIT = "5540d893a849db9772081229a86d5f68537ad037"
VERSION = "0.6.2"
CODE = "TorchSim"
PROBLEM_NOISE = "minimization/lj38-noise"
SEED = 20261016

FIRE = {
    "method": "fire",
    "fire_flavor": "ase_fire",
    "dt_start": 0.1,
    "alpha_start": 0.1,
    "dt_max": 1.0,
    "n_min": 5,
    "f_inc": 1.1,
    "f_dec": 0.5,
    "f_alpha": 0.99,
    "max_step": 0.2,
}
GD = {
    "method": "gradient_descent",
    "pos_lr": 0.01,
}

_CLEAN = (
    ("minimization/lj38", "lj38", 0.01),
    ("minimization/morse-bulk", "morse", 1.0e-3),
)
_METHODS = (
    ("fire", "FIRE", FIRE),
    ("gd", "GD", GD),
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


def _note(algorithm: str, problem: str, sigma: float | None) -> str:
    if algorithm == "FIRE":
        base = (
            "Default profile. FIRE uses the ASE flavor at the constructor defaults: "
            "dt_start 0.1, alpha_start 0.1, dt_max 1, n_min 5, f_inc 1.1, f_dec 0.5, "
            "f_alpha 0.99, max_step 0.2. SpecPotentialModel evaluates the problem potential."
        )
    else:
        base = (
            "Default profile. Gradient descent uses the constructor pos_lr 0.01. "
            "The step is that rate times the force, and this optimizer has no displacement cap. "
            "SpecPotentialModel evaluates the problem potential."
        )
        if problem in {"minimization/lj38", "minimization/lj38-noise"}:
            base += (
                " Opening forces on these Lennard-Jones starts are large enough "
                "that this step can leave the cutoff."
            )
    if problem == "minimization/morse-bulk":
        base += " The orthorhombic cell stays fixed."
    if sigma is not None:
        base += (
            " Gaussian force noise uses the problem seed, kappa, and window. "
            "The stop is the noise-aware force norm and, at positive sigma, the clean-energy window. "
            "The index is the clean Hessian with the rigid-mode rule."
        )
    return base


def _spec(
    ident: str,
    problem: str,
    algorithm: str,
    section: dict,
    force_norm: float,
    sigma: float | None,
) -> dict:
    row = {
        "id": ident,
        "engine": "torchsim",
        "problem": problem,
        "code": CODE,
        "code_version": VERSION,
        "commit": COMMIT,
        "algorithm": algorithm,
        "tuning": "default",
        "note": _note(algorithm, problem, sigma),
        "torchsim": dict(section),
        "stop": {"norm": "l2", "force_norm": float(force_norm), "max_force_calls": 10000},
    }
    if sigma is not None:
        row["noise"] = {"sigma": float(sigma), "seed": SEED}
    return row


def specs() -> list[dict]:
    """One dict per campaign template, clean problems first, then noise."""
    found: list[dict] = []
    for problem, slug, force_norm in _CLEAN:
        for method_slug, algorithm, section in _METHODS:
            found.append(
                _spec(
                    f"torchsim/{slug}-{method_slug}",
                    problem,
                    algorithm,
                    section,
                    force_norm,
                    None,
                )
            )
    for method_slug, algorithm, section in _METHODS:
        for sigma in SIGMAS:
            found.append(
                _spec(
                    f"torchsim/lj38-noise-{method_slug}-{sigma_slug(sigma)}",
                    PROBLEM_NOISE,
                    algorithm,
                    section,
                    0.01,
                    float(sigma),
                )
            )
    return found


def _render(spec: dict) -> str:
    lines = []
    for key in (
        "id",
        "engine",
        "problem",
        "code",
        "code_version",
        "commit",
        "algorithm",
        "tuning",
        "note",
    ):
        lines.append(f"{key} = {_toml_value(spec[key])}")
    lines.append("")
    lines.append("[torchsim]")
    for key, value in spec["torchsim"].items():
        lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    lines.append("[stop]")
    for key, value in spec["stop"].items():
        lines.append(f"{key} = {_toml_value(value)}")
    if "noise" in spec:
        lines.append("")
        lines.append("[noise]")
        lines.append(f"sigma = {_toml_value(spec['noise']['sigma'])}")
        lines.append(f"seed = {int(spec['noise']['seed'])}")
    lines.append("")
    return "\n".join(lines)


def write_templates(root: Path) -> list[Path]:
    """Write templates/torchsim and return the paths."""
    directory = root / "templates" / "torchsim"
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
    return written
