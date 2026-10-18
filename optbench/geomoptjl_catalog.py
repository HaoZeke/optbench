"""Templates for GeometryOptimization.jl on the minimization problems.

LBFGS, CG, and GradientDescent are the package solvers OptimLBFGS,
OptimCG, and OptimSD. The linesearch and the 0.8 bohr maxstep stay at
the library default. The force stop and the call cap come from the
problem file.
"""

from __future__ import annotations

from pathlib import Path

COMMIT = "5b145886c19042f2ef13e21627f28d86027f719c"
VERSION = "0.1.6"
CODE = "GeometryOptimization.jl"
SEED = 20261016
SIGMAS = ((0.0, "s0"), (1e-4, "s1em4"), (1e-3, "s1em3"), (1e-2, "s1em2"))
METHODS = (
    ("lbfgs", "LBFGS", "OptimLBFGS uses the Hager-Zhang initial step."),
    ("cg", "CG", "OptimCG keeps the package conjugate-gradient defaults."),
    ("gradientdescent", "GradientDescent", "OptimSD calls Optim.jl GradientDescent."),
)


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sigma_text(sigma: float) -> str:
    if sigma == 0.0:
        return "0.0"
    if sigma == 1e-4:
        return "1e-4"
    if sigma == 1e-3:
        return "1e-3"
    if sigma == 1e-2:
        return "1e-2"
    raise ValueError(f"sigma {sigma} is not a documented level")


def _note(method_blurb: str, problem: str, sigma: float | None) -> str:
    base = (
        f"{method_blurb} BackTracking linesearch of order 2 and maxstep 0.8 bohr. "
        "The iteration ceiling is the force-call cap. The potential is the harness callback."
    )
    if problem == "minimization/lj38":
        return (
            base
            + " The callback stops at the Euclidean force norm 0.01. "
            "Positions are labeled bohr and the energy is labeled hartree, so austrip leaves the spec numbers unchanged."
        )
    if problem == "minimization/morse-bulk":
        return (
            base
            + " The callback stops at the Euclidean force norm 0.001 eV/Å. "
            "Positions are angstrom and the energy is eV. The solver converts those to atomic units internally. "
            "The recorded force is the spec gradient in eV/Å."
        )
    sentence = (
        " The callback stops on the noise-aware force norm. "
        "At positive sigma the clean energy over the last window is part of that stop. "
        "The index is the clean Hessian with the rigid-mode rule. "
            "Positions are labeled bohr and the energy is labeled hartree, so austrip leaves the spec numbers unchanged."
    )
    if sigma == 0.0:
        sentence = (
            " Sigma is 0, so the stop is the clean Euclidean force norm 0.01. "
            "The index is the clean Hessian with the rigid-mode rule. "
            "Positions are labeled bohr and the energy is labeled hartree, so austrip leaves the spec numbers unchanged."
        )
    return base + sentence


def specs() -> list[dict]:
    found = []
    for method, algorithm, blurb in METHODS:
        opener = f"GeometryOptimization 0.1.6 {blurb}"
        for problem, stem, force, cap in (
            ("minimization/lj38", "lj38", 0.01, 10000),
            ("minimization/morse-bulk", "morse", 1.0e-3, 10000),
        ):
            found.append(
                {
                    "id": f"geomoptjl/{stem}-{method}",
                    "problem": problem,
                    "algorithm": algorithm,
                    "method": method,
                    "force": force,
                    "cap": cap,
                    "sigma": None,
                    "note": _note(opener, problem, None),
                }
            )
        for sigma, slug in SIGMAS:
            found.append(
                {
                    "id": f"geomoptjl/noise-{method}-{slug}",
                    "problem": "minimization/lj38-noise",
                    "algorithm": algorithm,
                    "method": method,
                    "force": 0.01,
                    "cap": 10000,
                    "sigma": sigma,
                    "note": _note(opener, "minimization/lj38-noise", sigma),
                }
            )
    return found


def render(spec: dict) -> str:
    lines = [
        f"id = {_quote(spec['id'])}",
        'engine = "geomoptjl"',
        f"problem = {_quote(spec['problem'])}",
        f"code = {_quote(CODE)}",
        f"code_version = {_quote(VERSION)}",
        f"commit = {_quote(COMMIT)}",
        f"algorithm = {_quote(spec['algorithm'])}",
        'tuning = "default"',
        f"note = {_quote(spec['note'])}",
        "",
        "[geomoptjl]",
        f"method = {_quote(spec['method'])}",
        "",
        "[ini.Optimizer]",
        f"converged_force = {spec['force']}",
        f"max_iterations = {spec['cap']}",
        "",
    ]
    if spec["sigma"] is not None:
        lines.extend(
            [
                "[noise]",
                f"sigma = {_sigma_text(spec['sigma'])}",
                f"seed = {SEED}",
                "",
            ]
        )
    return "\n".join(lines)


def write_templates(root: Path) -> list[Path]:
    directory = root / "templates" / "geomoptjl"
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"{spec['id'].split('/', 1)[1]}.toml" for spec in specs()}
    for path in directory.glob("*.toml"):
        if path.name not in expected:
            path.unlink()
    written = []
    for spec in specs():
        path = directory / (spec["id"].split("/", 1)[1] + ".toml")
        path.write_text(render(spec), encoding="utf-8")
        written.append(path)
    return written
