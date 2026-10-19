"""SciPy minimization templates for the clean problems.

Lennard-Jones 38 and Morse bulk. The noise rows live in
``optbench.noise_catalog`` because that file owns the sigma grid.
"""

from __future__ import annotations

from pathlib import Path

from optbench.runners.scipy_min import COMMIT, METHODS, VERSION

PROBLEMS = (
    (
        "minimization/lj38",
        "lj38",
        0.01,
        True,
    ),
    (
        "minimization/morse-bulk",
        "morse",
        0.001,
        False,
    ),
)

_ORDER = (
    ("lbfgsb", "L-BFGS-B"),
    ("bfgs", "BFGS"),
    ("cg", "CG"),
    ("trust-krylov", "trust-krylov"),
    ("trust-ncg", "trust-ncg"),
    ("newton-cg", "Newton-CG"),
)


def _toml(value: object) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _note(method: str, lj: bool) -> str:
    if method in {"BFGS", "CG", "Newton-CG"}:
        cap = (
            "The per-atom cap is 0.2, the same length as the eOn max move. "
            "A larger trial is mapped back inside that ball before the potential runs. "
            "The map is the identity through 0.18 and smooth beyond it. "
        )
    else:
        cap = (
            "The per-atom cap is 0.2, the same length as the eOn max move. "
            "A trial past that cap is refused before the potential runs. "
        )
    harness = (
        "Default row. fun and jac are the harness kernel. "
        "A repeated coordinate returns the cached energy and forces and does not add a call. "
    )
    budget = "The iteration ceiling sits above the force-call cap. The harness counter is the budget."
    if method in {"L-BFGS-B", "BFGS", "CG"}:
        norm = (
            "The SciPy infinity-norm gtol is the problem force norm divided by sqrt(n), "
            "so a library stop is inside the Euclidean stop. "
            "The callback stops when the Euclidean norm meets the problem value. "
        )
        if method == "L-BFGS-B":
            norm += "L-BFGS-B ftol is machine epsilon, so the relative energy test does not return ahead of the force stop. "
        if lj:
            return harness + cap + norm + budget
        return (
            harness
            + "Morse bulk has no per-atom cap. The cell is periodic. "
            + norm
            + budget
        )
    products = (
        "Each analytic Hessian-vector product counts as one force call. "
    )
    if method == "Newton-CG":
        stop = (
            "Newton-CG xtol is 0, so the step test does not stop the search. "
            "The callback stops when the Euclidean force norm meets the problem value. "
        )
    else:
        stop = (
            "The trust-region gtol is the Euclidean problem force norm. "
            + (
                "The trust radius cannot exceed 0.2 times sqrt(atoms). "
                if lj
                else "Library trust-radius defaults are unchanged. "
            )
        )
    if lj:
        return harness + products + cap + stop + budget
    return harness + products + "Morse bulk has no per-atom cap. " + stop + budget


def specs() -> list[dict]:
    found = []
    for problem, slug, force_norm, lj in PROBLEMS:
        for name, method in _ORDER:
            section = {"method": method}
            if lj:
                section["maxstep"] = 0.2
            found.append(
                {
                    "id": f"scipy/{slug}-{name}",
                    "engine": "scipy",
                    "problem": problem,
                    "code": "SciPy",
                    "code_version": VERSION,
                    "commit": COMMIT,
                    "algorithm": method,
                    "tuning": "default",
                    "note": _note(method, lj),
                    "scipy": section,
                    "force_norm": force_norm,
                }
            )
    if [row["algorithm"] for row in found if row["problem"] == "minimization/lj38"] != [method for _name, method in _ORDER]:
        raise RuntimeError("scipy method order drifted")
    if len(found) != len(PROBLEMS) * len(METHODS):
        raise RuntimeError("scipy template count drifted")
    return found


def _render(spec: dict) -> str:
    lines = []
    for key in ("id", "engine", "problem", "code", "code_version", "commit", "algorithm", "tuning", "note"):
        lines.append(f"{key} = {_toml(spec[key])}")
    lines.append("")
    lines.append("[scipy]")
    for key, value in spec["scipy"].items():
        lines.append(f"{key} = {_toml(value)}")
    lines.append("")
    lines.append("[ini.Optimizer]")
    lines.append(f"converged_force = {_toml(spec['force_norm'])}")
    lines.append("max_iterations = 10000")
    lines.append("")
    return "\n".join(lines)


def write_templates(root: Path) -> list[Path]:
    directory = root / "templates" / "scipy"
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"{spec['id'].split('/', 1)[1]}.toml" for spec in specs()}
    for path in directory.glob("*.toml"):
        if path.name not in expected:
            path.unlink()
    written = []
    for spec in specs():
        path = directory / (spec["id"].split("/", 1)[1] + ".toml")
        path.write_text(_render(spec), encoding="utf-8")
        written.append(path)
    return written
