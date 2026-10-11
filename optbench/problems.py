"""Load problem contracts from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path


def load_problem(path: Path) -> dict:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("id", "suite", "problem", "contract"):
        if key not in data:
            raise ValueError(f"{path} is missing {key}")
    if data["id"] != f"{data['suite']}/{data['problem']}":
        raise ValueError(f"{path} id does not match suite/problem")
    metrics = data.get("metrics") or {}
    if "required" not in metrics:
        raise ValueError(f"{path} is missing metrics.required")
    data["_path"] = path
    return data


def load_all(root: Path) -> dict[str, dict]:
    problems = {}
    for path in sorted(root.rglob("*.toml")):
        problem = load_problem(path)
        ident = problem["id"]
        if ident in problems:
            raise ValueError(f"duplicate problem id {ident}")
        problems[ident] = problem
    return problems
