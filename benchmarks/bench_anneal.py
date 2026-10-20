"""Anneal cluster search, one class per template.

The default grid is one start. ``OPTBENCH_ASV_FULL=1`` selects every
archived start. ``setup`` raises ``NotImplementedError`` unless
``OPTBENCH_ASV_ANNEAL=1``, so a plain asv invocation does not start a search.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    ("Lj38Communicating", "anneal/lj38-communicating", "globalopt/lj38", 2000000),
    ("Lj38Recommended", "anneal/lj38-recommended", "globalopt/lj38", 2000000),
    ("Lj75Communicating", "anneal/lj75-communicating", "globalopt/lj75", 1000000),
    ("Lj75Recommended", "anneal/lj75-recommended", "globalopt/lj75", 1000000),
    ("Lj98Communicating", "anneal/lj98-communicating", "globalopt/lj98", 1000000000),
    ("Lj98Recommended", "anneal/lj98-recommended", "globalopt/lj98", 1000000000),
    ("Lj150Communicating", "anneal/lj150-communicating", "globalopt/lj150", 10000000),
    ("Lj150Recommended", "anneal/lj150-recommended", "globalopt/lj150", 10000000),
)


def selected_starts(problem_id: str) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") != "1":
        return ["0"]
    name = problem_id.split("/", 1)[1]
    data = tomllib.loads((ROOT / "problems" / "globalopt" / f"{name}.toml").read_text(encoding="utf-8"))
    return [str(index) for index in range(int(data["sample"]["starts"]))]


def _bind(class_name: str, template_id: str, problem_id: str, cap: int) -> type:
    class AnnealSearch(TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], selected_starts(problem_id)]
        timeout = 86400

        def setup(self, seed, start):
            del seed, start
            if os.environ.get("OPTBENCH_ASV_ANNEAL") != "1":
                raise NotImplementedError("OPTBENCH_ASV_ANNEAL is unset")

        def time_search(self, seed, start):
            del seed, start
            raise NotImplementedError("OPTBENCH_ASV_ANNEAL is unset")

        def track_force_calls(self, seed, start) -> float:
            del seed, start
            return float("nan")

        def track_success(self, seed, start) -> float:
            del seed, start
            return float("nan")

        def track_failure_cap(self, seed, start) -> float:
            del seed, start
            return float("nan")

        def track_failure_wrong_stationary(self, seed, start) -> float:
            del seed, start
            return float("nan")

        def track_failure_divergence(self, seed, start) -> float:
            del seed, start
            return float("nan")

        def track_failure_crash(self, seed, start) -> float:
            del seed, start
            return float("nan")

    AnnealSearch.__name__ = class_name
    AnnealSearch.template_id = template_id
    AnnealSearch.problem_id = problem_id
    AnnealSearch.cap = cap
    AnnealSearch.track_force_calls.unit = "force calls"
    AnnealSearch.track_success.unit = "fraction"
    AnnealSearch.track_failure_cap.unit = "count"
    AnnealSearch.track_failure_wrong_stationary.unit = "count"
    AnnealSearch.track_failure_divergence.unit = "count"
    AnnealSearch.track_failure_crash.unit = "count"
    return AnnealSearch


for _name, _template, _problem, _cap in SPECS:
    globals()[_name] = _bind(_name, _template, _problem, _cap)
