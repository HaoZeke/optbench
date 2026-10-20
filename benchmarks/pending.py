"""One skipped benchmark for every problem and registry code.

A real code copies ``benchmarks/quadratic.py`` and implements
``measure``. This class is the placeholder so the grid exists. It has
``track_force_calls`` and ``time_search`` only. The success and failure
tracks are on the quadratic classes, which are the runs that produce
numbers. Multiplying those tracks across every unimplemented pair
would spawn a process per track.

``setup`` raises ``NotImplementedError``. asv-runner records that as a
skip (NaN) and does not save an exit code. See ``docs/asv-gaps.md``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def problem_ids() -> list[str]:
    found = []
    for path in sorted((ROOT / "problems").rglob("*.toml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("id = "):
                found.append(line.split("=", 1)[1].strip().strip('"'))
                break
        else:
            raise ValueError(f"{path} has no id")
    return found


# Scored by a harness campaign, outside this skipped grid.
_CAMPAIGN_ONLY = {"EON", "eon-develop-500350d9"}


def code_ids() -> list[str]:
    data = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    return [row["id"] for row in data["code"] if row["id"] not in _CAMPAIGN_ONLY]


class Pending:
    param_names = ["problem", "code", "seed", "start"]
    params = [problem_ids(), code_ids(), ["0"], ["0"]]
    repeat = 1
    rounds = 1
    number = 1
    min_run_count = 1
    sample_time = 0
    warmup_time = 0
    timeout = 60

    def setup(self, problem, code, seed, start):
        raise NotImplementedError(
            f"no benchmark for {problem} with {code}; copy benchmarks/quadratic.py"
        )

    def track_force_calls(self, problem, code, seed, start) -> float:
        raise NotImplementedError(problem)

    def time_search(self, problem, code, seed, start):
        raise NotImplementedError(problem)


Pending.track_force_calls.unit = "force calls"
