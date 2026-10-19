"""Sella 0.1.1 LJ38 transition-state tracks.

The campaign is ``results/campaign/sella-0.1.1-lj38-ts``. ``measure``
reads that file. It does not import Sella 0.1.1, and asv discovery does
not rerun the trust-region search.
"""

from __future__ import annotations

import json
from pathlib import Path

from optbench.asv_tracks import TrackedSearch
from optbench.failures import failure_of

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ID = "sella-0.1.1/lj38-ts"
PROBLEM_ID = "tsopt/LJ38"
CAP = 10000
_STARTS = [f"{index:04d}" for index in range(200)]


def _load(start: str) -> dict:
    path = ROOT / "results" / "campaign" / "sella-0.1.1-lj38-ts.starts.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("start")) == str(start):
            return row
    raise KeyError(f"{TEMPLATE_ID} has no start {start}")


class Lj38Ts(TrackedSearch):
    param_names = ["seed", "start"]
    params = [["0"], _STARTS]
    timeout = 60
    template_id = TEMPLATE_ID
    problem_id = PROBLEM_ID
    cap = CAP

    def measure(self, seed, start):
        row = _load(start)
        success = bool(row["success"])
        return {
            "force_calls": float(row["force_calls"]),
            "success": success,
            "failure": None if success else failure_of(row, CAP),
        }
