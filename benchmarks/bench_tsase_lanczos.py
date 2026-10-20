"""TSASE 1.0 lowest-eigenvector track.

The campaign is ``results/campaign/tsase-1.0-lj38-lanczos``. ``measure``
reads that file. It does not rebuild the Lanczos search.
"""

from __future__ import annotations

import json
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ID = "tsase/lj38-lanczos"
PROBLEM_ID = "lowest-eigenvector/lj38"
CAP = 10000
_STARTS = [f"{index:04d}" for index in range(200)]


def _load(start: str) -> dict:
    path = ROOT / "results" / "campaign" / "tsase-1.0-lj38-lanczos.starts.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("start")) == str(start):
            return row
    raise KeyError(f"{TEMPLATE_ID} has no start {start}")


class Lj38Lanczos(TrackedSearch):
    param_names = ["seed", "start"]
    params = [["0"], _STARTS]
    timeout = 60
    template_id = TEMPLATE_ID
    problem_id = PROBLEM_ID
    cap = CAP

    def measure(self, seed, start):
        row = _load(start)
        failure = row.get("failure")
        return {
            "force_calls": float(row["force_calls"]),
            "success": bool(row["success"]),
            "failure": None if failure in (None, "", "none") else failure,
        }
