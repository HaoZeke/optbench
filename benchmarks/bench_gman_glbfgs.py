"""Gman GL-BFGS Hessian tracks for the stored Pt-heptamer campaign.

``measure`` reads ``results/campaign/gman-vtst-r128-neb-pt-glbfgs-hess.starts.jsonl``.
It does not relax a band. asv discovery does not rerun the Morse NEB.
``force_calls`` in that file is the undivided evaluation total.
"""

from __future__ import annotations

import json
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ID = "gman/neb-pt-glbfgs-hess"
PROBLEM_ID = "neb/Pt-heptamer"
CAP = 10000
_STARTS = [f"min{index:02d}" for index in range(1, 14)]


def _load(start: str) -> dict:
    path = ROOT / "results" / "campaign" / "gman-vtst-r128-neb-pt-glbfgs-hess.starts.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("start")) == str(start):
            return row
    raise KeyError(f"{TEMPLATE_ID} has no start {start}")


class GmanGlbfgsHess(TrackedSearch):
    param_names = ["seed", "start"]
    params = [["0"], _STARTS]
    timeout = 60
    template_id = TEMPLATE_ID
    problem_id = PROBLEM_ID
    cap = CAP

    def measure(self, seed, start):
        del seed
        row = _load(start)
        return {
            "force_calls": row["force_calls"],
            "success": bool(row["success"]),
            "failure": None if row["success"] else "divergence",
        }


GmanGlbfgsHess.track_force_calls.unit = "force calls"
GmanGlbfgsHess.track_success.unit = "fraction"
GmanGlbfgsHess.track_failure_cap.unit = "count"
GmanGlbfgsHess.track_failure_wrong_stationary.unit = "count"
GmanGlbfgsHess.track_failure_divergence.unit = "count"
GmanGlbfgsHess.track_failure_crash.unit = "count"
