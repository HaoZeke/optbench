"""Force-call tracks for the Lennard-Jones 38 score.

One measurement is every start in the problem file, stopped by that
file. The value ASV stores is the unrounded mean. benchmark.dat still
carries the rounded mean.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _client() -> Path:
    env = os.environ.get("EONCLIENT")
    if env:
        return Path(env)
    found = shutil.which("eonclient")
    if not found:
        raise RuntimeError("pass eonclient on PATH or set EONCLIENT")
    return Path(found)


class Lj38EonLbfgsM100:
    """All 1000 starts, template eon-client/lbfgs-m100."""

    timeout = 86400
    repeat = 1
    rounds = 1
    problem = "minimization/lj38"
    template = "eon-client/lbfgs-m100"

    def setup_cache(self):
        from optbench.campaign import run

        work = Path(tempfile.mkdtemp(prefix="asv-optbench-"))
        try:
            summary = run(
                ROOT,
                self.problem,
                self.template,
                client=_client(),
                output=work / "benchmark.dat",
                work=work / "runs",
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return {
            "mean_unrounded": summary["mean_unrounded"],
            "force_calls_min": summary["force_calls_min"],
            "force_calls_max": summary["force_calls_max"],
            "nfailed": summary["nfailed"],
            "starts": summary["starts"],
            "counts": summary["counts"],
        }

    def track_force_calls(self, data):
        return float(data["mean_unrounded"])

    track_force_calls.unit = "force calls"

    def track_force_calls_min(self, data):
        return float(data["force_calls_min"])

    track_force_calls_min.unit = "force calls"

    def track_force_calls_max(self, data):
        return float(data["force_calls_max"])

    track_force_calls_max.unit = "force calls"

    def track_nfailed(self, data):
        return int(data["nfailed"])

    track_nfailed.unit = "starts"
