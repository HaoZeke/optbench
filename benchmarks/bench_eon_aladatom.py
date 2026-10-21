"""asv tracks for the two aluminum adatom AKMC templates.

The campaign states three repetitions. The default grid is the first
start, seed 20261018 and ``rep-1``. ``measure`` reads that committed
campaign start. It does not launch eonclient and it does not import
eOn. The recorded cost is ``eon_search_calls`` from the 151-atom
Voter-Chen ``eam_al`` catalogue with prefactor ``1e12``.
"""

from __future__ import annotations

import json
from pathlib import Path

from optbench.asv_tracks import TrackedSearch
from optbench.campaign import load_template

ROOT = Path(__file__).resolve().parents[1]
SEED = "20261018"
START = "rep-1"


def _campaign_record(template_name: str, seed: str, start: str) -> dict:
    template = load_template(ROOT, template_name)
    stem = f"eon-{template['code_version']}-{template_name.split('/', 1)[1]}"
    path = ROOT / "results" / "campaign" / f"{stem}.starts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("seed")) == str(seed) and str(row.get("start")) == str(start):
            failure = row.get("failure")
            return {
                "force_calls": float(row["force_calls"]),
                "success": bool(row["success"]),
                "failure": None if failure in (None, "") else str(failure),
                "total_searches": float(row.get("total_searches") or 0),
                "rate": float(row["rate"]) if row.get("rate") is not None else float("nan"),
                "counter": str(row.get("counter") or ""),
                "count_key": str(row.get("count_key") or ""),
            }
    raise ValueError(f"no campaign start {template_name} seed {seed} start {start}")


def _bind(spec: dict):
    class _Benchmark(TrackedSearch):
        param_names = ["seed", "start"]
        params = [[SEED], [START]]
        timeout = 7200
        template_name = spec["template"]
        problem_name = spec["problem"]
        cap = spec["cap"]
        potential = "eam_al"

        def setup(self, seed, start):
            return None

        def measure(self, seed, start):
            self.record = _campaign_record(self.template_name, str(seed), str(start))
            return self.record

        def time_search(self, seed, start):
            self.measure(seed, start)

        def track_force_calls(self, seed, start) -> float:
            return float(self.measure(seed, start)["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "crash" else 0.0

        def track_total_searches(self, seed, start) -> float:
            return float(self.measure(seed, start)["total_searches"])

        def track_rate(self, seed, start) -> float:
            return float(self.measure(seed, start)["rate"])

    _Benchmark.__name__ = spec["cls"]
    _Benchmark.__qualname__ = spec["cls"]
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    _Benchmark.track_total_searches.unit = "searches"
    _Benchmark.track_rate.unit = "per second"
    return _Benchmark


def _spec(cls: str, template_name: str) -> dict:
    template = load_template(ROOT, template_name)
    return {
        "cls": cls,
        "template": template["id"],
        "problem": template["problem"],
        "cap": int(template["failure_cap"]),
    }


EonAladatomAkmc = _bind(_spec("EonAladatomAkmc", "eon/aladatom-akmc"))
EonStandardAladatomAkmc = _bind(_spec("EonStandardAladatomAkmc", "eon/standard-aladatom-akmc"))
