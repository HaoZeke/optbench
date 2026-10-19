"""asv-runner methods shared by optbench benchmarks.

``measure`` returns a record with ``force_calls``, ``success``, and
``failure``. Track methods return plain floats. A timing dict would be
passed through ``compute_stats`` and the force-call count would be
replaced. ``time_search`` is the wall-clock sample of one search.

``repeat`` is 1 so a saved run is one sample. asv-runner treats
``repeat == 0`` as up to ten repeats.
"""

from __future__ import annotations


class TrackedSearch:
    repeat = 1
    rounds = 1
    number = 1
    min_run_count = 1
    sample_time = 0
    warmup_time = 0
    timeout = 120

    def setup(self, *params):
        self.record = self.measure(*params)

    def measure(self, *params):
        raise NotImplementedError("measure")

    def time_search(self, *params):
        self.measure(*params)

    def track_force_calls(self, *params) -> float:
        return float(self.record["force_calls"])

    def track_success(self, *params) -> float:
        return 1.0 if self.record["success"] else 0.0

    def track_failure_cap(self, *params) -> float:
        return 1.0 if self.record["failure"] == "cap" else 0.0

    def track_failure_wrong_stationary(self, *params) -> float:
        return 1.0 if self.record["failure"] == "wrong_stationary" else 0.0

    def track_failure_divergence(self, *params) -> float:
        return 1.0 if self.record["failure"] == "divergence" else 0.0

    def track_failure_crash(self, *params) -> float:
        return 1.0 if self.record["failure"] == "crash" else 0.0

    def track_failure_not_converged(self, *params) -> float:
        return 1.0 if self.record["failure"] == "not_converged" else 0.0


TrackedSearch.track_force_calls.unit = "force calls"
TrackedSearch.track_success.unit = "fraction"
TrackedSearch.track_failure_cap.unit = "count"
TrackedSearch.track_failure_wrong_stationary.unit = "count"
TrackedSearch.track_failure_divergence.unit = "count"
TrackedSearch.track_failure_crash.unit = "count"
TrackedSearch.track_failure_not_converged.unit = "count"
