"""pele transition-state searches as airspeed-velocity benchmarks.

One class is one template. Parameters are the seed and the archived
start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every geometry start, and it is read at
import. The saddle-search classes stay on the first displacement.

``time_search`` writes a sidecar. The ``track_*`` methods read that
sidecar. ``setup`` does not search. A track whose sidecar is missing
searches, so a track-only run is a real number.

Sidecars live under ``results/pele-asv`` and are not the published
store. The published numbers are the asv result file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]

_FAILURES = ("cap", "wrong_stationary", "divergence", "crash")
CLASS_TEMPLATE: dict[str, str] = {}
CLASS_PROBLEM: dict[str, str] = {}
CLASS_CAP: dict[str, int] = {}


def result_root() -> Path:
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def starts_for(problem: str) -> list[str]:
    if problem == "tsopt/pt-island":
        return [f"initial_{index}" for index in range(49)]
    if problem == "saddle-search/pt-island":
        return ["rep1-search1"]
    return [f"{index:04d}" for index in range(200)]


def selected_starts(problem: str) -> list[str]:
    names = starts_for(problem)
    if problem == "saddle-search/pt-island":
        return names
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def sidecar_path(template_name: str, seed: str, start: str) -> Path:
    slug = template_name.replace("/", "-")
    return result_root() / "results" / "pele-asv" / slug / str(seed) / f"{start}.json"


def _write_sidecar(template_name: str, seed: str, start: str, record: dict) -> None:
    path = sidecar_path(template_name, seed, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _failure_of(record: dict) -> str | None:
    if record.get("success"):
        return None
    label = record.get("failure")
    if label in _FAILURES:
        return str(label)
    return "crash"


def _bind(class_name: str, problem: str, template: str, cap: int):
    starts = selected_starts(problem)

    class _Benchmark(TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = 1800
        problem_name = problem
        template_name = template

        def setup(self, seed, start):
            try:
                import pele  # noqa: F401
            except ImportError as exc:
                raise NotImplementedError("pele is not installed") from exc
            path = sidecar_path(self.template_name, str(seed), str(start))
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.record = None

        def measure(self, seed, start):
            from optbench.pele_ts import measure_start

            try:
                record = measure_start(ROOT, self.template_name, str(start))
            except Exception as exc:
                record = {
                    "schema": "optbench.start/v2",
                    "start": str(start),
                    "seed": str(seed),
                    "force_calls": 0,
                    "verify_calls": 0,
                    "counter": "harness",
                    "comparable": True,
                    "success": False,
                    "failure": "crash",
                    "reasons": ["crash"],
                    "error": type(exc).__name__,
                    "problem": self.problem_name,
                    "template": self.template_name,
                }
            record["seed"] = str(seed)
            record["force_calls"] = float(record["force_calls"])
            record["success"] = bool(record["success"])
            record["failure"] = _failure_of(record)
            if record["success"]:
                record["failure"] = None
            return record

        def time_search(self, seed, start):
            print(f"start {self.template_name} {start}", file=sys.stderr, flush=True)
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            _write_sidecar(self.template_name, str(seed), str(start), record)
            self.record = record
            print(
                f"done {self.template_name} {start} calls={record['force_calls']} "
                f"success={record['success']} failure={record['failure']}",
                file=sys.stderr,
                flush=True,
            )

        def _ready(self, seed, start) -> dict:
            record = getattr(self, "record", None)
            if record is None:
                record = self.measure(seed, start)
                _write_sidecar(self.template_name, str(seed), str(start), record)
                self.record = record
            return record

        def track_force_calls(self, seed, start) -> float:
            return float(self._ready(seed, start)["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "crash" else 0.0

    _Benchmark.cap = cap
    _Benchmark.problem_id = problem
    _Benchmark.template_id = template
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    CLASS_TEMPLATE[class_name] = template
    CLASS_PROBLEM[class_name] = problem
    CLASS_CAP[class_name] = cap
    return _Benchmark


_ROWS = (
    ("PeleLj38Ts", "tsopt/LJ38", "pele/lj38-ts", 10000),
    ("PeleLj38TsFfd", "tsopt/LJ38", "pele/lj38-ts-ffd", 10000),
    ("PeleLj38TsCd", "tsopt/LJ38", "pele/lj38-ts-cd", 10000),
    ("PelePtIslandTs", "tsopt/pt-island", "pele/pt-island-ts", 10000),
    ("PelePtIslandTsFfd", "tsopt/pt-island", "pele/pt-island-ts-ffd", 10000),
    ("PeleLj38Loweig", "lowest-eigenvector/lj38", "pele/lj38-loweig", 10000),
    ("PeleLj38LoweigForward", "lowest-eigenvector/lj38", "pele/lj38-loweig-forward", 10000),
    ("PeleLj38LoweigCentral", "lowest-eigenvector/lj38", "pele/lj38-loweig-central", 10000),
    ("PelePtIslandSearch", "saddle-search/pt-island", "pele/pt-island-search", 5064),
    ("PelePtIslandSearchFfd", "saddle-search/pt-island", "pele/pt-island-search-ffd", 5064),
)

for _name, _problem, _template, _cap in _ROWS:
    globals()[_name] = _bind(_name, _problem, _template, _cap)
