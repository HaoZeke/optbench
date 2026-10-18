"""CatLearn ML-NEB band searches as airspeed-velocity benchmarks.

One class is one problem and one template. Parameters are the seed and
the archived start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start, and it is read at import.

``time_search`` is the timed search. The ``track_*`` methods read the
record from that search. The seed is ``"0"``. ML-NEB does not draw
random numbers. Importing this module does not import CatLearn.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]

SPECS = (
    ("Lj38Default", "double-ended-saddle-search/lj38", "catlearn/de-lj38-default"),
    ("Lj38Tuned", "double-ended-saddle-search/lj38", "catlearn/de-lj38-tuned"),
    ("Lj38TwoTsDefault", "double-ended-saddle-search/lj38_2ts_separation", "catlearn/de-lj38-2ts-default"),
    ("Lj38TwoTsTuned", "double-ended-saddle-search/lj38_2ts_separation", "catlearn/de-lj38-2ts-tuned"),
    ("PtIslandDefault", "double-ended-saddle-search/pt-island", "catlearn/de-pt-island-default"),
    ("PtIslandTuned", "double-ended-saddle-search/pt-island", "catlearn/de-pt-island-tuned"),
    ("HeptamerDefault", "neb/Pt-heptamer", "catlearn/neb-pt-heptamer-default"),
    ("HeptamerTuned", "neb/Pt-heptamer", "catlearn/neb-pt-heptamer-tuned"),
)

_FAILURES = ("cap", "wrong_stationary", "divergence", "crash")


def all_starts(problem_name: str) -> list[str]:
    from optbench.campaign import resolve_problem
    from optbench.catlearn_band import cached_pairs

    problem = resolve_problem(ROOT, problem_name)
    return list(cached_pairs(ROOT, problem))


def problem_cap(problem_name: str) -> int:
    from optbench.campaign import resolve_problem

    return int(resolve_problem(ROOT, problem_name)["convergence"]["max_force_calls"])


def selected_starts(names: list[str]) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return list(names)
    return list(names[:1])


def _failure_of(record: dict) -> str | None:
    if record.get("success"):
        return None
    label = record.get("failure")
    if label in _FAILURES:
        return str(label)
    return "crash"


def _bind(class_name: str, problem: str, template: str, starts: list[str], cap: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = 86400
        problem_name = problem
        template_name = template
        template_id = template
        problem_id = problem

        def setup(self, seed, start):
            return None

        def measure(self, seed, start):
            from optbench.catlearn_band import search_named

            try:
                record = search_named(ROOT, self.problem_name, self.template_name, str(start), str(seed))
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
                    "crash": type(exc).__name__,
                    "problem": self.problem_name,
                    "template": self.template_name,
                }
            record["force_calls"] = float(record["force_calls"])
            record["success"] = bool(record["success"])
            record["failure"] = _failure_of(record)
            if record["success"]:
                record["failure"] = None
            return record

    _Benchmark.cap = cap
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    return _Benchmark


for _class_name, _problem, _template in SPECS:
    globals()[_class_name] = _bind(
        _class_name,
        _problem,
        _template,
        selected_starts(all_starts(_problem)),
        problem_cap(_problem),
    )
