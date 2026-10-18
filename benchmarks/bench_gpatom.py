"""ase-gpatom AIDNEB as airspeed-velocity benchmarks.

One class is one problem. The template is ``gpatom/aidneb``. Parameters
are the seed and the archived start. The default grid is one seed and
the first geometry. ``OPTBENCH_ASV_FULL=1`` selects every start, and it
is read at import.

``time_search`` is the timed search. It writes a sidecar. The ``track_*``
methods read that sidecar, so a full ``asv run`` records force calls,
success, the four failure classes, and wall time from the same search.
A missing sidecar searches, so a track-only run is a real number.

The seed is ``"0"``. AIDNEB does not draw from it. Counting is a
``HarnessCounter`` on the spec potential. Gaussian-process predictions
are not force calls. The module does not import ase-gpatom at import.
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

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "gpatom/aidneb"
PIN = "572bd64ef61ccf498e061162609bb5f16e39d8a4"

PROBLEMS = (
    ("Lj38De", "double-ended-saddle-search/lj38"),
    ("Lj38TwoTs", "double-ended-saddle-search/lj38_2ts_separation"),
    ("PtIsland", "double-ended-saddle-search/pt-island"),
    ("Heptamer", "neb/Pt-heptamer"),
)


def result_root() -> Path:
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def sidecar_path(problem_name: str, seed: str, start: str) -> Path:
    slug = problem_name.replace("/", "-")
    return result_root() / "results" / "gpatom-asv" / slug / str(seed) / f"{start}.json"


def all_starts(problem_name: str) -> list[str]:
    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.pes import load_pairs

    problem = resolve_problem(ROOT, problem_name)
    pairs = load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )
    return [pair.name for pair in pairs]


def selected_starts(names: list[str]) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return list(names)
    return list(names[:1])


def _ensure_gpatom() -> None:
    try:
        import gpatom  # noqa: F401
    except ImportError:
        import subprocess

        url = f"ase-gpatom @ git+https://gitlab.com/gpatom/ase-gpatom.git@{PIN}"
        subprocess.check_call(
            ["uv", "pip", "install", "--python", sys.executable, "ase==3.29.0", "scipy", url],
        )


def _write_sidecar(problem_name: str, seed: str, start: str, record: dict) -> None:
    path = sidecar_path(problem_name, seed, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _load(problem_name: str, seed: str, start: str) -> dict:
    path = sidecar_path(problem_name, seed, start)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _measure(problem_name, seed, start)


def _measure(problem_name: str, seed: str, start: str) -> dict:
    _ensure_gpatom()
    from optbench.gpatom_band import search_named

    record = search_named(ROOT, problem_name, TEMPLATE, str(start), str(seed))
    record["force_calls"] = float(record["force_calls"])
    record["success"] = bool(record["success"])
    if record["success"]:
        record["failure"] = None
    _write_sidecar(problem_name, str(seed), str(start), record)
    return record


def _bind(class_name: str, problem_name: str, starts: list[str]) -> type:
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = 86400
        problem_name = problem_name

        def setup(self, seed, start):
            path = sidecar_path(self.problem_name, str(seed), str(start))
            self.record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

        def measure(self, seed, start):
            return _measure(self.problem_name, str(seed), str(start))

        def time_search(self, seed, start):
            print(f"start {self.problem_name} {start}", file=sys.stderr, flush=True)
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            _write_sidecar(self.problem_name, str(seed), str(start), record)
            self.record = record

        def _ready(self, seed, start) -> dict:
            record = getattr(self, "record", None)
            if record is None:
                record = _load(self.problem_name, str(seed), str(start))
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

    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    return _Benchmark


for _prefix, _problem in PROBLEMS:
    globals()[f"{_prefix}GpatomAidneb"] = _bind(f"{_prefix}GpatomAidneb", _problem, selected_starts(all_starts(_problem)))
