"""pyGSM double-ended growing string as airspeed-velocity benchmarks.

One class is one problem and the ``pygsm/de-gsm`` template. Parameters
are the seed and the archived start. The default grid is one seed and
the first geometry. ``OPTBENCH_ASV_FULL=1`` selects every start, and
it is read at import.

``time_search`` is the timed search. It writes a sidecar. The ``track_*``
methods read that sidecar, so a full ``asv run`` records force calls,
success, the four failure classes, and wall time from the same search.
``setup`` does not search. A track whose sidecar is missing searches,
so a track-only run is a real number.

The seed is the string ``"0"``. The string does not draw random numbers.
Sidecars live under ``results/pygsm-asv`` and are not the published
store. The published numbers are the asv result file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
_PIN = "1b4d5a5557a61adc6a3df7405d6b7255b1fa4865"
_FULL = os.environ.get("OPTBENCH_ASV_FULL") == "1"

CLASS_PROBLEM: dict[str, str] = {}
CLASS_TEMPLATE: dict[str, str] = {}
CLASS_CAP: dict[str, int] = {}


def ensure_pygsm() -> None:
    """Install the pinned pyGSM commit into this interpreter when it is absent."""
    try:
        import pyGSM  # noqa: F401

        return
    except ImportError:
        pass
    uv = shutil.which("uv")
    if uv is None:
        candidate = Path.home() / ".local" / "bin" / "uv"
        if candidate.is_file():
            uv = str(candidate)
    if uv is None:
        raise RuntimeError("uv is required to install pyGSM into the ASV environment")
    lock = Path("/tmp/optbench-pygsm-install.lock")
    import fcntl

    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            import pyGSM  # noqa: F401

            return
        except ImportError:
            subprocess.check_call(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "networkx",
                    "setuptools==80.10.2",
                ]
            )
            subprocess.check_call(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--no-deps",
                    f"pygsm @ git+https://github.com/ZimmermanGroup/pyGSM.git@{_PIN}",
                ]
            )


def result_root() -> Path:
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def sidecar_path(problem_name: str, template_name: str, seed: str, start: str) -> Path:
    slug = problem_name.replace("/", "-") + "--" + template_name.replace("/", "-")
    return result_root() / "results" / "pygsm-asv" / slug / str(seed) / f"{start}.json"


def _starts(problem_name: str) -> list[str]:
    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.pes import load_pairs

    problem = resolve_problem(ROOT, problem_name)
    pairs = load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )
    names = [pair.name for pair in pairs]
    if _FULL:
        return names
    return names[:1]


def _write_sidecar(problem_name: str, template_name: str, seed: str, start: str, record: dict) -> None:
    path = sidecar_path(problem_name, template_name, seed, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _bind(class_name: str, problem: str, cap: int, timeout: int):
    starts = _starts(problem)
    template = "pygsm/de-gsm"

    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = timeout
        problem_name = problem
        template_id = template
        cap = cap

        def setup(self, seed, start):
            ensure_pygsm()
            path = sidecar_path(self.problem_name, self.template_id, str(seed), str(start))
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.record = None

        def measure(self, seed, start):
            from optbench.pygsm_band import search_named

            try:
                record = search_named(ROOT, self.problem_name, self.template_id, str(start), str(seed))
            except Exception as exc:
                record = {
                    "force_calls": 0.0,
                    "success": False,
                    "failure": "crash",
                    "crash": type(exc).__name__,
                }
            record["force_calls"] = float(record["force_calls"])
            record["success"] = bool(record["success"])
            if record["success"]:
                record["failure"] = None
            return record

        def time_search(self, seed, start):
            print(f"start {self.template_id} {start}", file=sys.stderr, flush=True)
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            _write_sidecar(self.problem_name, self.template_id, str(seed), str(start), record)
            self.record = record
            print(
                f"done {self.template_id} {start} calls={record['force_calls']} "
                f"success={record['success']} failure={record['failure']}",
                file=sys.stderr,
                flush=True,
            )

        def _ready(self, seed, start) -> dict:
            record = getattr(self, "record", None)
            if record is None:
                record = self.measure(seed, start)
                _write_sidecar(self.problem_name, self.template_id, str(seed), str(start), record)
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
    CLASS_PROBLEM[class_name] = problem
    CLASS_TEMPLATE[class_name] = template
    CLASS_CAP[class_name] = cap
    return _Benchmark


PygsmLj38De = _bind("PygsmLj38De", "double-ended-saddle-search/lj38", 2500, 1800)
PygsmLj38TwoTs = _bind(
    "PygsmLj38TwoTs",
    "double-ended-saddle-search/lj38_2ts_separation",
    10000,
    7200,
)
