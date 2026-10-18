"""CP2K GEO_OPT dimer entries as airspeed-velocity benchmarks.

One class is one template. Parameters are the seed and the archived
start. The default grid is seed ``"0"`` and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start, and it is read at import.

``time_search`` writes a sidecar. The ``track_*`` methods read that
sidecar, so one search supplies the force calls, the success bit, the
failure class, and the wall time. ``setup`` does not search.

The saddle-search catalog is a campaign: one row is one displacement,
and the class grid here is the fixed tsopt starts. A missing
``CP2K_BIN`` is a ``RuntimeError``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OPTBENCH_CP2K_CACHE", "/tmp/optbench-asv-cp2k"))


def lj38_start_names() -> list[str]:
    return [f"{index:04d}" for index in range(200)]


def pt_start_names() -> list[str]:
    return [f"initial_{index}" for index in range(49)]


def _selected(names: list[str]) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return list(names)
    return list(names[:1])


def _sidecar(template_id: str, start: str, seed: str) -> Path:
    return CACHE / template_id.replace("/", "-") / f"{start}__seed{seed}.json"


def _require_binary() -> None:
    from optbench.cp2k_ts import cp2k_binary

    cp2k_binary()


def _execute(template_id: str, start: str, seed: str) -> dict:
    from optbench.cp2k_ts import run_start, spec_for_start

    spec = spec_for_start(ROOT, template_id, start, work=CACHE / "runs" / template_id.replace("/", "-"))
    spec["seed"] = str(seed)
    record = run_start(spec)
    record.pop("x", None)
    record.pop("mode_axis", None)
    record["seed"] = str(seed)
    path = _sidecar(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def _load(template_id: str, start: str, seed: str) -> dict:
    path = _sidecar(template_id, start, seed)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _execute(template_id, start, seed)


def _cap(template_id: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_id)
    cap = _check_stop(resolve_problem(ROOT, template["problem"]), template)
    if cap is None:
        raise RuntimeError(f"{template_id} has no force-call cap")
    return int(cap)


def _bind(class_name: str, template_id: str, problem_id: str, starts: list[str], timeout: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]

        def setup(self, seed, start):
            _require_binary()

        def measure(self, seed, start):
            return _execute(template_id, str(start), str(seed))

        def time_search(self, seed, start):
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            path = _sidecar(template_id, str(start), str(seed))
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            self.record = record

        def track_force_calls(self, seed, start) -> float:
            return float(_load(template_id, str(start), str(seed))["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "crash" else 0.0

        track_force_calls.unit = "force calls"
        track_success.unit = "fraction"
        track_failure_cap.unit = "count"
        track_failure_wrong_stationary.unit = "count"
        track_failure_divergence.unit = "count"
        track_failure_crash.unit = "count"

    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.timeout = timeout
    _Benchmark.cap = _cap(template_id)
    _Benchmark.template_id = template_id
    _Benchmark.problem_id = problem_id
    return _Benchmark


_ROWS = [
    ("Cp2kLj38DimerBfgs", "cp2k/lj38-dimer-bfgs", "tsopt/LJ38", _selected(lj38_start_names()), 3600),
    ("Cp2kLj38DimerBfgsTuned", "cp2k/lj38-dimer-bfgs-tuned", "tsopt/LJ38", _selected(lj38_start_names()), 3600),
    ("Cp2kLj38DimerCg", "cp2k/lj38-dimer-cg", "tsopt/LJ38", _selected(lj38_start_names()), 3600),
    ("Cp2kLj38DimerCgTuned", "cp2k/lj38-dimer-cg-tuned", "tsopt/LJ38", _selected(lj38_start_names()), 3600),
    ("Cp2kPtDimerBfgs", "cp2k/pt-island-dimer-bfgs", "tsopt/pt-island", _selected(pt_start_names()), 7200),
    ("Cp2kPtDimerBfgsTuned", "cp2k/pt-island-dimer-bfgs-tuned", "tsopt/pt-island", _selected(pt_start_names()), 7200),
    ("Cp2kPtDimerCg", "cp2k/pt-island-dimer-cg", "tsopt/pt-island", _selected(pt_start_names()), 7200),
    ("Cp2kPtDimerCgTuned", "cp2k/pt-island-dimer-cg-tuned", "tsopt/pt-island", _selected(pt_start_names()), 7200),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind(_class_name, _template_id, _problem_id, _starts, _timeout)
