"""OPTIM searches as airspeed-velocity benchmarks.

One class is one problem and one template. Parameters are the seed and
the archived start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start. It is read when the name
list is built.

``time_search`` is the timed search. The ``track_*`` methods read the
same record, so a full ``asv run`` records force calls, success, the
four failure classes, and wall time from that search. Counting is the
harness socket. A missing ``OPTIM_BIN`` raises in ``setup``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OPTBENCH_OPTIM_CACHE", "/tmp/optbench-asv-optim"))
RUNNER_REVISION = "socket-1"


def lj38_start_names() -> list[str]:
    return [f"{index:04d}" for index in range(200)]


def pt_start_names() -> list[str]:
    return [f"initial_{index}" for index in range(49)]


def selected_lj38() -> list[str]:
    names = lj38_start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def selected_pt() -> list[str]:
    names = pt_start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def _cache_path(template_id: str, start: str, seed: str) -> Path:
    safe = template_id.replace("/", "-")
    return CACHE / RUNNER_REVISION / safe / f"{start}__seed{seed}.json"


def _execute(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    from optbench.campaign import load_template, resolve_problem
    from optbench.optim_campaign import _geometry_specs, _prepare, install_relay, optim_binary

    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, problem_id)
    work = CACHE / "geometries" / problem_id.replace("/", "-")
    specs = _geometry_specs(ROOT, problem, template, work, optim_binary(), install_relay())
    spec = next((item for item in specs if item["name"] == start), None)
    if spec is None:
        raise RuntimeError(f"{problem_id} has no start {start}")
    record = _prepare(spec)
    record["seed"] = str(seed)
    record["template"] = template_id
    record["problem"] = problem_id
    path = _cache_path(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def _load(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    path = _cache_path(template_id, start, seed)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _execute(template_id, problem_id, start, seed)


def _cap(template_id: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, template["problem"])
    cap = _check_stop(problem, template)
    if cap is None:
        return int(((template.get("ini") or {}).get("Optimizer") or {}).get("max_iterations") or 10000)
    return int(cap)


def _bind(class_name: str, template_id: str, problem_id: str, starts: list[str], limit: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]
        timeout = limit

        def setup(self, seed, start):
            from optbench.optim_campaign import optim_binary

            optim_binary()

        def measure(self, seed, start):
            return _execute(template_id, problem_id, start, seed)

        def time_search(self, seed, start):
            self.record = self.measure(seed, start)

        def track_force_calls(self, seed, start) -> float:
            return float(_load(template_id, problem_id, start, seed)["force_calls"])

        def track_success(self, seed, start) -> float:
            record = _load(template_id, problem_id, start, seed)
            return 1.0 if record["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "crash" else 0.0

        track_force_calls.unit = "force calls"
        track_success.unit = "fraction"
        track_failure_cap.unit = "count"
        track_failure_wrong_stationary.unit = "count"
        track_failure_divergence.unit = "count"
        track_failure_crash.unit = "count"

    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    _Benchmark.cap = _cap(template_id)
    _Benchmark.template_id = template_id
    _Benchmark.problem_id = problem_id
    return _Benchmark


_ROWS = [
    ("OptimLj38Hef", "optim/hef-ffd-ts-lj38", "tsopt/LJ38", selected_lj38(), 1800),
    ("OptimLj38Bfgsts", "optim/bfgsts-ts-lj38", "tsopt/LJ38", selected_lj38(), 1800),
    ("OptimPtHef", "optim/hef-ffd-ts-pt", "tsopt/pt-island", selected_pt(), 3600),
    ("OptimPtBfgsts", "optim/bfgsts-ts-pt", "tsopt/pt-island", selected_pt(), 3600),
    ("OptimLj38RayleighRitz", "optim/rr-loweig", "lowest-eigenvector/lj38", selected_lj38(), 1800),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind(_class_name, _template_id, _problem_id, _starts, _timeout)
