"""SaddleSearch.jl dimer searches as airspeed-velocity benchmarks.

One class is one geometry template. Parameters are the seed and the
archived start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start, and it is read at import.

``time_search`` is the timed search. It writes a cache record. The
``track_*`` methods read that cache, so a full ``asv run`` records
force calls, success, the four failure classes, and wall time from the
same search. ``setup`` does not search.

The seed is the string ``"0"``. Saddle-search displacements stay on
the campaign runner. The published campaign is the ranked row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JULIA_NUM_THREADS", "1")
os.environ.setdefault("JULIA", str(Path.home() / "julia" / "bin" / "julia"))
os.environ.setdefault("SADDLESEARCHJL_SRC", str(Path.home() / "src" / "SaddleSearch.jl"))

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OPTBENCH_SADDLESEARCHJL_CACHE", "/tmp/optbench-asv-saddlesearchjl"))
RUNNER_REVISION = "harness-julia-1"


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
    from optbench.campaign import _check_stop, load_template, resolve_problem
    from optbench.rgsaddle_minmode import _geometries
    from optbench.saddlesearchjl import JuliaSession, _jsonable, _spec_from_geometry, execute

    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, problem_id)
    cap = _check_stop(problem, template)
    work = CACHE / "geometries" / problem_id.replace("/", "-")
    rows = _geometries(ROOT, problem, work)
    row = next((item for item in rows if item["name"] == start), None)
    if row is None:
        raise RuntimeError(f"{problem_id} has no start {start}")
    optimizer = (template.get("ini") or {}).get("Optimizer") or {}
    convergence = problem.get("convergence") or {}
    force_tol = float(optimizer.get("converged_force", convergence.get("force_norm", 1e-3)))
    spec = _spec_from_geometry(row, problem, template, cap, force_tol)
    spec["seed"] = seed
    session = JuliaSession.start()
    try:
        record = execute(spec, session)
    finally:
        session.close()
    stored = _jsonable(record)
    stored["seed"] = seed
    stored["template"] = template_id
    stored["problem"] = problem_id
    path = _cache_path(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(stored, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return stored


def _load(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    path = _cache_path(template_id, start, seed)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _execute(template_id, problem_id, start, seed)


def _cap(template_id: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_id)
    return int(_check_stop(resolve_problem(ROOT, template["problem"]), template))


def _bind(class_name: str, template_id: str, problem_id: str, starts: list[str], limit: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]
        timeout = limit

        def setup(self, seed, start):
            return None

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
    _Benchmark.cap = _cap(template_id)
    _Benchmark.template_id = template_id
    _Benchmark.problem_id = problem_id
    return _Benchmark


_ROWS = [
    ("SsjlLj38Static", "saddlesearchjl/lj38-static", "tsopt/LJ38", selected_lj38(), 600),
    ("SsjlLj38Bb", "saddlesearchjl/lj38-bb", "tsopt/LJ38", selected_lj38(), 600),
    ("SsjlLj38Superlinear", "saddlesearchjl/lj38-superlinear", "tsopt/LJ38", selected_lj38(), 600),
    ("SsjlLj38SuperlinearTuned", "saddlesearchjl/lj38-superlinear-tuned", "tsopt/LJ38", selected_lj38(), 600),
    ("SsjlLj38Ode", "saddlesearchjl/lj38-ode", "tsopt/LJ38", selected_lj38(), 600),
    ("SsjlPtStatic", "saddlesearchjl/pt-static", "tsopt/pt-island", selected_pt(), 900),
    ("SsjlPtBb", "saddlesearchjl/pt-bb", "tsopt/pt-island", selected_pt(), 900),
    ("SsjlPtSuperlinear", "saddlesearchjl/pt-superlinear", "tsopt/pt-island", selected_pt(), 900),
    ("SsjlPtSuperlinearTuned", "saddlesearchjl/pt-superlinear-tuned", "tsopt/pt-island", selected_pt(), 900),
    ("SsjlPtOde", "saddlesearchjl/pt-ode", "tsopt/pt-island", selected_pt(), 900),
    ("SsjlModeStatic", "saddlesearchjl/mode-static", "lowest-eigenvector/lj38", selected_lj38(), 600),
    ("SsjlModeBb", "saddlesearchjl/mode-bb", "lowest-eigenvector/lj38", selected_lj38(), 600),
    ("SsjlModeSuperlinear", "saddlesearchjl/mode-superlinear", "lowest-eigenvector/lj38", selected_lj38(), 600),
    ("SsjlModeSuperlinearTuned", "saddlesearchjl/mode-superlinear-tuned", "lowest-eigenvector/lj38", selected_lj38(), 600),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind(_class_name, _template_id, _problem_id, _starts, _timeout)
