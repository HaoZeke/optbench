"""rgsaddle saddle searches as asv benchmarks.

``param_names`` is ``["seed", "start"]``. The default grid is seed ``"0"``
and the first geometry. ``OPTBENCH_ASV_FULL=1`` at import selects every
start (200 on LJ38, 49 on the Pt island). The saved run is that grid:

``OPTBENCH_ASV_FULL=1 uv run asv run COMMIT^! -b bench_rgsaddle``

Pass ``-b`` so the eOn Lennard-Jones grid is not expanded in the same
process. A missing ``RGSADDLE_LIB`` is a ``RuntimeError``. It is not a
skipped parameter.

Counting is ``RgpotCounter.calculate`` on a ``HarnessCounter``. The
class inherits ``TrackedSearch``. ``time_search`` is one search and
writes the record. The ``track_*`` methods read that record. asv runs
``time_*`` before ``track_*``.

Nonlinear CG is not a row. rgsaddle refuses a line-search stepper on a
projected force, and the C ABI maps every accepted code other than
L-BFGS to FIRE. Pt P-RFO and the saddle-search campaign are not in
this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OPTBENCH_RGSADDLE_CACHE", "/tmp/optbench-asv-rgsaddle"))
RUNNER_REVISION = "harness-rgpot-1"


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


def _require_library() -> None:
    if not os.environ.get("RGSADDLE_LIB"):
        raise RuntimeError("set RGSADDLE_LIB to the pinned librgsaddle.so")


def _cache_path(template_id: str, start: str, seed: str) -> Path:
    safe = template_id.replace("/", "-")
    return CACHE / RUNNER_REVISION / safe / f"{start}__seed{seed}.json"


def _execute(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    os.environ["RAYON_NUM_THREADS"] = "1"
    _require_library()
    from optbench.campaign import _check_stop, load_template, resolve_problem
    from optbench.rgsaddle_minmode import _dispatch, _geometries, _spec_from_geometry

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
    record = _dispatch(spec)
    record["seed"] = seed
    record["template"] = template_id
    record["problem"] = problem_id
    record.pop("x", None)
    path = _cache_path(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
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
    return _check_stop(resolve_problem(ROOT, template["problem"]), template)


def _bind(class_name: str, template_id: str, problem_id: str, starts: list[str], limit: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]
        timeout = limit

        def setup(self, seed, start):
            _require_library()

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


# (class, template, problem, starts, timeout seconds)
_ROWS = [
    ("RgsaddleLj38DimerLbfgs", "rgsaddle/dimer-lbfgs", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38DimerLbfgsTuned", "rgsaddle/dimer-lbfgs-tuned", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38DimerFire", "rgsaddle/dimer-fire", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38DimerFireTuned", "rgsaddle/dimer-fire-tuned", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38Lanczos", "rgsaddle/lanczos", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38LanczosTuned", "rgsaddle/lanczos-tuned", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38DimerRotskip", "rgsaddle/dimer-rotskip", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38DimerRotskipTuned", "rgsaddle/dimer-rotskip-tuned", "tsopt/LJ38", selected_lj38(), 1800),
    ("RgsaddleLj38Prfo", "rgsaddle/prfo", "tsopt/LJ38", selected_lj38(), 7200),
    ("RgsaddleLj38PrfoTuned", "rgsaddle/prfo-tuned", "tsopt/LJ38", selected_lj38(), 7200),
    ("RgsaddlePtDimerLbfgs", "rgsaddle/pt-dimer-lbfgs", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtDimerLbfgsTuned", "rgsaddle/pt-dimer-lbfgs-tuned", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtDimerFire", "rgsaddle/pt-dimer-fire", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtDimerFireTuned", "rgsaddle/pt-dimer-fire-tuned", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtLanczos", "rgsaddle/pt-lanczos", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtLanczosTuned", "rgsaddle/pt-lanczos-tuned", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtDimerRotskip", "rgsaddle/pt-dimer-rotskip", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddlePtDimerRotskipTuned", "rgsaddle/pt-dimer-rotskip-tuned", "tsopt/pt-island", selected_pt(), 7200),
    ("RgsaddleModeDimer", "rgsaddle/mode-dimer", "lowest-eigenvector/lj38", selected_lj38(), 1800),
    ("RgsaddleModeDimerTuned", "rgsaddle/mode-dimer-tuned", "lowest-eigenvector/lj38", selected_lj38(), 1800),
    ("RgsaddleModeLanczos", "rgsaddle/mode-lanczos", "lowest-eigenvector/lj38", selected_lj38(), 1800),
    ("RgsaddleModeLanczosTuned", "rgsaddle/mode-lanczos-tuned", "lowest-eigenvector/lj38", selected_lj38(), 1800),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind(_class_name, _template_id, _problem_id, _starts, _timeout)
