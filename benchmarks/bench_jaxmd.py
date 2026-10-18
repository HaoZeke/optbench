"""JAX MD minimization, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` is the
archived geometry. The solvers have no optimizer seed, so the seed is
not passed into FIRE, gradient descent, or L-BFGS. Noise templates add
their own force-noise seed inside the job.

The default grid is that seed and start ``0``, which keeps ``asv check``
and ``asv run --quick`` on one parameter. ``OPTBENCH_ASV_FULL=1`` selects
every archived start. It is read at import, so the process that launches
asv has to set it.

``time_search`` is the minimization. Track methods read the record that
call wrote. The force count is ``RgpotCounter.calculate`` on a
``HarnessCounter``. ``setup`` raises ``NotImplementedError`` when JAX MD,
jaxopt, or optax is not installed, and asv-runner stores that parameter
as NaN.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-jaxmd"))


def _jaxmd_ready() -> bool:
    try:
        import jax_md  # noqa: F401
        import jaxopt  # noqa: F401
        import optax  # noqa: F401
    except ImportError:
        return False
    return True


def selected_indices(count: int) -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def asv_benchmark_pattern(class_name: str) -> str:
    """Anchored selector for one JAX MD class."""
    return rf"^bench_jaxmd\.{class_name}\."


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Jaxmd" + "".join(part.capitalize() for part in slug.split("-"))


def _archive_count(problem_id: str) -> int:
    from optbench.campaign import geometry_archive, resolve_problem

    archive = geometry_archive(ROOT, resolve_problem(ROOT, problem_id))
    with tarfile.open(archive) as packed:
        return sum(1 for name in packed.getnames() if name.endswith(".con"))


def _geometries(problem_id: str):
    if problem_id not in _GEOMETRIES:
        from optbench.campaign import resolve_problem
        from optbench.runners.rgmin_min import _load_geometries

        problem = resolve_problem(ROOT, problem_id)
        work = Path("/tmp/optbench-asv-geo")
        _GEOMETRIES[problem_id] = _load_geometries(ROOT, problem, None, work, work / "unused.dat")
    return _GEOMETRIES[problem_id]


_GEOMETRIES: dict[str, list] = {}


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem

    found = []
    counts: dict[str, int] = {}
    for template in list_templates(ROOT):
        if template.get("engine") != "jaxmd":
            continue
        problem_id = str(template["problem"])
        if problem_id not in counts:
            counts[problem_id] = _archive_count(problem_id) if os.environ.get("OPTBENCH_ASV_FULL") == "1" else 1
        problem = resolve_problem(ROOT, problem_id)
        convergence = problem.get("convergence") or {}
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "template": template,
                "gtol": float(convergence["force_norm"]),
                "cap": int(convergence["max_force_calls"]),
                "n": counts[problem_id],
            }
        )
    return found


def _job(spec: dict, start: str) -> dict:
    from optbench.campaign import resolve_problem
    from optbench.noise import KAPPA, window_energy_tolerance
    from optbench.runners.jaxmd_min import _kind, _options

    index = int(start)
    geometries = _geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, box = geometries[index]
    template = spec["template"]
    options = _options(template)
    problem = resolve_problem(ROOT, spec["problem_id"])
    noisy = spec["problem_id"] == "minimization/lj38-noise"
    sigma = float((template.get("noise") or {}).get("sigma") or 0.0)
    seed = int((template.get("noise") or {}).get("seed") or 0)
    gtol = float(spec["gtol"])
    energy_tol = gtol
    if noisy:
        energy_tol = window_energy_tolerance(
            float(problem["convergence"]["energy_tol"]),
            sigma,
            float((problem.get("noise") or {}).get("kappa", KAPPA)),
        )
    kind = _kind(spec["problem_id"])
    return {
        "index": index,
        "start": name,
        "coords": coords.tolist(),
        "box": None if kind == "lj" or box is None else box.tolist(),
        "kind": kind,
        "method": str(options["method"]),
        "maxstep": float(options["maxstep"]),
        "step_size": float(options.get("step_size") or 0.0),
        "history_size": int(options.get("history_size") or 10),
        "memory_size": int(options.get("memory_size") or 10),
        "gtol": gtol,
        "clean_tol": gtol,
        "energy_tol": energy_tol,
        "cap": int(spec["cap"]),
        "sigma": sigma,
        "seed": seed,
        "noisy": noisy,
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    """Minimize this start and store the record. The caller is the timed sample."""
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.runners.jaxmd_min import _minimize_job

    record = _minimize_job(_job(spec, start))
    record["seed"] = SEED
    path = _cache_path(spec["template_id"], SEED, str(int(start)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], SEED, str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _JaxmdSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed minimization."""

    def setup(self, seed, start):
        if not _jaxmd_ready():
            raise NotImplementedError("jax-md, jaxopt, or optax is not installed in this environment")
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.spec, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _load(self.spec, seed, start)
        return self.record

    def track_force_calls(self, seed, start) -> float:
        return float(self._stored(seed, start)["force_calls"])

    def track_success(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["success"] else 0.0

    def track_failure_cap(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "cap" else 0.0

    def track_failure_wrong_stationary(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "wrong_stationary" else 0.0

    def track_failure_divergence(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "divergence" else 0.0

    def track_failure_crash(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "crash" else 0.0


_JaxmdSearch.track_force_calls.unit = "force calls"
_JaxmdSearch.track_success.unit = "fraction"
_JaxmdSearch.track_failure_cap.unit = "count"
_JaxmdSearch.track_failure_wrong_stationary.unit = "count"
_JaxmdSearch.track_failure_divergence.unit = "count"
_JaxmdSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_JaxmdSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = limit
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


def _preload() -> None:
    if os.environ.get("OPTBENCH_ASV_FULL") != "1" or not _jaxmd_ready():
        return
    seen = set()
    for spec in _specs():
        if spec["problem_id"] not in seen:
            seen.add(spec["problem_id"])
            _geometries(spec["problem_id"])


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)

_preload()
