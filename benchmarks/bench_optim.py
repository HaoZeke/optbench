"""OPTIM LBFGS minimization, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` is the
archived geometry. OPTIM has no optimizer seed. The noise templates add
their own force-noise seed inside the job.

The default grid is that seed and start ``0``. ``OPTBENCH_ASV_FULL=1``
selects every archived start. It is read at import, so the process that
launches asv has to set it.

``time_search`` is the minimization. ``setup`` raises
``NotImplementedError`` when the OPTIM binary is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Private alias. asv discovers every public class that defines track_*.
from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-optim"))


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def asv_benchmark_pattern(class_name: str) -> str:
    """Anchored selector. The trailing dot keeps a prefix from matching a tuned class."""
    return rf"^bench_optim\.{class_name}\."


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Optim" + "".join(part.capitalize() for part in slug.split("-"))


def _archive_count(problem_id: str) -> int:
    import tarfile

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
    from optbench.noise import KAPPA, window_energy_tolerance
    from optbench.runners.optim_min import _kind, _options

    found = []
    counts: dict[str, int] = {}
    for template in list_templates(ROOT):
        # ASE-socket rows and the unhooked template share this engine name.
        if template.get("engine") != "optim" or str(template.get("code_version")) != "02aaf45245af":
            continue
        problem_id = str(template["problem"])
        if problem_id not in counts:
            counts[problem_id] = _archive_count(problem_id) if os.environ.get("OPTBENCH_ASV_FULL") == "1" else 1
        problem = resolve_problem(ROOT, problem_id)
        convergence = problem.get("convergence") or {}
        updates, maxbfgs = _options(template)
        noise = template.get("noise") or {}
        sigma = float(noise.get("sigma", 0.0))
        if problem_id == "minimization/lj38-noise":
            kappa = float((problem.get("noise") or {}).get("kappa", KAPPA))
            energy_tol = window_energy_tolerance(float(convergence["energy_tol"]), sigma, kappa)
            seed = int(noise.get("seed", (problem.get("noise") or {})["seed"]))
        else:
            energy_tol = 0.0
            seed = 0
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "kind": _kind(problem_id),
                "updates": updates,
                "maxbfgs": maxbfgs,
                "sigma": sigma,
                "noise_seed": seed,
                "energy_tol": energy_tol,
                "gtol": float(convergence["force_norm"]),
                "cap": int(convergence["max_force_calls"]),
                "code_version": str(template["code_version"]),
                "n": counts[problem_id],
            }
        )
    return found


def _job(spec: dict, start: str) -> dict:
    from optbench.runners.optim_min import optim_binary

    binary = optim_binary()
    if binary is None:
        raise RuntimeError("OPTIM is not on PATH and $OPTIM is unset")
    index = int(start)
    geometries = _geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, box = geometries[index]
    kind = spec["kind"]
    return {
        "index": index,
        "start": name,
        "coords": [float(value) for value in coords],
        "box": None if kind == "lj" or box is None else [float(value) for value in box],
        "kind": kind,
        "sigma": spec["sigma"],
        "seed": spec["noise_seed"],
        "clean_tol": spec["gtol"],
        "energy_tol": spec["energy_tol"],
        "cap": spec["cap"],
        "updates": spec["updates"],
        "maxbfgs": spec["maxbfgs"],
        "binary": str(binary),
        "code_version": spec["code_version"],
        "work": str(_cache_dir() / "work" / spec["template_id"].replace("/", "-") / f"{SEED}-{index}"),
        "record_path": "",
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.runners.optim_min import _minimize_job

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


class _OptimSearch(_TrackedSearch):
    def setup(self, seed, start):
        from optbench.runners.optim_min import optim_binary

        if optim_binary() is None:
            raise NotImplementedError("OPTIM is not installed in this environment")
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


_OptimSearch.track_force_calls.unit = "force calls"
_OptimSearch.track_success.unit = "fraction"
_OptimSearch.track_failure_cap.unit = "count"
_OptimSearch.track_failure_wrong_stationary.unit = "count"
_OptimSearch.track_failure_divergence.unit = "count"
_OptimSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_OptimSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = limit
        spec = bound

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


def _preload() -> None:
    from optbench.runners.optim_min import optim_binary

    if os.environ.get("OPTBENCH_ASV_FULL") != "1" or optim_binary() is None:
        return
    seen = set()
    for spec in _specs():
        if spec["problem_id"] not in seen:
            seen.add(spec["problem_id"])
            _geometries(spec["problem_id"])


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)

_preload()
