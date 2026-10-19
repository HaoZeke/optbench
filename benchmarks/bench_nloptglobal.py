"""NLopt global optimization on Lennard-Jones 38.

``param_names`` are the seed and the start index. Seed ``0`` is the
archived geometry. The NLopt generator is seeded with the start index.

The default grid is that seed and start ``0``, which keeps ``asv check``
on one parameter. ``OPTBENCH_ASV_FULL=1`` selects every archived start.
It is read at import, so the process that launches asv has to set it.

``time_search`` is the search. Track methods read the record that call
wrote. The force count is a ``HarnessCounter`` around the Lennard-Jones
cluster. ``setup`` raises ``NotImplementedError`` when nlopt is not
installed, and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _nlopt_ready() -> bool:
    try:
        import nlopt  # noqa: F401
    except ImportError:
        return False
    return True


def selected_indices(count: int) -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Nlopt" + "".join(part.capitalize() for part in slug.split("-"))


def _archive_count(problem_id: str) -> int:
    from optbench.campaign import geometry_archive, resolve_problem

    archive = geometry_archive(ROOT, resolve_problem(ROOT, problem_id))
    with tarfile.open(archive) as packed:
        return sum(1 for name in packed.getnames() if name.endswith(".con"))


def _geometries(problem_id: str):
    if problem_id not in _GEOMETRIES:
        from optbench.campaign import resolve_problem
        from optbench.runners.nlopt_global import _load_geometries

        problem = resolve_problem(ROOT, problem_id)
        work = Path("/tmp/optbench-asv-geo")
        _GEOMETRIES[problem_id] = _load_geometries(ROOT, problem, None, work, work / "unused.dat")
    return _GEOMETRIES[problem_id]


_GEOMETRIES: dict[str, list] = {}


def _specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem
    from optbench.globalopt import energy_tolerance, reference_energy
    from optbench.runners.nlopt_global import _settings

    found = []
    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    for template in list_templates(ROOT):
        if template.get("engine") != "nlopt":
            continue
        problem_id = str(template["problem"])
        problem = resolve_problem(ROOT, problem_id)
        settings = _settings(problem, template)
        count = _archive_count(problem_id) if full else 1
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "settings": settings,
                "reference": reference_energy(problem),
                "tolerance": energy_tolerance(problem),
                "cap": settings["cap"],
                "n": count,
            }
        )
    return found


def _job(spec: dict, start: str) -> dict:
    from optbench.runners.nlopt_global import _job as make_job

    index = int(start)
    geometries = _geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords = geometries[index]
    return make_job(index, name, coords, spec["settings"], spec["reference"], spec["tolerance"])


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.runners.nlopt_global import run_job

    record = run_job(_job(spec, start))
    record["seed_param"] = SEED
    return record


class _NloptSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed search."""

    timeout = 7200

    def setup(self, seed, start):
        if not _nlopt_ready():
            raise NotImplementedError("nlopt is not installed in this environment")
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.spec, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _compute(self.spec, seed, start)
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


_NloptSearch.track_force_calls.unit = "force calls"
_NloptSearch.track_success.unit = "fraction"
_NloptSearch.track_failure_cap.unit = "count"
_NloptSearch.track_failure_wrong_stationary.unit = "count"
_NloptSearch.track_failure_divergence.unit = "count"
_NloptSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])

    class _Benchmark(_NloptSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = bound["cap"]
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
