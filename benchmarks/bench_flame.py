"""FLAME minhocao, one class per globalopt template.

``param_names`` are the seed and the start index. The seed is the
template seed. The default grid is that seed and start 0.
``OPTBENCH_ASV_FULL=1`` selects every archived start. It is read at
import.

``time_search`` is one minhocao run. It always evaluates the potential.
``setup`` raises ``NotImplementedError`` when the FLAME binary is missing,
and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]


def _binary_ready() -> bool:
    from optbench.runners.flame_go import flame_binary

    return flame_binary().is_file()


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Flame" + "".join(part.capitalize() for part in slug.split("-"))


def _count(problem_id: str) -> int:
    from optbench.campaign import resolve_problem
    from optbench.runners.flame_go import load_geometries

    problem = resolve_problem(ROOT, problem_id)
    return len(load_geometries(ROOT, problem, None, Path("/tmp/optbench-asv-flame-geo"), Path("/tmp/unused-flame.dat")))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem
    from optbench.runners.flame_go import cap_of, success_ceiling

    found = []
    counts: dict[str, int] = {}
    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    for template in list_templates(ROOT):
        if template.get("engine") != "flame":
            continue
        problem_id = str(template["problem"])
        if problem_id not in counts:
            counts[problem_id] = _count(problem_id) if full else 1
        problem = resolve_problem(ROOT, problem_id)
        flame = template.get("flame") or {}
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "seed": str(int(flame.get("seed", 314159))),
                "cap": cap_of(problem, template),
                "ceiling": success_ceiling(problem, template),
                "nlminx": int(flame.get("nlminx", 100000)),
                "n": counts[problem_id],
            }
        )
    return found


def _job(spec: dict, start: str) -> dict:
    from optbench.campaign import resolve_problem
    from optbench.runners.flame_go import flame_binary, load_geometries

    index = int(start)
    problem = resolve_problem(ROOT, spec["problem_id"])
    geometries = load_geometries(
        ROOT,
        problem,
        None,
        Path("/tmp/optbench-asv-flame-geo"),
        Path("/tmp/unused-flame.dat"),
    )
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, _box = geometries[index]
    return {
        "index": index,
        "start": name,
        "coords": coords.reshape(-1).tolist(),
        "port": 26000 + (os.getpid() % 1000) + index,
        "work": str(Path("/tmp/optbench-asv-flame") / spec["template_id"].replace("/", "-") / f"{index:04d}"),
        "binary": str(flame_binary()),
        "seed": int(spec["seed"]),
        "nlminx": spec["nlminx"],
        "cap": spec["cap"],
        "target": spec["ceiling"],
        "idle_timeout": 300,
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != spec["seed"]:
        raise ValueError(f"seed {seed} is not the template seed {spec['seed']}")
    from optbench.runners.flame_go import _run_start

    return _run_start(_job(spec, start))


class _FlameSearch(TrackedSearch):
    def setup(self, seed, start):
        if not _binary_ready():
            raise NotImplementedError("FLAME binary is not available")
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


_FlameSearch.track_force_calls.unit = "force calls"
_FlameSearch.track_success.unit = "fraction"
_FlameSearch.track_failure_cap.unit = "count"
_FlameSearch.track_failure_wrong_stationary.unit = "count"
_FlameSearch.track_failure_divergence.unit = "count"
_FlameSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec

    class _Benchmark(_FlameSearch):
        param_names = ["seed", "start"]
        params = [[spec["seed"]], selected_indices(spec["n"])]
        timeout = 86400
        cap = spec["cap"]
        spec = bound
        template_id = spec["template_id"]
        problem_id = spec["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
