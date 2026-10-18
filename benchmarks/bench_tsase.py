"""TSASE basin hopping and minima hopping, one class per template.

``param_names`` are the seed and the start index. The seed is the archived
global-optimization seed, and every start of one template uses it.

The default grid is that seed and start ``0``. ``OPTBENCH_ASV_FULL=1``
selects every geometry. It is read at import, so the process that launches
asv has to set it.

``time_search`` is the search. Track methods read the record that call
wrote. ``setup`` raises ``NotImplementedError`` when TSASE is not installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "314159"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-tsase"))


def _tsase_ready() -> bool:
    try:
        import tsase  # noqa: F401
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
    return "Tsase" + "".join(part.capitalize() for part in slug.split("-"))


def _geometry_count(problem_id: str) -> int:
    if os.environ.get("OPTBENCH_ASV_FULL") != "1":
        return 1
    from optbench.campaign import resolve_problem
    from optbench.runners.tsase_global import load_geometries

    return len(load_geometries(ROOT, resolve_problem(ROOT, problem_id)))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    counts: dict[str, int] = {}
    for template in list_templates(ROOT):
        if template.get("engine") != "tsase":
            continue
        problem_id = str(template["problem"])
        if problem_id not in counts:
            counts[problem_id] = _geometry_count(problem_id)
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "cap": int(template["stop"]["max_force_calls"]),
                "n": counts[problem_id],
            }
        )
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    """Search this start. The caller is the timed sample."""
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived seed {SEED}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.tsase_global import (
        basin_settings,
        energy_threshold,
        load_geometries,
        minima_settings,
        run_start,
    )

    template = load_template(ROOT, spec["template_id"])
    section = template["tsase"]
    method = str(section["method"])
    problem = resolve_problem(ROOT, spec["problem_id"])
    geometries = load_geometries(ROOT, problem)
    index = int(start)
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, positions = geometries[index]
    threshold = energy_threshold(problem)
    record = run_start(
        {
            "index": index,
            "start": name,
            "positions": positions.tolist(),
            "method": method,
            "cap": spec["cap"],
            "threshold": threshold,
            "seed": int(seed),
            "workdir": str(_cache_dir() / "work" / spec["class_name"] / f"start-{index}"),
            "bh_settings": basin_settings(section, threshold) if method == "basin_hopping" else {},
            "mh_settings": minima_settings(section) if method == "minima_hopping" else {},
        }
    )
    record["seed"] = SEED
    path = _cache_path(spec["template_id"], SEED, str(index))
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


class _TsaseSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed search."""

    def setup(self, seed, start):
        if not _tsase_ready():
            raise NotImplementedError("tsase is not installed in this environment")
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


_TsaseSearch.track_force_calls.unit = "force calls"
_TsaseSearch.track_success.unit = "fraction"
_TsaseSearch.track_failure_cap.unit = "count"
_TsaseSearch.track_failure_wrong_stationary.unit = "count"
_TsaseSearch.track_failure_divergence.unit = "count"
_TsaseSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])

    class _Benchmark(_TsaseSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = bound["cap"]
        spec = bound
        template_id = spec["template_id"]
        problem_id = spec["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
