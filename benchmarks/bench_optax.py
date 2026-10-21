"""Optax mlopt training, one class per template.

``param_names`` are the template seed and the start index. The default
grid is that seed and start ``0``. ``OPTBENCH_ASV_FULL=1`` selects every
initialization. It is read at import.

``time_search`` trains the start. Track methods read the record that
call wrote. The force count is the harness loss-and-gradient count.
``setup`` raises ``NotImplementedError`` when optax is not installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-optax"))


def _ready() -> bool:
    try:
        import optax  # noqa: F401
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
    return "Optax" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "optax":
            continue
        problem_id = str(template["problem"])
        problem = resolve_problem(ROOT, problem_id)
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "cap": int(template["failure_cap"]),
                "seed": str(template["optax"]["seed"]),
                "n": int(problem["sample"]["initializations"]),
            }
        )
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{int(start)}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != str(spec["seed"]):
        raise ValueError(f"seed {seed} is not the template seed {spec['seed']}")
    from optbench.runners.optax_ml import train_indices

    record = train_indices(ROOT, spec["template_id"], [int(start)])[0]
    record["seed"] = str(seed)
    path = _cache_path(spec["template_id"], str(seed), str(int(start)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], str(seed), str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _OptaxSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed training."""

    def setup(self, seed, start):
        if not _ready():
            raise NotImplementedError("optax is not installed in this environment")
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


_OptaxSearch.track_force_calls.unit = "force calls"
_OptaxSearch.track_success.unit = "fraction"
_OptaxSearch.track_failure_cap.unit = "count"
_OptaxSearch.track_failure_wrong_stationary.unit = "count"
_OptaxSearch.track_failure_divergence.unit = "count"
_OptaxSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_OptaxSearch):
        param_names = ["seed", "start"]
        params = [[bound["seed"]], starts]
        timeout = 14400
        cap = limit
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]
        spec = bound

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
