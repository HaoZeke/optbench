"""AGOX global optimization, one class per template.

``param_names`` are the seed base and the start index. The search seed
is the base plus the index, the same rule as the campaign runner.

The default grid is seed base 314159 and start ``0``. ``asv check`` and
a one-start run stay on that pair. ``OPTBENCH_ASV_FULL=1`` selects every
archived start. It is read at import, so the process that launches asv
has to set it.

``time_search`` is the search. Track methods read the record that call
wrote. The force count is one ASE ``calculate`` on the harness
Lennard-Jones calculator. ``setup`` raises ``NotImplementedError`` when
AGOX is not installed, and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED_BASE = "314159"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-agox"))


def _agox_ready() -> bool:
    try:
        import agox  # noqa: F401
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
    return "Agox" + "".join(part.capitalize() for part in slug.split("-"))


def _start_count(problem_id: str) -> int:
    from optbench.campaign import resolve_problem
    from optbench.runners.agox_go import _load_geometries

    problem = resolve_problem(ROOT, problem_id)
    stated = (problem.get("sample") or {}).get("starts")
    if stated is not None:
        return int(stated)
    loaded = _load_geometries(ROOT, problem, None, Path("/tmp/optbench-asv-agox-geo"))
    return len(loaded)


def _geometries(problem_id: str):
    if problem_id not in _GEOMETRIES:
        from optbench.campaign import resolve_problem
        from optbench.runners.agox_go import _load_geometries

        problem = resolve_problem(ROOT, problem_id)
        _GEOMETRIES[problem_id] = _load_geometries(
            ROOT, problem, None, Path("/tmp/optbench-asv-agox-geo")
        )
    return _GEOMETRIES[problem_id]


_GEOMETRIES: dict[str, list] = {}


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{int(start)}.json"


def _specs() -> list[dict]:
    from optbench.campaign import _check_stop, list_templates, resolve_problem

    found = []
    counts: dict[str, int] = {}
    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    for template in list_templates(ROOT):
        if template.get("engine") != "agox":
            continue
        problem_id = str(template["problem"])
        if problem_id not in counts:
            counts[problem_id] = _start_count(problem_id) if full else 1
        problem = resolve_problem(ROOT, problem_id)
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "cap": int(_check_stop(problem, template)),
                "n": counts[problem_id],
            }
        )
    return found


def _job(spec: dict, seed: str, start: str) -> dict:
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.agox_go import _section, reference_energy, target_energy

    if str(seed) != SEED_BASE:
        raise ValueError(f"seed {seed} is not the campaign seed base {SEED_BASE}")
    index = int(start)
    geometries = _geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords = geometries[index]
    template = load_template(ROOT, spec["template_id"])
    settings = _section(template)
    problem = resolve_problem(ROOT, spec["problem_id"])
    reference, _source = reference_energy(ROOT, problem, int(len(coords)))
    target = target_energy(problem, reference)
    return {
        "index": index,
        "start": name,
        "coords": coords.tolist(),
        "cap": int(spec["cap"]),
        "seed": int(seed) + index,
        "method": str(settings["method"]),
        "settings": {key: value for key, value in settings.items() if key != "method"},
        "target": target,
        "reference": reference,
        "work": str(_cache_dir() / "work" / spec["template_id"].replace("/", "-") / f"{index:04d}-{name}"),
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    from optbench.runners.agox_go import search_job

    record = search_job(_job(spec, seed, start))
    record["seed"] = f"{seed}+{int(start)}"
    path = _cache_path(spec["template_id"], seed, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], seed, start)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _AgoxSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed search."""

    def setup(self, seed, start):
        if not _agox_ready():
            raise NotImplementedError("agox is not installed in this environment")
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


_AgoxSearch.track_force_calls.unit = "force calls"
_AgoxSearch.track_success.unit = "fraction"
_AgoxSearch.track_failure_cap.unit = "count"
_AgoxSearch.track_failure_wrong_stationary.unit = "count"
_AgoxSearch.track_failure_divergence.unit = "count"
_AgoxSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])

    class _Benchmark(_AgoxSearch):
        param_names = ["seed", "start"]
        params = [[SEED_BASE], starts]
        timeout = 7200
        cap = int(bound["cap"])
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
