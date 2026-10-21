"""pele searches, one class per template.

Minimization templates use ``pele.optimize``. Basin-hopping templates use
``pele.basinhopping.BasinHopping`` with the ``lbfgs_cpp`` quench.

``param_names`` are the seed and the start index. Seed ``0`` selects the
archived start. A basin-hopping generator seed is the integer in that
start name. The pele minimizers take no seed.

The default grid is seed ``0`` and start index ``0``. That keeps
``asv check`` on one parameter. ``OPTBENCH_ASV_FULL=1`` selects every
archived start. It is read at import, so the process that launches asv
has to set it.

``time_search`` is the search. Track methods read the record that call
wrote. The force count is a ``HarnessCounter`` entry through
``getEnergyGradient``. ``setup`` raises ``NotImplementedError`` when
pele is not installed. The ranked row is the campaign file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-pele"))


def _pele_ready() -> bool:
    try:
        import pele  # noqa: F401
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
    return "Pele" + "".join(part.capitalize() for part in slug.split("-"))


def _basin_hopping(template: dict) -> bool:
    return str(template.get("algorithm") or "") == "basin hopping"


def _cache_path(template_id: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{SEED}-{int(start)}.json"


def _archive_count(problem_id: str) -> int:
    import tarfile

    from optbench.campaign import geometry_archive, resolve_problem

    archive = geometry_archive(ROOT, resolve_problem(ROOT, problem_id))
    with tarfile.open(archive) as packed:
        return sum(1 for name in packed.getnames() if name.endswith(".con"))


_MIN_GEOMETRIES: dict[str, list] = {}
_BH_GEOMETRIES: dict[str, list] = {}


def _min_geometries(problem_id: str):
    if problem_id not in _MIN_GEOMETRIES:
        from optbench.campaign import resolve_problem
        from optbench.runners.pele_min import _load_geometries

        problem = resolve_problem(ROOT, problem_id)
        work = Path("/tmp/optbench-asv-geo")
        _MIN_GEOMETRIES[problem_id] = _load_geometries(
            ROOT, problem, None, work, work / "unused.dat"
        )
    return _MIN_GEOMETRIES[problem_id]


def _bh_geometries(problem_id: str):
    if problem_id not in _BH_GEOMETRIES:
        from optbench.campaign import list_templates, resolve_problem
        from optbench.runners.pele_bh import _load_geometries

        problem = resolve_problem(ROOT, problem_id)
        template = next(
            row
            for row in list_templates(ROOT)
            if row.get("engine") == "pele"
            and _basin_hopping(row)
            and row.get("problem") == problem_id
        )
        work = Path("/tmp/optbench-asv-geo")
        _BH_GEOMETRIES[problem_id] = _load_geometries(
            ROOT, problem, template, None, work, work / "unused.dat"
        )
    return _BH_GEOMETRIES[problem_id]


def _min_specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem
    from optbench.runners.pele_min import _kind, _section

    found = []
    counts: dict[str, int] = {}
    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    for template in list_templates(ROOT):
        if template.get("engine") != "pele" or _basin_hopping(template):
            continue
        problem_id = str(template["problem"])
        if problem_id not in counts:
            counts[problem_id] = _archive_count(problem_id) if full else 1
        problem = resolve_problem(ROOT, problem_id)
        convergence = problem.get("convergence") or {}
        method, options = _section(template)
        found.append(
            {
                "runner": "min",
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "kind": _kind(problem_id),
                "method": method,
                "options": options,
                "gtol": float(convergence["force_norm"]),
                "cap": int(convergence["max_force_calls"]),
                "n": counts[problem_id],
                "timeout": 7200,
            }
        )
    return found


def _bh_specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    for template in list_templates(ROOT):
        if template.get("engine") != "pele" or not _basin_hopping(template):
            continue
        problem_id = str(template["problem"])
        found.append(
            {
                "runner": "bh",
                "class_name": _class_name(str(template["id"])),
                "template": template,
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "cap": int(template["stop"]["max_force_calls"]),
                "n": len(_bh_geometries(problem_id)) if full else 1,
                "timeout": 14400,
            }
        )
    return found


def _min_job(spec: dict, start: str) -> dict:
    index = int(start)
    geometries = _min_geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, box = geometries[index]
    kind = spec["kind"]
    return {
        "index": index,
        "start": name,
        "coords": coords.tolist(),
        "box": None if kind == "lj" or box is None else box.tolist(),
        "kind": kind,
        "method": spec["method"],
        "options": dict(spec["options"]),
        "gtol": spec["gtol"],
        "cap": spec["cap"],
    }


def _bh_job(spec: dict, start: str) -> dict:
    from optbench.campaign import resolve_problem
    from optbench.runners.pele_bh import _options, _rule

    index = int(start)
    geometries = _bh_geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, seed = geometries[index]
    template = spec["template"]
    minimum, tolerance, _partial = _rule(resolve_problem(ROOT, spec["problem_id"]), template)
    return {
        "index": index,
        "start": name,
        "seed": seed,
        "coords": coords.tolist(),
        "cap": spec["cap"],
        "energy_minimum": minimum,
        "energy_tolerance": tolerance,
        "problem": spec["problem_id"],
        "template": spec["template_id"],
        "code_version": str(template["code_version"]),
        "commit": str(template["commit"]),
        "tuning": str(template.get("tuning") or ""),
        **_options(template),
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    if spec["runner"] == "bh":
        from optbench.runners.pele_bh import _hop_job

        record = _hop_job(_bh_job(spec, start))
        record["seed_param"] = SEED
    else:
        from optbench.runners.pele_min import _minimize_job

        record = _minimize_job(_min_job(spec, start))
        record["seed"] = SEED
    path = _cache_path(spec["template_id"], start)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], start)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _PeleSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed search."""

    def setup(self, seed, start):
        if not _pele_ready():
            raise NotImplementedError("pele is not installed in this environment")
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


_PeleSearch.track_force_calls.unit = "force calls"
_PeleSearch.track_success.unit = "fraction"
_PeleSearch.track_failure_cap.unit = "count"
_PeleSearch.track_failure_wrong_stationary.unit = "count"
_PeleSearch.track_failure_divergence.unit = "count"
_PeleSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]
    template_name = spec["template_id"]
    problem_name = spec["problem_id"]
    class_name = spec["class_name"]
    budget = spec["timeout"]

    class _Benchmark(_PeleSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = budget
        cap = limit
        spec = bound
        template_id = template_name
        problem_id = problem_name

    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    return _Benchmark


for _spec in _min_specs() + _bh_specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
