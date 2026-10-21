"""SCGO basin hopping on the harness Lennard-Jones potential.

``param_names`` are the seed and the start index. The seed is the
template seed. The default grid is that seed and start ``0``, which
keeps ``asv check`` on one parameter. ``OPTBENCH_ASV_FULL=1`` selects
every archived start. It is read at import, so the process that
launches asv has to set it.

``time_search`` is the search. Track methods read the record that call
wrote. ``setup`` raises ``NotImplementedError`` when scgo is not
installed, and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

from optbench.campaign import geometry_archive, load_template, resolve_problem

ROOT = Path(__file__).resolve().parents[1]


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-scgo"))


def _start_count(problem: dict, template: dict) -> int:
    sample = (problem.get("sample") or {}).get("starts")
    if sample not in (None, "", 0):
        return int(sample)
    archive_path = str((template.get("scgo") or {}).get("archive_path") or "")
    if not archive_path:
        raise ValueError(f"{template['id']} has no start count")
    archive = geometry_archive(ROOT, {"id": problem["id"], "input": {"archive_path": archive_path}})
    with tarfile.open(archive) as packed:
        count = sum(
            1
            for name in packed.getnames()
            if Path(name).name.startswith("coords.") or name.endswith(".con")
        )
    if count <= 0:
        raise ValueError(f"{archive_path} has no start files")
    return count


def _specs() -> list[dict]:
    found = []
    for template_id, class_name in (
        ("scgo/lj38-bh", "ScgoLj38Bh"),
        ("scgo/lj75-bh", "ScgoLj75Bh"),
        ("scgo/lj98-bh", "ScgoLj98Bh"),
        ("scgo/lj150-bh", "ScgoLj150Bh"),
    ):
        template = load_template(ROOT, template_id)
        problem = resolve_problem(ROOT, str(template["problem"]))
        found.append(
            {
                "class_name": class_name,
                "template_id": template_id,
                "problem_id": str(problem["id"]),
                "template": template,
                "problem": problem,
                "seed": str(int(template["scgo"]["seed"])),
                "cap": int(template["failure_cap"]),
                "n": _start_count(problem, template),
            }
        )
    return found


_SPECS = {spec["class_name"]: spec for spec in _specs()}
_GEOMETRIES: dict[str, list] = {}


def selected_starts(count: int | None = None) -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    total = _SPECS["ScgoLj38Bh"]["n"] if count is None else count
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(total)]
    return ["0"]


def _geometries(spec: dict):
    problem_id = spec["problem_id"]
    if problem_id not in _GEOMETRIES:
        from optbench.runners.scgo_go import load_geometries

        work = Path("/tmp/optbench-asv-geo")
        archive_path = str((spec["template"].get("scgo") or {}).get("archive_path") or "") or None
        _GEOMETRIES[problem_id] = load_geometries(
            ROOT,
            spec["problem"],
            None,
            work,
            work / "scgo.dat",
            archive_path=archive_path,
        )
    return _GEOMETRIES[problem_id]


def _cache_path(spec: dict, seed: str, start: str) -> Path:
    return _cache_dir() / spec["template_id"].replace("/", "-") / f"{seed}-{int(start)}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    """Run this start and store the record. The caller is the timed sample."""
    if str(seed) != spec["seed"]:
        raise ValueError(f"seed {seed} is not the template seed {spec['seed']}")
    from optbench.runners.scgo_go import _one_start, bh_settings, make_job

    index = int(start)
    geometries = _geometries(spec)
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, _box = geometries[index]
    settings = bh_settings(spec["problem"], spec["template"], len(coords) // 3)
    if int(settings["cap"]) != spec["cap"]:
        raise RuntimeError(f"template cap {spec['cap']} is not the budget {settings['cap']}")
    record = _one_start(
        make_job(
            spec["problem"],
            spec["template"],
            settings,
            index=index,
            name=name,
            coords=coords,
            work=str(Path("/tmp/optbench-asv-scgo-work") / spec["template_id"].replace("/", "-") / f"{index:04d}-{name}"),
            seed=int(seed),
        )
    )
    record["seed"] = spec["seed"]
    path = _cache_path(spec, spec["seed"], str(index))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec, spec["seed"], str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


def _bind(spec: dict):
    bound = spec

    class _Benchmark:
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]
        cap = bound["cap"]
        param_names = ["seed", "start"]
        params = [[bound["seed"]], selected_starts(bound["n"])]
        repeat = 1
        rounds = 1
        number = 1
        min_run_count = 1
        sample_time = 0
        warmup_time = 0
        timeout = 600
        spec = bound

        def setup(self, seed, start):
            try:
                import scgo  # noqa: F401
            except ImportError as exc:
                raise NotImplementedError("scgo is not installed in this environment") from exc
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

    _Benchmark.__name__ = bound["class_name"]
    _Benchmark.__qualname__ = bound["class_name"]
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    return _Benchmark


for _spec in _SPECS.values():
    globals()[_spec["class_name"]] = _bind(_spec)

SEED = _SPECS["ScgoLj38Bh"]["seed"]
CAP = _SPECS["ScgoLj38Bh"]["cap"]
TEMPLATE_ID = _SPECS["ScgoLj38Bh"]["template_id"]
PROBLEM_ID = _SPECS["ScgoLj38Bh"]["problem_id"]
START_COUNT = _SPECS["ScgoLj38Bh"]["n"]
