"""GASP global optimization, one class per template.

The default grid is seed ``0`` and start ``0``. ``OPTBENCH_ASV_FULL=1``
selects every archived start. ``setup`` raises ``NotImplementedError``
when the pinned GASP tree or the LAMMPS binary is absent, and asv-runner
stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _ready() -> bool:
    try:
        from optbench.runners.gasp_global import _find_gasp, _find_lammps
    except Exception:
        return False
    try:
        _find_lammps()
        _find_gasp({"commit": "0c8d993c82e0e1c69a05b3c34bbb2fcbbdbb7f07"})
    except Exception:
        return False
    return True


def _specs() -> list[dict]:
    from optbench.campaign import list_templates
    from optbench.runners.gasp_global import search_force_cap

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "gasp":
            continue
        section = template.get("gasp") or {}
        if not section.get("energy_minimum"):
            continue
        problem_id = str(template["problem"])
        problem_path = ROOT / "problems" / f"{problem_id}.toml"
        text = problem_path.read_text(encoding="utf-8")
        starts = 1
        for line in text.splitlines():
            if line.startswith("starts = "):
                starts = int(line.split("=", 1)[1].strip())
        if os.environ.get("OPTBENCH_ASV_FULL") != "1":
            starts = 1
        cap = search_force_cap(int(section["num_energy_calcs"]), int(section["min_eval"]))
        slug = str(template["id"]).split("/", 1)[1]
        found.append(
            {
                "class_name": "Gasp" + "".join(part.capitalize() for part in slug.split("-")),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "cap": cap,
                "n": starts,
            }
        )
    return found


def _cache(spec: dict, start: str) -> Path:
    root = Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-gasp"))
    return root / spec["template_id"].replace("/", "-") / f"{SEED}-{int(start)}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not {SEED}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.gasp_global import _jobs, search_job

    template = load_template(ROOT, spec["template_id"])
    problem = resolve_problem(ROOT, spec["problem_id"])
    work = Path("/tmp/optbench-asv-gasp-work")
    jobs = _jobs(ROOT, problem, template, work, limit=0)
    index = int(start)
    if index < 0 or index >= len(jobs):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    record = search_job(jobs[index])
    record["seed"] = SEED
    path = _cache(spec, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


class _GaspSearch(TrackedSearch):
    timeout = 7200

    def setup(self, seed, start):
        if not _ready():
            raise NotImplementedError("GASP or LAMMPS is not installed in this environment")
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.spec, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            path = _cache(self.spec, start)
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
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


_GaspSearch.track_force_calls.unit = "force calls"
_GaspSearch.track_success.unit = "fraction"
_GaspSearch.track_failure_cap.unit = "count"
_GaspSearch.track_failure_wrong_stationary.unit = "count"
_GaspSearch.track_failure_divergence.unit = "count"
_GaspSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec

    class _Benchmark(_GaspSearch):
        param_names = ["seed", "start"]
        params = [[SEED], [str(index) for index in range(spec["n"])]]
        timeout = 7200
        cap = spec["cap"]
        spec = bound
        template_id = spec["template_id"]
        problem_id = spec["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
