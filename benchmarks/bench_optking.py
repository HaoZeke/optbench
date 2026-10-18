"""OptKing transition-state searches, one class per template.

``param_names`` are the seed and the start name. Seed ``0`` is the
archived geometry. OptKing does not draw a random number for these
steps, so the seed is not passed into the optimizer.

The default grid is that seed and the first geometry, which keeps
``asv check`` on one parameter. ``OPTBENCH_ASV_FULL=1`` selects every
archived start. It is read at import, so the process that launches asv
has to set it.

``time_search`` is the search. Track methods read the record that call
wrote. The force count is a ``HarnessCounter`` entry. The initial
Hessian displacements are included. ``setup`` raises
``NotImplementedError`` when OptKing is not installed, and asv-runner
stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"
PIN = "767e2c0c13ff2b851ac82101ccd5aab9a558a4f1"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-optking"))


def _optking_ready() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        text = version("optking")
    except PackageNotFoundError:
        return False
    return PIN[:8] in text or text.endswith("g767e2c0c1")


def ensure_optking() -> None:
    """Install the pinned OptKing git revision when it is absent."""
    if _optking_ready():
        return
    uv = shutil.which("uv")
    if uv is None:
        candidate = Path.home() / ".local" / "bin" / "uv"
        if candidate.is_file():
            uv = str(candidate)
    if uv is None:
        raise NotImplementedError("optking is not installed and uv is not available")
    lock = Path("/tmp/optbench-optking-install.lock")
    import fcntl

    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if _optking_ready():
            return
        subprocess.check_call(
            [
                uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                f"optking @ git+https://github.com/psi-rking/optking.git@{PIN}",
            ]
        )
    if not _optking_ready():
        raise NotImplementedError("optking pin 767e2c0c did not install")


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Optking" + "".join(part.capitalize() for part in slug.split("-"))


def _first_start(problem_id: str) -> str:
    if problem_id == "tsopt/LJ38":
        return "0000"
    if problem_id == "tsopt/pt-island":
        return "initial_0"
    raise ValueError(problem_id)


def _start_names(problem_id: str) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") != "1":
        return [_first_start(problem_id)]
    from optbench.campaign import resolve_problem
    from optbench.runners.optking_ts import load_jobs

    problem = resolve_problem(ROOT, problem_id)
    template = next(row for row in _template_rows() if row["problem"] == problem_id)
    work = Path("/tmp/optbench-asv-optking-geo")
    jobs = load_jobs(ROOT, problem, template, geometries=None, work=work, output=work / "unused.dat")
    return [str(job["start"]) for job in jobs]


def _template_rows() -> list[dict]:
    from optbench.campaign import list_templates

    return [row for row in list_templates(ROOT) if row.get("engine") == "optking"]


def _specs() -> list[dict]:
    from optbench.campaign import resolve_problem

    found = []
    names: dict[str, list[str]] = {}
    for template in _template_rows():
        problem_id = str(template["problem"])
        if problem_id not in names:
            names[problem_id] = _start_names(problem_id)
        problem = resolve_problem(ROOT, problem_id)
        convergence = problem.get("convergence") or {}
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "cap": int(convergence["max_force_calls"]),
                "starts": names[problem_id],
            }
        )
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    ensure_optking()
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.optking_ts import load_jobs, search_start

    template = load_template(ROOT, spec["template_id"])
    problem = resolve_problem(ROOT, spec["problem_id"])
    work = Path("/tmp/optbench-asv-optking-geo")
    jobs = load_jobs(ROOT, problem, template, geometries=None, work=work, output=work / "unused.dat")
    matched = [job for job in jobs if job["start"] == str(start)]
    if len(matched) != 1:
        raise IndexError(f"{spec['template_id']} has no start {start}")
    job = dict(matched[0])
    job["scratch"] = str(_cache_dir() / "scratch" / spec["template_id"].replace("/", "-") / str(start))
    record = search_start(job)
    record["seed"] = SEED
    path = _cache_path(spec["template_id"], SEED, str(start))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], SEED, str(start))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _OptkingSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed search."""

    def setup(self, seed, start):
        if not _optking_ready():
            raise NotImplementedError("optking is not installed in this environment")
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


_OptkingSearch.track_force_calls.unit = "force calls"
_OptkingSearch.track_success.unit = "fraction"
_OptkingSearch.track_failure_cap.unit = "count"
_OptkingSearch.track_failure_wrong_stationary.unit = "count"
_OptkingSearch.track_failure_divergence.unit = "count"
_OptkingSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = list(spec["starts"])

    class _Benchmark(_OptkingSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 14400
        cap = int(bound["cap"])
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
