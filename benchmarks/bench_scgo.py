"""SCGO basin hopping on Lennard-Jones 38.

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
from pathlib import Path

from optbench.campaign import load_template, resolve_problem

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ID = "scgo/lj38-bh"
PROBLEM_ID = "globalopt/lj38"
_TEMPLATE = load_template(ROOT, TEMPLATE_ID)
_PROBLEM = resolve_problem(ROOT, PROBLEM_ID)
SEED = str(int(_TEMPLATE["scgo"]["seed"]))
CAP = int(_TEMPLATE["failure_cap"])
START_COUNT = int(_PROBLEM["sample"]["starts"])


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-scgo"))


def selected_starts() -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(START_COUNT)]
    return ["0"]


def _geometries():
    global _LOADED
    if _LOADED is None:
        from optbench.runners.scgo_go import load_geometries

        work = Path("/tmp/optbench-asv-geo")
        _LOADED = load_geometries(ROOT, _PROBLEM, None, work, work / "scgo.dat")
    return _LOADED


_LOADED = None


def _cache_path(seed: str, start: str) -> Path:
    return _cache_dir() / TEMPLATE_ID.replace("/", "-") / f"{seed}-{int(start)}.json"


def _compute(seed: str, start: str) -> dict:
    """Run this start and store the record. The caller is the timed sample."""
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the template seed {SEED}")
    from optbench.runners.scgo_go import _one_start, bh_settings, make_job

    index = int(start)
    geometries = _geometries()
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{TEMPLATE_ID} start {start} is outside the archive")
    name, coords, _box = geometries[index]
    settings = bh_settings(_PROBLEM, _TEMPLATE, len(coords) // 3)
    if int(settings["cap"]) != CAP:
        raise RuntimeError(f"template cap {CAP} is not the budget {settings['cap']}")
    record = _one_start(
        make_job(
            _PROBLEM,
            _TEMPLATE,
            settings,
            index=index,
            name=name,
            coords=coords,
            work=str(Path("/tmp/optbench-asv-scgo-work") / f"{index:04d}-{name}"),
            seed=int(seed),
        )
    )
    record["seed"] = SEED
    path = _cache_path(SEED, str(index))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(seed: str, start: str) -> dict:
    path = _cache_path(SEED, str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(seed, start)


class ScgoLj38Bh:
    """One basin-hopping search from an archived Lennard-Jones 38 geometry."""

    template_id = TEMPLATE_ID
    problem_id = PROBLEM_ID
    cap = CAP
    param_names = ["seed", "start"]
    params = [[SEED], selected_starts()]
    repeat = 1
    rounds = 1
    number = 1
    min_run_count = 1
    sample_time = 0
    warmup_time = 0
    timeout = 600

    def setup(self, seed, start):
        try:
            import scgo  # noqa: F401
        except ImportError as exc:
            raise NotImplementedError("scgo is not installed in this environment") from exc
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _load(seed, start)
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


ScgoLj38Bh.track_force_calls.unit = "force calls"
ScgoLj38Bh.track_success.unit = "fraction"
ScgoLj38Bh.track_failure_cap.unit = "count"
ScgoLj38Bh.track_failure_wrong_stationary.unit = "count"
ScgoLj38Bh.track_failure_divergence.unit = "count"
ScgoLj38Bh.track_failure_crash.unit = "count"
