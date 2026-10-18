"""n2p2 Kalman training, one class per template.

The default grid is seed 20261015 and start 0. ``OPTBENCH_ASV_FULL=1``
selects every initialization. The seed is the base; start ``k`` draws
``PCG64(20261015 + k)``.

``setup`` raises ``NotImplementedError`` when n2p2 or ASE is absent, and
asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "20261015"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-n2p2"))


def _ready() -> bool:
    try:
        import ase  # noqa: F401
    except ImportError:
        return False
    source = Path(os.environ.get("N2P2_SRC", "/tmp/n2p2"))
    return (source / "src" / "libnnptrain" / "KalmanFilter.cpp").is_file()


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "N2p2" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "n2p2":
            continue
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": str(template["problem"]),
                "cap": int(template["stop"]["max_force_calls"]),
                "tol": float(template["stop"]["grad_abs"]),
                "n": int(template.get("starts") or 50),
                "tuning": str(template.get("tuning") or "default"),
            }
        )
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not {SEED}")
    from optbench.n2p2_ml import CODE, CODE_VERSION, N2P2_COMMIT, PYAMFF_COMMIT, cache_dataset, dataset_from_cache, run_start
    from optbench.runners.n2p2_ml import _cache_path as _dataset_cache

    dataset_path = cache_dataset(ROOT, spec["problem_id"], _dataset_cache(Path("/tmp/optbench-n2p2"), spec["problem_id"]))
    dataset = dataset_from_cache(dataset_path)
    record = run_start(dataset, int(start), cap=spec["cap"], tol=spec["tol"], shuffle=True)
    record.update(
        {
            "schema": "optbench.start/v2",
            "counter": "harness",
            "comparable": True,
            "code": CODE,
            "code_version": CODE_VERSION,
            "commit": N2P2_COMMIT,
            "template": spec["template_id"],
            "problem": spec["problem_id"],
            "tuning": spec["tuning"],
            "cap": spec["cap"],
            "pyamff_commit": PYAMFF_COMMIT,
            "seed_param": SEED,
        }
    )
    path = _cache_path(spec["template_id"], SEED, str(int(start)))
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


class _N2p2Search(_TrackedSearch):
    def setup(self, seed, start):
        if not _ready():
            raise NotImplementedError("n2p2 source or ASE is not available")
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


_N2p2Search.track_force_calls.unit = "force calls"
_N2p2Search.track_success.unit = "fraction"
_N2p2Search.track_failure_cap.unit = "count"
_N2p2Search.track_failure_wrong_stationary.unit = "count"
_N2p2Search.track_failure_divergence.unit = "count"
_N2p2Search.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_N2p2Search):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = limit
        spec = bound
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
