"""PANNA Adam on the mlopt problems, one class per template.

The default grid is seed ``20261015`` and start ``0``.
``OPTBENCH_ASV_FULL=1`` selects all 50 initializations. The campaign
writer stores the full grid. Importing this module does not import
TensorFlow or JAX.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = "20261015"


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Panna" + "".join(part.capitalize() for part in slug.split("-"))


def selected_indices() -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(50)]
    return ["0"]


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "panna":
            continue
        found.append(
            {
                "class_name": _class_name(template["id"]),
                "template_id": template["id"],
                "problem_id": template["problem"],
                "cap": int(template["stop"]["max_force_calls"]),
            }
        )
    return found


def _ready() -> bool:
    try:
        import jax  # noqa: F401
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


class _PannaSearch:
    repeat = 1
    rounds = 1
    number = 1
    min_run_count = 1
    sample_time = 0
    warmup_time = 0
    timeout = 7200

    def setup(self, seed, start):
        if not _ready():
            raise NotImplementedError("PANNA trainers are not installed in this environment")
        if str(seed) != SEED:
            raise ValueError(f"seed {seed} is not {SEED}")
        self.record = None

    def measure(self, seed, start):
        if str(seed) != SEED:
            raise ValueError(f"seed {seed} is not {SEED}")
        raise NotImplementedError(
            f"{self.template_id} start {start} is recorded by python -m optbench.panna_train"
        )

    def time_search(self, seed, start):
        self.measure(seed, start)

    def track_force_calls(self, seed, start) -> float:
        return float(self.record["force_calls"])

    def track_success(self, seed, start) -> float:
        return 1.0 if self.record["success"] else 0.0

    def track_failure_cap(self, seed, start) -> float:
        return 1.0 if self.record["failure"] == "cap" else 0.0

    def track_failure_wrong_stationary(self, seed, start) -> float:
        return 1.0 if self.record["failure"] == "wrong_stationary" else 0.0

    def track_failure_divergence(self, seed, start) -> float:
        return 1.0 if self.record["failure"] == "divergence" else 0.0

    def track_failure_crash(self, seed, start) -> float:
        return 1.0 if self.record["failure"] == "crash" else 0.0


_PannaSearch.track_force_calls.unit = "force calls"
_PannaSearch.track_success.unit = "fraction"
_PannaSearch.track_failure_cap.unit = "count"
_PannaSearch.track_failure_wrong_stationary.unit = "count"
_PannaSearch.track_failure_divergence.unit = "count"
_PannaSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec

    class _Benchmark(_PannaSearch):
        param_names = ["seed", "start"]
        params = [[SEED], selected_indices()]
        cap = bound["cap"]
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
