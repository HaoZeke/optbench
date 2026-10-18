"""KLIFF neural-network training, one class per template.

``param_names`` are the seed base and the start index. The start seed is
``20261015`` plus that index, the same draw the campaign uses.

The default grid is start ``0``. ``OPTBENCH_ASV_FULL=1`` selects every
initialization. It is read at import, so the process that launches asv
has to set it.

``time_search`` is the training run. Track methods read the record that
call wrote. The force count is one loss and parameter-gradient evaluation
on a ``HarnessCounter``. ``setup`` raises ``NotImplementedError`` when
KLIFF is not installed, and asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch
from optbench.kliff_ml import SEED_BASE

ROOT = Path(__file__).resolve().parents[1]


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-kliff"))


def _kliff_ready() -> bool:
    try:
        from optbench.kliff_ml import _kliff_loss

        _kliff_loss()
    except Exception:
        return False
    return True


def selected_indices(count: int) -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Kliff" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "kliff":
            continue
        sample = int((template.get("kliff") or {}).get("initializations") or 50)
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "cap": int((template.get("stop") or {})["max_force_calls"]),
                "n": sample if os.environ.get("OPTBENCH_ASV_FULL") == "1" else 1,
            }
        )
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if int(seed) != SEED_BASE:
        raise ValueError(f"seed {seed} is not the campaign seed base {SEED_BASE}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.kliff_ml import _settings, train_start
    from optbench.ml_loss import cache_file, ensure_features, parse_fp

    template = load_template(ROOT, spec["template_id"])
    problem = resolve_problem(ROOT, template["problem"])
    settings = _settings(template, problem)
    spec_fp = parse_fp(ROOT / settings["fp_parameters"])
    ensure_features(ROOT, problem, spec_fp, need_jacobian=settings["force_coefficient"] != 0.0)
    import numpy as np

    stored = np.load(cache_file(problem["id"]), allow_pickle=False)
    features = {"fps": stored["fps"], "energy": stored["energy"], "symbols": stored["symbols"]}
    if settings["force_coefficient"] != 0.0:
        features["jacobian"] = stored["jacobian"]
        features["forces"] = stored["forces"]
    index = int(start)
    job = {
        "index": index,
        "seed": SEED_BASE + index,
        "method": settings["method"],
        "learning_rate": settings["learning_rate"],
        "force_coefficient": settings["force_coefficient"],
        "energy_coefficient": 1.0,
        "grad_tol": settings["grad_tol"],
        "cap": settings["cap"],
        "max_epochs": settings["max_epochs"],
        "hidden": list(settings["hidden"]),
        "elements": list(spec_fp.elements),
        "width": spec_fp.width,
    }
    record = train_start(features, job)
    path = _cache_path(spec["template_id"], str(SEED_BASE), str(index))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], str(int(seed)), str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _KliffSearch(TrackedSearch):
    """Shared track methods. ``time_search`` is the only timed training run."""

    def setup(self, seed, start):
        if not _kliff_ready():
            raise NotImplementedError("KLIFF is not installed in this environment")
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


_KliffSearch.track_force_calls.unit = "force calls"
_KliffSearch.track_success.unit = "fraction"
_KliffSearch.track_failure_cap.unit = "count"
_KliffSearch.track_failure_wrong_stationary.unit = "count"
_KliffSearch.track_failure_divergence.unit = "count"
_KliffSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec

    class _Benchmark(_KliffSearch):
        param_names = ["seed", "start"]
        params = [[str(SEED_BASE)], selected_indices(spec["n"])]
        timeout = 14400
        cap = spec["cap"]
        spec = bound

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
