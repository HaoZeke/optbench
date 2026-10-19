"""GMIN basin-hopping, one class per template.

``param_names`` are the seed and the start index. Seed ``0`` selects the
archived geometry. The GMIN random seed is the template base plus that
index.

The grid is seed ``0`` and archived start ``0``.

``setup`` raises ``NotImplementedError`` when ``OPTBENCH_GMIN`` is unset
and ``GMIN`` is not on ``PATH``. asv-runner stores that parameter as NaN.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"


def _binary_ready() -> bool:
    if os.environ.get("OPTBENCH_GMIN"):
        return Path(os.environ["OPTBENCH_GMIN"]).is_file()
    return shutil.which("GMIN") is not None


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Gmin" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "gmin":
            continue
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": str(template["problem"]),
                "cap": int(template["failure_cap"]),
                "n": 1,
            }
        )
    return found


class _GminSearch(TrackedSearch):
    def setup(self, seed, start):
        if not _binary_ready():
            raise NotImplementedError("GMIN binary is not available")
        if str(seed) != SEED:
            raise ValueError(f"seed {seed} is not {SEED}")
        self.record = None

    def time_search(self, seed, start):
        from optbench.campaign import load_template, resolve_problem
        from optbench.runners.gmin_go import run_template

        output = Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-gmin"))
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"{self.spec['template_id'].replace('/', '-')}-{start}.dat"
        if str(start) != "0":
            raise ValueError("this class measures archived start 0")
        run_template(
            ROOT,
            resolve_problem(ROOT, self.spec["problem_id"]),
            load_template(ROOT, self.spec["template_id"]),
            output=target,
            workers=1,
            work=output / "work" / self.spec["template_id"].replace("/", "-"),
            limit=1,
            resume=True,
        )
        line = target.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.record = json.loads(line)

    def track_force_calls(self, seed, start) -> float:
        return float(self.record["force_calls"]) if self.record else float("nan")

    def track_success(self, seed, start) -> float:
        return 1.0 if self.record and self.record["success"] else 0.0


_GminSearch.track_force_calls.unit = "force calls"
_GminSearch.track_success.unit = "fraction"


def _benchmark(spec: dict):
    # The class body treats any assigned name as local, so the outer spec
    # is copied into names that the class only reads.
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_GminSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = limit
        spec = bound

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
