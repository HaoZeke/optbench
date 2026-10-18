"""ASE basin hopping and minima hopping, one class per template.

``param_names`` is ``["seed", "start"]``. The default grid is the template
seed and the first geometry. ``OPTBENCH_ASV_FULL=1`` selects every start.
It is read at import.

``measure`` returns the harness record. A one-start result is not the
published mean.
"""

from __future__ import annotations

import os
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
_FULL = os.environ.get("OPTBENCH_ASV_FULL") == "1"


def _class_name(spec: dict) -> str:
    rest = spec["id"].split("/", 1)[1]
    parts = [part.capitalize() for part in rest.replace("-", " ").split()]
    return "Asebh" + "".join(parts)


def _starts(spec: dict) -> list[str]:
    from optbench.ase_global import _archive_file, list_start_names
    from optbench.problems import load_problem

    problem = load_problem(ROOT / "problems" / f"{spec['problem']}.toml")
    return list_start_names(_archive_file(ROOT, problem, spec))


def _bind(spec: dict):
    starts = _starts(spec)
    if not _FULL:
        starts = starts[:1]
    seed = str(int(spec.get("seed", 0)))
    template_id = spec["id"]
    cap = int(spec["stop"]["max_force_calls"])

    class _Benchmark(TrackedSearch):
        param_names = ["seed", "start"]
        params = [[seed], starts]
        timeout = 7200

        def measure(self, seed, start):
            from optbench.ase_global import run_point

            record = run_point(ROOT, template_id, start, int(seed))
            return {
                "force_calls": float(record["force_calls"]),
                "success": bool(record["success"]),
                "failure": record.get("failure"),
            }

    _Benchmark.__name__ = _class_name(spec)
    _Benchmark.__qualname__ = _Benchmark.__name__
    _Benchmark.template_id = template_id
    _Benchmark.cap = cap
    return _Benchmark


def _register() -> None:
    import tomllib

    directory = ROOT / "templates" / "asebh"
    for path in sorted(directory.glob("*.toml")):
        spec = tomllib.loads(path.read_text(encoding="utf-8"))
        benchmark = _bind(spec)
        globals()[benchmark.__name__] = benchmark


_register()
