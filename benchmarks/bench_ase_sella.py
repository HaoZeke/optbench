"""ASE 3.29.0 and Sella 2.6.0 searches, one ASV parameter pair per start.

``param_names`` is ``["seed", "start"]``. The default grid is one seed and
the first geometry. ``OPTBENCH_ASV_FULL=1`` selects every start, and it is
read at import, so it has to be set in the process that launches asv.

``measure`` returns the harness record. ``TrackedSearch`` publishes that
record as ``track_*`` and times a second call as ``time_search``. This
module is only the ASE and Sella suite.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from optbench.asv_tracks import TrackedSearch as _TrackedSearch
from optbench.failures import failure_of

ROOT = Path(__file__).resolve().parents[1]
_FULL = os.environ.get("OPTBENCH_ASV_FULL") == "1"


def _class_name(spec: dict) -> str:
    engine = "Ase" if spec["engine"] == "ase" else "Sella"
    rest = spec["id"].split("/", 1)[1]
    parts = [part.capitalize() for part in rest.replace("-", " ").split()]
    return engine + "".join(parts)


def _axes(spec: dict) -> tuple[list[str], list[str]]:
    from optbench.ase_sella import discover, materialize

    material = materialize(ROOT, spec, spec, Path("/tmp/optbench-geometry"))
    jobs = discover(material, spec)
    if not jobs:
        raise RuntimeError(f"{spec['id']} has no starts")
    if spec["task"] in {"dimer", "sella-search"}:
        starts = [jobs[0][1][0].name]
        seeds = [str(0 if seed is None else seed) for _name, _paths, seed in jobs]
    else:
        starts = [name for name, _paths, _seed in jobs]
        seeds = ["0"]
    if not _FULL:
        return [seeds[0]], [starts[0]]
    return seeds, starts


def _timeout(spec: dict) -> int:
    if spec["task"] in {"neb", "sella-neb"}:
        return 1800
    if spec["task"] == "sella-search":
        # Two searches (setup and the timed call) on the Pt island do not
        # fit in 900s. The class timeout has to cover both.
        return 3600
    if spec["task"] == "dimer":
        return 900
    return 600


def ensure_campaign() -> None:
    """Install ASE and Sella into this interpreter when they are absent."""
    try:
        import ase  # noqa: F401
        import numpy  # noqa: F401
        import sella  # noqa: F401

        return
    except ImportError:
        pass
    uv = shutil.which("uv")
    if uv is None:
        candidate = Path.home() / ".local" / "bin" / "uv"
        if candidate.is_file():
            uv = str(candidate)
    if uv is None:
        raise RuntimeError("uv is required to install ASE and Sella into the ASV environment")
    lock = Path("/tmp/optbench-campaign-install.lock")
    import fcntl

    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            import ase  # noqa: F401
            import numpy  # noqa: F401
            import sella  # noqa: F401

            return
        except ImportError:
            subprocess.check_call(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "ase==3.29.0",
                    "sella==2.6.0",
                    "numpy",
                    "scipy",
                ]
            )


def _bind(spec: dict):
    seeds, starts = _axes(spec)
    template_id = spec["id"]
    cap = int(spec["stop"]["max_force_calls"])

    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [seeds, starts]
        timeout = _timeout(spec)

        def measure(self, seed, start):
            ensure_campaign()
            from optbench.ase_sella import run_point

            record = run_point(ROOT, template_id, start, int(seed))
            calls = float(record["force_calls"])
            success = bool(record["success"]) and calls < cap
            reasons = [str(reason) for reason in record.get("reasons") or []]
            if "error" in reasons and "crash" not in reasons:
                reasons = ["crash", *reasons]
            if not success and not reasons:
                reasons = ["cap"] if calls >= cap else ["divergence"]
            judged = {"force_calls": calls, "success": success, "reasons": reasons}
            return {
                "force_calls": calls,
                "success": success,
                "failure": failure_of(judged, cap),
            }

    _Benchmark.__name__ = _class_name(spec)
    _Benchmark.__qualname__ = _Benchmark.__name__
    _Benchmark.template_id = template_id
    _Benchmark.cap = cap
    return _Benchmark


def _register() -> None:
    from optbench.ase_catalog import all_specs

    for spec in all_specs():
        benchmark = _bind(spec)
        globals()[benchmark.__name__] = benchmark


_register()
