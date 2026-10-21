"""geomeTRIC 1.1.1 transition-state searches.

``param_names`` is ``["seed", "start"]``. The default grid is one seed and
the first geometry. ``OPTBENCH_ASV_FULL=1`` selects every start, and it is
read at import, so it has to be set in the process that launches asv.
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
_GEOMETRIC = "geometric @ git+https://github.com/leeping/geomeTRIC.git@f657e69bae9c81bf7a23e7375db98502ca5c8c30"


def _class_name(spec: dict) -> str:
    rest = spec["id"].split("/", 1)[1]
    parts = [part.capitalize() for part in rest.replace("-", " ").split()]
    return "Geometric" + "".join(parts)


def _axes(spec: dict) -> tuple[list[str], list[str]]:
    from optbench.ase_sella import discover, materialize

    material = materialize(ROOT, spec, spec, Path("/tmp/optbench-geometry"))
    jobs = discover(material, spec)
    if not jobs:
        raise RuntimeError(f"{spec['id']} has no starts")
    starts = [name for name, _paths, _seed in jobs]
    seeds = ["0"]
    if not _FULL:
        return [seeds[0]], [starts[0]]
    return seeds, starts


def _timeout(spec: dict) -> int:
    if "pt-island" in spec["id"]:
        return 1200
    return 600


def ensure_campaign() -> None:
    """Install ASE and geomeTRIC into this interpreter when they are absent."""
    try:
        import ase  # noqa: F401
        import geometric  # noqa: F401
        import numpy  # noqa: F401

        return
    except ImportError:
        pass
    uv = shutil.which("uv")
    if uv is None:
        candidate = Path.home() / ".local" / "bin" / "uv"
        if candidate.is_file():
            uv = str(candidate)
    if uv is None:
        raise RuntimeError("uv is required to install geomeTRIC into the ASV environment")
    lock = Path("/tmp/optbench-campaign-install.lock")
    import fcntl

    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            import ase  # noqa: F401
            import geometric  # noqa: F401
            import numpy  # noqa: F401

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
                    "numpy",
                    "scipy",
                    _GEOMETRIC,
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
            from optbench.geometric_ts import run_point

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
    import tomllib

    specs = []
    folder = ROOT / "templates" / "geometric"
    for path in sorted(folder.glob("*.toml")):
        spec = tomllib.loads(path.read_text(encoding="utf-8"))
        if spec.get("engine") == "geometric" and spec.get("ranked", True):
            specs.append(spec)
    for spec in specs:
        benchmark = _bind(spec)
        globals()[benchmark.__name__] = benchmark


_register()
