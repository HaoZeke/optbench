"""ASE 3.29.0 and Sella 2.6.0 benchmarks, one ASV parameter pair per start.

``start`` is the geometry name. ``seed`` is the search seed. A minimizer
uses seed ``0``. A dimer or single-ended Sella search varies ``seed`` and
keeps one geometry in ``start``.

``time_optimize`` is the wall time of that start, counted by ASV around the
harness run. The ``track_*`` methods report the harness force calls, success,
and the failure causes. They read the record ``time_optimize`` just wrote, and
they run the start themselves when that record is missing.

This module is the ASE and Sella suite only. Shared ASV configuration stays
in ``asv.conf.json``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = "1"
CACHE_ROOT = Path("/tmp/optbench-asv-cache")


def _class_name(spec: dict) -> str:
    engine = "Ase" if spec["engine"] == "ase" else "Sella"
    rest = spec["id"].split("/", 1)[1]
    parts = [part.capitalize() for part in rest.replace("-", " ").split()]
    return engine + "".join(parts)


def _grid(spec: dict) -> tuple[list[str], list[str]]:
    from optbench.ase_sella import discover, materialize

    material = materialize(ROOT, spec, spec, Path("/tmp/optbench-geometry"))
    jobs = discover(material, spec)
    if not jobs:
        raise RuntimeError(f"{spec['id']} has no starts")
    if spec["task"] in {"dimer", "sella-search"}:
        starts = [jobs[0][1][0].name]
        seeds = [str(0 if seed is None else seed) for _name, _paths, seed in jobs]
        return starts, seeds
    return [name for name, _paths, _seed in jobs], ["0"]


def _stamp(spec: dict) -> str:
    payload = {
        "version": CACHE_VERSION,
        "id": spec["id"],
        "optimizer": spec.get("optimizer"),
        "neb": spec.get("neb"),
        "sella": spec.get("sella"),
        "sella_search": spec.get("sella_search"),
        "stop": spec.get("stop"),
        "potential": spec.get("potential"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _cache_file(spec: dict, start: str, seed: str) -> Path:
    return CACHE_ROOT / _stamp(spec) / f"{start}--{seed}.json"


def _write_record(spec: dict, start: str, seed: str, record: dict) -> None:
    path = _cache_file(spec, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _read_record(spec: dict, start: str, seed: str) -> dict | None:
    path = _cache_file(spec, start, seed)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_campaign() -> None:
    """Install ASE, Sella, and NumPy into this interpreter when they are absent.

    The ASV environment installs the OptBench package. The campaign pins are
    not part of that package, so the first benchmark call installs them.
    """
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
    lock.parent.mkdir(parents=True, exist_ok=True)
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


def _cause(record: dict, name: str) -> float:
    reasons = set(record.get("reasons") or [])
    return 1.0 if name in reasons else 0.0


def _bind(spec: dict):
    starts, seeds = _grid(spec)
    template_id = spec["id"]

    class _Benchmark:
        param_names = ["start", "seed"]
        params = [starts, seeds]
        timeout = 3600
        repeat = 1
        rounds = 1
        number = 1
        warmup_time = 0
        min_run_count = 1

        def setup(self, start: str, seed: str) -> None:
            ensure_campaign()
            self.start = start
            self.seed = seed

        def _compute(self, start: str, seed: str) -> dict:
            from optbench.ase_sella import run_point

            record = run_point(ROOT, template_id, start, int(seed))
            _write_record(spec, start, seed, record)
            return record

        def time_optimize(self, start: str, seed: str) -> None:
            self._compute(start, seed)

        def _load(self, start: str, seed: str) -> dict:
            found = _read_record(spec, start, seed)
            if found is None:
                found = self._compute(start, seed)
            return found

        def track_force_calls(self, start: str, seed: str) -> float:
            return float(self._load(start, seed)["force_calls"])

        def track_success(self, start: str, seed: str) -> float:
            return 1.0 if self._load(start, seed)["success"] else 0.0

        def track_failure_cap(self, start: str, seed: str) -> float:
            return _cause(self._load(start, seed), "cap")

        def track_failure_force(self, start: str, seed: str) -> float:
            return _cause(self._load(start, seed), "force_norm")

        def track_failure_index(self, start: str, seed: str) -> float:
            return _cause(self._load(start, seed), "hessian_index")

        def track_failure_error(self, start: str, seed: str) -> float:
            return _cause(self._load(start, seed), "error")

    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "success"
    _Benchmark.track_failure_cap.unit = "failure"
    _Benchmark.track_failure_force.unit = "failure"
    _Benchmark.track_failure_index.unit = "failure"
    _Benchmark.track_failure_error.unit = "failure"
    _Benchmark.__name__ = _class_name(spec)
    _Benchmark.__qualname__ = _Benchmark.__name__
    _Benchmark.template_id = template_id
    return _Benchmark


def benchmark_classes() -> list[type]:
    from optbench.ase_catalog import all_specs

    return [_bind(spec) for spec in all_specs()]


for _benchmark in benchmark_classes():
    globals()[_benchmark.__name__] = _benchmark


def _product(params: list) -> list[tuple]:
    import itertools

    if not params:
        return [()]
    return list(itertools.product(*[list(axis) for axis in params]))


def _load_asv_results(results_dir: Path) -> dict[str, dict]:
    """Map a benchmark name to ``{"result": [...], "params": [...]}``."""
    found: dict[str, dict] = {}
    if not results_dir.is_dir():
        return found
    for path in results_dir.rglob("*.json"):
        if path.name in {"machine.json", "benchmarks.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        results = payload.get("results")
        if not isinstance(results, dict):
            continue
        for name, row in results.items():
            if not isinstance(row, list) or not row:
                continue
            result = row[0]
            params = row[1] if len(row) > 1 else None
            if not isinstance(result, list):
                continue
            found[name] = {"result": result, "params": params, "commit": payload.get("commit_hash", "")}
    return found


def _track(loaded: dict[str, dict], prefix: str, suffix: str) -> dict | None:
    for name, row in loaded.items():
        if name.endswith(prefix + "." + suffix) or name.endswith("." + prefix + "." + suffix):
            return row
    return None


def export_campaign(results_dir: Path, campaign_dir: Path) -> list[Path]:
    """Write campaign files from an ``asv run`` result tree.

    A template is exported only when ``track_force_calls`` has a finite value
    for every ``(start, seed)``. The numbers are the ASV result values.
    """
    import math

    from optbench.ase_catalog import all_specs
    from optbench.ase_sella import entry_id, write_rows

    loaded = _load_asv_results(results_dir)
    written: list[Path] = []
    for spec in all_specs():
        class_name = _class_name(spec)
        calls = _track(loaded, class_name, "track_force_calls")
        if calls is None:
            continue
        params = calls["params"] if calls.get("params") else _grid(spec)
        # ``_grid`` returns (starts, seeds). ASV stores the same pair of lists.
        if isinstance(params, tuple):
            params = list(params)
        grid = _product(params)
        values = calls["result"]
        if len(values) != len(grid) or any(value is None or (isinstance(value, float) and math.isnan(value)) for value in values):
            continue
        success = _track(loaded, class_name, "track_success")
        cap = _track(loaded, class_name, "track_failure_cap")
        force = _track(loaded, class_name, "track_failure_force")
        index = _track(loaded, class_name, "track_failure_index")
        error = _track(loaded, class_name, "track_failure_error")
        timing = _track(loaded, class_name, "time_optimize")

        def _at(row: dict | None, position: int) -> float | None:
            if row is None or position >= len(row["result"]) or row["result"][position] is None:
                return None
            return float(row["result"][position])

        rows = []
        for position, (start, seed) in enumerate(grid):
            reasons = []
            if _at(cap, position) == 1.0:
                reasons.append("cap")
            if _at(force, position) == 1.0:
                reasons.append("force_norm")
            if _at(index, position) == 1.0:
                reasons.append("hessian_index")
            if _at(error, position) == 1.0:
                reasons.append("error")
            ok = _at(success, position)
            if ok is None:
                ok = 0.0 if reasons else 1.0
            record_start = start
            if spec["task"] in {"dimer", "sella-search"}:
                record_start = f"seed-{int(seed):04d}"
            cached = _read_record(spec, str(start), str(seed))
            row = {
                "schema": "optbench.start/v2",
                "start": record_start,
                "seed": int(seed),
                "force_calls": float(values[position]),
                "counter": "harness",
                "comparable": True,
                "success": bool(ok) and not reasons,
                "reasons": reasons,
                "reference": "none",
            }
            if cached:
                for key in ("verify_calls", "force_norm", "hessian_index", "expected_index", "energy", "parameters"):
                    if key in cached:
                        row[key] = cached[key]
                if float(cached["force_calls"]) != float(values[position]):
                    row["asv_force_calls"] = float(values[position])
            wall = _at(timing, position)
            if wall is not None:
                row["wall_seconds"] = wall
            rows.append(row)
        output = campaign_dir / f"{entry_id(spec)}.dat"
        write_rows(ROOT, spec["id"], rows, output)
        written.append(output)
    return written
