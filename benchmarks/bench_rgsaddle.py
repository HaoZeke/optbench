"""rgsaddle band entries as airspeed-velocity benchmarks.

One class is one problem and one template. Parameters are the seed and
the archived start. ``time_search`` runs that start once and writes a
sidecar. The ``track_*`` methods read the sidecar, so a full ``asv run``
records force calls, success, failure causes, and wall time from the
same search.

The seed is the string ``"0"``. The band does not draw random numbers;
the parameter is there so every entry has the same seed and start axes.
Set ``OPTBENCH_RESULT_ROOT`` to the repository that should keep the
sidecars. The benchmark process itself runs with a temporary working
directory.

``setup`` raises ``NotImplementedError`` when ``librgsaddle.so`` is
absent, which asv records as a skip. This module does not configure
the shared ``asv.conf.json``.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]

PROBLEMS = (
    ("Heptamer", "neb/Pt-heptamer"),
    ("PtIsland", "double-ended-saddle-search/pt-island"),
    ("Lj38De", "double-ended-saddle-search/lj38"),
    ("Lj38TwoTs", "double-ended-saddle-search/lj38_2ts_separation"),
)

TEMPLATES = (
    ("Neb", "rgsaddle/neb"),
    ("NebLbfgs", "rgsaddle/neb-lbfgs"),
    ("CiNeb", "rgsaddle/ci-neb"),
    ("CiNebLbfgs", "rgsaddle/ci-neb-lbfgs"),
    ("EwCiNeb", "rgsaddle/ew-ci-neb"),
    ("EwCiNebLbfgs", "rgsaddle/ew-ci-neb-lbfgs"),
    ("IdppCiNeb", "rgsaddle/idpp-ci-neb"),
    ("IdppCiNebLbfgs", "rgsaddle/idpp-ci-neb-lbfgs"),
    ("SidppCiNeb", "rgsaddle/sidpp-ci-neb"),
    ("SidppCiNebLbfgs", "rgsaddle/sidpp-ci-neb-lbfgs"),
    ("OciNeb", "rgsaddle/oci-neb"),
    ("OciNebLbfgs", "rgsaddle/oci-neb-lbfgs"),
)

TRACKS = (
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_force_norm",
    "track_failure_hessian_index",
    "track_failure_other",
)

_KNOWN_REASONS = ("cap", "force_norm", "hessian_index")
CLASS_PROBLEM: dict[str, str] = {}
CLASS_TEMPLATE: dict[str, str] = {}


def result_root() -> Path:
    """Directory that holds ``results/rgsaddle-asv`` sidecars."""
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    workspace = Path("/workspace")
    if (workspace / "optbench").is_dir():
        return workspace
    return ROOT


def sidecar_path(template_name: str, seed: str, start: str) -> Path:
    """One JSON record for one asv parameter pair."""
    slug = template_name.replace("/", "-")
    return result_root() / "results" / "rgsaddle-asv" / slug / str(seed) / f"{start}.json"


def _start_names(problem_name: str) -> list[str]:
    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.pes import load_pairs

    problem = resolve_problem(ROOT, problem_name)
    pairs = load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )
    return [pair.name for pair in pairs]


def _failure_flag(record: dict, reason: str) -> float:
    reasons = list(record["reasons"])
    if reason == "other":
        return 1.0 if any(item not in _KNOWN_REASONS for item in reasons) else 0.0
    return 1.0 if reason in reasons else 0.0


def _bind(class_name: str, problem: str, template: str, starts: list[str]):
    class _Benchmark:
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = 1800
        repeat = 1
        rounds = 1
        number = 1
        min_run_count = 1
        warmup_time = 0
        processes = 1
        problem_name = problem
        template_name = template

        def setup(self, seed: str, start: str) -> None:
            from optbench.rgsaddle_band import library_path

            if library_path() is None:
                raise NotImplementedError("librgsaddle.so is not available")
            if start not in self.params[1]:
                raise KeyError(start)

        def time_search(self, seed: str, start: str) -> None:
            from optbench.rgsaddle_band import search_named

            print(f"start {self.template_name} {start}", file=sys.stderr, flush=True)
            began = time.perf_counter()
            record = search_named(ROOT, self.problem_name, self.template_name, start, seed)
            record["search_seconds"] = time.perf_counter() - began
            path = sidecar_path(self.template_name, seed, start)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"done {self.template_name} {start} calls={record['force_calls']} "
                f"success={record['success']} reasons={record['reasons']}",
                file=sys.stderr,
                flush=True,
            )

        def _record(self, seed: str, start: str) -> dict:
            path = sidecar_path(self.template_name, seed, start)
            if not path.is_file():
                raise FileNotFoundError(path)
            return json.loads(path.read_text(encoding="utf-8"))

        def track_force_calls(self, seed: str, start: str) -> float:
            return float(self._record(seed, start)["force_calls"])

        def track_success(self, seed: str, start: str) -> float:
            return 1.0 if self._record(seed, start)["success"] else 0.0

        def track_failure_cap(self, seed: str, start: str) -> float:
            return _failure_flag(self._record(seed, start), "cap")

        def track_failure_force_norm(self, seed: str, start: str) -> float:
            return _failure_flag(self._record(seed, start), "force_norm")

        def track_failure_hessian_index(self, seed: str, start: str) -> float:
            return _failure_flag(self._record(seed, start), "hessian_index")

        def track_failure_other(self, seed: str, start: str) -> float:
            return _failure_flag(self._record(seed, start), "other")

    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "success"
    _Benchmark.track_failure_cap.unit = "failure"
    _Benchmark.track_failure_force_norm.unit = "failure"
    _Benchmark.track_failure_hessian_index.unit = "failure"
    _Benchmark.track_failure_other.unit = "failure"
    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    CLASS_PROBLEM[class_name] = problem
    CLASS_TEMPLATE[class_name] = template
    return _Benchmark


def _expected_track(record: dict, method: str) -> float:
    if method == "track_force_calls":
        return float(record["force_calls"])
    if method == "track_success":
        return 1.0 if record["success"] else 0.0
    if method == "track_failure_cap":
        return _failure_flag(record, "cap")
    if method == "track_failure_force_norm":
        return _failure_flag(record, "force_norm")
    if method == "track_failure_hessian_index":
        return _failure_flag(record, "hessian_index")
    if method == "track_failure_other":
        return _failure_flag(record, "other")
    raise KeyError(method)


def _asv_column(document: dict, benchmark: str, column: str):
    row = document["results"][benchmark]
    index = document["result_columns"].index(column)
    if index >= len(row) or row[index] is None:
        raise KeyError(f"{benchmark} has no {column}")
    return row[index]


def verify_asv_tracks(document: dict) -> None:
    """Every rgsaddle ``track_*`` value equals the sidecar from that search."""
    for name in document["results"]:
        parts = name.split(".")
        if len(parts) < 2 or parts[-1] not in TRACKS:
            continue
        class_name = parts[-2]
        if class_name not in CLASS_TEMPLATE:
            continue
        method = parts[-1]
        params = _asv_column(document, name, "params")
        values = _asv_column(document, name, "result")
        pairs = list(itertools.product(*params))
        if len(values) != len(pairs):
            raise ValueError(f"{name} has {len(values)} results for {len(pairs)} parameters")
        template_name = CLASS_TEMPLATE[class_name]
        for (seed, start), value in zip(pairs, values, strict=True):
            if value is None or value != value:
                raise ValueError(f"{name}({seed}, {start}) has no result")
            record = json.loads(sidecar_path(template_name, seed, start).read_text(encoding="utf-8"))
            expected = _expected_track(record, method)
            if float(value) != expected:
                raise ValueError(f"{name}({seed}, {start}) is {value}, the sidecar has {expected}")


def attach_wall_seconds(document: dict, directory: Path) -> int:
    """Copy each ``time_search`` sample onto the sidecar as ``wall_seconds``.

    The sample is asv's timed call. Returns how many sidecars were updated.
    """
    updated = 0
    for name, row in document["results"].items():
        parts = name.split(".")
        if len(parts) < 2 or parts[-1] != "time_search":
            continue
        class_name = parts[-2]
        if class_name not in CLASS_TEMPLATE:
            continue
        index = document["result_columns"].index("result")
        if index >= len(row) or row[index] is None:
            raise KeyError(f"{name} has no result")
        params = row[document["result_columns"].index("params")]
        values = row[index]
        template_name = CLASS_TEMPLATE[class_name]
        for (seed, start), value in zip(itertools.product(*params), values, strict=True):
            if value is None or value != value:
                raise ValueError(f"{name}({seed}, {start}) has no wall time")
            path = directory / template_name.replace("/", "-") / str(seed) / f"{start}.json"
            if not path.is_file():
                path = sidecar_path(template_name, seed, start)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["wall_seconds"] = float(value)
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            updated += 1
    return updated


for _problem_prefix, _problem_name in PROBLEMS:
    _starts = _start_names(_problem_name)
    for _template_prefix, _template_name in TEMPLATES:
        _class_name = f"{_problem_prefix}Rgsaddle{_template_prefix}"
        globals()[_class_name] = _bind(_class_name, _problem_name, _template_name, _starts)
