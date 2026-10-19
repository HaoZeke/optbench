"""CP2K BAND entries as airspeed-velocity benchmarks.

One class is one problem and one template. Parameters are the seed and
the archived start. The default grid is one seed and the first geometry.
``OPTBENCH_ASV_FULL=1`` selects every start, and it is read at import.

``time_search`` is the timed search. It writes a sidecar. The ``track_*``
methods read that sidecar, so a full ``asv run`` records force calls,
success, the four failure classes, and wall time from the same search.
``setup`` does not search. A missing ``CP2K_BIN`` and no local binary is
``NotImplementedError``.

The seed is the string ``"0"``. The band does not draw random numbers.
Sidecars live under ``results/cp2k-asv`` and are not the published store.
The published numbers are the asv result file.
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
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from optbench.asv_tracks import TrackedSearch as _TrackedSearch

ROOT = Path(__file__).resolve().parents[1]

PROBLEMS = (
    ("Heptamer", "neb/Pt-heptamer"),
    ("PtIsland", "double-ended-saddle-search/pt-island"),
    ("Lj38De", "double-ended-saddle-search/lj38"),
)

TEMPLATES = (
    ("CiNeb", "cp2k/ci-neb"),
    ("CiNebTuned", "cp2k/ci-neb-tuned"),
    ("ItNeb", "cp2k/it-neb"),
    ("ItNebTuned", "cp2k/it-neb-tuned"),
    ("String", "cp2k/string"),
    ("StringTuned", "cp2k/string-tuned"),
)

TRACKS = (
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)

_FAILURES = ("cap", "wrong_stationary", "divergence", "crash")
CLASS_PROBLEM: dict[str, str] = {}
CLASS_TEMPLATE: dict[str, str] = {}
CLASS_CAP: dict[str, int] = {}


def result_root() -> Path:
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def sidecar_path(problem_name: str, template_name: str, seed: str, start: str) -> Path:
    slug = problem_name.replace("/", "-") + "--" + template_name.replace("/", "-")
    return result_root() / "results" / "cp2k-asv" / slug / str(seed) / f"{start}.json"


def all_starts(problem_name: str) -> list[str]:
    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.pes import load_pairs

    problem = resolve_problem(ROOT, problem_name)
    pairs = load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )
    return [pair.name for pair in pairs]


def problem_cap(problem_name: str) -> int:
    from optbench.campaign import resolve_problem

    return int(resolve_problem(ROOT, problem_name)["convergence"]["max_force_calls"])


def selected_starts(names: list[str]) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return list(names)
    return list(names[:1])


def _write_sidecar(problem_name: str, template_name: str, seed: str, start: str, record: dict) -> None:
    path = sidecar_path(problem_name, template_name, seed, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _failure_of(record: dict) -> str | None:
    if record.get("success"):
        return None
    label = record.get("failure")
    if label in _FAILURES:
        return str(label)
    from optbench.failures import classify_failure

    return classify_failure([str(reason) for reason in record.get("reasons") or []]) or "crash"


def _bind(class_name: str, problem: str, template: str, starts: list[str], cap: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = 7200
        problem_name = problem
        template_name = template

        def setup(self, seed, start):
            from optbench.cp2k_band import binary_path

            if binary_path() is None:
                raise NotImplementedError("cp2k binary is not available")
            path = sidecar_path(self.problem_name, self.template_name, str(seed), str(start))
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.record = None

        def measure(self, seed, start):
            from optbench.cp2k_band import search_named

            try:
                record = search_named(ROOT, self.problem_name, self.template_name, str(start), str(seed))
            except Exception as exc:
                record = {
                    "schema": "optbench.start/v2",
                    "start": str(start),
                    "seed": str(seed),
                    "force_calls": 0,
                    "verify_calls": 0,
                    "counter": "harness",
                    "comparable": True,
                    "success": False,
                    "failure": "crash",
                    "reasons": ["crash"],
                    "crash": type(exc).__name__,
                    "problem": self.problem_name,
                    "template": self.template_name,
                }
            record["force_calls"] = float(record["force_calls"])
            record["success"] = bool(record["success"])
            record["failure"] = _failure_of(record)
            if record["success"]:
                record["failure"] = None
            return record

        def time_search(self, seed, start):
            print(f"start {self.template_name} {start}", file=sys.stderr, flush=True)
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            _write_sidecar(self.problem_name, self.template_name, str(seed), str(start), record)
            self.record = record
            print(
                f"done {self.template_name} {start} calls={record['force_calls']} "
                f"success={record['success']} failure={record['failure']}",
                file=sys.stderr,
                flush=True,
            )

        def _ready(self, seed, start) -> dict:
            record = getattr(self, "record", None)
            if record is None:
                path = sidecar_path(self.problem_name, self.template_name, str(seed), str(start))
                if path.is_file():
                    record = json.loads(path.read_text(encoding="utf-8"))
                else:
                    record = self.measure(seed, start)
                    _write_sidecar(self.problem_name, self.template_name, str(seed), str(start), record)
                self.record = record
            return record

        def track_force_calls(self, seed, start) -> float:
            return float(self._ready(seed, start)["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if self._ready(seed, start)["failure"] == "crash" else 0.0

    _Benchmark.cap = cap
    _Benchmark.track_force_calls.unit = "force calls"
    _Benchmark.track_success.unit = "fraction"
    _Benchmark.track_failure_cap.unit = "count"
    _Benchmark.track_failure_wrong_stationary.unit = "count"
    _Benchmark.track_failure_divergence.unit = "count"
    _Benchmark.track_failure_crash.unit = "count"
    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    CLASS_PROBLEM[class_name] = problem
    CLASS_TEMPLATE[class_name] = template
    CLASS_CAP[class_name] = cap
    return _Benchmark


def _asv_column(document: dict, benchmark: str, column: str):
    row = document["results"][benchmark]
    index = document["result_columns"].index(column)
    if index >= len(row) or row[index] is None:
        raise KeyError(f"{benchmark} has no {column}")
    return row[index]


def verify_asv_tracks(document: dict) -> None:
    """Every CP2K ``track_*`` value equals the sidecar from that search."""
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
            record = json.loads(
                sidecar_path(CLASS_PROBLEM[class_name], template_name, seed, start).read_text(encoding="utf-8")
            )
            if method == "track_force_calls":
                expected = float(record["force_calls"])
            elif method == "track_success":
                expected = 1.0 if record["success"] else 0.0
            else:
                failure = _failure_of(record)
                expected = 1.0 if failure == method.removeprefix("track_failure_") else 0.0
            if float(value) != expected:
                raise ValueError(f"{name}({seed}, {start}) is {value}, the sidecar has {expected}")


for _problem_prefix, _problem_name in PROBLEMS:
    _names = all_starts(_problem_name)
    _cap = problem_cap(_problem_name)
    _selected = selected_starts(_names)
    for _template_prefix, _template_name in TEMPLATES:
        _class_name = f"{_problem_prefix}Cp2k{_template_prefix}"
        globals()[_class_name] = _bind(_class_name, _problem_name, _template_name, _selected, _cap)


# GEO_OPT dimer benchmarks. Band classes above keep their own sidecar layout.
CACHE = Path(os.environ.get("OPTBENCH_CP2K_CACHE", "/tmp/optbench-asv-cp2k"))


def lj38_start_names() -> list[str]:
    return [f"{index:04d}" for index in range(200)]


def pt_start_names() -> list[str]:
    return [f"initial_{index}" for index in range(49)]


def _selected(names: list[str]) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return list(names)
    return list(names[:1])


def _sidecar(template_id: str, start: str, seed: str) -> Path:
    return CACHE / template_id.replace("/", "-") / f"{start}__seed{seed}.json"


def _require_binary() -> None:
    from optbench.cp2k_ts import cp2k_binary

    cp2k_binary()


def _execute(template_id: str, start: str, seed: str) -> dict:
    from optbench.cp2k_ts import run_start, spec_for_start

    spec = spec_for_start(ROOT, template_id, start, work=CACHE / "runs" / template_id.replace("/", "-"))
    spec["seed"] = str(seed)
    record = run_start(spec)
    record.pop("mode_axis", None)
    record["seed"] = str(seed)
    path = _sidecar(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def _load(template_id: str, start: str, seed: str) -> dict:
    path = _sidecar(template_id, start, seed)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _execute(template_id, start, seed)


def _cap(template_id: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_id)
    cap = _check_stop(resolve_problem(ROOT, template["problem"]), template)
    if cap is None:
        raise RuntimeError(f"{template_id} has no force-call cap")
    return int(cap)


def _bind_dimer(class_name: str, template_id: str, problem_id: str, starts: list[str], timeout: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]

        def setup(self, seed, start):
            _require_binary()

        def measure(self, seed, start):
            return _execute(template_id, str(start), str(seed))

        def time_search(self, seed, start):
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            path = _sidecar(template_id, str(start), str(seed))
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            self.record = record

        def track_force_calls(self, seed, start) -> float:
            return float(_load(template_id, str(start), str(seed))["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if _load(template_id, str(start), str(seed))["failure"] == "crash" else 0.0

        track_force_calls.unit = "force calls"
        track_success.unit = "fraction"
        track_failure_cap.unit = "count"
        track_failure_wrong_stationary.unit = "count"
        track_failure_divergence.unit = "count"
        track_failure_crash.unit = "count"

    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.timeout = timeout
    _Benchmark.cap = _cap(template_id)
    _Benchmark.template_id = template_id
    _Benchmark.problem_id = problem_id
    return _Benchmark


_ROWS = [
    ("Cp2kLj38DimerCg", "cp2k/lj38-dimer-cg", "tsopt/LJ38", _selected(lj38_start_names()), 3600),
    ("Cp2kLj38DimerCgTuned", "cp2k/lj38-dimer-cg-tuned", "tsopt/LJ38", _selected(lj38_start_names()), 3600),
    ("Cp2kPtDimerCg", "cp2k/pt-island-dimer-cg", "tsopt/pt-island", _selected(pt_start_names()), 7200),
    ("Cp2kPtDimerCgTuned", "cp2k/pt-island-dimer-cg-tuned", "tsopt/pt-island", _selected(pt_start_names()), 7200),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind_dimer(_class_name, _template_id, _problem_id, _starts, _timeout)
