"""OPTIM searches as airspeed velocity benchmarks.

Transition-state and lowest-eigenvector classes are one problem and one
template. Parameters are the seed and the archived start. The default
grid is one seed and the first geometry. ``OPTBENCH_ASV_FULL=1`` selects
every start. It is read when the name list is built.

``time_search`` is the timed search. The ``track_*`` methods read the
same record, so a full ``asv run`` records force calls, success, the
four failure classes, and wall time from that search. Counting is the
harness socket. A missing ``OPTIM_BIN`` raises in ``setup``.

Doubly-nudged elastic band classes use ``optim/dneb-hef``. Parameters
are the seed and the archived pair. The default grid is one seed and
the first geometry. ``time_search`` writes a sidecar under
``results/optim-asv``. The connect run does not draw random numbers.

USERPOT LBFGS minimization classes are one class per template. Their
``param_names`` are the seed and the start index. Seed ``0`` is the
archived geometry. OPTIM has no optimizer seed. The noise templates add
their own force-noise seed inside the job. The default grid is that
seed and start ``0``. ``setup`` raises ``NotImplementedError`` when the
OPTIM binary is missing. Those helpers are named ``_cache_path_min`` and
``_load_min`` so they do not replace the transition-state cache.
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
CACHE = Path(os.environ.get("OPTBENCH_OPTIM_CACHE", "/tmp/optbench-asv-optim"))
RUNNER_REVISION = "socket-1"


def lj38_start_names() -> list[str]:
    return [f"{index:04d}" for index in range(200)]


def pt_start_names() -> list[str]:
    return [f"initial_{index}" for index in range(49)]


def selected_lj38() -> list[str]:
    names = lj38_start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def selected_pt() -> list[str]:
    names = pt_start_names()
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return names
    return names[:1]


def _cache_path(template_id: str, start: str, seed: str) -> Path:
    safe = template_id.replace("/", "-")
    return CACHE / RUNNER_REVISION / safe / f"{start}__seed{seed}.json"


def _execute(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    from optbench.campaign import load_template, resolve_problem
    from optbench.optim_campaign import _geometry_specs, _prepare, install_relay, optim_binary

    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, problem_id)
    work = CACHE / "geometries" / problem_id.replace("/", "-")
    specs = _geometry_specs(ROOT, problem, template, work, optim_binary(), install_relay())
    spec = next((item for item in specs if item["name"] == start), None)
    if spec is None:
        raise RuntimeError(f"{problem_id} has no start {start}")
    record = _prepare(spec)
    record["seed"] = str(seed)
    record["template"] = template_id
    record["problem"] = problem_id
    path = _cache_path(template_id, start, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return record


def _load(template_id: str, problem_id: str, start: str, seed: str) -> dict:
    path = _cache_path(template_id, start, seed)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _execute(template_id, problem_id, start, seed)


def _cap(template_id: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_id)
    problem = resolve_problem(ROOT, template["problem"])
    cap = _check_stop(problem, template)
    if cap is None:
        return int(((template.get("ini") or {}).get("Optimizer") or {}).get("max_iterations") or 10000)
    return int(cap)


def _bind(class_name: str, template_id: str, problem_id: str, starts: list[str], limit: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], starts]
        timeout = limit

        def setup(self, seed, start):
            from optbench.optim_campaign import optim_binary

            optim_binary()

        def measure(self, seed, start):
            return _execute(template_id, problem_id, start, seed)

        def time_search(self, seed, start):
            self.record = self.measure(seed, start)

        def track_force_calls(self, seed, start) -> float:
            return float(_load(template_id, problem_id, start, seed)["force_calls"])

        def track_success(self, seed, start) -> float:
            record = _load(template_id, problem_id, start, seed)
            return 1.0 if record["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if _load(template_id, problem_id, start, seed)["failure"] == "crash" else 0.0

        track_force_calls.unit = "force calls"
        track_success.unit = "fraction"
        track_failure_cap.unit = "count"
        track_failure_wrong_stationary.unit = "count"
        track_failure_divergence.unit = "count"
        track_failure_crash.unit = "count"

    _Benchmark.__name__ = class_name
    _Benchmark.__qualname__ = class_name
    _Benchmark.__module__ = __name__
    _Benchmark.cap = _cap(template_id)
    _Benchmark.template_id = template_id
    _Benchmark.problem_id = problem_id
    return _Benchmark


_ROWS = [
    ("OptimLj38Hef", "optim/hef-ffd-ts-lj38", "tsopt/LJ38", selected_lj38(), 1800),
    ("OptimLj38Bfgsts", "optim/bfgsts-ts-lj38", "tsopt/LJ38", selected_lj38(), 1800),
    ("OptimPtHef", "optim/hef-ffd-ts-pt", "tsopt/pt-island", selected_pt(), 3600),
    ("OptimPtBfgsts", "optim/bfgsts-ts-pt", "tsopt/pt-island", selected_pt(), 3600),
    ("OptimLj38RayleighRitz", "optim/rr-loweig", "lowest-eigenvector/lj38", selected_lj38(), 1800),
]

for _class_name, _template_id, _problem_id, _starts, _timeout in _ROWS:
    globals()[_class_name] = _bind(_class_name, _template_id, _problem_id, _starts, _timeout)

PROBLEMS = (
    ("Heptamer", "neb/Pt-heptamer"),
    ("PtIsland", "double-ended-saddle-search/pt-island"),
    ("Lj38De", "double-ended-saddle-search/lj38"),
    ("Lj38TwoTs", "double-ended-saddle-search/lj38_2ts_separation"),
)

TEMPLATE = "optim/dneb-hef"

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
    """Directory that holds ``results/optim-asv`` sidecars."""
    env = os.environ.get("OPTBENCH_RESULT_ROOT")
    if env:
        return Path(env)
    return ROOT


def sidecar_path(problem_name: str, template_name: str, seed: str, start: str) -> Path:
    """One JSON record for one asv parameter pair."""
    slug = problem_name.replace("/", "-") + "--" + template_name.replace("/", "-")
    return result_root() / "results" / "optim-asv" / slug / str(seed) / f"{start}.json"


def all_starts(problem_name: str) -> list[str]:
    """Every archived pair name for ``problem_name``, in archive order."""
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
    """The first geometry, or every name when ``OPTBENCH_ASV_FULL=1``."""
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


def _expected_track(record: dict, method: str) -> float:
    if method == "track_force_calls":
        return float(record["force_calls"])
    if method == "track_success":
        return 1.0 if record["success"] else 0.0
    failure = _failure_of(record)
    name = method.removeprefix("track_failure_")
    return 1.0 if failure == name else 0.0


def _bind_dneb(class_name: str, problem: str, template: str, starts: list[str], cap: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], list(starts)]
        timeout = 3600
        problem_name = problem
        template_name = template
        problem_id = problem
        template_id = template

        def setup(self, seed, start):
            from optbench.optim_connect import optim_binary

            if optim_binary() is None:
                raise NotImplementedError("OPTIM is not available")
            path = sidecar_path(self.problem_name, self.template_name, str(seed), str(start))
            if path.is_file():
                self.record = json.loads(path.read_text(encoding="utf-8"))
            else:
                self.record = None

        def measure(self, seed, start):
            from optbench.optim_connect import search_named

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
            print(f"start {self.template_name} {self.problem_name} {start}", file=sys.stderr, flush=True)
            began = time.perf_counter()
            record = self.measure(seed, start)
            record["search_seconds"] = time.perf_counter() - began
            _write_sidecar(self.problem_name, self.template_name, str(seed), str(start), record)
            self.record = record
            print(
                f"done {self.template_name} {self.problem_name} {start} calls={record['force_calls']} "
                f"success={record['success']} failure={record['failure']}",
                file=sys.stderr,
                flush=True,
            )

        def _ready(self, seed, start) -> dict:
            record = getattr(self, "record", None)
            if record is None:
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
    """Every OPTIM ``track_*`` value equals the sidecar from that search."""
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
            expected = _expected_track(record, method)
            if float(value) != expected:
                raise ValueError(f"{name}({seed}, {start}) is {value}, the sidecar has {expected}")


def attach_wall_seconds(document: dict, directory: Path) -> int:
    """Copy each ``time_search`` sample onto the sidecar as ``wall_seconds``."""
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
            path = (
                directory
                / (CLASS_PROBLEM[class_name].replace("/", "-") + "--" + template_name.replace("/", "-"))
                / str(seed)
                / f"{start}.json"
            )
            if not path.is_file():
                path = sidecar_path(CLASS_PROBLEM[class_name], template_name, seed, start)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["wall_seconds"] = float(value)
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            updated += 1
    return updated


for _problem_prefix, _problem_name in PROBLEMS:
    _names = all_starts(_problem_name)
    _cap = problem_cap(_problem_name)
    _selected = selected_starts(_names)
    _class_name = f"{_problem_prefix}OptimDnebHef"
    globals()[_class_name] = _bind_dneb(_class_name, _problem_name, TEMPLATE, _selected, _cap)


SEED = "0"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-optim"))


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def asv_benchmark_pattern(class_name: str) -> str:
    """Anchored selector. The trailing dot keeps a prefix from matching a tuned class."""
    return rf"^bench_optim\.{class_name}\."


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Optim" + "".join(part.capitalize() for part in slug.split("-"))


def _archive_count(problem_id: str) -> int:
    import tarfile

    from optbench.campaign import geometry_archive, resolve_problem

    archive = geometry_archive(ROOT, resolve_problem(ROOT, problem_id))
    with tarfile.open(archive) as packed:
        return sum(1 for name in packed.getnames() if name.endswith(".con"))


def _geometries(problem_id: str):
    if problem_id not in _GEOMETRIES:
        from optbench.campaign import resolve_problem
        from optbench.runners.rgmin_min import _load_geometries

        problem = resolve_problem(ROOT, problem_id)
        work = Path("/tmp/optbench-asv-geo")
        _GEOMETRIES[problem_id] = _load_geometries(ROOT, problem, None, work, work / "unused.dat")
    return _GEOMETRIES[problem_id]


_GEOMETRIES: dict[str, list] = {}


def _cache_path_min(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{start}.json"


def _specs() -> list[dict]:
    from optbench.campaign import list_templates, resolve_problem
    from optbench.noise import KAPPA, window_energy_tolerance
    from optbench.runners.optim_min import _kind, _options

    found = []
    counts: dict[str, int] = {}
    for template in list_templates(ROOT):
        # ASE-socket rows and the unhooked template share this engine name.
        if template.get("engine") != "optim" or str(template.get("code_version")) != "02aaf45245af":
            continue
        problem_id = str(template["problem"])
        if problem_id not in counts:
            counts[problem_id] = _archive_count(problem_id) if os.environ.get("OPTBENCH_ASV_FULL") == "1" else 1
        problem = resolve_problem(ROOT, problem_id)
        convergence = problem.get("convergence") or {}
        updates, maxbfgs = _options(template)
        noise = template.get("noise") or {}
        sigma = float(noise.get("sigma", 0.0))
        if problem_id == "minimization/lj38-noise":
            kappa = float((problem.get("noise") or {}).get("kappa", KAPPA))
            energy_tol = window_energy_tolerance(float(convergence["energy_tol"]), sigma, kappa)
            seed = int(noise.get("seed", (problem.get("noise") or {})["seed"]))
        else:
            energy_tol = 0.0
            seed = 0
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "kind": _kind(problem_id),
                "updates": updates,
                "maxbfgs": maxbfgs,
                "sigma": sigma,
                "noise_seed": seed,
                "energy_tol": energy_tol,
                "gtol": float(convergence["force_norm"]),
                "cap": int(convergence["max_force_calls"]),
                "code_version": str(template["code_version"]),
                "n": counts[problem_id],
            }
        )
    return found


def _job(spec: dict, start: str) -> dict:
    from optbench.runners.optim_min import optim_binary

    binary = optim_binary()
    if binary is None:
        raise RuntimeError("OPTIM is not on PATH and $OPTIM is unset")
    index = int(start)
    geometries = _geometries(spec["problem_id"])
    if index < 0 or index >= len(geometries):
        raise IndexError(f"{spec['template_id']} start {start} is outside the archive")
    name, coords, box = geometries[index]
    kind = spec["kind"]
    return {
        "index": index,
        "start": name,
        "coords": [float(value) for value in coords],
        "box": None if kind == "lj" or box is None else [float(value) for value in box],
        "kind": kind,
        "sigma": spec["sigma"],
        "seed": spec["noise_seed"],
        "clean_tol": spec["gtol"],
        "energy_tol": spec["energy_tol"],
        "cap": spec["cap"],
        "updates": spec["updates"],
        "maxbfgs": spec["maxbfgs"],
        "binary": str(binary),
        "code_version": spec["code_version"],
        "work": str(_cache_dir() / "work" / spec["template_id"].replace("/", "-") / f"{SEED}-{index}"),
        "record_path": "",
    }


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not the archived-start seed {SEED}")
    from optbench.runners.optim_min import _minimize_job

    record = _minimize_job(_job(spec, start))
    record["seed"] = SEED
    path = _cache_path_min(spec["template_id"], SEED, str(int(start)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load_min(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path_min(spec["template_id"], SEED, str(int(start)))
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _OptimSearch(_TrackedSearch):
    def setup(self, seed, start):
        from optbench.runners.optim_min import optim_binary

        if optim_binary() is None:
            raise NotImplementedError("OPTIM is not installed in this environment")
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.spec, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _load_min(self.spec, seed, start)
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


_OptimSearch.track_force_calls.unit = "force calls"
_OptimSearch.track_success.unit = "fraction"
_OptimSearch.track_failure_cap.unit = "count"
_OptimSearch.track_failure_wrong_stationary.unit = "count"
_OptimSearch.track_failure_divergence.unit = "count"
_OptimSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])
    limit = spec["cap"]

    class _Benchmark(_OptimSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 7200
        cap = limit
        spec = bound

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


def _preload() -> None:
    from optbench.runners.optim_min import optim_binary

    if os.environ.get("OPTBENCH_ASV_FULL") != "1" or optim_binary() is None:
        return
    seen = set()
    for spec in _specs():
        if spec["problem_id"] not in seen:
            seen.add(spec["problem_id"])
            _geometries(spec["problem_id"])


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)

_preload()
