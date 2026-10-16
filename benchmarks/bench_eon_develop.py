"""eOn develop benchmarks for asv.

One class is one template. Parameters are the seed and the start.
``time_run`` is the client (or AKMC server) invocation, so asv's timing
is the wall time. It runs before the ``track_*`` methods because the
benchmark name sorts first. Those tracks read the result ``time_run``
wrote.

``track_force_calls`` is the client ``results.dat`` key
``total_force_calls`` (energy and gradient evaluations on the
potential). The AKMC classes use ``sad-fcs + mins-fcs`` through the
27th good saddle below 1.5 eV instead. Neither number is a Python
harness wrap.

``track_success`` is 1 when the termination reason is 0, and 0
otherwise. ``track_failure_cause`` is that reason when it is an
integer, 100 when the lowest-mode overlap was not reached, 101 when an
AKMC repetition did not reach 27 saddles, and 102 when the client wrote
no result. A missing force-call count is NaN, not a filled-in cap.

GPR-dimer is not a class: the private gpr_optim tree was not available
and this eonclient reports GPRD disabled.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EON_COMMIT = "7095b23ac17d49f434a5e08ccb072897ceca698e"
CAUSE_NO_OVERLAP = 100
CAUSE_INCOMPLETE = 101
CAUSE_MISSING = 102
CAUSE_OTHER = 199

_CACHE = Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv"))


def failure_cause(reason: str | None) -> float:
    """Integer cause code stored by ``track_failure_cause``."""
    if reason is None or reason == "":
        return float(CAUSE_MISSING)
    if reason == "0":
        return 0.0
    if reason == "no-overlap":
        return float(CAUSE_NO_OVERLAP)
    if reason == "incomplete":
        return float(CAUSE_INCOMPLETE)
    try:
        return float(int(reason))
    except ValueError:
        return float(CAUSE_OTHER)


def success_flag(reason: str | None) -> float:
    return 1.0 if reason == "0" else 0.0


def _client() -> str:
    env = os.environ.get("EONCLIENT")
    if env:
        return env
    found = shutil.which("eonclient")
    if not found:
        raise RuntimeError("put eonclient on PATH or set EONCLIENT")
    return found


def _extract(archive: Path, member: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as packed:
        extracted = packed.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} is not in {archive}")
        dest.write_bytes(extracted.read())
    return dest


def _index(start: str) -> int:
    if "_" in start:
        return int(start.rsplit("_", 1)[1])
    return int(start)


def _cache_path(template_name: str, seed: int, start: str) -> Path:
    safe = template_name.replace("/", "-")
    return _CACHE / EON_COMMIT / safe / str(seed) / f"{start}.json"


def _write_cache(template_name: str, seed: int, start: str, row: dict) -> None:
    path = _cache_path(template_name, seed, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _read_cache(template_name: str, seed: int, start: str) -> dict | None:
    path = _cache_path(template_name, seed, start)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _calls(raw) -> float:
    if raw is None:
        return float("nan")
    return float(raw)


def _finish_row(raw, reason: str | None, **extra) -> dict:
    row = {
        "failure_cause": failure_cause(reason),
        "force_calls": None if raw is None else int(raw),
        "reason": reason,
        "success": success_flag(reason),
    }
    row.update(extra)
    return row


def _ini(template: dict, seed: int) -> str:
    from optbench.campaign import format_ini

    sections = json.loads(json.dumps(template["ini"]))
    sections.setdefault("Main", {})["random_seed"] = int(seed)
    return format_ini(sections)


def _client_row(template: dict, seed: int, work: Path, copies, texts) -> dict:
    from optbench.eon_jobs import _execute

    spec = (
        _client(),
        _ini(template, seed),
        str(work / "run"),
        copies,
        texts,
        "total_force_calls",
    )
    found = _execute(spec)
    return _finish_row(found["raw"], found["reason"], extrema=found.get("extrema"))


def run_case(template_name: str, kind: str, seed: int, start: str) -> dict:
    """One seed and one start. The dict is what the tracks report."""
    from optbench.campaign import load_template
    from optbench.eon_jobs import (
        _coords,
        _difference_mode,
        _lowest_job,
        _numpy_direction,
        _seeded_mode,
        _stored,
    )

    template = load_template(ROOT, template_name)
    work = Path(tempfile.mkdtemp(prefix="optbench-eon-"))
    try:
        if kind == "ts-lj38":
            archive = _stored(ROOT, "website/tsopt/LJ38/coords-con.tar.gz")
            con = _extract(archive, f"coords-con/{start}.con", work / "src.con")
            text = con.read_text(encoding="utf-8", errors="replace")
            mode = _seeded_mode(len(_coords(text)), _index(start))
            return _client_row(
                template,
                seed,
                work,
                [(str(con), "pos.con"), (str(con), "displacement.con")],
                [("direction.dat", mode)],
            )
        if kind == "ts-pt":
            archive = _stored(ROOT, "website/tsopt/pt-island/pt-island-con.tgz")
            reactant = _extract(archive, "pt-island-con/reactant.con", work / "reactant.con")
            initial = _extract(
                archive,
                f"pt-island-con/initial_{_index(start)}.con",
                work / "initial.con",
            )
            mode = _difference_mode(reactant, initial)
            return _client_row(
                template,
                seed,
                work,
                [(str(reactant), "pos.con"), (str(initial), "displacement.con")],
                [("direction.dat", mode)],
            )
        if kind == "de-lj38":
            archive = _stored(ROOT, "website/double-ended-saddle-search/lj38/coords-con.tgz")
            index = _index(start)
            reactant = _extract(archive, f"coords-con/start_{index:02d}.con", work / "start.con")
            product = _extract(archive, f"coords-con/end_{index:02d}.con", work / "end.con")
            lammps = ROOT / "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/in.lammps"
            return _client_row(
                template,
                seed,
                work,
                [
                    (str(reactant), "reactant.con"),
                    (str(product), "product.con"),
                    (str(lammps), "in.lammps"),
                ],
                [],
            )
        if kind == "de-pt":
            archive = _stored(ROOT, "website/double-ended-saddle-search/pt-island/pt-island-con.tgz")
            reactant = _extract(archive, "pt-island-con/reactant.con", work / "reactant.con")
            product = _extract(
                archive,
                f"pt-island-con/product_{_index(start)}.con",
                work / "product.con",
            )
            return _client_row(
                template,
                seed,
                work,
                [(str(reactant), "reactant.con"), (str(product), "product.con")],
                [],
            )
        if kind == "de-2ts":
            from optbench.eon_parse import xyz_to_con

            archive = _stored(ROOT, "website/double-ended-saddle-search/lj38_2ts_separation/coords.tgz")
            index = _index(start)
            start_xyz = _extract(archive, f"coords/start_{index}.xyz", work / "start.xyz")
            end_xyz = _extract(archive, f"coords/end_{index}.xyz", work / "end.xyz")
            return _client_row(
                template,
                seed,
                work,
                [],
                [
                    ("reactant.con", xyz_to_con(start_xyz.read_text(encoding="utf-8"))),
                    ("product.con", xyz_to_con(end_xyz.read_text(encoding="utf-8"))),
                ],
            )
        if kind == "lowest":
            archive = _stored(ROOT, "website/lowest-eigenvector/lj38/coords-con.tar.gz")
            con = _extract(archive, f"coords-con/{start}.con", work / "src.con")
            mode = _extract(archive, f"coords-con/{start}_mode", work / "ref.mode")
            sections = json.loads(json.dumps(template["ini"]))
            sections.setdefault("Main", {})["random_seed"] = int(seed)
            method = sections["Saddle Search"]["min_mode_method"]
            found = _lowest_job(
                (
                    _client(),
                    json.dumps(sections),
                    str(work / "run"),
                    str(con),
                    mode.read_text(encoding="utf-8", errors="replace"),
                    _numpy_direction(),
                    method,
                    int(template.get("iteration_limit", 1000)),
                    float(template.get("overlap", 0.99)),
                )
            )
            return _finish_row(
                found["raw"],
                found["reason"],
                client_reason=found.get("client_reason"),
                k=found.get("k"),
                overlap=found.get("overlap"),
            )
        if kind == "akmc":
            from optbench.eon_akmc import _one_repetition

            row = _one_repetition(
                ROOT,
                template,
                Path(_client()),
                work / "akmc",
                int(seed),
                min(8, os.cpu_count() or 1),
                passes=int(template.get("pass_limit", 20000)),
            )
            reason = "0" if row["reached"] else "incomplete"
            return _finish_row(
                row["force_calls"],
                reason,
                fraction_connected=row["fraction_connected"],
                passes=row.get("passes"),
                qualifying=row["qualifying"],
                total_searches=row["total_searches"],
                unique_saddles=row["unique_saddles"],
            )
        raise ValueError(f"unknown eon benchmark kind {kind}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _ensure(bench, seed, start) -> dict:
    seed = int(seed)
    start = str(start)
    cached = _read_cache(bench.template_name, seed, start)
    if cached is not None:
        return cached
    row = run_case(bench.template_name, bench.kind, seed, start)
    _write_cache(bench.template_name, seed, start, row)
    return row


def _bind(spec: dict, akmc: bool):
    class _Benchmark:
        param_names = ["seed", "start"]
        params = [spec["seeds"], spec["starts"]]
        timeout = spec["timeout"]
        repeat = 1
        rounds = 1
        number = 1
        warmup_time = 0
        version = EON_COMMIT
        template_name = spec["template"]
        kind = spec["kind"]

        def time_run(self, seed, start):
            _ensure(self, seed, start)

        def track_force_calls(self, seed, start) -> float:
            return _calls(_ensure(self, seed, start)["force_calls"])

        def track_success(self, seed, start) -> float:
            return float(_ensure(self, seed, start)["success"])

        def track_failure_cause(self, seed, start) -> float:
            return float(_ensure(self, seed, start)["failure_cause"])

        track_force_calls.unit = "force calls"
        track_success.unit = "success"
        track_failure_cause.unit = "cause"

    if akmc:

        def track_unique_saddles(self, seed, start) -> float:
            value = _ensure(self, seed, start)["unique_saddles"]
            return float("nan") if value is None else float(value)

        def track_fraction_connected(self, seed, start) -> float:
            value = _ensure(self, seed, start)["fraction_connected"]
            return float("nan") if value is None else float(value)

        def track_total_searches(self, seed, start) -> float:
            value = _ensure(self, seed, start)["total_searches"]
            return float("nan") if value is None else float(value)

        track_unique_saddles.unit = "saddles"
        track_fraction_connected.unit = "fraction"
        track_total_searches.unit = "searches"
        _Benchmark.track_unique_saddles = track_unique_saddles
        _Benchmark.track_fraction_connected = track_fraction_connected
        _Benchmark.track_total_searches = track_total_searches

    if spec["kind"] == "de-2ts":

        def track_extrema(self, seed, start) -> float:
            value = _ensure(self, seed, start).get("extrema")
            return float("nan") if value is None else float(value)

        track_extrema.unit = "extrema"
        _Benchmark.track_extrema = track_extrema

    return _Benchmark


def _specs() -> list[dict]:
    lj = [f"{i:04d}" for i in range(200)]
    return [
        {
            "cls": "EonDevelopTsLj38Lanczos",
            "template": "eon/develop-ts-lj38-lanczos",
            "kind": "ts-lj38",
            "seeds": [1],
            "starts": lj,
            "timeout": 600,
        },
        {
            "cls": "EonDevelopTsLj38Dimer",
            "template": "eon/develop-ts-lj38-dimer",
            "kind": "ts-lj38",
            "seeds": [1],
            "starts": lj,
            "timeout": 600,
        },
        {
            "cls": "EonDevelopTsPtLanczos",
            "template": "eon/develop-ts-pt-lanczos",
            "kind": "ts-pt",
            "seeds": [1],
            "starts": [f"initial_{i}" for i in range(49)],
            "timeout": 600,
        },
        {
            "cls": "EonDevelopTsPtDimer",
            "template": "eon/develop-ts-pt-dimer",
            "kind": "ts-pt",
            "seeds": [1],
            "starts": [f"initial_{i}" for i in range(49)],
            "timeout": 600,
        },
        {
            "cls": "EonDevelopDeLj38Ew",
            "template": "eon/develop-de-lj38-ew",
            "kind": "de-lj38",
            "seeds": [1],
            "starts": [f"pair_{i:02d}" for i in range(50)],
            "timeout": 600,
        },
        {
            "cls": "EonDevelopDeLj38Oci",
            "template": "eon/develop-de-lj38-oci",
            "kind": "de-lj38",
            "seeds": [1],
            "starts": [f"pair_{i:02d}" for i in range(50)],
            "timeout": 600,
        },
        {
            "cls": "EonDevelopDeLj38Cap",
            "template": "eon/develop-de-lj38-cap",
            "kind": "de-lj38",
            "seeds": [1],
            "starts": [f"pair_{i:02d}" for i in range(50)],
            "timeout": 3600,
        },
        {
            "cls": "EonDevelopDePtEw",
            "template": "eon/develop-de-pt-ew",
            "kind": "de-pt",
            "seeds": [1],
            "starts": [f"product_{i}" for i in range(59)],
            "timeout": 1800,
        },
        {
            "cls": "EonDevelopDePtOci",
            "template": "eon/develop-de-pt-oci",
            "kind": "de-pt",
            "seeds": [1],
            "starts": [f"product_{i}" for i in range(59)],
            "timeout": 1800,
        },
        {
            "cls": "EonDevelopDe2ts",
            "template": "eon/develop-de-2ts",
            "kind": "de-2ts",
            "seeds": [1],
            "starts": [f"pair_{i:02d}" for i in range(51)],
            "timeout": 1800,
        },
        {
            "cls": "EonDevelopDe2tsOci",
            "template": "eon/develop-de-2ts-oci",
            "kind": "de-2ts",
            "seeds": [1],
            "starts": [f"pair_{i:02d}" for i in range(51)],
            "timeout": 1800,
        },
        {
            "cls": "EonDevelopDe2tsEw",
            "template": "eon/develop-de-2ts-ew",
            "kind": "de-2ts",
            "seeds": [1],
            "starts": [f"pair_{i:02d}" for i in range(51)],
            "timeout": 1800,
        },
        {
            "cls": "EonDevelopLowLj38Dimer",
            "template": "eon/develop-low-lj38-dimer",
            "kind": "lowest",
            "seeds": [1],
            "starts": lj,
            "timeout": 1800,
        },
        {
            "cls": "EonDevelopLowLj38Lanczos",
            "template": "eon/develop-low-lj38-lanczos",
            "kind": "lowest",
            "seeds": [1],
            "starts": lj,
            "timeout": 1800,
        },
        {
            "cls": "EonDevelopSaddlePtDimer",
            "template": "eon/develop-saddle-pt-dimer",
            "kind": "akmc",
            "seeds": list(range(1000, 1010)),
            "starts": ["pt-island"],
            "timeout": 7200,
        },
        {
            "cls": "EonDevelopSaddlePtLanczos",
            "template": "eon/develop-saddle-pt-lanczos",
            "kind": "akmc",
            "seeds": list(range(1000, 1010)),
            "starts": ["pt-island"],
            "timeout": 7200,
        },
    ]


for _spec in _specs():
    _cls = _bind(_spec, _spec["kind"] == "akmc")
    _cls.__name__ = _spec["cls"]
    _cls.__qualname__ = _spec["cls"]
    globals()[_spec["cls"]] = _cls
