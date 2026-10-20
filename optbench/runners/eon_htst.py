"""Stock dimer-CG AKMC on the Al hTST reactant, counted by the harness.

The client is latest EON r2549 or latest eOn develop. ``potential =
ext_pot`` and the kernel is the r2549 Aluminum Fortran ``force_``. One
force call is one ``HarnessCounter`` charge. Parallel searches share
that counter, so a window's harness cost is its share of the server
total in proportion to the ``sad-fcs`` of the searches that closed it.

The dimer block is the published algorithm: CG, 50 rotations, angle 1.
Every other server and client option stays at that code's constructor
default, including confidence, the displacement, and the prefactor.
``num_jobs`` and ``number_of_CPUs`` follow the campaign worker count.
They are how the run is executed, not optimizer parameters.

A barrier window and a rate window use the 2014 report's scoring rule
on the stock log: the eight reference barriers, ``rate_ref``
592041917, accuracy 0.999, prefactor 1.2e12, and 300 K. That rule
scores the log. It is not written into the ini. The archive averaged
639 barrier windows and 20 rate windows. This run stops when both of
those counts are in, or when ``pass_limit`` AKMC passes have finished.
Stock confidence does not end a reactant that has registered none of
the reference processes, so the pass bound is the observation. A side
that closes nothing keeps one ranked record per finished search.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from optbench.campaign import format_ini, geometry_archive
from optbench.eon_harness import bridge_binary, eam_al
from optbench.potserver import PotServer, remote_count
from optbench.runners.eon_min import _client
from optbench.score import score_v2
from optbench.v2 import write_starts

_ROOTS = {
    "svn": "EON_SVN_ROOT",
    "develop": "EON_DEVELOP_ROOT",
}
_ROOT_DEFAULTS = {
    "svn": Path("/tmp/eon-r2549"),
    "develop": Path("/tmp/eon-develop-src"),
}
_BARRIERS = [0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372]
_RATE_REF = 592041917.0
_RATE_ACCURACY = 0.999
_PREFACTOR = 1.2e12
_KB = 8.6173324e-5
_TEMPERATURE = 300.0
_BARRIER_TARGET = 639
_RATE_TARGET = 20


def _python_root(family: str) -> Path:
    env = os.environ.get(_ROOTS[family])
    path = Path(env) if env else _ROOT_DEFAULTS[family]
    if not (path / "eon" / "__init__.py").is_file():
        raise RuntimeError(f"{path} has no eon package for {family}")
    return path


def _searches(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        result = " ".join(fields[7:]) if len(fields) > 7 else fields[-1]
        status = result
        process_id = None
        if "-" in result:
            status, rest = result.split("-", 1)
            if status in ("good", "repeat"):
                try:
                    process_id = int(rest)
                except ValueError:
                    process_id = None
        rows.append(
            {
                "sad": int(float(fields[4])),
                "barrier": float(fields[2]),
                "status": status,
                "process_id": process_id,
            }
        )
    return rows


def _reference_ids(text: str) -> tuple[list[int], list[float]]:
    """Process ids that match the reference barrier multiset, and the barriers still missing.

    A later window may close only when the missing list is empty. A reactant
    that has found one of the eight barriers has not found the window.
    """
    wanted = list(_BARRIERS)
    found: list[int] = []
    if not text:
        return found, wanted
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == "proc" or len(fields) < 7:
            continue
        barrier = round(float(fields[6]), 3)
        if barrier in wanted:
            found.append(int(fields[0]))
            wanted.remove(barrier)
    return found, wanted


def _windows(
    rows: list[dict],
    harness: list[float],
    reference: list[int],
    missed: list[float],
) -> tuple[list[dict], list[dict]]:
    barrier_ids: list[int] = []
    rate_ids: list[int] = []
    rate_sum = 0.0
    threshold = _RATE_REF * _RATE_ACCURACY
    barrier_jobs = barrier_sad = barrier_harness = 0
    rate_jobs = rate_sad = rate_harness = 0
    barriers: list[dict] = []
    rates: list[dict] = []
    for row, cost in zip(rows, harness, strict=True):
        barrier_jobs += 1
        rate_jobs += 1
        barrier_sad += int(row["sad"])
        rate_sad += int(row["sad"])
        barrier_harness += float(cost)
        rate_harness += float(cost)
        process_id = row["process_id"]
        if process_id is None:
            continue
        # The eight reference barriers are one window. A shorter list of
        # processes that happen to match some of those barriers is not.
        if not missed and reference and process_id in reference and process_id not in barrier_ids:
            barrier_ids.append(process_id)
        if not missed and reference and len(barrier_ids) == len(reference):
            barriers.append(
                {
                    "jobs": barrier_jobs,
                    "sad_fcs": barrier_sad,
                    "force_calls": int(round(barrier_harness)),
                }
            )
            barrier_ids = []
            barrier_jobs = barrier_sad = 0
            barrier_harness = 0.0
        if rate_sum < threshold and process_id not in rate_ids:
            rate_ids.append(process_id)
            rate_sum += _PREFACTOR * math.exp(-float(row["barrier"]) / (_KB * _TEMPERATURE))
        if rate_sum >= threshold:
            rates.append(
                {
                    "jobs": rate_jobs,
                    "sad_fcs": rate_sad,
                    "force_calls": int(round(rate_harness)),
                }
            )
            rate_ids = []
            rate_jobs = rate_sad = 0
            rate_harness = 0.0
            rate_sum = 0.0
    return barriers, rates


def _assign(rows: list[dict], snapshots: list[tuple[int, int]]) -> list[float]:
    """Split each pass's harness delta across the searches that pass added."""
    assigned: list[float] = []
    previous_lines = 0
    previous_calls = 0
    for lines, calls in snapshots:
        group = rows[previous_lines:lines]
        delta = max(0, int(calls) - previous_calls)
        weights = [max(int(row["sad"]), 0) for row in group]
        total = sum(weights)
        if group:
            if total <= 0:
                assigned.extend([delta / len(group)] * len(group))
            else:
                assigned.extend([delta * weight / total for weight in weights])
        previous_lines = lines
        previous_calls = calls
    if len(assigned) < len(rows):
        assigned.extend([0.0] * (len(rows) - len(assigned)))
    return assigned[: len(rows)]


def _search_records(rows: list[dict], harness: list[float], template: dict, label: str) -> list[dict]:
    """One finished search per record when no reference window closed.

    A not-connected search is still a completed stock trial. ``success``
    stays true so the PAR-2 mean is the harness cost of those trials.
    ``window_count`` on the dat is zero, which is the gap.
    """
    records = []
    for index, (row, cost) in enumerate(zip(rows, harness, strict=True), 1):
        records.append(
            {
                "schema": "optbench.start/v2",
                "start": f"search-{index:04d}",
                "force_calls": int(round(float(cost))),
                "sad_fcs": int(row["sad"]),
                "result": row["status"],
                "counter": "harness",
                "comparable": True,
                "success": True,
                "failure": None,
                "problem": template.get("problem"),
                "template": template["id"] if label == "barrier" else template.get("rate_template"),
                "code": template.get("code"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
                "tuning": "default",
                "windows": 0,
            }
        )
    return records


def _records(windows: list[dict], template: dict, label: str) -> list[dict]:
    rows = []
    for index, window in enumerate(windows, 1):
        rows.append(
            {
                "schema": "optbench.start/v2",
                "start": f"{label}-{index:04d}",
                "force_calls": int(window["force_calls"]),
                "sad_fcs": int(window["sad_fcs"]),
                "jobs": int(window["jobs"]),
                "counter": "harness",
                "comparable": True,
                "success": True,
                "failure": None,
                "problem": template.get("problem"),
                "template": template["id"] if label == "barrier" else template.get("rate_template"),
                "code": template.get("code"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
                "tuning": "default",
            }
        )
    return rows


def _whole_record(template: dict, calls: int, label: str) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": "al",
        "force_calls": int(calls),
        "counter": "harness",
        "comparable": True,
        "success": True,
        "failure": None,
        "problem": template.get("problem"),
        "template": template["id"] if label == "barrier" else template.get("rate_template"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "tuning": "default",
        "windows": 0,
    }


def _dat(template: dict, records: list[dict], metrics: dict, *, label: str, windows: list[dict], total_calls: int, searches: int, partial: bool) -> str:
    sad = [int(window["sad_fcs"]) for window in windows]
    jobs = [int(window["jobs"]) for window in windows]
    calls = [int(record["force_calls"]) for record in records]
    rows = {
        "force_calls": int(round(sum(calls) / len(calls))) if calls else None,
        "force_calls_min": min(calls) if calls else None,
        "force_calls_max": max(calls) if calls else None,
        "nfailed": metrics.get("nfailed"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "algorithm": template.get("algorithm"),
        "template": template["id"] if label == "barrier" else template.get("rate_template"),
        "tuning": "default",
        "counter": "harness",
        "problem": template.get("problem"),
        "reducer": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "cap": int(template["failure_cap"]),
        "window": label,
        "window_count": len(windows),
        "searches": searches,
        "harness_total": int(total_calls),
    }
    window_calls = [int(window["force_calls"]) for window in windows]
    if label == "barrier":
        rows["barrier_jobs"] = int(round(sum(jobs) / len(jobs))) if jobs else 0
        rows["barrier_force_calls"] = int(round(sum(window_calls) / len(window_calls))) if window_calls else 0
        rows["barrier_sad_fcs"] = int(round(sum(sad) / len(sad))) if sad else 0
    else:
        rows["rate_jobs"] = int(round(sum(jobs) / len(jobs))) if jobs else 0
        rows["rate_force_calls"] = int(round(sum(window_calls) / len(window_calls))) if window_calls else 0
        rows["rate_sad_fcs"] = int(round(sum(sad) / len(sad))) if sad else 0
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one ext_pot evaluation of the r2549 Aluminum EAM.",
        "# A window's calls are its share of the server total, split by sad-fcs inside each pass.",
        "# The dimer block is CG, rotations_max 50, converged_angle 1. Other options stay at the constructor.",
        f"# {label} windows closed: {len(windows)}. Archive averages used 639 barrier windows and 20 rate windows.",
    ]
    if not windows:
        lines.append(
            "# No reference window closed. force_calls is the mean harness cost of the finished searches, not a window mean."
        )
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


def _write(path: Path, template: dict, records: list[dict], *, label: str, windows: list[dict], total_calls: int, searches: int, partial: bool, cap: int) -> dict:
    metrics = score_v2(
        [{"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]} for record in records],
        cap,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _dat(
            template,
            records,
            metrics,
            label=label,
            windows=windows,
            total_calls=total_calls,
            searches=searches,
            partial=partial,
        ),
        encoding="utf-8",
    )
    write_starts(path.with_suffix(".starts.jsonl"), records)
    return metrics


def _prepare_server(template: dict, client: str, seed: int, workers: int, wrapper: Path) -> str:
    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["job"] = "akmc"
    main["random_seed"] = int(seed)
    dimer = sections.setdefault("Dimer", {})
    for key in ("rotations_max", "converged_angle", "opt_method"):
        if key not in dimer:
            raise ValueError(f"{template['id']} is missing the dimer algorithm key {key}")
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    comm = sections.setdefault("Communicator", {})
    comm["type"] = "local"
    comm["client_path"] = client
    comm["number_of_CPUs"] = int(workers)
    comm["num_jobs"] = int(workers)
    return format_ini(sections)


def _flush_pending(akmc_mod, config) -> None:
    """Register jobs the pass just finished.

    ``akmc`` registers the previous pass and then submits the next one.
    The local communicator runs that batch before returning, so a second
    ``register_results`` is what puts those rows in ``search_results.txt``.
    """
    import inspect

    from eon import explorer

    kT = config.main_temperature / 11604.5
    getter = akmc_mod.get_akmc_metadata
    if "config" in inspect.signature(getter).parameters:
        meta = getter(config)
        states = akmc_mod.get_statelist(kT, config)
    else:
        meta = getter()
        states = akmc_mod.get_statelist(kT)
    start_state_num, _time, previous_state_num, _first, _previous_temperature = meta
    current = states.get_state(start_state_num)
    if int(previous_state_num) == -1:
        previous = current
    else:
        previous = states.get_state(previous_state_num)
    factory = explorer.get_minmodexplorer
    if "config" in inspect.signature(factory).parameters:
        state_explorer = factory(config)(states, previous, current, superbasin=None, config=config)
    else:
        state_explorer = factory()(states, previous, current, superbasin=None)
    state_explorer.register_results()


def _load_akmc(family: str):
    root = _python_root(family)
    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if name == "eon" or name.startswith("eon."):
            del sys.modules[name]
    import inspect

    from eon import akmc as akmc_mod
    from eon.config import ConfigClass
    from eon.config import config as shared_config

    parameters = inspect.signature(akmc_mod.get_akmc_metadata).parameters
    if "config" in parameters:
        config = ConfigClass()
    else:
        config = shared_config
        config.init_done = False
    return akmc_mod, config


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
) -> dict:
    del geometries
    family = str(template.get("client_family") or "")
    if family not in _ROOTS:
        raise ValueError(f"{template['id']} needs client_family svn or develop")
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    client = _client(family)
    wrapper = bridge_binary()
    cap = int(template["failure_cap"])
    passes = int(template.get("pass_limit") or _BARRIER_TARGET)
    partial = bool(limit)
    if limit:
        passes = int(limit)
    seed = int((template.get("ini") or {}).get("Main", {}).get("random_seed", 42))
    dest = (work or output.parent / "work" / template["id"].replace("/", "-")).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
        (dest / relative).mkdir(parents=True, exist_ok=True)
    geometry = geometry_archive(root, {"id": problem["id"], "input": {"archive_path": template["geometry"]}})
    shutil.copy(geometry, dest / "pos.con")
    (dest / "config.ini").write_text(_prepare_server(template, client, seed, workers, wrapper), encoding="utf-8")
    server = PotServer(eam_al, cap=cap).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    previous = Path.cwd()
    snapshots = [(0, 0)]
    error = ""
    started = time.monotonic()
    try:
        os.chdir(dest)
        akmc_mod, config = _load_akmc(family)
        config.init()
        from eon import communicator

        if hasattr(communicator.get_communicator, "comm"):
            del communicator.get_communicator.comm
        for index in range(passes):
            try:
                akmc_mod.akmc(config)
                _flush_pending(akmc_mod, config)
            except SystemExit as exc:
                error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                break
            results = dest / "states" / "0" / "search_results.txt"
            text = results.read_text(encoding="utf-8", errors="replace") if results.is_file() else ""
            rows = _searches(text)
            calls, _verify = remote_count(server.host, server.port)
            if snapshots and rows and len(rows) == snapshots[-1][0]:
                print(f"{template['id']} pass {index + 1} reactant idle at {len(rows)} searches", flush=True)
                break
            snapshots.append((len(rows), int(calls)))
            table = dest / "states" / "0" / "processtable"
            reference, missed = _reference_ids(
                table.read_text(encoding="utf-8", errors="replace") if table.is_file() else ""
            )
            harness = _assign(rows, snapshots)
            barriers, rates = _windows(rows, harness, reference, missed)
            print(
                f"{template['id']} pass {index + 1} searches={len(rows)} "
                f"barrier_windows={len(barriers)} rate_windows={len(rates)} calls={calls}",
                flush=True,
            )
            if len(barriers) >= _BARRIER_TARGET and len(rates) >= _RATE_TARGET:
                break
        else:
            pass
    finally:
        os.chdir(previous)
        try:
            calls, _verify = remote_count(server.host, server.port)
        except Exception:
            calls = snapshots[-1][1] if snapshots else 0
        server.stop()
        os.environ.pop("OPTBENCH_POTSERVER", None)
    results = dest / "states" / "0" / "search_results.txt"
    text = results.read_text(encoding="utf-8", errors="replace") if results.is_file() else ""
    rows = _searches(text)
    table = dest / "states" / "0" / "processtable"
    reference, missed = _reference_ids(table.read_text(encoding="utf-8", errors="replace") if table.is_file() else "")
    if len(snapshots) == 1 or snapshots[-1][0] != len(rows):
        snapshots.append((len(rows), int(calls)))
    harness = _assign(rows, snapshots)
    barriers, rates = _windows(rows, harness, reference, missed)
    barrier_records = (
        _records(barriers, template, "barrier")
        or _search_records(rows, harness, template, "barrier")
        or [_whole_record(template, int(calls), "barrier")]
    )
    rate_template = dict(template)
    rate_template["id"] = template.get("rate_template") or template["id"]
    rate_records = (
        _records(rates, template, "rate")
        or _search_records(rows, harness, rate_template, "rate")
        or [_whole_record(template, int(calls), "rate")]
    )
    barrier_metrics = _write(
        output,
        template,
        barrier_records,
        label="barrier",
        windows=barriers,
        total_calls=int(calls),
        searches=len(rows),
        partial=partial,
        cap=cap,
    )
    rate_output = output.with_name(str(template.get("rate_stem") or "htst-rate") + ".dat")
    if template.get("rate_template"):
        _write(
            rate_output,
            rate_template,
            rate_records,
            label="rate",
            windows=rates,
            total_calls=int(calls),
            searches=len(rows),
            partial=partial,
            cap=cap,
        )
    document = {
        "schema": "optbench.htst/stock",
        "error": error,
        "seconds": time.monotonic() - started,
        "searches": len(rows),
        "reference_ids": reference,
        "harness_total": int(calls),
        "barrier_windows": barriers,
        "rate_windows": rates,
        "workers": int(workers),
    }
    output.with_suffix(".windows.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        f"{template['id']} searches={len(rows)} barrier_windows={len(barriers)} "
        f"rate_windows={len(rates)} calls={calls} error={error.splitlines()[:1]}",
        flush=True,
    )
    return {
        "starts": len(barrier_records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": barrier_metrics["par2_mean"],
        "nfailed": barrier_metrics["nfailed"],
        "partial": partial,
        "barrier_windows": len(barriers),
        "rate_windows": len(rates),
        "searches": len(rows),
        "harness_total": int(calls),
        "rate_output": str(rate_output) if template.get("rate_template") else "",
    }
