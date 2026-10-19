"""AKMC catalogues for the aluminum adatom problems.

One repetition stays on the reactant until the old-scheme confidence
of the rate-relevant streak reaches the problem value and the search
count reaches the template floor. A process is rate-relevant when its
rate is at least 1 percent of the current thermal-window rate. A new
process at that share resets the streak. A repeat below it does not
increment the streak. eOn's own confidence
is left above 1 so that call does not take a KMC step off state 0.
The ranked cost is ``eon_search_calls``, the sum of ``sad-fcs`` and
``mins-fcs``. That sum matches ``HarnessCounter`` when the client
reaches the potential through ``ext_pot`` and the prefactor is a
constant. These catalogues keep the built-in ``eam_al`` potential.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np

from .amd import (
    catalogue_accuracy,
    distribution,
    force_calls_per_simulated_time,
    match_product,
    mobile_indices,
    mobile_rmsd,
    old_scheme_confidence,
    parse_con,
    rate_accuracy,
    rate_distribution,
    rate_summary,
    select_thermal,
    simulated_time_seconds,
)
from .campaign import format_ini, geometry_archive, load_template, resolve_problem, write_starts
from .eon_akmc import prepare_client_env
from .score import eon_process_calls, eon_search_calls

COUNT_KEY = "eon_search_calls"
PREFACTOR_COUNT_KEY = "eon_search_calls+pref-fcs"

# A process below this share of the thermal-window rate does not
# advance the old-scheme streak. One percent keeps the 0.17 eV
# exchange and leaves out the 0.43 eV hops once a faster process exists.
RATE_RELEVANT_SHARE = 0.01


def _fmt(value: float) -> str:
    return f"{value:.10g}"


def _search_lines(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines()[2:]:
        fields = line.split()
        if len(fields) >= 6:
            rows.append(fields)
    return rows


def search_cost(text: str, *, include_prefactor: bool = False) -> dict:
    """Sum saddle and minimisation calls, and count repeat lines.

    Prefactor calls are the seventh field. They stay out of the sum
    unless ``include_prefactor`` is set, which is the Vineyard row.
    """
    force_calls = 0
    searches = 0
    repeats = 0
    for fields in _search_lines(text):
        searches += 1
        saddle_calls = int(float(fields[4]))
        minimization_calls = int(float(fields[5]))
        if include_prefactor:
            prefactor_calls = int(float(fields[6])) if len(fields) > 6 else 0
            force_calls += eon_process_calls(saddle_calls, minimization_calls, prefactor_calls)
        else:
            force_calls += eon_search_calls(saddle_calls, minimization_calls)
        if any(token.startswith("repeat-") for token in fields):
            repeats += 1
    return {
        "force_calls": force_calls,
        "total_searches": searches,
        "repeat_lines": repeats,
        "repeats": repeats,
        "confidence": old_scheme_confidence(repeats),
    }


def _registered_event(fields: list[str]) -> tuple[str, int] | None:
    """A ``good-`` or ``repeat-`` id from one search-result row."""
    for token in fields[7:]:
        if token.startswith("good-") or token.startswith("repeat-"):
            kind, _, raw = token.partition("-")
            try:
                return kind, int(raw)
            except ValueError:
                return None
    return None


def _rate_relevant(
    process_id: int,
    known: dict[int, dict],
    temperature: float,
    window: float,
    min_barrier: float,
    share: float,
) -> bool:
    """True when this process carries at least ``share`` of the in-window rate."""
    thermal = select_thermal(list(known.values()), temperature, window, min_barrier)
    total = sum(float(item["rate"]) for item in thermal)
    if total <= 0:
        return False
    proc = known[process_id]
    if not any(item is proc for item in thermal):
        return False
    return float(proc["rate"]) >= share * total


def rate_relevant_repeats(
    text: str,
    processes: list[dict],
    temperature: float,
    window: float,
    min_barrier: float,
    share: float = RATE_RELEVANT_SHARE,
) -> int:
    """Old-scheme streak over processes that affect the escape rate.

    The walk follows the search results. A process enters only when its
    ``good-`` or ``repeat-`` line is reached, so a saddle found later
    does not change the earlier window. Finding a process at or above
    ``share`` resets the streak. Repeating one increments it. A slower
    process does neither.
    """
    if not math.isfinite(share) or share <= 0.0 or share > 1.0:
        raise ValueError("rate share must be in (0, 1]")
    by_id = {int(proc["id"]): proc for proc in processes}
    known: dict[int, dict] = {}
    streak = 0
    for fields in _search_lines(text):
        event = _registered_event(fields)
        if event is None:
            continue
        kind, process_id = event
        proc = by_id.get(process_id)
        if proc is None:
            continue
        if kind == "good" or process_id not in known:
            known[process_id] = proc
            if _rate_relevant(process_id, known, temperature, window, min_barrier, share):
                streak = 0
            continue
        if _rate_relevant(process_id, known, temperature, window, min_barrier, share):
            streak += 1
    return streak


def catalogue_reached(progress: dict, target: float, min_searches: int) -> bool:
    """True when the reactant search is allowed to stop.

    The old-scheme confidence of the rate-relevant streak has to clear
    ``target`` and the thermal-window rate has to be positive. A short
    run can repeat the 0.43 eV hops twenty times before any exchange
    has been found, so the stop also waits for ``min_searches``.
    """
    if min_searches < 0:
        raise ValueError("minimum searches must be non-negative")
    return (
        float(progress.get("confidence") or 0.0) >= target
        and float(progress.get("rate") or 0.0) > 0.0
        and int(progress.get("total_searches") or 0) >= min_searches
    )


def catalogue_confidence(
    text: str,
    processes: list[dict],
    temperature: float,
    window: float,
    min_barrier: float,
    metadata_repeats: int | None,
    *,
    include_prefactor: bool = False,
) -> dict:
    """Confidence that stops the reactant search.

    ``metadata_repeats`` is eOn's counter. It includes every process
    inside the thermal window. The streak does not, and it is the
    count the old-scheme confidence uses.
    """
    progress = search_cost(text, include_prefactor=include_prefactor)
    progress["eon_repeats"] = (
        metadata_repeats if metadata_repeats is not None else progress["repeat_lines"]
    )
    streak = rate_relevant_repeats(text, processes, temperature, window, min_barrier)
    progress["repeats"] = streak
    progress["confidence"] = old_scheme_confidence(streak)
    return progress


def _metadata_repeats(path: Path) -> int | None:
    if not path.is_file():
        return None
    parser = ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except Exception:
        return None
    if not parser.has_option("MetaData", "repeats"):
        return None
    try:
        return parser.getint("MetaData", "repeats")
    except ValueError:
        return None


def _parse_processes(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            rows.append(
                {
                    "id": int(parts[0]),
                    "prefactor": float(parts[2]),
                    "barrier": float(parts[6]),
                    "rate": float(parts[7]),
                }
            )
        except ValueError:
            continue
    return rows


def _write_config(
    template: dict,
    dest: Path,
    client: Path,
    seed: int,
    workers: int,
    temperature: float,
) -> None:
    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["random_seed"] = seed
    main["temperature"] = temperature
    comm = sections.setdefault("Communicator", {})
    comm["client_path"] = str(client.resolve())
    comm["type"] = "local"
    comm["number_of_CPUs"] = workers
    buffered = int(comm.get("num_jobs", workers))
    comm["num_jobs"] = max(workers, buffered)
    comm["jobs_per_bundle"] = comm["num_jobs"]
    # Above 1 the old scheme never reaches the threshold, so akmc()
    # does not step from state 0. The harness stops on its own count.
    akmc = sections.setdefault("AKMC", {})
    akmc["confidence"] = 1.1
    akmc["confidence_scheme"] = "old"
    dest.write_text(format_ini(sections), encoding="utf-8")


def _load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return parse_con(path.read_text(encoding="utf-8", errors="replace"))


def _catalogue_paths(root: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        paths.append(
            geometry_archive(
                root,
                {
                    "id": "accelerated-md",
                    "input": {
                        "archive_path": f"website/accelerated-md/products/product{index}.con"
                    },
                },
            )
        )
    return paths


def _score_products(
    work: Path,
    processes: list[dict],
    reactant: np.ndarray,
    products: list[np.ndarray],
    mobile: np.ndarray,
    *,
    min_barrier: float,
    tolerance: float,
    temperature: float,
    window: float,
) -> dict:
    channels: list[tuple[int | None, float]] = []
    detailed: list[dict] = []
    procdata = work / "states" / "0" / "procdata"
    kept_rate = 0.0
    thermal = select_thermal(processes, temperature, window, min_barrier)
    for process in thermal:
        if process["barrier"] < min_barrier or process["rate"] <= 0:
            continue
        product_path = procdata / f"product_{process['id']}.con"
        channel: int | None
        if not product_path.is_file():
            channel = None
        else:
            points, _fixed = _load_geometry(product_path)
            if points.shape != reactant.shape:
                channel = None
            elif mobile_rmsd(points, reactant, mobile) < tolerance:
                continue
            else:
                channel = match_product(points, products, mobile, tolerance)
        channels.append((channel, process["rate"]))
        kept_rate += process["rate"]
        detailed.append(
            {
                "barrier": float(process["barrier"]),
                "prefactor": float(process.get("prefactor") or 0.0),
                "rate": float(process["rate"]),
                "channel": channel,
            }
        )
    return {"channels": channels, "rate": kept_rate, "prefactors": detailed}


def _read_progress(
    work: Path,
    temperature: float,
    window: float,
    min_barrier: float,
    tolerance: float,
    reactant: np.ndarray,
    products: list[np.ndarray],
    mobile: np.ndarray,
    *,
    include_prefactor: bool = False,
) -> dict:
    """Confidence, rate, and force calls from the reactant state on disk."""
    results = work / "states" / "0" / "search_results.txt"
    text = results.read_text(encoding="utf-8", errors="replace") if results.is_file() else ""
    table = work / "states" / "0" / "processtable"
    processes = (
        _parse_processes(table.read_text(encoding="utf-8", errors="replace"))
        if table.is_file()
        else []
    )
    progress = catalogue_confidence(
        text,
        processes,
        temperature,
        window,
        min_barrier,
        _metadata_repeats(work / "states" / "0" / "info"),
        include_prefactor=include_prefactor,
    )
    progress.update(
        _score_products(
            work,
            processes,
            reactant,
            products,
            mobile,
            min_barrier=min_barrier,
            tolerance=tolerance,
            temperature=temperature,
            window=window,
        )
    )
    return progress


def _client_tail(work: Path) -> str:
    jobs = work / "jobs"
    if not jobs.is_dir():
        return ""
    for stderr in sorted(jobs.rglob("stderr.dat")):
        try:
            text = stderr.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        tail = "\n".join(text.splitlines()[-6:])
        return f"{stderr.parent.name}: {tail}"
    return ""


def _one_repetition(
    root: Path,
    template: dict,
    problem: dict,
    client: Path,
    work: Path,
    seed: int,
    workers: int,
    products: list[np.ndarray],
    reactant: np.ndarray,
    mobile: np.ndarray,
    *,
    passes: int,
    on_pass=None,
) -> dict:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    reactant_path = geometry_archive(
        root,
        {
            "id": problem["id"],
            "input": {"archive_path": problem["input"]["reactant"]},
        },
    )
    shutil.copy(reactant_path, work / "pos.con")
    temperature = float(problem["thermal"]["temperature_K"])
    _write_config(template, work / "config.ini", client, seed, workers, temperature)
    prepare_client_env(client)
    target = float((problem.get("convergence") or {}).get("confidence", 0.95))
    min_searches = int(template.get("confidence_min_searches", 0))
    include_prefactor = bool(template.get("count_prefactor"))
    min_barrier = float((problem.get("convergence") or {}).get("min_barrier_eV", 0.01))
    tolerance = float((problem.get("convergence") or {}).get("catalogue_rmsd", 0.15))
    window = float(
        (problem.get("convergence") or {}).get(
            "thermally_accessible_window",
            ((template.get("ini") or {}).get("AKMC") or {}).get("thermally_accessible_window", 20),
        )
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    previous = Path.cwd()
    progress = search_cost("")
    last_error = ""
    started = time.monotonic()
    try:
        os.chdir(work)
        from eon.akmc import akmc
        from eon.config import ConfigClass

        try:
            config = ConfigClass()
            config.init()
        except SystemExit as exc:
            last_error = f"config exit {exc.code}"
            progress["error"] = last_error
            progress["seed"] = seed
            progress["passes"] = 0
            progress["seconds"] = time.monotonic() - started
            progress["rate"] = 0.0
            progress["channels"] = []
            progress["reached"] = False
            return progress
        for index in range(passes):
            try:
                akmc(config)
            except SystemExit as exc:
                last_error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            progress = _read_progress(
                work,
                temperature,
                window,
                min_barrier,
                tolerance,
                reactant,
                products,
                mobile,
                include_prefactor=include_prefactor,
            )
            progress["passes"] = index + 1
            print(
                f"seed {seed} pass {index + 1} confidence={progress['confidence']:.3f} "
                f"repeats={progress['repeats']} eon_repeats={progress.get('eon_repeats')} "
                f"rate={progress['rate']:.6g} "
                f"searches={progress['total_searches']} force_calls={progress['force_calls']}",
                flush=True,
            )
            if on_pass is not None:
                on_pass(progress)
            if catalogue_reached(progress, target, min_searches):
                progress["reached"] = True
                break
        else:
            progress["passes"] = passes
            progress["reached"] = False
        # The call that submits a batch does not register it. A last
        # call with an empty buffer records the finished searches and
        # does not start another batch.
        if not progress.get("reached"):
            config.comm_job_buffer_size = 0
            try:
                akmc(config)
            except SystemExit as exc:
                last_error = last_error or f"akmc exit {exc.code}"
            except Exception as exc:
                last_error = last_error or f"{type(exc).__name__}: {exc}"
            done = int(progress.get("passes") or passes)
            progress = _read_progress(
                work,
                temperature,
                window,
                min_barrier,
                tolerance,
                reactant,
                products,
                mobile,
                include_prefactor=include_prefactor,
            )
            progress["passes"] = done
            progress["reached"] = catalogue_reached(progress, target, min_searches)
            print(
                f"seed {seed} flush confidence={progress['confidence']:.3f} "
                f"repeats={progress['repeats']} eon_repeats={progress.get('eon_repeats')} "
                f"rate={progress['rate']:.6g} "
                f"searches={progress['total_searches']} force_calls={progress['force_calls']}",
                flush=True,
            )
            if on_pass is not None:
                on_pass(progress)
    finally:
        os.chdir(previous)
    progress["seed"] = seed
    progress["error"] = last_error
    progress.setdefault("passes", passes)
    progress.setdefault("reached", False)
    progress.setdefault("rate", 0.0)
    progress.setdefault("channels", [])
    progress["seconds"] = time.monotonic() - started
    if not progress.get("total_searches") and not progress.get("error"):
        tail = _client_tail(work)
        if tail:
            progress["error"] = tail
    jobs = work / "jobs"
    if jobs.exists() and not os.environ.get("OPTBENCH_KEEP_JOBS"):
        shutil.rmtree(jobs, ignore_errors=True)
    return progress


def _archive_rep(work: Path, output: Path, repetition: int) -> None:
    """Keep the state files that the score was read from."""
    state = work / "states" / "0"
    names = ("search_results.txt", "processtable", "info")
    for name in names:
        src = state / name
        if not src.is_file():
            continue
        dest = output.with_name(f"{output.stem}.rep{repetition}.{name}")
        dest.write_bytes(src.read_bytes())


def _failure_class(row: dict) -> str | None:
    success = bool(row.get("reached")) and float(row.get("rate") or 0) > 0
    if success:
        return None
    if row.get("error"):
        return "crash"
    return "divergence"


def _write_result(
    template: dict,
    problem: dict,
    rows: list[dict],
    output: Path,
    workers: int,
    *,
    in_progress: bool = False,
) -> dict:
    reference = float(problem["reference"]["true_rate"])
    probabilities = distribution(problem["catalogue"]["probabilities"])
    products = int(problem["sample"]["products"])
    kept = [row for row in rows if row.get("reached") and row.get("rate", 0) > 0]
    failed = len(rows) - len(kept)
    lines = []
    for row in rows:
        lines.append(
            f"# rep {row['repetition']} force_calls {row.get('force_calls', 0)} "
            f"total_searches {row.get('total_searches', 0)} "
            f"repeats {int(row.get('repeats') or 0)} "
            f"eon_repeats {int(row.get('eon_repeats') or 0)} "
            f"rate {_fmt(float(row.get('rate') or 0.0))} "
            f"confidence {_fmt(float(row.get('confidence') or 0.0))} "
            f"reached {int(bool(row.get('reached')))}"
        )
    summary: dict = {
        "true_rate": None,
        "uncertainty": None,
        "percent_uncertainty": None,
        "rate_accuracy": None,
        "catalogue_accuracy": None,
        "force_calls_per_simulated_time": None,
        "force_calls": None,
        "nfailed": failed,
        "starts": len(rows),
    }
    if kept:
        rates = rate_summary([float(row["rate"]) for row in kept])
        summary["true_rate"] = rates["true_rate"]
        summary["uncertainty"] = rates["uncertainty"]
        summary["percent_uncertainty"] = rates["percent_uncertainty"]
        summary["rate_accuracy"] = rate_accuracy(float(rates["true_rate"]), reference)
        accuracies = [
            catalogue_accuracy(rate_distribution(row["channels"]), probabilities) for row in kept
        ]
        summary["catalogue_accuracy"] = sum(accuracies) / len(accuracies)
        calls = sum(int(row["force_calls"]) for row in kept)
        seconds = sum(simulated_time_seconds(float(row["rate"]), products) for row in kept)
        summary["force_calls"] = calls / len(kept)
        summary["force_calls_per_simulated_time"] = force_calls_per_simulated_time(calls, seconds)
    lines.append(f"code {template.get('code', 'eOn')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    count_key = PREFACTOR_COUNT_KEY if template.get("count_prefactor") else COUNT_KEY
    lines.append("counter harness")
    lines.append(f"count_keys {count_key}")
    lines.append("contributor Rohit Goswami")
    lines.append(f"repetitions {len(rows)}")
    lines.append(f"workers {workers}")
    lines.append(f"temperature_K {problem['thermal']['temperature_K']}")
    lines.append("thermostat akmc")
    lines.append(f"reference_rate {_fmt(reference)}")
    lines.append(f"reference_products {products}")
    if summary["true_rate"] is not None:
        lines.append(f"true_rate {_fmt(float(summary['true_rate']))}")
    if summary["uncertainty"] is not None:
        lines.append(f"uncertainty {_fmt(float(summary['uncertainty']))}")
    if summary["percent_uncertainty"] is not None:
        lines.append(f"percent_uncertainty {_fmt(float(summary['percent_uncertainty']))}")
    for key in (
        "rate_accuracy",
        "catalogue_accuracy",
        "force_calls_per_simulated_time",
        "force_calls",
    ):
        value = summary[key]
        if value is None:
            continue
        lines.append(f"{key} {_fmt(float(value))}")
    lines.append(f"nfailed {failed}")
    if template.get("count_prefactor"):
        for row in rows:
            for item in row.get("prefactors") or []:
                channel = item.get("channel")
                lines.append(
                    "# vineyard "
                    f"rep {row.get('repetition')} "
                    f"barrier {_fmt(float(item['barrier']))} "
                    f"prefactor {_fmt(float(item['prefactor']))} "
                    f"rate {_fmt(float(item['rate']))} "
                    f"channel {channel}"
                )
    if template.get("algorithm"):
        lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append("reducer amd-catalogue")
    if in_progress or len(rows) < int(problem["sample"]["repetitions"]):
        lines.append("partial true")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    start_rows = []
    for row in rows:
        success = _failure_class(row) is None
        record = {
            "schema": "optbench.start/v2",
            "start": f"rep-{row['repetition']}",
            "geometry": f"rep-{row['repetition']}",
            "force_calls": int(row.get("force_calls") or 0),
            "count_key": count_key,
            "counter": "harness",
            "comparable": True,
            "success": success,
            "failure": _failure_class(row),
            "seed": row.get("seed"),
            "rate": row.get("rate"),
            "confidence": row.get("confidence"),
            "repeats": row.get("repeats"),
            "eon_repeats": row.get("eon_repeats"),
            "total_searches": row.get("total_searches"),
            "channels": [
                [None if channel is None else int(channel), float(rate)]
                for channel, rate in (row.get("channels") or [])
            ],
            "prefactors": [
                {
                    "barrier": float(item["barrier"]),
                    "prefactor": float(item["prefactor"]),
                    "rate": float(item["rate"]),
                    "channel": None if item["channel"] is None else int(item["channel"]),
                }
                for item in (row.get("prefactors") or [])
            ],
            "seconds": row.get("seconds"),
            "problem": problem["id"],
            "template": template["id"],
        }
        if row.get("error"):
            record["error"] = row["error"]
        start_rows.append(record)
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in start_rows)
    jsonl = output.with_suffix(".starts.jsonl")
    jsonl.write_text(body, encoding="utf-8")
    write_starts(output.with_suffix(".starts"), start_rows)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    return summary


def run_amd(
    root: Path,
    template_name: str,
    client: Path,
    output: Path,
    work: Path,
    workers: int,
    repetitions: int | None = None,
) -> dict:
    """Run AKMC catalogues and write benchmark.dat after each one."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    template = load_template(root, template_name)
    problem = resolve_problem(root, str(template["problem"]))
    if repetitions is None:
        repetitions = int(problem["sample"]["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    passes = int(template.get("pass_limit", 200))
    reactant_path = geometry_archive(
        root,
        {"id": problem["id"], "input": {"archive_path": problem["input"]["reactant"]}},
    )
    reactant, fixed = _load_geometry(reactant_path)
    mobile = mobile_indices(fixed)
    catalogue = [_load_geometry(path)[0] for path in _catalogue_paths(root, 24)]
    rows: list[dict] = []
    summary: dict = {}
    for index in range(repetitions):
        repetition = index + 1
        rep_dir = work / f"rep-{repetition}"

        def _checkpoint(progress: dict, repetition: int = repetition) -> None:
            progress["repetition"] = repetition
            _write_result(
                template,
                problem,
                rows + [progress],
                output,
                workers,
                in_progress=True,
            )

        row = _one_repetition(
            root,
            template,
            problem,
            client,
            rep_dir,
            20261018 + index,
            workers,
            catalogue,
            reactant,
            mobile,
            passes=passes,
            on_pass=_checkpoint,
        )
        row["repetition"] = repetition
        rows.append(row)
        print(
            f"rep {repetition} reached={row.get('reached')} rate={row.get('rate')} "
            f"force_calls={row.get('force_calls')} error={row.get('error', '')}",
            flush=True,
        )
        _archive_rep(rep_dir, output, repetition)
        summary = _write_result(template, problem, rows, output, workers)
        if rep_dir.exists():
            shutil.rmtree(rep_dir, ignore_errors=True)
    return summary
