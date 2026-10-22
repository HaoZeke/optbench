"""AKMC saddle campaign on the Pt island.

One repetition runs the eOn server until ``states/0/search_results.txt``
contains 27 good saddles below 1.5 eV, or the pass limit is hit. The
force calls are ``sad-fcs + mins-fcs`` through that line. That is the
client counter the 2014 ``done.py`` summed. It is not a harness wrap.
Prefactor calls stay out of the sum.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from .campaign import format_ini, geometry_archive, load_template, write_starts
from .eon_parse import saddle_progress


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_error(values: list[float]) -> float:
    """Population standard deviation divided by sqrt(n), as in report.sh."""
    count = len(values)
    center = _mean(values)
    second = sum(value * value for value in values) / count
    variance = second - center * center
    if variance < 0:
        variance = 0.0
    return (variance ** 0.5) / (count ** 0.5)


def _fmt(value: float) -> str:
    return f"{value:.10g}"


def prepare_client_env(client: Path) -> None:
    """Put the installed eOn libraries on the path this process already uses.

    The server imports the Python package. The local communicator then
    spawns ``eonclient``, which needs ``librgpot`` and ``libeonclib``.
    """
    root = client.resolve().parent.parent
    lib = root / "lib"
    py_pkg = lib / "python3" / "dist-packages"
    py_deps = root / "py"
    entries = [str(py_pkg), str(py_deps)]
    if lib.is_dir():
        entries.extend(str(path) for path in sorted(lib.glob("python*/site-packages")))
    for entry in reversed(entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ["PATH"] = str(client.resolve().parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    pythonpath = os.pathsep.join(entries)
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = pythonpath if not previous else pythonpath + os.pathsep + previous


def _write_config(template: dict, dest: Path, client: Path, seed: int, workers: int) -> None:
    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["random_seed"] = seed
    comm = sections.setdefault("Communicator", {})
    comm["client_path"] = str(client.resolve())
    comm["type"] = "local"
    comm["number_of_CPUs"] = workers
    # The template buffer can be larger than the CPU count. One server
    # pass then finishes that many searches, eight at a time.
    buffered = int(comm.get("num_jobs", workers))
    comm["num_jobs"] = max(workers, buffered)
    dest.write_text(format_ini(sections), encoding="utf-8")


def _harness_total(prior_calls: int) -> int | None:
    """Calls on this process's server plus the calls stored before a resume.

    None means the server did not answer. Callers leave the checkpoint
    alone in that case, so a dead server cannot replace a real total with 0.
    """
    endpoint = os.environ.get("OPTBENCH_POTSERVER", "")
    if ":" not in endpoint:
        return int(prior_calls)
    host, port = endpoint.rsplit(":", 1)
    try:
        from .potserver import remote_count

        calls, _verify = remote_count(host, int(port))
    except Exception:
        return None
    return int(prior_calls) + int(calls)


def _remember_calls(work: Path, total: int | None) -> None:
    if total is None:
        return
    if int(total) < stored_calls(work):
        return
    _write_call_total(work, int(total))


def _write_call_total(work: Path, total: int) -> None:
    path = work / "harness-calls.txt"
    # The search loop and the checkpoint thread both write this file.
    # A shared temporary name makes one replace delete the other's file.
    temporary = path.with_name(f".harness-calls.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(f"{int(total)}\n", encoding="utf-8")
    temporary.replace(path)


def stored_calls(work: Path) -> int:
    """Harness calls already spent in this work directory."""
    path = work / "harness-calls.txt"
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _left_initial_state(work: Path) -> bool:
    states = work / "states"
    if not states.is_dir():
        return False
    for path in states.iterdir():
        if path.is_dir() and path.name.isdigit() and int(path.name) > 0:
            return True
    return False


def _one_repetition(
    root: Path,
    template: dict,
    client: Path,
    work: Path,
    seed: int,
    workers: int,
    *,
    passes: int,
    geometry: Path | None = None,
    cap: int | None = None,
    resume: bool = False,
    prior_calls: int = 0,
) -> dict:
    if work.exists() and not resume:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if resume:
        # Scratch from a killed pass is not part of the state list.
        for name in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete"):
            stale = work / name
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
    if geometry is None:
        pos = geometry_archive(
            root,
            {
                "id": "saddle",
                "input": {"archive_path": "website/saddle-search/pt-island/pt-island.con"},
            },
        )
    else:
        pos = Path(geometry)
    shutil.copy(pos, work / "pos.con")
    _write_config(template, work / "config.ini", client, seed, workers)
    prepare_client_env(client)
    for name in ("akmc", "eon", "state", "communicator"):
        logging.getLogger(name).setLevel(logging.WARNING)
    previous = Path.cwd()
    progress = saddle_progress("")
    last_error = ""
    started = time.monotonic()
    capped = False
    stop_poll = threading.Event()

    def _poll_calls() -> None:
        while not stop_poll.wait(15.0):
            try:
                _remember_calls(work, _harness_total(prior_calls))
            except Exception:
                continue

    poller = threading.Thread(target=_poll_calls, name="optbench-harness-calls", daemon=True)
    poller.start()
    for relative in (
        "jobs/scratch",
        "jobs/in",
        "jobs/out",
        "jobs/incomplete",
        "states",
        "potfiles",
    ):
        (work / relative).mkdir(parents=True, exist_ok=True)
    # r2549 creates states/state_table only when it creates the states
    # directory. This harness creates that directory first, so the KMC
    # step opens a missing file once confidence is reached. An empty
    # table is the file the server would have written.
    table = work / "states" / "state_table"
    if not table.exists():
        table.write_text("", encoding="utf-8")
    try:
        os.chdir(work)
        import inspect

        try:
            from eon import akmc as akmc_mod
            from eon.config import ConfigClass
            from eon.config import config as shared_config
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            progress["error"] = last_error
            progress["seed"] = seed
            progress["passes"] = 0
            progress["seconds"] = time.monotonic() - started
            return progress

        try:
            # r2549 helpers read the module-level config object. 3.5.0
            # passes the instance into those helpers, so a fresh object
            # still sees the ini.
            parameters = inspect.signature(akmc_mod.get_akmc_metadata).parameters
            if "config" in parameters:
                config = ConfigClass()
            else:
                config = shared_config
                config.init_done = False
            config.init()
            # get_communicator caches the first Local object. A second
            # seed in this process would keep the old scratch path.
            from eon import communicator

            if hasattr(communicator.get_communicator, "comm"):
                del communicator.get_communicator.comm
        except SystemExit as exc:
            last_error = f"config exit {exc.code}"
            progress["error"] = last_error
            progress["seed"] = seed
            progress["passes"] = 0
            progress["seconds"] = time.monotonic() - started
            return progress
        for index in range(passes):
            spent = _harness_total(prior_calls)
            if spent is None:
                spent = stored_calls(work)
            _remember_calls(work, spent)
            if cap is not None and spent >= cap:
                capped = True
                progress["passes"] = index
                break
            if _left_initial_state(work):
                last_error = "left the initial state"
                progress["passes"] = index
                break
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                last_error = f"akmc exit {exc.code}"
                progress["passes"] = index + 1
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                progress["passes"] = index + 1
                break
            results = work / "states" / "0" / "search_results.txt"
            if results.is_file():
                progress = saddle_progress(results.read_text(encoding="utf-8", errors="replace"))
            spent = _harness_total(prior_calls)
            if spent is None:
                spent = stored_calls(work)
            progress["harness_calls"] = spent
            _remember_calls(work, spent)
            if (index + 1) % 25 == 0 or progress["reached"] or (cap is not None and spent >= cap):
                print(
                    f"seed {seed} pass {index + 1} qualifying={progress['qualifying']} "
                    f"searches={progress['total_searches']} harness_calls={spent}",
                    flush=True,
                )
            if progress["reached"]:
                progress["passes"] = index + 1
                break
            if cap is not None and spent >= cap:
                capped = True
                progress["passes"] = index + 1
                break
            if _left_initial_state(work):
                last_error = "left the initial state"
                progress["passes"] = index + 1
                break
        else:
            progress["passes"] = passes
        # akmc() registers the previous pass, then submits this one.
        # A last call with an empty buffer records the searches that
        # already finished and does not start another batch.
        if not progress.get("reached") and not capped and not _left_initial_state(work):
            config.comm_job_buffer_size = 0
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                last_error = last_error or f"akmc exit {exc.code}"
            except Exception as exc:
                last_error = last_error or f"{type(exc).__name__}: {exc}"
            results = work / "states" / "0" / "search_results.txt"
            if results.is_file():
                progress = saddle_progress(results.read_text(encoding="utf-8", errors="replace"))
                progress["passes"] = passes
    finally:
        stop_poll.set()
        try:
            _remember_calls(work, _harness_total(prior_calls))
        except Exception:
            pass
        os.chdir(previous)
    progress["seed"] = seed
    progress["error"] = last_error
    final_calls = _harness_total(prior_calls)
    if final_calls is None:
        final_calls = stored_calls(work)
    progress["capped"] = bool(capped) or (cap is not None and final_calls >= cap)
    progress["harness_calls"] = final_calls
    _remember_calls(work, final_calls)
    progress.setdefault("passes", passes)
    progress["seconds"] = time.monotonic() - started
    jobs = work / "jobs"
    if jobs.exists():
        shutil.rmtree(jobs, ignore_errors=True)
    return progress


def _summary(rows: list[dict], failed: int) -> dict:
    kept = [row for row in rows if row["reached"]]
    if not kept:
        return {
            "force_calls": None,
            "force_calls_mean_error": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "unique_saddles": None,
            "fraction_connected": None,
            "total_searches": None,
            "nfailed": failed,
        }
    calls = [float(row["force_calls"]) for row in kept]
    unique = [float(row["unique_saddles"]) for row in kept]
    fractions = [float(row["fraction_connected"]) for row in kept]
    searches = [float(row["total_searches"]) for row in kept]
    return {
        "force_calls": _mean(calls),
        "force_calls_mean_error": _mean_error(calls),
        "force_calls_min": int(min(calls)),
        "force_calls_max": int(max(calls)),
        "unique_saddles": _mean(unique),
        "fraction_connected": _mean(fractions),
        "total_searches": _mean(searches),
        "nfailed": failed,
    }


def _write_saddle(template: dict, rows: list[dict], output: Path, workers: int) -> dict:
    failed = sum(1 for row in rows if not row["reached"])
    summary = _summary(rows, failed)
    lines = []
    for row in rows:
        fraction = row["fraction_connected"]
        fraction_text = "na" if fraction is None else _fmt(fraction)
        lines.append(
            f"# rep {row['repetition']} force_calls {row['force_calls']} "
            f"unique_saddles {row['unique_saddles']} "
            f"fraction_connected {fraction_text} "
            f"total_searches {row['total_searches']} "
            f"reached {int(row['reached'])}"
        )
    lines.append(f"code {template.get('code', 'eOn')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append("counter client-search-results")
    lines.append("count_keys sad-fcs+mins-fcs")
    lines.append("contributor Rohit Goswami")
    lines.append(f"repetitions {len(rows)}")
    lines.append(f"min_mode_method {template['ini']['Saddle Search']['min_mode_method']}")
    dimer = template["ini"].get("Dimer") or {}
    if "improved" in dimer:
        lines.append(f"dimer_improved {str(dimer['improved']).lower()}")
    lines.append(f"communicator_jobs {workers}")
    for key in (
        "force_calls",
        "force_calls_mean_error",
        "force_calls_min",
        "force_calls_max",
        "unique_saddles",
        "fraction_connected",
        "total_searches",
        "nfailed",
    ):
        value = summary[key]
        if value is None:
            continue
        if key in ("force_calls_min", "force_calls_max", "nfailed"):
            lines.append(f"{key} {int(value)}")
        else:
            lines.append(f"{key} {_fmt(float(value))}")
    if template.get("algorithm"):
        lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {template.get('problem', '')}")
    lines.append(f"reducer {template.get('reducer', 'akmc-until-27')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    start_rows = []
    for row in rows:
        start_rows.append(
            {
                "failed": not row["reached"],
                "force_calls": row["force_calls"],
                "fraction_connected": row["fraction_connected"],
                "geometry": f"rep-{row['repetition']}",
                "passes": row.get("passes"),
                "qualifying": row["qualifying"],
                "reached": row["reached"],
                "reason": "0" if row["reached"] else "incomplete",
                "seconds": row.get("seconds"),
                "seed": row["seed"],
                "total_searches": row["total_searches"],
                "unique_saddles": row["unique_saddles"],
            }
        )
    write_starts(output.with_suffix(".starts"), start_rows)
    summary["starts"] = len(rows)
    summary["template"] = template["id"]
    summary["problem"] = template.get("problem")
    return summary


def run_saddle(
    root: Path,
    template_name: str,
    client: Path,
    output: Path,
    work: Path,
    workers: int,
    repetitions: int = 10,
) -> dict:
    """Repeat the AKMC search and write benchmark.dat.

    A repetition that does not reach 27 qualifying saddles stays out of
    the mean and counts as failed. The dat is rewritten after every
    finished repetition, and ``repetitions`` is the number finished,
    not a requested count that never ran.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    template = load_template(root, template_name)
    passes = int(template.get("pass_limit", 20000))
    rows = []
    summary: dict = {}
    for index in range(repetitions):
        rep_work = work / f"rep-{index + 1}"
        row = _one_repetition(
            root,
            template,
            client,
            rep_work,
            1000 + index,
            workers,
            passes=passes,
        )
        row["repetition"] = index + 1
        rows.append(row)
        print(
            f"rep {index + 1} reached={row['reached']} "
            f"force_calls={row['force_calls']} searches={row['total_searches']} "
            f"error={row.get('error', '')}",
            flush=True,
        )
        summary = _write_saddle(template, rows, output, workers)
    return summary


def _insert_pythonpath(pythonpath: str) -> None:
    entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    for entry in reversed(entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if not entries:
        return
    joined = os.pathsep.join(entries)
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = joined if not previous else joined + os.pathsep + previous


def _limit_threads() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def _stock_record(template: dict, repetition: int, seed: int, progress: dict) -> dict:
    from .eon_harness import classify_akmc

    cap = int(template["failure_cap"])
    calls = int(progress.get("harness_calls") or 0)
    error = str(progress.get("error") or "")
    left = "left the initial state" in error
    crashed = left or (bool(error) and not progress.get("total_searches") and not progress.get("capped"))
    record = classify_akmc(
        calls=calls,
        cap=cap,
        reached=bool(progress.get("reached")) and not left,
        crashed=crashed,
    )
    record.update(
        {
            "call_unit": "total",
            "client_calls": progress.get("force_calls"),
            "count_key": "harness",
            "counter": "harness",
            "error": error,
            "fraction_connected": progress.get("fraction_connected"),
            "geometry": f"run-{repetition}",
            "passes": progress.get("passes"),
            "qualifying": progress.get("qualifying"),
            "raw": calls,
            "repetition": repetition,
            "seconds": progress.get("seconds"),
            "seed": seed,
            "template": template["id"],
            "total_searches": progress.get("total_searches"),
            "unique_saddles": progress.get("unique_saddles"),
        }
    )
    # A config or import failure did not spend a search. The next launch retries it.
    record["finished"] = not (
        calls == 0 and not progress.get("total_searches") and not progress.get("reached") and bool(error)
    )
    return record


def _stock_worker(spec: dict) -> dict:
    """One archive run: seed ``1000 + run - 1``, no KMC step, stop at 27 or the cap."""
    _limit_threads()
    _insert_pythonpath(str(spec.get("pythonpath") or ""))
    from .campaign import load_template
    import subprocess

    from .eon_harness import bridge_binary, fast_server_binary, preload_library

    root = Path(spec["root"])
    template = load_template(root, spec["template"])
    work = Path(spec["work"])
    repetition = int(spec["repetition"])
    seed = int(spec["seed"])
    result_path = work / "optbench-result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("finished") and int(existing.get("seed", -1)) == seed:
            print(f"run {repetition} seed {seed} already finished", flush=True)
            return existing
    cap = int(template["failure_cap"])
    passes = int(template.get("pass_limit", cap))
    resume = (work / "pos.con").is_file() and (work / "states").is_dir()
    prior = stored_calls(work) if resume else 0
    patched = dict(template)
    patched["ini"] = json.loads(json.dumps(template["ini"]))
    patched["ini"]["Potential"] = {
        "potential": "ext_pot",
        "ext_pot_path": str(bridge_binary()),
    }
    # The Morse kernel is the compiled pair sum. A C server charges one
    # harness call per request and skips the Python parser. The preload
    # answers the client's system() in-process so the bridge is not forked.
    from .surfaces import morse_library_file

    server = subprocess.Popen(
        [str(fast_server_binary()), "--cap", str(cap), "--kernel", str(morse_library_file())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert server.stdout is not None
    endpoint = server.stdout.readline().strip()
    if not endpoint or server.poll() is not None:
        error = server.stderr.read() if server.stderr is not None else ""
        raise RuntimeError(f"extpot server failed to start: {error}")
    os.environ["OPTBENCH_POTSERVER"] = endpoint
    preload = str(preload_library())
    previous_preload = os.environ.get("LD_PRELOAD", "")
    if preload not in previous_preload.split(":"):
        os.environ["LD_PRELOAD"] = preload if not previous_preload else preload + ":" + previous_preload
    try:
        progress = _one_repetition(
            root,
            patched,
            Path(spec["client"]),
            work,
            seed,
            1,
            passes=passes,
            cap=cap,
            resume=resume,
            prior_calls=prior,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=2)
        os.environ.pop("OPTBENCH_POTSERVER", None)
    record = _stock_record(template, repetition, seed, progress)
    if record["finished"]:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(result_path)
    print(
        f"run {repetition} seed {seed} success={record['success']} failure={record['failure']} "
        f"force_calls={record['force_calls']} qualifying={record.get('qualifying')} "
        f"error={record.get('error', '')}",
        flush=True,
    )
    return record


def _write_stock(template: dict, rows: list[dict], output: Path, *, scored: int) -> None:
    from .campaign import benchmark_text

    ordered = sorted(rows, key=lambda row: int(row.get("repetition") or 0))
    for row in ordered:
        row["in_mean"] = int(row.get("repetition") or 0) <= scored
    kept = [row for row in ordered if row["in_mean"]]
    finished_ok = [row for row in kept if row.get("success")]
    failed = len(kept) - len(finished_ok)
    summary: dict = {"call_unit": "total", "nfailed": failed}
    if finished_ok:
        calls = [float(row["force_calls"]) for row in finished_ok]
        summary["force_calls"] = _mean(calls)
        summary["force_calls_mean_error"] = _mean_error(calls)
        summary["force_calls_min"] = int(min(calls))
        summary["force_calls_max"] = int(max(calls))
        summary["unique_saddles"] = _mean([float(row["unique_saddles"]) for row in finished_ok])
        summary["fraction_connected"] = _mean([float(row["fraction_connected"]) for row in finished_ok])
        summary["total_searches"] = _mean([float(row["total_searches"]) for row in finished_ok])
    text = benchmark_text(summary, template)
    metrics = []
    for key in (
        "force_calls_mean_error",
        "unique_saddles",
        "fraction_connected",
        "total_searches",
    ):
        value = summary.get(key)
        if value is None:
            continue
        metrics.append(f"{key} {_fmt(float(value))}")
    extra = [
        *metrics,
        "counter harness",
        "count_keys HarnessCounter",
        "tuning stock",
        "potential morse/smooth/linear",
        "scored_runs 1-9",
        "launched_runs 1-10",
    ]
    comments = [
        (
            f"# run {int(row['repetition'])} seed {row.get('seed')} "
            f"force_calls {int(float(row['force_calls']))} "
            f"qualifying {row.get('qualifying')} total_searches {row.get('total_searches')}"
        )
        for row in ordered
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(comments) + "\n" + text + "\n".join(extra) + "\n", encoding="utf-8")
    write_starts(output.with_suffix(".starts"), ordered)


def run_stock_shard(
    root: Path,
    template_name: str,
    client: Path,
    output: Path,
    work: Path,
    *,
    pythonpath: str = "",
    slots: int = 8,
    launched: int = 10,
    scored: int = 9,
) -> dict:
    """Archive runs 1..``launched`` on seeds 1000.., scored through run ``scored``.

    ``report.sh`` globs ``run-?`` and therefore keeps runs 1 through 9.
    Run 10 is launched with seed 1009 and left out of the mean. Each run
    is one process, checkpointed under ``work``, and resumed from the
    stored harness count.
    """
    if launched <= 0 or scored <= 0 or scored > launched:
        raise ValueError("scored runs must sit inside the launched runs")
    if slots <= 0:
        raise ValueError("slots must be positive")
    _limit_threads()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    work.mkdir(parents=True, exist_ok=True)
    pending = []
    done: dict[int, dict] = {}
    for repetition in range(1, launched + 1):
        rep_work = work / f"run-{repetition}"
        result_path = rep_work / "optbench-result.json"
        seed = 999 + repetition
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("finished") and int(existing.get("seed", -1)) == seed:
                done[repetition] = existing
                continue
        pending.append(
            {
                "root": str(root),
                "template": template_name,
                "client": str(client),
                "work": str(rep_work),
                "repetition": repetition,
                "seed": seed,
                "pythonpath": pythonpath,
            }
        )
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(slots, len(pending)), mp_context=context) as pool:
            futures = {pool.submit(_stock_worker, spec): int(spec["repetition"]) for spec in pending}
            for future in as_completed(futures):
                repetition = futures[future]
                try:
                    done[repetition] = future.result()
                except Exception as exc:
                    # A killed worker must not discard the other runs.
                    # The checkpoint stays, and the next launch resumes it.
                    print(f"run {repetition} failed: {type(exc).__name__}: {exc}", flush=True)
    finished = [
        done[index] for index in range(1, launched + 1) if index in done and done[index].get("finished")
    ]
    scored_done = [row for row in finished if int(row.get("repetition") or 0) <= scored]
    if len(scored_done) == scored and len(finished) == launched:
        _write_stock(load_template(root, template_name), finished, output, scored=scored)
    missing = [index for index in range(1, launched + 1) if index not in done or not done[index].get("finished")]
    return {
        "scored": len(scored_done),
        "launched": launched,
        "missing": missing,
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Shard the Pt-island stock saddle search.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--template", required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--pythonpath", default="")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--launched", type=int, default=10)
    parser.add_argument("--scored", type=int, default=9)
    args = parser.parse_args(argv)
    result = run_stock_shard(
        args.root,
        args.template,
        args.client,
        args.output,
        args.work,
        pythonpath=args.pythonpath,
        slots=args.slots,
        launched=args.launched,
        scored=args.scored,
    )
    print(result, flush=True)
    return 0 if not result["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
