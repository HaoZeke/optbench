"""AKMC hTST campaign reduced by :mod:`optbench.htst`.

One run stays on ``states/0``. A start is one completed barrier window.
The run stops when the barrier windows reach ``windows`` and the rate
windows reach ``rate_windows``. The written means keep only those first
windows, which are the archive's sample sizes. Force calls are
``sad-fcs`` through the reducer. eOn's own potential supplies the
energy, so the counter is ``htst-reducer``.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from .campaign import format_ini, geometry_archive, load_template, resolve_problem, write_starts
from .eon_akmc import prepare_client_env
from .htst import (
    HtstReduction,
    barrier_start_records,
    metric_lines,
    reduce_tables,
    spec_from_problem,
)


def _prepare_client(client: Path) -> None:
    """Load an installed eOn tree. Debian and a custom prefix use different Python dirs."""
    prepare_client_env(client)
    root = client.resolve().parent.parent
    lib = root / "lib"
    extra = []
    multiarch: list[Path] = []
    if lib.is_dir():
        extra.extend(str(path) for path in sorted(lib.glob("python*/site-packages")))
        extra.extend(str(path) for path in sorted(lib.glob("python*/dist-packages")))
        multiarch = sorted(lib.glob("*-linux-gnu"))
        extra.extend(str(path) for path in multiarch if path.is_dir())
    for entry in reversed(extra):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if extra:
        previous = os.environ.get("PYTHONPATH")
        pythonpath = os.pathsep.join(extra)
        os.environ["PYTHONPATH"] = pythonpath if not previous else pythonpath + os.pathsep + previous
        lib_path = os.pathsep.join(str(path) for path in [lib, *multiarch] if Path(path).is_dir())
        os.environ["LD_LIBRARY_PATH"] = lib_path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def _write_config(template: dict, dest: Path, client: Path, seed: int, workers: int) -> None:
    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["random_seed"] = seed
    comm = sections.setdefault("Communicator", {})
    comm["client_path"] = str(client.resolve())
    comm["type"] = "local"
    comm["number_of_CPUs"] = workers
    buffered = int(comm.get("num_jobs", workers))
    comm["num_jobs"] = max(workers, buffered)
    dest.write_text(format_ini(sections), encoding="utf-8")


def _left_state0(states: Path) -> bool:
    """True when AKMC has a state directory other than state 0.

    ``state_table`` sits beside the numbered states and is not a KMC state.
    """
    if not states.is_dir():
        return False
    return any(path.name.isdigit() and path.name != "0" for path in states.iterdir())


def _read(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _published(reduction: HtstReduction, barrier_target: int, rate_target: int) -> HtstReduction:
    """Keep the first archive-sized samples. Later windows stay out of the means."""
    return replace(
        reduction,
        barrier_jobs=list(reduction.barrier_jobs[:barrier_target]),
        barrier_force_calls=list(reduction.barrier_force_calls[:barrier_target]),
        rate_jobs=list(reduction.rate_jobs[:rate_target]),
        rate_force_calls=list(reduction.rate_force_calls[:rate_target]),
    )


def _complete(reduction: HtstReduction, barrier_target: int, rate_target: int) -> bool:
    return (
        reduction.barrier_windows >= barrier_target
        and reduction.rate_windows >= rate_target
        and not reduction.missed_barriers
    )


def _campaign_text(
    template: dict,
    reduction: HtstReduction,
    *,
    target: int,
    rate_target: int,
    workers: int,
    passes: int,
    seed: int,
    seconds: float,
    error: str,
    left_state0: bool,
) -> str:
    lines = metric_lines(_published(reduction, target, rate_target))
    lines.append(f"searches {reduction.searches}")
    lines.append(f"missed_barriers {len(reduction.missed_barriers)}")
    lines.append(f"unexpected_barriers {len(reduction.unexpected_barriers)}")
    lines.append(f"reference_processes {len(reduction.reference_ids)}")
    lines.append(f"code {template.get('code', 'eOn')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append("counter htst-reducer")
    lines.append("reducer harness-htst")
    lines.append("contributor Rohit Goswami")
    lines.append(f"random_seed {seed}")
    lines.append(f"communicator_jobs {workers}")
    lines.append(f"passes {passes}")
    lines.append(f"seconds {seconds:.6g}")
    if template.get("algorithm"):
        lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {template.get('problem', '')}")
    lines.append("count_keys sad-fcs")
    if left_state0:
        lines.append("left_state0 true")
    if not _complete(reduction, target, rate_target):
        lines.append("partial true")
    if error:
        lines.append(f"error {error}")
    return "\n".join(lines) + "\n"


def _write_campaign(
    template: dict,
    reduction: HtstReduction,
    output: Path,
    *,
    target: int,
    rate_target: int,
    workers: int,
    passes: int,
    seed: int,
    seconds: float,
    error: str,
    left_state0: bool,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    published = _published(reduction, target, rate_target)
    output.write_text(
        _campaign_text(
            template,
            reduction,
            target=target,
            rate_target=rate_target,
            workers=workers,
            passes=passes,
            seed=seed,
            seconds=seconds,
            error=error,
            left_state0=left_state0,
        ),
        encoding="utf-8",
    )
    starts_path = output.with_name(output.name[: -len(output.suffix)] + ".starts.jsonl")
    rows = barrier_start_records(
        published,
        problem=str(template.get("problem") or ""),
        template=str(template["id"]),
        code=str(template.get("code") or "eOn"),
        code_version=str(template.get("code_version") or ""),
        commit=str(template.get("commit") or ""),
        tuning=str(template.get("algorithm") or ""),
    )
    write_starts(starts_path, rows)
    return {
        "barrier_windows": reduction.barrier_windows,
        "rate_windows": reduction.rate_windows,
        "published_barrier_windows": published.barrier_windows,
        "published_rate_windows": published.rate_windows,
        "searches": reduction.searches,
        "starts": len(rows),
        "target": target,
        "rate_target": rate_target,
        "partial": not _complete(reduction, target, rate_target),
        "template": template["id"],
        "problem": template.get("problem"),
        "barrier_jobs": None if published.barrier_jobs_mean is None else int(published.barrier_jobs_mean),
        "barrier_force_calls": published.barrier_force_calls_mean,
        "rate_jobs": None if published.rate_jobs_mean is None else int(published.rate_jobs_mean),
        "rate_force_calls": published.rate_force_calls_mean,
        "missed_barriers": len(reduction.missed_barriers),
        "passes": passes,
        "seconds": seconds,
        "error": error,
    }


def run_htst(
    root: Path,
    template_name: str,
    client: Path,
    output: Path,
    work: Path,
    workers: int,
    *,
    resume: bool = False,
) -> dict:
    """Run AKMC until the archive's barrier and rate window counts.

    A pass that adds no search lines five times in a row stops the loop.
    The file is rewritten after every pass. ``partial true`` stays in the
    dat until both window counts are reached and every reference barrier
    has been matched. The means keep the first ``windows`` barrier windows
    and the first ``rate_windows`` rate windows.
    """
    if workers <= 0:
        raise ValueError("workers must be positive")
    template = load_template(root, template_name)
    problem = resolve_problem(root, str(template["problem"]))
    spec = spec_from_problem(problem)
    sample = problem.get("sample") or {}
    target = int(template.get("windows") or sample.get("starts") or 0)
    rate_target = int(template.get("rate_windows") or 0)
    if target <= 0 or rate_target <= 0:
        raise ValueError(
            f"{template_name} needs windows and rate_windows; "
            f"{problem['id']} sample.starts does not replace the rate count"
        )
    seed = int(((template.get("ini") or {}).get("Main") or {}).get("random_seed", 1))
    passes_limit = int(template.get("pass_limit", 4000))
    if work.exists() and not resume:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "pos.con").is_file():
        pos = geometry_archive(root, problem)
        shutil.copy(pos, work / "pos.con")
    _write_config(template, work / "config.ini", client, seed, workers)
    _prepare_client(client)
    for name in ("akmc", "eon", "state", "communicator"):
        logging.getLogger(name).setLevel(logging.WARNING)
    previous = Path.cwd()
    started = time.monotonic()
    last_error = ""
    reduction: HtstReduction | None = None
    seen = -1
    stall = 0
    finished = 0
    try:
        os.chdir(work)
        from eon import akmc as akmc_mod
        from eon.config import ConfigClass
        from eon.config import config as shared_config

        try:
            # r2549 helpers read the module-level config object. Later
            # eOn passes the instance into those helpers.
            parameters = inspect.signature(akmc_mod.get_akmc_metadata).parameters
            if "config" in parameters:
                config = ConfigClass()
            else:
                config = shared_config
                config.init_done = False
            config.init()
            from eon import communicator

            if hasattr(communicator.get_communicator, "comm"):
                del communicator.get_communicator.comm
        except SystemExit as exc:
            last_error = f"config exit {exc.code}"
            empty = reduce_tables("", "header\nheader\n", spec)
            return _write_campaign(
                template,
                empty,
                output,
                target=target,
                rate_target=rate_target,
                workers=workers,
                passes=0,
                seed=seed,
                seconds=time.monotonic() - started,
                error=last_error,
                left_state0=False,
            )
        for index in range(passes_limit):
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                last_error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            finished = index + 1
            state = work / "states" / "0"
            processtable = _read(state / "processtable")
            searches = _read(state / "search_results.txt")
            if not searches:
                stall += 1
            else:
                try:
                    reduction = reduce_tables(processtable, searches, spec)
                except ValueError as exc:
                    last_error = str(exc)
                    break
                if reduction.searches == seen:
                    stall += 1
                else:
                    stall = 0
                    seen = reduction.searches
            left_state0 = _left_state0(work / "states")
            if reduction is not None:
                summary = _write_campaign(
                    template,
                    reduction,
                    output,
                    target=target,
                    rate_target=rate_target,
                    workers=workers,
                    passes=finished,
                    seed=seed,
                    seconds=time.monotonic() - started,
                    error=last_error,
                    left_state0=left_state0,
                )
                print(
                    f"pass {finished} searches={summary['searches']} "
                    f"barrier_windows={summary['barrier_windows']}/{target} "
                    f"rate_windows={summary['rate_windows']}/{rate_target} "
                    f"missed={summary['missed_barriers']}",
                    flush=True,
                )
                if not summary["partial"]:
                    summary["seconds"] = time.monotonic() - started
                    return summary
            # jobs/scratch holds the batch this pass just ran. The next
            # akmc() call registers it. Removing the directory first
            # drops those searches.
            if stall >= 5:
                last_error = last_error or "search log stopped growing"
                break
        else:
            finished = passes_limit
        # Register the batch the last submitting pass left in jobs/scratch.
        config.comm_job_buffer_size = 0
        try:
            akmc_mod.akmc(config)
        except SystemExit as exc:
            last_error = last_error or f"akmc exit {exc.code}"
        except Exception as exc:
            last_error = last_error or f"{type(exc).__name__}: {exc}"
        else:
            finished += 1
            state = work / "states" / "0"
            processtable = _read(state / "processtable")
            searches = _read(state / "search_results.txt")
            if searches:
                try:
                    reduction = reduce_tables(processtable, searches, spec)
                except ValueError as exc:
                    last_error = last_error or str(exc)
    finally:
        os.chdir(previous)
        jobs = work / "jobs"
        if jobs.exists():
            shutil.rmtree(jobs, ignore_errors=True)
    if reduction is None:
        reduction = reduce_tables("", "header\nheader\n", spec)
    left_state0 = _left_state0(work / "states")
    summary = _write_campaign(
        template,
        reduction,
        output,
        target=target,
        rate_target=rate_target,
        workers=workers,
        passes=finished,
        seed=seed,
        seconds=time.monotonic() - started,
        error=last_error,
        left_state0=left_state0,
    )
    return summary
