"""AKMC hTST campaign for the TSASE Dimer-QM platinum row.

The server is eOn's Python AKMC. The client is TSASE 1.0 SSDimer.
One run stays on ``states/0``. Force calls are ``sad-fcs`` through
:mod:`optbench.htst`. The quoted means are every closed window in the
search log. Registration stops once 4 barrier windows and 60 rate
windows have both closed. Rate windows can pass 60 while a barrier
window is still open, and those later windows stay in the mean.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from optbench.campaign import format_ini, load_template, write_starts
from optbench.htst import (
    HtstReduction,
    HtstSpec,
    barrier_start_records,
    metric_lines,
    reduce_tables,
)

PT_SPEC = HtstSpec(
    barriers=(
        0.601,
        0.601,
        0.620,
        0.986,
        0.987,
        0.989,
        0.986,
        0.987,
        0.989,
        1.196,
        1.196,
        1.196,
        1.196,
        1.207,
        1.207,
    ),
    rate_ref=155055901.244,
    temperature=700.0,
    prefactor=1.2e12,
    barrier_ceiling=1.3,
)
BARRIER_WINDOWS = 4
RATE_WINDOWS = 60


def _geometry(root: Path, template: dict) -> Path:
    wanted = str(template.get("geometry") or "")
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == wanted:
            return root / row["stored"]
    raise FileNotFoundError(f"{wanted} is not in the geometry manifest")


def _client_script(dest: Path) -> None:
    python = Path(os.path.abspath(sys.executable))
    dest.write_text(
        f"#!{python}\n"
        "import os\n"
        "os.environ.setdefault('OMP_NUM_THREADS', '1')\n"
        "from optbench.tsase_dimer_client import main\n"
        "main()\n",
        encoding="utf-8",
    )
    dest.chmod(0o755)


def _write_config(template: dict, dest: Path, client: Path, workers: int) -> None:
    sections = json.loads(json.dumps(template["ini"]))
    comm = sections.setdefault("Communicator", {})
    comm["client_path"] = str(client.resolve())
    comm["type"] = "local"
    comm["number_of_CPUs"] = workers
    comm["num_jobs"] = workers
    comm["jobs_per_bundle"] = 1
    dest.write_text(format_ini(sections), encoding="utf-8")


def _read(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _left_state0(states: Path) -> bool:
    if not states.is_dir():
        return False
    return any(path.name.isdigit() and path.name != "0" for path in states.iterdir())


def _campaign_text(
    template: dict,
    reduction: HtstReduction,
    *,
    workers: int,
    passes: int,
    seed: int,
    seconds: float,
    error: str,
    left_state0: bool,
) -> str:
    lines = metric_lines(reduction)
    lines.append(f"searches {reduction.searches}")
    lines.append(f"missed_barriers {len(reduction.missed_barriers)}")
    lines.append(f"unexpected_barriers {len(reduction.unexpected_barriers)}")
    lines.append(f"reference_processes {len(reduction.reference_ids)}")
    lines.append(f"code {template.get('code', 'tsase')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append("counter harness")
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
    if error:
        lines.append(f"error {error}")
    return "\n".join(lines) + "\n"


def _write_campaign(
    template: dict,
    reduction: HtstReduction,
    output: Path,
    *,
    workers: int,
    passes: int,
    seed: int,
    seconds: float,
    error: str,
    left_state0: bool,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _campaign_text(
            template,
            reduction,
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
        reduction,
        problem=str(template.get("problem") or ""),
        template=str(template["id"]),
        code=str(template.get("code") or "tsase"),
        code_version=str(template.get("code_version") or ""),
        commit=str(template.get("commit") or ""),
        tuning=str(template.get("algorithm") or ""),
    )
    write_starts(starts_path, rows)
    return {
        "barrier_windows": reduction.barrier_windows,
        "rate_windows": reduction.rate_windows,
        "searches": reduction.searches,
        "starts": len(rows),
        "short_sample": reduction.barrier_windows < BARRIER_WINDOWS or reduction.rate_windows < RATE_WINDOWS,
        "template": template["id"],
        "problem": template.get("problem"),
        "barrier_jobs": None if reduction.barrier_jobs_mean is None else int(reduction.barrier_jobs_mean),
        "barrier_force_calls": reduction.barrier_force_calls_mean,
        "rate_jobs": None if reduction.rate_jobs_mean is None else int(reduction.rate_jobs_mean),
        "rate_force_calls": reduction.rate_force_calls_mean,
        "missed_barriers": len(reduction.missed_barriers),
        "passes": passes,
        "seconds": seconds,
        "error": error,
    }


def _order_job_dirs() -> None:
    """Register a finished batch in wuid order.

    The local communicator lists the scratch directory. Lexical order
    puts ``0_10`` before ``0_2``. The displacement sequence is the
    wuid sequence.
    """
    import eon.communicator as communicator

    original = communicator.os.listdir

    def listed(path):
        names = list(original(path))
        if "jobs" not in str(path):
            return names

        def key(item):
            index, name = item
            parts = name.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (0, int(parts[0]), int(parts[1]))
            return (1, index, name)

        return [name for index, name in sorted(enumerate(names), key=key)]

    communicator.os.listdir = listed


def _done(reduction: HtstReduction) -> bool:
    return (
        reduction.barrier_windows >= BARRIER_WINDOWS
        and reduction.rate_windows >= RATE_WINDOWS
        and not reduction.missed_barriers
    )


def run_htst(
    root: Path,
    template_name: str,
    output: Path,
    work: Path,
    workers: int,
    *,
    resume: bool = False,
) -> dict:
    """Run until 4 barrier windows and 60 rate windows are in the log."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    template = load_template(root, template_name)
    seed = int(((template.get("ini") or {}).get("Main") or {}).get("random_seed", 42))
    passes_limit = int(os.environ.get("HTST_TSASE_PASS_LIMIT") or template.get("pass_limit", 4000))
    if work.exists() and not resume:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "pos.con").is_file():
        shutil.copy(_geometry(root, template), work / "pos.con")
    client = work / "dimer_ase.py"
    _client_script(client)
    _write_config(template, work / "config.ini", client, workers)
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
        _order_job_dirs()
        from eon import akmc as akmc_mod
        from eon.config import config as shared_config

        try:
            shared_config.init_done = False
            shared_config.init()
            from eon import communicator

            if hasattr(communicator.get_communicator, "comm"):
                del communicator.get_communicator.comm
        except SystemExit as exc:
            last_error = f"config exit {exc.code}"
            empty = reduce_tables("", "header\nheader\n", PT_SPEC)
            return _write_campaign(
                template,
                empty,
                work / "progress.dat",
                workers=workers,
                passes=0,
                seed=seed,
                seconds=time.monotonic() - started,
                error=last_error,
                left_state0=False,
            )
        for index in range(passes_limit):
            try:
                akmc_mod.akmc(shared_config)
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
                    reduction = reduce_tables(processtable, searches, PT_SPEC)
                except ValueError as exc:
                    last_error = str(exc)
                    break
                if reduction.searches == seen:
                    stall += 1
                else:
                    stall = 0
                    seen = reduction.searches
            left = _left_state0(work / "states")
            if reduction is not None:
                summary = _write_campaign(
                    template,
                    reduction,
                    work / "progress.dat",
                    workers=workers,
                    passes=finished,
                    seed=seed,
                    seconds=time.monotonic() - started,
                    error=last_error,
                    left_state0=left,
                )
                if _done(reduction):
                    _write_campaign(
                        template,
                        reduction,
                        output,
                        workers=workers,
                        passes=finished,
                        seed=seed,
                        seconds=time.monotonic() - started,
                        error=last_error,
                        left_state0=left,
                    )
                print(
                    f"pass {finished} searches={summary['searches']} "
                    f"barrier_windows={summary['barrier_windows']}/{BARRIER_WINDOWS} "
                    f"rate_windows={summary['rate_windows']}/{RATE_WINDOWS} "
                    f"missed={summary['missed_barriers']} "
                    f"barrier_jobs={summary['barrier_jobs']} "
                    f"rate_jobs={summary['rate_jobs']}",
                    flush=True,
                )
                if _done(reduction):
                    summary["seconds"] = time.monotonic() - started
                    return summary
            if stall >= 5:
                last_error = last_error or "search log stopped growing"
                break
            if left:
                last_error = last_error or "left state 0"
                break
        else:
            finished = passes_limit
    finally:
        os.chdir(previous)
    if reduction is None:
        reduction = reduce_tables("", "header\nheader\n", PT_SPEC)
    destination = output if _done(reduction) else work / "progress.dat"
    return _write_campaign(
        template,
        reduction,
        destination,
        workers=workers,
        passes=finished,
        seed=seed,
        seconds=time.monotonic() - started,
        error=last_error,
        left_state0=_left_state0(work / "states"),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "campaign" / "eon-htst-pt-tsase-dimer-qm.dat"
    work = Path(os.environ.get("HTST_TSASE_WORK", "/tmp/htst-pt-tsase"))
    workers = int(os.environ.get("HTST_TSASE_WORKERS", "7"))
    resume = os.environ.get("HTST_TSASE_RESUME", "") == "1"
    summary = run_htst(root, "eon/htst-pt-tsase-dimer-qm", output, work, workers, resume=resume)
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
