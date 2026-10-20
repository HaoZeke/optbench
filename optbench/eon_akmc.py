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


def honor_configured_void_bias() -> None:
    """Use ``Saddle Search`` ``void_bias_fraction`` from the client config.

    eOn 3.6.0 ``Displace.__init__`` assigns 0.2 and does not read the
    config. The yaml default is 0, which is the 2014 displacement: a
    Gaussian kick with no void repulsion. The hardcoded mix pulls every
    Pt-island search toward a void and finds the 27 low barriers early.
    """
    try:
        from eon import displace
    except ImportError:
        return
    if getattr(displace.Displace.__init__, "_optbench_void_bias", False):
        return
    original = displace.Displace.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        config = getattr(self, "config", None)
        fraction = 0.0 if config is None else float(getattr(config, "void_bias_fraction", 0.0))
        self.void_bias_fraction = fraction

    init._optbench_void_bias = True
    displace.Displace.__init__ = init
    print("eon displacement void_bias_fraction follows the config value", flush=True)


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
) -> dict:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
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
    for relative in (
        "jobs/scratch",
        "jobs/in",
        "jobs/out",
        "jobs/incomplete",
        "states",
        "potfiles",
    ):
        (work / relative).mkdir(parents=True, exist_ok=True)
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
            honor_configured_void_bias()
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
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                last_error = f"akmc exit {exc.code}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                break
            results = work / "states" / "0" / "search_results.txt"
            if not results.is_file():
                continue
            progress = saddle_progress(results.read_text(encoding="utf-8", errors="replace"))
            if (index + 1) % 25 == 0 or progress["reached"]:
                print(
                    f"seed {seed} pass {index + 1} qualifying={progress['qualifying']} "
                    f"searches={progress['total_searches']} force_calls={progress['force_calls']}",
                    flush=True,
                )
            if progress["reached"]:
                progress["passes"] = index + 1
                break
        else:
            progress["passes"] = passes
        # akmc() registers the previous pass, then submits this one.
        # A last call with an empty buffer records the searches that
        # already finished and does not start another batch.
        if not progress.get("reached"):
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
        os.chdir(previous)
    progress["seed"] = seed
    progress["error"] = last_error
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
    searches = sum(int(row["total_searches"]) for row in rows if row.get("total_searches") is not None)
    lines.append(f"n {searches}")
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
