"""Langevin parallel replica on the aluminum adatom.

The archived reference is a parallel-replica escape rate with Langevin
friction 0.001 per fs. Each replica starts from the published reactant,
dephases, and runs until a quench differs from that reactant. The local
client runs the whole batch, so the physical time of a batch that
escaped is the replica count times the earliest transition. Batches
with no escape add the sum of their block times. The rate is the number
of escapes divided by that time, and the uncertainty is the rate
divided by the square root of the number of escapes.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .amd import (
    catalogue_accuracy,
    distribution,
    force_calls_per_simulated_time,
    match_product,
    mobile_indices,
    parse_con,
    rate_accuracy,
    rate_distribution,
    simulated_time_seconds,
)
from .campaign import format_ini, geometry_archive, load_template, resolve_problem, write_starts
from .eon_akmc import prepare_client_env
from .eon_amd import _catalogue_paths, _fmt

COUNT_KEY = "eon_dynamics_calls"


def poisson_rate(events: int, total_time: float) -> dict[str, float]:
    """Escape rate and its Poisson standard error.

    ``events`` independent escapes in ``total_time`` seconds give
    rate = events / total_time. The standard error is rate / sqrt(events),
    which is the 2012 percent uncertainty when the sample is exponential.
    """
    if isinstance(events, bool) or not isinstance(events, int) or events < 1:
        raise ValueError("events must be a positive integer")
    if not math.isfinite(total_time) or total_time <= 0:
        raise ValueError("total time must be positive")
    rate = events / total_time
    uncertainty = rate / math.sqrt(events)
    return {
        "true_rate": rate,
        "uncertainty": uncertainty,
        "percent_uncertainty": 100.0 * uncertainty / rate,
        "events": float(events),
        "time_s": total_time,
    }


def batch_account(rows: list[dict]) -> dict:
    """Physical time and cost of one parallel-replica batch.

    A batch with no transition contributes the sum of the block times.
    A batch with a transition stops at the earliest one. Every replica
    contributes the time it actually ran up to that instant.
    """
    if not rows:
        raise ValueError("a batch has no replicas")
    if any(row.get("error") for row in rows):
        raise ValueError("a batch has a replica without results")
    calls = 0
    hits = []
    for row in rows:
        calls += int(row["force_calls"])
        if "transition_time_s" in row:
            hits.append(row)
    if not hits:
        total = 0.0
        for row in rows:
            block = float(row["simulation_time_s"])
            if not math.isfinite(block) or block <= 0:
                raise ValueError("a block time must be positive")
            total += block
        return {
            "escaped": False,
            "time_s": total,
            "force_calls": calls,
            "replicas": len(rows),
        }
    earliest = min(float(row["transition_time_s"]) for row in hits)
    if not math.isfinite(earliest) or earliest <= 0:
        raise ValueError("a transition time must be positive")
    physical = 0.0
    for row in rows:
        if "transition_time_s" in row:
            duration = float(row["transition_time_s"])
        else:
            duration = float(row["simulation_time_s"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("a replica time must be positive")
        physical += min(duration, earliest)
    winners = [row for row in hits if float(row["transition_time_s"]) == earliest]
    return {
        "escaped": True,
        "time_s": physical,
        "t_min": earliest,
        "force_calls": calls,
        "replicas": len(rows),
        "earliest": winners,
    }


def _result_fields(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            fields[parts[1]] = parts[0]
    return fields


def read_batch(scratch: Path) -> list[dict]:
    """Replica outcomes left in the scratch directory."""
    if not scratch.is_dir():
        return []
    rows = []
    for job in sorted(path for path in scratch.iterdir() if path.is_dir()):
        results = job / "results.dat"
        if not results.is_file():
            rows.append({"job": job.name, "error": "no results"})
            continue
        fields = _result_fields(results)
        if "total_force_calls" not in fields:
            rows.append({"job": job.name, "error": "no force calls"})
            continue
        row = {
            "job": job.name,
            "force_calls": int(float(fields["total_force_calls"])),
            "reactant_energy": float(fields["potential_energy_reactant"])
            if "potential_energy_reactant" in fields
            else None,
            "product_energy": float(fields["potential_energy_product"])
            if "potential_energy_product" in fields
            else None,
            "product_text": (job / "product.con").read_text(encoding="utf-8", errors="replace")
            if (job / "product.con").is_file()
            else "",
        }
        if fields.get("transition_found") == "1":
            row["transition_time_s"] = float(fields["transition_time_s"])
        elif "simulation_time_s" in fields:
            row["simulation_time_s"] = float(fields["simulation_time_s"])
        else:
            row["error"] = "no time"
        rows.append(row)
    return rows


def _empty_dir(path: Path) -> None:
    """Remove a batch's files and leave the directory in place.

    The local communicator creates the next job inside this directory.
    Deleting the directory itself makes that create fail.
    """
    path.mkdir(parents=True, exist_ok=True)
    for child in list(path.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass


def _write_config(template: dict, dest: Path, client: Path, seed: int, workers: int) -> None:
    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["random_seed"] = seed
    comm = sections.setdefault("Communicator", {})
    comm["client_path"] = str(client.resolve())
    comm["type"] = "local"
    comm["number_of_CPUs"] = workers
    comm["num_jobs"] = int(comm.get("num_jobs", workers))
    comm["jobs_per_bundle"] = 1
    dest.write_text(format_ini(sections), encoding="utf-8")


def _channel_of(text: str, products: list[np.ndarray], mobile: np.ndarray, tolerance: float):
    if not text:
        return None
    points, _fixed = parse_con(text)
    return match_product(points, products, mobile, tolerance)


def _one_seed(
    work: Path,
    client: Path,
    template: dict,
    seed: int,
    workers: int,
    products: list[np.ndarray],
    mobile: np.ndarray,
    tolerance: float,
    *,
    escapes: int,
    batch_limit: int,
    on_escape=None,
) -> dict:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    reactant = geometry_archive(
        Path(template["_root"]),
        {
            "id": template["_problem_id"],
            "input": {"archive_path": template["_reactant"]},
        },
    )
    shutil.copy(reactant, work / "pos.con")
    _write_config(template, work / "config.ini", client, seed, workers)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("UCX_MEM_EVENTS", "no")
    previous = Path.cwd()
    started = time.monotonic()
    found: list[dict] = []
    calls = 0
    total_time = 0.0
    batches = 0
    error = ""
    reactant_energy = None
    attempt_time = 0.0
    try:
        os.chdir(work)
        from eon import communicator
        from eon.config import ConfigClass
        from eon.parallelreplica import get_statelist, make_searches

        config = ConfigClass()
        config.init()
        states = get_statelist(config)
        state = states.get_state(0)
        comm = communicator.get_communicator(config)
        wuid = 0
        attempt_time = 0.0
        attempt_calls = 0
        attempt_batches = 0
        while len(found) < escapes:
            wuid = make_searches(comm, state, wuid, config)
            if hasattr(comm, "joblist"):
                comm.joblist.clear()
            scratch = Path(config.path_scratch)
            rows = read_batch(scratch)
            account = batch_account(rows)
            _empty_dir(scratch)
            _empty_dir(Path(config.path_jobs_in))
            batches += 1
            attempt_batches += 1
            calls += int(account["force_calls"])
            attempt_calls += int(account["force_calls"])
            attempt_time += float(account["time_s"])
            if reactant_energy is None:
                for row in rows:
                    if row.get("reactant_energy") is not None:
                        reactant_energy = float(row["reactant_energy"])
                        break
            if not account["escaped"]:
                print(
                    f"seed {seed} batch {batches} no escape "
                    f"time_s={attempt_time:.6e} calls={calls}",
                    flush=True,
                )
                if attempt_batches >= batch_limit:
                    error = "batch limit before an escape"
                    break
                continue
            winners = account["earliest"]
            total_time += attempt_time
            for winner in winners:
                try:
                    channel = _channel_of(
                        winner.get("product_text") or "", products, mobile, tolerance
                    )
                except (ValueError, IndexError):
                    channel = None
                event = {
                    "escape": len(found) + 1,
                    "time_s": attempt_time,
                    "t_min": float(account["t_min"]),
                    "force_calls": attempt_calls,
                    "batches": attempt_batches,
                    "replicas": int(account["replicas"]),
                    "ties": len(winners),
                    "channel": channel,
                    "product_energy": winner.get("product_energy"),
                    "reactant_energy": reactant_energy,
                }
                found.append(event)
            print(
                f"seed {seed} escape {len(found)} time_s={attempt_time:.6e} "
                f"channel={found[-1]['channel']} ties={len(winners)} calls={calls}",
                flush=True,
            )
            if on_escape is not None:
                on_escape(
                    {
                        "events": found,
                        "force_calls": calls,
                        "time_s": total_time,
                        "batches": batches,
                        "reactant_energy": reactant_energy,
                    }
                )
            attempt_time = 0.0
            attempt_calls = 0
            attempt_batches = 0
    except SystemExit as exc:
        error = f"exit {exc.code}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
    reached = len(found) >= escapes and not error
    rate = None
    if found and total_time > 0:
        rate = poisson_rate(len(found), total_time)
    channels: dict[int | None, int] = {}
    for event in found:
        key = event["channel"]
        channels[key] = channels.get(key, 0) + 1
    return {
        "seed": seed,
        "reached": reached,
        "error": error,
        "events": found,
        "event_count": len(found),
        "time_s": total_time,
        "force_calls": calls,
        "batches": batches,
        "rate": None if rate is None else rate["true_rate"],
        "uncertainty": None if rate is None else rate["uncertainty"],
        "channels": [(key, float(value)) for key, value in sorted(channels.items(), key=lambda item: (item[0] is None, item[0] or 0))],
        "reactant_energy": reactant_energy,
        "censored_time_s": attempt_time,
        "seconds": time.monotonic() - started,
    }


def _failure_class(row: dict) -> str | None:
    if row.get("reached") and float(row.get("rate") or 0) > 0:
        return None
    if row.get("error"):
        return "crash" if "Error" in str(row.get("error")) or "Exception" in str(row.get("error")) else "divergence"
    return "divergence"


def _write_result(
    template: dict,
    problem: dict,
    rows: list[dict],
    output: Path,
    workers: int,
    *,
    escapes: int,
    in_progress: bool,
) -> dict:
    reference = float(problem["reference"]["true_rate"])
    probabilities = distribution(problem["catalogue"]["probabilities"])
    products = int(problem["sample"]["products"])
    kept = [row for row in rows if row.get("event_count")]
    failed = sum(1 for row in rows if not row.get("reached"))
    lines = []
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"# rep {index} force_calls {int(row.get('force_calls') or 0)} "
            f"events {int(row.get('event_count') or 0)} "
            f"time_s {_fmt(float(row.get('time_s') or 0.0))} "
            f"rate {_fmt(float(row.get('rate') or 0.0))} "
            f"uncertainty {_fmt(float(row.get('uncertainty') or 0.0))} "
            f"reached {int(bool(row.get('reached')))}"
        )
    events = sum(int(row.get("event_count") or 0) for row in kept)
    total_time = sum(float(row.get("time_s") or 0.0) for row in kept)
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
        "events": events,
    }
    if events > 0 and total_time > 0:
        pooled = poisson_rate(events, total_time)
        summary["true_rate"] = pooled["true_rate"]
        summary["uncertainty"] = pooled["uncertainty"]
        summary["percent_uncertainty"] = pooled["percent_uncertainty"]
        summary["rate_accuracy"] = rate_accuracy(pooled["true_rate"], reference)
        weights = []
        for row in kept:
            weights.append(rate_distribution(row.get("channels") or []))
        if weights:
            merged: dict[int, float] = {}
            for item in weights:
                for key, value in item.items():
                    merged[key] = merged.get(key, 0.0) + value
            summary["catalogue_accuracy"] = catalogue_accuracy(merged, probabilities)
        calls = sum(int(row.get("force_calls") or 0) for row in kept)
        # The problem divides catalogue calls by the seconds 5000 escapes
        # would last. Scaling the measured calls to that many escapes is
        # the same number as the calls divided by the physical time.
        seconds = simulated_time_seconds(pooled["true_rate"], products)
        scaled_calls = calls * (products / events)
        summary["force_calls"] = calls / len(kept)
        summary["force_calls_per_simulated_time"] = force_calls_per_simulated_time(
            scaled_calls, seconds
        )
        summary["time_s"] = total_time
    lines.append(f"code {template.get('code', 'eOn')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append("counter harness")
    lines.append(f"count_keys {COUNT_KEY}")
    lines.append("contributor Rohit Goswami")
    lines.append(f"repetitions {len(rows)}")
    lines.append(f"workers {workers}")
    lines.append(f"escapes {escapes}")
    lines.append(f"events {events}")
    if events > 0 and total_time > 0:
        lines.append(f"time_s {_fmt(total_time)}")
    elif rows:
        censored = sum(float(row.get("censored_time_s") or 0.0) for row in rows)
        if censored > 0:
            lines.append(f"censored_time_s {_fmt(censored)}")
    energies = [
        float(row["reactant_energy"])
        for row in rows
        if row.get("reactant_energy") is not None
    ]
    if energies:
        lines.append(f"reactant_energy {_fmt(energies[0])}")
    lines.append(f"temperature_K {problem['thermal']['temperature_K']}")
    lines.append("thermostat langevin_0.001")
    lines.append(f"reference_rate {_fmt(reference)}")
    lines.append(f"reference_products {products}")
    if summary["true_rate"] is not None:
        lines.append(f"true_rate {_fmt(float(summary['true_rate']))}")
        lines.append(f"uncertainty {_fmt(float(summary['uncertainty']))}")
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
    if template.get("algorithm"):
        lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append("reducer amd-catalogue")
    target = int(problem["sample"]["repetitions"])
    if in_progress or len(rows) < target or any(not row.get("reached") for row in rows):
        lines.append("partial true")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    start_rows = []
    for index, row in enumerate(rows, start=1):
        failure = _failure_class(row)
        record = {
            "schema": "optbench.start/v2",
            "start": f"rep-{index}",
            "geometry": "reactant",
            "force_calls": int(row.get("force_calls") or 0),
            "count_key": COUNT_KEY,
            "counter": "harness",
            "comparable": True,
            "success": failure is None,
            "failure": failure,
            "seed": row.get("seed"),
            "rate": row.get("rate"),
            "uncertainty": row.get("uncertainty"),
            "events": int(row.get("event_count") or 0),
            "time_s": row.get("time_s"),
            "channels": [
                [None if channel is None else int(channel), float(count)]
                for channel, count in (row.get("channels") or [])
            ],
            "reactant_energy": row.get("reactant_energy"),
            "escapes": [
                {
                    "time_s": float(event["time_s"]),
                    "t_min": float(event["t_min"]),
                    "channel": None if event["channel"] is None else int(event["channel"]),
                    "product_energy": event.get("product_energy"),
                    "force_calls": int(event["force_calls"]),
                }
                for event in row.get("events") or []
            ],
            "seconds": row.get("seconds"),
            "problem": problem["id"],
            "template": template["id"],
        }
        if row.get("error"):
            record["error"] = row["error"]
        start_rows.append(record)
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in start_rows)
    output.with_suffix(".starts.jsonl").write_text(body, encoding="utf-8")
    write_starts(output.with_suffix(".starts"), start_rows)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    return summary


def run_pr(
    root: Path,
    template_name: str,
    client: Path,
    output: Path,
    work: Path,
    workers: int,
    repetitions: int | None = None,
    escapes: int | None = None,
) -> dict:
    """Run parallel-replica escape samples and write benchmark.dat."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    template = load_template(root, template_name)
    problem = resolve_problem(root, str(template["problem"]))
    if repetitions is None:
        repetitions = int(problem["sample"]["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if escapes is None:
        escapes = int(template.get("escapes", 1))
    if escapes <= 0:
        raise ValueError("escapes must be positive")
    batch_limit = int(template.get("batch_limit", 500))
    template = dict(template)
    template["_root"] = str(root)
    template["_problem_id"] = problem["id"]
    template["_reactant"] = problem["input"]["reactant"]
    tolerance = float((problem.get("convergence") or {}).get("catalogue_rmsd", 0.15))
    reactant_path = geometry_archive(
        root,
        {"id": problem["id"], "input": {"archive_path": problem["input"]["reactant"]}},
    )
    _points, fixed = parse_con(reactant_path.read_text(encoding="utf-8", errors="replace"))
    mobile = mobile_indices(fixed)
    catalogue = [
        parse_con(path.read_text(encoding="utf-8", errors="replace"))[0]
        for path in _catalogue_paths(root, 24)
    ]
    prepare_client_env(client)
    rows: list[dict] = []
    summary: dict = {}
    for index in range(repetitions):
        seed = 20261018 + index

        def _checkpoint(progress: dict, rows: list[dict] = rows, seed: int = seed) -> None:
            partial = {
                "seed": seed,
                "reached": False,
                "error": "",
                "events": progress["events"],
                "event_count": len(progress["events"]),
                "time_s": progress["time_s"],
                "force_calls": progress["force_calls"],
                "rate": poisson_rate(len(progress["events"]), progress["time_s"])["true_rate"]
                if progress["events"] and progress["time_s"]
                else 0.0,
                "uncertainty": poisson_rate(len(progress["events"]), progress["time_s"])["uncertainty"]
                if progress["events"] and progress["time_s"]
                else 0.0,
                "channels": [],
                "reactant_energy": progress.get("reactant_energy"),
                "seconds": None,
            }
            grouped: dict[int | None, int] = {}
            for event in progress["events"]:
                grouped[event["channel"]] = grouped.get(event["channel"], 0) + 1
            partial["channels"] = [
                (key, float(value))
                for key, value in sorted(grouped.items(), key=lambda item: (item[0] is None, item[0] or 0))
            ]
            _write_result(template, problem, rows + [partial], output, workers, escapes=escapes, in_progress=True)

        row = _one_seed(
            work / f"rep-{index + 1}",
            client,
            template,
            seed,
            workers,
            catalogue,
            mobile,
            tolerance,
            escapes=escapes,
            batch_limit=batch_limit,
            on_escape=_checkpoint,
        )
        rows.append(row)
        print(
            f"rep {index + 1} reached={row.get('reached')} rate={row.get('rate')} "
            f"events={row.get('event_count')} force_calls={row.get('force_calls')} "
            f"error={row.get('error', '')}",
            flush=True,
        )
        summary = _write_result(
            template, problem, rows, output, workers, escapes=escapes, in_progress=False
        )
    return summary
