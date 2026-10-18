"""Campaign driver for n2p2 Kalman training on the mlopt problems.

One start is one uniform initialization. The harness counts loss-gradient
evaluations. Starts run in separate processes so each Kalman matrix stays
on one core.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from optbench.campaign import write_starts
from optbench.n2p2_ml import (
    CODE,
    CODE_VERSION,
    EPOCH_CAP,
    FORCE_RMSE,
    GRAD_TOL,
    N2P2_COMMIT,
    N_STARTS,
    PYAMFF_COMMIT,
    SEED_BASE,
    cache_dataset,
    dataset_from_cache,
    run_start,
)
from optbench.score import half_even, score_v2


def _cache_path(work: Path, problem_id: str) -> Path:
    slug = problem_id.replace("/", "-")
    return work / f"{slug}-fingerprints.npz"


def _job(payload: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    dataset = dataset_from_cache(Path(payload["cache"]))
    started = time.perf_counter()
    record = run_start(
        dataset,
        int(payload["index"]),
        cap=int(payload["cap"]),
        tol=float(payload["tol"]),
        shuffle=True,
    )
    record["seconds"] = time.perf_counter() - started
    record["counter"] = "harness"
    record["comparable"] = True
    record["schema"] = "optbench.start/v2"
    record["code"] = CODE
    record["code_version"] = CODE_VERSION
    record["commit"] = N2P2_COMMIT
    record["template"] = payload["template"]
    record["problem"] = payload["problem"]
    record["tuning"] = payload["tuning"]
    record["cap"] = int(payload["cap"])
    record["pyamff_commit"] = PYAMFF_COMMIT
    return record


def _success_stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.median(values)


def _benchmark_text(summary: dict, records: list[dict], template: dict) -> str:
    successes = [record for record in records if record["success"]]
    calls = [float(record["force_calls"]) for record in successes]
    rmse = [float(record["energy_rmse"]) for record in successes if record.get("energy_rmse") is not None]
    mean_calls, median_calls = _success_stats(calls)
    mean_rmse, median_rmse = _success_stats(rmse)
    lines = [
        "# n2p2 standard Kalman filter on the pyamff energy-only squared error.",
        "# One force call is one loss-gradient evaluation. The filter updates once per structure.",
        "# Structures are permuted each epoch with PCG64, after the uniform[-1, 1) weights are drawn.",
        f"# loss commit {PYAMFF_COMMIT}. gradient tolerance {GRAD_TOL}. force RMSE is {FORCE_RMSE} because forces are not trained.",
        f"n {len(records)}",
        f"nfailed {summary['nfailed']}",
        f"cap {int(template['stop']['max_force_calls'])}",
        "counter harness",
        f"code {CODE}",
        f"code_version {CODE_VERSION}",
        f"commit {N2P2_COMMIT}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"problem {template['problem']}",
        f"tuning {template['tuning']}",
        f"mean_force_rmse {FORCE_RMSE}",
        f"median_force_rmse {FORCE_RMSE}",
    ]
    if mean_calls is not None:
        lines.append(f"force_calls {half_even(mean_calls)}")
        lines.append(f"mean_epoch {mean_calls:.8f}")
        lines.append(f"median_epoch {median_calls:.8f}")
    if mean_rmse is not None:
        lines.append(f"mean_energy_rmse {mean_rmse:.16g}")
        lines.append(f"median_energy_rmse {median_rmse:.16g}")
    for key in (
        "par2_mean",
        "median",
        "median_low",
        "median_high",
        "wilson_low",
        "wilson_high",
        "n_cap",
        "n_divergence",
        "n_crash",
        "n_wrong_stationary",
    ):
        value = summary.get(key)
        if value is not None:
            lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


def _load_done(path: Path) -> dict[int, dict]:
    found: dict[int, dict] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        found[int(record["start"])] = record
    return found


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None = None,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Train every initialization and write benchmark.dat plus starts.jsonl."""
    del geometries
    if template.get("problem") and template["problem"] != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {template['problem']}")
    cap = int(template["stop"]["max_force_calls"])
    tol = float(template["stop"]["grad_abs"])
    count = int(template.get("starts") or N_STARTS)
    if limit:
        count = min(count, int(limit))
    scratch = work or Path("/tmp/optbench-n2p2")
    cache = cache_dataset(root, problem["id"], _cache_path(scratch, problem["id"]))
    from optbench.n2p2_ml import build_library

    build_library()
    output.parent.mkdir(parents=True, exist_ok=True)
    starts_path = output.with_name(output.stem + ".starts.jsonl")
    done = _load_done(starts_path) if resume else {}
    pending = [index for index in range(count) if index not in done]
    jobs = [
        {
            "cache": str(cache),
            "index": index,
            "cap": cap,
            "tol": tol,
            "template": template["id"],
            "problem": problem["id"],
            "tuning": template.get("tuning") or "default",
        }
        for index in pending
    ]
    records = dict(done)
    if jobs:
        if workers <= 1:
            for job in jobs:
                record = _job(job)
                records[int(record["start"])] = record
                _append(starts_path, record)
                _report(record)
        else:
            context = get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(_job, job): job["index"] for job in jobs}
                for future in as_completed(futures):
                    record = future.result()
                    records[int(record["start"])] = record
                    _append(starts_path, record)
                    _report(record)
    ordered = [records[index] for index in range(count)]
    summary = score_v2(ordered, cap, seed=SEED_BASE)
    text = _benchmark_text(summary, ordered, template)
    output.write_text(text, encoding="utf-8")
    write_starts(starts_path, ordered)
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["force_calls"] = _legacy_force_calls(ordered)
    summary["nfailed"] = int(summary["nfailed"])
    return summary


def _legacy_force_calls(records: list[dict]) -> int | None:
    calls = [int(record["force_calls"]) for record in records if record["success"]]
    if not calls:
        return None
    return half_even(float(np.mean(calls)))


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _report(record: dict) -> None:
    print(
        f"start {record['start']} force_calls={record['force_calls']} "
        f"success={record['success']} failure={record['failure']} "
        f"energy_rmse={record['energy_rmse']} seconds={record['seconds']:.1f}",
        flush=True,
    )
