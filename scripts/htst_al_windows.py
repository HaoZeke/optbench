#!/usr/bin/env python3
"""hTST/Al dimer windows, one fresh AKMC per seed, counted by the harness.

A barrier seed stops when the eight reference processes (four 0.233 eV
and four 0.372 eV) have each been seen. A rate seed stops when the sum
of unique-process rates, prefactor 1.2e12 at 300 K, reaches 0.999 of
592041917. The same command resumes a seed from its work directory and
jsonl. Method settings stay the shipped ini.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "htst-al-dimer-cg2" / "windows"
WORK = ROOT / "results" / "htst-al-dimer-cg2" / "work"

BARRIERS = [0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372]
RATE_REF = 592041917.0
RATE_ACCURACY = 0.999
PREFACTOR = 1.2e12
KB = 8.6173324e-5
TEMP = 300.0
KT = KB * TEMP
THRESHOLD = RATE_REF * RATE_ACCURACY
BARRIER_TARGET = 432
RATE_TARGET = 11

CODES = {
    "svn": {
        "code": "EON (SVN r2549)",
        "code_version": "r2549",
        "commit": "r2549",
        "template": "eon-svn/htst-al-dimer-cg2-r2549",
        "client_env": "OPTBENCH_EON_SVN_CLIENT",
        "root_env": "OPTBENCH_EON_SVN_ROOT",
        "client": "/tmp/eon-svn-r2549-git/client/eonclient",
        "source": "/tmp/eon-svn-r2549-git",
    },
    "eon": {
        "code": "eOn (develop 500350d9)",
        "code_version": "500350d9",
        "commit": "500350d9fae881f06c7ef194e5868da079c4be5a",
        "template": "eon/develop-htst-al-dimer-cg2",
        "client_env": "OPTBENCH_EON_CLIENT",
        "root_env": "OPTBENCH_EON_ROOT",
        "client": "/tmp/eon-500350d9/bbdir/client/eonclient",
        "source": "/tmp/eon-500350d9",
    },
}


def _template(spec: dict) -> dict:
    return {
        "id": spec["template"],
        "engine": "eon",
        "problem": "hTST/Al",
        "code": spec["code"],
        "code_version": spec["code_version"],
        "commit": spec["commit"],
        "algorithm": "Dimer-CG rotation: max=10 tol=1",
        "reducer": "harness",
        "failure_cap": 100_000_000,
        "kernel": "eam",
        "seeds": [42],
        "starts": ["al"],
        "geometry": "website/hTST/Al/eon-dimer-cg2/pos.con",
        "pass_limit": 1,
        "workers": 1,
        "ini": {
            "Main": {"job": "akmc", "temperature": 300, "random_seed": 42},
            "Potential": {"potential": "eam_al"},
            "Optimizer": {"converged_force": 0.001, "max_iterations": 5000},
            "AKMC": {"confidence": 1.95},
            "Prefactor": {"default_value": 1.2e12},
            "Process Search": {"minimize_first": True},
            "Communicator": {
                "type": "local",
                "num_jobs": 1,
                "number_of_CPUs": 1,
                "client_path": "eonclient",
            },
            "Saddle Search": {
                "min_mode_method": "dimer",
                "displace_listed_atom_weight": 1.0,
                "displace_atom_list": "384",
                "displace_radius": 3.3,
                "displace_magnitude": 0.5,
                "max_energy": 100.0,
            },
            "Dimer": {"opt_method": "cg", "converged_angle": 1, "rotations_max": 10},
        },
    }


def _paths(code: str, seed: int) -> tuple[Path, Path, Path]:
    record = RECORD / code
    work = WORK / code / f"seed-{seed}"
    return record / f"seed-{seed}.jsonl", record / f"seed-{seed}.summary.json", work


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _windows(rows: list[dict]) -> dict:
    """First barrier window and first rate window on one seed."""
    needed = list(BARRIERS)
    ref_ids: list[int] = []
    ref_heights: list[float] = []
    seen: list[int] = []
    rate_ids: list[int] = []
    rate_sum = 0.0
    harness = 0
    client = 0
    sad = 0
    mins = 0
    barrier = None
    rate = None
    measured: list[float] = []
    for index, row in enumerate(rows):
        harness += int(row["harness_delta"])
        client += int(row["client_calls"])
        sad += int(row["sad_fcs"])
        mins += int(row["mins_fcs"])
        result = str(row["result"])
        kind, _, ident = result.partition("-")
        good = kind in {"good", "repeat"} and ident != ""
        process_id = int(ident) if good else None
        height = float(row["barrier"])
        if good and process_id not in ref_ids and process_id not in seen:
            rounded = round(height, 3)
            if rounded in needed:
                needed.remove(rounded)
                ref_ids.append(process_id)
                ref_heights.append(rounded)
        if good and process_id in ref_ids and process_id not in seen:
            seen.append(process_id)
            measured.append(height)
        if barrier is None and len(seen) == len(BARRIERS):
            barrier = {
                "jobs": index + 1,
                "force_calls": harness,
                "client_calls": client,
                "sad_fcs": sad,
                "mins_fcs": mins,
                "barrier": min(measured) if measured else None,
                "reference_barriers": list(ref_heights),
            }
        if good and rate_sum < THRESHOLD and process_id not in rate_ids:
            rate_ids.append(process_id)
            rate_sum += PREFACTOR * math.exp(-height / KT)
        if rate is None and rate_sum >= THRESHOLD:
            rate = {
                "jobs": index + 1,
                "force_calls": harness,
                "client_calls": client,
                "sad_fcs": sad,
                "mins_fcs": mins,
                "rate": rate_sum,
                "processes": len(rate_ids),
            }
    matched = all(int(row["harness_delta"]) == int(row["sad_fcs"]) + int(row["mins_fcs"]) for row in rows)
    return {
        "barrier": barrier,
        "rate": rate,
        "rate_sum": rate_sum,
        "client_matches_harness": matched if rows else None,
        "searches": len(rows),
    }


def _done(found: dict, until: str) -> bool:
    if until == "barrier":
        return found["barrier"] is not None
    if until == "rate":
        return found["rate"] is not None
    raise ValueError(until)


def _summary(spec: dict, seed: int, until: str, found: dict) -> dict:
    barrier = found["barrier"] or {}
    rate = found["rate"] or {}
    return {
        "code": spec["code"],
        "code_version": spec["code_version"],
        "commit": spec["commit"],
        "template": spec["template"],
        "seed": seed,
        "until": until,
        "barrier_jobs": barrier.get("jobs"),
        "barrier_force_calls": barrier.get("force_calls"),
        "barrier": barrier.get("barrier"),
        "client_barrier_force_calls": barrier.get("client_calls"),
        "client_barrier_sad_fcs": barrier.get("sad_fcs"),
        "client_barrier_mins_fcs": barrier.get("mins_fcs"),
        "rate_jobs": rate.get("jobs"),
        "rate_force_calls": rate.get("force_calls"),
        "rate": rate.get("rate"),
        "client_rate_force_calls": rate.get("client_calls"),
        "client_rate_sad_fcs": rate.get("sad_fcs"),
        "client_rate_mins_fcs": rate.get("mins_fcs"),
        "client_matches_harness": found["client_matches_harness"],
        "searches": found["searches"],
        "rate_sum": found["rate_sum"],
    }


def _write_summary(
    path: Path,
    spec: dict,
    seed: int,
    until: str,
    rows: list[dict],
    elapsed_s: float | None = None,
) -> dict:
    found = _windows(rows)
    payload = _summary(spec, seed, until, found)
    if elapsed_s is not None:
        payload["elapsed_s"] = elapsed_s
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _locate(spec: dict) -> tuple[Path, Path]:
    client = Path(os.environ.get(spec["client_env"], spec["client"]))
    source = Path(os.environ.get(spec["root_env"], spec["source"]))
    if not client.is_file():
        raise SystemExit(f"missing client {client}")
    if not (source / "eon").is_dir():
        raise SystemExit(f"missing eon package in {source}")
    return client, source


def _prepare_imports(client: Path, source: Path) -> None:
    lib_dirs = []
    for path in (client.parent, client.parent.parent / "lib", source / "bbdir" / "client"):
        if path.is_dir() and str(path) not in lib_dirs:
            lib_dirs.append(str(path))
    for path in (source / "bbdir").glob("**/*.so"):
        folder = str(path.parent)
        if folder not in lib_dirs:
            lib_dirs.append(folder)
    os.environ["PATH"] = str(client.parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        ["/usr/lib/gcc/x86_64-linux-gnu/13", *lib_dirs, os.environ.get("LD_LIBRARY_PATH", "")]
    )
    os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


# Clock of the run that wrote the svn seed-42 jsonl. The file in this tree
# was copied after that run, so its birth time is not the run.
SVN_SEED_42_WALL_S = 18525.879


def _birth_epoch(path: Path) -> float | None:
    try:
        raw = subprocess.check_output(["stat", "-c", "%W", str(path)], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    value = float(raw)
    return value if value > 0 else None


def _elapsed_seconds(code: str, seed: int, jsonl: Path, old: dict, searches: int) -> float | None:
    recorded = old.get("elapsed_s")
    if isinstance(recorded, (int, float)) and old.get("searches") == searches:
        return float(recorded)
    if (code, seed) == ("svn", 42):
        return SVN_SEED_42_WALL_S
    born = _birth_epoch(jsonl)
    if born is None:
        return None
    elapsed = jsonl.stat().st_mtime - born
    if elapsed <= 0:
        return None
    return round(elapsed, 3)


def _until_for(code: str, seed: int, found: dict, old: dict) -> str:
    if found["rate"] is not None:
        return "rate"
    if old.get("until") == "rate":
        return "rate"
    if code == "eon" and seed < 11:
        return "rate"
    if code == "svn" and (seed < 10 or seed == 42):
        return "rate"
    return "barrier"


def _load_rows_repair(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    rows = []
    kept = []
    dropped = False
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            dropped = True
            continue
        kept.append(line)
    if dropped:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return rows


def _refresh_summaries() -> None:
    """Rewrite each summary from its jsonl and keep a wall-clock stamp."""
    for name, spec in CODES.items():
        record = RECORD / name
        if not record.is_dir():
            continue
        for jsonl in sorted(record.glob("seed-*.jsonl"), key=lambda path: int(path.stem.split("-", 1)[1])):
            seed = int(jsonl.stem.split("-", 1)[1])
            summary_path = record / f"seed-{seed}.summary.json"
            old = {}
            if summary_path.is_file():
                try:
                    old = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    old = {}
            rows = _load_rows_repair(jsonl)
            found = _windows(rows)
            elapsed = _elapsed_seconds(name, seed, jsonl, old, len(rows))
            _write_summary(summary_path, spec, seed, _until_for(name, seed, found, old), rows, elapsed)


def _remainder(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        delta = int(row["harness_delta"]) - int(row["sad_fcs"]) - int(row["mins_fcs"])
        if delta == 0:
            continue
        kind = str(row["result"]).split("-", 1)[0]
        key = f"{kind}:{delta:+d}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def collect(code: str | None = None) -> dict:
    """Windows already on disk. A rate seed also contributes its barrier window."""
    codes = [code] if code else list(CODES)
    document = {}
    for name in codes:
        spec = CODES[name]
        rows = []
        parsed_rows: list[list[dict]] = []
        record = RECORD / name
        elapsed: dict[int, float] = {}
        if record.is_dir():
            for path in sorted(record.glob("seed-*.jsonl"), key=lambda item: int(item.stem.split("-", 1)[1])):
                parsed = _load_rows(path)
                if not parsed:
                    continue
                seed = int(path.stem.split("-", 1)[1])
                rows.append((seed, _windows(parsed)))
                parsed_rows.append(parsed)
                summary_path = record / f"seed-{seed}.summary.json"
                if summary_path.is_file():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        summary = {}
                    if isinstance(summary.get("elapsed_s"), (int, float)):
                        elapsed[seed] = float(summary["elapsed_s"])
        barriers = [(item[0], item[1]["barrier"]) for item in rows if item[1]["barrier"]]
        rates = [(item[0], item[1]["rate"]) for item in rows if item[1]["rate"]]
        matched = [item[1]["client_matches_harness"] for item in rows if item[1]["searches"]]
        remainder: dict[str, int] = {}
        search_count = 0
        for parsed in parsed_rows:
            search_count += len(parsed)
            for key, count in _remainder(parsed).items():
                remainder[key] = remainder.get(key, 0) + count
        barrier_values = [item[1] for item in barriers]
        rate_values = [item[1] for item in rates]
        stopped = []
        for seed, found in rows:
            barrier = found["barrier"]
            if barrier and found["searches"] == barrier["jobs"] and seed in elapsed:
                stopped.append((elapsed[seed], barrier["jobs"]))
        rate_wall = []
        for seed, rate in rates:
            if seed in elapsed:
                rate_wall.append((elapsed[seed], rate["jobs"]))
        windows = []
        for seed, found in rows:
            barrier = found["barrier"] or {}
            rate = found["rate"] or {}
            windows.append(
                {
                    "seed": seed,
                    "searches": found["searches"],
                    "elapsed_s": elapsed.get(seed),
                    "client_matches_harness": found["client_matches_harness"],
                    "barrier_jobs": barrier.get("jobs"),
                    "barrier_force_calls": barrier.get("force_calls"),
                    "barrier": barrier.get("barrier"),
                    "client_barrier_sad_fcs": barrier.get("sad_fcs"),
                    "client_barrier_mins_fcs": barrier.get("mins_fcs"),
                    "rate_jobs": rate.get("jobs"),
                    "rate_force_calls": rate.get("force_calls"),
                    "rate": rate.get("rate"),
                    "client_rate_sad_fcs": rate.get("sad_fcs"),
                    "client_rate_mins_fcs": rate.get("mins_fcs"),
                    "rate_sum": found["rate_sum"],
                }
            )
        document[name] = {
            "code": spec["code"],
            "commit": spec["commit"],
            "seeds": len(rows),
            "searches": search_count,
            "barrier_windows": len(barriers),
            "barrier_target": BARRIER_TARGET,
            "rate_windows": len(rates),
            "rate_target": RATE_TARGET,
            "client_matches_harness": all(matched) if matched else None,
            "client_remainder": remainder,
            "barrier_jobs": _stats([float(item["jobs"]) for item in barrier_values]),
            "barrier_force_calls": _stats([float(item["force_calls"]) for item in barrier_values]),
            "barrier": _stats([float(item["barrier"]) for item in barrier_values if item.get("barrier") is not None]),
            "client_barrier_force_calls": _stats([float(item["client_calls"]) for item in barrier_values]),
            "client_barrier_sad_fcs": _stats([float(item["sad_fcs"]) for item in barrier_values]),
            "client_barrier_mins_fcs": _stats([float(item["mins_fcs"]) for item in barrier_values]),
            "rate_jobs": _stats([float(item["jobs"]) for item in rate_values]),
            "rate_force_calls": _stats([float(item["force_calls"]) for item in rate_values]),
            "rate": _stats([float(item["rate"]) for item in rate_values]),
            "client_rate_force_calls": _stats([float(item["client_calls"]) for item in rate_values]),
            "client_rate_sad_fcs": _stats([float(item["sad_fcs"]) for item in rate_values]),
            "client_rate_mins_fcs": _stats([float(item["mins_fcs"]) for item in rate_values]),
            "barrier_wall_s": _stats([item[0] for item in stopped]),
            "barrier_seconds_per_search": _stats([item[0] / item[1] for item in stopped if item[1]]),
            "rate_wall_s": _stats([item[0] for item in rate_wall]),
            "rate_seconds_per_search": _stats([item[0] / item[1] for item in rate_wall if item[1]]),
            "windows": windows,
        }
    return document


def run_seed(code: str, seed: int, until: str, limit: int | None = None) -> dict:
    spec = CODES[code]
    jsonl_path, summary_path, work = _paths(code, seed)
    rows = _load_rows(jsonl_path)
    found = _windows(rows)
    if _done(found, until) or (limit is not None and len(rows) >= limit and rows):
        payload = _write_summary(summary_path, spec, seed, until, rows)
        print(json.dumps(payload), flush=True)
        return payload
    client, source = _locate(spec)
    _prepare_imports(client, source)
    sys.path.insert(0, str(ROOT))
    from optbench.campaign import geometry_archive
    from optbench.eon_akmc import _write_config
    from optbench.eon_harness import bridge_binary, eam_al, ensure_eam
    from optbench.potserver import PotServer, remote_count

    ensure_eam()
    label = _template(spec)
    akmc_dir = work / "akmc"
    if not akmc_dir.is_dir():
        if rows:
            _write_summary(summary_path, spec, seed, until, rows)
            print(
                f"{code} seed {seed} has {len(rows)} searches recorded and no work directory",
                flush=True,
            )
            raise SystemExit(2)
        akmc_dir.mkdir(parents=True)
        for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
            (akmc_dir / relative).mkdir(parents=True, exist_ok=True)
        geometry = geometry_archive(ROOT, {"id": label["geometry"], "input": {"archive_path": label["geometry"]}})
        shutil.copy(geometry, akmc_dir / "pos.con")
        patched = json.loads(json.dumps(label))
        patched["ini"]["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(bridge_binary())}
        patched["ini"]["Main"]["random_seed"] = seed
        _write_config(patched, akmc_dir / "config.ini", client, seed, 1)
    import inspect
    import logging

    for name in ("akmc", "eon", "state", "communicator", "explorer"):
        logging.getLogger(name).setLevel(logging.ERROR)
    server = PotServer(eam_al, cap=int(label["failure_cap"])).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    previous = Path.cwd()
    os.chdir(akmc_dir)
    try:
        from eon import akmc as akmc_mod
        from eon.config import ConfigClass
        from eon.config import config as shared_config

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
        costs = [int(row["harness_delta"]) for row in rows]
        accounted = int(rows[-1]["accounted"]) if rows else 0
        offset = int(rows[-1]["harness_exit"]) if rows else 0
        calls = len(rows)
        while True:
            if limit is not None and len(costs) >= limit:
                break
            if _done(_windows(rows), until):
                break
            calls += 1
            h_enter = offset + int(remote_count(server.host, server.port)[0])
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                print(f"{code} seed {seed} akmc exit {exc.code}", flush=True)
                break
            except Exception:
                print(f"{code} seed {seed} akmc error\n{traceback.format_exc()}", flush=True)
                break
            h_exit = offset + int(remote_count(server.host, server.port)[0])
            result_path = akmc_dir / "states" / "0" / "search_results.txt"
            result_text = result_path.read_text(encoding="utf-8", errors="replace") if result_path.is_file() else ""
            lines = []
            for line in result_text.splitlines():
                fields = line.split()
                if len(fields) < 6 or fields[0] in {"wuid", "----"} or set(fields[0]) <= {"-"}:
                    continue
                if fields[2][:1].isdigit():
                    lines.append(fields)
            new = len(lines) - len(costs)
            if new < 0:
                raise RuntimeError(f"search_results shrank from {len(costs)} to {len(lines)}")
            if new > 1:
                raise RuntimeError(f"registered {new} searches in one pass")
            if new == 0:
                print(f"{code} seed {seed} call {calls} registered 0", flush=True)
                continue
            delta = h_enter - accounted
            if delta < 0:
                raise RuntimeError(f"negative harness delta {delta}")
            fields = lines[-1]
            row = {
                "seed": seed,
                "jobs": len(costs) + 1,
                "harness_delta": delta,
                "harness_exit": h_exit,
                "accounted": h_enter,
                "client_calls": int(fields[4]) + int(fields[5]),
                "sad_fcs": int(fields[4]),
                "mins_fcs": int(fields[5]),
                "barrier": float(fields[2]),
                "result": fields[-1],
            }
            costs.append(delta)
            accounted = h_enter
            rows.append(row)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            found = _windows(rows)
            print(
                f"{code} seed {seed} job {row['jobs']} barrier {fields[2]} {fields[-1]} "
                f"harness {delta} client {row['client_calls']} "
                f"ref {0 if not found['barrier'] else 8}/8 rate {found['rate_sum']:.6e}",
                flush=True,
            )
        payload = _write_summary(summary_path, spec, seed, until, rows)
        print("SUMMARY " + json.dumps(payload), flush=True)
        return payload
    finally:
        os.chdir(previous)
        server.stop()


def _parse_seeds(text: str) -> list[int]:
    seeds = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" in piece:
            start, stop = piece.split(":", 1)
            seeds.extend(range(int(start), int(stop)))
        else:
            seeds.append(int(piece))
    return seeds


def shard(code: str, seeds: list[int], until: str, jobs: int) -> int:
    pending = []
    for seed in seeds:
        jsonl_path, _summary_path, _work = _paths(code, seed)
        found = _windows(_load_rows(jsonl_path))
        if not _done(found, until):
            pending.append(seed)
    if not pending:
        print(f"{code} nothing to run", flush=True)
        return 0
    print(f"{code} seeds {len(pending)} until {until} jobs {jobs}", flush=True)
    running: list[tuple[int, subprocess.Popen]] = []
    failures = 0
    script = str(Path(__file__).resolve())

    def launch(seed: int) -> None:
        log_path = RECORD / code / f"seed-{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-u", script, "run", "--code", code, "--seed", str(seed), "--until", until],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running.append((seed, process))

    cursor = 0
    while cursor < len(pending) or running:
        while cursor < len(pending) and len(running) < jobs:
            launch(pending[cursor])
            cursor += 1
        time.sleep(2)
        still = []
        for seed, process in running:
            status = process.poll()
            if status is None:
                still.append((seed, process))
            elif status == 2:
                print(f"{code} seed {seed} held", flush=True)
            elif status != 0:
                failures += 1
                print(f"{code} seed {seed} exit {status}", flush=True)
            else:
                print(f"{code} seed {seed} done", flush=True)
        running = still
    return failures


ARCHIVE = {
    "barrier_jobs": 46,
    "barrier_force_calls": 21070,
    "barrier_windows": BARRIER_TARGET,
    "rate_jobs": 1671,
    "rate_force_calls": 766300,
    "rate_windows": RATE_TARGET,
    "barrier_eV": 0.233,
    "rate_ref": RATE_REF,
    "force_calls": "sad-fcs",
}


def _print_report(document: dict) -> None:
    archive = document.get("archive", ARCHIVE)
    for name, block in document.items():
        if name not in CODES:
            continue
        print(f"{block['code']} barrier {block['barrier_windows']}/{block['barrier_target']} rate {block['rate_windows']}/{block['rate_target']}")
        print(f"  client_matches_harness {block['client_matches_harness']} remainder {block['client_remainder']}")
        for key in (
            "barrier_jobs",
            "barrier_force_calls",
            "client_barrier_sad_fcs",
            "client_barrier_mins_fcs",
            "barrier",
            "barrier_wall_s",
            "barrier_seconds_per_search",
            "rate_jobs",
            "rate_force_calls",
            "client_rate_sad_fcs",
            "client_rate_mins_fcs",
            "rate",
            "rate_wall_s",
            "rate_seconds_per_search",
        ):
            stats = block[key]
            archived = archive.get(key)
            if stats is None:
                print(f"  {key} none archive {archived}")
                continue
            print(
                f"  {key} n={stats['n']} mean={stats['mean']:.6g} median={stats['median']:.6g} "
                f"min={stats['min']:.6g} max={stats['max']:.6g} archive={archived}"
            )


def _self_test() -> None:
    rows = []
    order = [1, 4, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    heights = {1: 0.233, 2: 0.372, 3: 0.233, 4: 0.500, 5: 0.233, 6: 0.233, 7: 0.372, 8: 0.372, 9: 0.372}
    for index, process_id in enumerate(order):
        height = heights.get(process_id, 0.372)
        rows.append(
            {
                "harness_delta": 100,
                "client_calls": 100,
                "sad_fcs": 80,
                "mins_fcs": 20,
                "barrier": height,
                "result": f"good-{process_id}",
            }
        )
    found = _windows(rows)
    if found["barrier"] is None or found["barrier"]["jobs"] != 9:
        raise SystemExit(f"self-test barrier {found['barrier']}")
    if found["barrier"]["force_calls"] != 900:
        raise SystemExit(f"self-test barrier calls {found['barrier']}")
    if found["rate"] is None or found["rate"]["jobs"] <= found["barrier"]["jobs"]:
        raise SystemExit(f"self-test rate {found['rate']} sum {found['rate_sum']}")
    if found["client_matches_harness"] is not True:
        raise SystemExit("self-test match")
    mismatch = json.loads(json.dumps(rows[-1]))
    mismatch["harness_delta"] = 101
    if _windows(rows[:-1] + [mismatch])["client_matches_harness"] is not False:
        raise SystemExit("self-test mismatch")
    stats = _stats([3, 1, 2])
    if stats != {"n": 3, "mean": 2, "median": 2, "min": 1, "max": 3}:
        raise SystemExit(f"self-test stats {stats}")
    print("self-test ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hTST/Al dimer windows")
    parser.add_argument("command", choices=("run", "shard", "report", "selftest"))
    parser.add_argument("--code", choices=tuple(CODES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--until", choices=("barrier", "rate"), default="barrier")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command == "selftest":
        _self_test()
        return 0
    if args.command == "report":
        _refresh_summaries()
        document = collect(args.code)
        document["archive"] = dict(ARCHIVE)
        destination = RECORD / "aggregate.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        _print_report(document)
        print(f"wrote {destination}")
        return 0
    if args.code is None:
        raise SystemExit("--code is required")
    if args.command == "run":
        if args.seed is None:
            raise SystemExit("--seed is required")
        run_seed(args.code, args.seed, args.until, args.limit)
        return 0
    if not args.seeds:
        raise SystemExit("--seeds is required")
    return shard(args.code, _parse_seeds(args.seeds), args.until, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
