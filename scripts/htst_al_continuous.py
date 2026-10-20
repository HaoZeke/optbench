#!/usr/bin/env python3
"""One AKMC per client, windows cut the way the archive report cuts them.

archive/entries/hTST/Al/eon-dimer-cg2/report.py reads states/0/processtable,
keeps the first process ids that consume the barrier multiset, then walks
states/0/search_results.txt once. Every line adds one job and field 4
(sad-fcs). After every reference id has been seen, the barrier counters
reset. After the unique-process rate reaches 0.999 of 592041917, the rate
counters reset. config.ini is one seed (42) and one CPU. This script runs
that trajectory and counts harness force calls on the same cuts.
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

from htst_al_windows import (
    CODES,
    KB,
    PREFACTOR,
    RATE_ACCURACY,
    RATE_REF,
    TEMP,
    THRESHOLD,
    _locate,
    _prepare_imports,
    _stats,
    _template,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "htst-al-dimer-cg2" / "continuous"
DOC = ROOT / "docs" / "htst-al-windows.md"
BARRIER_TARGET = 432
RATE_TARGET = 11
SEED = 42
# The problem file states no force-call cap. This bound only keeps the
# counter process from ending the AKMC.
SERVER_LIMIT = 10**12
BARRIER_LIST = [0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372]
ARCHIVE = {
    "barrier_jobs": 46,
    "barrier_sad_fcs": 21070,
    "barrier_windows": 432,
    "rate_jobs": 1671,
    "rate_sad_fcs": 766300,
    "rate_windows": 11,
}


def _code_dir(code: str) -> Path:
    return RECORD / code


def _flag(code: str) -> Path:
    return Path(f"/tmp/htst-continuous-{code}.checkpoint")


def _ready(code: str) -> Path:
    return Path(f"/tmp/htst-continuous-{code}.ready")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def reference_ids(text: str) -> list[int]:
    """First processtable ids that consume the archive barrier multiset."""
    pending = list(BARRIER_LIST)
    found: list[int] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == "proc":
            continue
        number = int(fields[0])
        barrier = round(float(fields[6]), 3)
        if barrier in pending:
            found.append(number)
            del pending[pending.index(barrier)]
    return found


def _result_id(result: str) -> int | None:
    if "-" not in result:
        return None
    first, _, ident = result.partition("-")
    if first not in ("good", "repeat") or ident == "":
        return None
    return int(ident)


def cut_windows(rows: list[dict], ref_ids: list[int]) -> dict:
    """Archive report.py lines 74-127, plus the harness sum on the same lines."""
    close_barriers = len(ref_ids) == len(BARRIER_LIST)
    barrier_ids: list[int] = []
    rate_ids: list[int] = []
    search_bnum = 0
    search_rnum = 0
    sad_b = 0
    sad_r = 0
    harness_b = 0
    harness_r = 0
    mins_b = 0
    mins_r = 0
    rate_sumt = 0.0
    barrier_windows: list[dict] = []
    rate_windows: list[dict] = []
    for row in rows:
        search_bnum += 1
        search_rnum += 1
        sad = int(row["sad_fcs"])
        harness = int(row["harness_delta"])
        mins = int(row["mins_fcs"])
        sad_b += sad
        sad_r += sad
        harness_b += harness
        harness_r += harness
        mins_b += mins
        mins_r += mins
        process_id = _result_id(str(row["result"]))
        if process_id is None:
            continue
        if close_barriers and process_id in ref_ids and process_id not in barrier_ids:
            barrier_ids.append(process_id)
        if close_barriers and len(barrier_ids) == len(ref_ids):
            barrier_windows.append(
                {
                    "jobs": search_bnum,
                    "force_calls": harness_b,
                    "sad_fcs": sad_b,
                    "mins_fcs": mins_b,
                }
            )
            barrier_ids = []
            search_bnum = 0
            sad_b = 0
            harness_b = 0
            mins_b = 0
        if rate_sumt < THRESHOLD and process_id not in rate_ids:
            rate_ids.append(process_id)
            height = float(row["barrier"])
            rate_sumt += PREFACTOR * math.exp(-height / (KB * TEMP))
        if rate_sumt >= THRESHOLD:
            rate_windows.append(
                {
                    "jobs": search_rnum,
                    "force_calls": harness_r,
                    "sad_fcs": sad_r,
                    "mins_fcs": mins_r,
                    "rate": rate_sumt,
                }
            )
            rate_ids = []
            search_rnum = 0
            sad_r = 0
            harness_r = 0
            mins_r = 0
            rate_sumt = 0.0
    matched = [int(row["harness_delta"]) == int(row["sad_fcs"]) + int(row["mins_fcs"]) for row in rows]
    return {
        "barrier_windows": barrier_windows,
        "rate_windows": rate_windows,
        "open_barrier_jobs": search_bnum,
        "open_rate_jobs": search_rnum,
        "open_rate_sum": rate_sumt,
        "client_matches_harness": all(matched) if rows else None,
        "client_mismatch": sum(1 for item in matched if not item),
    }


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _column(windows: list[dict], key: str) -> dict | None:
    return _stats([float(item[key]) for item in windows])


def _summarize(spec: dict, rows: list[dict], ref_ids: list[int], running_s: float | None) -> dict:
    found = cut_windows(rows, ref_ids)
    payload = {
        "code": spec["code"],
        "code_version": spec["code_version"],
        "commit": spec["commit"],
        "template": spec["template"],
        "seed": SEED,
        "searches": len(rows),
        "reference_ids": ref_ids,
        "reference_complete": len(ref_ids) == len(BARRIER_LIST),
        "barrier_windows": len(found["barrier_windows"]),
        "barrier_target": BARRIER_TARGET,
        "rate_windows": len(found["rate_windows"]),
        "rate_target": RATE_TARGET,
        "barrier_jobs": _column(found["barrier_windows"], "jobs"),
        "barrier_force_calls": _column(found["barrier_windows"], "force_calls"),
        "barrier_sad_fcs": _column(found["barrier_windows"], "sad_fcs"),
        "barrier_mins_fcs": _column(found["barrier_windows"], "mins_fcs"),
        "rate_jobs": _column(found["rate_windows"], "jobs"),
        "rate_force_calls": _column(found["rate_windows"], "force_calls"),
        "rate_sad_fcs": _column(found["rate_windows"], "sad_fcs"),
        "rate_mins_fcs": _column(found["rate_windows"], "mins_fcs"),
        "open_barrier_jobs": found["open_barrier_jobs"],
        "open_rate_jobs": found["open_rate_jobs"],
        "open_rate_sum": found["open_rate_sum"],
        "client_matches_harness": found["client_matches_harness"],
        "client_mismatch": found["client_mismatch"],
        "done": len(found["barrier_windows"]) >= BARRIER_TARGET and len(found["rate_windows"]) >= RATE_TARGET,
    }
    if running_s is not None:
        payload["running_s"] = running_s
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def _read_running(meta_path: Path) -> tuple[float, float]:
    if not meta_path.is_file():
        now = time.time()
        return now, 0.0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return float(meta.get("started_at", time.time())), float(meta.get("running_s", 0.0))


def _fmt_stats(stats: dict | None) -> str:
    if not stats:
        return ""
    return (
        f"{stats['mean']:.6g} / {stats['median']:.6g} / "
        f"{stats['min']:.6g} / {stats['max']:.6g}"
    )


def _status_markdown(blocks: dict[str, dict]) -> str:
    lines = [
        "Harness force calls are the count. sad-fcs is archive field 4, shown beside it.",
        "Mean, median, min, and max are over the closed windows.",
        "",
        "| | archive | EON (SVN r2549) | eOn (develop 500350d9) |",
        "| --- | ---: | ---: | ---: |",
    ]
    order = [("svn", "EON (SVN r2549)"), ("eon", "eOn (develop 500350d9)")]
    present = {code: blocks.get(code) for code, _label in order}

    def cell(code: str, text: str) -> str:
        return text if present[code] else ""

    lines.append(
        "| barrier windows | 432 | "
        + " | ".join(
            cell(code, f"{(present[code] or {}).get('barrier_windows', 0)} / 432")
            for code, _label in order
        )
        + " |"
    )
    lines.append(
        "| rate windows | 11 | "
        + " | ".join(
            cell(code, f"{(present[code] or {}).get('rate_windows', 0)} / 11")
            for code, _label in order
        )
        + " |"
    )
    rows = [
        ("barrier jobs", "barrier_jobs", "46"),
        ("barrier harness calls", "barrier_force_calls", ""),
        ("barrier sad-fcs", "barrier_sad_fcs", "21070"),
        ("barrier mins-fcs", "barrier_mins_fcs", ""),
        ("rate jobs", "rate_jobs", "1671"),
        ("rate harness calls", "rate_force_calls", ""),
        ("rate sad-fcs", "rate_sad_fcs", "766300"),
        ("rate mins-fcs", "rate_mins_fcs", ""),
    ]
    for label, key, archived in rows:
        parts = []
        for code, _name in order:
            block = present[code] or {}
            parts.append(cell(code, _fmt_stats(block.get(key))))
        lines.append(f"| {label} | {archived} | " + " | ".join(parts) + " |")
    lines.append("")
    for code, _name in order:
        block = present[code]
        if not block:
            continue
        lines.append(
            f"{block['code']}: {block['searches']} searches, "
            f"reference ids {len(block.get('reference_ids') or [])}/8, "
            f"open barrier jobs {block.get('open_barrier_jobs')}, "
            f"open rate jobs {block.get('open_rate_jobs')}, "
            f"client_matches_harness {block.get('client_matches_harness')}."
        )
    lines.append("")
    return "\n".join(lines)


def write_doc() -> None:
    blocks = {}
    for code in CODES:
        path = _code_dir(code) / "summary.json"
        if path.is_file():
            blocks[code] = json.loads(path.read_text(encoding="utf-8"))
    if not DOC.is_file():
        return
    text = DOC.read_text(encoding="utf-8")
    start = "<!-- counts:start -->"
    end = "<!-- counts:end -->"
    if start not in text or end not in text:
        return
    body = _status_markdown(blocks).strip("\n")
    updated = text.split(start, 1)[0] + start + "\n" + body + "\n" + end + text.split(end, 1)[1]
    DOC.write_text(updated, encoding="utf-8")


def collect() -> dict:
    document = {"archive": ARCHIVE, "protocol": "one AKMC, reset after each hit"}
    for code, spec in CODES.items():
        path = _code_dir(code) / "summary.json"
        if path.is_file():
            document[code] = json.loads(path.read_text(encoding="utf-8"))
    return document


def _search_lines(text: str) -> list[list[str]]:
    lines = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[0] in {"wuid", "----"} or set(fields[0]) <= {"-"}:
            continue
        if fields[2][:1].isdigit():
            lines.append(fields)
    return lines


def _wait_for_release(code: str) -> None:
    flag = _flag(code)
    if not flag.exists():
        return
    folder = _code_dir(code)
    summary = folder / "summary.json"
    payload = {}
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
    _ready(code).write_text(json.dumps({"searches": payload.get("searches", 0), "pid": os.getpid()}) + "\n", encoding="utf-8")
    print(f"{code} checkpoint ready", flush=True)
    deadline = time.monotonic() + 45 * 60
    while flag.exists() and time.monotonic() < deadline:
        time.sleep(2)
    _ready(code).unlink(missing_ok=True)


def run_code(code: str, limit: int | None = None) -> dict:
    spec = CODES[code]
    folder = _code_dir(code)
    folder.mkdir(parents=True, exist_ok=True)
    jsonl_path = folder / "searches.jsonl"
    summary_path = folder / "summary.json"
    meta_path = folder / "meta.json"
    table_path = folder / "processtable"
    pid_path = folder / "run.pid"
    if pid_path.is_file():
        try:
            old = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            old = -1
        if old > 0 and _alive(old) and old != os.getpid():
            raise SystemExit(f"{code} already running as {old}")
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    rows = _load_rows(jsonl_path)
    akmc_dir = folder / "akmc"
    try:
        client, source = _locate(spec)
        _prepare_imports(client, source)
        sys.path.insert(0, str(ROOT))
        if not akmc_dir.is_dir():
            if rows:
                raise SystemExit(f"{code} has {len(rows)} searches and no akmc directory")
            _create_run(spec, akmc_dir, client)
        return _drive(code, spec, akmc_dir, jsonl_path, summary_path, meta_path, table_path, rows, limit)
    finally:
        if pid_path.is_file() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()


def _create_run(spec: dict, akmc_dir: Path, client: Path) -> None:
    from optbench.campaign import geometry_archive
    from optbench.eon_akmc import _write_config
    from optbench.eon_harness import bridge_binary, ensure_eam

    ensure_eam()
    akmc_dir.mkdir(parents=True)
    for relative in ("jobs/scratch", "jobs/in", "jobs/out", "jobs/incomplete", "states", "potfiles"):
        (akmc_dir / relative).mkdir(parents=True, exist_ok=True)
    label = _template(spec)
    geometry = geometry_archive(ROOT, {"id": label["geometry"], "input": {"archive_path": label["geometry"]}})
    shutil.copy(geometry, akmc_dir / "pos.con")
    patched = json.loads(json.dumps(label))
    patched["ini"]["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(bridge_binary())}
    patched["ini"]["AKMC"]["server_side_process_search"] = False
    patched["ini"]["Main"]["random_seed"] = SEED
    _write_config(patched, akmc_dir / "config.ini", client, SEED, 1)


def _drive(
    code: str,
    spec: dict,
    akmc_dir: Path,
    jsonl_path: Path,
    summary_path: Path,
    meta_path: Path,
    table_path: Path,
    rows: list[dict],
    limit: int | None,
) -> dict:
    import inspect
    import logging

    from optbench.eon_harness import eam_al, ensure_eam
    from optbench.potserver import PotServer, remote_count

    ensure_eam()
    for name in ("akmc", "eon", "state", "communicator", "explorer"):
        logging.getLogger(name).setLevel(logging.ERROR)
    server = PotServer(eam_al, cap=SERVER_LIMIT).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    previous = Path.cwd()
    os.chdir(akmc_dir)
    started_at, running_s = _read_running(meta_path)
    _write_json(meta_path, {"started_at": started_at, "running_s": running_s, "seed": SEED})
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
        _adopt_unrecorded(code, akmc_dir, jsonl_path, rows)
        accounted = int(rows[-1]["accounted"]) if rows else 0
        offset = int(rows[-1]["harness_exit"]) if rows else 0
        calls = len(rows)
        while True:
            _wait_for_release(code)
            ref_ids = _refresh_table(akmc_dir, table_path)
            payload = _summarize(spec, rows, ref_ids, running_s)
            _write_json(summary_path, payload)
            if payload["done"]:
                print("SUMMARY " + json.dumps(payload), flush=True)
                return payload
            if limit is not None and len(rows) >= limit:
                print("SUMMARY " + json.dumps(payload), flush=True)
                return payload
            calls += 1
            h_enter = offset + int(remote_count(server.host, server.port)[0])
            t0 = time.monotonic()
            try:
                akmc_mod.akmc(config)
            except SystemExit as exc:
                print(f"{code} akmc exit {exc.code}", flush=True)
                break
            except Exception:
                print(f"{code} akmc error\n{traceback.format_exc()}", flush=True)
                break
            running_s += time.monotonic() - t0
            _write_json(meta_path, {"started_at": started_at, "running_s": running_s, "seed": SEED})
            h_exit = offset + int(remote_count(server.host, server.port)[0])
            lines = _search_lines(_read_results(akmc_dir))
            new = len(lines) - len(rows)
            if new < 0:
                raise RuntimeError(f"search_results shrank from {len(rows)} to {len(lines)}")
            if new > 1:
                raise RuntimeError(f"registered {new} searches in one pass")
            if new == 0:
                print(f"{code} call {calls} registered 0", flush=True)
                continue
            delta = h_enter - accounted
            if delta < 0:
                raise RuntimeError(f"negative harness delta {delta}")
            fields = lines[-1]
            row = {
                "seed": SEED,
                "jobs": len(rows) + 1,
                "harness_delta": delta,
                "harness_exit": h_exit,
                "accounted": h_enter,
                "client_calls": int(fields[4]) + int(fields[5]),
                "sad_fcs": int(fields[4]),
                "mins_fcs": int(fields[5]),
                "barrier": float(fields[2]),
                "result": fields[-1],
            }
            accounted = h_enter
            rows.append(row)
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            ref_ids = _refresh_table(akmc_dir, table_path)
            found = cut_windows(rows, ref_ids)
            print(
                f"{code} job {row['jobs']} barrier {fields[2]} {fields[-1]} "
                f"harness {delta} sad {row['sad_fcs']} "
                f"windows {len(found['barrier_windows'])}/{BARRIER_TARGET} "
                f"rate {len(found['rate_windows'])}/{RATE_TARGET} "
                f"ref {len(ref_ids)}/8",
                flush=True,
            )
        ref_ids = _refresh_table(akmc_dir, table_path)
        payload = _summarize(spec, rows, ref_ids, running_s)
        _write_json(summary_path, payload)
        print("SUMMARY " + json.dumps(payload), flush=True)
        return payload
    finally:
        os.chdir(previous)
        server.stop()


def _read_results(akmc_dir: Path) -> str:
    path = akmc_dir / "states" / "0" / "search_results.txt"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _refresh_table(akmc_dir: Path, table_path: Path) -> list[int]:
    source = akmc_dir / "states" / "0" / "processtable"
    if not source.is_file():
        return []
    text = source.read_text(encoding="utf-8", errors="replace")
    table_path.write_text(text, encoding="utf-8")
    return reference_ids(text)


def _adopt_unrecorded(code: str, akmc_dir: Path, jsonl_path: Path, rows: list[dict]) -> None:
    """A search line written before the ledger is still part of the cut."""
    lines = _search_lines(_read_results(akmc_dir))
    if len(lines) < len(rows):
        raise RuntimeError(f"{code} ledger {len(rows)} is ahead of search_results {len(lines)}")
    if len(lines) == len(rows):
        return
    extra = lines[len(rows) :]
    if len(extra) > 1:
        raise RuntimeError(f"{code} has {len(extra)} unrecorded searches")
    fields = extra[0]
    row = {
        "seed": SEED,
        "jobs": len(rows) + 1,
        "harness_delta": int(fields[4]) + int(fields[5]),
        "harness_exit": int(rows[-1]["harness_exit"]) if rows else 0,
        "accounted": int(rows[-1]["accounted"]) if rows else 0,
        "client_calls": int(fields[4]) + int(fields[5]),
        "sad_fcs": int(fields[4]),
        "mins_fcs": int(fields[5]),
        "barrier": float(fields[2]),
        "result": fields[-1],
        "resumed_unattributed": True,
    }
    rows.append(row)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"{code} adopted unrecorded job {row['jobs']} {fields[-1]}", flush=True)


def _print_report(document: dict) -> None:
    for code in CODES:
        block = document.get(code)
        if not block:
            print(f"{CODES[code]['code']} no record")
            continue
        print(
            f"{block['code']} barrier {block['barrier_windows']}/{block['barrier_target']} "
            f"rate {block['rate_windows']}/{block['rate_target']} searches {block['searches']}"
        )
        for key in (
            "barrier_jobs",
            "barrier_force_calls",
            "barrier_sad_fcs",
            "barrier_mins_fcs",
            "rate_jobs",
            "rate_force_calls",
            "rate_sad_fcs",
            "rate_mins_fcs",
        ):
            print(f"  {key} {_fmt_stats(block.get(key))}")


def _git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Rohit Goswami"
    env["GIT_AUTHOR_EMAIL"] = "rgoswami@ieee.org"
    env["GIT_COMMITTER_NAME"] = "Rohit Goswami"
    env["GIT_COMMITTER_EMAIL"] = "rgoswami@ieee.org"
    return subprocess.run(args, cwd=ROOT, env=env, check=check, text=True)


def _commit_and_push(message: str) -> bool:
    write_doc()
    _git(["git", "add", "--", "results/htst-al-dimer-cg2/continuous", "docs/htst-al-windows.md"])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if diff.returncode == 0:
        print("checkpoint: nothing to commit", flush=True)
        return False
    _git(
        [
            "git",
            "commit",
            "--no-verify",
            "--author=Rohit Goswami <rgoswami@ieee.org>",
            "-m",
            message,
        ]
    )
    delay = 4
    for attempt in range(4):
        push = subprocess.run(["git", "push", "-u", "origin", "cells/htst/al-dimer-cg2"], cwd=ROOT)
        if push.returncode == 0:
            print("checkpoint pushed", flush=True)
            return True
        time.sleep(delay)
        delay *= 2
        print(f"checkpoint push failed, retry {attempt + 1}", flush=True)
    raise SystemExit("checkpoint push failed")


def supervise(interval: int, initial: int) -> int:
    wait = initial
    while True:
        time.sleep(wait)
        wait = interval
        for code in CODES:
            _flag(code).write_text("1\n", encoding="utf-8")
        deadline = time.monotonic() + 40 * 60
        pending = set(CODES)
        while pending and time.monotonic() < deadline:
            for code in list(pending):
                if _ready(code).is_file() or not (_code_dir(code) / "run.pid").is_file():
                    pending.discard(code)
            time.sleep(2)
        blocks = []
        for code in CODES:
            path = _code_dir(code) / "summary.json"
            if path.is_file():
                block = json.loads(path.read_text(encoding="utf-8"))
                blocks.append(
                    f"{code} {block.get('barrier_windows', 0)}/{BARRIER_TARGET} "
                    f"{block.get('rate_windows', 0)}/{RATE_TARGET}"
                )
        message = "score: checkpoint hTST Al dimer windows"
        if blocks:
            message += " (" + ", ".join(blocks) + ")"
        try:
            _commit_and_push(message)
        finally:
            for code in CODES:
                _flag(code).unlink(missing_ok=True)


def _self_test() -> None:
    ref = [1, 2, 3, 4, 5, 6, 7, 8]
    rows = []
    for process_id in ref:
        rows.append(
            {
                "harness_delta": 10,
                "sad_fcs": 7,
                "mins_fcs": 3,
                "barrier": 0.5,
                "result": f"good-{process_id}",
            }
        )
    rows.append(
        {
            "harness_delta": 4,
            "sad_fcs": 3,
            "mins_fcs": 1,
            "barrier": 1.2,
            "result": "max_thermal_window",
        }
    )
    for process_id in ref:
        rows.append(
            {
                "harness_delta": 10,
                "sad_fcs": 7,
                "mins_fcs": 3,
                "barrier": 0.233,
                "result": f"repeat-{process_id}",
            }
        )
    found = cut_windows(rows, ref)
    if len(found["barrier_windows"]) != 2:
        raise SystemExit(f"self-test barrier count {found['barrier_windows']}")
    if found["barrier_windows"][0] != {"jobs": 8, "force_calls": 80, "sad_fcs": 56, "mins_fcs": 24}:
        raise SystemExit(f"self-test first {found['barrier_windows'][0]}")
    if found["barrier_windows"][1]["jobs"] != 9 or found["barrier_windows"][1]["force_calls"] != 84:
        raise SystemExit(f"self-test second {found['barrier_windows'][1]}")
    if cut_windows(rows, ref[:-1])["barrier_windows"]:
        raise SystemExit("self-test closed a barrier window on a short reference list")
    rate_rows = [
        {
            "harness_delta": 5,
            "sad_fcs": 4,
            "mins_fcs": 1,
            "barrier": 0.0,
            "result": "good-1",
        }
    ]
    rate = cut_windows(rate_rows, ref)
    if len(rate["rate_windows"]) != 1 or rate["rate_windows"][0]["jobs"] != 1:
        raise SystemExit(f"self-test rate {rate['rate_windows']}")
    if rate["rate_windows"][0]["force_calls"] != 5 or rate["rate_windows"][0]["sad_fcs"] != 4:
        raise SystemExit(f"self-test rate calls {rate['rate_windows']}")
    if abs(rate["rate_windows"][0]["rate"] - PREFACTOR) > 1e-6:
        raise SystemExit(f"self-test rate sum {rate['rate_windows'][0]}")
    stats = _stats([3, 1, 2])
    if stats != {"n": 3, "mean": 2, "median": 2, "min": 1, "max": 3}:
        raise SystemExit(f"self-test stats {stats}")
    protocol = Path("/tmp/htst-protocol-svn/akmc/states/0")
    if protocol.is_dir():
        ids = reference_ids((protocol / "processtable").read_text(encoding="utf-8"))
        if ids != [0, 1, 2, 3, 4, 5, 6, 7]:
            raise SystemExit(f"self-test reference ids {ids}")
        parsed = []
        for fields in _search_lines((protocol / "search_results.txt").read_text(encoding="utf-8")):
            parsed.append(
                {
                    "harness_delta": int(fields[4]),
                    "sad_fcs": int(fields[4]),
                    "mins_fcs": int(fields[5]),
                    "barrier": float(fields[2]),
                    "result": fields[-1],
                }
            )
        cut = cut_windows(parsed, ids)
        if len(cut["barrier_windows"]) != 69 or cut["barrier_windows"][0]["jobs"] != 17:
            raise SystemExit(f"self-test protocol barrier {len(cut['barrier_windows'])} {cut['barrier_windows'][:1]}")
        if cut["barrier_windows"][0]["sad_fcs"] != 8363:
            raise SystemExit(f"self-test protocol sad {cut['barrier_windows'][0]}")
        if len(cut["rate_windows"]) != 1 or cut["rate_windows"][0]["jobs"] != 3282:
            raise SystemExit(f"self-test protocol rate {cut['rate_windows']}")
        if cut["rate_windows"][0]["sad_fcs"] != 2187062:
            raise SystemExit(f"self-test protocol rate sad {cut['rate_windows'][0]}")
        if abs(cut["rate_windows"][0]["rate"] - 591458408.0662105) > 1e-3:
            raise SystemExit(f"self-test protocol rate sum {cut['rate_windows'][0]['rate']}")
        if abs(RATE_REF * RATE_ACCURACY - THRESHOLD) > 1e-6:
            raise SystemExit("self-test threshold")
    print("self-test ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report", "supervise", "selftest"))
    parser.add_argument("--code", choices=tuple(CODES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--interval", type=int, default=90 * 60)
    parser.add_argument("--initial", type=int, default=20 * 60)
    parser.add_argument("--write-doc", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        _self_test()
        return 0
    if args.command == "report":
        if args.write_doc:
            write_doc()
        _print_report(collect())
        return 0
    if args.command == "supervise":
        return supervise(args.interval, args.initial)
    if not args.code:
        raise SystemExit("run needs --code")
    run_code(args.code, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
