#!/usr/bin/env python3
"""Score a serial GMIN no-symmetrisation campaign.

The archive row is the gsaveit target line, which is what run_gmin.sh reads,
plus the mc timer on the same hit. Failures are starts with no target hit.
They stay out of the means.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

HIT = re.compile(r"Target hit after\s+(\d+)\s+quenches\s+(\d+)\s+function calls")
TIME = re.compile(
    r"target hit, quenches, function calls, time:\s+\d+\s+\d+\s+\d+\s+([0-9.]+)"
)
ENERGY = re.compile(r"gsaveit> Energy=\s+([-\d.]+)\s+target=")
MISSED = "target missed"


def parse_output(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    hit = HIT.search(text)
    if hit is None:
        return {"success": False, "missed": MISSED in text}
    clock = TIME.search(text)
    energy = ENERGY.search(text)
    if clock is None:
        raise SystemExit(f"{path} hit the target and has no mc time")
    return {
        "success": True,
        "quenches": int(hit.group(1)),
        "calls": int(hit.group(2)),
        "cpu_time": float(clock.group(1)),
        "energy": float(energy.group(1)) if energy else None,
    }


def collect(root: Path, count: int) -> list[dict]:
    rows = []
    for index in range(1, count + 1):
        path = root / f"run-{index}" / "output"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        parsed = parse_output(path)
        parsed["start"] = str(index)
        rows.append(parsed)
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_campaign(
    rows: list[dict],
    dat_path: Path,
    starts_path: Path,
    *,
    code_version: str,
    commit: str,
) -> dict:
    ok = [row for row in rows if row["success"]]
    failed = [row for row in rows if not row["success"]]
    if any(not row["success"] and not row.get("missed") for row in rows):
        raise SystemExit("a run neither hit nor missed the target")
    summary = {
        "basin_hopping_steps_mean": _mean([row["quenches"] for row in ok]),
        "basin_hopping_steps_stddev": _std([row["quenches"] for row in ok]),
        "energy_calls_mean": _mean([row["calls"] for row in ok]),
        "energy_calls_stddev": _std([row["calls"] for row in ok]),
        "cpu_time_mean": _mean([row["cpu_time"] for row in ok]),
        "cpu_time_stddev": _std([row["cpu_time"] for row in ok]),
        "nfailed": len(failed),
        "n": len(rows),
    }
    lines = [
        "# GMIN gsaveit quenches and function calls, and the mc timer.",
        "# Means leave out starts with no Target hit after line.",
        "problem globalopt/lj150",
        "template gmin/lj150-nosym",
        "code GMIN",
        f"code_version {code_version}",
        f"commit {commit}",
        "algorithm no symmetrisation",
        "tuning archive",
        "counter harness",
        "reducer harness",
        "cap 10000000",
        f"n {summary['n']}",
        f"nfailed {summary['nfailed']}",
        f"basin_hopping_steps_mean {summary['basin_hopping_steps_mean']:.10g}",
        f"basin_hopping_steps_stddev {summary['basin_hopping_steps_stddev']:.10g}",
        f"energy_calls_mean {summary['energy_calls_mean']:.10g}",
        f"energy_calls_stddev {summary['energy_calls_stddev']:.10g}",
        f"cpu_time_mean {summary['cpu_time_mean']:.10g}",
        f"cpu_time_stddev {summary['cpu_time_stddev']:.10g}",
    ]
    dat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    start_lines = []
    for row in rows:
        record = {
            "schema": "optbench.start/v2",
            "start": row["start"],
            "force_calls": row["calls"] if row["success"] else 0,
            "counter": "harness",
            "comparable": True,
            "success": row["success"],
            "quenches": row.get("quenches"),
            "cpu_time": row.get("cpu_time"),
            "energy": row.get("energy"),
        }
        if not row["success"]:
            record["failure"] = "cap"
            record["force_calls"] = 0
        start_lines.append(json.dumps(record, sort_keys=True))
    starts_path.write_text("\n".join(start_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dat", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--code-version", default="df8da5992675")
    parser.add_argument(
        "--commit",
        default="738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31",
    )
    args = parser.parse_args()
    rows = collect(args.root, args.count)
    summary = write_campaign(
        rows,
        args.dat,
        args.starts,
        code_version=args.code_version,
        commit=args.commit,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
