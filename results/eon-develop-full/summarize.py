"""Summarise a full-grid eOn develop asv result.

Reads the track values asv stored and calls ``score_v2``. Force-call
mean, median, min, and max are over successes only. A missing parameter
is not a failure and is not given a number.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["OPTBENCH_ASV_FULL"] = "1"
os.environ.pop("OPTBENCH_ASV_SLICE", None)

from benchmarks.bench_eon_develop import EON_COMMIT, _specs  # noqa: E402
from optbench.score import score_v2  # noqa: E402


def _unwrap(value: str):
    return ast.literal_eval(value)


def _values(payload: dict, cls: str, method: str) -> list | None:
    name = f"bench_eon_develop.{cls}.{method}"
    if name not in payload["results"]:
        return None
    return payload["results"][name][0]


def _failure(success, cap, wrong, divergence, crash) -> str | None:
    if success == 1.0:
        return None
    if crash == 1.0:
        return "crash"
    if divergence == 1.0:
        return "divergence"
    if cap == 1.0:
        return "cap"
    if wrong == 1.0:
        return "wrong_stationary"
    return None


def _stat(values: list[float], fn):
    if not values:
        return None
    return fn(values)


def summarise(payload: dict) -> dict:
    classes = []
    for spec in _specs():
        order = [
            (str(seed), str(start))
            for seed, start in itertools.product(spec["campaign_seeds"], spec["campaign_starts"])
        ]
        calls = _values(payload, spec["cls"], "track_force_calls")
        success = _values(payload, spec["cls"], "track_success")
        if calls is None or success is None:
            classes.append(
                {
                    "class": spec["cls"],
                    "template": spec["template"],
                    "kind": spec["kind"],
                    "cap": spec["cap"],
                    "n": len(order),
                    "finished": 0,
                    "missing": [list(pair) for pair in order],
                    "ran": False,
                }
            )
            continue
        if len(calls) != len(order) or len(success) != len(order):
            raise ValueError(f"{spec['cls']} has {len(calls)} force calls for {len(order)} parameters")
        tracks = {
            name: _values(payload, spec["cls"], f"track_{name}")
            for name in (
                "failure_cap",
                "failure_wrong_stationary",
                "failure_divergence",
                "failure_crash",
            )
        }
        records = []
        missing = []
        for index, pair in enumerate(order):
            if calls[index] is None or success[index] is None or (
                isinstance(calls[index], float) and math.isnan(calls[index])
            ):
                missing.append(list(pair))
                continue
            flags = {}
            for name, values in tracks.items():
                if values is None:
                    raise ValueError(f"{spec['cls']} is missing {name}")
                flags[name] = None if values[index] is None else float(values[index])
            ok = float(success[index]) == 1.0
            failure = _failure(
                1.0 if ok else 0.0,
                flags["failure_cap"],
                flags["failure_wrong_stationary"],
                flags["failure_divergence"],
                flags["failure_crash"],
            )
            if not ok and failure is None:
                raise ValueError(f"{spec['cls']} {pair} failed without a class")
            records.append(
                {
                    "seed": pair[0],
                    "start": pair[1],
                    "force_calls": float(calls[index]),
                    "success": ok,
                    "failure": failure,
                }
            )
        finished_calls = [row["force_calls"] for row in records if row["success"]]
        counts = {"cap": 0, "divergence": 0, "crash": 0, "wrong_stationary": 0}
        for row in records:
            if row["failure"] is not None:
                counts[row["failure"]] += 1
        entry = {
            "class": spec["cls"],
            "template": spec["template"],
            "kind": spec["kind"],
            "cap": spec["cap"],
            "n": len(order),
            "finished": len(records),
            "missing": missing,
            "successes": sum(row["success"] for row in records),
            "failures": counts,
            "force_calls_success": {
                "n": len(finished_calls),
                "mean": _stat(finished_calls, statistics.fmean),
                "median": _stat(finished_calls, statistics.median),
                "min": _stat(finished_calls, min),
                "max": _stat(finished_calls, max),
            },
            "ran": len(records) == len(order),
        }
        if records and not missing:
            entry["score_v2"] = score_v2(records, spec["cap"])
        classes.append(entry)
    return {
        "eon_commit": EON_COMMIT,
        "asv_commit": payload.get("commit_hash"),
        "counter": "harness",
        "classes": classes,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize.py RESULT.json SUMMARY.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    summary = summarise(payload)
    Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{'class':<32} {'n':>4} {'ok':>4} {'cap':>4} {'div':>4} {'crash':>5} {'wrong':>5} "
        f"{'mean':>12} {'median':>12} {'min':>10} {'max':>10} {'par2':>12}"
    )
    for row in summary["classes"]:
        if "force_calls_success" not in row:
            print(f"{row['class']:<32} {row['n']:>4} missing {row['n'] - row['finished']}")
            continue
        calls = row["force_calls_success"]
        par2 = None if "score_v2" not in row else row["score_v2"]["par2_mean"]
        print(
            f"{row['class']:<32} {row['finished']:>4} {row['successes']:>4} "
            f"{row['failures']['cap']:>4} {row['failures']['divergence']:>4} "
            f"{row['failures']['crash']:>5} {row['failures']['wrong_stationary']:>5} "
            f"{calls['mean']!s:>12} {calls['median']!s:>12} {calls['min']!s:>10} {calls['max']!s:>10} "
            f"{par2!s:>12}"
        )


if __name__ == "__main__":
    main()
