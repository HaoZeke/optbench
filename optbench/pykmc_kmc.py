"""KMC scores for the pyKMC rows, and the saddle endpoint file.

hTST follows ``archive/entries/hTST/*/report.py``: a barrier cycle ends
when every reference barrier has been seen, and a rate cycle ends when
the sum of ``prefactor * exp(-barrier / kT)`` reaches 0.999 of the
reference rate. The aluminum adatom score uses the same columns as the
eOn AKMC reducer in ``optbench/eon_amd.py``.

The saddle catalogue writes one endpoint record per search. A harness
success is a connected index-1 saddle at the force stop. The barrier
window and the novelty test count the catalogue. Failure classes come
from ``optbench.failures``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# archive/entries/hTST/*/report.py
KB_EV = 8.6173324e-5

HTST = {
    "hTST/Pt": {
        "temperature": 700.0,
        "prefactor": 1.2e12,
        "kb": KB_EV,
        "rate_accuracy": 0.999,
        "rate_ref": 155055901.244,
        "barriers": [
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
        ],
        "barrier_cycles": 13,
        "rate_cycles": 224,
        "source": "archive/entries/hTST/Pt/eon-dimer-lbfgs/report.py",
    },
    "hTST/Al": {
        "temperature": 300.0,
        "prefactor": 1.2e12,
        "kb": KB_EV,
        "rate_accuracy": 0.999,
        "rate_ref": 592041917.0,
        "barriers": [0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372],
        "barrier_cycles": 639,
        "rate_cycles": 20,
        "source": "archive/entries/hTST/Al/eon-dimer-cg/report.py",
    },
}

ENDPOINT_SCHEMA = "optbench.endpoint-minima/v1"
PROFILE = "beland-2015-pt111"


def searches_to_launch(
    searches: int,
    stall: int,
    connected: int,
    goal: int,
    max_searches: int,
    workers: int,
    *,
    stall_after: int = 160,
    stall_start: int = 200,
) -> int:
    """How many searches may start, assuming each one fails to connect.

    The catalogue stops once ``stall_start`` searches have run and
    ``stall_after`` of them have added no connected saddle. Counting
    the stall up front keeps a parallel batch inside that stop.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if connected >= goal or searches >= max_searches:
        return 0
    launched = 0
    cursor = searches
    missed = stall
    while launched < workers and cursor < max_searches and connected < goal:
        if cursor >= stall_start and missed >= stall_after:
            break
        cursor += 1
        missed += 1
        launched += 1
    return launched


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def reduce_htst(rows: Sequence[Mapping], spec: Mapping) -> dict:
    """Barrier and rate cycles from one KMC state, in search order.

    ``process_id`` is set only for a search the archive would call good
    or repeat. The first time an id appears, its barrier is rounded to
    three decimals and matched against the reference list. A later
    search with the same id is a repeat. This is the second pass of
    ``report.py``.
    """
    remaining = [float(value) for value in spec["barriers"]]
    full = len(remaining)
    ref_ids: list[int] = []
    known: dict[int, float] = {}
    barrier_ids: list[int] = []
    search_b = 0
    calls_b = 0
    barrier_searches: list[int] = []
    barrier_calls: list[int] = []
    rate_ids: list[int] = []
    search_r = 0
    calls_r = 0
    rate_sum = 0.0
    rate_searches: list[int] = []
    rate_calls: list[int] = []
    threshold = float(spec["rate_ref"]) * float(spec["rate_accuracy"])
    temperature = float(spec["temperature"])
    prefactor = float(spec["prefactor"])
    kb = float(spec["kb"])
    for row in rows:
        spent = int(row["force_calls"])
        search_b += 1
        search_r += 1
        calls_b += spent
        calls_r += spent
        pid = row.get("process_id")
        if pid is None:
            continue
        ident = int(pid)
        barrier = float(row["barrier"])
        if ident not in known:
            known[ident] = barrier
            key = round(barrier, 3)
            if key in remaining:
                remaining.remove(key)
                ref_ids.append(ident)
        if ident in ref_ids and ident not in barrier_ids:
            barrier_ids.append(ident)
        if full and len(ref_ids) == full and len(barrier_ids) == len(ref_ids):
            barrier_searches.append(search_b)
            barrier_calls.append(calls_b)
            barrier_ids = []
            search_b = 0
            calls_b = 0
        if rate_sum < threshold and ident not in rate_ids:
            rate_ids.append(ident)
            rate_sum += prefactor * math.exp(-barrier / (kb * temperature))
        if rate_sum >= threshold:
            rate_searches.append(search_r)
            rate_calls.append(calls_r)
            rate_ids = []
            search_r = 0
            calls_r = 0
            rate_sum = 0.0
    return {
        "searches": len(rows),
        "processes": len(known),
        "matched_barriers": len(ref_ids),
        "reference_barriers": full,
        "unmatched_barriers": remaining,
        "barrier_cycles": len(barrier_searches),
        "barrier_jobs": _mean(barrier_searches),
        "barrier_force_calls": _mean(barrier_calls),
        "barrier_call_list": barrier_calls,
        "rate_cycles": len(rate_searches),
        "rate_jobs": _mean(rate_searches),
        "rate_force_calls": _mean(rate_calls),
        "rate_call_list": rate_calls,
    }


def endpoint_record(
    *,
    start: str,
    search: int,
    repetition: int,
    minimum: list,
    box: list,
    free: list,
    pbc: list,
    endpoint_plus,
    endpoint_minus,
) -> dict:
    """One search in the endpoint file. Null minima mean the climb had no mode."""
    return {
        "schema": ENDPOINT_SCHEMA,
        "start": start,
        "search": int(search),
        "repetition": int(repetition),
        "minimum": minimum,
        "box": box,
        "free": free,
        "pbc": pbc,
        "endpoint_plus": endpoint_plus,
        "endpoint_minus": endpoint_minus,
    }


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def problem_force_cap(problem: Mapping) -> int | None:
    """Force-call cap from the problem file.

    ``score_v2`` leaves PAR-2 unset when this is ``None``. A template
    ``failure_cap`` is not a substitute for a number the problem omits.
    """
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    if stated in (None, ""):
        return None
    return int(stated)


def htst_dat(
    summary: Mapping,
    *,
    template: Mapping,
    problem: Mapping,
    spec: Mapping,
    searches: int,
    minimize_calls: int,
    partial: bool,
    comments: Sequence[str],
) -> str:
    """Archive keys plus the cycle counts. ``average over`` is split so both survive."""
    lines = [line for line in comments if line]
    if summary["barrier_jobs"] is not None:
        lines.append(f"barrier_jobs {_fmt(float(summary['barrier_jobs']))}")
        lines.append(f"barrier_force_calls {_fmt(float(summary['barrier_force_calls']))}")
        lines.append(f"barrier_average_over {int(summary['barrier_cycles'])}")
    if summary["rate_jobs"] is not None:
        lines.append(f"rate_jobs {_fmt(float(summary['rate_jobs']))}")
        lines.append(f"rate_force_calls {_fmt(float(summary['rate_force_calls']))}")
        lines.append(f"rate_average_over {int(summary['rate_cycles'])}")
    lines.append(f"temperature_K {_fmt(float(spec['temperature']))}")
    lines.append(f"prefactor {_fmt(float(spec['prefactor']))}")
    lines.append(f"rate_ref {_fmt(float(spec['rate_ref']))}")
    lines.append(f"matched_barriers {int(summary['matched_barriers'])}")
    lines.append(f"reference_barriers {int(summary['reference_barriers'])}")
    lines.append(f"processes {int(summary['processes'])}")
    lines.append(f"total_searches {int(searches)}")
    lines.append(f"minimize_calls {int(minimize_calls)}")
    lines.append(f"code {template.get('code', 'pyKMC')}")
    for key in ("code_version", "commit", "algorithm", "tuning"):
        value = str(template.get(key) or "").strip()
        if value:
            lines.append(f"{key} {value}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append("counter harness")
    lines.append(f"reducer {template.get('reducer', 'htst-cycles')}")
    cap = problem_force_cap(problem)
    if cap is not None:
        lines.append(f"cap {int(cap)}")
    if partial:
        lines.append("partial true")
    return "\n".join(lines) + "\n"


def amd_dat(
    summary: Mapping,
    *,
    template: Mapping,
    problem: Mapping,
    workers: int,
    partial: bool,
    comments: Sequence[str],
) -> str:
    """Same columns as the eOn AKMC catalogue. The counter is the harness."""
    lines = [line for line in comments if line]
    lines.append(f"code {template.get('code', 'pyKMC')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append("counter harness")
    lines.append("count_keys harness")
    lines.append("contributor Rohit Goswami")
    lines.append(f"repetitions {int(summary['repetitions'])}")
    lines.append(f"workers {int(workers)}")
    lines.append(f"temperature_K {problem['thermal']['temperature_K']}")
    lines.append("thermostat akmc")
    lines.append(f"reference_rate {_fmt(float(problem['reference']['true_rate']))}")
    lines.append(f"reference_products {int(problem['sample']['products'])}")
    for key in (
        "true_rate",
        "uncertainty",
        "percent_uncertainty",
        "rate_accuracy",
        "catalogue_accuracy",
        "force_calls_per_simulated_time",
        "force_calls",
    ):
        value = summary.get(key)
        if value is None:
            continue
        lines.append(f"{key} {_fmt(float(value))}")
    lines.append(f"nfailed {int(summary['nfailed'])}")
    if template.get("algorithm"):
        lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append("reducer amd-catalogue")
    lines.append(f"cap {int(template['failure_cap'])}")
    if partial:
        lines.append("partial true")
    return "\n".join(lines) + "\n"
