"""hTST barrier-window lines for a benchmark.dat file.

A blank identity field is omitted. The archive parser rejects a key
with no value, and an eOn template often has no tuning string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def problem_force_cap(problem: Mapping) -> int | None:
    """Force-call cap from the problem file.

    A template ``failure_cap`` is not a substitute for a number the
    problem omits.
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
    unfinished: bool,
    comments: Sequence[str],
) -> str:
    """Archive keys plus the cycle counts.

    ``average over`` is split so the barrier count and the rate count
    both survive. Empty code, commit, algorithm, and tuning values are
    left out.
    """
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
    lines.append(f"code {template.get('code', 'eOn')}")
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
    if unfinished:
        lines.append("partial true")
    return "\n".join(lines) + "\n"
