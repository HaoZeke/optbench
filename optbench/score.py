"""Reducers for force-call tables.

The archive mean is awk ``printf "%.0f"``. On halves that rounds to even,
which is what Python ``round`` does for these ties.
"""

from __future__ import annotations


def half_even(value: float) -> int:
    """Round to an integer, ties to even."""
    return round(value)


def summarize_capped(counts: list[int], cap: int) -> dict[str, int | None]:
    """Mean, min, and max of counts below ``cap``, and how many meet the cap.

    A count at or above ``cap`` is a failure and is left out of the mean.
    The mean is rounded with :func:`half_even`.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    ok = [count for count in counts if count < cap]
    failed = len(counts) - len(ok)
    if not ok:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "nfailed": failed,
        }
    return {
        "force_calls": half_even(sum(ok) / len(ok)),
        "force_calls_min": min(ok),
        "force_calls_max": max(ok),
        "nfailed": failed,
    }
