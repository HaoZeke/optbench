"""Reducers for force-call tables.

The archive mean is awk ``printf "%.0f"``. On halves that rounds to even,
which is what Python ``round`` does for these ties.
"""

from __future__ import annotations


def half_even(value: float) -> int:
    """Round to an integer, ties to even."""
    return round(value)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2


def summarize_all(counts: list[int], nfailed: int) -> dict[str, int | float | None]:
    """Mean, min, max, and median of every count, failures included.

    Transition-state rows use this reducer. A start that hit the step cap
    stays in the mean, and ``nfailed`` counts those starts.
    """
    if nfailed < 0:
        raise ValueError("nfailed must be zero or positive")
    if not counts:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "force_calls_median": None,
            "nfailed": nfailed,
        }
    return {
        "force_calls": half_even(sum(counts) / len(counts)),
        "force_calls_min": min(counts),
        "force_calls_max": max(counts),
        "force_calls_median": _median([float(count) for count in counts]),
        "nfailed": nfailed,
    }


def summarize_divided(
    counts: list[int],
    failed: list[bool],
    *,
    images: int,
    cap: int,
) -> dict[str, int | float | None]:
    """Per-image mean of counts that converged under ``cap``.

    ``cap`` is compared to ``count / images``, which is how the Pt-heptamer
    report turns a band total into one image. The raw mean of the kept
    totals is ``force_calls_raw``.
    """
    if images <= 0:
        raise ValueError("images must be positive")
    if cap <= 0:
        raise ValueError("cap must be positive")
    if len(counts) != len(failed):
        raise ValueError("counts and failed must have the same length")
    ok_raw: list[int] = []
    ok_per: list[float] = []
    nfailed = 0
    for count, flag in zip(counts, failed):
        per_image = count / images
        if flag or per_image >= cap:
            nfailed += 1
            continue
        ok_raw.append(count)
        ok_per.append(per_image)
    if not ok_per:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "force_calls_median": None,
            "force_calls_raw": None,
            "nfailed": nfailed,
        }
    return {
        "force_calls": half_even(sum(ok_per) / len(ok_per)),
        "force_calls_min": half_even(min(ok_per)),
        "force_calls_max": half_even(max(ok_per)),
        "force_calls_median": _median(ok_per),
        "force_calls_raw": half_even(sum(ok_raw) / len(ok_raw)),
        "nfailed": nfailed,
    }


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
