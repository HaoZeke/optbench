"""Summaries of a force-call sample.

ASV stores one result per start. These are the summaries of that vector.
The rounded mean in benchmark.dat is a separate record.
"""

from __future__ import annotations

import statistics


def sample_stats(counts: list[float]) -> dict[str, float | int]:
    """Mean, standard deviation, and quartiles of a force-call sample."""
    if not counts:
        raise ValueError("counts is empty")
    ordered = sorted(float(count) for count in counts)
    n = len(ordered)

    def quantile(fraction: float) -> float:
        if n == 1:
            return ordered[0]
        position = (n - 1) * fraction
        low = int(position)
        high = min(low + 1, n - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return {
        "n": n,
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if n > 1 else 0.0,
        "q25": quantile(0.25),
        "q75": quantile(0.75),
        "min": ordered[0],
        "max": ordered[-1],
    }
