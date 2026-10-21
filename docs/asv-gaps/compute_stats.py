"""Show the keys compute_stats returns for a timing sample.

Run: uv run python docs/asv-gaps/compute_stats.py
"""

from asv_runner.statistics import compute_stats

value, stats = compute_stats([10.0, 20.0, 30.0, 40.0], 1)
print(f"value: {value}")
print("keys: " + " ".join(sorted(stats)))
needed = {"par2_mean", "median_low"}
overlap = needed & set(stats)
print(f"score keys present: {sorted(overlap) or 'none'}")
if overlap:
    raise SystemExit("compute_stats grew score keys; this reproducer is stale")
