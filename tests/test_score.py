import subprocess

from optbench.score import half_even, summarize_capped


def test_cap_splits_failures_from_the_mean():
    summary = summarize_capped([207, 1153, 10000], cap=10000)
    assert summary["force_calls"] == 680
    assert summary["force_calls_min"] == 207
    assert summary["force_calls_max"] == 1153
    assert summary["nfailed"] == 1


def test_half_even_matches_awk_printf():
    for value in (0.5, 1.5, 2.5, -1.5):
        awk = subprocess.check_output(
            ["awk", "-v", f"x={value}", 'BEGIN{printf "%.0f", x}'],
            text=True,
        )
        assert str(half_even(value)) == awk.strip()
