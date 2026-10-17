import subprocess

from optbench.score import half_even, summarize_all, summarize_capped, summarize_divided


def test_cap_splits_failures_from_the_mean():
    summary = summarize_capped([207, 1153, 10000], cap=10000)
    assert summary["force_calls"] == 680
    assert summary["force_calls_min"] == 207
    assert summary["force_calls_max"] == 1153
    assert summary["nfailed"] == 1


def test_summarize_all_keeps_a_failed_start_in_the_mean():
    summary = summarize_all([34, 68, 1846], nfailed=1)
    assert summary["nfailed"] == 1
    assert summary["force_calls_max"] == 1846
    assert summary["force_calls_min"] == 34
    assert summary["force_calls"] == round((34 + 68 + 1846) / 3)


def test_divided_heptamer_mean_is_per_image():
    summary = summarize_divided([80, 160, 80000], [False, False, False], images=8, cap=10000)
    assert summary["nfailed"] == 1
    assert summary["force_calls"] == 15
    assert summary["force_calls_raw"] == 120


def test_half_even_matches_awk_printf():
    for value in (0.5, 1.5, 2.5, -1.5):
        awk = subprocess.check_output(
            ["awk", "-v", f"x={value}", 'BEGIN{printf "%.0f", x}'],
            text=True,
        )
        assert str(half_even(value)) == awk.strip()
