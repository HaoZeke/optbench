from optbench.eon_jobs import report_summary


def test_report_keeps_a_value_under_the_cap_and_counts_a_bad_reason():
    rows = [
        {"raw": 40, "reason": "0"},
        {"raw": 80, "reason": "1"},
        {"raw": 2500, "reason": "0"},
        {"raw": None, "reason": None},
    ]
    summary = report_summary(rows, cap=2500)
    assert summary["force_calls"] == 60
    assert summary["force_calls_min"] == 40
    assert summary["force_calls_max"] == 80
    assert summary["nfailed"] == 2
    assert summary["force_calls_median"] == 60


def test_heptamer_mean_divides_the_band_counter_by_the_image_count():
    rows = [{"raw": 320, "reason": "0"}, {"raw": 400, "reason": "0"}]
    summary = report_summary(rows, cap=10000, images=8)
    assert summary["force_calls"] == 45
    assert summary["force_calls_min"] == 40
    assert summary["force_calls_max"] == 50
    assert summary["nfailed"] == 0
