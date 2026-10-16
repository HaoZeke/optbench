import os

import pytest

from benchmarks.bench_eon_develop import (
    EonDevelopDe2ts,
    EonDevelopSaddlePtDimer,
    EonDevelopSaddlePtLanczos,
    EonDevelopTsLj38Lanczos,
    classify_akmc,
    classify_client,
    lj_cut,
    morse_cluster,
    selected,
)
from optbench.asv_tracks import TrackedSearch


def test_default_grid_is_one_real_start():
    bench = EonDevelopTsLj38Lanczos
    assert bench.param_names == ["seed", "start"]
    assert bench.params == [["1"], ["0000"]]
    assert len(bench.campaign_starts) == 200
    assert bench.repeat == 1
    assert bench.rounds == 1
    assert bench.number == 1
    assert bench.sample_time == 0
    assert bench.warmup_time == 0
    assert bench.cap == 10000
    assert bench.published_mean is False
    assert bench.potential == "ljcluster"
    assert bench.track_force_calls.unit == "force calls"
    assert bench.track_success.unit == "fraction"
    assert bench.track_failure_cap.unit == "count"
    assert bench.track_failure_wrong_stationary.unit == "count"
    assert bench.track_failure_divergence.unit == "count"
    assert bench.track_failure_crash.unit == "count"
    assert bench.measure is not TrackedSearch.measure
    assert bench.time_search is not TrackedSearch.time_search
    assert not hasattr(bench, "time_run")
    assert not hasattr(bench, "track_failure_cause")
    assert EonDevelopDe2ts.campaign_starts[-1] == "pair_50"
    assert len(EonDevelopDe2ts.campaign_starts) == 51
    assert EonDevelopDe2ts.params[1] == ["pair_00"]
    assert EonDevelopDe2ts.track_extrema.unit == "count"
    assert EonDevelopSaddlePtDimer.campaign_seeds == [str(seed) for seed in range(1000, 1010)]
    assert EonDevelopSaddlePtDimer.params == [["1000"], ["pt-island"]]
    assert EonDevelopSaddlePtDimer.passes == 1
    assert EonDevelopSaddlePtDimer.potential == "morse_pt"
    assert EonDevelopSaddlePtLanczos.campaign_seeds == ["1000", "1001"]
    assert EonDevelopSaddlePtLanczos.params[0] == ["1000"]
    assert EonDevelopSaddlePtDimer.track_total_searches.unit == "searches"
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    assert selected(["0000", "0001"]) == ["0000"]


def test_full_grid_follows_the_environment(monkeypatch):
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert selected(["0000", "0001"]) == ["0000", "0001"]


def test_failure_classes_follow_the_harness_order():
    assert classify_client(calls=10000, reason="2", cap=10000)["failure"] == "cap"
    under = classify_client(calls=40, reason="5", cap=10000)
    assert under["failure"] == "divergence"
    assert under["success"] is False
    mode = classify_client(calls=43, reason="5", cap=10000, overlap_ok=True)
    assert mode["success"] is True
    assert mode["failure"] is None
    missed = classify_client(calls=20, reason="5", cap=10000, overlap_ok=False)
    assert missed["failure"] == "wrong_stationary"
    assert classify_client(calls=None, reason=None, cap=100)["failure"] == "crash"
    assert classify_client(calls=12, reason="0", cap=100)["success"] is True
    stalled = classify_akmc(calls=50, cap=100, reached=False, crashed=False)
    assert stalled["failure"] == "cap"
    assert stalled["success"] is False


def test_morse_and_lj_cut_kernels():
    energy, forces = morse_cluster([0.0, 0.0, 0.0, 2.897, 0.0, 0.0])
    assert energy == pytest.approx(0.0, abs=1e-12)
    assert forces == pytest.approx([0.0] * 6, abs=1e-8)
    distance = 2.0 ** (1.0 / 6.0)
    energy, forces = lj_cut([0.0, 0.0, 0.0, distance, 0.0, 0.0])
    assert energy == pytest.approx(-1.0, abs=1e-8)
    assert forces == pytest.approx([0.0] * 6, abs=1e-6)
    far, _forces = lj_cut([0.0, 0.0, 0.0, 40.0, 0.0, 0.0])
    assert far == 0.0
