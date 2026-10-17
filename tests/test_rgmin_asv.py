"""rgmin ASV classes follow the shared parameter and track names."""

import pytest

import re

from benchmarks.bench_rgmin import (
    SEED,
    asv_benchmark_pattern,
    selected_indices,
    RgminLj38LbfgsTuned,
    RgminMorseLbfgs,
    _specs,
)
from optbench.failures import FAILURES


def test_rgmin_asv_params_are_seed_and_start_index(monkeypatch):
    assert RgminLj38LbfgsTuned.param_names == ["seed", "start"]
    assert RgminLj38LbfgsTuned.params[0] == [SEED]
    assert RgminLj38LbfgsTuned.params[1] == ["0"]
    assert RgminMorseLbfgs.params[1] == ["0"]
    assert RgminLj38LbfgsTuned.repeat == 1
    assert RgminLj38LbfgsTuned.rounds == 1
    assert RgminLj38LbfgsTuned.number == 1
    assert RgminLj38LbfgsTuned.warmup_time == 0
    assert RgminLj38LbfgsTuned.sample_time == 0
    assert RgminLj38LbfgsTuned.cap == 10000
    assert RgminLj38LbfgsTuned.track_force_calls.unit == "force calls"
    assert RgminLj38LbfgsTuned.track_success.unit == "fraction"
    for cause in FAILURES:
        method = getattr(RgminLj38LbfgsTuned, f"track_failure_{cause}")
        assert method.unit == "count"
    assert selected_indices(1000) == ["0"]
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert selected_indices(1000) == [str(index) for index in range(1000)]
    assert selected_indices(100) == [str(index) for index in range(100)]


def test_asv_class_filter_is_anchored_and_literal():
    names = [spec["class_name"] for spec in _specs()]
    assert len(names) == 64
    assert len(set(names)) == 64
    for name in names:
        pattern = re.compile(asv_benchmark_pattern(name))
        assert pattern.search(f"bench_rgmin.{name}.time_search('0', '0')")
        for other in names:
            if other != name:
                assert pattern.search(f"bench_rgmin.{other}.time_search('0', '0')") is None


def test_one_tuned_lbfgs_start_is_a_harness_count():
    pytest.importorskip("rgmin")
    bench = RgminLj38LbfgsTuned()
    bench.setup(SEED, "0")
    bench.time_search(SEED, "0")
    calls = bench.track_force_calls(SEED, "0")
    assert calls == int(calls)
    assert 0 < calls < 10000
    assert bench.record["counter"] == "harness"
    assert bench.record["failure"] in (None, *FAILURES)
    assert bench.track_success(SEED, "0") == (1.0 if bench.record["success"] else 0.0)
    assert bench.track_failure_cap(SEED, "0") == (1.0 if bench.record["failure"] == "cap" else 0.0)
