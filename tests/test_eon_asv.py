import os

import pytest

from benchmarks.bench_eon_develop import (
    EonDevelopDe2ts,
    EonDevelopDeLj38Cap,
    EonDevelopSaddlePtDimer,
    EonDevelopSaddlePtLanczos,
    EonDevelopTsLj38Lanczos,
    _akmc_server_ini,
    _ini,
    classify_akmc,
    classify_client,
    kept_lowest,
    lj_cut,
    morse_cluster,
    selected,
)
from optbench.asv_tracks import TrackedSearch as _TrackedSearch
from optbench.counter import HarnessCounter


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
    assert bench.measure is not _TrackedSearch.measure
    assert bench.time_search is not _TrackedSearch.time_search
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
    spent = classify_client(calls=2500, reason=None, cap=2500, crashed=True)
    assert spent["failure"] == "cap"
    assert spent["success"] is False
    assert spent["force_calls"] == 2500
    assert classify_client(calls=2499, reason=None, cap=2500)["failure"] == "crash"
    assert EonDevelopDeLj38Cap.cap == 2500


def test_ext_pot_ini_uses_one_potential(tmp_path):
    text = _ini({"ini": {"Main": {"job": "dimer"}, "Potential": {"potential": "ljcluster"}}}, 1, tmp_path / "bridge")
    assert "potential = ext_pot" in text
    assert "parallel = false" in text
    assert "ext_pot_path" in text


def test_akmc_server_ini_omits_parallel(tmp_path):
    # The Python AKMC schema rejects Main.parallel and exits before a search.
    bridge = tmp_path / "bridge"
    ini = _akmc_server_ini(
        {"ini": {"Main": {"job": "akmc", "random_seed": 1}, "Potential": {"potential": "morse_pt"}}},
        bridge,
    )
    assert "parallel" not in ini["Main"]
    assert ini["Main"]["job"] == "akmc"
    assert ini["Potential"] == {"potential": "ext_pot", "ext_pot_path": str(bridge)}


def test_lowest_mode_keeps_one_attempt_not_the_sum():
    # The first attempt misses the overlap. The counter is reset before
    # the second, so the kept cost is 2, not 28 + 2.
    index = kept_lowest([(28, 0.2), (2, 0.995)], 0.99)
    assert index == 1
    attempts = [(28, 0.2), (2, 0.995)]
    assert attempts[index][0] == 2
    assert attempts[index][0] != sum(calls for calls, _overlap in attempts)
    assert kept_lowest([(10, 0.1), (4, 0.4)], 0.99) == 1
    assert kept_lowest([], 0.99) is None


def test_reactant_prefetch_is_one_force_call_per_search():
    """The AKMC reactant energy is a real evaluation outside sad-fcs.

    Thirty-two searches add thirty-two calls. A later pass that does not
    evaluate the potential does not add them again.
    """
    counter = HarnessCounter(cache=False)
    searches = 32
    for index in range(searches):
        counter.charge((float(index),))
        for step in range(3):
            counter.charge((float(index), float(step)))
    window = counter.calls - searches
    assert counter.calls == searches * 4
    assert window == searches * 3
    assert counter.calls - window == searches


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
