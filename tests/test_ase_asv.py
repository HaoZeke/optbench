"""ASE and Sella benchmarks use the shared asv track methods."""

from benchmarks.bench_ase_sella import AseLj38LbfgsDefault, SellaPtIslandSearchDefault


def test_lj38_lbfgs_uses_the_shared_track_grid():
    assert AseLj38LbfgsDefault.param_names == ["seed", "start"]
    assert AseLj38LbfgsDefault.params[0] == ["0"]
    assert len(AseLj38LbfgsDefault.params[1]) == 1
    assert AseLj38LbfgsDefault.params[1][0].startswith("cluster_")
    assert AseLj38LbfgsDefault.repeat == 1
    assert AseLj38LbfgsDefault.rounds == 1
    assert AseLj38LbfgsDefault.number == 1
    assert AseLj38LbfgsDefault.sample_time == 0
    assert AseLj38LbfgsDefault.warmup_time == 0
    assert AseLj38LbfgsDefault.cap > 0
    assert AseLj38LbfgsDefault.track_force_calls.unit == "force calls"
    assert AseLj38LbfgsDefault.track_success.unit == "fraction"
    assert AseLj38LbfgsDefault.track_failure_cap.unit == "count"
    assert AseLj38LbfgsDefault.track_failure_wrong_stationary.unit == "count"
    assert callable(AseLj38LbfgsDefault.time_search)
    assert AseLj38LbfgsDefault.template_id == "ase/lj38-lbfgs-default"


def test_sella_search_varies_the_seed_on_one_geometry():
    assert SellaPtIslandSearchDefault.param_names == ["seed", "start"]
    assert SellaPtIslandSearchDefault.params[0] == ["0"]
    assert len(SellaPtIslandSearchDefault.params[1]) == 1
    assert SellaPtIslandSearchDefault.template_id == "sella/pt-island-search-default"
