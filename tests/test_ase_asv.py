"""ASE and Sella ASV classes expose a start and a seed."""

from benchmarks.bench_ase_sella import AseLj38LbfgsDefault, SellaPtIslandSearchDefault


def test_ase_lbfgs_tracks_each_lj38_start_and_a_seed():
    assert AseLj38LbfgsDefault.param_names == ["start", "seed"]
    assert AseLj38LbfgsDefault.params[1] == ["0"]
    assert len(AseLj38LbfgsDefault.params[0]) == 1000
    assert AseLj38LbfgsDefault.repeat == 1
    assert AseLj38LbfgsDefault.number == 1
    assert AseLj38LbfgsDefault.track_force_calls.unit == "force calls"
    assert callable(AseLj38LbfgsDefault.time_optimize)
    assert callable(AseLj38LbfgsDefault.track_success)
    assert callable(AseLj38LbfgsDefault.track_failure_cap)
    assert callable(AseLj38LbfgsDefault.track_failure_force)
    assert callable(AseLj38LbfgsDefault.track_failure_index)
    assert callable(AseLj38LbfgsDefault.track_failure_error)


def test_sella_search_varies_the_seed_on_one_geometry():
    assert SellaPtIslandSearchDefault.param_names == ["start", "seed"]
    assert len(SellaPtIslandSearchDefault.params[0]) == 1
    assert SellaPtIslandSearchDefault.params[1] == [str(seed) for seed in range(16)]
    assert SellaPtIslandSearchDefault.template_id == "sella/pt-island-search-default"
