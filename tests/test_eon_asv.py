from benchmarks.bench_eon_develop import (
    CAUSE_INCOMPLETE,
    CAUSE_MISSING,
    CAUSE_NO_OVERLAP,
    EonDevelopDe2ts,
    EonDevelopSaddlePtDimer,
    EonDevelopTsLj38Lanczos,
    failure_cause,
    success_flag,
)


def test_failure_causes_are_stable_integers():
    assert failure_cause("0") == 0
    assert failure_cause("5") == 5
    assert failure_cause("no-overlap") == CAUSE_NO_OVERLAP
    assert failure_cause("incomplete") == CAUSE_INCOMPLETE
    assert failure_cause(None) == CAUSE_MISSING
    assert success_flag("0") == 1
    assert success_flag("5") == 0


def test_develop_benchmarks_parametrize_seed_and_start():
    assert EonDevelopTsLj38Lanczos.param_names == ["seed", "start"]
    assert EonDevelopTsLj38Lanczos.params[0] == [1]
    assert EonDevelopTsLj38Lanczos.params[1][0] == "0000"
    assert len(EonDevelopTsLj38Lanczos.params[1]) == 200
    assert EonDevelopTsLj38Lanczos.repeat == 1
    assert EonDevelopTsLj38Lanczos.track_force_calls.unit == "force calls"
    assert EonDevelopDe2ts.params[1][-1] == "pair_50"
    assert len(EonDevelopDe2ts.params[1]) == 51
    assert hasattr(EonDevelopDe2ts, "track_extrema")
    assert EonDevelopSaddlePtDimer.params[0] == list(range(1000, 1010))
    assert EonDevelopSaddlePtDimer.params[1] == ["pt-island"]
    assert hasattr(EonDevelopSaddlePtDimer, "track_total_searches")
    assert hasattr(EonDevelopSaddlePtDimer, "time_run")
