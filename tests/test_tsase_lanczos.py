"""TSASE r410 Lanczos matches the archived LJ38 lowest-eigenvector score."""

from optbench.tsase_lanczos import _coords_dir, force_calls, paper_mean, read_mode, read_positions, run_all, shared_mode


def test_two_starts_match_the_r410_force_calls():
    coords = _coords_dir()
    mode = shared_mode()
    fastest = force_calls(read_positions(coords / "0126.con"), read_mode(coords / "0126_mode"), mode)
    first = force_calls(read_positions(coords / "0000.con"), read_mode(coords / "0000_mode"), mode)
    assert fastest == 13
    assert first == 27


def test_the_paper_mean_is_the_archived_integer():
    calls = run_all(_coords_dir())
    assert len(calls) == 200
    assert min(calls) == 13
    assert max(calls) == 52
    assert paper_mean(calls) == 25
    assert all(value < 10000 for value in calls)
