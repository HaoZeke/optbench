"""Shared Pt-island start. One seed per search, one listed epicentre.

The displacement matches OPTIM ``displace_listed``: every free atom
inside 3.3 Å moves by an independent normal component of standard
deviation 0.1. A second search uses a new generator. It does not
continue the first search's stream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import load_template, resolve_problem
from optbench.engines.geometries import load_starts
from optbench.eon_displace import (
    as_bool,
    displace_listed,
    draw_search_start,
    problem_pbc,
    search_seed,
)
from optbench.surfaces import minimum_image

ROOT = Path(__file__).resolve().parents[1]
LISTED = [0, 1, 2, 3, 4, 5, 6]


def _island(work: Path):
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    start = load_starts(ROOT, problem, work)[0]
    positions = np.asarray(start["positions"], dtype=np.float64)
    fixed = np.asarray(start["fixed"], dtype=bool)
    cell = np.asarray(start["box"], dtype=np.float64)
    return problem, positions, fixed, cell


def _support(delta: np.ndarray) -> np.ndarray:
    return np.linalg.norm(delta, axis=1) > 0.0


class _ZeroRng:
    def integers(self, low, high):
        del high
        return int(low)

    def normal(self, loc, scale, size):
        del loc, scale
        return np.zeros(size, dtype=np.float64)


def test_search_seed_is_the_optim_formula():
    assert search_seed(1, 1) == (20261015 + 1) * 1_000_003 + 1
    assert search_seed(2, 5) == (20261015 + 2) * 1_000_003 + 5
    assert search_seed(1, 1) != search_seed(1, 2)


@pytest.mark.parametrize(
    "name",
    [
        "rgsaddle/saddle-dimer-lbfgs",
        "rgsaddle/saddle-dimer-fire",
        "rgsaddle/saddle-dimer-rotskip",
        "rgsaddle/saddle-lanczos",
    ],
)
def test_archived_rgsaddle_templates_name_the_shared_radius(name):
    search = load_template(ROOT, name)["ini"]["Search"]
    assert search["displace_atoms"] == "0,1,2,3,4,5,6"
    assert search["displace_magnitude"] == 0.1
    assert search["displace_radius"] == 3.3
    assert "random_mode" not in search
    assert as_bool("false", True) is False
    assert as_bool("true", False) is True


def test_the_same_seed_is_the_same_start_in_two_callers(tmp_path):
    _problem, positions, fixed, cell = _island(tmp_path / "work")
    pbc = (True, True, False)
    kwargs = dict(
        positions=positions,
        fixed=fixed,
        cell=cell,
        pbc=pbc,
        listed=LISTED,
        rep=1,
        attempt=4,
        radius=3.3,
        magnitude=0.1,
    )
    left = draw_search_start(**kwargs)
    right = draw_search_start(**kwargs)
    assert left["seed"] == search_seed(1, 4)
    assert left["epicenter"] == right["epicenter"]
    assert left["epicenter"] in LISTED
    assert np.array_equal(left["start"], right["start"])
    assert np.array_equal(left["kick"], right["kick"])
    assert np.array_equal(left["vector"], right["vector"])


def test_a_continued_stream_is_not_the_next_search():
    positions = np.zeros((8, 3))
    positions[:, 0] = np.arange(8, dtype=np.float64)
    fixed = np.zeros(8, dtype=bool)
    pbc = (False, False, False)
    fresh = draw_search_start(
        positions, fixed, None, pbc, LISTED, rep=1, attempt=2, radius=3.3, magnitude=0.1
    )
    rng = np.random.default_rng(search_seed(1, 1))
    displace_listed(positions, fixed, None, pbc, LISTED, 3.3, 0.1, rng)
    continued = displace_listed(positions, fixed, None, pbc, LISTED, 3.3, 0.1, rng)
    assert not np.array_equal(continued["start"], fresh["start"])


def test_one_gaussian_is_both_the_shift_and_the_kick(tmp_path):
    _problem, positions, fixed, cell = _island(tmp_path / "work")
    drawn = draw_search_start(
        positions,
        fixed,
        cell,
        problem_pbc(resolve_problem(ROOT, "saddle-search/pt-island")),
        LISTED,
        rep=1,
        attempt=1,
        radius=3.3,
        magnitude=0.1,
    )
    delta = drawn["start"] - positions
    # Adding the Gaussian onto coordinates of several angstroms and subtracting
    # them back is the same vector up to roundoff. A second mode draw would
    # point somewhere else.
    assert np.allclose(drawn["kick"], delta[~fixed].reshape(-1), rtol=0.0, atol=1e-12)
    assert int(drawn["moved"]) > 1
    expected = np.zeros(len(positions), dtype=bool)
    for index, frozen in enumerate(fixed):
        if frozen:
            continue
        if index == drawn["epicenter"]:
            distance = 0.0
        else:
            distance = float(
                np.linalg.norm(minimum_image(positions[index] - positions[drawn["epicenter"]], cell, (True, True, False)))
            )
        expected[index] = distance <= 3.3
    assert _support(delta).tolist() == expected.tolist()
    assert not _support(delta)[fixed].any()


def test_radius_keeps_the_boundary_and_the_periodic_neighbor():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.3, 0.0, 0.0],
            [3.300001, 0.0, 0.0],
            [18.2088, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    fixed = np.array([False, False, False, False, True, False])
    cell = np.array([19.2088, 19.0118, 30.0])
    drawn = displace_listed(
        positions,
        fixed,
        cell,
        (True, False, False),
        [0],
        3.3,
        0.1,
        np.random.default_rng(1),
    )
    assert drawn["epicenter"] == 0
    assert _support(drawn["start"] - positions).tolist() == [True, True, False, True, False, False]


def test_a_zero_draw_shifts_the_epicentre_along_x():
    positions = np.zeros((3, 3))
    fixed = np.array([False, False, True])
    drawn = displace_listed(
        positions,
        fixed,
        None,
        (False, False, False),
        [0, 1],
        3.3,
        0.1,
        _ZeroRng(),
    )
    assert drawn["epicenter"] == 0
    assert drawn["start"][0, 0] == pytest.approx(0.1)
    assert drawn["kick"][0] == pytest.approx(1.0)
    assert _support(drawn["start"] - positions).tolist() == [True, False, False]


def test_a_frozen_list_is_rejected():
    positions = np.zeros((2, 3))
    fixed = np.ones(2, dtype=bool)
    with pytest.raises(RuntimeError, match="no free atom"):
        displace_listed(
            positions,
            fixed,
            None,
            (False, False, False),
            [0],
            3.3,
            0.1,
            np.random.default_rng(0),
        )
