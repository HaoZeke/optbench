"""eOn r2205 endpoint link on the Pt island.

``ProcessSearchJob`` calls ``Matter::compare`` with ``indistinguishable``
left false. Two minima are the same when ``distance_difference`` 0.1 Å
is strictly above ``perAtomNorm``. A search is connected when exactly
one minimized endpoint matches the reactant. ``fraction_connected`` is
the 2014 ``done.py`` ratio of searches that are not ``Not Connected``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import geometry_archive, resolve_problem
from optbench.connection import (
    DISTANCE_DIFFERENCE,
    ENERGY_DIFFERENCE,
    ProcessCatalog,
    connected_endpoints,
    done_fraction,
    geometry_status,
    link_tolerance,
    per_atom_max,
    relink,
    result_is_not_connected,
    same_minimum,
    same_process,
)
from optbench.eon_parse import saddle_progress
from optbench.surfaces import read_con

ROOT = Path(__file__).resolve().parents[1]


def _island():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    path = geometry_archive(
        ROOT,
        {"id": problem["id"], "input": {"archive_path": "website/saddle-search/pt-island/pt-island.con"}},
    )
    positions, fixed, cell = read_con(path.read_text(encoding="utf-8"))
    return positions, fixed, cell


def _cartesian_rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
    delta = np.asarray(moved, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def test_link_tolerance_reads_the_problem_threshold():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    assert link_tolerance(problem) == 0.1
    assert link_tolerance(None) == DISTANCE_DIFFERENCE
    assert problem["connection"]["rule"] == "eon-r2205-per-atom-max"


def test_single_adatom_hop_is_a_distinct_minimum():
    positions, fixed, cell = _island()
    free = ~fixed
    assert int(free.sum()) == 175
    assert cell == pytest.approx([19.2088, 19.0118, 30.0])
    assert not bool(fixed[0])
    hopped = positions.copy()
    hopped[0, 0] += 2.71
    assert per_atom_max(hopped, positions, cell) == pytest.approx(2.71)
    assert not same_minimum(hopped, positions, cell)
    rmsd = _cartesian_rmsd(hopped[free], positions[free])
    assert rmsd == pytest.approx(2.71 / np.sqrt(175))
    assert rmsd < 0.5


def test_small_relaxation_is_the_same_minimum():
    positions, _fixed, cell = _island()
    relaxed = positions.copy()
    relaxed[0, 0] += 0.05
    assert per_atom_max(relaxed, positions, cell) == pytest.approx(0.05)
    assert same_minimum(relaxed, positions, cell)


def test_exactly_one_endpoint_links_the_search():
    positions, _fixed, cell = _island()
    hopped = positions.copy()
    hopped[0, 0] += 2.71
    relaxed = positions.copy()
    relaxed[0, 0] += 0.05
    assert connected_endpoints(positions, hopped, relaxed, cell)
    assert connected_endpoints(positions, relaxed, hopped, cell)
    assert not connected_endpoints(positions, hopped, hopped, cell)
    assert not connected_endpoints(positions, relaxed, relaxed, cell)


def test_distance_difference_is_strict():
    reactant = np.zeros((1, 3))
    moved = np.array([[0.1, 0.0, 0.0]])
    assert per_atom_max(moved, reactant, None) == 0.1
    assert not same_minimum(moved, reactant, None)
    inside = np.array([[0.099, 0.0, 0.0]])
    assert same_minimum(inside, reactant, None)


def test_minimum_image_folds_every_positive_axis():
    positions, _fixed, cell = _island()
    wrapped = positions.copy()
    wrapped[0, 0] += float(cell[0]) - 0.04
    assert per_atom_max(wrapped, positions, cell) == pytest.approx(0.04, abs=1e-9)
    assert same_minimum(wrapped, positions, cell)
    matrix = np.diag(cell)
    assert per_atom_max(wrapped, positions, matrix) == pytest.approx(0.04, abs=1e-9)
    negative = positions.copy()
    negative[0, 1] -= float(cell[1]) - 0.04
    assert per_atom_max(negative, positions, cell) == pytest.approx(0.04, abs=1e-9)


def test_relink_needs_endpoint_coordinates():
    positions, _fixed, cell = _island()
    hopped = positions.copy()
    hopped[0, 0] += 2.71
    relaxed = positions.copy()
    relaxed[0, 0] += 0.05
    linked = {
        "success": False,
        "failure": "divergence",
        "reasons": ["not_connected"],
        "minimum": positions.tolist(),
        "endpoint_plus": relaxed.tolist(),
        "endpoint_minus": hopped.tolist(),
    }
    assert relink(linked, cell) == "updated"
    assert linked["success"] is True
    assert linked["reasons"] == []
    assert linked["failure"] is None
    both = {
        "success": True,
        "failure": None,
        "reasons": [],
        "minimum": positions.tolist(),
        "endpoint_plus": hopped.tolist(),
        "endpoint_minus": hopped.tolist(),
    }
    assert relink(both, cell) == "updated"
    assert both["success"] is False
    assert both["reasons"] == ["not_connected"]
    scalars = {"success": False, "reasons": ["not_connected"], "rmsd_plus": 0.22, "rmsd_minus": 0.32}
    assert geometry_status(scalars).startswith("scalars:")
    assert relink(scalars, cell) == "missing"
    assert scalars["success"] is False
    capped = {
        "success": False,
        "failure": "cap",
        "reasons": ["cap"],
        "minimum": positions.tolist(),
        "endpoint_plus": relaxed.tolist(),
        "endpoint_minus": hopped.tolist(),
    }
    assert relink(capped, cell) == "unchanged"
    assert capped["success"] is False
    assert geometry_status({}) == "absent"


def test_find_repeat_keeps_a_single_atom_hop_that_rmsd_merges():
    positions, fixed, cell = _island()
    free = ~fixed
    hopped = positions.copy()
    hopped[0, 0] += 2.71
    other = positions.copy()
    other[1, 1] += 2.71
    assert _cartesian_rmsd(hopped[free], positions[free]) < 0.5
    assert _cartesian_rmsd(other[free], hopped[free]) < 0.5
    assert not same_process(hopped, positions, cell, 0.60, 0.60)
    assert not same_process(other, hopped, cell, 0.60, 0.605)
    nudged = hopped.copy()
    nudged[0, 0] += 0.04
    assert same_process(nudged, hopped, cell, 0.60, 0.609)
    assert not same_process(nudged, hopped, cell, 0.60, 0.60 + ENERGY_DIFFERENCE + 1e-9)
    catalog = ProcessCatalog(cell)
    first = catalog.observe(judged=True, linked=True, structure=hopped, barrier=0.60)
    second = catalog.observe(judged=True, linked=True, structure=other, barrier=0.70)
    repeat = catalog.observe(judged=True, linked=True, structure=nudged, barrier=0.602)
    missed = catalog.observe(judged=True, linked=False, structure=other, barrier=0.4)
    catalog.observe(judged=False)
    assert first["success"] is True
    assert second["success"] is True
    assert repeat["reasons"] == ["duplicate"]
    assert missed["reasons"] == ["not_connected"]
    assert catalog.unique == 2
    assert catalog.qualifying == 2
    assert catalog.not_connected == 1
    assert catalog.searches == 5
    assert catalog.fraction_connected == done_fraction(5, 1)


def test_fraction_connected_is_the_r2205_done_py_ratio():
    lines = [
        "    wuid       type    barrier   max-dist    sad-fcs   mins-fcs   pref-fcs    result",
        "-------------------------------------------------------------------------------------",
        "       1     random    0.40000    0.00000         10          5          0    good-0",
        "       2     random    0.00000    0.00000         10          5          0    Not Connected",
        "       3     random    1.60000    0.00000         10          5          0    good-1",
        "       4     random    0.20000    0.00000         10          5          0    good-2",
    ]
    progress = saddle_progress("\n".join(lines) + "\n", target=10)
    assert result_is_not_connected("Not Connected")
    assert not result_is_not_connected("good-0")
    assert progress["fraction_connected"] == done_fraction(4, 1)
    assert progress["unique_saddles"] == 3
    assert progress["qualifying"] == 2
    catalog = ProcessCatalog([19.2088, 19.0118, 30.0])
    for result, height in (("good-0", 0.4), ("Not Connected", 0.0), ("good-1", 1.6), ("good-2", 0.2)):
        catalog.observe(
            judged=True,
            linked=not result_is_not_connected(result),
            barrier=height,
            structure=None,
        )
    assert catalog.fraction_connected == progress["fraction_connected"]
