"""The mlopt loss, fingerprints, and parameter-set gradient."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.board import stated_full_n
from optbench.ml_landscape import (
    HIDDEN,
    build_landscape,
    compute_fingerprints,
    initial_parameters,
    layout_for,
    loss_and_grad,
    parameter_set_norms,
    parse_fingerprints,
    squared_error,
)
from optbench.problems import load_all

ROOT = Path(__file__).resolve().parents[1]
PT_FP = ROOT / "archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0/fpParas.dat"
C2H4_FP = ROOT / "archive/entries/mlopt/C2H4/pyamff-adam10-7-fcoeff0/fpParas.dat"


def _cell(length: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3) * length, np.array([False, False, False])


def test_archived_fingerprint_files_match_the_published_widths():
    pt = parse_fingerprints(PT_FP.read_text(encoding="utf-8"))
    c2h4 = parse_fingerprints(C2H4_FP.read_text(encoding="utf-8"))
    assert pt.elements == ("Pt",)
    assert pt.counts() == {"Pt": 8}
    assert c2h4.elements == ("C", "H")
    assert c2h4.counts() == {"C": 7, "H": 7}
    assert layout_for(pt.counts(), pt.elements, HIDDEN).size == 175
    assert layout_for(c2h4.counts(), c2h4.elements, HIDDEN).size == 330


def test_initial_draw_is_uniform_and_shared_by_start_index():
    layout = layout_for({"Pt": 8}, ("Pt",), HIDDEN)
    first = initial_parameters(layout, 0)
    again = initial_parameters(layout, 0)
    second = initial_parameters(layout, 1)
    assert first.shape == (175,)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, second)
    assert np.all(first >= -1.0)
    assert np.all(first < 1.0)


def test_squared_error_is_the_archived_sum():
    predicted_energy = np.array([[1.0, 4.0], [3.0, 1.0]])
    energy = np.array([0.0, 1.0])
    n_atoms = np.array([2.0, 2.0])
    predicted_force = np.zeros((2, 2, 2, 3))
    predicted_force[0, 0, 0, 0] = 1.0
    predicted_force[0, 1, 0, 0] = 2.0
    force = np.zeros((2, 2, 3))
    energy_loss, force_loss = squared_error(predicted_energy, energy, predicted_force, force, n_atoms)
    assert energy_loss == pytest.approx([1.25, 4.0])
    assert force_loss == pytest.approx([1.0 / 6.0, 4.0 / 6.0])


def test_g1_matches_the_pair_formula_and_ignores_the_far_image():
    positions = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=np.float64)
    text = "\n".join(
        [
            "BP",
            "A",
            "0.0",
            "1 0",
            "G1 A A 1.0 0.0 3.0",
        ]
    )
    parameters = parse_fingerprints(text)
    symbols = ("A", "A", "B")
    cell, pbc = _cell()
    fingerprints, derivatives = compute_fingerprints(
        positions, symbols, cell, pbc, parameters, with_derivatives=True
    )
    distance = 2.0
    rc = 3.0
    expected = np.exp(-((distance / rc) ** 2)) * 0.5 * (np.cos(np.pi * distance / rc) + 1.0)
    assert fingerprints[0, 0, 0] == pytest.approx(expected)
    assert fingerprints[0, 1, 0] == pytest.approx(expected)
    assert fingerprints[0, 2, 0] == 0.0
    wrapped = positions.copy()
    wrapped[0, 1, 0] = 9.8
    cell_p = np.eye(3) * 10.0
    pbc_on = np.array([True, False, False])
    wrapped_fp, _unused = compute_fingerprints(
        wrapped, ("A", "A", "B"), cell_p, pbc_on, parameters, with_derivatives=False
    )
    near = 0.2
    near_value = np.exp(-((near / rc) ** 2)) * 0.5 * (np.cos(np.pi * near / rc) + 1.0)
    assert wrapped_fp[0, 0, 0] == pytest.approx(near_value)
    shifted = positions.copy()
    shifted[0, 1, 0] += 1e-6
    plus, _unused = compute_fingerprints(shifted, symbols, cell, pbc, parameters, with_derivatives=False)
    numerical = (plus[0, 0, 0] - fingerprints[0, 0, 0]) / 1e-6
    assert derivatives[0, 0, 0, 1, 0] == pytest.approx(numerical, rel=1e-5, abs=1e-7)


def test_g2_uses_index_order_and_drops_a_long_third_side():
    positions = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float64)
    text = "\n".join(
        [
            "BP",
            "C",
            "0.0",
            "0 2",
            "G2 C H H 0.005 1.0 1.0 0.0 3.0",
            "G2 C H C 0.005 1.0 1.0 0.0 3.0",
        ]
    )
    parameters = parse_fingerprints(text)
    symbols = ("C", "H", "H")
    cell, pbc = _cell()
    fingerprints, _derivatives = compute_fingerprints(
        positions, symbols, cell, pbc, parameters, with_derivatives=True
    )
    rc = 3.0
    eta = 0.005
    side = 1.0
    diagonal = np.sqrt(2.0)
    cos = 0.0
    fc = 0.5 * (np.cos(np.pi * side / rc) + 1.0)
    fc_diag = 0.5 * (np.cos(np.pi * diagonal / rc) + 1.0)
    expo = np.exp(-eta * (side**2 + side**2 + diagonal**2) / rc**2)
    expected = (1.0 + cos) * expo * fc * fc * fc_diag
    assert fingerprints[0, 0, 0] == pytest.approx(expected)
    assert fingerprints[0, 0, 1] == 0.0
    long = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.1, 0.0]]], dtype=np.float64)
    short = "\n".join(
        [
            "BP",
            "C",
            "0.0",
            "0 1",
            "G2 C H H 0.005 1.0 1.0 0.0 1.5",
        ]
    )
    dropped, _unused = compute_fingerprints(
        long, ("C", "H", "H"), cell, pbc, parse_fingerprints(short), with_derivatives=False
    )
    assert dropped[0, 0, 0] == 0.0
    inside, deriv = compute_fingerprints(
        positions, symbols, cell, pbc, parameters, with_derivatives=True
    )
    nudged = positions.copy()
    nudged[0, 1, 1] += 1e-6
    plus, _unused = compute_fingerprints(nudged, symbols, cell, pbc, parameters, with_derivatives=False)
    numerical = (plus[0, 0, 0] - inside[0, 0, 0]) / 1e-6
    assert deriv[0, 0, 0, 1, 1] == pytest.approx(numerical, rel=1e-4, abs=1e-6)


def test_parameter_gradient_matches_a_finite_difference():
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.1, 0.2]],
            [[0.1, 0.0, 0.2], [1.3, 0.2, -0.1], [0.0, 1.0, 0.4]],
        ],
        dtype=np.float64,
    )
    text = "\n".join(
        [
            "BP",
            "A",
            "0.0",
            "1 1",
            "G1 A A 0.5 0.0 3.0",
            "G2 A A A 0.005 1.0 -1.0 0.0 3.0",
        ]
    )
    parameters = parse_fingerprints(text)
    symbols = ("A", "A", "A")
    cell, pbc = _cell()
    energies = np.array([0.4, -0.2])
    forces = np.zeros((2, 3, 3))
    forces[0, 1, 0] = 0.3
    forces[1, 2, 1] = -0.2
    landscape = build_landscape(
        positions,
        symbols,
        cell,
        pbc,
        energies,
        forces,
        parameters,
        with_forces=True,
        hidden=(2, 2),
    )
    start = initial_parameters(landscape.layout, 3)
    base = loss_and_grad(start, landscape, energy_coefficient=1.0, force_coefficient=0.1)
    step = 1e-6
    numerical = np.zeros_like(start)
    for index in range(start.size):
        plus = start.copy()
        minus = start.copy()
        plus[index] += step
        minus[index] -= step
        high = loss_and_grad(plus, landscape, energy_coefficient=1.0, force_coefficient=0.1)["loss"][0]
        low = loss_and_grad(minus, landscape, energy_coefficient=1.0, force_coefficient=0.1)["loss"][0]
        numerical[index] = (high - low) / (2.0 * step)
    assert base["gradient"][0] == pytest.approx(numerical, rel=1e-5, abs=1e-6)
    groups = parameter_set_norms(base["gradient"], landscape.layout)
    assert groups.shape == (1,)
    assert groups[0] == pytest.approx(np.sqrt(np.max([np.sum(base["gradient"][0, spec.offset : spec.offset + spec.size] ** 2) for spec in landscape.layout.specs])))


def test_energy_only_reports_the_missing_force_rmse():
    positions = np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]], dtype=np.float64)
    text = "\n".join(["BP", "A", "0.0", "1 0", "G1 A A 0.2 0.0 4.0"])
    parameters = parse_fingerprints(text)
    cell, pbc = _cell()
    landscape = build_landscape(
        positions,
        ("A", "A"),
        cell,
        pbc,
        np.array([1.0]),
        np.zeros((1, 2, 3)),
        parameters,
        with_forces=False,
        hidden=(2, 2),
    )
    result = loss_and_grad(initial_parameters(landscape.layout, 0), landscape, force_coefficient=0.0)
    assert result["force_rmse"][0] == -1.0
    assert result["energy_rmse"][0] > 0.0


def test_mlopt_full_start_count_is_the_initialization_count():
    problems = load_all(ROOT / "problems")
    assert stated_full_n(ROOT, problems["mlopt/emt-pt38"]) == 50
    assert stated_full_n(ROOT, problems["mlopt/C2H4"]) == 50
