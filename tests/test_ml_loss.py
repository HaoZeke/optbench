"""The mlopt squared-error loss and its fingerprint loops."""

import math
from pathlib import Path

import numpy as np
import pytest

from optbench.board import stated_full_n
from optbench.ml_loss import (
    G1,
    G2,
    g1_value,
    g2_value,
    parameter_count,
    parse_fp,
    squared_error,
)
from optbench.problems import load_problem

ROOT = Path(__file__).resolve().parents[1]


def test_g1_matches_the_radial_formula():
    row = G1(center="Pt", neighbor="Pt", eta=0.05, rs=0.0, rc=5.0, slot=0)
    value = g1_value(["Pt", "H"], [2.5, 1.0], row)
    expected = math.exp(-0.05 * (2.5**2) / 25.0) * 0.5 * (math.cos(math.pi * 2.5 / 5.0) + 1.0)
    assert value == pytest.approx(expected)


def test_g2_uses_the_lower_index_symbol_and_the_raw_jk_distance():
    row = G2(
        center="C",
        neighbor1="H",
        neighbor2="C",
        eta=0.005,
        zeta=1.0,
        lam=1.0,
        thetas=0.0,
        rc=6.5,
        slot=0,
    )
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    # Center atom 0. Neighbors sorted by index are 1 then 2, symbols H then C.
    value = g2_value(
        [1, 2],
        ["H", "C"],
        [1.0, 1.0],
        [90.0],
        positions,
        row,
    )
    swapped = G2(
        center="C",
        neighbor1="C",
        neighbor2="H",
        eta=0.005,
        zeta=1.0,
        lam=1.0,
        thetas=0.0,
        rc=6.5,
        slot=0,
    )
    assert g2_value([1, 2], ["H", "C"], [1.0, 1.0], [90.0], positions, swapped) == 0.0
    rjk = math.sqrt(2.0)
    pref = 2.0 ** (1.0 - 1.0)
    angular = 1.0 + 1.0 * math.cos(math.radians(90.0))
    radial = (
        math.exp(-0.005 * 1.0 / 6.5**2)
        * math.exp(-0.005 * 1.0 / 6.5**2)
        * math.exp(-0.005 * rjk**2 / 6.5**2)
    )

    def fc(distance: float) -> float:
        return 0.5 * (math.cos(math.pi * distance / 6.5) + 1.0)

    expected = pref * angular * radial * fc(1.0) * fc(1.0) * fc(rjk)
    assert value == pytest.approx(expected)


def test_squared_error_sums_per_image_and_drops_forces_at_coefficient_zero():
    energy = squared_error([2.0, 4.0], [0.0, 0.0], None, None, 2, 1.0, 0.0)
    assert energy["energyloss"] == pytest.approx(5.0)
    assert energy["loss"] == pytest.approx(5.0)
    assert energy["energy_rmse"] == pytest.approx(math.sqrt(2.5))
    assert energy["force_rmse"] == -1.0
    pred_f = np.zeros((2, 2, 3))
    ref_f = np.zeros((2, 2, 3))
    pred_f[0, 0, 0] = 3.0
    both = squared_error([0.0, 0.0], [0.0, 0.0], pred_f, ref_f, 2, 1.0, 0.1)
    # One component of 3 on the first image: sum of squares 9, divided by 3N = 6.
    assert both["forceloss"] == pytest.approx(9.0 / 6.0)
    assert both["loss"] == pytest.approx(0.1 * 1.5)


def test_shipped_fingerprint_files_have_the_archived_widths():
    pt = parse_fp(ROOT / "archive/entries/mlopt/emt-pt38/pyamff-lbfgs10-7-fcoeff0/fpParas.dat")
    mol = parse_fp(ROOT / "archive/entries/mlopt/C2H4/pyamff-lbfgs10-7-fcoeff0/fpParas.dat")
    assert pt.elements == ("Pt",)
    assert pt.n_fp == {"Pt": 8}
    assert pt.rc == pytest.approx(5.0)
    assert parameter_count(8) == 175
    assert mol.elements == ("C", "H")
    assert mol.n_fp == {"C": 7, "H": 7}
    assert mol.rc == pytest.approx(6.5)
    assert parameter_count(7) * 2 == 330
    assert all(row.thetas == 0.0 for row in mol.g2)


def test_mlopt_full_n_is_the_initialization_count():
    emt = load_problem(ROOT / "problems/mlopt/emt-pt38.toml")
    c2h4 = load_problem(ROOT / "problems/mlopt/C2H4.toml")
    assert stated_full_n(ROOT, emt) == 50
    assert stated_full_n(ROOT, c2h4) == 50


def test_torch_fingerprints_match_the_pyamff_loops_and_the_jacobian():
    pytest.importorskip("ase")
    pytest.importorskip("torch")
    from ase import Atoms

    from optbench.ml_loss import fingerprints_and_jacobian, neighbor_data, reference_fingerprints

    spec = parse_fp(ROOT / "archive/entries/mlopt/C2H4/pyamff-lbfgs10-7-fcoeff0/fpParas.dat")
    atoms = Atoms(
        "CHHC",
        positions=[[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.2, 1.2, 0.1], [2.0, 1.5, 0.3]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    geometry = neighbor_data(atoms, spec.rc)
    reference = reference_fingerprints(geometry, spec)
    values, jacobian = fingerprints_and_jacobian(geometry, spec)
    assert values == pytest.approx(reference, abs=1e-8, rel=1e-8)
    point = geometry["positions"].copy()
    step = 1e-6
    # One coordinate is enough to lock the derivative.
    bumped = dict(geometry)
    shifted = point.copy()
    shifted[1, 0] += step
    bumped["positions"] = shifted
    # Distances and angles in the reference come from ASE, so the finite
    # difference uses the torch values, which are the function being differentiated.
    import torch

    from optbench.ml_loss import fingerprints_torch

    def at(positions):
        tensor = torch.tensor(positions, dtype=torch.float64)
        cell = torch.tensor(geometry["cell"], dtype=torch.float64)
        return fingerprints_torch(tensor, geometry["symbols"], geometry["neighbors"], cell, spec).detach().numpy()

    plus = at(shifted)
    minus_pos = point.copy()
    minus_pos[1, 0] -= step
    minus = at(minus_pos)
    numeric = (plus - minus) / (2.0 * step)
    assert jacobian[:, :, 1, 0] == pytest.approx(numeric, abs=1e-5, rel=1e-5)
