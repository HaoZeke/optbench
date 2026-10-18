"""Fingerprints, the squared-energy loss, and one n2p2 Kalman step."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optbench.n2p2_ml import (
    EPOCH_CAP,
    FingerprintSpec,
    G1,
    G2,
    MlDataset,
    element_width,
    fingerprints_frame,
    g2_value,
    init_weights,
    image_jacobian,
    loss_and_gradient,
    n_weights,
    parameter_slices,
    parse_fpparas,
    reference_kalman_update,
    run_start,
    start_rng,
)

ROOT = Path(__file__).resolve().parents[1]
N2P2 = Path("/tmp/n2p2/src/libnnptrain/KalmanFilter.cpp")


def _equilateral() -> tuple[np.ndarray, np.ndarray]:
    height = 2.5 * math.sqrt(3.0) / 2.0
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, height, 0.0]], dtype=np.float64)
    return positions, np.array([30.0, 30.0, 30.0])


def test_parser_counts_the_archived_fingerprints():
    pt = parse_fpparas((ROOT / "archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0/fpParas.dat").read_text())
    assert pt.elements == ("Pt",)
    assert pt.n_fp == (8,)
    assert [feature.index for feature in pt.g1] == [0, 1, 2, 3]
    assert [feature.index for feature in pt.g2] == [4, 5, 6, 7]
    ethene = parse_fpparas((ROOT / "archive/entries/mlopt/C2H4/pyamff-adam10-7-fcoeff0/fpParas.dat").read_text())
    assert ethene.elements == ("C", "H")
    assert ethene.n_fp == (7, 7)
    carbon = [feature for feature in ethene.g2 if feature.center == 0]
    assert [(feature.neighbor_a, feature.neighbor_b, feature.index) for feature in carbon] == [
        (0, 0, 3),
        (0, 1, 4),
        (0, 1, 5),
        (1, 1, 6),
    ]
    assert carbon[1].eta == carbon[2].eta
    assert carbon[1].zeta == carbon[2].zeta
    assert carbon[1].lam == carbon[2].lam
    assert n_weights(pt.n_fp) == 175
    assert n_weights(ethene.n_fp) == 330
    assert element_width(8) == 175
    assert element_width(7) == 165


def test_fcoeff0_fpparas_copies_match():
    for problem, stem in (("emt-pt38", "pt"), ("C2H4", "c2h4")):
        names = [
            f"pyamff-adam10-7-fcoeff0/fpParas.dat",
            f"pyamff-lbfgs10-7-fcoeff0/fpParas.dat",
            f"pyamff-rprop10-7-fcoeff0/fpParas.dat",
        ]
        base = (ROOT / "archive/entries/mlopt" / problem / names[0]).read_bytes()
        for name in names[1:]:
            assert (ROOT / "archive/entries/mlopt" / problem / name).read_bytes() == base
        del stem


def test_equilateral_g1_and_g2_match_the_fortran_products():
    positions, lengths = _equilateral()
    text = "\n".join(
        [
            "#FP type",
            "BP",
            "#Elements",
            "Pt",
            "0.0",
            "#nG1s nG2s",
            "1 1",
            "#G1",
            "G1 Pt Pt 0.05 0.0 5.0",
            "#G2",
            "G2 Pt Pt Pt 0.005 1.0 1.0 0.0 5.0",
        ]
    )
    spec = parse_fpparas(text)
    frame = fingerprints_frame(positions, ["Pt", "Pt", "Pt"], lengths, spec)
    distance = 2.5
    rc = 5.0
    g1 = math.exp(-0.05 * distance * distance / (rc * rc)) * 0.5 * (math.cos(math.pi * distance / rc) + 1.0)
    assert frame[:, 0] == pytest.approx(2.0 * g1)
    feature = spec.g2[0]
    vij = positions[1] - positions[0]
    vik = positions[2] - positions[0]
    expected = g2_value(distance, distance, distance, vij, vik, feature)
    cosine = 0.5
    theta = math.acos(cosine)
    base = 1.0 + 1.0 * math.cos(theta)
    radial = (math.exp(-distance * distance / (rc * rc)) ** 3) ** 0.005
    cutoff = (0.5 * (math.cos(math.pi * distance / rc) + 1.0)) ** 3
    formula = base ** (1.0 - 1.0) * base * radial * cutoff * 2.0 ** (1.0 - 1.0)
    assert expected == pytest.approx(formula)
    assert frame[:, 1] == pytest.approx(expected)


def test_duplicate_angular_slots_receive_the_same_value():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.2, 1.1, 0.0], [-0.4, -0.9, 0.2]],
        dtype=np.float64,
    )
    spec = parse_fpparas((ROOT / "archive/entries/mlopt/C2H4/pyamff-adam10-7-fcoeff0/fpParas.dat").read_text())
    frame = fingerprints_frame(positions, ["C", "C", "H", "H"], np.array([20.0, 20.0, 20.0]), spec)
    carbon = frame[0]
    assert carbon[4] != 0.0
    assert carbon[4] == pytest.approx(carbon[5])
    assert carbon.shape == (7,)


def test_loss_gradient_matches_a_finite_difference():
    rng = np.random.Generator(np.random.PCG64(11))
    n_fp = (2,)
    weights = init_weights(n_fp, rng)
    element_index = np.array([0, 0], dtype=np.int32)
    fingerprints = rng.normal(size=(2, 2, 2))
    energy = np.array([0.3, -0.2])
    loss, _rmse, grad = loss_and_gradient(weights, element_index, n_fp, fingerprints, energy, 2)
    step = 1e-6
    numeric = np.zeros_like(weights)
    for index in range(weights.size):
        bumped = weights.copy()
        bumped[index] += step
        bumped_loss, _rmse, _grad = loss_and_gradient(bumped, element_index, n_fp, fingerprints, energy, 2)
        numeric[index] = (bumped_loss - loss) / step
    assert grad == pytest.approx(numeric, rel=1e-5, abs=1e-5)
    peaks = [float(np.max(np.abs(grad[start:stop]))) for start, stop in parameter_slices(n_fp)]
    assert len(peaks) == 6


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_library_matches_the_reference_loss_and_one_filter_pass():
    from optbench.n2p2_ml import KALMAN_ETAMAX, KALMAN_ETATAU, KALMAN_ETA0, KALMAN_Q0, KALMAN_QMIN, KALMAN_QTAU, KalmanSession

    rng = np.random.Generator(np.random.PCG64(19))
    n_fp = (3, 2)
    element_index = np.array([0, 1, 1], dtype=np.int32)
    fingerprints = rng.normal(size=(2, 3, 3))
    energy = np.array([1.5, -0.4])
    dataset = MlDataset(("A", "B"), n_fp, element_index, fingerprints, energy)
    weights = init_weights(n_fp, rng)
    loss, rmse, grad = loss_and_gradient(weights, element_index, n_fp, fingerprints, energy, 3)
    session = KalmanSession(dataset, 0.02)
    try:
        session.set_weights(weights)
        evaluation = session.evaluate()
        assert evaluation["loss"] == pytest.approx(loss, rel=1e-10, abs=1e-10)
        assert evaluation["energy_rmse"] == pytest.approx(rmse, rel=1e-10, abs=1e-10)
        assert evaluation["grad"] == pytest.approx(grad, rel=1e-9, abs=1e-9)
        slices = parameter_slices(n_fp)
        peaks = [float(np.max(np.abs(grad[start:stop]))) for start, stop in slices]
        assert evaluation["tensor_maxabs"] == pytest.approx(peaks)
        updates = dataset.n_images
        state_w = weights.copy()
        state_p = np.eye(state_w.size) / 1.0e-2
        eta = KALMAN_ETA0
        q = KALMAN_Q0
        order = np.arange(updates)
        for image in order:
            pred, jacobian = image_jacobian(state_w, element_index, n_fp, fingerprints[int(image)])
            state_w, state_p, eta, q = reference_kalman_update(
                state_w,
                state_p,
                jacobian,
                float(energy[image]) - pred,
                eta,
                q,
                etatau=KALMAN_ETATAU / updates,
                etamax=KALMAN_ETAMAX,
                qtau=KALMAN_QTAU / updates,
                qmin=KALMAN_QMIN,
            )
        assert session.epoch(order) == 0
        assert session.get_weights() == pytest.approx(state_w, rel=1e-9, abs=1e-9)
        assert session.updates() == updates
    finally:
        session.close()


@pytest.mark.skipif(not N2P2.is_file(), reason="n2p2 source is not on this machine")
def test_harness_counts_each_loss_evaluation_until_the_cap():
    rng = start_rng(0)
    n_fp = (2,)
    weights_unused = init_weights(n_fp, rng)
    del weights_unused
    dataset = MlDataset(
        ("Pt",),
        n_fp,
        np.zeros(2, dtype=np.int32),
        np.zeros((4, 2, 2)),
        np.array([0.2, 0.2, 0.2, 0.2]),
    )
    record = run_start(dataset, 3, cap=4, tol=1e-30, shuffle=False)
    assert record["force_calls"] == 4
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert record["kalman_epochs"] == 3
    assert record["seed"] == 20261015 + 3
    assert EPOCH_CAP == 10000


def test_weight_draw_is_uniform_on_the_half_open_interval():
    weights = init_weights((8,), start_rng(4))
    assert weights.shape == (175,)
    assert weights.min() >= -1.0
    assert weights.max() < 1.0
    again = init_weights((8,), start_rng(4))
    assert again == pytest.approx(weights)


def test_spec_objects_are_the_feature_records():
    spec = FingerprintSpec(("Pt",), (1,), (G1(0, 0, 0.05, 0.0, 5.0, 0),), (G2(0, 0, 0, 0.005, 1.0, 1.0, 0.0, 5.0, 1),), 5.0)
    assert spec.stride == 1
