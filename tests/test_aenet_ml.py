"""Gradient and trainer checks for the aenet mlopt cell.

The Fortran library is optional. A machine without the aenet build still
runs the loss tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optbench.aenet_ml import (
    LossModel,
    Layout,
    build_descriptors,
    gradients_ok,
    initial_weights,
    lbfgsb_available,
    parse_fpparas,
    train_start,
)

FP = """
BP
Pt
0.0
1 1
G1 Pt Pt 0.05 0.0 5.0
G2 Pt Pt Pt 0.005 1.0 1.0 0.0 5.0
"""


def _model(seed: int = 0, *, forces: bool):
    rng = np.random.default_rng(seed)
    layout = Layout(["Pt"], {"Pt": 2})
    images, atoms, width = 3, 4, 2
    fp = {"Pt": rng.normal(size=(images, atoms, width))}
    energy = rng.normal(size=(images,))
    dG = None
    force_ref = None
    c_force = 0.0
    if forces:
        dG = {"Pt": rng.normal(scale=0.1, size=(images, atoms, width, atoms, 3))}
        force_ref = rng.normal(scale=0.1, size=(images, atoms, 3))
        c_force = 0.1
    model = LossModel(
        fp=fp,
        atoms={"Pt": np.arange(atoms)},
        energy=energy,
        forces=force_ref,
        layout=layout,
        c_energy=1.0,
        c_force=c_force,
        dG=dG,
    )
    return model, initial_weights(layout, 0)


def _finite_difference(model, weights, eps=1e-6):
    loss, grad, _aux = model.loss_grad(weights)
    numerical = np.empty_like(grad)
    for index in range(weights.size):
        plus = np.array(weights, copy=True)
        minus = np.array(weights, copy=True)
        plus[index] += eps
        minus[index] -= eps
        high, _g, _a = model.loss_grad(plus)
        low, _g, _a = model.loss_grad(minus)
        numerical[index] = (high - low) / (2.0 * eps)
    return loss, grad, numerical


def test_energy_gradient_matches_a_central_difference():
    model, weights = _model(forces=False)
    _loss, grad, numerical = _finite_difference(model, weights)
    assert np.allclose(grad, numerical, rtol=1e-5, atol=1e-7)


def test_force_gradient_matches_a_central_difference():
    model, weights = _model(forces=True)
    _loss, grad, numerical = _finite_difference(model, weights)
    assert np.allclose(grad, numerical, rtol=1e-5, atol=1e-6)


def test_residual_jacobian_is_the_loss_gradient():
    model, weights = _model(forces=True)
    loss, grad, aux = model.loss_grad(weights, jacobian=True)
    assert math.isclose(aux["sse"], loss, rel_tol=1e-10, abs_tol=1e-10)
    rebuilt = aux["jacobian"] @ aux["residuals"]
    assert np.allclose(rebuilt, grad, rtol=1e-8, atol=1e-8)


def test_g1_matches_the_closed_form():
    elements, specs = parse_fpparas(FP)
    pos = np.zeros((1, 2, 3))
    pos[0, 1, 0] = 2.0
    cell = np.diag([20.0, 20.0, 20.0])
    fp, dG, _atoms = build_descriptors(pos, cell, ["Pt", "Pt"], elements, specs, derivatives=True)
    radius = 2.0
    rc = 5.0
    eta = 0.05
    expected = math.exp(-eta * ((radius - 0.0) / rc) ** 2) * 0.5 * (math.cos(math.pi * radius / rc) + 1.0)
    assert fp["Pt"][0, 0, 0] == pytest.approx(expected)
    assert fp["Pt"][0, 1, 0] == pytest.approx(expected)
    # Move the second atom along x. The neighbor list stays the same.
    eps = 1e-6
    pos[0, 1, 0] = 2.0 + eps
    plus, _d, _a = build_descriptors(pos, cell, ["Pt", "Pt"], elements, specs, derivatives=False)
    pos[0, 1, 0] = 2.0 - eps
    minus, _d, _a = build_descriptors(pos, cell, ["Pt", "Pt"], elements, specs, derivatives=False)
    numerical = (plus["Pt"][0, :, 0] - minus["Pt"][0, :, 0]) / (2.0 * eps)
    assert numerical[0] == pytest.approx(dG["Pt"][0, 0, 0, 1, 0], rel=1e-6, abs=1e-8)
    assert numerical[1] == pytest.approx(dG["Pt"][0, 1, 0, 1, 0], rel=1e-6, abs=1e-8)


def test_g2_derivative_matches_a_central_difference():
    elements, specs = parse_fpparas(FP)
    rng = np.random.default_rng(1)
    pos = rng.normal(scale=0.4, size=(1, 4, 3))
    pos[0, :, 0] += np.arange(4) * 1.3
    cell = np.diag([30.0, 30.0, 30.0])
    symbols = ["Pt", "Pt", "Pt", "Pt"]
    _fp, dG, _atoms = build_descriptors(pos, cell, symbols, elements, specs, derivatives=True)
    eps = 1e-6
    atom, coord, center = 2, 1, 0
    pos[0, atom, coord] += eps
    plus, _d, _a = build_descriptors(pos, cell, symbols, elements, specs, derivatives=False)
    pos[0, atom, coord] -= 2.0 * eps
    minus, _d, _a = build_descriptors(pos, cell, symbols, elements, specs, derivatives=False)
    numerical = (plus["Pt"][0, center, 1] - minus["Pt"][0, center, 1]) / (2.0 * eps)
    assert numerical == pytest.approx(dG["Pt"][0, center, 1, atom, coord], rel=1e-5, abs=1e-7)


def test_parameter_l2_stop_uses_each_tensor():
    model, weights = _model(forces=False)
    _loss, grad, _aux = model.loss_grad(weights)
    assert gradients_ok(grad, model.layout, tolerance=1.0e9)
    assert gradients_ok(np.zeros_like(grad), model.layout, tolerance=0.02)


@pytest.mark.skipif(not lbfgsb_available(), reason="aenet liblbfgsb.a is not built")
def test_setulb_minimizes_a_quadratic():
    from optbench.aenet_ml import _LBFGS

    start = np.array([0.0, 0.0, 0.0])
    solver = _LBFGS(start)
    x = np.array(solver.x, copy=True)
    for _ in range(30):
        loss = 0.5 * float(np.dot(x - 3.0, x - 3.0))
        grad = x - 3.0
        if np.linalg.norm(grad) < 1e-8:
            break
        x = solver.step(loss, grad)
        if solver.stopped:
            break
    assert np.allclose(x, [3.0, 3.0, 3.0], atol=1e-5)


def test_adam_respects_the_cap_and_the_harness_counter():
    model, weights = _model(forces=False)
    record = train_start(model, weights, "adam", cap=3, tolerance=1e-30)
    assert record["force_calls"] == 3
    assert record["success"] is False
    assert record["failure"] == "cap"


def test_lm_counts_the_jacobian_after_the_error_pass():
    model, weights = _model(forces=False)
    record = train_start(model, weights, "lm", cap=8, tolerance=1e-30)
    assert record["force_calls"] == 8
    assert record["failure"] == "cap"
