"""Optax mlopt kernel: fingerprints, the harness count, and the stop."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optbench.board import stated_full_n
from optbench.campaign import resolve_problem
from optbench.ml_bp import fingerprints_from_arrays, load_trajectory
from optbench.ml_optax import TrainSettings, _enable_x64, _forward_factory, _restore_x64, initial_parameters, train

ROOT = Path(__file__).resolve().parents[1]


def _settings(**overrides) -> TrainSettings:
    base = dict(
        method="adam",
        learning_rate=0.01,
        hidden=(4, 3),
        seed=20261018,
        energy_coefficient=1.0,
        force_coefficient=0.0,
        grad_tol=0.02,
        energy_rmse_tol=0.0001,
        force_rmse_tol=0.01,
        cap=8,
    )
    base.update(overrides)
    return TrainSettings(**base)


def _synthetic(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_images, natoms, n_fp = 4, 3, 2
    return {
        "n_images": n_images,
        "natoms": natoms,
        "elements": ["A"],
        "n_fp": {"A": n_fp},
        "fps": {"A": rng.normal(size=(n_images, natoms, n_fp))},
        "dgdr": {"A": rng.normal(size=(n_images, natoms, n_fp, natoms, 3)) * 0.01},
        "energies": rng.normal(size=(n_images,)),
        "forces": rng.normal(size=(n_images, natoms, 3)) * 0.01,
    }


def _g1(distance: float, eta: float, rs: float, rc: float) -> float:
    cutoff = 0.5 * (math.cos(math.pi * distance / rc) + 1.0)
    return math.exp(-eta * (distance - rs) ** 2 / rc**2) * cutoff


def test_initializations_are_the_full_set():
    platinum = resolve_problem(ROOT, "mlopt/emt-pt38")
    ethylene = resolve_problem(ROOT, "mlopt/C2H4")
    assert stated_full_n(ROOT, platinum) == 50
    assert stated_full_n(ROOT, ethylene) == 50


def test_g1_matches_the_analytic_dimer():
    distance = 2.0
    eta, rs, rc = 0.05, 0.0, 6.0
    positions = np.zeros((1, 2, 3))
    positions[0, 1, 0] = distance
    neighbors = np.asarray([[[1, -1], [0, -1]]], dtype=np.int32)
    prepared = fingerprints_from_arrays(
        positions,
        np.asarray([[20.0, 20.0, 20.0]]),
        np.asarray([False, False, False]),
        ["A", "A"],
        neighbors,
        ["A"],
        {"A": [{"kind": "G1", "neighbor": "A", "eta": eta, "rs": rs, "rc": rc}]},
    )
    expected = _g1(distance, eta, rs, rc)
    # The two atoms share one value, so the range is replaced and the scale is 1.
    assert prepared["fps"]["A"].shape == (1, 2, 1)
    assert prepared["fps"]["A"][0, 0, 0] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert prepared["fps"]["A"][0, 1, 0] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_g2_right_angle_uses_the_raw_jk_distance():
    rc = 6.0
    positions = np.zeros((1, 3, 3))
    positions[0, 1, 0] = 1.0
    positions[0, 2, 1] = 1.0
    neighbors = np.asarray([[[1, 2], [0, 2], [0, 1]]], dtype=np.int32)
    prepared = fingerprints_from_arrays(
        positions,
        np.asarray([[30.0, 30.0, 30.0]]),
        np.asarray([False, False, False]),
        ["A", "A", "A"],
        neighbors,
        ["A"],
        {
            "A": [
                {
                    "kind": "G2",
                    "neighbor1": "A",
                    "neighbor2": "A",
                    "eta": 0.0,
                    "zeta": 1.0,
                    "lam": 1.0,
                    "rc": rc,
                }
            ]
        },
    )
    def cutoff(distance: float) -> float:
        return 0.5 * (math.cos(math.pi * distance / rc) + 1.0)

    expected = cutoff(1.0) * cutoff(1.0) * cutoff(math.sqrt(2.0))
    scaled = float(prepared["fps"]["A"][0, 0, 0])
    raw = (scaled - float(prepared["intercept"]["A"][0])) / float(prepared["magnitude"]["A"][0])
    assert raw == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fingerprint_derivative_matches_a_central_difference():
    rng = np.random.default_rng(1)
    positions = rng.normal(size=(1, 3, 3))
    neighbors = np.asarray([[[1, 2], [0, 2], [0, 1]]], dtype=np.int32)
    lengths = np.asarray([[30.0, 30.0, 30.0]])
    pbc = np.asarray([False, False, False])
    specs = {
        "A": [
            {"kind": "G1", "neighbor": "A", "eta": 0.05, "rs": 0.0, "rc": 6.0},
            {"kind": "G1", "neighbor": "A", "eta": 0.2, "rs": 1.0, "rc": 6.0},
        ]
    }
    # Normalization is a property of the batch, so the difference is on the raw G.
    base = fingerprints_from_arrays(
        positions, lengths, pbc, ["A", "A", "A"], neighbors, ["A"], specs, normalize=False
    )
    step = 1e-6
    for atom in range(3):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[0, atom, axis] += step
            minus[0, atom, axis] -= step
            high = fingerprints_from_arrays(
                plus, lengths, pbc, ["A", "A", "A"], neighbors, ["A"], specs, normalize=False
            )
            low = fingerprints_from_arrays(
                minus, lengths, pbc, ["A", "A", "A"], neighbors, ["A"], specs, normalize=False
            )
            finite = (high["fps"]["A"] - low["fps"]["A"]) / (2.0 * step)
            got = base["dgdr"]["A"][0, :, :, atom, axis]
            assert finite[0] == pytest.approx(got, rel=1e-5, abs=1e-5)


def test_load_trajectory_pads_images_with_different_neighbor_widths(tmp_path: Path):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    def image(distance: float) -> Atoms:
        atoms = Atoms("H3", positions=[[0, 0, 0], [distance, 0, 0], [0, distance, 0]], cell=[40, 40, 40], pbc=True)
        atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((3, 3)))
        return atoms

    traj = tmp_path / "tiny.traj"
    write(traj, [image(1.2), image(12.0)])
    fp_file = tmp_path / "fpParas.dat"
    fp_file.write_text(
        "\n".join(
            [
                "dummy",
                "BP",
                "",
                "",
                "",
                "",
                "1 0",
                "G1 H H 0.05 0.0 6.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prepared = load_trajectory(traj, fp_file, cutoff=6.0)
    assert prepared["n_images"] == 2
    assert prepared["fps"]["H"].shape[0] == 2
    assert np.isfinite(prepared["fps"]["H"]).all()
    assert np.isfinite(prepared["dgdr"]["H"]).all()


def test_initial_weights_lie_in_the_open_upper_interval():
    previous = _enable_x64()
    try:
        params = initial_parameters(_synthetic(), _settings(), [0, 1])
    finally:
        _restore_x64(previous)
    import jax

    leaves = jax.tree.leaves(params)
    assert leaves
    for leaf in leaves:
        values = np.asarray(leaf)
        assert values.shape[0] == 2
        assert np.min(values) >= -1.0
        assert np.max(values) < 1.0
    assert not np.array_equal(np.asarray(leaves[0][0]), np.asarray(leaves[0][1]))


def test_adam_at_the_cap_is_a_cap_failure():
    records = train(_synthetic(), _settings(cap=3, grad_tol=0.0, energy_rmse_tol=-1.0), [0, 1])
    assert len(records) == 2
    for record in records:
        assert record["force_calls"] == 3
        assert record["epoch"] == 3
        assert record["success"] is False
        assert record["failure"] == "cap"
        assert record["counter"] == "harness"


def test_a_loose_gradient_stop_succeeds_on_the_initial_evaluation():
    records = train(_synthetic(), _settings(cap=5, grad_tol=1e9), [3])
    assert records[0]["success"] is True
    assert records[0]["failure"] is None
    assert records[0]["force_calls"] == 1
    assert records[0]["epoch"] == 1
    assert records[0]["converged"] is True


def test_zero_learning_rate_still_charges_every_evaluation():
    first = train(_synthetic(2), _settings(cap=1, learning_rate=0.0, grad_tol=0.0, energy_rmse_tol=-1.0), [0])
    later = train(_synthetic(2), _settings(cap=4, learning_rate=0.0, grad_tol=0.0, energy_rmse_tol=-1.0), [0])
    assert later[0]["force_calls"] == 4
    assert later[0]["failure"] == "cap"
    assert later[0]["loss"] == pytest.approx(first[0]["loss"], rel=0.0, abs=0.0)


def test_zero_force_coefficient_drops_the_force_term_from_the_gradient():
    prepared = _synthetic(3)
    shifted = dict(prepared)
    shifted["forces"] = np.asarray(prepared["forces"]) * 5.0 + 1.0
    previous = _enable_x64()
    try:
        params = initial_parameters(prepared, _settings(), [0])

        def grad(data, force_coefficient: float):
            _elements, _loss, evaluate, _metrics = _forward_factory(data, _settings(force_coefficient=force_coefficient))
            result = evaluate(params)
            import jax

            return jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), result[1])

        energy_only = grad(prepared, 0.0)
        energy_shifted = grad(shifted, 0.0)
        with_force = grad(prepared, 0.1)
        with_force_shifted = grad(shifted, 0.1)
    finally:
        _restore_x64(previous)
    import jax

    for left, right in zip(jax.tree.leaves(energy_only), jax.tree.leaves(energy_shifted), strict=True):
        assert left == pytest.approx(right, rel=0.0, abs=0.0)
    differ = False
    for left, right in zip(jax.tree.leaves(with_force), jax.tree.leaves(with_force_shifted), strict=True):
        if not np.allclose(left, right):
            differ = True
    assert differ


def test_lbfgs_charges_line_search_steps_and_verifies_the_accepted_point():
    records = train(
        _synthetic(4),
        _settings(method="lbfgs", learning_rate=None, cap=8, grad_tol=0.0, energy_rmse_tol=-1.0, force_rmse_tol=-1.0),
        [0],
    )
    record = records[0]
    # The initial point is an epoch. Each accepted outer step is another epoch
    # and one verification. Line-search trials make the harness count larger.
    assert record["failure"] == "cap"
    assert record["epoch"] > 1
    assert record["force_calls"] > record["epoch"]
    assert record["verify_calls"] == record["epoch"] - 1
    assert record["force_calls"] <= 8
