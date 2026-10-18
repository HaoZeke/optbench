"""CatLearn ML-NEB kernel match, counting, and template pin."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist, squareform

from optbench.campaign import list_templates, load_template
from optbench.catlearn_band import (
    _PIN,
    _VERSION,
    asv_document,
    gaussian_xx_gradients,
    gaussian_xxp_gradients,
)
from optbench.counter import HarnessCounter

ROOT = Path(__file__).resolve().parents[1]


def _gram(left: np.ndarray, right: np.ndarray | None, width: np.ndarray) -> np.ndarray:
    if right is None:
        gram = squareform(np.exp(-0.5 * pdist(left / width, metric="sqeuclidean")))
        np.fill_diagonal(gram, 1.0)
        return gram
    return np.exp(-0.5 * cdist(left / width, right / width, metric="sqeuclidean"))


def _xx_reference(features: np.ndarray, width: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """CatLearn's D>30 loop, copied from the 6a307f32 kernel."""
    count, dimension = features.shape
    cross = np.zeros((count, count * dimension))
    curvature = np.zeros((count * dimension, count * dimension))
    invsq = width ** (-2)
    identity = np.identity(dimension) * invsq
    for row in range(count):
        delta = invsq * (features - features[row])
        cross[:, dimension * row : dimension * (row + 1)] = (delta.T * gram[row]).T
        for col in range(row, count):
            block = (identity - np.outer(delta[col], delta[col])) * gram[row, col]
            curvature[row * dimension : (row + 1) * dimension, col * dimension : (col + 1) * dimension] = block
            if col != row:
                curvature[col * dimension : (col + 1) * dimension, row * dimension : (row + 1) * dimension] = block.T
    return np.block([[gram, cross], [cross.T, curvature]])


def _xxp_reference(left: np.ndarray, right: np.ndarray, width: np.ndarray, gram: np.ndarray) -> np.ndarray:
    count_right, dimension = right.shape
    cross = np.zeros((left.shape[0], count_right * dimension))
    invsq = width ** (-2)
    for row in range(left.shape[0]):
        scaled = invsq * (right - left[row]) * gram[row].reshape(count_right, 1)
        cross[row] = -scaled.reshape(1, count_right * dimension)
    return np.block([gram, cross])


def test_gradient_blocks_match_the_library_loop():
    rng = np.random.default_rng(20261018)
    cases = (
        (4, 8, np.linspace(0.3, 1.4, 8)),
        (5, 40, np.linspace(0.3, 1.4, 40)),
        (3, 40, np.array([0.75])),
    )
    for count, dimension, width in cases:
        left = rng.normal(size=(count, dimension))
        gram = _gram(left, None, width)
        got = gaussian_xx_gradients(left, width, gram)
        expect = _xx_reference(left, width, gram)
        assert got.shape == expect.shape
        assert np.array_equal(got, expect)
        right = rng.normal(size=(3, dimension))
        cross = _gram(left, right, width)
        got_cross = gaussian_xxp_gradients(left, right, width, cross)
        expect_cross = _xxp_reference(left, right, width, cross)
        assert np.array_equal(got_cross, expect_cross)


def test_installed_kernels_match_when_catlearn_is_present():
    pytest.importorskip("catlearn")
    import catlearn.regression.gpfunctions.kernels as kernels

    if kernels.gaussian_xx_gradients is gaussian_xx_gradients:
        pytest.skip("runtime shim already replaced the library kernels")
    rng = np.random.default_rng(7)
    left = rng.normal(size=(4, 36))
    width = np.linspace(0.5, 1.1, 36)
    gram = _gram(left, None, width)
    assert np.array_equal(gaussian_xx_gradients(left, width, gram), kernels.gaussian_xx_gradients(left, width, gram))


def test_diagonal_noise_matches_the_dense_covariance():
    pytest.importorskip("catlearn")
    import catlearn.regression.gpfunctions.covariance as covariance
    from optbench.catlearn_band import _covariance_with_diagonal_noise

    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(4, 12))
    kernel_list = [
        {
            "type": "gaussian",
            "width": [0.8],
            "dimension": "single",
            "bounds": ((0.1, 2.0),),
            "scaling": 1.5,
            "scaling_bounds": ((1.5, 1.5),),
        },
        {
            "type": "noise_multi",
            "hyperparameters": [0.005, 0.0005],
            "bounds": ((0.001, 0.005), (0.0005, 0.002)),
        },
    ]
    original = covariance.get_covariance
    if getattr(original, "__wrapped_noise__", False):
        pytest.skip("covariance shim is already installed")
    fast = _covariance_with_diagonal_noise(original)
    dense = original(kernel_list, False, matrix, regularization=0.0, eval_gradients=True)
    quick = fast(kernel_list, False, matrix, regularization=0.0, eval_gradients=True)
    assert np.array_equal(dense, quick)


def test_templates_use_the_registry_pin():
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("catlearn/")]
    assert len(found) == 8
    problems = {row["problem"] for row in found}
    assert problems == {
        "double-ended-saddle-search/lj38",
        "double-ended-saddle-search/lj38_2ts_separation",
        "double-ended-saddle-search/pt-island",
        "neb/Pt-heptamer",
    }
    for row in found:
        loaded = load_template(ROOT, row["id"])
        assert loaded["engine"] == "catlearn-mlneb"
        assert loaded["commit"] == _PIN
        assert loaded["code_version"] == _VERSION
        assert loaded["mlneb"]["steps"] == 500
        assert loaded["mlneb"]["acquisition"] == "acq_5"
        if loaded["tuning"] == "tuned":
            assert loaded["mlneb"]["interpolation"] == "idpp"
        else:
            assert loaded["mlneb"]["interpolation"] == "linear"


def test_spec_calculator_counts_one_call_and_zeros_fixed_forces():
    pytest.importorskip("ase")
    from optbench.catlearn_band import _freeze_endpoint, _spec_calculator
    from optbench.pes import evaluate

    free = np.array([True, False], dtype=bool)
    box = np.array([10.0, 10.0, 10.0])
    coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    counter = HarnessCounter(cache=False, cap=10)
    atoms = __import__("ase", fromlist=["Atoms"]).Atoms(symbols=["Pt", "Pt"], positions=coords, cell=box, pbc=True)
    _freeze_endpoint(atoms, "morse_pt", box, free, counter)
    assert counter.calls == 1
    energy, gradient = evaluate("morse_pt", coords, box)
    assert atoms.get_potential_energy() == pytest.approx(energy)
    assert counter.calls == 1
    forces = atoms.get_forces()
    assert counter.calls == 1
    assert forces[1] == pytest.approx(0.0)
    assert forces[0] == pytest.approx(-gradient[0])
    live = _spec_calculator("morse_pt", box, free)
    from optbench.counter import attach_ase_counter

    attach_ase_counter(live, cap=10, counter=counter)
    other = atoms.copy()
    other.calc = live
    other.positions[0, 0] = 0.2
    other.get_potential_energy()
    other.get_forces()
    assert counter.calls == 2
    assert other.get_forces()[1] == pytest.approx(0.0)


def test_asv_document_round_trips():
    asv_results = pytest.importorskip("asv.results")
    machine = {
        "arch": "x86_64",
        "cpu": "test",
        "machine": "cursor",
        "num_cpu": "8",
        "os": "Linux",
        "ram": "1",
    }
    document = asv_document(
        "abc12345abc12345abc12345abc12345abc12345",
        [
            {
                "name": "bench_catlearn.Lj38Default",
                "starts": [
                    {
                        "seed": "0",
                        "start": "pair_00",
                        "force_calls": 12,
                        "success": False,
                        "failure": "divergence",
                        "wall_seconds": 1.5,
                    }
                ],
            }
        ],
        machine,
        1_700_000_000_000,
    )
    root = Path(__file__).resolve().parent
    del root
    text = json.dumps(document)
    loaded = json.loads(text)
    assert loaded["version"] == 2
    key = "bench_catlearn.Lj38Default.track_force_calls"
    assert loaded["results"][key][0] == [12.0]
    assert ast.literal_eval(loaded["results"][key][1][1][0]) == "pair_00"
    assert "Results" in dir(asv_results)
