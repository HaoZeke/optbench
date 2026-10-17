import math

import numpy as np
import pytest

from optbench.stationary import (
    _rigid_basis,
    curvature_tolerance,
    jacobi_eigenvalues,
    judge_stationary,
    kabsch_rmsd,
    negative_index,
    numerical_hessian,
    vibrational_eigenvalues,
)


def test_symmetric_eigenvalues_match_the_matrix():
    matrix = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]
    values, vectors = jacobi_eigenvalues(matrix)
    assert values == sorted(values)
    for column, value in enumerate(values):
        vector = [vectors[row][column] for row in range(3)]
        product = [
            sum(matrix[row][col] * vector[col] for col in range(3))
            for row in range(3)
        ]
        for got, expected in zip(product, [value * component for component in vector], strict=True):
            assert got == pytest.approx(expected, abs=1e-8)


def test_known_two_by_two_spectrum():
    values, _vectors = jacobi_eigenvalues([[1.0, 2.0], [2.0, 1.0]])
    assert values == pytest.approx([-1.0, 3.0])
    assert negative_index(values, tol=1e-8) == 1


def test_kabsch_removes_a_proper_rotation_and_a_translation():
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.7]]
    moved = [[-y + 3.0, x - 4.0, z + 5.0] for x, y, z in points]
    assert kabsch_rmsd(moved, points) == pytest.approx(0.0, abs=1e-8)


def test_kabsch_does_not_match_an_enantiomer():
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    mirrored = [[x, y, -z] for x, y, z in points]
    assert kabsch_rmsd(mirrored, points) > 0.1


def test_numerical_hessian_of_a_saddle():
    def gradient(point):
        x, y = point
        return (-2.0 * x, 2.0 * y)

    hessian = numerical_hessian(gradient, [0.2, -0.4], step=1e-5)
    values, _vectors = jacobi_eigenvalues(hessian)
    assert values == pytest.approx([-2.0, 2.0], abs=1e-6)
    judged = judge_stationary(
        kind="saddle",
        eigenvalues=values,
        energy=0.0,
        coords=[[0.0, 0.0, 0.0]],
        reference_energy=0.0,
        reference_coords=[[0.0, 0.0, 0.0]],
        energy_tol=1e-8,
        rmsd_tol=1e-6,
    )
    assert judged["ok"] is True
    assert judged["hessian_index"] == 1


def test_curvature_tolerance_follows_the_force_stop_and_the_hessian_norm():
    assert curvature_tolerance(0.01, 12.0) == pytest.approx(0.01)
    assert curvature_tolerance(1e-8, 1.0) == pytest.approx(1e-6)
    assert curvature_tolerance(1e-12, 1.0e6) == pytest.approx(0.01)


def test_rigid_leakage_is_not_a_vibrational_mode_and_a_real_mode_remains():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.1]])
    basis = _rigid_basis(coords, rotations=True)
    rotation = basis[:, 3]
    leakage = -1.0e-3 * np.outer(rotation, rotation)
    raw = np.linalg.eigvalsh(leakage)
    assert negative_index(raw, 1e-6) == 1
    vibrational = vibrational_eigenvalues(leakage, coords, rotations=True)
    assert negative_index(vibrational, curvature_tolerance(0.01, np.linalg.norm(leakage))) == 0

    left, singular, _right = np.linalg.svd(basis, full_matrices=True)
    mode = left[:, int(np.sum(singular > 1e-8))]
    hessian = leakage - 2.0 * np.outer(mode, mode)
    vibrational = vibrational_eigenvalues(hessian, coords, rotations=True)
    assert min(vibrational) == pytest.approx(-2.0, abs=1e-8)
    assert negative_index(vibrational, curvature_tolerance(0.01, np.linalg.norm(hessian))) == 1


def test_a_minimum_is_not_an_index_1_saddle():
    values, _vectors = jacobi_eigenvalues([[2.0, 0.0], [0.0, 20.0]])
    judged = judge_stationary(
        kind="saddle",
        eigenvalues=values,
        energy=0.0,
        coords=[[1.0, -2.0, 0.0]],
        reference_energy=0.0,
        reference_coords=[[0.0, 0.0, 0.0]],
        energy_tol=1e-8,
        rmsd_tol=1e-6,
    )
    assert judged["ok"] is False
    assert "hessian_index" in judged["reasons"]
    assert "rmsd" in judged["reasons"]
    assert judged["rmsd"] == pytest.approx(math.sqrt(5.0))
