"""Morse bulk on a general cell for variable-cell SQNM.

The fixed-cell campaign keeps ``optbench.potentials.morse_bulk``. That
kernel is orthorhombic. ``periodic_sqnm`` updates a full lattice, so the
variable-cell rows need the same pair potential on that lattice.

``dE/dA`` is the derivative at fixed fractional coordinates. Python
``periodic_sqnm`` multiplies the supplied lattice derivative by minus
one before the SQNM step. ``sqnm_lattice_derivative`` is the array that
cancels that minus. The ASE wrapper builds its derivative as
``-det(A) stress inv(A).T``, and ``cell_stress`` is the stress that
makes that expression equal ``sqnm_lattice_derivative``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from optbench.potentials import (
    MORSE_ALPHA,
    MORSE_CUTOFF,
    MORSE_D0,
    MORSE_R0,
    _accumulate,
    _as_positions,
    _hessian_from_pairs,
)


def as_cell(cell: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a 3 by 3 matrix of lattice vectors stored columnwise."""
    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape == (3,):
        return np.diag(matrix)
    if matrix.shape != (3, 3):
        raise ValueError("cell must be three lengths or a 3 by 3 matrix")
    return matrix


def orthorhombic_lengths(cell: np.ndarray) -> np.ndarray:
    """Diagonal lengths. The fixed-cell Morse kernel takes this vector."""
    matrix = as_cell(cell)
    if np.max(np.abs(matrix - np.diag(np.diag(matrix)))) > 0.0:
        raise ValueError("fixed-cell Morse is orthorhombic")
    return np.diag(matrix).copy()


def _pairs(pos: np.ndarray, cell: np.ndarray):
    inv = np.linalg.inv(cell)
    frac = pos @ inv.T
    iu, ju = np.triu_indices(pos.shape[0], k=1)
    fractional = frac[iu] - frac[ju]
    fractional -= np.round(fractional)
    disp = fractional @ cell.T
    r2 = np.einsum("ij,ij->i", disp, disp)
    mask = (r2 > 0.0) & (r2 < MORSE_CUTOFF * MORSE_CUTOFF)
    return iu[mask], ju[mask], fractional[mask], disp[mask], r2[mask]


def _morse_terms(r2: np.ndarray):
    r = np.sqrt(r2)
    expo = np.exp(-MORSE_ALPHA * (r - MORSE_R0))
    raw = MORSE_D0 * (expo * expo - 2.0 * expo)
    cutoff = np.exp(-MORSE_ALPHA * (MORSE_CUTOFF - MORSE_R0))
    shift = MORSE_D0 * (cutoff * cutoff - 2.0 * cutoff)
    first = 2.0 * MORSE_D0 * MORSE_ALPHA * expo * (1.0 - expo)
    second = 2.0 * MORSE_D0 * MORSE_ALPHA * MORSE_ALPHA * expo * (2.0 * expo - 1.0)
    slope = first / r
    return raw - shift, slope, first, second


def morse_cell(coords: Sequence[float] | np.ndarray, cell: Sequence[float] | np.ndarray):
    """Energy, Cartesian gradient, and ``dE/dA`` at fixed fractional coordinates."""
    pos = _as_positions(coords)
    matrix = as_cell(cell)
    n = pos.shape[0]
    empty_grad = np.zeros(pos.size, dtype=np.float64)
    empty_cell = np.zeros((3, 3), dtype=np.float64)
    if abs(float(np.linalg.det(matrix))) < 1e-18:
        raise ValueError("cell determinant vanished")
    iu, ju, fractional, disp, r2 = _pairs(pos, matrix)
    if iu.size == 0:
        return 0.0, empty_grad, empty_cell
    energy, slope, _first, _second = _morse_terms(r2)
    grad = _accumulate(n, iu, ju, disp, slope)
    # (dE/dA)_ab = sum slope * d_a * s_b
    deriv = np.einsum("i,ia,ib->ab", slope, disp, fractional)
    return float(np.sum(energy)), grad.reshape(-1), deriv


def morse_cell_hessian(coords: Sequence[float] | np.ndarray, cell: Sequence[float] | np.ndarray) -> np.ndarray:
    """Atomic Cartesian Hessian at fixed cell."""
    pos = _as_positions(coords)
    matrix = as_cell(cell)
    iu, ju, _fractional, disp, r2 = _pairs(pos, matrix)
    if iu.size == 0:
        return np.zeros((pos.size, pos.size), dtype=np.float64)
    _energy, _slope, first, second = _morse_terms(r2)
    return _hessian_from_pairs(pos.shape[0], iu, ju, disp, r2, first, second)


def sqnm_lattice_derivative(deriv: np.ndarray) -> np.ndarray:
    """Lattice derivative consumed by Python ``periodic_sqnm``.

    The function multiplies this array by minus one and then steps
    against the product. Supplying ``-dE/dA`` makes that product the
    true derivative, so the step descends.
    """
    return -np.asarray(deriv, dtype=np.float64)


def cell_stress(deriv: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """ASE stress whose wrapper formula equals ``sqnm_lattice_derivative``.

    ``aseOptimizer`` sets ``deralat = -det(A) stress inv(A).T``.
    """
    matrix = as_cell(cell)
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-18:
        raise ValueError("cell determinant vanished")
    return np.asarray(deriv, dtype=np.float64) @ matrix.T / det
