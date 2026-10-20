"""TSASE Morse force for the archived bulk starts.

TSASE ``FORCE`` shifts the pair energy to zero at the cutoff and folds
orthorhombic minimum images with Fortran ``DNINT``.
"""

from __future__ import annotations

import numpy as np

_DTYPE = np.float64


def dnint(values: np.ndarray) -> np.ndarray:
    """Fortran ``DNINT``: round half away from zero."""
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def morse_force(
    positions: np.ndarray,
    cell: np.ndarray,
    u0: float,
    alpha: float,
    r0: float,
    rc: float,
) -> tuple[float, np.ndarray]:
    """TSASE Morse energy and forces for one orthorhombic cell.

    The pair loop is the upper triangle. A pair contributes when ``r^2 < rc^2``.
    Forces are ``2 * alpha * u0 * (x^2 - x) * rhat`` with ``x = exp(alpha*(r0-r))``.
    """
    pos = np.asarray(positions, dtype=_DTYPE)
    count = pos.shape[0]
    left, right = np.triu_indices(count, k=1)
    delta = pos[left] - pos[right]
    box = np.array([cell[0, 0], cell[1, 1], cell[2, 2]], dtype=_DTYPE)
    delta = delta - box * dnint(delta / box)
    r2 = np.einsum("ij,ij->i", delta, delta)
    rc2 = _DTYPE(rc) * _DTYPE(rc)
    mask = r2 < rc2
    energy = _DTYPE(0.0)
    forces = np.zeros_like(pos)
    if not np.any(mask):
        return float(energy), forces
    delta = delta[mask]
    r2 = r2[mask]
    left = left[mask]
    right = right[mask]
    distance = np.sqrt(r2)
    uu = np.exp(_DTYPE(alpha) * (_DTYPE(r0) - distance))
    duu = uu * uu - uu
    uu_shift = np.exp(_DTYPE(alpha) * (_DTYPE(r0) - _DTYPE(rc)))
    uu_shift = uu_shift * uu_shift - _DTYPE(2.0) * uu_shift
    hat = delta / distance[:, None]
    contrib = duu[:, None] * hat
    np.add.at(forces, left, contrib)
    np.add.at(forces, right, -contrib)
    forces *= _DTYPE(2.0) * _DTYPE(alpha) * _DTYPE(u0)
    energy = float(_DTYPE(u0) * np.sum(duu - uu - uu_shift))
    return energy, forces
