"""Pair potentials for a harness-counted rgmin campaign.

Lennard-Jones 38 is the cluster in ``problems/minimization/lj38.toml``:
epsilon 1, sigma 1, cutoff 40, no shift, no periodic images. Morse bulk
is the shifted LAMMPS Morse that ``archive/entries/minimization/morse-bulk``
states matches ``morse_pt``: D0 0.7102, alpha 1.6047, r0 2.897, cutoff 7,
energy shifted to zero at the cutoff, orthorhombic minimum image.

The gradient is dE/dx. Its Euclidean norm is the force norm.
Binary Lennard-Jones uses an arithmetic sigma for the unlike pair and the
per-pair epsilons the caller passes. A missing cutoff sums every pair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

LJ_EPSILON = 1.0
LJ_SIGMA = 1.0
LJ_CUTOFF = 40.0
# A pair at distance zero has no 12-6 value. This finite stand-in keeps the
# pair in the energy instead of dropping it.
COINCIDENT_PAIR_ENERGY = 1.0e6

MORSE_D0 = 0.7102
MORSE_ALPHA = 1.6047
MORSE_R0 = 2.897
MORSE_CUTOFF = 7.0


def _as_positions(coords: Sequence[float] | np.ndarray) -> np.ndarray:
    pos = np.asarray(coords, dtype=np.float64)
    if pos.ndim == 1:
        if pos.size % 3 != 0:
            raise ValueError("coordinates must be a multiple of 3")
        pos = pos.reshape(-1, 3)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3) or (3n,)")
    return pos


def _pairs(pos: np.ndarray, cutoff: float, box: np.ndarray | None):
    n = pos.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    diff = pos[iu] - pos[ju]
    if box is not None:
        lengths = np.asarray(box, dtype=np.float64)
        if lengths.shape != (3,):
            raise ValueError("box must be three orthorhombic lengths")
        diff -= lengths * np.round(diff / lengths)
    r2 = np.einsum("ij,ij->i", diff, diff)
    mask = (r2 > 0.0) & (r2 < cutoff * cutoff)
    return iu[mask], ju[mask], diff[mask], r2[mask]


def _accumulate(n: int, iu: np.ndarray, ju: np.ndarray, diff: np.ndarray, slope: np.ndarray):
    """Scatter pair contributions. ``slope`` is (1/r) dV/dr."""
    grad = np.zeros((n, 3), dtype=np.float64)
    piece = slope[:, None] * diff
    np.add.at(grad, iu, piece)
    np.add.at(grad, ju, -piece)
    return grad


def lennard_jones(coords: Sequence[float] | np.ndarray) -> tuple[float, np.ndarray]:
    """Truncated 12-6 Lennard-Jones. No shift and no periodic images.

    A pair at distance zero contributes ``COINCIDENT_PAIR_ENERGY`` and no
    force. Dropping that pair would omit the repulsion.
    """
    pos = _as_positions(coords)
    n = pos.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    diff = pos[iu] - pos[ju]
    r2 = np.einsum("ij,ij->i", diff, diff)
    coincident = int(np.count_nonzero(r2 == 0.0))
    keep = (r2 > 0.0) & (r2 < LJ_CUTOFF * LJ_CUTOFF)
    iu, ju, diff, r2 = iu[keep], ju[keep], diff[keep], r2[keep]
    penalty = COINCIDENT_PAIR_ENERGY * coincident
    if iu.size == 0:
        return float(penalty), np.zeros(pos.size, dtype=np.float64)
    inv_r2 = 1.0 / r2
    sr2 = (LJ_SIGMA * LJ_SIGMA) * inv_r2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = 4.0 * LJ_EPSILON * np.sum(sr12 - sr6) + penalty
    # (1/r) dV/dr = 24 eps (sr6 - 2 sr12) / r^2
    slope = 24.0 * LJ_EPSILON * (sr6 - 2.0 * sr12) * inv_r2
    grad = _accumulate(pos.shape[0], iu, ju, diff, slope)
    return float(energy), grad.reshape(-1)


def _species_index(species: Sequence[int] | Sequence[str] | np.ndarray, count: int) -> np.ndarray:
    labels = np.asarray(species)
    if labels.shape != (count,):
        raise ValueError("species must name one label per atom")
    if labels.dtype.kind in {"U", "S", "O"}:
        mapped = []
        for label in labels.tolist():
            text = str(label).upper()
            if text in {"A", "LA", "0"}:
                mapped.append(0)
            elif text in {"B", "LB", "1"}:
                mapped.append(1)
            else:
                raise ValueError(f"unknown species {label}")
        return np.asarray(mapped, dtype=np.int64)
    values = labels.astype(np.int64)
    if np.any((values != 0) & (values != 1)):
        raise ValueError("species integers are 0 for A and 1 for B")
    return values


def binary_lennard_jones(
    coords: Sequence[float] | np.ndarray,
    species: Sequence[int] | Sequence[str] | np.ndarray,
    *,
    sigma_aa: float = 1.0,
    sigma_bb: float = 1.3,
    epsilon_aa: float = 1.0,
    epsilon_bb: float = 1.0,
    epsilon_ab: float = 1.0,
    cutoff: float | None = None,
) -> tuple[float, np.ndarray]:
    """Binary 12-6 Lennard-Jones. Unlike sigma is the arithmetic mean.

    Species 0 is A and species 1 is B. ``epsilon_ab`` is the unlike well
    depth. AA and BB pairs use their own epsilon. ``cutoff`` None keeps
    every pair. There is no shift and no periodic image.
    """
    pos = _as_positions(coords)
    labels = _species_index(species, pos.shape[0])
    limit = math.inf if cutoff is None else float(cutoff)
    iu, ju, diff, r2 = _pairs(pos, limit, None)
    if iu.size == 0:
        return 0.0, np.zeros(pos.size, dtype=np.float64)
    sigma = np.where(labels == 0, float(sigma_aa), float(sigma_bb))
    sigma_ij = 0.5 * (sigma[iu] + sigma[ju])
    same = labels[iu] == labels[ju]
    epsilon_ij = np.where(
        same,
        np.where(labels[iu] == 0, float(epsilon_aa), float(epsilon_bb)),
        float(epsilon_ab),
    )
    inv_r2 = 1.0 / r2
    sr2 = (sigma_ij * sigma_ij) * inv_r2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = 4.0 * np.sum(epsilon_ij * (sr12 - sr6))
    slope = 24.0 * epsilon_ij * (sr6 - 2.0 * sr12) * inv_r2
    grad = _accumulate(pos.shape[0], iu, ju, diff, slope)
    return float(energy), grad.reshape(-1)


def morse_bulk(coords: Sequence[float] | np.ndarray, box: Sequence[float]) -> tuple[float, np.ndarray]:
    """Shifted Morse on an orthorhombic cell. The shift does not change the force."""
    pos = _as_positions(coords)
    lengths = np.asarray(box, dtype=np.float64)
    iu, ju, diff, r2 = _pairs(pos, MORSE_CUTOFF, lengths)
    if iu.size == 0:
        return 0.0, np.zeros(pos.size, dtype=np.float64)
    r = np.sqrt(r2)
    x = np.exp(-MORSE_ALPHA * (r - MORSE_R0))
    raw = MORSE_D0 * (x * x - 2.0 * x)
    rc = MORSE_CUTOFF
    xc = np.exp(-MORSE_ALPHA * (rc - MORSE_R0))
    shift = MORSE_D0 * (xc * xc - 2.0 * xc)
    energy = float(np.sum(raw - shift))
    # dV/dr = 2 D0 alpha x (1 - x); slope = (1/r) dV/dr
    slope = (2.0 * MORSE_D0 * MORSE_ALPHA * x * (1.0 - x)) / r
    grad = _accumulate(pos.shape[0], iu, ju, diff, slope)
    return energy, grad.reshape(-1)


def _hessian_from_pairs(
    n: int,
    iu: np.ndarray,
    ju: np.ndarray,
    diff: np.ndarray,
    r2: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Cartesian Hessian from pair derivatives dV/dr and d2V/dr2."""
    dim = 3 * n
    hessian = np.zeros((dim, dim), dtype=np.float64)
    if iu.size == 0:
        return hessian
    r = np.sqrt(r2)
    s = first / r
    # d(g_a)/d(x_b) = -((V'' - s) uhat uhat^T + s I), u = x_a - x_b.
    outer_scale = (second - s) / r2
    for a in range(3):
        for b in range(3):
            block = -outer_scale * diff[:, a] * diff[:, b]
            if a == b:
                block = block - s
            np.add.at(hessian, (iu * 3 + a, ju * 3 + b), block)
            np.add.at(hessian, (ju * 3 + b, iu * 3 + a), block)
            np.add.at(hessian, (iu * 3 + a, iu * 3 + b), -block)
            np.add.at(hessian, (ju * 3 + a, ju * 3 + b), -block)
    return 0.5 * (hessian + hessian.T)


def lennard_jones_hessian(coords: Sequence[float] | np.ndarray) -> np.ndarray:
    pos = _as_positions(coords)
    iu, ju, diff, r2 = _pairs(pos, LJ_CUTOFF, None)
    inv_r2 = 1.0 / r2 if r2.size else r2
    sr2 = (LJ_SIGMA * LJ_SIGMA) * inv_r2 if r2.size else r2
    sr6 = sr2 * sr2 * sr2 if r2.size else r2
    sr12 = sr6 * sr6 if r2.size else r2
    r = np.sqrt(r2) if r2.size else r2
    # V' = 24 eps (-2 r^{-13} + r^{-7}) with sigma = 1
    first = 24.0 * LJ_EPSILON * (-2.0 * sr12 + sr6) / r if r2.size else r2
    # V'' = 24 eps (26 r^{-14} - 7 r^{-8})
    second = 24.0 * LJ_EPSILON * (26.0 * sr12 - 7.0 * sr6) * inv_r2 if r2.size else r2
    return _hessian_from_pairs(pos.shape[0], iu, ju, diff, r2, first, second)


def morse_bulk_hessian(coords: Sequence[float] | np.ndarray, box: Sequence[float]) -> np.ndarray:
    pos = _as_positions(coords)
    lengths = np.asarray(box, dtype=np.float64)
    iu, ju, diff, r2 = _pairs(pos, MORSE_CUTOFF, lengths)
    r = np.sqrt(r2) if r2.size else r2
    x = np.exp(-MORSE_ALPHA * (r - MORSE_R0)) if r2.size else r2
    first = 2.0 * MORSE_D0 * MORSE_ALPHA * x * (1.0 - x) if r2.size else r2
    second = 2.0 * MORSE_D0 * MORSE_ALPHA * MORSE_ALPHA * x * (2.0 * x - 1.0) if r2.size else r2
    return _hessian_from_pairs(pos.shape[0], iu, ju, diff, r2, first, second)


def read_con(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return positions (n, 3) and orthorhombic box lengths from an eOn ``.con``."""
    lines = text.splitlines()
    box = None
    for i, line in enumerate(lines[:-1]):
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            lengths = [float(part) for part in parts]
            angles = [float(part) for part in lines[i + 1].split()]
        except ValueError:
            continue
        if len(angles) == 3 and all(abs(angle - 90.0) < 1e-4 for angle in angles):
            box = np.asarray(lengths, dtype=np.float64)
            break
    if box is None:
        raise ValueError("con file has no orthorhombic box")
    start = next(i for i, line in enumerate(lines) if line.startswith("Coordinates of Component"))
    rows = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            break
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            break
        if len(parts) >= 4 and parts[3] != "0":
            raise ValueError("frozen atoms are not a minimization coordinate in this runner")
    if not rows:
        raise ValueError("con file has no coordinates")
    return np.asarray(rows, dtype=np.float64), box
