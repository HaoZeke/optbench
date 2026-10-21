"""Lennard-Jones pair potential matching ASE's free-cluster calculator.

``ase.calculators.lj.LennardJones`` with ``smooth`` false shifts each pair
by its value at the cutoff and leaves the force as the derivative of the
unshifted 12-6 term inside that cutoff. Pairs at or beyond the cutoff are
omitted. The cell is ignored, which is the free cluster used by the
global-optimization geometries. The harness counts one ``calculate`` entry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class LennardJonesASE:
    """Energy and forces for the ASE Lennard-Jones parameters.

    The object is an ASE calculator once :meth:`build` is called. The
    numpy kernel is separate so a test can compare it without constructing
    an atoms object.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, rc: float = 40.0):
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.rc = float(rc)
        if self.rc <= 0.0:
            raise ValueError("cutoff must be positive")
        ratio = self.sigma / self.rc
        attractive = ratio**6
        self.e0 = 4.0 * self.epsilon * attractive * (attractive - 1.0)

    def energy_forces(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        """Pair sum. ``positions`` has shape ``(natoms, 3)``."""
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must be natoms by 3")
        n = pos.shape[0]
        forces = np.zeros((n, 3), dtype=float)
        if n < 2:
            return 0.0, forces
        # delta[i, j] points from i toward j. ASE's neighbor list is bothways,
        # and each atom sums those pair forces. The energy is half of that sum.
        delta = pos[None, :, :] - pos[:, None, :]
        r2 = np.einsum("ijk,ijk->ij", delta, delta)
        rc2 = self.rc * self.rc
        mask = (r2 > 0.0) & (r2 <= rc2)
        ratio = np.zeros((n, n), dtype=float)
        np.divide(self.sigma * self.sigma, r2, out=ratio, where=mask)
        c6 = np.zeros((n, n), dtype=float)
        np.power(ratio, 3, out=c6, where=mask)
        c12 = c6 * c6
        pair = np.zeros((n, n), dtype=float)
        pair[mask] = 4.0 * self.epsilon * (c12[mask] - c6[mask]) - self.e0
        scalar = np.zeros((n, n), dtype=float)
        np.divide(
            -24.0 * self.epsilon * (2.0 * c12 - c6),
            r2,
            out=scalar,
            where=mask,
        )
        forces = (scalar[:, :, None] * delta).sum(axis=1)
        return 0.5 * float(pair.sum()), forces

    def build(self):
        """Return an ASE calculator with this kernel."""
        from ase.calculators.calculator import Calculator
        from ase.calculators.calculator import all_changes

        kernel = self

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                energy, forces = kernel.energy_forces(self.atoms.positions)
                self.results = {"energy": energy, "forces": forces}

        return _Calc()


def lj_energy_forces(
    positions: Sequence[Sequence[float]],
    *,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    rc: float = 40.0,
) -> tuple[float, np.ndarray]:
    """Energy and ``(natoms, 3)`` forces for one free cluster."""
    return LennardJonesASE(epsilon, sigma, rc).energy_forces(np.asarray(positions, dtype=float))
