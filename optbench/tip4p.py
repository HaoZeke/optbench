"""Rigid TIP4P water, in the GMIN angle-axis chart.

Constants are ``DEFTIP4`` in GMIN ``source/newtip.f90`` from the group source
tarball https://www-wales.ch.cam.ac.uk/svn.tar.bz2 (linked from
https://www-wales.ch.cam.ac.uk/GMIN/). C6 is 2552.24 kJ/mol Angstrom**6,
C12 is 2510.4D3 kJ/mol Angstrom**12, and the Coulomb prefactor HTOKJ is
1389.354848. Charges are 0, 0.52, 0.52, -1.04 on oxygen, the two hydrogens,
and the M site. The geometry is rOH 0.9572 Angstrom, angle 104.52 degrees,
and the M site 0.15 Angstrom from oxygen along the H-O-H bisector. Site
vectors are then shifted to the centre of mass with masses 16, 1, 1, 0.

The rotation is the angle-axis map in GMIN ``source/tip.f90``: with alpha the
length of the angle-axis vector omega,

    c2 = cos(alpha), c3 = (cos(alpha) - 1) / alpha**2, s = sin(alpha) / alpha
    r = c2 * p - c3 * (omega · p) * omega - s * (omega × p)

A coordinate block is cx, cy, cz, lx, ly, lz for each molecule. The centre is
the centre of mass. The gradient is dE/d(centre, angle-axis). Lennard-Jones
acts between oxygens. Coulomb acts between the charged sites.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

TIP4P_C6 = 2552.24
TIP4P_C12 = 2510.4e3
TIP4P_COULOMB = 1389.354848
TIP4P_CHARGES = np.array([0.0, 0.52, 0.52, -1.04], dtype=np.float64)
TIP4P_ROH = 0.9572
TIP4P_ANGLE_DEG = 104.52
TIP4P_ROM = 0.15
TIP4P_MASSES = np.array([16.0, 1.0, 1.0, 0.0], dtype=np.float64)

_CHARGED = (1, 2, 3)


def reference_sites() -> np.ndarray:
    """O, H, H, M in the centre-of-mass frame. Shape (4, 3)."""
    theta = math.radians(TIP4P_ANGLE_DEG)
    half = 0.5 * theta
    oxygen = np.zeros(3, dtype=np.float64)
    hydrogen_1 = np.array([0.0, math.sin(half) * TIP4P_ROH, math.cos(half) * TIP4P_ROH])
    hydrogen_2 = np.array([0.0, -math.sin(half) * TIP4P_ROH, math.cos(half) * TIP4P_ROH])
    m_site = np.array([0.0, 0.0, TIP4P_ROM])
    sites = np.stack([oxygen, hydrogen_1, hydrogen_2, m_site])
    centre = np.sum(sites * TIP4P_MASSES[:, None], axis=0) / float(np.sum(TIP4P_MASSES))
    return sites - centre


def _rotate(omega: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GMIN rotation of ``points`` (m, 3).

    The derivative array has shape (m, 3, 3) and ``dr[:, :, k]`` is d r / d omega_k.
    """
    omega = np.asarray(omega, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    alpha2 = float(np.dot(omega, omega))
    alpha = math.sqrt(alpha2)
    small = alpha < 1e-8
    if small:
        cosine = 1.0
        c3 = -0.5
        sine_over = 1.0
        d_cosine = d_c3 = d_sine = 0.0
        d_alpha = np.zeros(3, dtype=np.float64)
    else:
        cosine = math.cos(alpha)
        sine = math.sin(alpha)
        c3 = (cosine - 1.0) / alpha2
        sine_over = sine / alpha
        d_cosine = -sine
        d_c3 = (-sine * alpha2 - (cosine - 1.0) * 2.0 * alpha) / (alpha2 * alpha2)
        d_sine = (cosine * alpha - sine) / alpha2
        d_alpha = omega / alpha
    projection = points @ omega
    crossed = np.cross(omega, points)
    rotated = cosine * points - c3 * projection[:, None] * omega - sine_over * crossed
    derivative = np.zeros((points.shape[0], 3, 3), dtype=np.float64)
    for axis in range(3):
        unit = np.zeros(3, dtype=np.float64)
        unit[axis] = 1.0
        scale_cosine = 0.0 if small else d_cosine * d_alpha[axis]
        scale_c3 = 0.0 if small else d_c3 * d_alpha[axis]
        scale_sine = 0.0 if small else d_sine * d_alpha[axis]
        term = scale_cosine * points
        term = term - scale_c3 * projection[:, None] * omega
        term = term - c3 * points[:, axis : axis + 1] * omega
        term = term - c3 * projection[:, None] * unit
        term = term - scale_sine * crossed
        term = term - sine_over * np.cross(unit, points)
        derivative[:, :, axis] = term
    return rotated, derivative


def _split_rigid(coords: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 6:
        block = values
    elif values.ndim == 1 and values.size % 6 == 0:
        block = values.reshape(-1, 6)
    else:
        raise ValueError("rigid TIP4P coordinates have shape (n, 6) or (6n,)")
    return block[:, :3].copy(), block[:, 3:].copy()


def site_energy_forces(placed: np.ndarray) -> tuple[float, np.ndarray]:
    """Energy and dE/d(site) for placed molecules of shape (n, 4, 3)."""
    placed = np.asarray(placed, dtype=np.float64)
    count = placed.shape[0]
    gradient = np.zeros_like(placed)
    energy = 0.0
    for left in range(count - 1):
        for right in range(left + 1, count):
            delta = placed[left, 0] - placed[right, 0]
            distance2 = float(np.dot(delta, delta))
            inverse = 1.0 / distance2
            r6 = inverse * inverse * inverse
            r12 = r6 * r6
            energy += TIP4P_C12 * r12 - TIP4P_C6 * r6
            slope = (-12.0 * TIP4P_C12 * r12 + 6.0 * TIP4P_C6 * r6) * inverse
            force = slope * delta
            gradient[left, 0] += force
            gradient[right, 0] -= force
            for a in _CHARGED:
                for b in _CHARGED:
                    delta = placed[left, a] - placed[right, b]
                    distance2 = float(np.dot(delta, delta))
                    distance = math.sqrt(distance2)
                    pair = TIP4P_COULOMB * TIP4P_CHARGES[a] * TIP4P_CHARGES[b] / distance
                    energy += pair
                    slope = -pair / distance2
                    force = slope * delta
                    gradient[left, a] += force
                    gradient[right, b] -= force
    return float(energy), gradient


def tip4p(coords: Sequence[float] | np.ndarray) -> tuple[float, np.ndarray]:
    """Rigid TIP4P energy and dE/d(centre, angle-axis), flattened."""
    centres, axes = _split_rigid(coords)
    count = centres.shape[0]
    references = reference_sites()
    placed = np.zeros((count, 4, 3), dtype=np.float64)
    derivative = np.zeros((count, 4, 3, 3), dtype=np.float64)
    for index in range(count):
        rotated, rotated_derivative = _rotate(axes[index], references)
        placed[index] = centres[index] + rotated
        derivative[index] = rotated_derivative
    energy, site_gradient = site_energy_forces(placed)
    gradient = np.zeros((count, 6), dtype=np.float64)
    for index in range(count):
        gradient[index, :3] = np.sum(site_gradient[index], axis=0)
        gradient[index, 3:] = np.einsum("sak,sa->k", derivative[index], site_gradient[index])
    return energy, gradient.reshape(-1)


def sites_from_ohh(coords: Sequence[float] | np.ndarray) -> np.ndarray:
    """Build O, H, H, M from Cartesian O, H, H blocks. M lies on the bisector."""
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 3)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] % 3 != 0:
        raise ValueError("OHH coordinates are O, H, H repeated")
    molecules = values.reshape(-1, 3, 3)
    placed = np.zeros((molecules.shape[0], 4, 3), dtype=np.float64)
    for index, molecule in enumerate(molecules):
        oxygen, hydrogen_1, hydrogen_2 = molecule
        bond_1 = hydrogen_1 - oxygen
        bond_2 = hydrogen_2 - oxygen
        bond_1 /= np.linalg.norm(bond_1)
        bond_2 /= np.linalg.norm(bond_2)
        bisector = bond_1 + bond_2
        bisector /= np.linalg.norm(bisector)
        placed[index, 0] = oxygen
        placed[index, 1] = hydrogen_1
        placed[index, 2] = hydrogen_2
        placed[index, 3] = oxygen + TIP4P_ROM * bisector
    return placed


def rigid_from_ohh(coords: Sequence[float] | np.ndarray) -> np.ndarray:
    """Centre-of-mass and GMIN angle-axis coordinates for an O, H, H geometry."""
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 3)
    molecules = values.reshape(-1, 3, 3)
    references = reference_sites()[:3]
    rigid = np.zeros((molecules.shape[0], 6), dtype=np.float64)
    masses = TIP4P_MASSES[:3]
    for index, molecule in enumerate(molecules):
        centre = np.sum(molecule * masses[:, None], axis=0) / float(np.sum(masses))
        displacement = molecule - centre
        correlation = references.T @ displacement
        left, _scale, right = np.linalg.svd(correlation)
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right = right.copy()
            right[-1] *= -1.0
            rotation = right.T @ left.T
        trace = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        theta = math.acos(trace)
        if theta < 1e-12:
            axis = np.zeros(3, dtype=np.float64)
        else:
            axis = np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ],
                dtype=np.float64,
            )
            axis /= 2.0 * math.sin(theta)
            axis *= -theta
        rigid[index, :3] = centre
        rigid[index, 3:] = axis
    return rigid


def tip4p_ohh(coords: Sequence[float] | np.ndarray) -> tuple[float, np.ndarray]:
    """Energy of Cartesian O, H, H water and the rigid gradient at that geometry."""
    return tip4p(rigid_from_ohh(coords))


# eOn ``tip4p_unit_system.hpp`` electron-volt system. One eV is this many kJ/mol.
E_CHARGE_C = 1.60217653e-19
AVOGADRO = 6.0221415e23
KJ_PER_EV = E_CHARGE_C * AVOGADRO / 1000.0

# One campaign-file line. Both clients use this conversion.
CONVERSION_LEDGER = (
    "GMIN .aa is centre then angle-axis, six floats per molecule, rotated by "
    "optbench.tip4p.reference_sites with the GMIN angle-axis map. "
    "EON (SVN r2549) and eOn (develop 7095b23ac17d49f434a5e08ccb072897ceca698e) "
    "each get the same .con: H, H pairs then oxygens, the stride in tip4p_ccl.cpp "
    "computeHH_O_. The harness kernel is uncut rigid TIP4P in kJ/mol divided by "
    "e_charge*N_A/1000 from tip4p_unit_system.hpp. The Cambridge test multiplies "
    "that client energy back to kJ/mol. eOn built with -Dwith_water=true routes "
    "potential tip4p to rgpot::TIP4PPot and is not this kernel. "
    "EON (SVN r2549) is a different code and is not rebuilt."
)


def parse_aa(text: str) -> np.ndarray:
    """GMIN ``.aa`` text: centre then angle-axis, six floats per molecule.

    Lines hold three floats. Molecules are consecutive blocks of six.
    """
    values: list[float] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            values.extend(float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("an angle-axis file has a non-numeric field") from exc
    if not values or len(values) % 6 != 0:
        raise ValueError("angle-axis coordinates are six floats per molecule")
    return np.asarray(values, dtype=np.float64)


def sites_from_aa(coords: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rigid O, H, H, M sites from GMIN centre and angle-axis coordinates."""
    centres, axes = _split_rigid(coords)
    references = reference_sites()
    placed = np.zeros((centres.shape[0], 4, 3), dtype=np.float64)
    for index in range(centres.shape[0]):
        rotated, _derivative = _rotate(axes[index], references)
        placed[index] = centres[index] + rotated
    return placed


def ccl_from_sites(sites: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CCL atom order used by ``Tip4p::computeHH_O_``.

    Pairs of hydrogens, one pair per molecule, then every oxygen. The M site
    stays virtual. ``tip4p_ccl.cpp`` strides the first hydrogen by six doubles
    and the second by the three doubles that follow it, and puts oxygen at
    ``n_molecules * 6``.
    """
    placed = np.asarray(sites, dtype=np.float64)
    count = placed.shape[0]
    positions = np.zeros((3 * count, 3), dtype=np.float64)
    for index in range(count):
        positions[2 * index] = placed[index, 1]
        positions[2 * index + 1] = placed[index, 2]
    positions[2 * count :] = placed[:, 0]
    numbers = np.empty(3 * count, dtype=np.int64)
    numbers[: 2 * count] = 1
    numbers[2 * count :] = 8
    return positions, numbers


def ohh_from_ccl(positions: Sequence[float], numbers: Sequence[int]) -> np.ndarray:
    """Inverse of ``ccl_from_sites`` for the real atoms. Shape ``(n, 3, 3)`` is O, H, H."""
    coords = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    atomic = np.asarray(numbers, dtype=np.int64).reshape(-1)
    if coords.shape[0] != atomic.shape[0] or coords.shape[0] % 3 != 0:
        raise ValueError("TIP4P Cartesian input is three atoms per molecule")
    count = coords.shape[0] // 3
    if not np.all(atomic[: 2 * count] == 1) or not np.all(atomic[2 * count :] == 8):
        raise ValueError("TIP4P Cartesian order is H, H pairs then oxygens")
    molecules = np.zeros((count, 3, 3), dtype=np.float64)
    for index in range(count):
        molecules[index, 1] = coords[2 * index]
        molecules[index, 2] = coords[2 * index + 1]
        molecules[index, 0] = coords[2 * count + index]
    return molecules


def rigid_cartesian(positions: Sequence[float], numbers: Sequence[int]) -> tuple[float, np.ndarray]:
    """Uncut rigid TIP4P energy and Cartesian forces, in kJ/mol and kJ/mol/Å.

    The energy is ``tip4p`` of the rigid fit. Forces are the rigid-body
    forces of that gradient: the total force and the spatial torque, spread
    onto O, H, H by mass. A rigid displacement does not strain the molecule,
    which is the motion basin hopping follows from a rigid start.
    """
    molecules = ohh_from_ccl(positions, numbers)
    rigid = rigid_from_ohh(molecules.reshape(-1))
    centres, axes = _split_rigid(rigid)
    references = reference_sites()
    placed = np.zeros((centres.shape[0], 4, 3), dtype=np.float64)
    for index in range(centres.shape[0]):
        rotated, _derivative = _rotate(axes[index], references)
        placed[index] = centres[index] + rotated
    energy, site_gradient = site_energy_forces(placed)
    masses = TIP4P_MASSES[:3]
    mass = float(np.sum(masses))
    forces = np.zeros_like(molecules)
    eye = np.eye(3, dtype=np.float64)
    for index in range(molecules.shape[0]):
        centre = centres[index]
        relative = placed[index] - centre
        gradient = site_gradient[index]
        total = np.sum(gradient, axis=0)
        torque = np.sum(np.cross(relative, gradient), axis=0)
        inertia = np.zeros((3, 3), dtype=np.float64)
        arms = molecules[index] - centre
        for atom in range(3):
            arm = arms[atom]
            inertia += masses[atom] * (float(np.dot(arm, arm)) * eye - np.outer(arm, arm))
        alpha = np.linalg.solve(inertia, torque)
        for atom in range(3):
            forces[index, atom] = -(masses[atom] / mass) * total - masses[atom] * np.cross(alpha, arms[atom])
    ordered = np.zeros((molecules.shape[0] * 3, 3), dtype=np.float64)
    count = molecules.shape[0]
    for index in range(count):
        ordered[2 * index] = forces[index, 1]
        ordered[2 * index + 1] = forces[index, 2]
    ordered[2 * count :] = forces[:, 0]
    return float(energy), ordered.reshape(-1)
