"""ASE 3.6.0 LBFGS and the TSASE Morse force used by the 2013 bulk script.

The archived ``run.py`` counts ``LBFGS.force_calls``. With line search off,
that counter increments once per accepted step inside ``step``, not on the
convergence probe ``atoms.get_forces``. The inverse Hessian stays ``1/alpha``.
TSASE ``FORCE`` (svn r338) shifts the pair energy to zero at the cutoff and
folds orthorhombic minimum images with Fortran ``DNINT``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DTYPE = np.float64


def dnint(values: np.ndarray) -> np.ndarray:
    """Fortran ``DNINT``: round half away from zero."""
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def read_poscar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a VASP POSCAR the way ASE 3.6 ``read_vasp`` does.

    Returns Cartesian positions and the cell, both float64.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 1
    scale = float(lines[cursor].split()[0])
    cursor += 1
    cell = np.array(
        [[float(part) for part in lines[cursor + row].split()[:3]] for row in range(3)],
        dtype=_DTYPE,
    )
    cell *= scale
    cursor += 3
    counts = lines[cursor].split()
    cursor += 1
    try:
        int(counts[0])
    except ValueError:
        counts = lines[cursor].split()
        cursor += 1
    natoms = sum(int(count) for count in counts)
    mode = lines[cursor].strip().lower()
    cursor += 1
    if mode.startswith("s"):
        mode = lines[cursor].strip().lower()
        cursor += 1
    coords = np.array(
        [[float(part) for part in lines[cursor + row].split()[:3]] for row in range(natoms)],
        dtype=_DTYPE,
    )
    cartesian = mode.startswith("c") or mode.startswith("k")
    if cartesian:
        return coords * scale, cell
    return coords @ cell, cell


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


def _longest_scale(step: np.ndarray, maxstep: float) -> np.ndarray:
    """Scale the step so the longest per-atom move is ``maxstep``."""
    lengths = np.sqrt(np.sum(step * step, axis=1))
    longest = float(np.max(lengths))
    if longest >= maxstep:
        step *= _DTYPE(maxstep) / _DTYPE(longest)
    return step


def minimize(
    positions: np.ndarray,
    cell: np.ndarray,
    potential: dict,
    parameters: dict,
    tol: float,
    max_steps: int,
) -> dict:
    """Run ASE 3.6.0 LBFGS and return its ``force_calls``."""
    memory = int(parameters.get("memory", 100))
    maxstep = float(parameters.get("maxstep", 0.04))
    alpha = float(parameters.get("alpha", 10.0))
    damping = float(parameters.get("damping", 1.0))
    h0 = _DTYPE(1.0 / alpha)
    u0 = float(potential["D"])
    morse_alpha = float(potential["alpha"])
    r0 = float(potential["r0"])
    rc = float(potential["rc"])
    current = np.array(positions, dtype=_DTYPE, copy=True)
    history_s: list[np.ndarray] = []
    history_y: list[np.ndarray] = []
    history_rho: list[float] = []
    iteration = 0
    force_calls = 0
    previous_r: np.ndarray | None = None
    previous_f: np.ndarray | None = None
    initial_norm = 0.0
    energy = 0.0
    norm = 0.0
    for step_index in range(max_steps):
        energy, forces = morse_force(current, cell, u0, morse_alpha, r0, rc)
        norm = float(np.linalg.norm(forces))
        if step_index == 0:
            initial_norm = norm
        if norm <= tol:
            break
        if iteration > 0:
            assert previous_r is not None and previous_f is not None
            s0 = current.reshape(-1) - previous_r.reshape(-1)
            y0 = previous_f.reshape(-1) - forces.reshape(-1)
            history_s.append(s0)
            history_y.append(y0)
            history_rho.append(float(1.0 / np.dot(y0, s0)))
        if iteration > memory:
            history_s.pop(0)
            history_y.pop(0)
            history_rho.pop(0)
        loopmax = min(memory, iteration)
        q = -forces.reshape(-1)
        coeff = np.empty(loopmax, dtype=_DTYPE)
        for index in range(loopmax - 1, -1, -1):
            coeff[index] = history_rho[index] * np.dot(history_s[index], q)
            q -= coeff[index] * history_y[index]
        z = h0 * q
        for index in range(loopmax):
            beta = history_rho[index] * np.dot(history_y[index], z)
            z += history_s[index] * (coeff[index] - beta)
        direction = -z.reshape((-1, 3))
        force_calls += 1
        displacement = _longest_scale(direction, maxstep) * _DTYPE(damping)
        previous_r = current.copy()
        previous_f = forces.copy()
        iteration += 1
        current = current + displacement
    else:
        energy, forces = morse_force(current, cell, u0, morse_alpha, r0, rc)
        norm = float(np.linalg.norm(forces))
    return {
        "force_calls": force_calls,
        "force_norm": norm,
        "initial_force_norm": initial_norm,
        "energy": energy,
        "failed": norm > tol,
        "parameters": {
            "memory": memory,
            "maxstep": maxstep,
            "alpha": alpha,
            "damping": damping,
            "H0": float(h0),
            "use_line_search": False,
        },
    }


def minimize_poscar(
    path: Path,
    potential: dict,
    parameters: dict,
    tol: float,
    max_steps: int,
) -> dict:
    positions, cell = read_poscar(path)
    return minimize(positions, cell, potential, parameters, tol, max_steps)
