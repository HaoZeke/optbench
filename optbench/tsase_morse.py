"""TSASE Morse calculator, the January 2013 ``force_`` kernel.

The Fortran source is ``optbench/vendor/tsase/morse.f``. It is the
pair loop shipped with TSASE: depth 0.7102, alpha 1.6047, r0 2.897,
cutoff 9.5, energy shifted by the cutoff value, minimum image on each
box edge. The shared library is built with gfortran on first use.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).resolve().parent / "vendor" / "tsase" / "morse.f"
_CACHE = Path.home() / ".cache" / "optbench" / "tsase-morse"


def library_path() -> Path:
    """Compile ``morse.f`` once and return the shared library."""
    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:12]
    dest = _CACHE / f"libmorse-{digest}.so"
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gfortran", "-O3", "-fPIC", "-shared", "-o", str(dest), str(_SOURCE)],
        check=True,
    )
    return dest


def _load():
    lib = ctypes.CDLL(str(library_path()))
    lib.force_.argtypes = [ctypes.POINTER(ctypes.c_int)] + [ctypes.POINTER(ctypes.c_double)] * 9
    lib.force_.restype = None
    return lib.force_


_FORCE = None


def _force():
    global _FORCE
    if _FORCE is None:
        _FORCE = _load()
    return _FORCE


class Morse:
    """ASE-style calculator. ``get_forces`` returns a copy of the raw forces.

    Fixed-atom constraints are applied by ASE after this copy, so the
    cache stays the unconstrained Fortran result.
    """

    def __init__(self, u0: float = 0.7102, alpha: float = 1.6047, r0: float = 2.8970, rc: float = 9.5):
        self.u0 = float(u0)
        self.alpha = float(alpha)
        self.r0 = float(r0)
        self.rc = float(rc)
        self.force_calls = 0
        self._positions: np.ndarray | None = None
        self._cell: np.ndarray | None = None
        self._energy = 0.0
        self._forces: np.ndarray | None = None

    def _current(self, atoms) -> bool:
        if self._positions is None or self._forces is None or self._cell is None:
            return False
        return np.array_equal(self._positions, atoms.positions) and np.array_equal(self._cell, np.asarray(atoms.cell))

    def _evaluate(self, atoms) -> None:
        positions = np.ascontiguousarray(atoms.positions, dtype=np.float64)
        ra = np.ascontiguousarray(positions.ravel(), dtype=np.float64)
        fa = np.zeros_like(ra)
        energy = np.zeros(1, dtype=np.float64)
        cell = np.asarray(atoms.cell, dtype=np.float64)
        count = ctypes.c_int(positions.shape[0])
        bx = ctypes.c_double(cell[0, 0])
        by = ctypes.c_double(cell[1, 1])
        bz = ctypes.c_double(cell[2, 2])
        depth = ctypes.c_double(self.u0)
        alpha = ctypes.c_double(self.alpha)
        r0 = ctypes.c_double(self.r0)
        cutoff = ctypes.c_double(self.rc)
        _force()(
            ctypes.byref(count),
            ra.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            fa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            energy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(bx),
            ctypes.byref(by),
            ctypes.byref(bz),
            ctypes.byref(depth),
            ctypes.byref(alpha),
            ctypes.byref(r0),
            ctypes.byref(cutoff),
        )
        self._positions = positions.copy()
        self._cell = cell.copy()
        self._energy = float(energy[0])
        self._forces = fa.reshape((-1, 3)).copy()
        self.force_calls += 1

    def get_potential_energy(self, atoms=None, force_consistent: bool = False):
        del force_consistent
        if atoms is None or not self._current(atoms):
            if atoms is None:
                raise RuntimeError("Morse.get_potential_energy needs atoms")
            self._evaluate(atoms)
        return self._energy

    def get_forces(self, atoms):
        if not self._current(atoms):
            self._evaluate(atoms)
        return self._forces.copy()
