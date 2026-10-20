#!/usr/bin/env python3
"""Stock TSASE 1.0 SSDimer client for one EON process_search job.

The pin's ``SSDimer_atoms`` takes no ``alpha`` or ``alpha2``. This client
calls that constructor with ``ss=False`` and the archive knobs the pin
still accepts: ``phi_tol`` 5, ``rotationMax`` 50, ``maxStep`` 0.2.
``search(minForce=0.001)`` and the endpoint FIRE ``fmax`` follow
``Optimizer converged_force`` in the archived config. The Morse energy
is the pin's Fortran calculator. ASE 3.22 renamed
``read_vasp_configuration``; the alias below only lets the pin's ``io``
package import.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "1")

import ase.io.vasp as _vasp

if not hasattr(_vasp, "read_vasp_configuration"):
    _vasp.read_vasp_configuration = _vasp.read_vasp

_TSASE = os.environ.get("TSASE_ROOT", "/tmp/src/tsase")
if _TSASE not in sys.path:
    sys.path.insert(0, _TSASE)

import numpy as np
from ase.optimize.fire import FIRE

import tsase  # noqa: F401  the pin sets numpy error mode on import
from tsase.calculators.morse import morse
from tsase.dimer.ssdimer import SSDimer_atoms
from tsase.io.con import read_con, write_con
from tsase.neb.util import vmag

# The pin raises on underflow. Numpy 2 hits that in the dimer trig.
np.seterr(all="ignore")

MIN_FORCE = 0.001
MAX_STEPS = 5000
ROTATION_MAX = 50
MAX_STEP = 0.20
PHI_TOL = 5
R_DIFF = 0.3
PREFACTOR = 1.2e12


def _endpoint(mode, origin, destination, step=1.0):
    natom = len(origin)
    volume = origin.get_volume()
    jacob = (volume / natom) ** (1.0 / 3.0) * natom**0.5
    shift = step * mode
    cell0 = origin.get_cell()
    cell1 = cell0 + np.dot(cell0, shift[-3:]) / jacob
    destination.set_cell(cell1, scale_atoms=True)
    scaled = origin.get_scaled_positions()
    destination.set_positions(np.dot(scaled, cell1) + shift[:-3])


def _write_results(status, displaced, reactant, saddle, product, saddle_calls, opt_calls):
    total = saddle_calls + opt_calls
    if status != 0:
        reactant = 0.0
        product = 0.0
        saddle = 0.0
    lines = [
        f"{status} termination_reason",
        "saddle_search job_type",
        f"{total} total_force_calls",
        f"{saddle_calls} force_calls_saddle",
        f"{opt_calls} force_calls_minimization",
        f"{displaced} displacement_saddle_distance",
        f"{reactant} potential_energy_reactant",
        f"{saddle} potential_energy_saddle",
        f"{product} potential_energy_product",
        f"{saddle - reactant} barrier_reactant_to_product",
        f"{saddle - product} barrier_product_to_reactant",
        "0 iterations",
        "42 random_seed",
        "morse_pt potential_type",
        f"{PREFACTOR} prefactor_product_to_reactant",
        f"{PREFACTOR} prefactor_reactant_to_product",
        "0 force_calls_prefactors",
    ]
    with open("results.dat", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    calc = morse()
    reactant = read_con("pos.con")
    reactant.set_calculator(calc)
    reactant_energy = float(reactant.get_potential_energy())
    write_con("reactant.con", reactant)

    displaced = read_con("displacement.con")
    displaced.set_calculator(calc)
    distance = float(vmag(displaced.get_positions() - reactant.get_positions()))
    mode = np.loadtxt("direction.dat")

    dimer = SSDimer_atoms(
        displaced,
        mode=mode,
        rotationMax=ROTATION_MAX,
        maxStep=MAX_STEP,
        phi_tol=PHI_TOL,
        ss=False,
    )
    # quiet=True freezes the curvature test in this pin, so the search
    # runs to the force-call cap. The archive client left quiet off.
    dimer.search(minForce=MIN_FORCE, movie=None, quiet=False)
    saddle_calls = int(dimer.forceCalls)
    status = 5 if saddle_calls > 100000 else -1
    mode = dimer.get_mode()
    np.savetxt("mode.dat", mode)
    write_con("saddle.con", dimer.R0)
    saddle_energy = float(dimer.get_potential_energy())
    opt_calls = 0
    product_energy = 0.0

    if status != 5:
        forward = dimer.R0.copy()
        backward = dimer.R0.copy()
        forward.set_calculator(calc)
        backward.set_calculator(calc)
        padded = np.vstack((mode, np.zeros((3, 3))))
        _endpoint(padded, dimer.R0, backward)
        _endpoint(-padded, dimer.R0, forward)
        first = FIRE(backward, logfile=None, dt=0.2, maxstep=0.2, dtmax=0.2)
        first.run(fmax=MIN_FORCE, steps=MAX_STEPS)
        second = FIRE(forward, logfile=None, dt=0.2, maxstep=0.2, dtmax=0.2)
        second.run(fmax=MIN_FORCE, steps=MAX_STEPS)
        opt_calls = int(first.nsteps + second.nsteps)
        back_distance = float(vmag(backward.get_positions() - reactant.get_positions()))
        forward_distance = float(vmag(forward.get_positions() - reactant.get_positions()))
        back_is_reactant = back_distance <= R_DIFF
        forward_is_reactant = forward_distance <= R_DIFF
        if back_is_reactant and forward_is_reactant:
            status = 6
        elif back_is_reactant:
            status = 0
            write_con("product.con", forward)
            product_energy = float(forward.get_potential_energy())
        elif forward_is_reactant:
            status = 0
            write_con("product.con", backward)
            product_energy = float(backward.get_potential_energy())
        else:
            status = 6
        write_con("CONT_backward.con", backward)
        write_con("CONT_forward.con", forward)

    _write_results(
        status,
        distance,
        reactant_energy,
        saddle_energy,
        product_energy,
        saddle_calls,
        opt_calls,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
