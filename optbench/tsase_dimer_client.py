"""AKMC client for the latest TSASE solid-state dimer on Morse platinum.

The saddle search is ``SSDimer_atoms`` from TSASE 1.0 (98aac60) with
stock constructor settings. The archive convergence criterion is
``rotationMax=50``, ``phi_tol=5``, and a maximum atom force of 1e-4.
Both endpoints are relaxed with FIRE (``dt=0.2``, ``maxstep=0.2``,
``dtmax=0.2``, ``fmax=1e-4``, 5000 steps). A result is connected when
exactly one endpoint lies within 0.3 of the reactant.
``force_calls_saddle`` is the harness ASE ``calculate`` count during
the dimer search. The minimizer count is recorded and is not part of
the hTST sum.
"""

from __future__ import annotations

import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from optbench.counter import HarnessCounter, attach_ase_counter
from optbench.engines.tsase_search import _load_dimer
from optbench.tsase_morse import Morse

DIMER_EDIFFG = 0.0001
OPT_EDIFFG = 0.0001
RDIFF = 0.3
_DIMER = _load_dimer()


def vmag(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


class MorseCalculator(Calculator):
    """ASE calculator over the TSASE Morse kernel. One ``calculate`` is one harness call."""

    implemented_properties = ("energy", "forces")
    nolabel = True

    def __init__(self) -> None:
        super().__init__()
        self._kernel = Morse()

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        if system_changes is None:
            system_changes = ["positions"]
        Calculator.calculate(self, atoms, properties, system_changes)
        energy = self._kernel.get_potential_energy(self.atoms)
        forces = self._kernel.get_forces(self.atoms)
        self.results = {"energy": float(energy), "forces": np.asarray(forces, dtype=np.float64)}


def _box(lengths: np.ndarray, angles: np.ndarray) -> np.ndarray:
    box = np.zeros((3, 3))
    radians = angles * np.pi / 180.0
    box[0][0] = 1.0
    box[1][0] = np.cos(radians[0])
    box[1][1] = np.sin(radians[0])
    box[2][0] = np.cos(radians[1])
    box[2][1] = (np.cos(radians[2]) - box[1][0] * box[2][0]) / box[1][1]
    box[2][2] = np.sqrt(1.0 - box[2][0] ** 2 - box[2][1] ** 2)
    box[0, :] *= lengths[0]
    box[1, :] *= lengths[1]
    box[2, :] *= lengths[2]
    return box


def _lengths_angles(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(cell, axis=1)
    angles = np.zeros(3)
    angles[0] = np.arccos(np.dot(cell[0] / lengths[0], cell[1] / lengths[1]))
    angles[1] = np.arccos(np.dot(cell[0] / lengths[0], cell[2] / lengths[2]))
    angles[2] = np.arccos(np.dot(cell[1] / lengths[1], cell[2] / lengths[2]))
    return lengths, angles * 180.0 / np.pi


def read_con(path: Path) -> Atoms:
    """Read one con frame the way TSASE ``read_con`` indexes the lines."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lengths = np.array([float(value) for value in lines[2].split()[0:3]])
    angles = np.array([float(value) for value in lines[3].split()[0:3]])
    cell = _box(lengths, angles)
    ntypes = int(lines[6].split()[0])
    counts = [int(value) for value in lines[7].split()[0:ntypes]]
    masses_of_type = [float(value) for value in lines[8].split()[0:ntypes]]
    frozen: list[int] = []
    positions: list[list[float]] = []
    symbols: list[str] = []
    masses: list[float] = []
    cursor = 9
    atom_index = 0
    for type_index in range(ntypes):
        symbol = lines[cursor].strip()
        mass = masses_of_type[type_index]
        cursor += 2
        for _ in range(counts[type_index]):
            fields = lines[cursor].split()
            positions.append([float(value) for value in fields[0:3]])
            symbols.append(symbol)
            masses.append(mass)
            if fields[3] != "0":
                frozen.append(atom_index)
            atom_index += 1
            cursor += 1
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=(True, True, True),
        masses=masses,
    )
    atoms.set_constraint(FixAtoms(frozen))
    return atoms


def write_con(path: Path, atoms: Atoms) -> None:
    """Write a con file eOn ``loadcon`` can read back."""
    cell = np.asarray(atoms.cell, dtype=np.float64)
    lengths, angles = _lengths_angles(cell)
    frozen = set()
    if atoms.constraints:
        index = atoms.constraints[0].index
        frozen = set(int(i) for i in np.atleast_1d(index))
    symbols = atoms.get_chemical_symbols()
    order: list[str] = []
    counts: dict[str, int] = {}
    type_mass: list[float] = []
    for i, symbol in enumerate(symbols):
        if symbol not in counts:
            order.append(symbol)
            counts[symbol] = 0
            type_mass.append(float(atoms.get_masses()[i]))
        counts[symbol] += 1
    lines = ["Generated by tsase", ""]
    lines.append(" ".join(f"{value:12.6f}" for value in lengths))
    lines.append(" ".join(f"{value:12.6f}" for value in angles))
    lines.append("")
    lines.append("")
    lines.append(str(len(order)))
    lines.append(" ".join(str(counts[name]) for name in order))
    lines.append(" ".join(f"{value:12.6f}" for value in type_mass))
    positions = atoms.get_positions()
    serial = 0
    for component, name in enumerate(order, start=1):
        lines.append(name)
        lines.append(f"Coordinates of Component {component}")
        for i, symbol in enumerate(symbols):
            if symbol != name:
                continue
            flag = 1 if i in frozen else 0
            x, y, z = positions[i]
            lines.append(f"{x:12.6f} {y:12.6f} {z:12.6f} {flag} {serial}")
            serial += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _endpoint(mode: np.ndarray, origin: Atoms, dest: Atoms, jacob: float, sign: float) -> None:
    dRvec = sign * mode
    cell0 = origin.get_cell()
    cell1 = cell0 + np.dot(cell0, dRvec[-3:]) / jacob
    dest.set_cell(cell1, scale_atoms=True)
    scaled = origin.get_scaled_positions()
    dest.set_positions(np.dot(scaled, cell1) + dRvec[:-3])


def _results(
    status: int,
    total: int,
    saddle_calls: int,
    opt_calls: int,
    displaced: float,
    reactant: float,
    saddle: float,
    product: float,
) -> str:
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
        "eam_al potential_type",
        f"{1.2e12} prefactor_product_to_reactant",
        f"{1.2e12} prefactor_reactant_to_product",
        "0 force_calls_prefactors",
    ]
    return "\n".join(lines)


def run_job(directory: Path) -> None:
    """Run one saddle search in ``directory`` and write ``results.dat``."""
    os.chdir(directory)
    np.seterr(all="ignore")
    reactant = read_con(Path("pos.con"))
    calc = MorseCalculator()
    reactant.calc = calc
    e_reactant = float(reactant.get_potential_energy())
    write_con(Path("reactant.con"), reactant)
    displaced = read_con(Path("displacement.con"))
    displaced.calc = calc
    distance = vmag(displaced.get_positions() - reactant.get_positions())
    mode = np.loadtxt("direction.dat")
    status = -1
    counter = HarnessCounter(cache=False)
    attach_ase_counter(calc, counter=counter)
    dimer = _DIMER(displaced, mode=mode, rotationMax=50, phi_tol=5)
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        dimer.search(minForce=DIMER_EDIFFG, maxForceCalls=100000)
    saddle_calls = int(counter.calls)
    if int(dimer.forceCalls) >= 100000 or not dimer.converged:
        status = 5
    mode = np.asarray(dimer.get_mode(), dtype=np.float64)
    if mode.shape[0] == len(reactant):
        mode = np.vstack((mode, np.zeros((3, 3))))
    write_con(Path("saddle.con"), dimer.R0)
    np.savetxt("mode.dat", mode)
    e_saddle = float(dimer.R0.get_potential_energy())
    opt_calls = 0
    e_product = 0.0
    if status != 5:
        natom = len(reactant)
        volume = reactant.get_volume()
        jacob = (volume / natom) ** (1.0 / 3.0) * natom**0.5
        forward = dimer.R0.copy()
        backward = forward.copy()
        forward.calc = calc
        backward.calc = calc
        _endpoint(mode, dimer.R0, backward, jacob, 1.0)
        _endpoint(mode, dimer.R0, forward, jacob, -1.0)
        backward.calc = calc
        forward.calc = calc
        dyn0 = FIRE(backward, logfile=None, dt=0.2, maxstep=0.2, dtmax=0.2)
        dyn0.run(fmax=OPT_EDIFFG, steps=5000)
        dyn2 = FIRE(forward, logfile=None, dt=0.2, maxstep=0.2, dtmax=0.2)
        dyn2.run(fmax=OPT_EDIFFG, steps=5000)
        opt_calls = int(dyn0.nsteps + dyn2.nsteps)
        d_back = vmag(backward.get_positions() - reactant.get_positions())
        d_forward = vmag(forward.get_positions() - reactant.get_positions())
        back_connected = d_back <= RDIFF
        forward_connected = d_forward <= RDIFF
        write_con(Path("CONT_backward.con"), backward)
        write_con(Path("CONT_forward.con"), forward)
        if back_connected and forward_connected:
            status = 6
        elif back_connected:
            status = 0
            write_con(Path("product.con"), forward)
            e_product = float(forward.get_potential_energy())
        elif forward_connected:
            status = 0
            write_con(Path("product.con"), backward)
            e_product = float(backward.get_potential_energy())
        else:
            status = 6
    if status != 0:
        e_reactant = 0.0
        e_product = 0.0
        e_saddle = 0.0
    Path("results.dat").write_text(
        _results(status, saddle_calls + opt_calls, saddle_calls, opt_calls, distance, e_reactant, e_saddle, e_product),
        encoding="utf-8",
    )


def main() -> None:
    try:
        run_job(Path.cwd())
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise


if __name__ == "__main__":
    main()
