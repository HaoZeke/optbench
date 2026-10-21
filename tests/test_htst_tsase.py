"""The TSASE Morse client and the hTST reducer for the Pt row."""

from pathlib import Path

import numpy as np
import pytest

from optbench.htst import reduce_tables
from optbench.htst_tsase import PT_SPEC
from optbench.pes import morse_pt
from optbench.tsase_dimer_client import read_con
from optbench.tsase_morse import Morse

ROOT = Path(__file__).resolve().parents[1]
POS = ROOT / "inputs/sha256/16acc6c442c11bcd917ba7fd44f6a4f285a5591dbdb3996d62572bd0dece1482/pos.con"


def test_fortran_morse_matches_the_shifted_pair_potential():
    atoms = read_con(POS)
    calc = Morse()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    raw = atoms.get_forces(apply_constraint=False)
    forces = atoms.get_forces()
    reference_energy, gradient = morse_pt(atoms.positions, np.asarray(atoms.cell.lengths(), dtype=np.float64))
    assert energy == pytest.approx(reference_energy, abs=1e-8)
    assert raw == pytest.approx(-gradient, abs=1e-8)
    free = np.ones(len(atoms), dtype=bool)
    free[atoms.constraints[0].index] = False
    assert np.max(np.linalg.norm(forces[free], axis=1)) < 1e-5
    assert np.max(np.linalg.norm(forces[~free], axis=1)) == 0.0


def test_reducer_uses_sad_fcs_and_the_constant_prefactor():
    processtable = "\n".join(
        [
            "proc  saddle  prefactor  product  product_energy  product_prefactor  barrier  rate  repeats",
            "0 0 1 0 0 1 0.601 1 0",
            "1 0 1 0 0 1 0.601 1 0",
            "2 0 1 0 0 1 0.620 1 0",
            "3 0 1 0 0 1 0.986 1 0",
            "4 0 1 0 0 1 0.987 1 0",
            "5 0 1 0 0 1 0.989 1 0",
            "6 0 1 0 0 1 0.986 1 0",
            "7 0 1 0 0 1 0.987 1 0",
            "8 0 1 0 0 1 0.989 1 0",
            "9 0 1 0 0 1 1.196 1 0",
            "10 0 1 0 0 1 1.196 1 0",
            "11 0 1 0 0 1 1.196 1 0",
            "12 0 1 0 0 1 1.196 1 0",
            "13 0 1 0 0 1 1.207 1 0",
            "14 0 1 0 0 1 1.207 1 0",
        ]
    )
    header = "h\nh\n"
    rows = []
    for process_id in range(15):
        rows.append(f"0 random 0.10000 0.1 10 0 0 good-{process_id}")
    reduction = reduce_tables(processtable, header + "\n".join(rows) + "\n", PT_SPEC)
    assert reduction.missed_barriers == []
    assert reduction.barrier_windows == 1
    assert reduction.barrier_jobs == [15]
    assert reduction.barrier_force_calls == [150]
