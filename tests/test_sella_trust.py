"""Sella 0.1.1 trust radius and reactant mode on Sella 2.6.0."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sella")

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from sella import Sella

from optbench.ase_sella import _assign_reactant_mode


def test_archive_delta0_is_an_l2_radius_and_the_mode_uses_ufree():
    atoms = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=40.0)
    atoms.set_constraint(FixAtoms(indices=[2, 3]))
    reference = atoms.positions.copy()
    reference[0, 0] += 0.2
    reference[1, 1] += 0.1
    opt = Sella(
        atoms,
        logfile=None,
        order=1,
        delta0=1.3e-3,
        rs="tr",
        eta=1e-6,
        gamma=0.4,
    )
    _assign_reactant_mode(opt, atoms, reference)
    basis = np.asarray(opt.pes.get_Ufree())
    assert opt.delta == pytest.approx(1.3e-3 * basis.shape[1])
    displacement = (atoms.positions - reference).ravel()
    expected = basis.T @ (displacement / np.linalg.norm(displacement))
    expected = expected / np.linalg.norm(expected)
    assert np.dot(np.asarray(opt.pes.v0), expected) == pytest.approx(1.0)
