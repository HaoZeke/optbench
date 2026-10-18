"""Harness Hessians used by the geomeTRIC transition-state rows."""

from pathlib import Path

import pytest

from optbench.campaign import _check_stop, load_template, resolve_problem
from optbench.geometric_ts import lennard_jones_hessian, smooth_morse_hessian
from optbench.potentials import lennard_jones_hessian as pinned_lj_hessian

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f657e69bae9c81bf7a23e7375db98502ca5c8c30"


def test_templates_name_the_pin_and_the_problem_stop():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "geometric")
    assert pinned["commit"] == COMMIT
    assert pinned["ref"] == "1.1.1"
    names = [
        "geometric/lj38-ts-fd",
        "geometric/lj38-ts-harness",
        "geometric/pt-island-ts-fd",
        "geometric/pt-island-ts-harness",
    ]
    registered = {row["id"] for row in registry["template"]}
    for name in names:
        assert name in registered
        template = load_template(ROOT, name)
        assert template["commit"] == COMMIT
        assert template["code_version"] == "1.1.1"
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        assert template["stop"]["force_norm"] == problem["convergence"]["force_norm"]
        assert int(template["stop"]["max_force_calls"]) == int(problem["convergence"]["max_force_calls"])


def test_lennard_jones_hessian_matches_the_harness_potential():
    import numpy as np

    rng = np.random.default_rng(7)
    pos = rng.normal(size=(5, 3))
    got = lennard_jones_hessian(pos, 1.0, 1.0, 40.0)
    ref = pinned_lj_hessian(pos.ravel())
    assert np.allclose(got, ref, atol=1e-10)


def test_smooth_morse_hessian_matches_calculator_forces():
    pytest.importorskip("ase")
    import numpy as np
    from ase import Atoms

    from optbench.ase_sella import _inner_calculator

    rng = np.random.default_rng(11)
    pos = rng.uniform(0.0, 8.0, size=(5, 3))
    atoms = Atoms("Pt5", positions=pos, cell=[12.0, 11.0, 14.0], pbc=[True, True, False])
    potential = {
        "kind": "morse",
        "D": 0.7102,
        "alpha": 1.6047,
        "r0": 2.897,
        "rc": 9.5,
        "form": "smooth",
    }
    atoms.calc = _inner_calculator(potential)
    hessian = smooth_morse_hessian(
        atoms.get_positions(),
        atoms.cell.array,
        atoms.pbc,
        potential["D"],
        potential["alpha"],
        potential["r0"],
        potential["rc"],
    )
    step = 1e-5
    base = np.array(atoms.get_positions(), dtype=float)
    column = 4
    atom, axis = divmod(column, 3)
    plus = base.copy()
    minus = base.copy()
    plus[atom, axis] += step
    minus[atom, axis] -= step
    atoms.set_positions(plus)
    grad_plus = -np.array(atoms.get_forces(), dtype=float).ravel()
    atoms.set_positions(minus)
    grad_minus = -np.array(atoms.get_forces(), dtype=float).ravel()
    finite = (grad_plus - grad_minus) / (2.0 * step)
    assert float(np.max(np.abs(finite - hessian[:, column]))) < 1e-4
