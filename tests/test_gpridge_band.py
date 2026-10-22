from pathlib import Path

import h5py
import numpy as np
import pytest

from optbench import gpridge_band as native
from optbench.pes import Pair


def pair():
    return Pair("start", np.array([[0., 0., 0.], [3., 0., 0.]]),
                np.array([[1., 0., 0.], [3., 0., 0.]]),
                np.array([True, False]), np.array([20., 20., 30.]))


def settings():
    return {"n_interior": 8, "force_norm": .01, "max_move": .2, "cap": 10000}


def test_command_keeps_the_problem_stop_and_geometry(tmp_path):
    argv = native.command("gprn", tmp_path, settings(), {})
    flags = dict(zip(argv[1::2], argv[2::2]))
    assert flags["--nimages"] == "8"
    assert flags["--convergence-force-norm"] == "l2"
    assert flags["--converged-force"] == "0.01"
    assert flags["--climbing-image-converged-only"] == "false"
    assert flags["--spring-constant"] == flags["--spring-constant-min"] == "5"
    assert flags["--neb-lbfgs-max-move"] == "0.2"
    assert flags["--gp-nonperiodic"] == "false"
    assert flags["--endpoint-assign"] == flags["--sidpp"] == "false"


def test_input_roundtrip_retains_fixed_atoms_cell_and_cutoff(tmp_path):
    import tomllib
    from ase.io import read

    directory = tmp_path / "case"
    native.write_inputs(directory, pair())
    for name, positions in (("reactant", pair().reactant), ("product", pair().product)):
        atoms = read(directory / f"{name}.con", format="eon")
        np.testing.assert_allclose(atoms.positions, positions, atol=1e-12)
        np.testing.assert_allclose(atoms.cell.lengths(), pair().box)
        assert atoms.constraints[0].index.tolist() == [1]
    params = tomllib.loads((directory / "potential.toml").read_text())["oracle"]["parameters"]["morse"]
    assert params == {"de": .7102, "a": 1.6047, "re": 2.897, "cutoff": 9.5}
    assert (directory / "environment-types.txt").read_text() == "0 78 0\n1 78 0\n"


def test_missing_and_separate_search_counts():
    assert native.parse_calls("search unfinished") == (None, None)
    assert native.parse_calls("Oracle calls (to convergence): 17  (endpoints=2)\n"
                              "Oracle calls (with validation): 29 (validation=12)\n") == (17, 29)


def write_band(path, positions, gradient=None, energy=0.):
    with h5py.File(path, "w") as out:
        out["summary/max_energy_image"] = [1.]
        out["converged/1/positions"] = np.asarray(positions).reshape(-1)
        out["converged/1/gradient"] = np.zeros(3) if gradient is None else gradient
        out["converged/1/energy"] = [energy]


def test_independent_check_accepts_only_an_index_one_stationary_point(tmp_path, monkeypatch):
    def saddle(positions, box):
        coeff = np.array([-1., 2., 3.])
        gradient = np.zeros_like(positions)
        gradient[0] = coeff * positions[0]
        return .5 * float(np.dot(gradient[0], positions[0])), gradient

    monkeypatch.setattr(native, "morse_pt", saddle)
    path = tmp_path / "band.h5"
    write_band(path, [0., 0., 0.])
    record = native.verify_band(path, pair(), settings())
    assert record["reasons"] == []
    assert record["hessian_index"] == 1
    assert record["verify_calls"] == 7
    assert record["gradient_error"] == 0
    write_band(path, [0.1, 0., 0.], [-.1, 0., 0.], -.005)
    record = native.verify_band(path, pair(), settings())
    assert record["reasons"] == ["force_norm"]
    assert record["hessian_index"] is None


def test_surface_mismatch_and_moved_substrate_are_rejected(tmp_path):
    path = tmp_path / "band.h5"
    write_band(path, [0., 0., 0.])
    assert "potential_mismatch" in native.verify_band(path, pair(), settings())["reasons"]
    write_band(path, [[0., 0., 0.], [3.1, 0., 0.]])
    with pytest.raises(ValueError, match="fixed substrate"):
        native.verify_band(path, pair(), settings())
