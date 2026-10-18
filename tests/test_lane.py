"""Harness checks for the globalopt-codes lane."""

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem
from optbench.lane_load import GeometryRefusal, load_starts
from optbench.lane_methods import _scipy_options, run_job
from optbench.potentials import binary_lennard_jones
from optbench.runners.harness import preflight

ROOT = Path(__file__).resolve().parents[1]


def test_binary_lennard_jones_uses_the_arithmetic_mean_sigma():
    left = np.array([0.0, 0.0, 0.0, 1.2, 0.0, 0.0])
    same = binary_lennard_jones(left, [1, 1])
    mixed = binary_lennard_jones(left, [1, 2])
    assert same[1].shape == (6,)
    assert np.isfinite(same[0]) and np.isfinite(mixed[0])
    assert mixed[0] != pytest.approx(same[0])


def test_tip4p_archives_are_refused_before_a_result_file(tmp_path: Path):
    problem = resolve_problem(ROOT, "globalopt/tip4p10")
    with pytest.raises(GeometryRefusal, match="no TIP4P potential"):
        load_starts(ROOT, problem, work=tmp_path, output=tmp_path / "out.dat")


def test_gmin_preflight_names_the_missing_interface():
    template = load_template(ROOT, "sqnm/min-lj38-default")
    template = dict(template)
    template["code"] = "gmin"
    template["search"] = {"method": "gmin", "family": "minimize"}
    with pytest.raises(RuntimeError, match="02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"):
        preflight(template)


def test_harness_templates_match_problem_stops():
    found = [row for row in list_templates(ROOT) if row.get("engine") == "harness"]
    assert len(found) >= 90
    for template in found:
        _check_stop(resolve_problem(ROOT, template["problem"]), template)
    default = load_template(ROOT, "sqnm/min-lj38-default")
    tuned = load_template(ROOT, "sqnm/min-lj38-tuned")
    assert default["ranked"] is False
    assert tuned["ranked"] is True
    assert load_template(ROOT, "geometric/min-lj38")["search"]["coordsys"] == "cart"


def test_tnc_does_not_receive_maxiter():
    options = _scipy_options("TNC", 10000)
    assert "maxiter" not in options
    assert options["maxfun"] == 10000


def test_scipy_lbfgs_relaxes_a_lennard_jones_dimer():
    coords = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    record = run_job(
        {
            "index": 0,
            "start": "dimer",
            "coords": coords,
            "box": None,
            "species": None,
            "mode": None,
            "kind": "lj",
            "layout": "coords",
            "family": "minimize",
            "method": "scipy-minimize",
            "scipy_method": "L-BFGS-B",
            "profile": "default",
            "cap": 200,
            "gtol": 0.01,
            "seed": 1,
            "target": None,
            "energy_window": 0.001,
            "sigma": 0.0,
            "kappa": 3.0,
            "noise_window": 5,
            "energy_tol": 0.01,
            "noise_seed": 1,
        }
    )
    assert record["success"] is True
    assert record["force_calls"] < 200
    assert record["energy"] == pytest.approx(-1.0, abs=1e-3)
    assert record["hessian_index"] == 0


def test_nlopt_tnewton_relaxes_a_lennard_jones_dimer():
    record = run_job(
        {
            "index": 0,
            "start": "dimer",
            "coords": [0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
            "box": None,
            "species": None,
            "mode": None,
            "kind": "lj",
            "layout": "coords",
            "family": "minimize",
            "method": "nlopt-minimize",
            "nlopt_algorithm": "LD_TNEWTON",
            "profile": "default",
            "cap": 200,
            "gtol": 0.01,
            "seed": 1,
            "target": None,
            "energy_window": 0.001,
            "sigma": 0.0,
            "kappa": 3.0,
            "noise_window": 5,
            "energy_tol": 0.01,
            "noise_seed": 1,
        }
    )
    assert record["success"] is True
    assert record.get("error") is None
    assert record["force_calls"] < 200
    assert record["energy"] == pytest.approx(-1.0, abs=1e-3)


def test_optking_cartesian_relaxes_a_lennard_jones_dimer():
    record = run_job(
        {
            "index": 0,
            "start": "dimer",
            "coords": [0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
            "box": None,
            "species": None,
            "mode": None,
            "kind": "lj",
            "layout": "coords",
            "family": "minimize",
            "method": "optking",
            "profile": "default",
            "cap": 200,
            "gtol": 0.01,
            "seed": 1,
            "target": None,
            "energy_window": 0.001,
            "sigma": 0.0,
            "kappa": 3.0,
            "noise_window": 5,
            "energy_tol": 0.01,
            "noise_seed": 1,
        }
    )
    assert record["success"] is True
    assert record["force_calls"] < 200
    assert record["energy"] == pytest.approx(-1.0, abs=1e-3)
    assert record["hessian_index"] == 0
