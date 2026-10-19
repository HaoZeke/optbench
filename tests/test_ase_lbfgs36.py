"""ASE 3.6.0 Morse L-BFGS counts accepted steps on the archived POSCAR set."""

from pathlib import Path

from optbench.ase_lbfgs36 import minimize_poscar
from optbench.ase_sella import _summarize, discover, materialize
from optbench.campaign import load_template, resolve_problem

ROOT = Path(__file__).resolve().parents[1]


def _template() -> dict:
    return load_template(ROOT, "ase-3.6.0/morse-lbfgs")


def test_poscar_zero_takes_forty_accepted_steps():
    template = _template()
    problem = resolve_problem(ROOT, template["problem"])
    material = materialize(ROOT, template, problem, Path("/tmp/optbench-geometry"))
    jobs = discover(material, template)
    assert len(jobs) == 100
    assert [name for name, _paths, _seed in jobs[:3]] == ["POSCAR_0", "POSCAR_1", "POSCAR_2"]
    result = minimize_poscar(
        jobs[0][1][0],
        template["potential"],
        template["optimizer"]["parameters"],
        float(template["stop"]["force_norm"]),
        int(template["stop"]["max_steps"]),
    )
    assert result["force_calls"] == 40
    assert result["force_norm"] <= 1e-3
    assert result["parameters"]["H0"] == 0.04
    assert result["parameters"]["use_line_search"] is False
    assert abs(result["initial_force_norm"] - 10.61572344446026) < 1e-8


def test_the_campaign_mean_stays_unrounded():
    template = _template()
    rows = [
        {"failed": False, "force_calls": 35},
        {"failed": False, "force_calls": 36},
    ]
    summary = _summarize(rows, template)
    assert summary["force_calls"] == 35.5
    assert summary["nfailed"] == 0
