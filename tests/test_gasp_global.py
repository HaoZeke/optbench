"""GASP log parsing and the force-call cap formula."""

from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.runners.gasp_global import (
    lammps_final_energy,
    lammps_force_evaluations,
    search_force_cap,
)

ROOT = Path(__file__).resolve().parents[1]

_LOG = """\
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final =
     -13.0  -138.5  -138.530000000001
  Iterations, force evaluations = 1000 1975
Loop time of 0.001 on 1 procs for 0 steps with 38 atoms
"""


def test_lammps_log_counts_minimize_and_run0():
    assert lammps_force_evaluations(_LOG) == 1976
    assert lammps_final_energy(_LOG) == -138.530000000001


def test_lammps_log_ignores_a_minimize_without_run0():
    text = "Iterations, force evaluations = 38 76\n"
    assert lammps_force_evaluations(text) == 76
    assert lammps_final_energy(text) is None


def test_two_minimizes_add_their_force_evaluations():
    text = (
        "Iterations, force evaluations = 1 2\n"
        "Energy initial, next-to-last, final =\n"
        "   -1.0  -2.0  -3.5\n"
        "Iterations, force evaluations = 10 20\n"
        "Energy initial, next-to-last, final =\n"
        "   -3.5  -4.0  -4.25\n"
        "Loop time of 1 on 1 procs for 0 steps with 38 atoms\n"
    )
    assert lammps_force_evaluations(text) == 23
    assert lammps_final_energy(text) == -4.25


def test_force_cap_sits_one_above_the_budget():
    assert search_force_cap(800, 20000) == 800 * 20001 + 1


def test_gasp_templates_match_the_cap_formula():
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "gasp":
            continue
        found.append(template["id"])
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        section = template["gasp"]
        assert cap == search_force_cap(int(section["num_energy_calcs"]), int(section["min_eval"]))
        assert template["commit"] == "0c8d993c82e0e1c69a05b3c34bbb2fcbbdbb7f07"
        assert template["code"] == "gasp"
    assert found == ["gasp/lj38-default"]
