"""Pt dimer-CG max=50: EON and eOn stay separate stock rows."""

from pathlib import Path

from optbench.campaign import load_template
from optbench.htst import Process, Search, reduce_htst, spec_from_problem

ROOT = Path(__file__).resolve().parents[1]

BARRIERS = (
    0.601,
    0.601,
    0.620,
    0.986,
    0.987,
    0.989,
    0.986,
    0.987,
    0.989,
    1.196,
    1.196,
    1.196,
    1.196,
    1.207,
    1.207,
)


def _untuned(template: dict) -> None:
    dimer = template["ini"]["Dimer"]
    assert dimer["rotations_max"] == 50
    assert dimer["converged_angle"] == 1.0
    assert "opt_method" not in dimer
    assert "improved" not in dimer
    assert template["ini"]["Potential"]["potential"] == "morse_pt"
    assert template["ini"]["Main"]["random_seed"] == 42
    assert template["ini"]["AKMC"]["confidence"] == 1.9
    assert "thermally_accessible_window" not in template["ini"]["AKMC"]
    assert "confidence_scheme" not in template["ini"]["AKMC"]
    assert "finite_difference" not in template["ini"]["Main"]
    optimizer = template["ini"]["Optimizer"]
    assert optimizer["converged_force"] == 0.001
    assert optimizer["max_iterations"] == 5000
    assert "opt_method" not in optimizer
    assert "convergence_metric" not in optimizer
    assert template["window_target"] == 9
    assert template.get("paper_baseline") is not True
    spec = spec_from_problem({"id": template["id"], "htst": template["htst"]})
    assert spec.barriers == BARRIERS
    assert spec.rate_ref == 155055901.244
    assert spec.prefactor == 1.2e12


def test_eon_svn_and_eon_git_are_separate_stock_rows():
    eon = load_template(ROOT, "EON/htst-pt-dimer-cg2")
    git = load_template(ROOT, "eOn/develop-htst-pt-dimer-cg2")
    assert eon["code"] == "EON (SVN r2549)"
    assert eon["code_version"] == "r2549"
    assert eon["commit"] == "r2549"
    assert git["code"] == "eOn (develop 500350d9)"
    assert git["code_version"] == "500350d9"
    assert git["commit"] == "500350d9fae881f06c7ef194e5868da079c4be5a"
    assert eon["code"] != git["code"]
    _untuned(eon)
    _untuned(git)
    process = Process(
        id=1,
        saddle_energy=-1.0,
        prefactor=1.2e12,
        barrier=0.601,
        barrier_raw=0.6012,
        rate=1.0,
        repeats=0,
    )
    search = Search(force_calls=10, barrier=0.6012, status="good", process_id=1)
    reduction = reduce_htst([process], [search], spec_from_problem({"id": eon["id"], "htst": eon["htst"]}))
    assert reduction.missed_barriers
    assert reduction.barrier_windows == 0
