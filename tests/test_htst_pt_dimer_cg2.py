"""The Pt dimer-CG max=50 template uses the 2014 barrier list."""

from pathlib import Path

from optbench.campaign import load_template
from optbench.htst import Process, Search, reduce_htst, spec_from_problem

ROOT = Path(__file__).resolve().parents[1]


def test_pt_dimer_cg2_keeps_the_archived_rotation():
    template = load_template(ROOT, "eon-svn/htst-pt-dimer-cg2")
    dimer = template["ini"]["Dimer"]
    assert dimer["opt_method"] == "cg"
    assert dimer["rotations_max"] == 50
    assert dimer["converged_angle"] == 1.0
    assert dimer["improved"] is True
    assert template["ini"]["Potential"]["potential"] == "morse_pt"
    assert template["ini"]["AKMC"]["thermally_accessible_window"] == 22
    assert template["ini"]["Optimizer"]["opt_method"] == "cg"
    assert template["ini"]["Optimizer"]["convergence_metric"] == "max_atom"
    assert template["window_target"] == 13
    spec = spec_from_problem({"id": template["id"], "htst": template["htst"]})
    assert spec.barriers == (
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
    assert spec.rate_ref == 155055901.244
    assert spec.prefactor == 1.2e12
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
    reduction = reduce_htst([process], [search], spec)
    assert reduction.missed_barriers
    assert reduction.barrier_windows == 0
