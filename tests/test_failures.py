"""Boundaries of the central failure classifier."""

import math

from optbench.board import _record_from_raw
from optbench.failures import (
    apply_taxonomy,
    classify_failure,
    classify_outcome,
    classify_stored,
    ended_on_step_limit,
)
from optbench.score import score_v2
from optbench.taxonomy_diff import legacy_failure_of, legacy_record_from_raw, par2_mean


def test_nonfinite_energy_or_force_or_geometry_is_divergence():
    assert classify_outcome(energy=float("inf"), calls=1, cap=10) == "divergence"
    assert classify_outcome(energy=float("-inf"), calls=1, cap=10) == "divergence"
    assert classify_outcome(energy=float("nan"), calls=1, cap=10) == "divergence"
    assert classify_outcome(energy=1.0, geometry=[float("nan"), 0.0, 0.0], calls=1, cap=10) == "divergence"
    assert classify_outcome(energy=1.0, force=float("nan"), calls=100, cap=100, force_tol=0.01) == "divergence"


def test_a_finite_energy_is_not_a_blowup():
    assert (
        classify_outcome(
            energy=1.0e20,
            geometry=[[0.0, 0.0, 0.0]],
            force=0.0,
            calls=1,
            cap=10,
            force_tol=0.01,
            hessian_index=0,
            expected_index=0,
        )
        is None
    )


def test_dissociation_is_divergence_with_a_finite_energy():
    assert (
        classify_outcome(
            energy=3.7e6,
            force=1.97e8,
            calls=40,
            cap=1000,
            force_tol=0.01,
            dissociated=True,
        )
        == "divergence"
    )


def test_displacement_equal_to_the_bound_is_not_over_it():
    geometry = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    reference = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    kwargs = dict(
        energy=1.0,
        geometry=geometry,
        reference_geometry=reference,
        force=0.0,
        calls=1,
        cap=10,
        force_tol=1.0,
        hessian_index=0,
        expected_index=0,
    )
    assert classify_outcome(displacement_bound=1.0, **kwargs) is None
    assert classify_outcome(displacement=1.0, displacement_bound=1.0, energy=1.0, force=0.0, calls=1, cap=10, force_tol=1.0) is None
    assert classify_outcome(displacement_bound=0.999, **kwargs) == "divergence"
    assert (
        classify_outcome(
            displacement=1.0 + 1e-12,
            displacement_bound=1.0,
            energy=1.0,
            force=0.0,
            calls=1,
            cap=10,
            force_tol=1.0,
        )
        == "divergence"
    )


def test_cap_is_the_call_count_and_beats_a_crash():
    assert classify_outcome(energy=1.0, force=0.001, calls=100, cap=100, force_tol=0.01) == "cap"
    assert classify_outcome(energy=1.0, force=0.02, calls=99, cap=100, force_tol=0.01) == "not_converged"
    assert (
        classify_outcome(energy=1.0, force=1.0, calls=100, cap=100, force_tol=0.01, process_failed=True)
        == "cap"
    )
    assert classify_failure(["crash", "cap"]) == "cap"


def test_force_equal_to_the_tolerance_is_not_above_it():
    assert (
        classify_outcome(
            energy=-1.0,
            geometry=[[0.0, 0.0, 0.0]],
            force=0.01,
            calls=10,
            cap=100,
            force_tol=0.01,
            hessian_index=0,
            expected_index=0,
        )
        is None
    )
    assert (
        classify_outcome(
            energy=-1.0,
            force=math.nextafter(0.01, 1.0),
            calls=10,
            cap=100,
            force_tol=0.01,
        )
        == "not_converged"
    )


def test_wrong_index_at_the_force_stop_is_wrong_stationary():
    assert (
        classify_outcome(
            energy=-1.0,
            force=0.01,
            calls=5,
            cap=100,
            force_tol=0.01,
            hessian_index=1,
            expected_index=0,
        )
        == "wrong_stationary"
    )
    assert classify_failure(["hessian_index", "rmsd"]) == "wrong_stationary"
    assert classify_failure(["not_connected", "duplicate", "barrier"]) == "wrong_stationary"


def test_a_dead_process_under_the_cap_is_a_crash_and_a_blowup_wins():
    assert classify_outcome(energy=1.0, force=5.0, calls=4, cap=100, force_tol=0.01, process_failed=True) == "crash"
    assert classify_outcome(energy=float("nan"), calls=4, cap=100, process_failed=True) == "divergence"
    assert classify_failure(["crash"]) == "crash"


def test_lammps_neb_step_limit_is_cap_only_at_the_harness_cap():
    limited = [
        [("force tolerance", 4), ("max iterations", 9)],
        [("force tolerance", 4), ("max iterations", 9)],
    ]
    assert ended_on_step_limit(limited)
    assert ended_on_step_limit([[("max iterations", 1)], [("max force evaluations", 2)]])
    assert ended_on_step_limit(["max iterations", "max force evaluations"])
    assert not ended_on_step_limit([[("max iterations", 4), ("force tolerance", 2)]])
    assert not ended_on_step_limit([[("max iterations", 9)], [("force tolerance", 3)]])
    assert not ended_on_step_limit([])
    under = dict(
        energy=-1.0,
        force=0.016,
        calls=2485,
        cap=2500,
        force_tol=0.001,
        replica_stops=limited,
    )
    assert classify_outcome(**under) == "not_converged"
    assert classify_outcome(**{**under, "calls": 2499}) == "not_converged"
    assert classify_outcome(**{**under, "calls": 2500}) == "cap"
    assert (
        classify_outcome(
            energy=-1.0,
            force=0.016,
            calls=2485,
            cap=2500,
            force_tol=0.001,
            step_limit=True,
        )
        == "not_converged"
    )
    assert (
        classify_outcome(
            energy=-1.0,
            force=0.001,
            calls=2485,
            cap=2500,
            force_tol=0.001,
            step_limit=True,
            hessian_index=1,
            expected_index=1,
        )
        is None
    )
    assert (
        classify_outcome(
            energy=-1.0,
            force=1.0e-7,
            calls=2485,
            cap=2500,
            force_tol=0.001,
            step_limit=True,
            hessian_index=0,
            expected_index=1,
        )
        == "wrong_stationary"
    )
    assert (
        classify_outcome(
            energy=-1.0,
            force=float("nan"),
            calls=100,
            cap=2500,
            force_tol=0.001,
            step_limit=True,
        )
        == "divergence"
    )
    assert (
        classify_outcome(
            energy=-1.0,
            force=0.016,
            calls=100,
            cap=2500,
            force_tol=0.001,
            step_limit=True,
            process_failed=True,
        )
        == "crash"
    )
    assert (
        classify_outcome(
            energy=-1.0,
            force=0.016,
            calls=400,
            cap=2500,
            force_tol=0.001,
            replica_stops=[[("max iterations", 9)], [("force tolerance", 3)]],
        )
        == "not_converged"
    )


def test_stored_lammps_step_limit_under_the_cap_is_not_converged():
    stored = {
        "success": False,
        "failure": "cap",
        "reasons": ["cap", "force_norm"],
        "energy": -1.0,
        "force_norm": 0.016,
        "force_calls": 2485,
        "lammps_stop": "max iterations",
    }
    assert classify_stored(stored, 2500, 0.001) == "not_converged"
    assert classify_stored({**stored, "lammps_stop": "max force evaluations"}, 2500, 0.001) == "not_converged"
    assert classify_stored({**stored, "force_calls": 2500}, 2500, 0.001) == "cap"
    blocks = {
        **stored,
        "lammps_stop": "mixed",
        "replica_blocks": [[("max iterations", 9)], [("max force evaluations", 4)]],
    }
    assert classify_stored(blocks, 2500, 0.001) == "not_converged"
    raw = _record_from_raw(stored, 2500, 0.001, "neb/Pt-heptamer")
    assert raw["failure"] == "not_converged"
    assert raw["success"] is False


def test_a_finished_band_above_the_force_stop_is_not_converged():
    # rgsaddle plain NEB: the optimizer reports converged, the highest
    # image is finite, and the force is still above the problem stop.
    highest = {
        "success": False,
        "failure": "divergence",
        "reasons": ["force_norm"],
        "energy": -1774.8240973223333,
        "force_norm": 0.12729021406522067,
        "force_calls": 164,
        "converged": True,
        "candidate": "highest",
        "hessian_index": None,
    }
    assert classify_stored(highest, 10000, 0.001) == "not_converged"
    climbing = {
        **highest,
        "candidate": "climbing",
        "force_calls": 400,
        "force_norm": 0.016,
    }
    assert classify_stored(climbing, 2500, 0.001) == "not_converged"
    # The same facts through the central call, without a stored label.
    assert (
        classify_outcome(
            energy=-1774.82,
            force=0.127,
            calls=164,
            cap=10000,
            force_tol=0.001,
        )
        == "not_converged"
    )
    # Above the stop, a stored index is not the class. The force was not met.
    indexed = {**highest, "hessian_index": 0, "expected_index": 1}
    assert classify_stored(indexed, 10000, 0.001) == "not_converged"


def test_a_band_at_the_stop_with_the_wrong_index_is_wrong_stationary():
    stored = {
        "success": False,
        "failure": "divergence",
        "reasons": ["hessian_index"],
        "energy": -169.1848886465046,
        "force_norm": 0.0005943210204850857,
        "force_calls": 9342,
        "hessian_index": 0,
        "candidate": "highest",
        "converged": True,
    }
    assert classify_stored(stored, 10000, 0.001) == "wrong_stationary"
    assert (
        classify_outcome(
            energy=-169.18,
            force=0.00059,
            calls=9342,
            cap=10000,
            force_tol=0.001,
            hessian_index=2,
            expected_index=1,
        )
        == "wrong_stationary"
    )
    assert (
        classify_outcome(
            energy=-169.18,
            force=0.001,
            calls=100,
            cap=10000,
            force_tol=0.001,
            hessian_index=1,
            expected_index=1,
        )
        is None
    )


def test_own_stop_or_missing_mode_under_the_cap_is_not_converged():
    assert classify_failure(["force_norm"]) == "not_converged"
    assert classify_failure(["mode_overlap"]) == "not_converged"
    assert classify_failure(["solver"]) == "not_converged"
    assert classify_failure(["cap", "force_norm"]) == "cap"
    assert classify_failure(["divergence", "force_norm"]) == "divergence"


def test_explicit_force_met_does_not_recompare_the_stored_force():
    assert (
        classify_outcome(
            energy=1.0,
            force=5.0,
            calls=3,
            cap=10,
            force_tol=0.01,
            force_above=False,
            hessian_index=0,
            expected_index=0,
        )
        is None
    )


def test_stored_residual_stop_is_not_divergence():
    stored = {
        "success": False,
        "failure": "divergence",
        "reasons": ["force_norm"],
        "energy": 1.0,
        "force_norm": 0.5,
        "force_calls": 12,
    }
    assert classify_stored(stored, 100, 0.01) == "not_converged"


def test_a_measured_dissociation_stays_divergence():
    stored = {
        "success": False,
        "failure": "divergence",
        "reasons": ["divergence"],
        "dissociated": True,
        "energy": -1.0e-7,
        "force_norm": 1.0e-8,
        "force_calls": 1692,
    }
    assert classify_stored(stored, 2500, 0.001) == "divergence"
    plain = dict(stored)
    plain.pop("dissociated")
    assert classify_stored(plain, 2500, 0.001) == "not_converged"
    from optbench.board import _record_from_raw

    revised = _record_from_raw(stored, 2500, 0.001, "double-ended-saddle-search/lj38")
    assert revised["failure"] == "divergence"


def test_stored_lost_atoms_stay_divergence():
    stored = {
        "success": False,
        "failure": "divergence",
        "reasons": ["divergence"],
        "energy": 3.7e6,
        "force_norm": 1.97e8,
        "force_calls": 40,
        "error": "RuntimeError: LAMMPS exposed 36 atoms, the geometry has 38",
    }
    assert classify_stored(stored, 1000, 0.01) == "divergence"


def test_link_rejection_without_a_force_is_wrong_stationary():
    stored = {
        "success": False,
        "failure": "divergence",
        "reasons": ["not_connected", "duplicate"],
        "force_calls": 80,
        "hessian_index": 1,
    }
    assert classify_stored(stored, 1000, 0.01) == "wrong_stationary"


def test_a_mode_overlap_success_keeps_its_recorded_hessian():
    stored = {
        "success": True,
        "failure": None,
        "force_calls": 23,
        "energy": -167.0,
        "force_norm": 2.99,
        "hessian_index": 3,
        "expected_index": 1,
        "overlap": 0.997,
    }
    revised = apply_taxonomy(stored, 10000, 0.01)
    assert revised["success"] is True
    assert revised["failure"] is None


def test_a_force_stop_success_with_the_wrong_index_is_wrong_stationary():
    stored = {
        "success": True,
        "failure": None,
        "force_calls": 10,
        "energy": -1.0,
        "force_norm": 0.001,
        "hessian_index": 1,
        "expected_index": 0,
    }
    revised = apply_taxonomy(stored, 1000, 0.01)
    assert revised["failure"] == "wrong_stationary"
    assert revised["success"] is False


def test_board_keeps_a_mode_overlap_success_and_a_noise_success():
    mode = {
        "force_calls": 23,
        "success": True,
        "energy": -167.0,
        "force_norm": 2.99,
        "hessian_index": 3,
        "expected_index": 1,
        "overlap": 0.997,
    }
    revised = _record_from_raw(mode, 10000, 0.01, "lowest-eigenvector/lj38")
    assert revised["success"] is True
    assert "failure" not in revised
    noisy = {
        "force_calls": 365,
        "success": True,
        "energy": -164.0,
        "force_norm": 0.02,
        "sigma": 0.001,
        "hessian_index": 0,
        "expected_index": 0,
    }
    kept = _record_from_raw(noisy, 10000, 0.01, "minimization/lj38-noise")
    assert kept["success"] is True
    bare = _record_from_raw(noisy, 10000, 0.01, None)
    assert bare["failure"] == "not_converged"


def test_a_label_without_numbers_is_kept():
    stored = {"success": False, "failure": "divergence", "force_calls": 10}
    revised = apply_taxonomy(stored, 100, 0.01)
    assert revised["failure"] == "divergence"
    assert revised["success"] is False


def test_noise_success_uses_the_noise_stop():
    # LJ38 has 114 force components. At sigma 0.01 the stop is 3 * 0.01 * sqrt(114).
    stop = 3 * 0.01 * math.sqrt(114)
    assert stop > 0.2
    kept = apply_taxonomy(
        {"success": True, "force_calls": 10, "energy": -1.0, "force_norm": 0.2, "sigma": 0.01},
        1000,
        0.01,
        components=114,
    )
    assert kept["success"] is True
    assert kept["failure"] is None
    moved = apply_taxonomy(
        {"success": True, "force_calls": 10, "energy": -1.0, "force_norm": stop + 0.01, "sigma": 0.01},
        1000,
        0.01,
        components=114,
    )
    assert moved["failure"] == "not_converged"
    assert moved["success"] is False


def test_rescore_keeps_par2_when_only_the_label_changes():
    raw = {
        "force_calls": 12,
        "success": False,
        "failure": "divergence",
        "reasons": ["force_norm"],
        "energy": 1.0,
        "force_norm": 0.5,
    }
    old = legacy_record_from_raw(raw, 100, 0.01)
    new = _record_from_raw(raw, 100, 0.01)
    assert legacy_failure_of(old, 100) == "divergence"
    assert new["failure"] == "not_converged"
    assert new["success"] is False
    assert par2_mean([old], 100) == par2_mean([new], 100)
    assert par2_mean([old], 100) == score_v2([old], 100, resamples=3, seed=2)["par2_mean"]


def test_a_stored_success_with_a_nan_energy_becomes_divergence():
    raw = {"force_calls": 12, "success": True, "energy": float("nan"), "force_norm": float("nan")}
    old = legacy_record_from_raw(raw, 100, 0.01)
    new = _record_from_raw(raw, 100, 0.01)
    assert old["success"] is True
    assert new["failure"] == "divergence"
    assert new["success"] is False
    assert par2_mean([old], 100) != par2_mean([new], 100)
