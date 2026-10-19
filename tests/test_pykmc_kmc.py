"""pyKMC KMC reducers and the saddle endpoint record."""

import math
import tomllib
from pathlib import Path

import numpy as np

from optbench.engines.pykmc_search import _link, _section
from optbench.failures import apply_taxonomy
from optbench.pykmc_kmc import (
    ENDPOINT_SCHEMA,
    HTST,
    endpoint_record,
    reduce_htst,
    searches_to_launch,
)


def test_catalogue_batch_stops_at_the_stall_limit():
    assert searches_to_launch(0, 0, 0, 27, 20000, 8) == 8
    assert searches_to_launch(200, 160, 0, 27, 20000, 8) == 0
    assert searches_to_launch(200, 159, 0, 27, 20000, 8) == 1
    assert searches_to_launch(199, 160, 10, 27, 20000, 8) == 1
    assert searches_to_launch(10, 0, 27, 27, 20000, 8) == 0


def test_htst_cycles_match_the_archive_reducer():
    spec = {
        "barriers": [0.500, 0.500],
        "rate_ref": 1.0,
        "rate_accuracy": 0.999,
        "temperature": 700.0,
        "prefactor": 1.2e12,
        "kb": 8.6173324e-5,
    }
    # One 0.5 eV event at 700 K is far above the tiny reference rate,
    # so each new process id closes a rate cycle. The fourth search
    # starts a fresh cycle after the third search reset the sum. The
    # barrier cycle waits for both reference entries.
    rows = [
        {"force_calls": 10, "process_id": 1, "barrier": 0.5004},
        {"force_calls": 12, "process_id": None},
        {"force_calls": 8, "process_id": 2, "barrier": 0.4996},
        {"force_calls": 9, "process_id": 1, "barrier": 0.5004},
    ]
    score = reduce_htst(rows, spec)
    assert score["matched_barriers"] == 2
    assert score["barrier_cycles"] == 1
    assert score["barrier_jobs"] == 3
    assert score["barrier_force_calls"] == 30
    assert score["rate_cycles"] == 3
    assert score["rate_jobs"] == 4.0 / 3.0
    assert score["rate_call_list"] == [10, 20, 9]
    assert score["barrier_call_list"] == [30]
    rate = 1.2e12 * math.exp(-0.5004 / (8.6173324e-5 * 700.0))
    assert rate > spec["rate_ref"] * spec["rate_accuracy"]
    assert HTST["hTST/Pt"]["barrier_cycles"] == 13
    assert HTST["hTST/Pt"]["rate_cycles"] == 224
    assert HTST["hTST/Al"]["barrier_cycles"] == 639
    assert len(HTST["hTST/Pt"]["barriers"]) == 15


def test_endpoint_record_keeps_both_minima():
    record = endpoint_record(
        start="rep1-search1",
        search=1,
        repetition=1,
        minimum=[[0.0, 0.0, 0.0]],
        box=[1.0, 1.0, 1.0],
        free=[True],
        pbc=[True, True, False],
        endpoint_plus=[[0.1, 0.0, 0.0]],
        endpoint_minus=[[-0.1, 0.0, 0.0]],
    )
    assert record["schema"] == ENDPOINT_SCHEMA
    assert record["endpoint_plus"] != record["endpoint_minus"]
    empty = endpoint_record(
        start="rep1-search2",
        search=2,
        repetition=1,
        minimum=[[0.0, 0.0, 0.0]],
        box=[1.0, 1.0, 1.0],
        free=[True],
        pbc=[False, False, False],
        endpoint_plus=None,
        endpoint_minus=None,
    )
    assert empty["endpoint_plus"] is None
    assert empty["endpoint_minus"] is None


def test_link_stores_both_endpoints_and_does_not_invent_a_connection():
    def evaluate(coords, _box):
        point = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        return 0.5 * float(np.sum(point * point)), point.copy()

    record = {
        "success": True,
        "reasons": [],
        "failure": None,
        "force_calls": 4,
        "energy": 0.5,
        "positions": np.array([[1.0, 0.0, 0.0]]),
        "lowest_mode": [1.0, 0.0, 0.0],
        "start": "only",
    }
    out = _link(
        record,
        evaluator=evaluate,
        free=np.array([True]),
        box=np.array([10.0, 10.0, 10.0]),
        minimum=np.zeros((1, 3)),
        min_energy=0.0,
        force_norm=0.001,
        barrier_max=1.5,
        connect_rmsd=0.5,
        found=[],
        pbc=(False, False, False),
        tolerance=0.1,
    )
    assert out["endpoint_plus"] is not None
    assert out["endpoint_minus"] is not None
    assert len(out["endpoint_plus"]) == 1
    assert len(out["endpoint_minus"]) == 1
    assert out["endpoint_plus"] != out["endpoint_minus"] or out["eon_connected"] is False
    # Both sides fall into the same well, so the r2205 test rejects it.
    assert out["per_atom_max_plus"] < 0.1
    assert out["per_atom_max_minus"] < 0.1
    assert out["success"] is False
    assert out["eon_connected"] is False


def test_minimum_profile_is_the_published_push():
    section = _section({})
    assert section["ninit"] == 1
    assert section["push_mode"] == "rad"
    assert section["push_step_size"] == 0.4
    assert section["push_dist_thr"] == 3.0
    assert section["eigen_step_size"] == 0.25
    assert section["eigval_thr"] == -0.05
    assert section["lanczos_min_size"] == 12
    assert section["lanczos_max_size"] == 15
    assert section["nsmooth"] == 2
    assert section["nnewchance"] == 2
    pinned = _section({"pykmc": {"ninit": 0, "push_mode": "list", "push_dist_thr": 1.0}})
    assert pinned["ninit"] == 0
    assert pinned["push_mode"] == "list"
    assert pinned["push_dist_thr"] == 1.0
    assert pinned["eigen_step_size"] == 0.25


def test_templates_pin_the_push_profiles():
    root = Path(__file__).resolve().parents[1] / "templates" / "pykmc"
    saddle = tomllib.loads((root / "saddle-pt.toml").read_text(encoding="utf-8"))
    assert saddle["pykmc"]["ninit"] == 1
    assert saddle["pykmc"]["push_dist_thr"] == 3.0
    assert saddle["pykmc"]["lanczos_min_size"] == 12
    assert saddle["pykmc"]["lanczos_max_size"] == 15
    assert saddle["search"]["displace_magnitude"] == 0.0
    assert saddle["kernel"] == "morse_shift"
    assert "failure_cap" not in tomllib.loads((root / "htst-pt.toml").read_text(encoding="utf-8"))
    assert "failure_cap" not in tomllib.loads((root / "htst-al.toml").read_text(encoding="utf-8"))
    assert tomllib.loads((root / "htst-pt.toml").read_text(encoding="utf-8"))["kernel"] == "morse_shift"
    assert tomllib.loads((root / "aladatom.toml").read_text(encoding="utf-8"))["kmc"]["seeds"] >= 3
    assert tomllib.loads((root / "standard-aladatom.toml").read_text(encoding="utf-8"))["kmc"]["seeds"] >= 3
    smooth = tomllib.loads((root / "saddle-pt-smooth.toml").read_text(encoding="utf-8"))
    assert smooth["kernel"] == "morse"
    assert smooth["id"] == "pykmc/saddle-pt-smooth"
    guess = tomllib.loads((root / "ts-pt.toml").read_text(encoding="utf-8"))
    assert guess["pykmc"]["ninit"] == 0
    assert guess["pykmc"]["push_mode"] == "list"
    assert guess["pykmc"]["eigval_thr"] == -0.01
    assert guess["pykmc"]["nsmooth"] == 3
    assert guess["pykmc"]["lanczos_disp"] == 0.0005
    assert guess["pykmc"]["dmax"] == 6.0


def _accept_linked(record, *, force_tol, cap):
    """Success bit only. ``apply_taxonomy`` names the failure class."""
    row = dict(record)
    calls = float(row["force_calls"])
    energy = row.get("energy")
    force = row.get("force_norm")
    hessian = row.get("hessian_index")
    expected = 1 if row.get("expected_index") is None else int(row["expected_index"])
    finite_energy = isinstance(energy, (int, float)) and math.isfinite(float(energy))
    finite_force = isinstance(force, (int, float)) and math.isfinite(float(force))
    blown = (energy is not None and not finite_energy) or (force is not None and not finite_force)
    at_cap = cap is not None and math.isfinite(calls) and calls >= float(cap)
    at_stop = finite_force and float(force) <= float(force_tol)
    index_ok = hessian is not None and int(hessian) == expected
    if row.get("eon_connected") and at_stop and index_ok and finite_energy and not blown and not at_cap:
        row["success"] = True
    return apply_taxonomy(row, cap, force_tol)


def test_connected_index_one_is_a_success_and_a_clean_stop_is_not_divergence():
    connected = _accept_linked(
        {
            "force_calls": 907,
            "energy": -1769.8,
            "force_norm": 0.00099,
            "hessian_index": 1,
            "expected_index": 1,
            "eon_connected": True,
            "barrier": 1.51,
            "reasons": ["barrier"],
            "success": False,
            "failure": "divergence",
        },
        force_tol=0.001,
        cap=20000,
    )
    assert connected["success"] is True
    assert connected["failure"] is None
    duplicate = _accept_linked(
        {
            "force_calls": 1100,
            "energy": -1770.0,
            "force_norm": 0.0004,
            "hessian_index": 1,
            "expected_index": 1,
            "eon_connected": True,
            "barrier": 0.62,
            "reasons": ["duplicate", "barrier"],
            "success": False,
            "failure": "divergence",
        },
        force_tol=0.001,
        cap=20000,
    )
    assert duplicate["success"] is True
    missed = _accept_linked(
        {
            "force_calls": 1261,
            "energy": -1769.9,
            "force_norm": 0.00094,
            "hessian_index": 1,
            "expected_index": 1,
            "eon_connected": False,
            "reasons": ["not_connected"],
            "success": False,
            "failure": "divergence",
        },
        force_tol=0.001,
        cap=20000,
    )
    assert missed["success"] is False
    assert missed["failure"] == "wrong_stationary"
    clean = _accept_linked(
        {
            "force_calls": 800,
            "energy": None,
            "force_norm": None,
            "hessian_index": None,
            "eon_connected": False,
            "reasons": ["divergence"],
            "error": "ErrorType.EVENT_NOT_FOUND: No event found",
            "success": False,
            "failure": "divergence",
        },
        force_tol=0.001,
        cap=20000,
    )
    assert clean["failure"] == "crash"
    assert clean["failure"] != "divergence"
    capped = _accept_linked(
        {
            "force_calls": 10000,
            "energy": -1.0,
            "force_norm": 0.5,
            "hessian_index": None,
            "eon_connected": False,
            "reasons": ["cap"],
            "success": False,
            "failure": "cap",
        },
        force_tol=0.001,
        cap=10000,
    )
    assert capped["failure"] == "cap"
