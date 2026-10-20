"""SoftSaddle templates and the compiled eigen-solvers."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.connection import ProcessCatalog
from optbench.endpoints import EndpointArchive, archive_error
from optbench.softsaddle_run import (
    _apply_novelty,
    _attach_endpoint,
    _depth_snapshot,
    _radius_update,
    _records_carry_endpoints,
    _search_seed,
    _starts_by_repetition,
    hypersphere,
    hypersphere_sha,
    library_path,
    matlab_benchmark,
    matlab_fraction,
    run_mode_point,
)

ROOT = Path(__file__).resolve().parents[1]


def test_softsaddle_templates_match_the_registry_pin():
    rows = [row for row in list_templates(ROOT) if str(row["id"]).startswith("softsaddle/")]
    assert {row["id"] for row in rows} == {
        "softsaddle/mode-lanczos",
        "softsaddle/mode-lanczos-tuned",
        "softsaddle/mode-davidson",
        "softsaddle/mode-davidson-tuned",
        "softsaddle/saddle-davidson-sm50",
        "softsaddle/saddle-lanczos-sm0",
        "softsaddle/saddle-lanczos-sm50",
    }
    for row in rows:
        assert row["engine"] == "softsaddle"
        assert row["code"] == "softsaddle"
        assert row["commit"] == "5a00aeede06fd822ef302624ec4c9bc7889d21fc0b5804b701a5785695c68d16"
        _check_stop(resolve_problem(ROOT, row["problem"]), row)


_HYPER_SHA = {
    1: "eb78f6d9cde1e9aae308e3f8ecec777fdb655e3432e356c9da339b950bd3267a",
    2: "aa994b50de6bbe02c6ea8ee0cdf4e47906913f1e60be91b58cdd4468010197c6",
    3: "99d59ea0e6075d0f852917d60f480a922572268c98eb01fb273aaf70ea950157",
    4: "e2aca928c2637c2487bc41c3025d5ac58b356916b29ee14978d2d302a96fc9d4",
    5: "efa0176136278fb2a4b353b2aca3b35c966265f058bf358dc9592ae32ff6a345",
    6: "27d3796fcc5a47df606cdc6f92a6b3df9dd3245a4e66489f12965e40cac165bd",
    7: "f043cb68a3c6e2fafadf5587395ab301294fccde8f048aad7cee0f24eee8ece3",
    8: "0a505b66a05a11387a866df88b135faa2573819ffb1a693161cc1de0aaeb3159",
    9: "1ce1f597259e75e1360337de5e7249ccbe186666583b62e667bc45e2fcf1c801",
    10: "6efd3373f4b6a8048137be6bec51b760da8b7cab9486f1be79ce77612a1a69ba",
}
_HYPER_FIRST = {
    1: (3013, 1358),
    2: (1519, 1208),
    3: (3470, 3396),
    4: (1550, 657),
    5: (344, 14),
    6: (4622, 393),
    7: (3000, 2620),
    8: (3448, 1310),
    9: (4945, 4223),
    10: (669, 70),
}


def test_hypersphere_is_the_sorted_5000_point_set():
    vectors, order = hypersphere()
    assert vectors.shape == (5000, 21)
    assert order.shape == (5000,)
    assert len(np.unique(order)) == 5000
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)
    pair = vectors[int(order[0])] - vectors[int(order[1])]
    assert float(np.linalg.norm(pair)) == pytest.approx(1.924429334848153, rel=1e-9)
    previous = vectors
    for repetition in range(1, 11):
        directions, direction_order = hypersphere(repetition)
        assert directions.shape == (5000, 21)
        assert len(np.unique(direction_order)) == 5000
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-8)
        assert hypersphere_sha(repetition) == _HYPER_SHA[repetition]
        assert (int(direction_order[0]), int(direction_order[1])) == _HYPER_FIRST[repetition]
        if repetition > 1:
            assert not np.array_equal(directions, previous)
        previous = directions


def test_later_repetitions_use_a_shifted_seed():
    assert _search_seed(1, 0) == 20261016
    assert _search_seed(1, 4999) == 20266015
    assert _search_seed(2, 0) == 20361016
    assert _search_seed(10, 4999) == 21166015


def test_matlab_benchmark_uses_the_website_reduction():
    reps = [
        {"force_calls": 1000 + index, "unique_saddles": 20, "connected": 4, "total_searches": 100}
        for index in range(10)
    ]
    pooled = matlab_benchmark(reps)
    assert pooled["unique_saddles"] == 4
    assert pooled["total_searches"] == 100
    assert pooled["fraction_connected"] == pytest.approx(matlab_fraction(20, 4, 100))
    assert pooled["force_calls"] == pytest.approx(1004.5)
    assert pooled["force_calls_mean_error"] == pytest.approx(
        float(np.std([1000 + index for index in range(10)], ddof=1))
    )


@pytest.mark.parametrize("solver", ["lanczos", "davidson"])
def test_quadratic_lowest_mode(tmp_path: Path, solver: str):
    if library_path(solver) is None:
        pytest.skip(f"{solver} library is not built")

    def evaluate(flat):
        x, y, z = np.asarray(flat, dtype=np.float64)
        energy = float(-x * x + 2.0 * y * y + 3.0 * z * z)
        gradient = np.array([-2.0 * x, 4.0 * y, 6.0 * z], dtype=np.float64)
        return energy, gradient

    record = run_mode_point(
        {
            "name": "quadratic",
            "evaluate": evaluate,
            "x": np.array([0.2, 0.3, -0.1], dtype=np.float64),
            "reference": np.array([1.0, 0.0, 0.0]),
            "cap": 10000,
            "overlap": 0.99,
            "solver": solver,
            "seed": 20261015,
            "work": str(tmp_path / solver),
            "settings": {
                "solver": solver,
                "seed_mode": "gradient",
                "iteration_limit": 8,
                "tolerance": 1e-8,
                "finite": 1e-6,
            },
        }
    )
    assert record["success"] is True
    assert record["overlap"] >= 0.99
    assert record["curvature"] == pytest.approx(-2.0, rel=1e-3, abs=1e-3)
    assert record["force_calls"] < 10000
    assert record["failure"] is None


def test_radius_keeps_when_linked_uniques_clear_the_efficiency():
    """Scheme 2 keeps the radius when new linked saddles are at least 20%.

    The 1.5 eV successes are the stop counter. A batch can clear 0.2 on
    the link test while most of those saddles sit above 1.5 eV.
    """
    batch = []
    for _index in range(25):
        batch.append(
            {
                "success": False,
                "reasons": ["not_connected", "barrier"],
                "per_atom_max_plus": 1.6,
                "per_atom_max_minus": 2.4,
                "lambda_distance": 4.0,
                "saddle_distance": 5.0,
            }
        )
    for index in range(6):
        batch[index]["per_atom_max_minus"] = 0.001
        batch[index]["reasons"] = [] if index < 3 else ["barrier"]
        batch[index]["success"] = index < 3
    settings = {"batch": 25, "mix_radius": True, "keep_efficiency": 0.2}
    assert _radius_update(0.1, batch, [], [], settings) == 0.1
    for index in range(2):
        batch[index]["per_atom_max_minus"] = 1.7
        batch[index]["reasons"] = ["not_connected"]
        batch[index]["success"] = False
    moved = _radius_update(0.1, batch, [], [], settings)
    assert moved != 0.1
    assert moved == pytest.approx(0.5 * (4.0 + 0.1))


def test_radius_counts_catalog_good_saddles():
    """find_repeat ``good`` is the efficiency count, including a high barrier."""
    batch = []
    for _index in range(25):
        batch.append(
            {
                "good": False,
                "success": False,
                "reasons": ["not_connected"],
                "per_atom_max_plus": 1.0,
                "per_atom_max_minus": 1.0,
                "lambda_distance": 9.0,
                "saddle_distance": 9.0,
            }
        )
    for index in range(6):
        batch[index]["good"] = True
        batch[index]["lambda_distance"] = 4.0
        batch[index]["reasons"] = [] if index < 3 else ["barrier"]
        batch[index]["success"] = index < 3
    settings = {"batch": 25, "mix_radius": True, "keep_efficiency": 0.2}
    assert _radius_update(0.1, batch, [], [], settings) == 0.1
    for index in range(3):
        batch[index]["good"] = False
    moved = _radius_update(0.1, batch, [], [], settings)
    assert moved == pytest.approx(0.5 * (4.0 + 0.1))


def test_depth_snapshot_counts_the_link_before_the_barrier_stop(tmp_path: Path):
    rows = [
        {
            "start": "rep1-search1",
            "repetition": 1,
            "success": True,
            "reasons": [],
            "per_atom_max_plus": 0.001,
            "per_atom_max_minus": 1.5,
            "force_calls": 10,
        },
        {
            "start": "rep1-search10",
            "repetition": 1,
            "success": False,
            "reasons": ["duplicate"],
            "per_atom_max_plus": 0.001,
            "per_atom_max_minus": 1.5,
            "force_calls": 40,
        },
        {
            "start": "rep1-search2",
            "repetition": 1,
            "success": False,
            "reasons": ["barrier"],
            "per_atom_max_plus": 1.5,
            "per_atom_max_minus": 0.001,
            "force_calls": 20,
        },
        {
            "start": "rep1-search3",
            "repetition": 1,
            "success": False,
            "reasons": ["not_connected"],
            "per_atom_max_plus": 0.001,
            "per_atom_max_minus": 0.001,
            "force_calls": 30,
        },
    ]
    path = tmp_path / "starts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ordered = _starts_by_repetition(path)[1]
    assert [row["start"] for row in ordered] == [
        "rep1-search1",
        "rep1-search2",
        "rep1-search3",
        "rep1-search10",
    ]
    snap = _depth_snapshot(ordered, 4)
    assert snap["below"] == 1
    assert snap["linked"] == 2
    assert snap["both"] == 1
    assert snap["novel"] == 3
    assert snap["calls"] == 100
    assert _depth_snapshot(ordered, 5) is None


def test_depth_snapshot_prefers_catalog_good():
    rows = [
        {
            "start": "rep1-search1",
            "repetition": 1,
            "success": False,
            "good": True,
            "accepted_novel": True,
            "reasons": ["barrier"],
            "per_atom_max_plus": 1.0,
            "per_atom_max_minus": 1.0,
            "force_calls": 10,
        },
        {
            "start": "rep1-search2",
            "repetition": 1,
            "success": False,
            "good": False,
            "accepted_novel": True,
            "reasons": ["not_connected"],
            "per_atom_max_plus": 0.001,
            "per_atom_max_minus": 0.001,
            "force_calls": 20,
        },
    ]
    snap = _depth_snapshot(rows, 2)
    assert snap["linked"] == 1
    assert snap["novel"] == 2
    assert snap["both"] == 1
    assert snap["below"] == 0


def test_novelty_uses_find_repeat():
    """A one-atom hop is a new process. A 0.05 angstrom move at the same barrier is not.

    Cartesian RMSD of the hop is under 0.5. A barrier 0.02 eV away is a
    different process even when the saddle geometry matches. A duplicate
    is divergence: the audit taxonomy sends find_repeat there.
    """
    count = 175
    reactant = np.zeros((count, 3))
    hop = reactant.copy()
    hop[0, 0] = 2.71
    rmsd = float(np.sqrt(np.mean(np.sum((hop - reactant) ** 2, axis=1))))
    assert rmsd < 0.5
    catalog = ProcessCatalog(None, tolerance=0.1, barrier_max=1.5)
    found: list = []
    judged = {
        "reasons": [],
        "success": True,
        "novel_candidate": True,
        "saddle_x": hop.reshape(-1),
        "barrier": 0.60,
    }
    out = _apply_novelty(judged, catalog, found, None, 0.1)
    assert out["success"] is True
    assert out["good"] is True
    assert out["failure"] is None
    assert catalog.unique == 1
    assert catalog.qualifying == 1
    shifted = {
        "reasons": [],
        "success": True,
        "novel_candidate": True,
        "saddle_x": hop.reshape(-1),
        "barrier": 0.62,
    }
    second = _apply_novelty(shifted, catalog, found, None, 0.1)
    assert second["good"] is True
    assert second["success"] is True
    assert catalog.unique == 2
    near = hop.copy()
    near[0, 0] = 2.76
    duplicate = _apply_novelty(
        {
            "reasons": [],
            "success": True,
            "novel_candidate": True,
            "saddle_x": near.reshape(-1),
            "barrier": 0.605,
        },
        catalog,
        found,
        None,
        0.1,
    )
    assert duplicate["success"] is False
    assert duplicate["good"] is False
    assert "duplicate" in duplicate["reasons"]
    assert duplicate["failure"] == "divergence"
    assert catalog.unique == 2
    high = _apply_novelty(
        {
            "reasons": [],
            "success": True,
            "novel_candidate": True,
            "saddle_x": hop.reshape(-1),
            "barrier": 1.8,
        },
        catalog,
        found,
        None,
        0.1,
    )
    assert high["good"] is True
    assert high["success"] is False
    assert "barrier" in high["reasons"]
    assert high["failure"] == "divergence"
    assert catalog.unique == 3
    assert catalog.qualifying == 2


def test_attach_endpoint_stamps_stored_and_not_minimized(tmp_path: Path):
    archive = EndpointArchive(tmp_path / "softsaddle-2019.04-saddle-demo.endpoints.npz", 2)
    archive.reset(np.array([19.2088, 19.0118, 30.0]), np.array([0, 4]))
    reactant = np.zeros((2, 3))
    judged = {
        "endpoint_plus": reactant + np.array([2.71, 0.0, 0.0]),
        "endpoint_minus": reactant + np.array([0.05, 0.0, 0.0]),
    }
    stored = {"start": "rep1-search1"}
    assert _attach_endpoint(archive, reactant, judged, stored) == "stored"
    assert "endpoint_plus" not in judged
    skipped = {"start": "rep1-search2"}
    assert _attach_endpoint(archive, reactant, {}, skipped) == "not_minimized"
    non_finite = {
        "endpoint_plus": reactant,
        "endpoint_minus": np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
    missing = {"start": "rep1-search3"}
    assert _attach_endpoint(archive, reactant, non_finite, missing) == "not_minimized"
    archive.save()
    rows = [stored, skipped, missing]
    assert archive_error(archive.path, rows, archive.path.name) is None
    assert _records_carry_endpoints(rows) is True
    assert _records_carry_endpoints([{**stored, "endpoint_status": "guessed"}]) is False
    assert [row["endpoint_index"] for row in rows] == [0, 1, 2]


_BANNED_WORDING = re.compile(
    r"reproduced|paper baseline|not_rerun|\blegacy\b|\bcurrent\b|\bband\b|\binside\b|\boutside\b",
    re.IGNORECASE,
)


def test_repetition_scores_ignore_the_search_log(tmp_path: Path):
    from optbench.board import _cap_of, _load_group_records, _score_records
    from optbench.dat import parse_lenient

    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    assert problem["potential"]["pair_style"] == "eon-morse"
    assert problem["goal"]["max_force_calls"] == 100_000_000
    assert problem["goal"]["max_searches"] == 5000
    assert "max_force_calls" not in (problem.get("convergence") or {})
    assert _cap_of(problem, {"failure_cap": 20_000_000}, {"cap": "5064"}) == 20_000_000
    assert _cap_of(problem, {"stop": {"max_force_calls": 10000}}, {}) == 10000
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    dat = campaign / "demo.dat"
    dat.write_text(
        "\n".join(
            [
                "# rep 1 force_calls 1000 connected 27 total_searches 10",
                "# rep 2 force_calls 2000 connected 26 total_searches 5000",
                "score_unit repetition",
                "goal 27",
                "max_searches 5000",
                "cap 5064",
                "problem saddle-search/pt-island",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (campaign / "demo.starts.jsonl").write_text(
        json.dumps({"force_calls": 10, "success": True, "failure": None, "start": "search"}) + "\n",
        encoding="utf-8",
    )
    fields, _duplicates = parse_lenient(dat.read_text(encoding="utf-8"))
    cap = _cap_of(problem, None, fields)
    assert cap == 100_000_000
    records, source = _load_group_records(
        {"dat": dat, "starts_jsonl": campaign / "demo.starts.jsonl"},
        cap,
        None,
    )
    assert source == "repetitions"
    assert [row["start"] for row in records] == ["rep1", "rep2"]
    summary = _score_records(records, cap)
    assert summary["n"] == 2
    assert summary["successes"] == 1
    assert summary["n_cap"] == 1
    assert summary["par2_mean"] == (1000 + 2 * 100_000_000) / 2


def test_published_saddle_rows_score_ten_repetitions():
    from optbench.board import _cap_of, _campaign_groups, _load_group_records, _score_records
    from optbench.dat import parse_lenient

    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    groups = _campaign_groups(ROOT / "results" / "campaign")
    expected = {
        "softsaddle-2019.04-saddle-davidson-sm50": (9, 20837407.1, 1021892.5, 932781.0),
        "softsaddle-2019.04-saddle-lanczos-sm0": (9, 21347760.7, 1612643.2, 1617334.0),
        "softsaddle-2019.04-saddle-lanczos-sm50": (7, 60964746.8, 1618641.9, 1732799.5),
    }
    for stem, (hits, par2, mean_calls, median_calls) in expected.items():
        group = groups[stem]
        fields, _duplicates = parse_lenient(group["dat"].read_text(encoding="utf-8"))
        cap = _cap_of(problem, None, fields)
        records, source = _load_group_records(group, cap, None)
        summary = _score_records(records, cap)
        assert source == "repetitions"
        assert summary["n"] == 10
        assert summary["successes"] == hits
        assert summary["n_cap"] == 10 - hits
        assert summary["par2_mean"] == pytest.approx(par2)
        assert summary["mean_calls"] == pytest.approx(mean_calls)
        assert summary["median_calls"] == pytest.approx(median_calls)
        assert cap == 100_000_000


def test_softsaddle_prose_states_the_gap_as_numbers():
    paths = [
        ROOT / "optbench" / "softsaddle_run.py",
        ROOT / "results" / "campaign" / "softsaddle-pt-island-repetitions.txt",
    ]
    for path in paths:
        match = _BANNED_WORDING.search(path.read_text(encoding="utf-8"))
        assert match is None, f"{path.name} has {match.group(0)!r}"
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    version2 = problem["scoring"]["v2"]
    for key in ("cost", "success", "note"):
        match = _BANNED_WORDING.search(version2[key])
        assert match is None, f"scoring.v2.{key} has {match.group(0)!r}"
    assert "eon-morse" in version2["note"]
    assert "morse/smooth/linear" not in version2["note"]
