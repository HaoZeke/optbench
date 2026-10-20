"""SoftSaddle templates and the compiled eigen-solvers."""

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.softsaddle_run import (
    _apply_novelty,
    _radius_update,
    _rep_comment,
    _search_seed,
    _summary_complete,
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
    1: "8fb181df53e977d267c46dbfd720e3dbd830f1d4d0c43ef24fb0b430c3084007",
    2: "dcb3f487e20f4abeb65e00bdd24e2476e0f7c25f0b4440e71f41097213370f9e",
    3: "18c61a78ab7c03d3260b483c85585cc2e7b18c8ebe6d52a5177ad3c0dc600bc4",
    4: "c6679971f4ce5fb0febe57dbde2233ca4ebf9e139fd079e88f119158c94f2ff3",
    5: "b557aa1f14ec87b4cfddd15a9ef6719d57614bb9c3bdbd1adfb99ab4ff1c57b9",
    6: "7468b822556ba3e0c990cfa17a3bc467b05b01ab4d8be9977807a51005026f73",
    7: "05d5e91fa3c47d96017f198f4aeae65312b15e93a5bad0bf8d4d5056d5dfa9cc",
    8: "109b6b113a757d66920f1048971248f8f6530ea30dddb9ec184a42869290ac6c",
    9: "2a3def70386aaf105d20423809d507d84b4debcdefe2070a25ff8ea8599121fb",
    10: "4af27df930ca3fed286d585f143543f857a86ee24818bef6f842998cb25e0480",
}
_HYPER_FIRST = {
    1: (1560, 1243),
    2: (4526, 3896),
    3: (2964, 950),
    4: (4770, 4385),
    5: (2192, 1730),
    6: (4924, 720),
    7: (5258, 2955),
    8: (4107, 440),
    9: (5094, 1047),
    10: (4868, 877),
}


def test_hypersphere_is_the_sorted_5500_point_set():
    vectors, order = hypersphere()
    assert vectors.shape == (5500, 21)
    assert order.shape == (5500,)
    assert len(np.unique(order)) == 5500
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)
    pair = vectors[int(order[0])] - vectors[int(order[1])]
    assert float(np.linalg.norm(pair)) == pytest.approx(1.9780898390727164, rel=1e-9)
    previous = vectors
    for repetition in range(1, 11):
        current, current_order = hypersphere(repetition)
        assert current.shape == (5500, 21)
        assert len(np.unique(current_order)) == 5500
        assert np.allclose(np.linalg.norm(current, axis=1), 1.0, atol=1e-8)
        assert hypersphere_sha(repetition) == _HYPER_SHA[repetition]
        assert (int(current_order[0]), int(current_order[1])) == _HYPER_FIRST[repetition]
        if repetition > 1:
            assert not np.array_equal(current, previous)
        previous = current


def test_later_repetitions_use_a_shifted_seed():
    assert _search_seed(1, 0) == 20261016
    assert _search_seed(1, 5499) == 20266515
    assert _search_seed(2, 0) == 20361016
    assert _search_seed(10, 5499) == 21166515


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


def test_qualifying_is_the_stop_and_connected_is_every_linked_saddle():
    assert _summary_complete({"connected": 40, "qualifying": 10, "total_searches": 100}, 5000, 27) is False
    assert _summary_complete({"connected": 40, "qualifying": 27, "total_searches": 100}, 5000, 27) is True
    assert _summary_complete({"connected": 27, "total_searches": 100}, 5000, 27) is True
    comment = _rep_comment(1, [{"force_calls": 10}, {"force_calls": 5}], 4, 2, 1, 3, True, 7, "abc")
    parsed_unique = 4
    parsed_connected = 2
    assert "qualifying 1" in comment
    assert "unique_saddles 4" in comment
    assert "connected 2" in comment
    assert matlab_fraction(parsed_unique, parsed_connected, 2) == pytest.approx(0.0)


def test_novelty_uses_per_atom_distance_and_the_radius_uses_paper_links():
    cell = np.diag([20.0, 20.0, 30.0])
    saddle = np.zeros(6)
    found: list[tuple[np.ndarray, float]] = []
    first = {
        "reasons": [],
        "success": False,
        "novel_candidate": True,
        "deferred": True,
        "barrier": 0.4,
        "saddle_x": saddle,
    }
    accepted = _apply_novelty(first, found, cell)
    assert accepted["accepted_novel"] is True
    assert accepted["failure"] is None
    assert len(found) == 1
    repeat = _apply_novelty(dict(first), found, cell)
    assert repeat.get("accepted_novel") is not True
    assert "duplicate" in repeat["reasons"]
    shifted = dict(first)
    shifted["saddle_x"] = saddle + np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    shifted["barrier"] = 0.4
    other = _apply_novelty(shifted, found, cell)
    assert other["accepted_novel"] is True
    radius = _radius_update(
        0.1,
        [
            {"paper_connected": True, "success": False, "lambda_distance": 0.4},
            {"paper_connected": False, "success": True, "lambda_distance": 9.0},
        ],
        [],
        [],
        {"batch": 25, "mix_radius": False, "keep_efficiency": 0.2},
    )
    assert radius == pytest.approx(0.4)


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
