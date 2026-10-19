"""SoftSaddle templates and the compiled eigen-solvers."""

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.softsaddle_run import (
    _search_seed,
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
        current, current_order = hypersphere(repetition)
        assert current.shape == (5000, 21)
        assert len(np.unique(current_order)) == 5000
        assert np.allclose(np.linalg.norm(current, axis=1), 1.0, atol=1e-8)
        assert hypersphere_sha(repetition) == _HYPER_SHA[repetition]
        assert (int(current_order[0]), int(current_order[1])) == _HYPER_FIRST[repetition]
        if repetition > 1:
            assert not np.array_equal(current, previous)
        previous = current


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
