"""NLopt on the harness counter.

The quadratic bowl has no archive. The Lennard-Jones start is cluster 0
from the problem archive, with the LJ38 step guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _nlopt():
    pytest.importorskip("numpy")
    from optbench.runners.nlopt_min import load_nlopt

    try:
        module, _built = load_nlopt()
    except ImportError as exc:
        pytest.skip(str(exc))
    return module


def test_quadratic_bowl_stops_for_every_gradient_method():
    import numpy as np

    from optbench.runners.nlopt_min import ALGORITHMS, drive

    _nlopt()

    def evaluate(coords):
        x = float(coords[0])
        y = float(coords[1])
        energy = (x - 1.0) ** 2 + 10.0 * (y + 2.0) ** 2
        gradient = np.array([2.0 * (x - 1.0), 20.0 * (y + 2.0)], dtype=np.float64)
        return energy, gradient

    def stop(_point, _energy, _gradient, norm: float) -> bool:
        return norm <= 1e-6

    for algorithm in ALGORITHMS:
        result = drive(
            np.array([0.0, 0.0]),
            evaluate,
            algorithm=algorithm,
            step="none",
            maxstep=None,
            stop=stop,
            cap=200,
        )
        point = result["point"]
        assert result["stopped"], algorithm
        assert result["error"] == "", algorithm
        assert abs(point[0] - 1.0) < 1e-4, algorithm
        assert abs(point[1] + 2.0) < 1e-4, algorithm


def test_scaled_lbfgs_minimizes_one_lj38_start():
    from optbench.campaign import resolve_problem
    from optbench.potentials import lennard_jones
    from optbench.runners.nlopt_min import _minimize_job
    from optbench.runners.rgmin_min import _load_geometries

    _nlopt()
    problem = resolve_problem(ROOT, "minimization/lj38")
    name, coords, _box = _load_geometries(ROOT, problem, None, Path("/tmp/optbench-nlopt-geo"), Path("/tmp/unused.dat"))[0]
    record = _minimize_job(
        {
            "index": 0,
            "start": name,
            "coords": coords.tolist(),
            "box": None,
            "kind": "lj",
            "algorithm": "LD_LBFGS",
            "step": "scale",
            "maxstep": 0.2,
            "gtol": 0.01,
            "cap": 10000,
        }
    )
    assert record["counter"] == "harness"
    assert record["success"] is True
    assert record["failure"] is None
    assert record["force_calls"] < 10000
    assert record["force_norm"] <= 0.01
    assert record["hessian_index"] == 0
    assert record["energy"] < -100.0
    energy, gradient = lennard_jones(coords)
    assert energy > record["energy"]
    assert sum(float(value) * float(value) for value in gradient) > 1.0
