"""EZGA leaves on a hit and on the force-call cap.

The engine catches Exception. These tests fail if the cap or a hit is
raised as Exception, because the stand-in generation loop swallows it
and keeps going.
"""

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.counter import ForceCallCap
from optbench.potentials import binary_lennard_jones
from optbench.runners.ezga_go import Budget, Hit, _Halt, load_starts, observe_point

ROOT = Path(__file__).resolve().parents[1]


def _swallowing_loop(call, generations: int) -> int:
    """Stand-in for EZGA's generation loop. It catches Exception only."""
    completed = 0
    for _generation in range(generations):
        completed += 1
        try:
            call()
        except Exception:
            continue
        else:
            continue
    return completed


def test_force_call_cap_is_not_swallowed():
    budget = Budget(lambda coords: (0.0, np.zeros_like(np.asarray(coords, dtype=np.float64))), cap=3)
    step = {"n": 0}

    def call() -> None:
        step["n"] += 1
        point = np.full(9, float(step["n"]))
        observe_point(budget, point, fmax=1.0)

    with pytest.raises(_Halt):
        _swallowing_loop(call, 30)
    assert budget.calls == 3
    assert not issubclass(_Halt, Exception)


def test_hit_is_not_swallowed():
    def evaluate(coords):
        return -1.0, np.zeros_like(np.asarray(coords, dtype=np.float64))

    budget = Budget(evaluate, cap=100, target=-1.0, window=0.01)
    point = np.zeros(9)

    def call() -> None:
        observe_point(budget, point, fmax=0.05)

    with pytest.raises(_Halt):
        _swallowing_loop(call, 30)
    assert budget.hit is True
    assert budget.calls == 1
    assert issubclass(Hit, Exception)
    assert issubclass(ForceCallCap, Exception)


def test_templates_use_the_problem_cap():
    found = [row for row in list_templates(ROOT) if row.get("engine") == "ezga"]
    assert {row["id"] for row in found} == {
        "ezga/lj38",
        "ezga/lj75",
        "ezga/lj98",
        "ezga/lj150",
        "ezga/binary",
    }
    for row in found:
        problem = resolve_problem(ROOT, row["problem"])
        assert int(row["stop"]["max_force_calls"]) == 2000000
        assert _check_stop(problem, row) == 2000000
        assert row["code_version"] == "1.2.31"
        assert row["commit"] == "1395e8de01f003e40a3352987377d5f863a2e388"
        assert row["tuning"] == "default"


def test_start_sets_match_the_problem_files():
    expected = {
        "globalopt/lj38": 100,
        "globalopt/lj75": 200,
        "globalopt/lj98": 100,
        "globalopt/lj150": 100,
        "globalopt/lj100-binary": 100,
    }
    for ident, count in expected.items():
        problem = resolve_problem(ROOT, ident)
        starts = load_starts(ROOT, problem)
        assert len(starts) == count
        assert starts[0]["index"] == 0


def test_binary_dimer_at_the_arithmetic_sigma_is_bound():
    sigma = 0.5 * (1.0 + 1.3)
    distance = sigma * 2.0 ** (1.0 / 6.0)
    coords = [0.0, 0.0, 0.0, distance, 0.0, 0.0]
    energy, gradient = binary_lennard_jones(coords, [0, 1], cutoff=20.0)
    assert energy == pytest.approx(-1.0, abs=1e-8)
    assert np.linalg.norm(gradient) == pytest.approx(0.0, abs=1e-6)
