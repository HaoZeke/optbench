"""GMIN start judging, kernels, and start files on the shared problem window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import resolve_problem
from optbench.potentials import binary_lennard_jones
from optbench.runners.gmin_go import (
    _classify,
    evaluator,
    load_geometries,
    potential_spec,
    reference_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]


def _job(cap: int = 100) -> dict:
    return {"start": "coords.1", "index": 0, "cap": cap, "reference": -10.0, "tolerance": 0.001, "replicas": 1}


def _snapshot(**values) -> dict:
    base = {"calls": 50, "cap_hit": False, "errors": 0, "lowest_energy": -9.5, "hit_calls": None, "hit_energy": None}
    base.update(values)
    return base


def test_success_counts_the_first_evaluation_inside_the_window():
    record = _classify(_job(), _snapshot(calls=60, hit_calls=42, hit_energy=-10.0005, lowest_energy=-10.0005), 0, "")
    assert record["success"] is True
    assert record["force_calls"] == 42
    assert record["total_calls"] == 60


def test_no_hit_at_the_cap_is_a_cap_failure():
    record = _classify(_job(), _snapshot(calls=100, cap_hit=True), 2, "")
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert record["force_calls"] == 100


def test_a_timeout_or_nonzero_exit_before_the_cap_is_a_crash():
    timed_out = _classify(_job(), _snapshot(hit_calls=10, hit_energy=-10.0), -9, "timeout")
    assert timed_out["success"] is False
    assert timed_out["failure"] == "crash"
    exited = _classify(_job(), _snapshot(), 1, "")
    assert exited["failure"] == "crash"


def test_a_hit_at_the_cap_count_is_not_a_success():
    record = _classify(_job(cap=100), _snapshot(calls=100, cap_hit=True, hit_calls=100, hit_energy=-10.0), 2, "")
    assert record["success"] is False


def test_binary_problem_uses_the_42_58_kernel():
    problem = resolve_problem(ROOT, "globalopt/lj100-binary")
    spec = potential_spec(problem)
    assert spec["n_A"] == 42 and spec["n_B"] == 58
    assert spec["sigma_AA"] == 1.0 and spec["sigma_BB"] == 1.3
    rng = np.random.default_rng(7)
    pos = rng.normal(scale=2.0, size=(100, 3))
    energy, grad = evaluator(spec)(pos)
    species = [0] * 42 + [1] * 58
    expected, expected_grad = binary_lennard_jones(pos, species, sigma_bb=1.3)
    assert energy == pytest.approx(expected)
    assert np.allclose(grad, expected_grad)
    points_energy, rms = reference_evaluation(ROOT, 100, spec)
    assert points_energy <= -604.796307 + 0.001
    assert rms < 1.0e-3


def test_binary_starts_are_the_100_con_files(tmp_path):
    problem = resolve_problem(ROOT, "globalopt/lj100-binary")
    loaded = load_geometries(ROOT, problem, tmp_path)
    assert len(loaded) == 100
    assert loaded[0][0] == "coords.0"
    assert loaded[-1][0] == "coords.99"
    assert all(pos.shape == (100, 3) for _name, pos in loaded)
    assert max(float(np.max(np.abs(pos))) for _name, pos in loaded) < 10.0
