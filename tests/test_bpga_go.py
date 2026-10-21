"""BPGA and PyBCGA on the harness Lennard-Jones potential."""

import os
from pathlib import Path

import numpy as np

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.counter import HarnessCounter
from optbench.runners.bpga_go import (
    QUENCH_FORCE,
    QUENCH_MAXFUN,
    _judge,
    _load_starts,
    _quench,
    _read_poscar,
    _runtime_cap,
    _score_cap,
    build_bpga,
    lj38_reference_energy,
    lj38_truncated_octahedron,
    scheduled_pool_cap,
    scheduled_pybcga_cap,
    search_job,
)

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPTBENCH_BPGA", "/tmp/src/bpga")
os.environ.setdefault("OPTBENCH_PYBCGA", "/tmp/src/PyBCGA")


def test_lj38_reference_is_the_truncated_octahedron():
    assert len(lj38_truncated_octahedron()) == 38
    energy, _coords = lj38_reference_energy()
    assert abs(energy - (-173.92842659)) < 1e-6
    counter = HarnessCounter(cache=True, cap=1000)
    quenched = _quench(lj38_truncated_octahedron(), counter, QUENCH_FORCE, QUENCH_MAXFUN)
    assert quenched["converged"]
    assert quenched["energy"] <= energy + 0.001
    assert quenched["force"] <= QUENCH_FORCE
    assert counter.calls < 100


def test_templates_use_the_scheduled_budgets():
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "bpga":
            continue
        found.append(template["id"])
        problem = resolve_problem(ROOT, template["problem"])
        stated = (problem.get("convergence") or {}).get("max_force_calls")
        if stated in (None, ""):
            assert _check_stop(problem, template) == int(template["failure_cap"])
        else:
            assert _check_stop(problem, template) == int(stated) == 2000000
            assert _score_cap(problem, template) == 2000000
            assert _runtime_cap(problem, template) == min(int(template["failure_cap"]), 2000000)
        section = template["bpga"]
        if section["method"] == "pybcga":
            scheduled = scheduled_pybcga_cap(
                int(section["pop_size"]),
                int(section["max_generation"]),
                int(section["offspring"]),
                int(section["quench_maxfun"]),
            )
            assert int(template["failure_cap"]) == scheduled
        else:
            scheduled = scheduled_pool_cap(
                int(section["n_pool"]),
                int(section["iterations"]),
                int(section["quench_maxfun"]),
            )
            assert int(template["failure_cap"]) == scheduled
    assert set(found) == {
        "bpga/lj38-pool",
        "bpga/lj38-pybcga",
        "bpga/lj75-pool",
        "bpga/lj75-pybcga",
        "bpga/lj98-pool",
        "bpga/lj98-pybcga",
        "bpga/lj150-pool",
        "bpga/lj150-pybcga",
    }


def test_archived_starts_match_the_manifests():
    lj38 = resolve_problem(ROOT, "globalopt/lj38")
    starts = _load_starts(ROOT, lj38, None, Path("/tmp/optbench-bpga-test-geo"), ROOT / "unused.dat")
    assert [name for name, _coords in starts] == [str(index) for index in range(100)]
    assert starts[0][1].shape == (38, 3)
    lj75 = resolve_problem(ROOT, "globalopt/lj75")
    wide = _load_starts(ROOT, lj75, None, Path("/tmp/optbench-bpga-test-geo"), ROOT / "unused.dat")
    assert [name for name, _coords in wide] == [f"coords.{index}" for index in range(1, 201)]
    assert wide[0][1].shape == (75, 3)
    for slug, atoms, count in (("lj98", 98, 100), ("lj150", 150, 100)):
        problem = resolve_problem(ROOT, f"globalopt/{slug}")
        loaded = _load_starts(ROOT, problem, None, Path("/tmp/optbench-bpga-test-geo"), ROOT / "unused.dat")
        assert len(loaded) == count
        assert loaded[0][1].shape == (atoms, 3)
        assert [name for name, _coords in loaded] == [f"coords.{index}" for index in range(1, count + 1)]


def test_poscar_roundtrip_is_cartesian():
    coords = lj38_truncated_octahedron()
    box = 20.0
    lines = ["label\n", f"{box}\n", "1.0 0.0 0.0\n", "0.0 1.0 0.0\n", "0.0 0.0 1.0\n", "Ni\n", "38\n", "Direct\n"]
    for x, y, z in coords:
        lines.append(f"{(x + box / 2) / box} {(y + box / 2) / box} {(z + box / 2) / box}\n")
    path = Path("/tmp/optbench-bpga-poscar")
    path.write_text("".join(lines), encoding="utf-8")
    restored, scale = _read_poscar(path)
    assert scale == box
    assert np.allclose(restored, coords)


def test_a_quenched_global_minimum_is_a_success():
    energy, coords = lj38_reference_energy()
    target = energy + 0.001
    job = {
        "index": 0,
        "start": "ideal",
        "coords": coords.reshape(-1).tolist(),
        "method": "pool",
        "cap": 2000,
        "target": target,
        "reference_energy": energy,
        "quench_force": QUENCH_FORCE,
        "quench_maxfun": QUENCH_MAXFUN,
        "seed": 20261015,
        "code": "BPGA",
        "code_version": "0",
        "commit": "b32cd227e8071429825f990449048eb21c7f8730",
        "tuning": "default",
        "template": "bpga/lj38-pool",
        "problem": "globalopt/lj38",
        "algorithm": "pool GA",
        "work": "/tmp/optbench-bpga-success",
        "pybcga_src": "/tmp/src/PyBCGA",
        "bpga_ga": str(build_bpga()),
        "pop_size": 2,
        "max_generation": 0,
        "offspring": 1,
        "mutant_rate": 0.0,
        "tournament": 2,
        "remove_duplicates": False,
        "mass_extinction": False,
        "n_pool": 2,
        "iterations": 0,
        "mutate": 0.1,
        "mut_type": "random",
        "cross": "weighted",
        "r_ij": 3.0,
        "box_add": 10.0,
    }
    row = search_job(job)
    assert row["success"] is True
    assert row["failure"] is None
    assert row["force_calls"] < 100
    assert row["energy"] <= target
    py = dict(job, method="pybcga", cap=2000, work="/tmp/optbench-bpga-success-py")
    py_row = search_job(py)
    assert py_row["success"] is True
    assert py_row["failure"] is None
    assert py_row["energy"] <= target


def test_missing_target_is_not_a_success_and_a_spent_budget_is_a_cap():
    assert _judge(-173.9, 1e-4, 10, 100, None, 1e-3)["success"] is False
    assert _judge(-173.9, 1e-4, 10, 100, None, 1e-3)["failure"] == "wrong_stationary"
    assert _judge(-173.9, 1e-4, 100, 100, -173.0, 1e-3)["failure"] == "cap"
    assert _judge(float("nan"), float("inf"), 100, 100, -173.0, 1e-3)["failure"] == "divergence"


def test_bpga_build_keeps_the_integer_index_fix():
    ga = build_bpga()
    text = (ga / "Database.py").read_text(encoding="utf-8")
    assert "eleList[i//3]" in text
    ran = (ga / "MinimiseRan.py").read_text(encoding="utf-8")
    assert ",self.calcNum,Index,self.eleNums" in ran
    overlap = (ga / "fixOverlap.py").read_text(encoding="utf-8")
    assert "pushes > 100000" in overlap
