"""Anneal cluster-search rows use the harness counter and the stock presets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optbench.globalopt import force_cap
from optbench.problems import load_problem
from optbench.runners.anneal_go import (
    ANNEAL_COMMIT,
    RGMIN_COMMIT,
    bind_potential,
    load_geometries,
    preset_flags,
    version_label,
)

ROOT = Path(__file__).resolve().parents[1]
CAPS = {
    "globalopt/lj38": 2000000,
    "globalopt/lj75": 1000000,
    "globalopt/lj98": 1000000000,
    "globalopt/lj150": 10000000,
}
STARTS = {
    "globalopt/lj38": 100,
    "globalopt/lj75": 200,
    "globalopt/lj98": 100,
    "globalopt/lj150": 100,
}


def test_problem_caps_and_bounds():
    for problem_id, cap in CAPS.items():
        problem = load_problem(ROOT / "problems" / f"{problem_id}.toml")
        assert force_cap(problem) == cap
        assert float(problem["convergence"]["energy_above_minimum"]) == 0.001
        assert problem["reference"]["energy"] < 0
        assert int(problem["sample"]["starts"]) == STARTS[problem_id]


def test_presets_are_the_two_stock_constructors():
    recommended = preset_flags("recommended")
    communicating = preset_flags("communicating")
    assert recommended == {
        "recommended": True,
        "derived": False,
        "communicating": False,
        "ras": False,
    }
    assert communicating["communicating"] is True
    assert communicating["derived"] is False
    assert communicating["recommended"] is True
    with pytest.raises(ValueError):
        preset_flags("for_cluster")
    with pytest.raises(ValueError):
        preset_flags("derived")


def test_templates_match_the_preset_label():
    import tomllib

    found = sorted(path.stem for path in (ROOT / "templates" / "anneal").glob("*.toml"))
    assert found == [
        "lj150-communicating",
        "lj150-recommended",
        "lj38-communicating",
        "lj38-recommended",
        "lj75-communicating",
        "lj75-recommended",
        "lj98-communicating",
        "lj98-recommended",
    ]
    for path in (ROOT / "templates" / "anneal").glob("*.toml"):
        template = tomllib.loads(path.read_text(encoding="utf-8"))
        preset = template["preset"]
        assert preset in {"communicating", "recommended"}
        assert template["code_version"] == version_label(preset)
        assert template["commit"] == ANNEAL_COMMIT
        assert "for_cluster" not in template["note"]
        assert "derived" not in template["note"]
        assert RGMIN_COMMIT in template["note"]
        problem = load_problem(ROOT / "problems" / f"{template['problem']}.toml")
        assert int(template["failure_cap"]) == force_cap(problem)


def test_energy_and_gradient_at_one_geometry_are_one_call():
    rng = np.random.default_rng(0)
    point = rng.normal(size=(8, 3))
    potential, state, objective, gradient = bind_potential(1000, -1.0e9)
    objective(point)
    gradient(point.reshape(-1))
    assert potential.calls == 1
    assert state["obj_calls"] == 1
    assert state["grad_calls"] == 1


def test_bound_stops_the_search():
    rng = np.random.default_rng(1)
    point = rng.normal(size=(6, 3))
    potential, state, objective, gradient = bind_potential(50, 1.0e9)
    with pytest.raises(KeyboardInterrupt):
        objective(point)
    assert state["hit"] is True
    assert state["stop"] is True
    assert potential.calls == 1


def test_cap_stops_without_a_hit():
    rng = np.random.default_rng(2)
    point = rng.normal(size=(5, 3))
    potential, state, objective, gradient = bind_potential(2, -1.0e9)
    objective(point)
    with pytest.raises(KeyboardInterrupt):
        gradient(point + 1.0)
    assert state["hit"] is False
    assert state["cap_hit"] is True
    assert potential.calls == 2


def test_archived_start_counts(tmp_path):
    for problem_id, count in STARTS.items():
        problem = load_problem(ROOT / "problems" / f"{problem_id}.toml")
        loaded = load_geometries(ROOT, problem, tmp_path / problem_id.replace("/", "-"))
        assert len(loaded) == count
        labels = [label for label, _pos in loaded]
        assert labels == sorted(labels, key=lambda label: int(label.split(".")[-1]))
        _label, pos = loaded[0]
        assert abs(float(pos.mean())) < 1e-8


def test_run_writes_a_ranked_row(tmp_path, monkeypatch):
    from optbench.campaign import load_template, run

    def fake(job):
        return {
            "index": job["index"],
            "start": job["start"],
            "seed": job["seed"],
            "force_calls": 10 + job["index"],
            "success": True,
            "failure": None,
            "reasons": [],
            "energy": -173.928427,
            "counter": "harness",
            "preset": job["preset"],
            "profile": {"quenches": 1, "improvements": 1},
            "error": "",
        }

    monkeypatch.setattr("optbench.runners.anneal_go.search_start", fake)
    monkeypatch.setattr("optbench.runners.anneal_go.installed_rgmin_pin", lambda: ("0.2.0", RGMIN_COMMIT))
    output = tmp_path / "anneal-0.11.1-lj38-communicating.dat"
    summary = run(
        ROOT,
        "globalopt/lj38",
        "anneal/lj38-communicating",
        output=output,
        workers=1,
        work=tmp_path / "work",
    )
    assert summary["starts"] == 100
    assert summary["nfailed"] == 0
    text = output.read_text(encoding="utf-8")
    assert "code_version 0.11.1 (communicating)" in text
    assert "counter harness" in text
    assert "ranked" not in text
    rows = [
        json.loads(line)
        for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 100
    assert rows[0]["counter"] == "harness"
    assert all(row["success"] for row in rows)
