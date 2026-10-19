"""pele transition-state templates and the harness potential."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.counter import ForceCallCap
from optbench.pele_ts import (
    _SADDLE_SEARCH_CAP,
    HarnessPotential,
    _transition_kwargs,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b7081b932a060be11342fbe8287ac35da8ada879"


def _pele_importable() -> bool:
    try:
        import pele  # noqa: F401
    except ImportError:
        return False
    return True


def test_pele_templates_use_the_problem_cap_and_the_pinned_commit():
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "pele":
            continue
        found.append(template["id"])
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        assert cap is not None
        assert cap >= 1000
        if problem["id"] == "saddle-search/pt-island":
            assert cap == 5064
        else:
            assert cap == 10000
        assert template["commit"] == COMMIT
        assert template["code"] == "pele"
        assert template["code_version"] == "0.1.2"
    assert "pele/lj38-ts" in found
    assert "pele/lj38-ts-ffd" in found
    assert "pele/lj38-ts-cd" in found
    assert "pele/pt-island-ts" in found
    assert "pele/pt-island-ts-ffd" in found
    assert "pele/lj38-loweig" in found
    assert "pele/lj38-loweig-forward" in found
    assert "pele/lj38-loweig-central" in found
    assert "pele/pt-island-search" in found
    assert "pele/pt-island-search-ffd" in found
    assert _SADDLE_SEARCH_CAP == 5064


def test_endpoint_arrays_leave_the_starts_file(tmp_path: Path):
    from optbench.pele_ts import _store_endpoints

    records = [
        {
            "start": "a",
            "endpoint_plus": [1.0, 2.0, 3.0],
            "endpoint_minus": [4.0, 5.0, 6.0],
            "minimum": [4.0, 5.0, 6.0],
            "force_calls": 1,
        },
        {"start": "b", "force_calls": 2},
    ]
    output = tmp_path / "run.dat"
    _store_endpoints(output, records)
    assert "endpoint_plus" not in records[0]
    assert "minimum" not in records[0]
    plus = np.load(tmp_path / "run.endpoints-plus.npz")
    minus = np.load(tmp_path / "run.endpoints-minus.npz")
    assert plus["endpoint_plus"].shape == (2, 3)
    assert plus["endpoint_plus"][0, 0] == 1.0
    assert np.isnan(plus["endpoint_plus"][1]).all()
    assert minus["endpoint_minus"][0, 2] == 6.0
    assert np.allclose(plus["minimum"][0], [4.0, 5.0, 6.0])
    assert np.isnan(plus["minimum"][1]).all()


def test_partial_checkpoint_drops_a_truncated_tail(tmp_path: Path):
    from optbench.pele_ts import _load_partial

    path = tmp_path / "run.partial.jsonl"
    path.write_text('{"start": "rep1-search1", "success": false}\n{"start": "rep1-sea\n', encoding="utf-8")
    rows = _load_partial(path)
    assert rows == [{"start": "rep1-search1", "success": False}]


def test_registry_pins_the_pele_commit():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "pele")
    assert row["commit"] == COMMIT
    assert row["name"] == "pele"
    assert row["committed"] == "2026-10-12"


def test_harness_potential_caches_one_geometry_and_stops_at_the_cap():
    seen = {"n": 0}

    def evaluate(coords):
        seen["n"] += 1
        values = np.asarray(coords, dtype=np.float64)
        return float(np.dot(values, values)), 2.0 * values

    pot = HarnessPotential(evaluate, cap=2)
    energy, gradient = pot.getEnergyGradient(np.array([1.0, 0.0, 0.0]))
    again = pot.getEnergy(np.array([1.0, 0.0, 0.0]))
    assert energy == again == 1.0
    assert pot.calls == 1
    assert seen["n"] == 1
    assert gradient.shape == (3,)
    pot.getEnergyGradient(np.array([0.0, 1.0, 0.0]))
    assert pot.calls == 2
    with pytest.raises(ForceCallCap):
        pot.getEnergyGradient(np.array([0.0, 0.0, 1.0]))
    assert pot.calls == 2
    assert seen["n"] == 2
    with pot.verification():
        pot.getEnergyGradient(np.array([0.0, 0.0, 1.0]))
    assert pot.calls == 2
    assert pot.verify_calls == 1


def test_rms_tolerance_matches_the_euclidean_stop():
    kwargs = _transition_kwargs({"nsteps": 100, "orthog": "translations-rotations"}, 114, 0.001, None)
    assert kwargs["tol"] == pytest.approx(0.001 / np.sqrt(114.0))
    assert kwargs["orthogZeroEigs"] == 0
    assert kwargs["verbosity"] == 0
    bare = _transition_kwargs({"orthog": "none"}, 525, 0.001, np.ones(525))
    assert bare["orthogZeroEigs"] is None
    assert bare["eigenvec0"].shape == (525,)


def test_pele_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["limit"] = kwargs["limit"]
        return {"template": template["id"], "problem": problem["id"], "starts": 0}

    monkeypatch.setattr("optbench.pele_ts.run_template", fake)
    summary = run(
        ROOT,
        "tsopt/LJ38",
        "pele/lj38-ts",
        output=tmp_path / "benchmark.dat",
        workers=3,
        limit=2,
    )
    assert captured == {
        "template": "pele/lj38-ts",
        "problem": "tsopt/LJ38",
        "workers": 3,
        "limit": 2,
    }
    assert summary["problem"] == "tsopt/LJ38"


@pytest.mark.skipif(not _pele_importable(), reason="pele is not installed")
def test_ffd_and_forward_match_one_archived_start(tmp_path: Path, monkeypatch):
    if Path("/tmp/pele").is_dir():
        monkeypatch.setenv("PELE_SRC", "/tmp/pele")
    from optbench.campaign import load_template
    from optbench.pele_ts import run_template

    ffd = run_template(
        ROOT,
        resolve_problem(ROOT, "tsopt/LJ38"),
        load_template(ROOT, "pele/lj38-ts-ffd"),
        output=tmp_path / "ffd.dat",
        workers=1,
        limit=1,
    )
    row = ffd["records"][0]
    assert row["success"], row
    assert row["hessian_index"] == 1
    assert row["force_calls"] < 500
    assert row["force_norm"] <= 0.001
    assert row["counter"] == "harness"
    assert row["screen_calls"] >= 0
    assert row["unscreened_calls"] + row["screen_calls"] == row["force_calls"]
    assert (tmp_path / "ffd.starts.jsonl").is_file()

    mode = run_template(
        ROOT,
        resolve_problem(ROOT, "lowest-eigenvector/lj38"),
        load_template(ROOT, "pele/lj38-loweig-forward"),
        output=tmp_path / "mode.dat",
        workers=1,
        limit=1,
    )
    found = mode["records"][0]
    assert found["success"], found
    assert found["overlap"] >= 0.99
    assert found["force_calls"] < 100
