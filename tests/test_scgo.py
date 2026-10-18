import importlib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem, run
from optbench.runners.scgo_go import (
    PINNED_COMMIT,
    SearchStop,
    default_bh_budget,
    lj38_reference_energy,
    load_geometries,
)

ROOT = Path(__file__).resolve().parents[1]


def test_lj38_reference_is_the_quenched_cambridge_geometry():
    energy, force_norm = lj38_reference_energy()
    assert energy == pytest.approx(-173.92842659062762)
    assert force_norm < 1e-4


def test_default_budget_matches_the_0_9_1_formula():
    assert default_bh_budget(38) == (131, 233, 30888)
    assert default_bh_budget(75) == (154, 266, 41385)


def test_search_stop_is_not_a_runtime_error():
    stop = SearchStop("target", -1.0, 0.0)
    assert not isinstance(stop, RuntimeError)
    assert stop.kind == "target"


def test_lj38_template_uses_the_budget_cap():
    template = load_template(ROOT, "scgo/lj38-bh")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    assert template["engine"] == "scgo"
    assert template["code"] == "SCGO"
    assert template["commit"] == PINNED_COMMIT
    assert template["code_version"] == "0.9.1"
    assert template["scgo"]["algorithm"] == "bh"
    assert template["scgo"]["symbol"] == "C"
    assert int(template["scgo"]["seed"]) == 314159
    assert _check_stop(problem, template) == 30888
    assert [row["id"] for row in list_templates(ROOT) if str(row["id"]).startswith("scgo/")] == [
        "scgo/lj38-bh"
    ]


def test_archived_starts_sort_numerically(tmp_path: Path):
    problem = resolve_problem(ROOT, "globalopt/lj38")
    loaded = load_geometries(ROOT, problem, None, tmp_path / "work", tmp_path / "out.dat")
    assert [name for name, _coords, _box in loaded[:12]] == [str(index) for index in range(12)]
    assert len(loaded) == 100
    assert len(loaded[0][1]) == 38 * 3


def test_asv_grid_is_one_start_until_full(monkeypatch):
    import benchmarks.bench_scgo as bench

    assert bench.ScgoLj38Bh.param_names == ["seed", "start"]
    assert bench.ScgoLj38Bh.params[0] == ["314159"]
    assert bench.ScgoLj38Bh.params[1] == ["0"]
    assert bench.ScgoLj38Bh.cap == 30888
    assert bench.ScgoLj38Bh.template_id == "scgo/lj38-bh"
    assert bench.ScgoLj38Bh.problem_id == "globalopt/lj38"
    assert bench.ScgoLj38Bh.repeat == 1
    assert bench.ScgoLj38Bh.track_force_calls.unit == "force calls"
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    reloaded = importlib.reload(bench)
    assert reloaded.ScgoLj38Bh.params[1] == [str(index) for index in range(100)]
    monkeypatch.delenv("OPTBENCH_ASV_FULL", raising=False)
    importlib.reload(bench)


def test_scgo_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["limit"] = kwargs["limit"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 1,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.scgo_go.run_template", fake)
    summary = run(
        ROOT,
        "globalopt/lj38",
        "scgo/lj38-bh",
        output=tmp_path / "benchmark.dat",
        workers=3,
        limit=1,
    )
    assert captured == {
        "template": "scgo/lj38-bh",
        "problem": "globalopt/lj38",
        "workers": 3,
        "limit": 1,
    }
    assert summary["template"] == "scgo/lj38-bh"


def test_installed_library_budget_matches_the_template():
    pytest.importorskip("scgo")
    from optbench.runners.scgo_go import bh_settings

    template = load_template(ROOT, "scgo/lj38-bh")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    settings = bh_settings(problem, template, 38)
    assert settings["niter"] == 131
    assert settings["niter_local_relaxation"] == 233
    assert settings["cap"] == 30888
    assert settings["optimizer"] == "FIRE"
    assert settings["fmax"] == pytest.approx(0.05)
    assert settings["dr"] == pytest.approx(0.2)
    assert settings["symbol"] == "C"
    assert settings["threshold"] == pytest.approx(settings["reference_energy"] + 0.001)
