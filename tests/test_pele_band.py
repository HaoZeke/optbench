"""pele band templates and a capped search on one LJ38 pair."""

import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem
from optbench.pele_band import _execute, _job, _registry, _settings, cached_pairs

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = (
    "double-ended-saddle-search/lj38",
    "double-ended-saddle-search/lj38_2ts_separation",
    "double-ended-saddle-search/pt-island",
    "neb/Pt-heptamer",
)


def test_pele_templates_match_the_registry_and_the_problem_stop():
    pin = _registry(ROOT)
    found = []
    for template in list_templates(ROOT):
        if not str(template["id"]).startswith("pele/"):
            continue
        found.append(template["id"])
        assert template["engine"] == "pele-band"
        assert template["code"] == "pele"
        assert template["code_version"] == "0.1.2"
        assert template["commit"] == pin["commit"]
        _check_stop(resolve_problem(ROOT, "minimization/lj38"), template)
    assert found == [
        "pele/connect",
        "pele/dneb-tuned",
        "pele/dneb",
        "pele/neb-tuned",
        "pele/neb",
    ]
    record = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pele = [row for row in record["code"] if row["id"] == "pele"]
    assert len(pele) == 1
    assert pele[0]["commit"] == "b7081b932a060be11342fbe8287ac35da8ada879"


def test_pele_settings_follow_each_problem():
    pin = _registry(ROOT)["commit"]
    expected = {
        "double-ended-saddle-search/lj38": (2500, 5, 0.001, "lj"),
        "double-ended-saddle-search/lj38_2ts_separation": (10000, 5, 0.001, "lj"),
        "double-ended-saddle-search/pt-island": (3000, 3, 0.001, "morse_pt"),
        "neb/Pt-heptamer": (10000, 8, 0.01, "morse_pt"),
    }
    for name, (cap, images, force, potential) in expected.items():
        template = load_template(ROOT, "pele/connect")
        settings = _settings(template, resolve_problem(ROOT, name), pin)
        assert settings["cap"] == cap
        assert settings["n_interior"] == images
        assert settings["force_norm"] == force
        assert settings["potential"] == potential
        assert settings["kind"] == "connect"
        assert settings["dneb"] is True
        neb = _settings(load_template(ROOT, "pele/neb"), resolve_problem(ROOT, name), pin)
        assert neb["kind"] == "neb"
        assert neb["dneb"] is False
        dneb = _settings(load_template(ROOT, "pele/dneb"), resolve_problem(ROOT, name), pin)
        assert dneb["dneb"] is True
        assert neb["climbing"] is False
        assert neb["stop"] == "residual"
        tuned = _settings(load_template(ROOT, "pele/neb-tuned"), resolve_problem(ROOT, name), pin)
        dneb_tuned = _settings(load_template(ROOT, "pele/dneb-tuned"), resolve_problem(ROOT, name), pin)
        assert tuned["kind"] == "neb"
        assert tuned["dneb"] is False
        assert dneb_tuned["kind"] == "dneb"
        assert dneb_tuned["dneb"] is True
        for row in (tuned, dneb_tuned):
            assert row["cap"] == cap
            assert row["n_interior"] == images
            assert row["force_norm"] == force
            assert row["climbing"] is True
            assert row["stop"] == "true_force"
            assert row["spring_k"] == 100.0
            assert row["max_erise"] == 1.0
            assert row["memory"] == 4
            assert row["h0"] == 0.1


def test_capped_pele_neb_stops_inside_the_budget():
    pytest.importorskip("pele")
    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    template = load_template(ROOT, "pele/neb")
    settings = _settings(template, problem, _registry(ROOT)["commit"])
    settings["cap"] = 30
    pair = next(iter(cached_pairs(ROOT, problem, settings).values()))
    record = _execute(_job(pair, settings))
    assert record["schema"] == "optbench.start/v2"
    assert record["counter"] == "harness"
    assert record["force_calls"] <= 30
    assert record["force_calls"] > 0
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert record["geometry"] is None or len(record["geometry"]) == 38
    assert "force" in record


def test_tuned_band_stores_the_final_geometry_and_force():
    pytest.importorskip("pele")
    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    template = load_template(ROOT, "pele/neb-tuned")
    settings = _settings(template, problem, _registry(ROOT)["commit"])
    settings["cap"] = 40
    pair = next(iter(cached_pairs(ROOT, problem, settings).values()))
    record = _execute(_job(pair, settings))
    assert record["force_calls"] <= 40
    assert record["force_calls"] > 0
    assert len(record["geometry"]) == pair.reactant.shape[0]
    assert len(record["geometry"][0]) == 3
    assert record["energy"] is None or isinstance(record["energy"], float)
    if record["force"] is not None:
        assert len(record["force"]) == len(record["geometry"])
        assert record["force_norm"] is not None


def test_pele_benchmarks_name_the_four_problems():
    import benchmarks.bench_pele as bench

    assert bench.Lj38DePeleNeb.problem_name == "double-ended-saddle-search/lj38"
    assert bench.Lj38DePeleNeb.template_name == "pele/neb"
    assert bench.Lj38DePeleNeb.cap == 2500
    assert bench.Lj38TwoTsPeleDneb.cap == 10000
    assert bench.PtIslandPeleConnect.problem_name == "double-ended-saddle-search/pt-island"
    assert bench.PtIslandPeleConnect.template_name == "pele/connect"
    assert bench.PtIslandPeleConnect.cap == 3000
    assert bench.HeptamerPeleDneb.problem_name == "neb/Pt-heptamer"
    assert bench.HeptamerPeleDneb.cap == 10000
    assert bench.Lj38DePeleNeb.param_names == ["seed", "start"]
    assert len(bench.CLASS_TEMPLATE) == 20
    assert bench.Lj38DePeleNebTuned.template_name == "pele/neb-tuned"
    assert bench.HeptamerPeleDnebTuned.template_name == "pele/dneb-tuned"
    assert set(bench.CLASS_PROBLEM.values()) == set(PROBLEMS)
