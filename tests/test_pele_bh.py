"""Archived pele basin hopping for Lennard-Jones 38."""

from pathlib import Path

from optbench.campaign import load_template, resolve_problem, run
from optbench.runners.pele_bh import _options, _rule

ROOT = Path(__file__).resolve().parents[1]


def test_template_is_the_latest_release_stock():
    template = load_template(ROOT, "pele/lj38-basinhopping")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    options = _options(template)
    minimum, tolerance, partial = _rule(problem, template)
    assert template["tuning"] == "stock"
    assert template["code"] == "pele"
    assert template["code_version"] == "0.1.2"
    assert template["commit"] == "b7081b932a060be11342fbe8287ac35da8ada879"
    assert options == {"stock": True}
    assert "0.1.2" in template["note"]
    assert minimum == -173.9284
    assert tolerance == 0.001
    assert partial == ""
    assert template["stop"]["max_force_calls"] == 4000000
    assert not (ROOT / "codes" / "pele-a68ec5-lbfgs-linesearch.patch").exists()


def test_current_pele_uses_the_same_driver_unpatched():
    template = load_template(ROOT, "current-pele/lj38-basinhopping")
    options = _options(template)
    assert template["code"] == "pele"
    assert template["code_version"] == "0.1.2"
    assert template["commit"] == "b7081b932a060be11342fbe8287ac35da8ada879"
    assert template["tuning"] == "archive"
    assert "current-pele" in template["note"]
    assert options["quench"] == "lbfgs_cpp"
    assert options["temperature"] == 1.0
    assert options["stepsize"] == 0.41
    assert options["acc_ratio"] == 0.5
    assert options["factor"] == 0.9
    assert options["interval"] == 100
    assert options["quench_tol"] == 1e-3


def test_campaign_dispatches_pele(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, *, geometries, output, workers, work, limit, resume):
        captured["problem"] = problem["id"]
        captured["template"] = template["id"]
        captured["workers"] = workers
        captured["resume"] = resume
        captured["output"] = output
        return {"starts": 0}

    monkeypatch.setattr("optbench.runners.pele_bh.run_template", fake)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "globalopt/lj38",
        "pele/lj38-basinhopping",
        output=output,
        workers=3,
        resume=True,
    )
    assert summary == {"starts": 0}
    assert captured["problem"] == "globalopt/lj38"
    assert captured["template"] == "pele/lj38-basinhopping"
    assert captured["workers"] == 3
    assert captured["resume"] is True
    assert captured["output"] == output
