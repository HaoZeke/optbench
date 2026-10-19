import tomllib
from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem, run

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "767e2c0c13ff2b851ac82101ccd5aab9a558a4f1"
VERSION = "0.5.1.dev2+g767e2c0c1"
EXPECTED = {
    ("tsopt/LJ38", "RS_I_RFO"),
    ("tsopt/LJ38", "P_RFO"),
    ("tsopt/pt-island", "RS_I_RFO"),
    ("tsopt/pt-island", "P_RFO"),
}


def test_optking_templates_cover_both_problems_and_both_steps():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "optking")
    assert pinned["commit"] == COMMIT
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("optking/")]
    assert len(found) == 4
    seen = set()
    for template in found:
        assert template["engine"] == "optking"
        assert template["commit"] == COMMIT
        assert template["code_version"] == VERSION
        assert template["tuning"] == "default"
        assert template["ranked"] is True
        assert template["optking"]["opt_coordinates"] == "CARTESIAN"
        assert template["optking"]["full_hess_every"] == 0
        assert template["optking"]["fd_step"] == 1.0e-5
        assert template["algorithm"] == template["optking"]["step_type"]
        problem = resolve_problem(ROOT, template["problem"])
        assert _check_stop(problem, template) == 10000
        assert float(template["stop"]["force_norm"]) == float(problem["convergence"]["force_norm"])
        seen.add((template["problem"], template["algorithm"]))
    assert seen == EXPECTED


def test_optking_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["limit"] = kwargs["limit"]
        captured["resume"] = kwargs["resume"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 0,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.optking_ts.run_template", fake)
    summary = run(
        ROOT,
        "tsopt/LJ38",
        "optking/lj38-rs-irfo",
        output=tmp_path / "benchmark.dat",
        workers=2,
        limit=3,
        resume=True,
    )
    assert captured == {
        "template": "optking/lj38-rs-irfo",
        "problem": "tsopt/LJ38",
        "workers": 2,
        "limit": 3,
        "resume": True,
    }
    assert summary["template"] == "optking/lj38-rs-irfo"
