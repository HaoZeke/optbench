import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.runners.nlopt_min import ALGORITHMS, require_step

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "673d2504f819bf554a7968e41de552ee102e7fb1"
LJ_STEPS = {
    "LD_LBFGS": ("scale", 0.2),
    "LD_VAR2": ("scale", 0.2),
    "LD_SLSQP": ("scale", 0.2),
    "LD_TNEWTON_PRECOND_RESTART": ("box", 0.2),
    "LD_MMA": ("none", None),
}


def test_nlopt_templates_cover_the_gradient_methods():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "nlopt")
    assert pinned["commit"] == COMMIT
    assert pinned["ref"] == "v2.11.0"
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("nlopt/")]
    assert len(found) == 10
    seen = set()
    for template in found:
        assert template["engine"] == "nlopt"
        assert template["code"] == "nlopt"
        assert template["commit"] == COMMIT
        assert template["code_version"] == "2.11.0"
        assert template["tuning"] == "default"
        assert template["algorithm"] in ALGORITHMS
        assert template["nlopt"]["algorithm"] == template["algorithm"]
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        step = template["nlopt"]["step"]
        maxstep = template["nlopt"].get("maxstep")
        require_step(template["problem"], template["algorithm"], step, maxstep)
        seen.add((template["problem"], template["algorithm"]))
    for problem in ("minimization/lj38", "minimization/morse-bulk"):
        for algorithm in ALGORITHMS:
            assert (problem, algorithm) in seen
    for template in found:
        if template["problem"] == "minimization/lj38":
            step, maxstep = LJ_STEPS[template["algorithm"]]
            assert template["nlopt"]["step"] == step
            assert template["nlopt"].get("maxstep") == maxstep
        else:
            assert template["nlopt"]["step"] == "none"
            assert "maxstep" not in template["nlopt"]


def test_lj38_identity_step_is_rejected():
    with pytest.raises(ValueError, match="step scale"):
        require_step("minimization/lj38", "LD_LBFGS", "none", None)
    with pytest.raises(ValueError, match="step box"):
        require_step("minimization/lj38-noise", "LD_TNEWTON_PRECOND_RESTART", "scale", 0.2)
    with pytest.raises(ValueError, match="library defaults"):
        require_step("minimization/morse-bulk", "LD_MMA", "scale", 0.2)


def test_nlopt_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["limit"] = kwargs["limit"]
        captured["resume"] = kwargs["resume"]
        return {"template": template["id"], "problem": problem["id"], "starts": 0}

    monkeypatch.setattr("optbench.runners.nlopt_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/lj38",
        "nlopt/lj38-lbfgs",
        output=tmp_path / "benchmark.dat",
        workers=2,
        limit=3,
        resume=True,
    )
    assert captured == {
        "template": "nlopt/lj38-lbfgs",
        "problem": "minimization/lj38",
        "workers": 2,
        "limit": 3,
        "resume": True,
    }
    assert summary["template"] == "nlopt/lj38-lbfgs"
