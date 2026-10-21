import tomllib
from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.jaxmd_catalog import (
    JAXMD_COMMIT,
    JAXMD_VERSION,
    JAXOPT_COMMIT,
    OPTAX_COMMIT,
    render,
    specs,
)
from optbench.noise_catalog import SIGMAS, sigma_slug

ROOT = Path(__file__).resolve().parents[1]


def test_registry_pins_jaxmd_and_the_lbfgs_neighbors():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "jaxmd")
    assert pinned["commit"] == JAXMD_COMMIT
    assert pinned["ref"] == "jax-md-v0.2.29"
    assert pinned["repository"] == "https://github.com/jax-md/jax-md"
    assert JAXOPT_COMMIT in pinned["note"]
    assert OPTAX_COMMIT in pinned["note"]


def test_jaxmd_templates_cover_the_three_problems():
    found = [row for row in list_templates(ROOT) if row.get("engine") == "jaxmd"]
    catalog = {row["id"]: row for row in specs()}
    assert len(found) == 24
    assert {row["id"] for row in found} == set(catalog)
    seen = set()
    for template in found:
        spec = catalog[template["id"]]
        assert template["id"] == spec["id"]
        assert template["engine"] == "jaxmd"
        assert template["code"] == "jaxmd"
        assert template["commit"] == JAXMD_COMMIT
        assert template["code_version"] == JAXMD_VERSION
        assert template["tuning"] == "default"
        assert template["jaxmd"]["maxstep"] == 0.2
        problem = resolve_problem(ROOT, template["problem"])
        assert _check_stop(problem, template) == 10000
        path = ROOT / "templates" / f"{template['id']}.toml"
        assert path.read_text(encoding="utf-8") == render(spec)
        seen.add((template["problem"], template["jaxmd"]["method"]))
    for method in ("fire", "gd", "lbfgs", "optax-lbfgs"):
        assert ("minimization/lj38", method) in seen
        assert ("minimization/morse-bulk", method) in seen
        for sigma in SIGMAS:
            match = [
                row
                for row in found
                if row["problem"] == "minimization/lj38-noise"
                and row["jaxmd"]["method"] == method
                and float(row["noise"]["sigma"]) == float(sigma)
            ]
            assert len(match) == 1
            assert match[0]["id"].endswith(sigma_slug(sigma))
            assert int(match[0]["noise"]["seed"]) == 20261016
    assert next(row for row in found if row["id"] == "jaxmd/lj38-gd")["jaxmd"]["step_size"] == 0.001
    assert next(row for row in found if row["id"] == "jaxmd/lj38-lbfgs")["jaxmd"]["history_size"] == 10
    assert next(row for row in found if row["id"] == "jaxmd/lj38-optax-lbfgs")["jaxmd"]["memory_size"] == 10


def test_jaxmd_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
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

    monkeypatch.setattr("optbench.runners.jaxmd_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/lj38",
        "jaxmd/lj38-fire",
        output=tmp_path / "benchmark.dat",
        workers=2,
        limit=3,
        resume=True,
    )
    assert captured == {
        "template": "jaxmd/lj38-fire",
        "problem": "minimization/lj38",
        "workers": 2,
        "limit": 3,
        "resume": True,
    }
    assert summary["template"] == "jaxmd/lj38-fire"
