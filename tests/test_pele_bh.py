import json
import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem, run
from optbench.runners.pele_bh import _load_geometries, _meets, _rule, run_template

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b7081b932a060be11342fbe8287ac35da8ada879"
PROBLEMS = ("globalopt/lj38", "globalopt/lj75", "globalopt/lj98", "globalopt/lj150")


def _pele_templates() -> list[dict]:
    return [
        row
        for row in list_templates(ROOT)
        if str(row.get("algorithm") or "") == "basin hopping"
    ]


def test_pele_templates_match_the_registry_pin():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pele")
    assert pinned["commit"] == COMMIT
    found = _pele_templates()
    assert len(found) == 4
    seen = set()
    for template in found:
        assert template["engine"] == "pele"
        assert template["commit"] == COMMIT
        assert template["code_version"] == "0.1.2"
        assert template["tuning"] == "default"
        assert template["algorithm"] == "basin hopping"
        assert template["pele"]["quench"] == "lbfgs_cpp"
        assert template["pele"]["temperature"] == 1.0
        assert template["pele"]["stepsize"] == 0.6
        problem = resolve_problem(ROOT, template["problem"])
        expected_cap = {
            "globalopt/lj38": 3_000_000,
            "globalopt/lj75": 1_000_000,
            "globalopt/lj98": 1_000_000_000,
            "globalopt/lj150": 10_000_000,
        }[template["problem"]]
        assert int(template["stop"]["max_force_calls"]) == expected_cap
        assert _check_stop(problem, template) == expected_cap
        minimum, tolerance, partial = _rule(problem, template)
        assert tolerance == 0.001
        assert partial == ""
        assert minimum == template["reference"]["energy_minimum"]
        assert minimum == float(problem["reference"]["energy"])
        seen.add(template["problem"])
    assert seen == set(PROBLEMS)


def test_duplicate_bytes_keep_the_numbered_name(tmp_path: Path):
    geo = tmp_path / "geo"
    geo.mkdir()
    body = "\n".join(f"0 0 {index}" for index in range(4)) + "\n"
    (geo / "coords").write_text(body, encoding="utf-8")
    (geo / "coords.2").write_text(body, encoding="utf-8")
    (geo / "coords.1").write_text("\n".join(f"1 0 {index}" for index in range(4)) + "\n", encoding="utf-8")
    problem = {"id": "scratch/plain", "sample": {}}
    template = {"id": "pele/plain", "input": {}}
    loaded = _load_geometries(ROOT, problem, template, geo, None, tmp_path / "out.dat")
    assert [(name, seed) for name, _coords, seed in loaded] == [("coords.1", 1), ("coords.2", 2)]


def test_lj38_archive_has_one_hundred_starts(tmp_path: Path):
    problem = resolve_problem(ROOT, "globalopt/lj38")
    template = load_template(ROOT, "pele/lj38-basinhopping")
    loaded = _load_geometries(ROOT, problem, template, None, tmp_path, tmp_path / "out.dat")
    assert len(loaded) == 100
    assert [seed for _name, _coords, seed in loaded] == list(range(100))
    assert loaded[0][1].shape == (114,)


def test_larger_archives_drop_the_bare_coords_duplicate(tmp_path: Path):
    expected = {"globalopt/lj75": 200, "globalopt/lj98": 100, "globalopt/lj150": 100}
    for problem_id, count in expected.items():
        problem = resolve_problem(ROOT, problem_id)
        template = load_template(ROOT, f"pele/{problem_id.split('/')[1]}-basinhopping")
        loaded = _load_geometries(ROOT, problem, template, None, tmp_path, tmp_path / "out.dat")
        names = [name for name, _coords, _seed in loaded]
        assert len(names) == count
        assert "coords" not in names
        assert names[0] == "coords.1"


def test_energy_window():
    minimum = -173.928427
    tolerance = 0.001
    assert _meets(minimum, minimum, tolerance, True, 10, 100)
    assert _meets(minimum + tolerance, minimum, tolerance, True, 10, 100)
    assert not _meets(minimum + tolerance + 1e-9, minimum, tolerance, True, 10, 100)
    assert not _meets(minimum, minimum, tolerance, True, 100, 100)
    assert _meets(minimum - 0.02, minimum, tolerance, True, 10, 100)
    assert not _meets(minimum, minimum, tolerance, False, 10, 100)
    assert not _meets(None, minimum, tolerance, True, 10, 100)


def test_pele_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
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
            "starts": 1,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.pele_bh.run_template", fake)
    summary = run(
        ROOT,
        "globalopt/lj38",
        "pele/lj38-basinhopping",
        output=tmp_path / "benchmark.dat",
        workers=2,
        limit=3,
        resume=True,
    )
    assert captured == {
        "template": "pele/lj38-basinhopping",
        "problem": "globalopt/lj38",
        "workers": 2,
        "limit": 3,
        "resume": True,
    }
    assert summary["template"] == "pele/lj38-basinhopping"


def test_cap_stops_one_lj38_start(tmp_path: Path):
    pytest.importorskip("pele")
    import pele.version as pele_version

    if getattr(pele_version, "git_revision", "") != COMMIT:
        pytest.skip("installed pele is not the pinned commit")
    template = load_template(ROOT, "pele/lj38-basinhopping")
    template["stop"] = dict(template.get("stop") or {})
    template["stop"]["max_force_calls"] = 200
    problem = resolve_problem(ROOT, "globalopt/lj38")
    output = tmp_path / "pele.dat"
    summary = run_template(
        ROOT,
        problem,
        template,
        output=output,
        workers=1,
        work=tmp_path / "work",
        limit=1,
    )
    assert summary["starts"] == 1
    assert summary["partial"]
    text = output.read_text(encoding="utf-8")
    assert "problem globalopt/lj38" in text
    assert "partial true" in text
    assert "counter harness" in text
    start = json.loads(output.with_suffix(".starts.jsonl").read_text(encoding="utf-8"))
    assert start["schema"] == "optbench.start/v2"
    assert start["counter"] == "harness"
    assert start["success"] is False
    assert start["failure"] == "cap"
    assert 0 < int(start["force_calls"]) <= 200
    assert start["commit"] == COMMIT
