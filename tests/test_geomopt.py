"""OptKing and PyBerny templates and a short harness minimization."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run

ROOT = Path(__file__).resolve().parents[1]
OPTKING_COMMIT = "767e2c0c13ff2b851ac82101ccd5aab9a558a4f1"
OPTKING_VERSION = "0.5.1.dev2+g767e2c0c1"
PYBERNY_COMMIT = "2c732230c543455dc4c318fed8ab8cb7b5f9face"
PYBERNY_VERSION = "0.7.0.post2.dev0+2c73223"


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_geomopt_templates_match_the_registry_pins():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pins = {row["id"]: row for row in registry["code"]}
    assert pins["optking"]["commit"] == OPTKING_COMMIT
    assert pins["pyberny"]["commit"] == PYBERNY_COMMIT
    found = [
        row
        for row in list_templates(ROOT)
        if str(row["id"]).startswith("optking/") or str(row["id"]).startswith("pyberny/")
    ]
    assert {row["id"] for row in found} == {
        "optking/lj38-rfo-cartesian",
        "optking/lj38-rfo",
        "pyberny/lj38-berny",
    }
    for template in found:
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        assert template["tuning"] == "default"
        assert template["problem"] == "minimization/lj38"
        if template["engine"] == "optking":
            assert template["commit"] == OPTKING_COMMIT
            assert template["code_version"] == OPTKING_VERSION
            assert template["optking"]["symbol"] == "H"
            assert template["optking"]["step_type"] == "RFO"
        else:
            assert template["commit"] == PYBERNY_COMMIT
            assert template["code_version"] == PYBERNY_VERSION
            assert template["pyberny"]["symbol"] == "H"
            assert template["pyberny"]["dihedral"] is True
    by_id = {row["id"]: row for row in found}
    assert by_id["optking/lj38-rfo-cartesian"]["optking"]["coordinates"] == "CARTESIAN"
    assert by_id["optking/lj38-rfo"]["optking"]["coordinates"] == "REDUNDANT"
    assert by_id["optking/lj38-rfo-cartesian"]["algorithm"] == "Cartesian RFO"
    assert by_id["optking/lj38-rfo"]["algorithm"] == "RFO"


def test_geomopt_engines_do_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["resume"] = kwargs["resume"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 0,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.geomopt_min.run_template", fake)
    for name in ("optking/lj38-rfo-cartesian", "pyberny/lj38-berny"):
        summary = run(
            ROOT,
            "minimization/lj38",
            name,
            output=tmp_path / "benchmark.dat",
            workers=2,
            resume=True,
        )
        assert captured["template"] == name
        assert captured["problem"] == "minimization/lj38"
        assert captured["workers"] == 2
        assert captured["resume"] is True
        assert summary["template"] == name


@pytest.mark.skipif(not _installed("optking"), reason="optking is not installed")
def test_cartesian_rfo_descends_on_a_dimer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from optbench.potentials import lennard_jones
    from optbench.runners.geomopt_min import _minimize_job

    start = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    initial, _gradient = lennard_jones(start)
    record = _minimize_job(
        {
            "index": 0,
            "start": "dimer",
            "coords": start,
            "engine": "optking",
            "options": {"coordinates": "CARTESIAN", "step_type": "RFO", "symbol": "H"},
            "gtol": 0.01,
            "cap": 25,
        }
    )
    assert record["schema"] == "optbench.start/v2"
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["force_calls"] >= 1
    assert record["force_calls"] < 25
    assert record["failure"] != "crash"
    assert record["energy"] < float(initial)
    assert not (ROOT / "opt_log.out").exists()
    assert not (tmp_path / "opt_log.out").exists()


@pytest.mark.skipif(not _installed("berny"), reason="pyberny is not installed")
def test_pyberny_records_a_short_cluster(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from optbench.runners.geomopt_min import _minimize_job

    record = _minimize_job(
        {
            "index": 0,
            "start": "tetra",
            "coords": [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.7, 1.2, 0.0, 0.7, 0.4, 1.1],
            "engine": "pyberny",
            "options": {"symbol": "H", "dihedral": True},
            "gtol": 0.01,
            "cap": 4,
        }
    )
    assert record["schema"] == "optbench.start/v2"
    assert record["counter"] == "harness"
    assert record["force_calls"] >= 1
    assert record["failure"] in {None, "crash", "divergence", "cap", "wrong_stationary"}


@pytest.mark.skipif(not _installed("optking"), reason="optking is not installed")
def test_noise_optking_accepts_positive_sigma(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from optbench.runners.noise_min import _minimize_job

    record = _minimize_job(
        {
            "index": 0,
            "start": "dimer",
            "coords": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            "family": "optking",
            "method": "",
            "optimizer": "",
            "options": {"coordinates": "CARTESIAN", "step_type": "RFO", "symbol": "H"},
            "sigma": 1e-4,
            "seed": 20261016,
            "clean_tol": 0.01,
            "energy_tol": 0.01,
            "cap": 2,
        }
    )
    assert record["counter"] == "harness"
    assert record["sigma"] == 1e-4
    assert record["force_calls"] >= 1
    assert record["failure"] != "crash" or "error" in record
