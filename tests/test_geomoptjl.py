"""GeometryOptimization.jl templates and the harness callback."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.geomoptjl_catalog import COMMIT, VERSION, specs
from optbench.runners.geomoptjl_min import CODE, manifest_pin
from optbench.runners.geomoptjl_session import Session

ROOT = Path(__file__).resolve().parents[1]


def test_templates_match_the_catalog_and_the_manifest_pin():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "geomoptjl")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == CODE
    assert manifest_pin(ROOT) == COMMIT
    catalog = specs()
    assert len(catalog) == 18
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("geomoptjl/")]
    assert {row["id"] for row in found} == {row["id"] for row in catalog}
    by_id = {row["id"]: row for row in found}
    for spec in catalog:
        template = by_id[spec["id"]]
        assert template["engine"] == "geomoptjl"
        assert template["commit"] == COMMIT
        assert template["code_version"] == VERSION
        assert template["code"] == CODE
        assert template["tuning"] == "default"
        assert template["geomoptjl"]["method"] == spec["method"]
        assert template["algorithm"] == spec["algorithm"]
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        if spec["sigma"] is None:
            assert "noise" not in template
        else:
            assert template["noise"]["sigma"] == spec["sigma"]
            assert template["noise"]["seed"] == 20261016
    assert {row["geomoptjl"]["method"] for row in found} == {"lbfgs", "cg", "gradientdescent"}


def test_repeated_coordinate_is_one_harness_call():
    session = Session(
        {
            "kind": "lj",
            "method": "lbfgs",
            "start": "pair",
            "index": 0,
            "coords": [0.0, 0.0, 0.0, 0.0, 0.0, 1.2],
            "box": None,
            "cap": 10,
            "gtol": 1e-6,
            "noise": False,
            "sigma": 0.0,
            "seed": 1,
            "energy_tol": 0.01,
        }
    )
    first = session.call(session.job_coords())
    second = session.call(session.job_coords())
    assert first["cap"] is False
    assert second["energy"] == first["energy"]
    assert session.calls() == 1
    shifted = [value + (0.05 if index == 5 else 0.0) for index, value in enumerate(session.job_coords())]
    third = session.call(shifted)
    assert third["cap"] is False
    assert session.calls() == 2


def test_cap_refuses_the_next_search_call():
    session = Session(
        {
            "kind": "lj",
            "method": "cg",
            "start": "pair",
            "index": 0,
            "coords": [0.0, 0.0, 0.0, 0.0, 0.0, 1.2],
            "box": None,
            "cap": 1,
            "gtol": 1e-12,
            "noise": False,
            "sigma": 0.0,
            "seed": 1,
            "energy_tol": 0.01,
        }
    )
    session.call(session.job_coords())
    shifted = [value + (0.2 if index == 5 else 0.0) for index, value in enumerate(session.job_coords())]
    refused = session.call(shifted)
    assert refused["cap"] is True
    assert session.calls() == 1


def _julia_present() -> bool:
    if shutil.which("julia"):
        return True
    return Path.home().joinpath("opt/julia/bin/julia").is_file()


@pytest.mark.skipif(not _julia_present(), reason="julia is not installed")
def test_lj38_lbfgs_start_meets_the_force_stop():
    from optbench.campaign import resolve_problem
    from optbench.runners.geomoptjl_min import JuliaWorker, _jobs
    from optbench.runners.rgmin_min import _load_geometries

    problem = resolve_problem(ROOT, "minimization/lj38")
    loaded = _load_geometries(ROOT, problem, None, Path("/tmp/optbench-geo"), Path("/tmp/optbench-geo/unused.dat"))
    template = next(row for row in list_templates(ROOT) if row["id"] == "geomoptjl/lj38-lbfgs")
    jobs, sigma = _jobs(problem, template, [loaded[0]], "lbfgs")
    assert sigma is None
    worker = JuliaWorker(ROOT)
    try:
        record = worker.run(jobs[0])
    finally:
        worker.close()
    assert record["counter"] == "harness"
    assert record["success"] is True
    assert record["failure"] is None
    assert record["force_calls"] < 10000
    assert record["force_norm"] <= 0.01
    assert record["hessian_index"] == 0
    assert record["geomopt_reason"] == "tolerance"
