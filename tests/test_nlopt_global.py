"""NLopt global-optimization templates and the Lennard-Jones 38 judge."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import list_templates, load_template
from optbench.potentials import lennard_jones, read_con
from optbench.runners.nlopt_global import (
    judge_global,
    objective_report,
    reference_minimum,
    search_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "88c424d4f458412787df96fcc95218acbca224fd"
TEMPLATES = (
    "nloptglobal/lj38-mlsl-lbfgs",
    "nloptglobal/lj38-crs2",
    "nloptglobal/lj38-isres",
    "nloptglobal/lj38-stogo",
)


def test_templates_pin_the_registry_commit():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "nloptglobal")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == "NLopt"
    assert pinned["ref"] == "v2.11.0"
    ids = {row["id"] for row in registry["template"]}
    for name in TEMPLATES:
        template = load_template(ROOT, name)
        assert template["commit"] == COMMIT
        assert template["code_version"] == "2.11.0"
        assert template["code"] == "NLopt"
        assert template["problem"] == "globalopt/lj38"
        assert template["engine"] == "nlopt"
        assert int(template["stop"]["max_force_calls"]) == 1_000_000
        assert float(template["stop"]["energy_above_minimum"]) == 0.001
        assert float(template["domain"]["half_width"]) == 6.0
        assert name in ids


def test_four_nlopt_templates_are_the_global_set():
    found = [row["id"] for row in list_templates(ROOT) if row.get("engine") == "nlopt"]
    assert found == sorted(TEMPLATES)


def test_reference_minimum_matches_the_polished_cluster():
    energy, force, coords = reference_minimum()
    assert energy == pytest.approx(-173.92842659062865, abs=1e-9)
    assert force < 1e-4
    assert coords.size == 114
    again, gradient = lennard_jones(coords)
    assert again == pytest.approx(energy)
    assert float(np.linalg.norm(gradient)) == pytest.approx(force)


def test_every_archived_start_and_the_minimum_fit_in_the_cube():
    import tarfile

    from optbench.campaign import geometry_archive, resolve_problem

    _energy, _force, minimum = reference_minimum()
    minimum = minimum.reshape(-1, 3)
    minimum = minimum - minimum.mean(axis=0)
    archive = geometry_archive(ROOT, resolve_problem(ROOT, "globalopt/lj38"))
    with tarfile.open(archive) as packed:
        names = sorted(
            (name for name in packed.getnames() if name.endswith(".con")),
            key=lambda name: int(Path(name).stem),
        )
        assert len(names) == 100
        for name in names:
            text = packed.extractfile(name).read().decode("utf-8", errors="replace")
            pos, _box = read_con(text)
            flat = np.asarray(pos, dtype=np.float64).reshape(-1)
            lower, upper = search_bounds(flat, 6.0)
            assert np.all(flat >= lower - 1e-8)
            assert np.all(flat <= upper + 1e-8)
            center = flat.reshape(-1, 3).mean(axis=0)
            placed = (minimum + center).reshape(-1)
            assert np.all(placed >= lower - 1e-8)
            assert np.all(placed <= upper + 1e-8)


def test_energy_inside_the_window_and_under_the_cap_is_a_success():
    success, failure, reasons = judge_global(
        calls=12,
        cap=100,
        energy=-173.928,
        force_norm=0.2,
        finite=True,
        nlopt_code=2,
        error="",
        limit=-173.92742659062864,
        floor=-173.92942659062865,
        stationary_force=0.01,
    )
    assert success is True
    assert failure is None
    assert reasons == []


def test_the_budget_is_a_cap_and_a_tight_local_minimum_is_the_wrong_point():
    success, failure, _reasons = judge_global(
        calls=100,
        cap=100,
        energy=-160.0,
        force_norm=1e-4,
        finite=True,
        nlopt_code=5,
        error="",
        limit=-173.92742659062864,
        floor=-173.92942659062865,
        stationary_force=0.01,
    )
    assert success is False
    assert failure == "cap"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-164.0,
        force_norm=1e-3,
        finite=True,
        nlopt_code=4,
        error="",
        limit=-173.92742659062864,
        floor=-173.92942659062865,
        stationary_force=0.01,
    )
    assert failure == "wrong_stationary"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-80.0,
        force_norm=20.0,
        finite=True,
        nlopt_code=1,
        error="",
        limit=-173.92742659062864,
        floor=-173.92942659062865,
        stationary_force=0.01,
    )
    assert failure == "divergence"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-182.9,
        force_norm=1e-4,
        finite=True,
        nlopt_code=1,
        error="",
        limit=-173.92742659062864,
        floor=-173.92942659062865,
        stationary_force=0.01,
    )
    assert success is False
    assert failure == "wrong_stationary"
    success, failure, _reasons = judge_global(
        calls=3,
        cap=100,
        energy=None,
        force_norm=None,
        finite=False,
        nlopt_code=-1,
        error="RuntimeError: boom",
        limit=-173.92742659062864,
        floor=-173.92942659062865,
        stationary_force=0.01,
    )
    assert failure == "crash"


def test_coincident_sites_are_outside_the_potential():
    energy, _force, coords = reference_minimum()
    pos = coords.reshape(-1, 3).copy()
    pos[1] = pos[0]
    raw, grad = lennard_jones(pos)
    reported, reported_grad, usable = objective_report(float(raw), grad, pos.reshape(-1))
    assert math.isfinite(raw)
    assert usable is False
    assert reported == pytest.approx(raw)
    assert reported_grad.shape == (coords.size,)
    kept, kept_grad, keep = objective_report(-180.0, np.zeros(coords.size), coords)
    assert keep is True
    assert kept == -180.0
    assert kept_grad.shape == (coords.size,)


def test_a_short_mlsl_start_is_counted_by_the_harness():
    nlopt = pytest.importorskip("nlopt")
    assert (nlopt.version_major(), nlopt.version_minor(), nlopt.version_bugfix()) == (2, 11, 0)
    import tarfile

    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.runners.nlopt_global import _settings, run_job

    template = load_template(ROOT, "nloptglobal/lj38-mlsl-lbfgs")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    settings = _settings(problem, template)
    archive = geometry_archive(ROOT, problem)
    with tarfile.open(archive) as packed:
        text = packed.extractfile("100_lj38/0.con").read().decode("utf-8", errors="replace")
    pos, _box = read_con(text)
    reference, _force, _coords = reference_minimum()
    job = {
        "index": 0,
        "start": "0",
        "coords": np.asarray(pos, dtype=np.float64).reshape(-1).tolist(),
        "seed": 0,
        "algorithm": "GN_MLSL",
        "local": "LD_LBFGS",
        "local_ftol_rel": 1e-15,
        "local_xtol_rel": 1e-7,
        "population": None,
        "half_width": settings["half_width"],
        "cap": 30,
        "limit": reference + settings["window"],
        "floor": reference - settings["window"],
        "stationary_force": settings["stationary_force"],
    }
    record = run_job(job)
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["schema"] == "optbench.start/v2"
    assert 0 < record["force_calls"] <= 30
    assert record["failure"] in {"cap", "divergence", "crash", "wrong_stationary"}
    assert record["success"] is False


def test_asv_classes_name_the_four_templates():
    import benchmarks.bench_nloptglobal as bench

    expected = {
        "NloptLj38MlslLbfgs": "nloptglobal/lj38-mlsl-lbfgs",
        "NloptLj38Crs2": "nloptglobal/lj38-crs2",
        "NloptLj38Isres": "nloptglobal/lj38-isres",
        "NloptLj38Stogo": "nloptglobal/lj38-stogo",
    }
    for name, template_id in expected.items():
        cls = getattr(bench, name)
        assert cls.template_id == template_id
        assert cls.problem_id == "globalopt/lj38"
        assert cls.cap == 1_000_000
        assert cls.params == [["0"], ["0"]]
