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
    capped_lennard_jones,
    judge_global,
    objective_report,
    reference_minimum,
    search_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "88c424d4f458412787df96fcc95218acbca224fd"
CUBE_TEMPLATES = (
    "nloptglobal/lj38-mlsl-lbfgs",
    "nloptglobal/lj38-crs2",
    "nloptglobal/lj38-isres",
    "nloptglobal/lj38-stogo",
)
HOP_CAPS = {
    "nloptglobal/lj38-lbfgs-bh": 1_000_000,
    "nloptglobal/lj75-lbfgs-bh": 1_000_000,
    "nloptglobal/lj98-lbfgs-bh": 1_000_000_000,
    "nloptglobal/lj150-lbfgs-bh": 10_000_000,
}


def test_templates_pin_the_registry_commit():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "nloptglobal")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == "NLopt"
    assert pinned["ref"] == "v2.11.0"
    ids = {row["id"] for row in registry["template"]}
    for name in (*CUBE_TEMPLATES, *HOP_CAPS):
        template = load_template(ROOT, name)
        assert template["commit"] == COMMIT
        assert template["code_version"] == "2.11.0"
        assert template["code"] == "NLopt"
        assert template["engine"] == "nlopt"
        assert float(template["stop"]["energy_above_minimum"]) == 0.001
        assert name in ids
        if name in HOP_CAPS:
            assert int(template["stop"]["max_force_calls"]) == HOP_CAPS[name]
            assert float(template["nlopt"]["displacement"]) == 0.5
            assert float(template["nlopt"]["temperature"]) == 0.8
            assert float(template["nlopt"]["maxstep"]) == 0.2
            assert float(template["nlopt"]["core_radius"]) == 0.9
        else:
            assert template["problem"] == "globalopt/lj38"
            assert int(template["stop"]["max_force_calls"]) == 1_000_000
            assert float(template["domain"]["half_width"]) == 6.0


def test_nlopt_templates_are_the_files_on_disk():
    files = sorted("nloptglobal/" + path.stem for path in (ROOT / "templates" / "nloptglobal").glob("*.toml"))
    found = [row["id"] for row in list_templates(ROOT) if row.get("engine") == "nlopt"]
    assert found == files
    assert set(CUBE_TEMPLATES).issubset(files)
    assert set(HOP_CAPS).issubset(files)


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


def test_the_cambridge_rule_is_at_or_below_the_minimum():
    reference = -173.928427
    tolerance = 0.001
    success, failure, reasons = judge_global(
        calls=12,
        cap=100,
        energy=reference + tolerance,
        finite=True,
        nlopt_code=2,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert success is True
    assert failure is None
    assert reasons == []
    below, below_failure, _reasons = judge_global(
        calls=12,
        cap=100,
        energy=reference - 1.0,
        finite=True,
        nlopt_code=2,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert below is True
    assert below_failure is None


def test_the_budget_is_a_cap_and_a_high_energy_is_the_wrong_point():
    reference = -173.928427
    tolerance = 0.001
    success, failure, _reasons = judge_global(
        calls=100,
        cap=100,
        energy=-160.0,
        finite=True,
        nlopt_code=5,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert success is False
    assert failure == "cap"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-164.0,
        finite=True,
        nlopt_code=4,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert failure == "wrong_stationary"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-80.0,
        finite=True,
        nlopt_code=1,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert failure == "wrong_stationary"
    success, failure, _reasons = judge_global(
        calls=3,
        cap=100,
        energy=None,
        finite=False,
        nlopt_code=-1,
        error="RuntimeError: boom",
        reference=reference,
        tolerance=tolerance,
    )
    assert failure == "crash"


def test_a_coincident_pair_stays_a_candidate_with_a_large_finite_energy():
    from optbench.potentials import COINCIDENT_PAIR_ENERGY

    stacked = np.zeros(6)
    raw, grad = lennard_jones(stacked)
    assert raw == pytest.approx(COINCIDENT_PAIR_ENERGY)
    assert np.all(np.isfinite(grad))
    reported, reported_grad, usable = objective_report(float(raw), grad, stacked)
    assert usable is True
    assert reported == pytest.approx(COINCIDENT_PAIR_ENERGY)
    assert reported_grad.shape == stacked.shape
    energy, _force, coords = reference_minimum()
    kept, kept_grad, keep = objective_report(energy, np.zeros(coords.size), coords)
    assert keep is True
    assert kept == pytest.approx(energy)
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
    from optbench.globalopt import reference_energy

    reference = reference_energy(problem)
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
        "reference": reference,
        "tolerance": settings["window"],
    }
    record = run_job(job)
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["schema"] == "optbench.start/v2"
    assert 0 < record["force_calls"] <= 30
    assert record["failure"] in {"cap", "divergence", "crash", "wrong_stationary"}
    assert record["success"] is False


def test_asv_classes_follow_the_templates():
    import benchmarks.bench_nloptglobal as bench

    found = [row for row in list_templates(ROOT) if row.get("engine") == "nlopt"]
    assert {row["id"] for row in found} >= set(CUBE_TEMPLATES) | set(HOP_CAPS)
    for template in found:
        slug = str(template["id"]).split("/", 1)[1]
        name = "Nlopt" + "".join(part.capitalize() for part in slug.split("-"))
        cls = getattr(bench, name)
        assert cls.template_id == template["id"]
        assert cls.problem_id == template["problem"]
        assert cls.cap == int(template["stop"]["max_force_calls"])
        assert cls.params == [["0"], ["0"]]


def test_the_core_cap_matches_the_harness_beyond_the_join():
    energy, _force, coords = reference_minimum()
    capped, grad, minimum = capped_lennard_jones(coords, 0.9)
    assert minimum > 0.9
    assert capped == pytest.approx(energy, abs=1e-12)
    raw, raw_grad = lennard_jones(coords)
    assert np.max(np.abs(grad - raw_grad)) < 1e-8
    pair = np.zeros(6)
    pair[3] = 1.2
    capped, grad, _minimum = capped_lennard_jones(pair, 0.9)
    raw, raw_grad = lennard_jones(pair)
    assert capped == pytest.approx(raw)
    assert np.allclose(grad, raw_grad)
    pair[3] = 0.4
    capped, grad, _minimum = capped_lennard_jones(pair, 0.9)
    assert math.isfinite(capped)
    assert capped < 1000.0
    eps = 1e-6
    finite = np.zeros_like(grad)
    for index in range(pair.size):
        pair[index] += eps
        above, _grad, _minimum = capped_lennard_jones(pair, 0.9)
        pair[index] -= 2.0 * eps
        below, _grad, _minimum = capped_lennard_jones(pair, 0.9)
        pair[index] += eps
        finite[index] = (above - below) / (2.0 * eps)
    assert np.max(np.abs(finite - grad)) < 1e-6


def test_a_polished_basin_is_a_cambridge_success():
    pytest.importorskip("nlopt")
    from optbench.campaign import resolve_problem
    from optbench.globalopt import reference_energy
    from optbench.runners.nlopt_global import _job, _settings, run_job

    template = load_template(ROOT, "nloptglobal/lj38-lbfgs-bh")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    settings = _settings(problem, template)
    _energy, _force, coords = reference_minimum()
    rng = np.random.default_rng(2)
    trial = coords + rng.normal(scale=0.05, size=coords.size)
    trial = trial.reshape(-1, 3)
    trial -= trial.mean(axis=0)
    reference = reference_energy(problem)
    job = _job(0, "gm", trial.reshape(-1), settings, reference, settings["window"])
    job["cap"] = 300
    record = run_job(job)
    assert record["success"] is True
    assert record["failure"] is None
    assert record["energy"] <= reference + 0.001
    assert record["energy"] == pytest.approx(-173.92842659062865, abs=1e-6)
    assert 0 < record["force_calls"] < 300
    assert record["verify_calls"] == 1
    assert record["counter"] == "harness"
