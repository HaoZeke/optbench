"""TSASE global-optimization helpers. The searches themselves are not run here."""

import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.runners.tsase_global import (
    PUBLISHED_LJ38_ENERGY,
    describe_parameters,
    energy_threshold,
    judge_global,
    lj38_minimum_energy,
    load_geometries,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "98aac60b5d9905e81ee4e6b8c385f169258d0218"


def test_published_lj38_minimum_matches_the_harness_potential():
    energy = lj38_minimum_energy()
    assert abs(energy - PUBLISHED_LJ38_ENERGY) < 1e-6
    problem = resolve_problem(ROOT, "globalopt/lj38")
    threshold = energy_threshold(problem)
    assert threshold == pytest.approx(energy + 0.001)


def test_lj75_has_no_energy_window_on_this_main():
    problem = resolve_problem(ROOT, "globalopt/lj75")
    assert energy_threshold(problem) is None


def test_success_is_the_energy_window_under_the_cap():
    success, failure, _reasons = judge_global(
        calls=10,
        cap=100,
        energy=-173.928,
        threshold=-173.927,
        crashed=False,
        exploded=False,
    )
    assert success is True
    assert failure is None
    success, failure, _reasons = judge_global(
        calls=100,
        cap=100,
        energy=-173.928,
        threshold=-173.927,
        crashed=False,
        exploded=False,
    )
    assert success is False
    assert failure == "cap"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-170.0,
        threshold=-173.927,
        crashed=False,
        exploded=False,
    )
    assert success is False
    assert failure == "wrong_stationary"


def test_an_unruled_start_is_not_a_success():
    success, failure, _reasons = judge_global(
        calls=12,
        cap=100,
        energy=-300.0,
        threshold=None,
        crashed=False,
        exploded=False,
    )
    assert success is False
    assert failure == "cap"


def test_templates_use_the_registry_pin_and_the_problem_stop():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "tsase")
    assert pinned["commit"] == COMMIT
    found = [
        row
        for row in list_templates(ROOT)
        if row.get("code") == "tsase" and str(row.get("problem") or "").startswith("globalopt/")
    ]
    assert {row["id"] for row in found} == {
        "tsase/lj38-basinhopping",
        "tsase/lj38-basinhopping-tuned",
        "tsase/lj38-minimahopping",
        "tsase/lj75-basinhopping",
        "tsase/lj75-basinhopping-tuned",
        "tsase/lj75-minimahopping",
    }
    for template in found:
        assert template["engine"] == "tsase"
        assert template["tsase"]["method"] in {"basin_hopping", "minima_hopping"}
        assert template["commit"] == COMMIT
        assert template["code_version"] == "1.0"
        assert template["tuning"] in {"default", "tuned"}
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        assert int(template["stop"]["max_force_calls"]) == 2_000_000
    tuned = next(row for row in found if row["id"] == "tsase/lj38-basinhopping-tuned")
    described = describe_parameters(tuned["tsase"], -173.927)
    assert described["fmax"] == 0.01
    assert described["dr"] == 0.3
    assert described["temperature"] == pytest.approx(8000 * 8.617333262145e-5)
    default = next(row for row in found if row["id"] == "tsase/lj38-basinhopping")
    described = describe_parameters(default["tsase"], -173.927)
    assert described["fmax"] == 0.1
    assert described["dr"] == 0.1


def test_lj75_archive_has_two_hundred_starts():
    problem = resolve_problem(ROOT, "globalopt/lj75")
    rows = load_geometries(ROOT, problem)
    assert len(rows) == 200
    assert rows[0][0] == "coords.1"
    assert rows[0][1].shape == (75, 3)
    assert rows[-1][0] == "coords.200"


def test_lj38_archive_has_the_hundred_starts():
    problem = resolve_problem(ROOT, "globalopt/lj38")
    rows = load_geometries(ROOT, problem)
    assert len(rows) == 100
    assert rows[0][0] == "0"
    assert rows[0][1].shape == (38, 3)
    assert rows[-1][0] == "99"
