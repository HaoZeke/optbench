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
    assert threshold == pytest.approx(-173.928427 + 0.001)
    assert abs(energy - (-173.928427)) < 1e-5


def test_lj75_window_is_the_cambridge_minimum():
    problem = resolve_problem(ROOT, "globalopt/lj75")
    assert energy_threshold(problem) == pytest.approx(-397.492331 + 0.001)
    assert int(problem["convergence"]["max_force_calls"]) == 1_000_000
    assert int(problem["sample"]["starts"]) == 200


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


def test_a_stalled_line_search_is_not_divergence():
    success, failure, reasons = judge_global(
        calls=3,
        cap=1_000_000,
        energy=3.72e6,
        threshold=-397.491331,
        crashed=False,
        exploded=False,
        stalled=True,
    )
    assert success is False
    assert failure == "stalled"
    assert reasons == ["stalled"]
    from optbench.score import penalised_cost

    assert penalised_cost(3, success, 1_000_000) == 2_000_000


def test_an_exploded_energy_is_divergence():
    success, failure, _reasons = judge_global(
        calls=4,
        cap=100,
        energy=1.0e8,
        threshold=None,
        crashed=False,
        exploded=True,
    )
    assert success is False
    assert failure == "divergence"


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
        "tsase/lj38-minimahopping",
        "tsase/lj75-basinhopping",
        "tsase/lj75-minimahopping",
    }
    for template in found:
        assert template["engine"] == "tsase"
        assert template["tsase"]["method"] in {"basin_hopping", "minima_hopping"}
        assert template["commit"] == COMMIT
        assert template["code_version"] == "1.0"
        assert template["tuning"] == "default"
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        expected_cap = 1_000_000 if template["problem"] == "globalopt/lj75" else 2_000_000
        assert int(template["stop"]["max_force_calls"]) == expected_cap
    default = next(row for row in found if row["id"] == "tsase/lj38-basinhopping")
    described = describe_parameters(default["tsase"], -173.927)
    assert described["fmax"] == 0.1
    assert described["dr"] == 0.1


def test_stock_quasinewton_stalls_on_an_overlapped_lj75_start():
    """BFGSLineSearch repeats one clamped step. The harness records the stall."""
    import numpy as np
    from ase.optimize import QuasiNewton

    from optbench.counter import HarnessCounter
    from optbench.runners.tsase_global import Stalled, _attach, _import_tsase, _make_atoms

    _import_tsase()
    geometries = dict(load_geometries(ROOT, resolve_problem(ROOT, "globalopt/lj75")))
    counter = HarnessCounter(cache=True, cap=8000)
    atoms = _make_atoms(geometries["coords.44"])
    _attach(atoms, counter, None)
    with pytest.raises(Stalled):
        QuasiNewton(atoms, logfile=None, trajectory=None).run(fmax=0.05, steps=4000)
    energy = float(counter.lowest_energy)
    assert counter.calls < 20
    assert np.isfinite(energy)
    success, failure, _reasons = judge_global(
        calls=counter.calls,
        cap=1_000_000,
        energy=energy,
        threshold=-397.491331,
        crashed=False,
        exploded=False,
        stalled=True,
    )
    assert success is False
    assert failure == "stalled"


def test_tsase_rows_are_labelled_with_the_short_commit():
    from optbench.board import _version_label

    label = _version_label(
        {
            "code": "tsase",
            "code_version": "1.0",
            "commit": "98aac60b5d9905e81ee4e6b8c385f169258d0218",
        }
    )
    assert label == "TSASE 1.0 (98aac60)"


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
