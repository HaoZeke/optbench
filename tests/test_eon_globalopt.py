"""Cambridge basin-hopping rows for the eOn harness."""

import math

import numpy as np
import pytest

from optbench.campaign import load_template
from optbench.eon_harness import (
    _cache_path,
    blj20,
    classify_cambridge,
    coords_to_con,
    template_starts,
)
from optbench.globalopt import energy_tolerance, force_cap, reference_energy
from optbench.potentials import binary_lennard_jones

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

STOCK = (
    "eon-svn/bh-lj75",
    "eon-svn/bh-lj98",
    "eon-svn/bh-lj150",
    "eon-svn/bh-lj100-binary",
    "eon-develop/bh-lj75",
    "eon-develop/bh-lj98",
    "eon-develop/bh-lj150",
    "eon-develop/bh-lj100-binary",
)


def test_coords_to_con_keeps_every_atom():
    text = "0 0 0\n1 0 0\n0 1 0\n"
    con = coords_to_con(text)
    assert "\n3\n" in con
    assert con.count("\n  ") == 3


def test_binary_kernel_matches_the_archived_type_order():
    sigma = 1.15 * 2.0 ** (1.0 / 6.0)
    positions = np.array([[0.0, 0.0, 0.0], [sigma, 0.0, 0.0]])
    energy, forces = blj20(positions, [1, 6])
    reference, gradient = binary_lennard_jones(
        positions,
        [1, 0],
        sigma_aa=1.0,
        sigma_bb=1.3,
        epsilon_aa=1.0,
        epsilon_bb=1.0,
        epsilon_ab=1.0,
        cutoff=20.0,
    )
    assert energy == pytest.approx(reference)
    assert forces == pytest.approx(tuple(float(value) for value in np.ravel(-gradient)))
    with pytest.raises(ValueError):
        blj20(positions, [1, 8])


def test_logged_basin_minimum_reads_the_global_min_column(tmp_path):
    from optbench.eon_harness import _logged_basin_minimum

    (tmp_path / "client.log").write_text(
        "\n".join(
            [
                "[Basin Hopping] step      current        trial   global min   fc    ar    md",
                "[Basin Hopping]     1     -381.399     -381.399     -381.399  302 1.000 0.500 A",
                "[Basin Hopping]   975     -394.741     -396.282     -396.282  180 0.350 0.420 A",
                "[Basin Hopping]  1200     -395.100     -390.000     -396.282  200 0.340 0.420 R",
            ]
        ),
        encoding="utf-8",
    )
    assert _logged_basin_minimum(tmp_path) == pytest.approx(-396.282)


def test_cambridge_cap_precedes_a_missing_energy():
    row = classify_cambridge(
        calls=10,
        cap=10,
        energy=None,
        returncode=1,
        reference=-1.0,
        tolerance=0.001,
    )
    assert row["failure"] == "cap"
    assert row["success"] is False


def test_cambridge_clean_exit_uses_the_energy_rule():
    hit = classify_cambridge(
        calls=12,
        cap=100,
        energy=-1.0005,
        returncode=0,
        reference=-1.0,
        tolerance=0.001,
    )
    assert hit["success"] is True
    assert hit["failure"] is None
    high = classify_cambridge(
        calls=12,
        cap=100,
        energy=-0.5,
        returncode=0,
        reference=-1.0,
        tolerance=0.001,
    )
    assert high["success"] is False
    assert high["failure"] == "wrong_stationary"
    crashed = classify_cambridge(
        calls=12,
        cap=100,
        energy=None,
        returncode=0,
        reference=-1.0,
        tolerance=0.001,
    )
    assert crashed["failure"] == "crash"


def test_stock_templates_copy_the_lj38_basin_settings():
    expected = {
        "eon-svn/bh-lj75": ("1", 200),
        "eon-develop/bh-lj75": ("1", 200),
        "eon-svn/bh-lj98": ("1", 100),
        "eon-develop/bh-lj98": ("1", 100),
        "eon-svn/bh-lj150": ("1", 100),
        "eon-develop/bh-lj150": ("1", 100),
        "eon-svn/bh-lj100-binary": ("0", 100),
        "eon-develop/bh-lj100-binary": ("0", 100),
    }
    for name in STOCK:
        template = load_template(ROOT, name)
        problem = __import__("optbench.campaign", fromlist=["resolve_problem"]).resolve_problem(
            ROOT, template["problem"]
        )
        assert template["cambridge"] is True
        assert template["tuning"] == "lj38"
        assert template["cache_token"] == "cambridge-1"
        assert template.get("budget_seconds") in (None, 0)
        assert int(template["failure_cap"]) == force_cap(problem)
        stop = float(template["ini"]["Basin Hopping"]["stop_energy"])
        assert stop == pytest.approx(reference_energy(problem) + energy_tolerance(problem), abs=0.0, rel=0.0)
        assert math.isclose(stop, reference_energy(problem) + 0.001, abs_tol=0.0, rel_tol=0.0)
        basin = template["ini"]["Basin Hopping"]
        assert basin["displacement"] == 0.5
        assert basin["target_ratio"] == 0.3
        assert basin["adjust_displacement"] is True
        assert template["ini"]["Main"]["temperature"] == 8000
        assert template["ini"]["Main"]["random_seed"] == 314159
        assert template["seeds"] == [314159]
        assert "displacement_distribution" not in basin
        first, count = expected[name]
        starts = template_starts(template)
        assert starts[0] == first
        assert len(starts) == count
        assert len(starts) == int(problem["sample"]["starts"])


def test_cambridge_cache_token_does_not_move_the_lj38_path():
    lj38 = _cache_path("svn", "eon-svn/bh-lj38", 314159, "0", 0)
    assert "cambridge-1" not in lj38.parts
    lj75 = _cache_path("svn", "eon-svn/bh-lj75", 314159, "1", 0)
    assert "cambridge-1" in lj75.parts
    develop = _cache_path("develop", "eon-develop/bh-lj75", 314159, "1", 0)
    assert develop.parts[2] == "7095b23" or "7095b23" in develop.parts


def test_asv_default_is_the_first_real_start():
    from benchmarks.bench_eon_go import EonDevelopBhLj75, EonDevelopBhLj100Binary
    from benchmarks.bench_eon_lane import EonSvnBhLj75, EonSvnBhLj98, EonSvnBhLj150, EonSvnBhLj100Binary

    assert EonSvnBhLj75.params == [[314159], ["1"]]
    assert EonSvnBhLj98.params == [[314159], ["1"]]
    assert EonSvnBhLj150.params == [[314159], ["1"]]
    assert EonSvnBhLj100Binary.params == [[314159], ["0"]]
    assert EonDevelopBhLj75.params == [[314159], ["1"]]
    assert EonDevelopBhLj100Binary.params == [[314159], ["0"]]
    assert EonSvnBhLj75.cap == 1000000
    assert EonDevelopBhLj75.template_name == "eon-develop/bh-lj75"
    assert EonSvnBhLj75.template_name == "eon-svn/bh-lj75"
