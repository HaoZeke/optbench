"""Cambridge basin-hopping rows for the eOn harness."""

import math
from pathlib import Path

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
        assert template["label"] == "target_ratio 0.3"
        assert template["label"] != "lj38"
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


def test_lj75_ratio_row_restores_the_documented_acceptance_target():
    for name, family in (
        ("eon-svn/bh-lj75-ratio", "svn"),
        ("eon-develop/bh-lj75-ratio", "develop"),
    ):
        template = load_template(ROOT, name)
        stock = load_template(ROOT, f"{name.split('/')[0]}/bh-lj75")
        assert template["tuning"] == "target-ratio-0.5"
        assert template["label"] == "target_ratio 0.5"
        assert template["cache_token"] == "cambridge-1-ratio-0.5"
        assert template["ini"]["Basin Hopping"]["target_ratio"] == 0.5
        assert template["ini"]["Main"]["temperature"] == stock["ini"]["Main"]["temperature"]
        assert template["ini"]["Basin Hopping"]["displacement"] == stock["ini"]["Basin Hopping"]["displacement"]
        assert template["failure_cap"] == stock["failure_cap"]
        assert template["ini"]["Basin Hopping"]["stop_energy"] == stock["ini"]["Basin Hopping"]["stop_energy"]
        assert len(template_starts(template)) == 200
        path = _cache_path(family, name, 314159, "1", 0)
        stock_path = _cache_path(family, stock["id"], 314159, "1", 0)
        assert "cambridge-1-ratio-0.5" in path.parts
        assert path != stock_path


def _read_con_atoms(text: str):
    """The nine-line header ``Matter::con2matter`` and readcon both accept."""
    lines = text.splitlines()
    assert len(lines) >= 9
    count = int(lines[6])
    sizes = [int(part) for part in lines[7].split()]
    masses = [float(part) for part in lines[8].split()]
    assert len(sizes) == count
    assert len(masses) == count
    cursor = 9
    atoms = []
    for component, size in enumerate(sizes, start=1):
        symbol = lines[cursor]
        assert lines[cursor + 1] == f"Coordinates of Component {component}"
        cursor += 2
        for _ in range(size):
            parts = lines[cursor].split()
            atoms.append((symbol, tuple(float(part) for part in parts[:3]), int(parts[3])))
            cursor += 1
    assert cursor == len(lines)
    return atoms


def test_aa_to_con_uses_the_shared_ccl_order():
    import tarfile

    from optbench.campaign import geometry_archive
    from optbench.eon_harness import aa_to_con
    from optbench.tip4p import ccl_from_sites, parse_aa, sites_from_aa

    archive = geometry_archive(
        ROOT, {"input": {"archive_path": "website/globalopt/tip4p10/10000_tip4p10.tar.gz"}}
    )
    with tarfile.open(archive) as packed:
        member = next(item for item in packed.getmembers() if Path(item.name).name == "1.aa")
        text = packed.extractfile(member).read().decode()
    positions, numbers = ccl_from_sites(sites_from_aa(parse_aa(text)))
    con = aa_to_con(text)
    atoms = _read_con_atoms(con)
    assert [symbol for symbol, _point, _fixed in atoms[:20]] == ["H"] * 20
    assert [symbol for symbol, _point, _fixed in atoms[20:]] == ["O"] * 10
    assert [int(number) for number in numbers.tolist()] == [1] * 20 + [8] * 10
    coords = positions.reshape(-1, 3)
    written = np.array([point for _symbol, point, _fixed in atoms])
    shifts = written - coords
    assert np.allclose(shifts, shifts[0], atol=1e-8)


def test_rigid_cartesian_force_matches_a_translation():
    from optbench.eon_harness import aa_to_con
    from optbench.tip4p import ccl_from_sites, parse_aa, rigid_cartesian, sites_from_aa

    aa = "0 0 0\n0 0 0\n3.2 0.4 -0.2\n0.1 -0.2 0.05\n"
    positions, numbers = ccl_from_sites(sites_from_aa(parse_aa(aa)))
    energy, forces = rigid_cartesian(positions.reshape(-1), numbers)
    step = 1e-6
    moved = positions.copy()
    # Molecule 0 is the first H, H pair and the first oxygen.
    moved[0] += np.array([step, 0.0, 0.0])
    moved[1] += np.array([step, 0.0, 0.0])
    moved[4] += np.array([step, 0.0, 0.0])
    moved_energy, _moved_forces = rigid_cartesian(moved.reshape(-1), numbers)
    molecule = forces.reshape(-1, 3)[[0, 1, 4]].sum(axis=0)
    assert (moved_energy - energy) / step == pytest.approx(-molecule[0], abs=1e-4)
    con = aa_to_con(aa)
    assert "\n2\n" in con
    assert "\n4 2\n" in con


def test_tip4p_templates_share_the_ledger_and_the_ev_stop():
    from optbench.campaign import resolve_problem
    from optbench.tip4p import CONVERSION_LEDGER, KJ_PER_EV

    names = [
        f"{family}/bh-{problem}"
        for family in ("eon-svn", "eon-develop")
        for problem in ("tip4p10", "tip4p15", "tip4p20")
    ]
    ledgers = set()
    for name in names:
        template = load_template(ROOT, name)
        problem = resolve_problem(ROOT, template["problem"])
        ledgers.add(template["ledger"])
        assert template["kernel"] == "tip4p"
        assert template["input"]["select"] == "aa"
        assert template["label"] == "target_ratio 0.3"
        if name.startswith("eon-svn/"):
            assert template["code"] == "EON (SVN r2549)"
        else:
            assert template["code"] == "eOn (develop 7095b23ac17d49f434a5e08ccb072897ceca698e)"
        assert template["ini"]["Potential"]["potential"] == "tip4p"
        assert float(template["energy_kj_per_ev"]) == KJ_PER_EV
        stop = float(template["ini"]["Basin Hopping"]["stop_energy"])
        threshold = reference_energy(problem) + energy_tolerance(problem)
        assert stop * KJ_PER_EV == pytest.approx(threshold, abs=1e-9)
        assert int(template["failure_cap"]) == force_cap(problem)
        assert len(template_starts(template)) == int(problem["sample"]["starts"])
        assert template["ini"]["Basin Hopping"]["target_ratio"] == 0.3
        assert template["ini"]["Main"]["temperature"] == 8000
    assert ledgers == {CONVERSION_LEDGER}


def test_complete_cap_row_is_ranked_with_par2(tmp_path):
    from optbench.dat import parse
    from optbench.eon_harness import write_campaign

    template = load_template(ROOT, "eon-svn/bh-lj75")
    rows = [
        (
            314159,
            str(start),
            {
                "force_calls": 1000000.0,
                "success": False,
                "failure": "cap",
                "minimum_energy": -396.282,
            },
        )
        for start in range(1, 201)
    ]
    summary = write_campaign(template, rows, tmp_path / "row.dat")
    fields = parse((tmp_path / "row.dat").read_text(encoding="utf-8"))
    assert summary["partial"] is False
    assert "partial" not in fields
    assert float(fields["par2_mean"]) == 2000000.0
    assert fields["n_cap"] == "200"
    assert fields["n"] == "200"
    assert fields["label"] == "target_ratio 0.3"
    assert "lj38" not in fields["label"]
    short = rows[:16]
    summary = write_campaign(template, short, tmp_path / "short.dat")
    assert summary["partial"] is True
    assert "partial true" in (tmp_path / "short.dat").read_text(encoding="utf-8")


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
