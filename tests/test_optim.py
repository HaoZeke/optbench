"""OPTIM DNEB/HEF connect input and the socket cap."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, geometry_archive, list_templates, load_template, resolve_problem
from optbench.noise_catalog import specs
from optbench.optim_connect import (
    _execute,
    _fortran_float,
    _job,
    _settings,
    minimum_image_finish,
    optim_binary,
    read_dumped_points,
    read_path_info,
    write_odata,
)
from optbench.pes import load_pairs
from optbench.runners.optim_min import fortran_d, gmax
from optbench.runners.optim_min import write_odata as write_userpot_odata

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
VERSION = "02aaf45245af"


def test_registry_pin_matches_the_template():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "optim")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == "OPTIM"
    template = load_template(ROOT, "optim/dneb-hef")
    assert template["engine"] == "optim-connect"
    assert template["commit"] == COMMIT
    assert template["code"] == "optim"
    assert template["code_version"] == "02aaf452"
    assert template["algorithm"] == "DNEB/HEF problem images"
    assert template["tuning"] == "problem-images"
    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    settings = _settings(template, problem, COMMIT)
    assert settings["n_interior"] == 5
    assert settings["cap"] == 2500
    assert settings["max_move"] == 0.05
    assert settings["potential"] == "lj"
    assert settings["neb_rms"] == 0.01
    assert settings["hef_rms"] == 1.0e-5
    assert settings["spring"] == 1.0


def test_odata_uses_the_problem_image_count_and_step():
    path = Path("/tmp/optim-odata-test")
    path.mkdir(exist_ok=True)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    target = path / "odata"
    write_odata(
        target,
        coords,
        interior=5,
        cap=2500,
        max_move=0.05,
        iteration_density=30,
        frozen=[2],
        neb_rms=0.01,
        spring=1.0,
        hef_rms=1.0e-5,
    )
    text = target.read_text(encoding="utf-8")
    assert "NEWCONNECT 1 1" in text
    assert "NEWNEB 5 150 0.01" in text
    assert "BFGSMIN" in text
    assert "BFGSMIN " not in text
    assert "STEPS 2500" in text
    assert "MAXSTEP 0.05" in text
    assert "MAXMAX 0.05" in text
    assert "MAXBFGS 0.05 0.05 0.05 0.05" in text
    assert "NEBK 1" in text
    assert "CONVERGE" not in text
    assert "FREEZE" in text
    assert text.strip().splitlines()[-1].startswith("XX 1.")
    assert "LJ" not in text.split("POINTS", 1)[0]


def test_finish_unwraps_free_atoms_and_leaves_frozen_atoms():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    product = np.array([[0.1, 0.0, 0.0], [9.2, 0.0, 0.0]], dtype=np.float64)
    free = np.array([False, True])
    box = np.array([10.0, 10.0, 10.0])
    finish = minimum_image_finish(reactant, product, free, box)
    assert finish[0].tolist() == [0.0, 0.0, 0.0]
    assert finish[1, 0] == pytest.approx(1.0 + (9.2 - 1.0 - 10.0))


def test_parsers_read_fortran_doubles_and_saddle_blocks():
    assert _fortran_float("1.5D-02") == pytest.approx(0.015)
    dumped = "     6\n 1.0D+00  2.0D+00  3.0D+00\n 4.0D+00  5.0D+00  6.0D+00\n"
    points = read_dumped_points(dumped, 2)
    assert len(points) == 1
    assert points[0][1].tolist() == pytest.approx([4.0, 5.0, 6.0])
    path = "\n".join(
        [
            "-1.0",
            "     1 C1",
            "0 0 0",
            "1 0 0",
            "-0.5",
            "     1 C1",
            "0.5 0 0",
            "0.5 0 0",
            "-0.8",
            "     1 C1",
            "1 0 0",
            "0 1 0",
        ]
    )
    saddles = read_path_info(path, 2)
    assert len(saddles) == 1
    assert saddles[0][0, 0] == pytest.approx(0.5)


def test_asv_classes_use_one_start_unless_the_full_grid_is_requested():
    from benchmarks.bench_optim import (
        HeptamerOptimDnebHef,
        Lj38DeOptimDnebHef,
        Lj38TwoTsOptimDnebHef,
        PtIslandOptimDnebHef,
    )

    classes = (
        HeptamerOptimDnebHef,
        PtIslandOptimDnebHef,
        Lj38DeOptimDnebHef,
        Lj38TwoTsOptimDnebHef,
    )
    assert [row.problem_id for row in classes] == [
        "neb/Pt-heptamer",
        "double-ended-saddle-search/pt-island",
        "double-ended-saddle-search/lj38",
        "double-ended-saddle-search/lj38_2ts_separation",
    ]
    assert [row.cap for row in classes] == [10000, 3000, 2500, 10000]
    for row in classes:
        assert row.template_id == "optim/dneb-hef"
        assert row.params[0] == ["0"]
        assert len(row.params[1]) == 1
        assert row.timeout == 3600


def test_cap_stops_the_first_lj38_pair():
    binary = optim_binary()
    if binary is None:
        pytest.skip("OPTIM binary is not on this machine")
    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    template = load_template(ROOT, "optim/dneb-hef")
    settings = _settings(template, problem, COMMIT)
    settings["cap"] = 5
    pairs = load_pairs(geometry_archive(ROOT, problem), settings["geometry"], settings["pairs"])
    record = _execute(_job(pairs[0], settings))
    assert record["failure"] == "cap"
    assert record["success"] is False
    assert record["force_calls"] == 5
    assert record["counter"] == "harness"
    assert "OPTIM version" not in (record.get("crash") or "")


def test_gmax_is_the_force_norm_over_sqrt_dof():
    assert gmax(0.01, 38) == 0.01 / math.sqrt(114)
    # The 2014 odata stored 9.36585D-04, that quotient to six figures.
    assert abs(gmax(0.01, 38) - 9.36585e-4) < 1e-9
    assert gmax(1.0e-3, 256) == 1.0e-3 / math.sqrt(768)


def test_odata_puts_userpot_last_and_writes_d_exponents(tmp_path: Path):
    path = tmp_path / "odata"
    write_userpot_odata(path, [0.1, -0.2, 0.3], gmax_value=gmax(0.01, 1), updates=4, maxbfgs=0.2, steps=10000)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("BFGSMIN ")
    assert "D" in lines[0]
    assert lines[1] == "BFGSSTEPS 10000"
    assert lines[2] == "UPDATES 4"
    assert lines[3].startswith("MAXBFGS ")
    assert lines[4] == "USERPOT"
    assert lines[5].startswith("X ")
    assert "e" not in lines[5]
    assert fortran_d(0.1).endswith("D-01") or "D" in fortran_d(0.1)


def test_asv_discovers_only_the_userpot_templates():
    from benchmarks import bench_optim

    classes = [
        value
        for value in vars(bench_optim).values()
        if isinstance(value, type) and getattr(value, "spec", None)
    ]
    assert {row.spec["template_id"] for row in classes} == {
        "optim/lj38-lbfgs",
        "optim/lj38-lbfgs-tuned",
        "optim/morse-lbfgs",
        "optim/morse-lbfgs-tuned",
        "optim/noise-lbfgs-s0",
        "optim/noise-lbfgs-s1em4",
        "optim/noise-lbfgs-s1em3",
        "optim/noise-lbfgs-s1em2",
    }
    assert all(row.spec["code_version"] == VERSION for row in classes)
    assert all(row.params == [["0"], ["0"]] for row in classes)


def test_optim_templates_match_the_tarball_pin():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "optim")
    assert pinned["commit"] == COMMIT
    assert pinned["committed"] == "2026-10-18"
    wanted = {
        "optim/lj38-lbfgs",
        "optim/lj38-lbfgs-tuned",
        "optim/morse-lbfgs",
        "optim/morse-lbfgs-tuned",
        "optim/noise-lbfgs-s0",
        "optim/noise-lbfgs-s1em4",
        "optim/noise-lbfgs-s1em3",
        "optim/noise-lbfgs-s1em2",
    }
    found = [row for row in list_templates(ROOT) if row["id"] in wanted]
    assert len(found) == len(wanted)
    catalog = {row["id"] for row in specs()}
    seen = set()
    for template in found:
        assert template["engine"] == "optim"
        assert template["commit"] == COMMIT
        assert template["code_version"] == VERSION
        assert template["code"] == "optim"
        assert template["algorithm"] == "LBFGS"
        assert template["id"] not in catalog
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        seen.add(template["id"])
    assert seen == {
        "optim/lj38-lbfgs",
        "optim/lj38-lbfgs-tuned",
        "optim/morse-lbfgs",
        "optim/morse-lbfgs-tuned",
        "optim/noise-lbfgs-s0",
        "optim/noise-lbfgs-s1em4",
        "optim/noise-lbfgs-s1em3",
        "optim/noise-lbfgs-s1em2",
    }
    by_id = {row["id"]: row for row in found}
    assert by_id["optim/lj38-lbfgs"]["optim"] == {"updates": 4, "maxbfgs": 0.2}
    assert by_id["optim/lj38-lbfgs-tuned"]["optim"]["updates"] == 1000
    assert by_id["optim/morse-lbfgs-tuned"]["optim"] == {"updates": 100, "maxbfgs": 2.0}
    assert by_id["optim/noise-lbfgs-s1em2"]["noise"]["sigma"] == 0.01
    assert by_id["optim/lj38-lbfgs"]["tuning"] == "default"
    assert by_id["optim/lj38-lbfgs-tuned"]["tuning"] == "tuned"
