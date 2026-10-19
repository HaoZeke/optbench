"""OPTIM DNEB/HEF connect input and the socket cap."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import geometry_archive, load_template, resolve_problem
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

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"


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
