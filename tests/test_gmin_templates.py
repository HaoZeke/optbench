"""GMIN globalopt templates use the registry pin and the harness cap."""

from __future__ import annotations

import tomllib
from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.runners.gmin_go import CODE_VERSION, TARBALL_SHA256, _coords_text, _data_text

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = (
    "globalopt/lj38",
    "globalopt/lj75",
    "globalopt/lj98",
    "globalopt/lj150",
)


def test_registry_pins_the_tarball():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "gmin")
    assert row["name"] == "GMIN"
    assert row["commit"] == TARBALL_SHA256
    assert row["ref"] == CODE_VERSION
    assert TARBALL_SHA256.startswith(row["ref"])


def test_eight_templates_cover_bh_and_bhpt():
    found = [item for item in list_templates(ROOT) if item.get("engine") == "gmin"]
    ids = sorted(item["id"] for item in found)
    assert ids == sorted(
        f"gmin/{problem.split('/')[1]}-{method}"
        for problem in PROBLEMS
        for method in ("bh", "bhpt")
    )
    for template in found:
        assert template["code"] == "GMIN"
        assert template["code_version"] == CODE_VERSION
        assert template["commit"] == TARBALL_SHA256
        assert template["problem"] in PROBLEMS
        assert int(template["failure_cap"]) == 2_000_000
        section = template["gmin"]
        assert section["method"] in {"bh", "bhpt"}
        if section["method"] == "bhpt":
            assert int(section["replicas"]) >= 2
        else:
            assert int(section["replicas"]) == 1
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        stated = (problem.get("convergence") or {}).get("energy_above_minimum")
        if stated is not None:
            assert abs(float(stated) - float(section["ediff"])) < 1e-12


def test_unstated_clusters_use_the_website_archives(tmp_path):
    from optbench.runners.gmin_go import load_geometries

    expected = {
        "globalopt/lj75": 200,
        "globalopt/lj98": 100,
        "globalopt/lj150": 100,
    }
    for problem_id, count in expected.items():
        problem = resolve_problem(ROOT, problem_id)
        assert (problem.get("input") or {}).get("archive_path") is None
        loaded = load_geometries(ROOT, problem, tmp_path / problem_id.replace("/", "-"))
        assert len(loaded) == count
        assert loaded[0][0] == "coords.1"
        assert loaded[-1][0] == f"coords.{count}"
        assert "coords" not in {name for name, _pos in loaded}


def test_bhpt_keyword_order_and_repeated_coords():
    job = {
        "replicas": 4,
        "pttmin": 0.1,
        "pttmax": 1.0,
        "exchprob": 0.1,
        "sloppyconv": 0.005,
        "tightconv": 0.001,
        "ediff": 0.001,
        "maxit": 600,
        "maxit2": 500,
        "radius": 50.0,
        "temperature": 0.1,
        "step": 0.4,
        "astep": 0.51,
        "steps": 1000000,
        "seed": 20261018,
        "reference": -173.928426511789,
        "cap": 2000000,
    }
    text = _data_text(job)
    assert text.startswith("MPI\nBHPT 0.10000000 1.00000000 0.10000000\n")
    assert "TARGET -173.928426511789" in text
    serial = dict(job)
    serial["replicas"] = 1
    serial_text = _data_text(serial)
    assert serial_text.startswith("MPI\n")
    assert "BHPT" not in serial_text
    from optbench.runners.gmin_go import _command

    command = _command("/tmp/gmin-build/GMIN", 4)
    assert command[command.index("-np") + 1] == "4"
    assert "mpi_yield_when_idle" in command
    coords = _coords_text([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 4)
    assert len(coords.splitlines()) == 8


def test_asv_classes_import_with_one_start():
    import benchmarks.bench_gmin as bench

    classes = [
        value
        for value in vars(bench).values()
        if isinstance(value, type) and getattr(value, "param_names", None) == ["seed", "start"]
    ]
    assert len(classes) == 8
    for benchmark in classes:
        assert benchmark.params == [["0"], ["0"]]
        assert benchmark.cap == 2_000_000
