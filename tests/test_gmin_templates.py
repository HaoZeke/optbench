"""GMIN globalopt templates use the registry pin and the harness cap."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.runners.gmin_go import CODE_VERSION, PIN_REF, TARBALL_SHA256, _coords_text, _data_text

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = (
    "globalopt/lj38",
    "globalopt/lj75",
    "globalopt/lj98",
    "globalopt/lj150",
)


def test_registry_pins_the_tarball():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "gmin-bh")
    assert row["name"] == "GMIN"
    assert row["commit"] == TARBALL_SHA256
    assert row["ref"] == PIN_REF
    assert CODE_VERSION == "df8da599"
    assert TARBALL_SHA256.startswith(row["ref"])
    quench = next(item for item in registry["code"] if item["id"] == "gmin")
    assert TARBALL_SHA256 in quench["note"]


def test_bh_bhpt_and_csm_templates():
    found = [item for item in list_templates(ROOT) if item.get("engine") == "gmin-bh"]
    ids = sorted(item["id"] for item in found)
    expected = [
        f"gmin/{problem.split('/')[1]}-{method}"
        for problem in PROBLEMS
        for method in ("bh", "bhpt")
    ]
    expected.append("gmin/lj38-csm")
    assert ids == sorted(expected)
    for template in found:
        assert template["code"] == "GMIN"
        assert template["code_version"] == CODE_VERSION
        assert template["commit"] == TARBALL_SHA256
        assert template["problem"] in PROBLEMS
        section = template["gmin"]
        assert section["method"] in {"bh", "bhpt"}
        if section["method"] == "bhpt":
            assert int(section["replicas"]) >= 2
        else:
            assert int(section["replicas"]) == 1
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        convergence = problem.get("convergence") or {}
        stated = convergence.get("energy_above_minimum")
        if stated is not None:
            assert abs(float(stated) - float(section["ediff"])) < 1e-12
        stated_cap = convergence.get("max_force_calls")
        if stated_cap is not None:
            assert int(template["failure_cap"]) == int(stated_cap)
        else:
            assert int(template["failure_cap"]) == 2_000_000


def test_cluster_starts_are_the_numbered_website_coords(tmp_path):
    from optbench.runners.gmin_go import load_geometries

    expected = {
        "globalopt/lj75": ("website/globalopt/lj75/lj75_clusters.tgz", 200),
        "globalopt/lj98": ("website/globalopt/lj98/lj98_clusters.tgz", 100),
        "globalopt/lj150": ("website/globalopt/lj150/lj150_clusters.tgz", 100),
    }
    for problem_id, (archive, count) in expected.items():
        problem = resolve_problem(ROOT, problem_id)
        assert (problem.get("input") or {}).get("archive_path") == archive
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
    coords = _coords_text([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 4)
    assert len(coords.splitlines()) == 8


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun is not on PATH")
def test_bhpt_command_launches_every_replica_under_mpirun():
    from optbench.runners.gmin_go import _command

    command = _command("/tmp/gmin-build/GMIN", 4)
    assert command[command.index("-np") + 1] == "4"
    assert "mpi_yield_when_idle" in command


def test_lj98_cap_is_the_problem_cap_and_the_data_file_uses_it():
    problem = resolve_problem(ROOT, "globalopt/lj98")
    cap = int(problem["convergence"]["max_force_calls"])
    assert cap == 1_000_000_000
    for name in ("gmin/lj98-bh", "gmin/lj98-bhpt"):
        template = next(item for item in list_templates(ROOT) if item["id"] == name)
        assert int(template["failure_cap"]) == cap
        section = template["gmin"]
        assert int(section["steps"]) == 100_000
        assert section["symmetrise"] == "1 0.01 0.05 0.3 0.1 0.6 3 0.3 2.1"
        text = _data_text(
            {
                "replicas": int(section["replicas"]),
                "pttmin": float(section["pttmin"]),
                "pttmax": float(section["pttmax"]),
                "exchprob": float(section["exchprob"]),
                "sloppyconv": float(section["sloppyconv"]),
                "tightconv": float(section["tightconv"]),
                "ediff": float(section["ediff"]),
                "maxit": int(section["maxit"]),
                "maxit2": int(section["maxit2"]),
                "radius": float(section["radius"]),
                "temperature": float(section["temperature"]),
                "step": float(section["step"]),
                "astep": float(section["astep"]),
                "steps": int(section["steps"]),
                "seed": int(section["seed"]),
                "reference": -543.665361,
                "cap": cap,
                "fixboth": section["fixboth"],
                "permdist": section["permdist"],
                "newrestart": section["newrestart"],
                "sort": section["sort"],
                "symmetrise": section["symmetrise"],
                "save": int(section["save"]),
                "tfac": float(section["tfac"]),
                "emit_seed": bool(section["emit_seed"]),
            }
        )
        assert f"MAX_NPCALL {cap}" in text
        assert "SYMMETRISE 1 0.01 0.05 0.3 0.1 0.6 3 0.3 2.1" in text
        assert "STEPS 100000 1.0" in text
        assert "RADIUS" not in text
        assert "RANSEED" not in text
        assert "MAX_NPCALL 1000000000" in text
        assert section["center"] is False
        if section["method"] == "bh":
            assert "BHPT" not in text
            assert "TEMPERATURE 1.30000000" in text
        else:
            assert text.startswith("MPI\nBHPT 0.10000000 1.30000000 0.10000000\n")


def test_lj38_csm_uses_the_archive_deck_on_df8da599():
    problem = resolve_problem(ROOT, "globalopt/lj38")
    assert problem["convergence"].get("max_force_calls") is None
    template = next(item for item in list_templates(ROOT) if item["id"] == "gmin/lj38-csm")
    assert template["code_version"] == "df8da599"
    assert int(template["failure_cap"]) == 2_000_000
    section = template["gmin"]
    text = _data_text(
        {
            "replicas": int(section["replicas"]),
            "pttmin": float(section["pttmin"]),
            "pttmax": float(section["pttmax"]),
            "exchprob": float(section["exchprob"]),
            "sloppyconv": float(section["sloppyconv"]),
            "tightconv": float(section["tightconv"]),
            "ediff": float(section["ediff"]),
            "maxit": int(section["maxit"]),
            "maxit2": int(section["maxit2"]),
            "radius": float(section["radius"]),
            "temperature": float(section["temperature"]),
            "step": float(section["step"]),
            "astep": float(section["astep"]),
            "steps": int(section["steps"]),
            "seed": int(section["seed"]),
            "reference": -173.928426511789,
            "cap": int(template["failure_cap"]),
            "fixboth": section["fixboth"],
            "permdist": section["permdist"],
            "newrestart": section["newrestart"],
            "sort": section["sort"],
            "save": int(section["save"]),
            "tfac": float(section["tfac"]),
            "emit_seed": bool(section["emit_seed"]),
            "avoid": section["avoid"],
            "symmetrisecsm": section["symmetrisecsm"],
        }
    )
    assert "SYMMETRISECSM 1 CI 1.0D-6 30 1 50" in text
    assert "AVOID 1.25 18" in text
    assert "TEMPERATURE 3.00000000" in text
    assert "NEWRESTART 30" in text
    assert "PERMDIST 0.00500000" in text
    assert "STEPS 90000000 1.0" in text
    assert "MAX_NPCALL 2000000" in text
    assert "EDIFF 0.00100000" in text
    assert "RANSEED" not in text
    assert "RADIUS" not in text
    assert "BHPT" not in text
    assert "wales-27058" in template["note"]
    assert "not published" in template["note"]


def test_asv_classes_import_with_one_start():
    import benchmarks.bench_gmin_bh as bench

    classes = [
        value
        for value in vars(bench).values()
        if isinstance(value, type) and getattr(value, "param_names", None) == ["seed", "start"]
    ]
    assert len(classes) == 9
    caps = {
        item["id"]: int(item["failure_cap"])
        for item in list_templates(ROOT)
        if item.get("engine") == "gmin-bh"
    }
    for benchmark in classes:
        assert benchmark.params == [["0"], ["0"]]
        assert benchmark.cap == caps[benchmark.spec["template_id"]]
