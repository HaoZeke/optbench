"""GMIN globalopt templates use the registry pin and the harness cap."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from optbench.campaign import resolve_problem
from optbench.runners.gmin_go import CODE_VERSION, PIN_REF, TARBALL_SHA256, _coords_text, _data_text

ROOT = Path(__file__).resolve().parents[1]


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
