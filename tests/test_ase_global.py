"""ASE basin hopping uses the harness counter and the Lennard-Jones stop."""

import json
from pathlib import Path

import pytest

from optbench.ase_global import list_start_names, target_energy
from optbench.ase_lj import LennardJonesASE
from optbench.campaign import geometry_archive, load_template
from optbench.problems import load_problem

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "globalopt/lj38": "website/globalopt/lj38/100_lj38.tar.gz",
    "globalopt/lj75": "website/globalopt/lj75/lj75_clusters.tgz",
    "globalopt/lj98": "website/globalopt/lj98/lj98_clusters.tgz",
    "globalopt/lj150": "website/globalopt/lj150/lj150_clusters.tgz",
}
COUNTS = {
    "globalopt/lj38": 100,
    "globalopt/lj75": 200,
    "globalopt/lj98": 100,
    "globalopt/lj150": 100,
}


def test_start_counts_match_the_archives():
    for problem, archive_path in ARCHIVES.items():
        archive = geometry_archive(ROOT, {"id": problem, "input": {"archive_path": archive_path}})
        names = list_start_names(archive)
        assert len(names) == COUNTS[problem]
        assert len(names) == len(set(names))


def test_lj38_stop_is_the_archived_eon_energy():
    problem = load_problem(ROOT / "problems" / "globalopt" / "lj38.toml")
    assert target_energy(problem) == -173.927427
    assert target_energy(load_problem(ROOT / "problems" / "globalopt" / "lj75.toml")) is None
    tuned = load_template(ROOT, "asebh/lj38-basin-tuned")
    assert tuned["method"]["parameters"]["temperature_kelvin"] == 8000
    assert tuned["method"]["parameters"]["dr"] == 0.5
    assert tuned["method"]["parameters"]["fmax"] == 0.001
    assert tuned["method"]["parameters"]["optimizer"] == "LBFGS"
    assert tuned["seed"] == 314159
    assert tuned["stop"]["max_force_calls"] == 2000000
    default = load_template(ROOT, "asebh/lj38-basin-default")
    assert "temperature_kelvin" not in default["method"]["parameters"]
    assert default["method"]["name"] == "BasinHopping"
    hopping = load_template(ROOT, "asebh/lj38-mh-default")
    assert hopping["method"]["name"] == "MinimaHopping"


def test_kernel_matches_the_ase_lennard_jones_calculator():
    ase = pytest.importorskip("ase")
    numpy = pytest.importorskip("numpy")
    del ase
    from ase.calculators.lj import LennardJones
    from ase.io import read

    archive = geometry_archive(
        ROOT,
        {"id": "globalopt/lj38", "input": {"archive_path": ARCHIVES["globalopt/lj38"]}},
    )
    import tarfile

    with tarfile.open(archive) as packed:
        raw = packed.extractfile("100_lj38/0.con").read()
    path = ROOT / ".pytest-lj38-0.con"
    path.write_bytes(raw)
    try:
        atoms = read(path, format="eon")
    finally:
        path.unlink()
    atoms.set_pbc(False)
    positions = [atoms.get_positions()]
    rng = numpy.random.default_rng(0)
    base = atoms.get_positions()
    for scale in (0.05, 0.2, 0.5):
        positions.append(base + rng.normal(scale=scale, size=base.shape))
    kernel = LennardJonesASE(epsilon=1.0, sigma=1.0, rc=40.0)
    calc = LennardJones(epsilon=1.0, sigma=1.0, rc=40.0, smooth=False)
    atoms.calc = calc
    for pos in positions:
        atoms.set_positions(pos)
        expected_energy = float(atoms.get_potential_energy())
        expected_forces = atoms.get_forces()
        energy, forces = kernel.energy_forces(pos)
        energy_error = abs(energy - expected_energy)
        force_error = float(numpy.max(numpy.abs(forces - expected_forces)))
        force_scale = max(float(numpy.max(numpy.abs(expected_forces))), 1.0)
        assert energy_error < 1e-8 or energy_error <= 1e-12 * max(abs(expected_energy), 1.0)
        assert force_error < 1e-8 or force_error <= 1e-12 * force_scale
    published = positions[0]
    atoms.set_positions(published)
    energy, forces = kernel.energy_forces(published)
    assert abs(energy - float(atoms.get_potential_energy())) < 1e-8
    assert numpy.max(numpy.abs(forces - atoms.get_forces())) < 1e-8


def test_a_high_target_stops_on_the_first_quench(tmp_path: Path):
    pytest.importorskip("ase")
    from optbench.ase_global import materialize, run_start

    archive = geometry_archive(
        ROOT,
        {"id": "globalopt/lj38", "input": {"archive_path": ARCHIVES["globalopt/lj38"]}},
    )
    cache = materialize(archive, tmp_path / "geo")
    payload = {
        "start": "100_lj38/0.con",
        "path": str(cache / "100_lj38/0.con"),
        "kind": "eon",
        "method": "BasinHopping",
        "parameters": {"local_steps": 50, "fmax": 0.1},
        "potential": {"epsilon": 1.0, "sigma": 1.0, "rc": 40.0},
        "seed": 0,
        "cap": 1000,
        "max_hops": 5,
        "target": -100.0,
        "problem": "globalopt/lj38",
        "template": "asebh/lj38-basin-default",
        "tuning": "default",
        "algorithm": "BasinHopping",
    }
    first = run_start(payload)
    second = run_start(payload)
    assert first["success"] is True
    assert first["failure"] is None
    assert first["counter"] == "harness"
    assert first["force_calls"] < 50
    assert first["lowest_energy"] <= -100.0
    assert first["seconds"] >= 0.0
    quiet = {key: value for key, value in first.items() if key != "seconds"}
    assert quiet == {key: value for key, value in second.items() if key != "seconds"}


def test_asv_tracks_round_trip(tmp_path: Path):
    pytest.importorskip("asv")
    from asv.results import iter_results

    from benchmarks.bench_asebh import _class_name
    from optbench.ase_asv import class_name, document
    from optbench.asv_score import track_rows

    template = "asebh/lj38-basin-tuned"
    assert class_name(template) == _class_name({"id": template})
    rows = [
        {
            "start": "100_lj38/0.con",
            "force_calls": 2000000,
            "success": False,
            "failure": "cap",
            "seconds": 3.5,
            "seed": 314159,
        },
        {
            "start": "100_lj38/1.con",
            "force_calls": 40,
            "success": True,
            "failure": None,
            "seconds": 0.25,
            "seed": 314159,
        },
    ]
    machine = {
        "arch": "x86_64",
        "cpu": "test",
        "machine": "cursor",
        "num_cpu": "8",
        "os": "Linux",
        "ram": "1",
    }
    body = document({template: rows}, commit="abc12345abc12345", machine=machine, started_at=1)
    dest_dir = tmp_path / "asv" / "cursor"
    dest_dir.mkdir(parents=True)
    (dest_dir / "machine.json").write_text(json.dumps({**machine, "version": 1}), encoding="utf-8")
    (dest_dir / "abc12345-uv-pip-py3.12.json").write_text(json.dumps(body), encoding="utf-8")
    found = list(iter_results(str(tmp_path / "asv")))
    assert len(found) == 1
    tracked = track_rows(found[0], class_name(template))
    assert [item["force_calls"] for item in tracked] == [2000000, 40]
    assert [item["success"] for item in tracked] == [False, True]
    assert [item["failure"] for item in tracked] == ["cap", None]


def test_the_force_cap_is_a_failure(tmp_path: Path):
    pytest.importorskip("ase")
    from optbench.ase_global import materialize, run_start

    archive = geometry_archive(
        ROOT,
        {"id": "globalopt/lj38", "input": {"archive_path": ARCHIVES["globalopt/lj38"]}},
    )
    cache = materialize(archive, tmp_path / "geo")
    payload = {
        "start": "100_lj38/0.con",
        "path": str(cache / "100_lj38/0.con"),
        "kind": "eon",
        "method": "BasinHopping",
        "parameters": {
            "local_steps": 1000,
            "temperature_kelvin": 8000,
            "dr": 0.5,
            "fmax": 0.001,
            "optimizer": "LBFGS",
        },
        "potential": {"epsilon": 1.0, "sigma": 1.0, "rc": 40.0},
        "seed": 314159,
        "cap": 30,
        "max_hops": 10,
        "target": -173.927427,
        "problem": "globalopt/lj38",
        "template": "asebh/lj38-basin-tuned",
        "tuning": "tuned",
        "algorithm": "BasinHopping",
    }
    row = run_start(payload)
    assert row["success"] is False
    assert row["failure"] == "cap"
    assert row["force_calls"] <= 30
    assert row["counter"] == "harness"


def test_minima_identity_matches_ase_compare():
    ase = pytest.importorskip("ase")
    numpy = pytest.importorskip("numpy")
    del ase
    from ase import Atoms
    from ase.optimize.minimahopping import ComparePositions

    from optbench.ase_global import _greedy_dmax, _translated_dmax

    rng = numpy.random.default_rng(2)
    translated = ComparePositions(translate=True)
    plain = ComparePositions(translate=False)
    for index in range(12):
        pos1 = rng.normal(scale=3.0, size=(38, 3))
        if index % 3 == 0:
            pos2 = pos1[rng.permutation(38)] + rng.normal(scale=1e-8, size=(38, 3))
        elif index % 3 == 1:
            pos2 = pos1 + rng.normal(scale=0.02, size=(38, 3))
        else:
            pos2 = rng.normal(scale=3.0, size=(38, 3))
        atoms1 = Atoms("C38", positions=pos1, pbc=False)
        atoms2 = Atoms("C38", positions=pos2, pbc=False)
        assert abs(float(translated(atoms1, atoms2)) - _translated_dmax(pos1, pos2)) < 1e-12
        assert abs(float(plain(atoms1, atoms2)) - _greedy_dmax(pos1, pos2)) < 1e-12


def test_minima_hopping_repeats_for_one_seed(tmp_path: Path):
    pytest.importorskip("ase")
    from optbench.ase_global import materialize, run_start

    archive = geometry_archive(
        ROOT,
        {"id": "globalopt/lj38", "input": {"archive_path": ARCHIVES["globalopt/lj38"]}},
    )
    cache = materialize(archive, tmp_path / "geo")
    payload = {
        "start": "100_lj38/0.con",
        "path": str(cache / "100_lj38/0.con"),
        "kind": "eon",
        "method": "MinimaHopping",
        "parameters": {"local_steps": 1000},
        "potential": {"epsilon": 1.0, "sigma": 1.0, "rc": 40.0},
        "seed": 0,
        "cap": 20000,
        "max_hops": 3,
        "target": -173.927427,
        "problem": "globalopt/lj38",
        "template": "asebh/lj38-mh-default",
        "tuning": "default",
        "algorithm": "MinimaHopping",
    }
    first = run_start(payload)
    second = run_start(payload)
    assert first["failure"] != "crash"
    assert first["hops"] == 3
    assert first["force_calls"] == second["force_calls"]
    assert first["lowest_energy"] == second["lowest_energy"]
    assert first["counter"] == "harness"
