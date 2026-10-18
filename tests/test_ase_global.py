"""ASE basin hopping uses the harness counter and the Lennard-Jones stop."""

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
