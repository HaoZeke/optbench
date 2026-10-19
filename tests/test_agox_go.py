"""AGOX templates, the harness Lennard-Jones calculator, and a short search."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.potentials import lennard_jones
from optbench.runners.agox_go import _xyz_rows, reference_energy, target_energy

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "22d94a6fae8cf2d08d51ecc8b9291a47667aa95c"
VERSION = "3.11.2.dev2+g22d94a6fa"
METHODS = ("basin_hopping", "rss", "ga", "gofee")


def _pinned() -> dict:
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    return next(row for row in registry["code"] if row["id"] == "agox")


def test_cambridge_coordinates_match_this_potential():
    lj38, _gradient = lennard_jones(_xyz_rows((ROOT / "optbench/data/lj38-minimum.txt").read_text(encoding="utf-8")))
    lj75, gradient75 = lennard_jones(_xyz_rows((ROOT / "optbench/data/lj75-minimum.txt").read_text(encoding="utf-8")))
    assert lj38 == pytest.approx(-173.9284265117895, abs=1e-9)
    assert lj75 == pytest.approx(-397.4923306811037, abs=1e-9)
    import numpy as np

    assert float(np.linalg.norm(gradient75)) < 0.02
    problem = resolve_problem(ROOT, "globalopt/lj38")
    energy, source = reference_energy(ROOT, problem, 38)
    assert source == "problem"
    assert energy == pytest.approx(-173.928427)
    assert target_energy(problem, energy) == pytest.approx(energy + 0.001)
    lj75_problem = resolve_problem(ROOT, "globalopt/lj75")
    energy75, source75 = reference_energy(ROOT, lj75_problem, 75)
    assert source75 == "problem"
    assert energy75 == pytest.approx(-397.492331)
    assert target_energy(lj75_problem, energy75) == pytest.approx(energy75 + 0.001)


def test_agox_templates_pin_the_registry_and_the_cap():
    pinned = _pinned()
    assert pinned["commit"] == COMMIT
    assert VERSION in pinned["note"]
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("agox/")]
    assert len(found) == 8
    seen = set()
    for template in found:
        assert template["engine"] == "agox"
        assert template["code"] == "AGOX"
        assert template["commit"] == COMMIT
        assert template["code_version"] == VERSION
        assert template["tuning"] == "default"
        assert template["agox"]["method"] in METHODS
        problem = resolve_problem(ROOT, template["problem"])
        expected = 1000000 if template["problem"] == "globalopt/lj75" else 500000
        assert _check_stop(problem, template) == expected
        seen.add((template["problem"], template["agox"]["method"]))
    for problem in ("globalopt/lj38", "globalopt/lj75"):
        for method in METHODS:
            assert (problem, method) in seen


def test_one_calculator_call_counts_energy_and_forces_once():
    pytest.importorskip("ase")
    from ase import Atoms

    from optbench.counter import attach_ase_counter
    from optbench.runners.agox_go import LennardJones

    atoms = Atoms("N2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    atoms.calc = attach_ase_counter(LennardJones(), cap=10)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    direct, _gradient = lennard_jones(atoms.positions)
    assert energy == pytest.approx(direct)
    assert forces.shape == (2, 3)
    assert atoms.calc.optbench_harness.calls == 1


def test_fast_lbfgs_lowers_a_stretched_pair():
    pytest.importorskip("ase")
    pytest.importorskip("scipy")
    from ase import Atoms

    from optbench.counter import attach_ase_counter
    from optbench.runners.agox_go import FastLBFGS, LennardJones

    atoms = Atoms("N2", positions=[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]])
    atoms.calc = attach_ase_counter(LennardJones(), cap=100)
    before = atoms.get_potential_energy()
    # The energy read above spent one call. Reset by constructing a fresh counter.
    atoms.calc = attach_ase_counter(LennardJones(), cap=100)
    opt = FastLBFGS(atoms, logfile=None)
    opt.run(fmax=0.05, steps=40)
    after = float(atoms.calc.last_energy)
    assert after < before
    assert atoms.calc.optbench_harness.calls < 100
    assert atoms.calc.optbench_harness.calls > 1


def test_overlapped_lj75_start_stays_inside_the_confinement(tmp_path: Path):
    pytest.importorskip("agox")
    pytest.importorskip("ase")
    import numpy as np
    from ase import Atoms
    from agox.environments import Environment

    from optbench.counter import attach_ase_counter
    from optbench.runners.agox_go import FastLBFGS, LennardJones, _load_geometries, _place

    loaded = _load_geometries(ROOT, resolve_problem(ROOT, "globalopt/lj75"), None, tmp_path)
    for start in ("coords.5", "coords.34", "coords.49", "coords.90", "coords.163"):
        coords = next(pos for name, pos in loaded if name == start)
        pos, cell, side, corner = _place(coords, 48.0, 36.0)
        template = Atoms(cell=np.eye(3) * cell, pbc=False)
        environment = Environment(
            template=template,
            symbols=f"N{len(pos)}",
            confinement_cell=np.eye(3) * side,
            confinement_corner=np.array([corner, corner, corner]),
            print_report=False,
        )
        atoms = Atoms(f"N{len(pos)}", positions=pos, cell=template.cell, pbc=False)
        atoms.set_constraint(environment.get_box_constraint())
        atoms.calc = attach_ase_counter(LennardJones(), cap=2000)
        FastLBFGS(atoms, logfile=None).run(fmax=0.05, steps=500)
        span = float((atoms.positions.max(axis=0) - atoms.positions.min(axis=0)).max())
        assert span < 12.0, start
        assert float(atoms.calc.last_energy) < -300.0, start


def test_basin_hopping_stops_at_the_cap(tmp_path: Path):
    pytest.importorskip("agox")
    from optbench.campaign import load_template
    from optbench.runners.agox_go import run_template

    template = load_template(ROOT, "agox/lj38-bh")
    template["stop"] = {"max_force_calls": 60}
    output = tmp_path / "benchmark.dat"
    summary = run_template(
        ROOT,
        resolve_problem(ROOT, "globalopt/lj38"),
        template,
        geometries=None,
        output=output,
        workers=1,
        work=tmp_path / "work",
        limit=1,
        resume=False,
    )
    assert summary["starts"] == 1
    assert summary["cap"] == 60
    text = output.read_text(encoding="utf-8")
    assert "counter harness" in text
    assert "partial true" in text
    starts = output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(starts) == 1
    import json

    record = json.loads(starts[0])
    assert record["counter"] == "harness"
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert record["force_calls"] == 60
    assert record["lowest_energy"] < -100
