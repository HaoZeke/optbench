"""The LJ38 two-step stop is the true-force L2 norm, not the NEB residual."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from optbench.ase_sella import CallCounter, _ase_stock_fmax, _run_sella_neb, _run_start


class _FlatForce(Calculator):
    """True force stays under the L2 stop. Springs still make a NEB residual."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        forces = np.zeros_like(positions)
        forces[:, 0] = 1.0e-6
        self.results["energy"] = float(positions[:, 0].sum())
        self.results["forces"] = forces


def _xyz(path: Path, positions: list[list[float]]) -> None:
    lines = [str(len(positions)), "pair"]
    for xyz in positions:
        lines.append("Ar {:f} {:f} {:f}".format(*xyz))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _payload(tmp_path: Path, *, norm: str, steps: int) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    left = tmp_path / "left.xyz"
    right = tmp_path / "right.xyz"
    _xyz(left, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _xyz(right, [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    return {
        "start": "0000",
        "task": "neb",
        "paths": [str(left), str(right)],
        "potential": {
            "kind": "lennard-jones",
            "epsilon": 1.0,
            "sigma": 1.0,
            "rc": 40.0,
            "pbc": [False, False, False],
        },
        "optimizer": {"name": "FIRE", "parameters": {}},
        "neb": {
            "moving_images": 1,
            "spring": 0.1,
            "climb": False,
            "method": "improvedtangent",
        },
        "seed": 0,
        "tol": 0.001,
        "norm": norm,
        "max_steps": steps,
        "raw_cap": 5000,
        "snapshot": False,
    }


def test_l2_stop_ends_before_the_neb_residual_does(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("optbench.ase_sella._inner_calculator", lambda _potential: _FlatForce())

    class _KeepGoing:
        """Yield until the harness breaks. A straight band has no spring residual."""

        def __init__(self, target, logfile=None, **kwargs):
            self.target = target

        def irun(self, fmax, steps):
            self.target.get_forces()
            yield False
            for _ in range(steps):
                for image in self.target.images[1:-1]:
                    image.positions[0, 0] += 0.01
                self.target.get_forces()
                yield False

    monkeypatch.setattr("optbench.ase_sella._build_optimizer", lambda spec, target: _KeepGoing(target))
    l2 = _run_start(_payload(tmp_path / "l2", norm="l2", steps=8))
    residual = _run_start(_payload(tmp_path / "fmax", norm="fmax", steps=8))
    assert l2["success"] is True
    assert l2["force_norm"] <= 0.001
    assert l2["force_calls"] < residual["force_calls"]


def test_l2_sella_band_keeps_the_whole_cap(monkeypatch):
    seen = {}

    def fake_neb(payload, bucket, band_limit=None, stock_fmax=None):
        seen["band_limit"] = band_limit
        seen["stock_fmax"] = stock_fmax
        seen["norm"] = payload.get("norm")

        def image(energy):
            atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=np.zeros((2, 3)),
            )
            return atoms

        return {
            "failed": True,
            "force_norm": 1.0e-8,
            "images": [image(0.0), image(1.0), image(0.0)],
            "parameters": {"spring": 0.1},
        }

    monkeypatch.setattr("optbench.ase_sella._run_neb", fake_neb)
    row = _run_sella_neb(
        {
            "norm": "l2",
            "tol": 0.001,
            "max_steps": 10,
            "snapshot": True,
            "optimizer": {"name": "Sella", "parameters": {"order": 1}},
            "neb": {"band_optimizer": "FIRE", "band_parameters": {}},
            "sella": {"order": 1, "v0": "none"},
        },
        CallCounter(1000),
    )
    assert seen["band_limit"] is None
    assert seen["stock_fmax"] == pytest.approx(_ase_stock_fmax())
    assert seen["norm"] == "l2"
    assert row["failed"] is False
    assert row["force_norm"] <= 0.001
    assert "reserve" not in row["parameters"]
    assert row["parameters"]["band_fmax"] == pytest.approx(_ase_stock_fmax())


def test_residual_rows_still_reserve_part_of_the_cap(monkeypatch):
    seen = {}

    def fake_neb(payload, bucket, band_limit=None, stock_fmax=None):
        seen["band_limit"] = band_limit
        seen["stock_fmax"] = stock_fmax
        return {"failed": True, "force_norm": 1.0, "images": None}

    monkeypatch.setattr("optbench.ase_sella._run_neb", fake_neb)
    _run_sella_neb(
        {
            "norm": "fmax",
            "tol": 0.001,
            "max_steps": 10,
            "snapshot": False,
            "optimizer": {"name": "Sella", "parameters": {"order": 1}},
            "neb": {"band_optimizer": "FIRE", "band_parameters": {}},
            "sella": {"order": 1, "v0": "none", "reserve": 0.2},
        },
        CallCounter(1000),
    )
    assert seen["band_limit"] == 800
    assert seen["stock_fmax"] is None
