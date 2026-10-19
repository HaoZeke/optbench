"""TorchSim minimization templates and the harness potential model."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.torchsim_catalog import COMMIT, VERSION, specs

ROOT = Path(__file__).resolve().parents[1]


def test_torchsim_templates_match_the_catalog_and_the_problem_stop():
    catalog = specs()
    assert len(catalog) == 12
    on_disk = [row for row in list_templates(ROOT) if str(row["id"]).startswith("torchsim/")]
    assert {row["id"] for row in on_disk} == {row["id"] for row in catalog}
    by_id = {row["id"]: row for row in on_disk}
    problems = {"minimization/lj38", "minimization/lj38-noise", "minimization/morse-bulk"}
    assert {row["problem"] for row in catalog} == problems
    for row in catalog:
        stored = by_id[row["id"]]
        assert stored["engine"] == "torchsim"
        assert stored["commit"] == COMMIT
        assert stored["code_version"] == VERSION
        assert stored["tuning"] == "default"
        assert stored["torchsim"]["method"] == row["torchsim"]["method"]
        problem = resolve_problem(ROOT, stored["problem"])
        _check_stop(problem, stored)
        assert float(stored["stop"]["force_norm"]) == float(problem["convergence"]["force_norm"])
        assert int(stored["stop"]["max_force_calls"]) == int(problem["convergence"]["max_force_calls"])
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "torchsim")
    assert pinned["commit"] == COMMIT
    assert pinned["ref"] == "v0.6.2"
    assert pinned["name"] == "TorchSim"
    fire = by_id["torchsim/lj38-fire"]["torchsim"]
    assert fire["fire_flavor"] == "ase_fire"
    assert fire["max_step"] == 0.2
    assert fire["dt_start"] == 0.1
    assert by_id["torchsim/lj38-gd"]["torchsim"]["pos_lr"] == 0.01
    assert "noise" not in by_id["torchsim/lj38-fire"]
    for sigma, slug in ((0.0, "s0"), (1e-4, "s1em4"), (1e-3, "s1em3"), (1e-2, "s1em2")):
        for method in ("fire", "gd"):
            stored = by_id[f"torchsim/lj38-noise-{method}-{slug}"]
            assert stored["noise"]["sigma"] == sigma
            assert stored["noise"]["seed"] == 20261016


def test_torchsim_engine_dispatches_without_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["limit"] = kwargs["limit"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 0,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.torchsim_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/lj38",
        "torchsim/lj38-fire",
        output=tmp_path / "benchmark.dat",
        workers=2,
    )
    assert captured["template"] == "torchsim/lj38-fire"
    assert captured["problem"] == "minimization/lj38"
    assert captured["workers"] == 2
    assert captured["limit"] == 0
    assert summary["template"] == "torchsim/lj38-fire"


def test_model_force_is_the_negative_spec_gradient():
    pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    import inspect

    import numpy as np
    import torch
    from torch_sim.optimizers.fire import fire_init, fire_step
    from torch_sim.optimizers.gradient_descent import gradient_descent_step

    from optbench.counter import HarnessCounter
    from optbench.potentials import lennard_jones, morse_bulk
    from optbench.potserver import RgpotCounter
    from optbench.torchsim_model import SpecPotentialModel

    assert inspect.signature(fire_init).parameters["dt_start"].default == 0.1
    assert inspect.signature(fire_init).parameters["alpha_start"].default == 0.1
    assert inspect.signature(fire_init).parameters["fire_flavor"].default == "ase_fire"
    step = inspect.signature(fire_step).parameters
    assert step["dt_max"].default == 1.0
    assert step["n_min"].default == 5
    assert step["f_inc"].default == 1.1
    assert step["f_dec"].default == 0.5
    assert step["f_alpha"].default == 0.99
    assert step["max_step"].default == 0.2
    assert step["fire_flavor"].default == "ase_fire"
    assert inspect.signature(gradient_descent_step).parameters["pos_lr"].default == 0.01

    rng = np.random.default_rng(0)
    coords = rng.normal(scale=0.4, size=(8, 3))
    counter = HarnessCounter(cache=True, cap=10)
    potential = RgpotCounter(lambda positions, _numbers=None, _box=None: lennard_jones(positions), counter)

    def evaluate(rows):
        energy, gradient = potential.calculate(np.asarray(rows, dtype=np.float64).reshape(-1))
        return energy, gradient

    import torch_sim as ts

    device = torch.device("cpu")
    dtype = torch.float64
    pos = torch.tensor(coords, dtype=dtype)
    state = ts.SimState(
        positions=pos,
        masses=torch.ones(len(coords), dtype=dtype),
        cell=torch.eye(3, dtype=dtype).unsqueeze(0) * 40.0,
        atomic_numbers=torch.ones(len(coords), dtype=torch.int64),
        pbc=False,
    )
    model = SpecPotentialModel([evaluate], device=device, dtype=dtype)
    output = model(state)
    energy, gradient = lennard_jones(coords)
    assert counter.calls == 1
    assert abs(float(output["energy"][0]) - energy) < 1e-12
    assert np.allclose(output["forces"].detach().cpu().numpy(), -gradient.reshape(-1, 3))
    model(state)
    assert counter.calls == 1

    box = np.array([12.0, 12.0, 12.0])
    morse_counter = HarnessCounter(cache=True, cap=4)
    morse_potential = RgpotCounter(
        lambda positions, _numbers=None, _box=None: morse_bulk(positions, box),
        morse_counter,
    )

    def morse_evaluate(rows):
        return morse_potential.calculate(np.asarray(rows, dtype=np.float64).reshape(-1))

    morse_model = SpecPotentialModel([morse_evaluate], device=device, dtype=dtype)
    morse_state = ts.SimState(
        positions=pos,
        masses=torch.ones(len(coords), dtype=dtype),
        cell=torch.diag(torch.tensor(box, dtype=dtype)).unsqueeze(0),
        atomic_numbers=torch.ones(len(coords), dtype=torch.int64),
        pbc=True,
    )
    morse_out = morse_model(morse_state)
    morse_energy, morse_gradient = morse_bulk(coords, box)
    assert morse_counter.calls == 1
    assert abs(float(morse_out["energy"][0]) - morse_energy) < 1e-12
    assert np.allclose(morse_out["forces"].detach().cpu().numpy(), -morse_gradient.reshape(-1, 3))
