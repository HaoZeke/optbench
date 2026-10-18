"""KLIFF template grid and a short harness training run."""

import tomllib
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "df2d4de2d8c0b3d4ba81cb9aa1834b52de299aa5"


def _templates():
    return [row for row in list_templates(ROOT) if str(row["id"]).startswith("kliff/")]


def test_kliff_templates_cover_both_problems():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "kliff")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == "KLIFF"
    found = _templates()
    assert len(found) == 20
    seen = set()
    for template in found:
        assert template["engine"] == "kliff"
        assert template["code"] == "KLIFF"
        assert template["commit"] == COMMIT
        assert template["code_version"] == "1.0.1"
        assert template["tuning"] in ("default", "tuned")
        assert template["kliff"]["method"] in ("Adam", "LBFGS", "Rprop")
        assert template["kliff"]["seed_base"] == 20261015
        assert template["model"]["hidden"] == [10, 7]
        problem = resolve_problem(ROOT, template["problem"])
        assert _check_stop(problem, template) == int(template["stop"]["max_force_calls"])
        force = float(template["kliff"]["force_coefficient"])
        assert force in (0.0, 0.1)
        if template["kliff"]["method"] == "Rprop":
            assert template["tuning"] == "default"
            assert "learning_rate" not in template["kliff"]
        if template["tuning"] == "tuned":
            assert float(template["kliff"]["learning_rate"]) == 0.01
        if template["tuning"] == "default":
            assert "learning_rate" not in template["kliff"]
        if template["problem"] == "mlopt/emt-pt38":
            assert int(template["stop"]["max_force_calls"]) == 10000
        elif force == 0.0:
            assert int(template["stop"]["max_force_calls"]) == 10000
        else:
            assert int(template["stop"]["max_force_calls"]) == 30000
        grad = 0.02 if force == 0.0 else 0.1
        assert float(template["stop"]["gradient_norm"]) == pytest.approx(grad)
        seen.add((template["problem"], template["kliff"]["method"], template["tuning"], force))
    for problem in ("mlopt/emt-pt38", "mlopt/C2H4"):
        for method in ("Adam", "LBFGS"):
            for force in (0.0, 0.1):
                assert (problem, method, "default", force) in seen
                assert (problem, method, "tuned", force) in seen
        for force in (0.0, 0.1):
            assert (problem, "Rprop", "default", force) in seen
            assert (problem, "Rprop", "tuned", force) not in seen


def test_a_short_run_counts_evaluations_and_stops():
    pytest.importorskip("torch")
    kliff = pytest.importorskip("kliff")
    del kliff
    import numpy as np
    import torch

    from kliff.legacy.loss import LossNeuralNetworkModel

    from optbench.kliff_ml import train_start

    assert "Adam" in LossNeuralNetworkModel.torch_minimize_methods
    assert "LBFGS" in LossNeuralNetworkModel.torch_minimize_methods
    assert "Rprop" in LossNeuralNetworkModel.torch_minimize_methods
    assert torch.optim.Adam(torch.nn.Linear(1, 1).parameters()).defaults["lr"] == 0.001
    lbfgs = torch.optim.LBFGS(torch.nn.Linear(1, 1).parameters())
    assert lbfgs.defaults["lr"] == 1
    assert lbfgs.defaults["max_iter"] == 20
    assert torch.optim.Rprop(torch.nn.Linear(1, 1).parameters()).defaults["lr"] == 0.01

    rng = np.random.default_rng(1)
    features = {
        "fps": rng.normal(size=(2, 2, 2)),
        "energy": rng.normal(size=(2,)),
        "symbols": np.array(["C", "C"]),
    }
    base = {
        "index": 0,
        "seed": 20261015,
        "method": "Adam",
        "learning_rate": None,
        "force_coefficient": 0.0,
        "energy_coefficient": 1.0,
        "hidden": [10, 7],
        "elements": ["C"],
        "width": 2,
        "max_epochs": 5,
    }
    success = train_start(features, {**base, "grad_tol": 1e9, "cap": 5})
    assert success["success"] is True
    assert success["failure"] is None
    assert success["force_calls"] == 1
    assert success["counter"] == "harness"
    assert success["force_rmse"] == -1.0

    capped = train_start(features, {**base, "grad_tol": -1.0, "cap": 3, "max_epochs": 3})
    assert capped["success"] is False
    assert capped["failure"] == "cap"
    assert capped["force_calls"] == 3

    forced = dict(features)
    forced["jacobian"] = rng.normal(size=(2, 2, 2, 2, 3))
    forced["forces"] = rng.normal(size=(2, 2, 3))
    with_forces = train_start(
        forced,
        {**base, "force_coefficient": 0.1, "grad_tol": 1e9, "cap": 4, "method": "Rprop"},
    )
    assert with_forces["success"] is True
    assert with_forces["force_calls"] == 1
    assert with_forces["force_rmse"] != -1.0
    assert with_forces["force_rmse"] > 0.0

    lbfgs_run = train_start(
        features,
        {**base, "method": "LBFGS", "grad_tol": -1.0, "cap": 4, "max_epochs": 4},
    )
    assert lbfgs_run["success"] is False
    assert lbfgs_run["failure"] == "cap"
    assert lbfgs_run["force_calls"] == 4
