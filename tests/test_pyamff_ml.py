"""PyAMFF templates stay on the two machine-learning problems."""

import tomllib
from pathlib import Path

from optbench.board import stated_full_n
from optbench.campaign import list_templates, resolve_problem
from optbench.runners.pyamff_ml import OPTIMIZERS, classify_stop

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "240f51d01ca22969369f9bc21760c61cbb3b6665"
PROBLEMS = ("mlopt/emt-pt38", "mlopt/C2H4")


def test_stop_classes():
    assert classify_stop(calls=12, cap=100, converged=True, finite=True, error="") == (True, None)
    assert classify_stop(calls=100, cap=100, converged=True, finite=True, error="") == (False, "cap")
    assert classify_stop(calls=100, cap=100, converged=False, finite=True, error="") == (False, "cap")
    assert classify_stop(calls=4, cap=100, converged=False, finite=False, error="") == (False, "divergence")
    assert classify_stop(calls=4, cap=100, converged=False, finite=True, error="RuntimeError") == (False, "crash")
    assert classify_stop(calls=4, cap=100, converged=False, finite=True, error="") == (False, "wrong_stationary")


def test_pyamff_templates_cover_both_problems_and_seven_optimizers():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pyamff")
    assert pinned["commit"] == COMMIT
    found = [row for row in list_templates(ROOT) if row.get("engine") == "pyamff"]
    assert len(found) == 28
    seen = set()
    for template in found:
        assert template["commit"] == COMMIT
        assert template["code_version"] == "0.1.0-beta"
        assert template["tuning"] == "default"
        assert template["problem"] in PROBLEMS
        problem = resolve_problem(ROOT, template["problem"])
        assert stated_full_n(ROOT, problem) == 50
        section = template["pyamff"]
        assert section["optimizer"] in OPTIMIZERS
        assert section["learning_rate"] == 0.01
        assert section["hidden_layers"] == [10, 7]
        assert section["activation"] == "tanh"
        assert section["initial_weights"] == "random_uniform"
        assert section["seed_base"] == 20261015
        stop = template["stop"]
        assert stop["norm"] == "l2"
        coeff = section["force_coefficient"]
        if template["id"].endswith("-force"):
            assert coeff == 0.1
            assert stop["max_steps"] == 30000
            assert stop["gradient_norm"] == 0.1
        else:
            assert coeff == 0.0
            assert stop["max_steps"] == 10000
            assert stop["gradient_norm"] == 0.02
        seen.add((template["problem"], section["optimizer"], coeff))
    for problem in PROBLEMS:
        for name in OPTIMIZERS:
            assert (problem, name, 0.0) in seen
            assert (problem, name, 0.1) in seen
