"""SciPy minimization templates and one measured start."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e4e854eaa8f18d807cd3496028e257e36caa93cc"
METHODS = ("L-BFGS-B", "BFGS", "CG", "trust-krylov", "trust-ncg", "Newton-CG")


def test_squash_stays_inside_the_cap_and_matches_its_hessian():
    import numpy as np

    from optbench.runners.scipy_min import SQUASH_FRACTION, _squash_hvp, _squash_image, _squash_grad, _squash_lengths

    mapped, first, _second = _squash_lengths(0.05, 0.2)
    assert mapped == 0.05 and first == 1.0
    mapped, _first, _second = _squash_lengths(0.25, 0.2)
    assert SQUASH_FRACTION * 0.2 < mapped < 0.2
    mapped, _first, _second = _squash_lengths(1.0, 0.2)
    assert mapped <= 0.2
    rng = np.random.default_rng(0)
    anchor = rng.normal(size=12)
    trial = anchor.copy()
    trial[0:3] += np.array([0.19, 0.0, 0.0])
    direction = rng.normal(size=12)

    def physical(point):
        values = np.asarray(point, dtype=np.float64).reshape(-1)
        return float(np.dot(values, values)), 2.0 * values

    def grad_only(point):
        _image, meta = _squash_image(point, anchor, 0.2)
        _energy, gradient = physical(_image)
        return _squash_grad(gradient, meta)

    step = 1e-6
    finite = (grad_only(trial + step * direction) - grad_only(trial - step * direction)) / (2 * step)
    image, meta = _squash_image(trial, anchor, 0.2)
    _energy, gradient = physical(image)
    analytic = _squash_hvp(lambda vector: 2.0 * np.asarray(vector, dtype=np.float64), direction, gradient, meta)
    assert np.max(np.abs(finite - analytic)) < 1e-5
    assert float(np.max(np.linalg.norm((image - anchor).reshape(-1, 3), axis=1))) < 0.2


def test_solver_options_match_the_problem_norm():
    from optbench.runners.scipy_min import solver_options

    import sys

    lbfgsb = solver_options("L-BFGS-B", 114, 0.01, 10000, 0.2, noise=False)
    assert math.isclose(lbfgsb["gtol"], 0.01 / math.sqrt(114))
    assert lbfgsb["ftol"] == sys.float_info.epsilon
    bfgs = solver_options("BFGS", 114, 0.01, 10000, 0.2, noise=False)
    assert math.isclose(bfgs["gtol"], 0.01 / math.sqrt(114))
    assert bfgs["norm"] == "inf"
    assert bfgs["maxiter"] == 200000
    noisy = solver_options("BFGS", 114, 0.01, 10000, 0.2, noise=True)
    assert noisy["gtol"] == 0.0
    trust = solver_options("trust-ncg", 114, 0.01, 10000, 0.2, noise=False)
    assert trust["gtol"] == 0.01
    assert math.isclose(trust["max_trust_radius"], 0.2 * math.sqrt(38))
    assert trust["initial_trust_radius"] < trust["max_trust_radius"]
    newton = solver_options("Newton-CG", 114, 0.01, 10000, 0.2, noise=False)
    assert newton["xtol"] == 0.0
    assert "gtol" not in newton
    morse = solver_options("trust-krylov", 300, 1e-3, 10000, None, noise=False)
    assert "max_trust_radius" not in morse
    assert morse["gtol"] == 1e-3
    with pytest.raises(ValueError, match="unknown scipy method"):
        solver_options("nelder-mead", 114, 0.01, 10000, 0.2, noise=False)


def test_scipy_templates_cover_lj38_and_morse():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "scipy")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == "SciPy"
    found = [row for row in list_templates(ROOT) if str(row["id"]).startswith("scipy/")]
    assert len(found) == 12
    seen = set()
    for template in found:
        assert template["engine"] == "scipy"
        assert template["commit"] == COMMIT
        assert template["code"] == "SciPy"
        assert template["code_version"] == "1.18.1"
        assert template["tuning"] == "default"
        assert template["scipy"]["method"] in METHODS
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        if template["problem"] == "minimization/lj38":
            assert template["scipy"]["maxstep"] == 0.2
            assert template["ini"]["Optimizer"]["converged_force"] == 0.01
        else:
            assert template["problem"] == "minimization/morse-bulk"
            assert "maxstep" not in template["scipy"]
            assert template["ini"]["Optimizer"]["converged_force"] == 0.001
        assert template["ini"]["Optimizer"]["max_iterations"] == 10000
        seen.add((template["problem"], template["scipy"]["method"]))
    for problem in ("minimization/lj38", "minimization/morse-bulk"):
        for method in METHODS:
            assert (problem, method) in seen


def test_scipy_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        captured["resume"] = kwargs["resume"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 0,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.scipy_min.run_template", fake)
    summary = run(
        ROOT,
        "minimization/lj38",
        "scipy/lj38-bfgs",
        output=tmp_path / "benchmark.dat",
        workers=2,
        resume=True,
    )
    assert captured == {
        "template": "scipy/lj38-bfgs",
        "problem": "minimization/lj38",
        "workers": 2,
        "resume": True,
    }
    assert summary["template"] == "scipy/lj38-bfgs"


def _lj_job(method: str, cap: int) -> dict:
    from optbench.runners.rgmin_min import _load_geometries

    problem = resolve_problem(ROOT, "minimization/lj38")
    rows = _load_geometries(ROOT, problem, None, Path("/tmp/optbench-scipy-geo"), Path("/tmp/optbench-scipy-geo/unused.dat"))
    name, coords, _box = rows[0]
    return {
        "index": 0,
        "start": name,
        "coords": coords.tolist(),
        "box": None,
        "kind": "lj",
        "method": method,
        "maxstep": 0.2,
        "gtol": 0.01,
        "cap": cap,
    }


def test_lj38_bfgs_start_stays_inside_the_cutoff():
    pytest.importorskip("scipy")
    import scipy

    if scipy.__version__ != "1.18.1":
        pytest.skip(f"scipy {scipy.__version__} is not the pinned 1.18.1")
    from optbench.runners.scipy_min import _minimize_job

    record = _minimize_job(_lj_job("BFGS", 10000))
    assert record["counter"] == "harness"
    assert record["energy"] < -10.0
    assert 0 < record["force_calls"] < 10000
    assert record["failure"] != "crash"


def test_each_analytic_product_counts_as_one_force_call():
    pytest.importorskip("scipy")
    import numpy as np
    import scipy

    if scipy.__version__ != "1.18.1":
        pytest.skip(f"scipy {scipy.__version__} is not the pinned 1.18.1")
    from optbench.counter import HarnessCounter
    from optbench.runners.scipy_min import search

    counter = HarnessCounter(cache=True, cap=30)
    start = np.linspace(0.2, 0.8, 9)

    def evaluate(coords):
        point = np.asarray(coords, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in point)
        fresh, cached = counter.charge(key)
        if not fresh and cached is not None:
            energy, gradient = cached
            return float(energy), np.asarray(gradient, dtype=np.float64)
        energy = float(np.dot(point, point))
        gradient = 2.0 * point
        counter.remember(key, (energy, tuple(float(value) for value in gradient)))
        return energy, gradient

    def hessian(coords):
        size = int(np.asarray(coords).reshape(-1).size)
        return 2.0 * np.eye(size)

    info = search(
        start,
        "Newton-CG",
        evaluate,
        hessian=hessian,
        force_tol=1e-10,
        cap=30,
        maxstep=None,
        counter=counter,
    )
    assert info["hessian_calls"] > 0
    assert counter.calls + info["hessian_calls"] <= 30
    assert info["cap_hit"] or info["stopped"]


def test_force_call_cap_stops_lbfgsb():
    pytest.importorskip("scipy")
    import scipy

    if scipy.__version__ != "1.18.1":
        pytest.skip(f"scipy {scipy.__version__} is not the pinned 1.18.1")
    from optbench.runners.scipy_min import _minimize_job

    record = _minimize_job(_lj_job("L-BFGS-B", 5))
    assert record["force_calls"] <= 5
    assert record["failure"] == "cap"
    assert record["counter"] == "harness"
