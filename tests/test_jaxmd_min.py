import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_md")
pytest.importorskip("jaxopt")
pytest.importorskip("optax")

from optbench.runners import jaxmd_min


class _Counter:
    def __init__(self) -> None:
        self.calls = 0


def _bowl(counter: _Counter):
    def evaluate(flat):
        counter.calls += 1
        point = np.asarray(flat, dtype=np.float64).reshape(-1)
        return float(point @ point), 2.0 * point

    return evaluate


def _job(counter: _Counter, method: str) -> dict:
    return {
        "method": method,
        "maxstep": 0.2,
        "step_size": 0.1,
        "history_size": 10,
        "memory_size": 10,
        "gtol": 1e-6,
        "counter": counter,
    }


def test_overlong_lbfgs_trial_does_not_call_the_potential():
    counter = _Counter()
    jaxmd_min._SLOT["evaluate"] = _bowl(counter)
    jaxmd_min._SLOT["guard"] = True
    jaxmd_min._SLOT["maxstep"] = 0.2
    jaxmd_min._SLOT["anchor"] = np.zeros(3)
    jaxmd_min._SLOT["e0"] = 1.5
    energy, grad = jaxmd_min._host_energy_grad(np.array([1.0, 0.0, 0.0]))
    assert counter.calls == 0
    assert energy > 1.5
    assert grad.shape == (3,)


def test_fire_and_gradient_descent_minimize_a_bowl():
    start = np.array([[0.4, -0.2, 0.1]], dtype=np.float64)
    for method, drive in (("fire", jaxmd_min._drive_fire), ("gd", jaxmd_min._drive_gd)):
        counter = _Counter()
        jaxmd_min._SLOT["evaluate"] = _bowl(counter)
        jaxmd_min._SLOT["guard"] = False
        jaxmd_min._SLOT["maxstep"] = 0.2
        point, stopped = drive(_job(counter, method), start.reshape(-1), 200, None)
        assert stopped
        assert counter.calls < 200
        assert np.linalg.norm(point) < 1e-4


def test_both_lbfgs_rows_minimize_a_bowl():
    start = np.array([0.4, -0.2, 0.1], dtype=np.float64)
    for method, drive in (
        ("lbfgs", jaxmd_min._drive_jaxopt),
        ("optax-lbfgs", jaxmd_min._drive_optax),
    ):
        counter = _Counter()
        jaxmd_min._SLOT["evaluate"] = _bowl(counter)
        jaxmd_min._SLOT["guard"] = True
        jaxmd_min._SLOT["maxstep"] = 0.2
        jaxmd_min._SLOT["anchor"] = None
        point, stopped = drive(_job(counter, method), start, 200, None)
        assert stopped, method
        assert counter.calls < 200
        assert np.linalg.norm(point) < 1e-4


def test_lj38_cap_is_a_cap_failure(tmp_path):
    from pathlib import Path

    from optbench.campaign import resolve_problem
    from optbench.runners.rgmin_min import _load_geometries

    repo = Path(__file__).resolve().parents[1]
    loaded = _load_geometries(
        repo,
        resolve_problem(repo, "minimization/lj38"),
        None,
        tmp_path / "work",
        tmp_path / "unused.dat",
    )
    name, coords, _box = loaded[0]
    record = jaxmd_min._minimize_job(
        {
            "index": 0,
            "start": name,
            "coords": coords.tolist(),
            "box": None,
            "kind": "lj",
            "method": "fire",
            "maxstep": 0.2,
            "step_size": 0.0,
            "history_size": 10,
            "memory_size": 10,
            "gtol": 0.01,
            "clean_tol": 0.01,
            "energy_tol": 0.01,
            "cap": 4,
            "sigma": 0.0,
            "seed": 0,
            "noisy": False,
        }
    )
    assert record["schema"] == "optbench.start/v2"
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["force_calls"] <= 4
    assert record["success"] is False
    assert record["failure"] == "cap"
