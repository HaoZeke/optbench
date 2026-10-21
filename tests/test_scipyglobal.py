"""SciPy global-optimizer templates and the harness stop."""

import copy
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem, run
from optbench.lj_minima import LJ38_MINIMUM, QUOTED_ENERGY
from optbench.potentials import lennard_jones
from optbench.runners.scipyglobal import (
    COORD_BOUND,
    HarnessObjective,
    energy_target,
    load_starts,
    search_start,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e4e854eaa8f18d807cd3496028e257e36caa93cc"
METHODS = (
    "basinhopping",
    "dual_annealing",
    "differential_evolution",
    "shgo",
)
BOUNDED = {"dual_annealing", "differential_evolution", "shgo"}


def _templates():
    return [row for row in list_templates(ROOT) if str(row["id"]).startswith("scipyglobal/")]


def test_scipyglobal_templates_match_the_registry_pin():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "scipyglobal")
    assert pinned["name"] == "SciPy"
    assert pinned["commit"] == COMMIT
    assert pinned["ref"] == "v1.18.1"
    found = _templates()
    assert len(found) == 8
    eon_cap = int(load_template(ROOT, "eon-svn/bh-lj38")["failure_cap"])
    seen = set()
    for template in found:
        assert template["engine"] == "scipyglobal"
        assert template["code"] == "SciPy"
        assert template["code_version"] == "1.18.1"
        assert template["commit"] == COMMIT
        method = template["scipy"]["method"]
        if method == "shgo" and template["problem"] == "globalopt/lj38":
            assert template["tuning"] == "sobol"
        else:
            assert template["tuning"] == "default"
        assert template["scipy"]["method"] in METHODS
        assert int(template["scipy"]["seed"]) == 314159
        problem = resolve_problem(ROOT, template["problem"])
        problem_cap = (problem.get("convergence") or {}).get("max_force_calls")
        expected_cap = int(problem_cap) if problem_cap is not None else eon_cap
        assert int(template["failure_cap"]) == expected_cap
        assert int(template["stop"]["max_force_calls"]) == expected_cap
        if problem_cap is None:
            assert expected_cap == 2000000
        if method in BOUNDED:
            assert float(template["scipy"]["bound"]) == COORD_BOUND
        else:
            assert "bound" not in template["scipy"]
        _check_stop(problem, template)
        seen.add((template["problem"], method))
    for problem in ("globalopt/lj38", "globalopt/lj75"):
        for method in METHODS:
            assert (problem, method) in seen


def test_lj38_minimum_matches_the_quoted_wales_energy():
    energy, gradient = lennard_jones(np.asarray(LJ38_MINIMUM, dtype=np.float64))
    assert abs(float(energy) - float(QUOTED_ENERGY)) < 1e-5
    assert float(np.linalg.norm(gradient)) < 0.02
    problem = resolve_problem(ROOT, "globalopt/lj38")
    window = float(problem["convergence"]["energy_above_minimum"])
    assert float(problem["reference"]["energy"]) == pytest.approx(-173.928427)
    target = energy_target(problem)
    assert target == pytest.approx(float(problem["reference"]["energy"]) + window)
    assert target == pytest.approx(-173.927427)
    lj75 = resolve_problem(ROOT, "globalopt/lj75")
    window = float(lj75["convergence"]["energy_above_minimum"])
    assert energy_target(lj75) == pytest.approx(float(lj75["reference"]["energy"]) + window)
    assert int(lj75["convergence"]["max_force_calls"]) == 1000000


def test_lj_starts_are_centered_inside_the_bound(tmp_path: Path):
    lj38 = load_starts(ROOT, resolve_problem(ROOT, "globalopt/lj38"), tmp_path)
    assert len(lj38) == 100
    assert [name for name, _coords in lj38] == [str(index) for index in range(100)]
    for _name, coords in lj38:
        assert coords.shape == (114,)
        assert coords.dtype == np.float64
        assert float(np.max(np.abs(coords))) <= COORD_BOUND
    lj75 = load_starts(ROOT, resolve_problem(ROOT, "globalopt/lj75"), tmp_path)
    assert len(lj75) == 200
    assert lj75[0][0] == "1"
    assert lj75[-1][0] == "200"
    for _name, coords in lj75:
        assert coords.shape == (225,)
        assert coords.dtype == np.float64
        assert float(np.max(np.abs(coords))) <= COORD_BOUND


def test_fun_and_jac_at_one_geometry_count_once():
    def bowl(x):
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        delta = arr - 1.0
        return float(np.dot(delta, delta)), 2.0 * delta

    objective = HarnessObjective(bowl, 20, None)
    point = np.zeros(4)
    assert objective.fun(point) == pytest.approx(4.0)
    jac = objective.jac(point)
    assert jac.shape == (4,)
    assert objective.counter.calls == 1


def _bowl(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    delta = arr - 1.0
    return float(np.dot(delta, delta)), 2.0 * delta


def _constant(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return 1.0, np.zeros_like(arr)


def _job(method: str, evaluate, cap: int, target, bound):
    coords = [0.0, 0.0]
    from optbench.runners.scipyglobal import method_parameters

    parameters = method_parameters(method, cap, len(coords), bound)
    parameters["seed"] = 314159
    return {
        "index": 0,
        "start": "0",
        "coords": coords,
        "cap": cap,
        "method": method,
        "seed": 314159,
        "bound": bound,
        "target": target,
        "commit": COMMIT,
        "code_version": "1.18.1",
        "parameters": parameters,
        "template_id": f"test/{method}",
        "evaluate": evaluate,
    }


@pytest.mark.parametrize("method", METHODS)
def test_cap_stops_at_five_calls(method: str):
    pytest.importorskip("scipy")
    from optbench.runners.scipyglobal import method_parameters

    bound = None if method == "basinhopping" else COORD_BOUND
    job = _job(method, _constant, 5, 0.0, bound)
    if method == "shgo":
        # Two coordinates request five simplicial vertices, which can finish
        # on the cap boundary. Four coordinates request 17 vertices, so the
        # fifth evaluation stops the search.
        job["coords"] = [0.0, 0.0, 0.0, 0.0]
        parameters = method_parameters("shgo", 5, 4, bound)
        parameters["seed"] = 314159
        job["parameters"] = parameters
    record = search_start(job)
    assert record["force_calls"] == 5
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert record["energy"] == pytest.approx(1.0)


def test_basinhopping_reaches_a_quadratic_bowl():
    pytest.importorskip("scipy")
    record = search_start(_job("basinhopping", _bowl, 200, 1e-8, None))
    assert record["success"] is True
    assert record["failure"] is None
    assert record["force_calls"] < 200
    assert record["energy"] <= 1e-8


def test_missing_window_writes_a_partial_row(tmp_path: Path):
    pytest.importorskip("scipy")
    template = dict(load_template(ROOT, "scipyglobal/lj75-basinhopping"))
    template["failure_cap"] = 12
    template["stop"] = {"max_force_calls": 12}
    problem = copy.deepcopy(resolve_problem(ROOT, "globalopt/lj75"))
    problem.pop("reference", None)
    problem["convergence"] = {"max_force_calls": 12}
    output = tmp_path / "benchmark.dat"
    summary = __import__("optbench.runners.scipyglobal", fromlist=["run_template"]).run_template(
        ROOT,
        problem,
        template,
        output=output,
        workers=1,
        work=tmp_path / "work",
        limit=1,
        check_source=False,
    )
    text = output.read_text(encoding="utf-8")
    assert "partial true\n" in text
    assert "no energy window" in text
    assert summary["starts"] == 1
    row = json.loads(output.with_suffix(".starts.jsonl").read_text(encoding="utf-8"))
    assert row["counter"] == "harness"
    assert row["force_calls"] == 12
    assert row["energy"] is not None
    assert "trace" not in row


def test_a_longer_trace_clips_to_the_problem_cap():
    from optbench.runners.scipyglobal import _reusable, method_parameters

    old_cap = 20
    new_cap = 10
    old_parameters = method_parameters("basinhopping", old_cap, 2, None)
    new_parameters = method_parameters("basinhopping", new_cap, 2, None)
    old_parameters["seed"] = 1
    new_parameters["seed"] = 1
    saved = {
        "schema": "optbench.start/v2",
        "parameters": old_parameters,
        "cap": old_cap,
        "commit": COMMIT,
        "seed": 1,
        "method": "basinhopping",
        "target": None,
        "force_calls": 20,
        "energy": -10.0,
        "best_calls": 15,
        "stop": "cap",
        "success": False,
        "failure": "cap",
        "reasons": ["cap"],
        "trace": [[3, -1.0], [8, -4.0], [15, -10.0]],
    }
    job = {
        "cap": new_cap,
        "commit": COMMIT,
        "seed": 1,
        "method": "basinhopping",
        "target": -5.0,
        "parameters": new_parameters,
        "coords": [0.0, 0.0],
        "bound": None,
    }
    capped = _reusable(saved, job)
    assert capped is not None
    assert capped["success"] is False
    assert capped["failure"] == "cap"
    assert capped["force_calls"] == 10
    assert capped["energy"] == pytest.approx(-4.0)
    assert capped["parameters"]["niter"] == 10
    job["target"] = -3.0
    hit = _reusable(saved, job)
    assert hit is not None
    assert hit["success"] is True
    assert hit["force_calls"] == 8
    assert hit["energy"] == pytest.approx(-4.0)
    assert hit["stop"] == "hit"


def test_shgo_crash_reuses_under_the_lower_cap():
    from optbench.runners.scipyglobal import _reusable, method_parameters

    old_parameters = method_parameters("shgo", 2000000, 4, COORD_BOUND)
    new_parameters = method_parameters("shgo", 1000000, 4, COORD_BOUND)
    old_parameters["seed"] = 314159
    new_parameters["seed"] = 314159
    saved = {
        "schema": "optbench.start/v2",
        "parameters": old_parameters,
        "cap": 2000000,
        "commit": COMMIT,
        "seed": 314159,
        "method": "shgo",
        "target": None,
        "force_calls": 0,
        "energy": None,
        "best_calls": None,
        "stop": "crash",
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "error": "MemoryError: ",
        "trace": [],
    }
    job = {
        "cap": 1000000,
        "commit": COMMIT,
        "seed": 314159,
        "method": "shgo",
        "target": -397.491331,
        "parameters": new_parameters,
        "coords": [0.0, 0.0, 0.0, 0.0],
        "bound": COORD_BOUND,
    }
    found = _reusable(saved, job)
    assert found is not None
    assert found["failure"] == "crash"
    assert found["force_calls"] == 0
    assert found["cap"] == 1000000
    assert found["energy"] is None


def test_declared_partial_reason_reaches_the_board(tmp_path: Path):
    from optbench.board import _campaign_rows

    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    (campaign / "row.dat").write_text(
        "partial true\n"
        "partial_reason no energy window on this problem\n"
        "counter harness\n"
        "template scipyglobal/lj75-basinhopping\n"
        "problem globalopt/lj75\n"
        "code SciPy\n"
        "cap 2000000\n",
        encoding="utf-8",
    )
    (campaign / "row.starts.jsonl").write_text(
        json.dumps(
            {
                "force_calls": 10,
                "success": False,
                "failure": "divergence",
                "counter": "harness",
                "start": "1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows, unparsed = _campaign_rows(tmp_path, {}, {})
    assert unparsed == []
    assert rows[0]["_declared_partial"] == "no energy window on this problem"
    assert rows[0]["code"] == "SciPy"


def test_scipyglobal_engine_does_not_need_eonclient(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, **kwargs):
        captured["template"] = template["id"]
        captured["problem"] = problem["id"]
        captured["workers"] = kwargs["workers"]
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 0,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.scipyglobal.run_template", fake)
    summary = run(
        ROOT,
        "globalopt/lj38",
        "scipyglobal/lj38-basinhopping",
        output=tmp_path / "benchmark.dat",
        workers=3,
    )
    assert captured == {
        "template": "scipyglobal/lj38-basinhopping",
        "problem": "globalopt/lj38",
        "workers": 3,
    }
    assert summary["problem"] == "globalopt/lj38"


def test_shgo_default_sampling_is_the_constructor_and_sobol_stays_available():
    from optbench.runners.scipyglobal import (
        SHGO_ADDRESS_BYTES,
        SHGO_SOBOL_ADDRESS_BYTES,
        method_parameters,
        shgo_sample_count,
    )

    assert shgo_sample_count(2) == 4
    assert shgo_sample_count(4) == 8
    assert shgo_sample_count(114) == 128
    assert shgo_sample_count(225) == 256
    stock = method_parameters("shgo", 1000000, 225, COORD_BOUND)
    assert stock["sampling_method"] == "simplicial"
    assert stock["n"] == 100
    assert stock["iters"] == 1
    assert stock["minimizer"] == "SLSQP"
    assert stock["ftol"] == 1e-12
    assert stock["jac"] is True
    assert stock["workers"] == 1
    assert stock["address_space_bytes"] == SHGO_ADDRESS_BYTES
    assert SHGO_ADDRESS_BYTES == 2 * 1024**3
    parameters = method_parameters("shgo", 2000000, 114, COORD_BOUND, "sobol")
    assert parameters["sampling_method"] == "sobol"
    assert parameters["n"] == 128
    assert parameters["iters"] == 1
    assert parameters["minimizer"] == "SLSQP"
    assert parameters["ftol"] == 1e-12
    assert parameters["jac"] is True
    assert parameters["address_space_bytes"] == SHGO_SOBOL_ADDRESS_BYTES
    assert SHGO_SOBOL_ADDRESS_BYTES == 10 * 1024**3
    wide = method_parameters("shgo", 1000000, 225, COORD_BOUND, "sobol")
    assert wide["n"] == 256
    assert wide["sampling_method"] == "sobol"


def test_asv_shgo_job_limits_its_address_space():
    from benchmarks.bench_scipyglobal import _job, _specs

    specs = {row["template_id"]: row for row in _specs()}
    for template_id, spec in specs.items():
        job = _job(spec, "0")
        assert job["limit_address_space"] is (spec["template"]["scipy"]["method"] == "shgo")
        assert template_id.startswith("scipyglobal/")
