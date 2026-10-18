"""GPR calculator templates, the import shim, and one counted band step."""

import json
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem
from optbench.failures import FAILURES
from optbench.gpr_neb import CODE_NAME, GPR_COMMIT, GPR_VERSION, _noise, _registry, _settings

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = (
    "double-ended-saddle-search/lj38",
    "double-ended-saddle-search/lj38_2ts_separation",
    "double-ended-saddle-search/pt-island",
    "neb/Pt-heptamer",
)


def test_templates_match_the_registry_pin():
    registry = _registry(ROOT)
    assert registry["commit"] == GPR_COMMIT
    assert registry["name"] == CODE_NAME
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "gprcalc":
            continue
        found.append(template["id"])
        loaded = load_template(ROOT, template["id"])
        assert loaded["commit"] == registry["commit"]
        assert loaded["code"] == CODE_NAME
        assert loaded["code_version"] == GPR_VERSION
        for problem_name in PROBLEMS:
            problem = resolve_problem(ROOT, problem_name)
            _check_stop(problem, loaded)
            settings = _settings(loaded, problem, registry["commit"])
            assert settings["cap"] == int(problem["convergence"]["max_force_calls"])
            assert settings["force_norm"] == float(problem["convergence"]["force_norm"])
            assert settings["n_interior"] == int(problem["band"]["interior_images"])
            assert settings["max_move"] == float(problem["band"]["max_move"])
    assert found == ["gprcalc/gp-ci-neb", "gprcalc/gp-neb"]


def test_default_problem_stop_accepts_the_templates():
    for template in list_templates(ROOT):
        if template.get("engine") != "gprcalc":
            continue
        _check_stop(resolve_problem(ROOT, "minimization/lj38"), template)


def test_force_stop_noise_uses_the_problem_threshold():
    template = load_template(ROOT, "gprcalc/gp-ci-neb")
    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    settings = _settings(template, problem, GPR_COMMIT)
    noise_e, noise_f = _noise(settings, 38)
    assert noise_f == pytest.approx(0.001)
    assert noise_e == pytest.approx(0.0002)
    default = _settings(load_template(ROOT, "gprcalc/gp-neb"), problem, GPR_COMMIT)
    assert _noise(default, 38) == (0.002, 0.1)


def test_energy_and_forces_are_one_base_call():
    ase = pytest.importorskip("ase")
    from ase import Atoms

    from optbench.counter import HarnessCounter, attach_ase_counter
    from optbench.gpr_neb import spec_calculator

    counter = HarnessCounter(cache=False, cap=10)
    calc = spec_calculator("lj", None)
    attach_ase_counter(calc, cap=10, counter=counter)
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert counter.calls == 1
    assert energy < 0.0
    assert forces.shape == (2, 3)
    assert ase.__name__ == "ase"


def test_sph_harm_shim_matches_the_old_argument_order():
    pytest.importorskip("scipy")
    pytest.importorskip("ase")
    from optbench.gpr_compat import install

    install()
    import scipy.special as special

    value = special.sph_harm(1, 2, 0.3, 0.7)
    expected = special.sph_harm_y(2, 1, 0.7, 0.3)
    assert complex(value) == complex(expected)


def test_one_band_step_counts_only_the_base_calculator(tmp_path: Path):
    pytest.importorskip("ase")
    from optbench.gpr_compat import install

    install()
    try:
        import gpr_calc.calculator  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("gpr_calc is not installed")
    import numpy as np

    from optbench.gpr_neb import run_gp_neb
    from optbench.score import score_v2

    reactant = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.0, 0.0]],
        dtype=np.float64,
    )
    product = reactant + np.array([[0.15, 0.02, 0.0], [-0.08, 0.12, 0.0], [0.0, -0.1, 0.0]])
    free = np.ones(3, dtype=bool)
    settings = {
        "potential": "lj",
        "n_interior": 1,
        "max_move": 0.05,
        "force_norm": 0.05,
        "cap": 40,
        "climbing": False,
        "algo": "BFGS",
        "spring_k": 0.1,
        "steps": 1,
        "kernel": "RBF",
        "zeta": 2.0,
        "noise_mode": "absolute",
        "noise_e": 0.002,
        "noise_f": 0.1,
        "noise_e_floor": 0.0002,
        "lmax": 2,
        "nmax": 2,
        "rcut": 3.0,
        "freq": 10,
        "neb_method": "improvedtangent",
    }
    record = run_gp_neb(
        name="tiny",
        reactant=reactant,
        product=product,
        free=free,
        box=None,
        settings=settings,
        scratch=tmp_path,
    )
    assert record["schema"] == "optbench.start/v2"
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["start"] == "tiny"
    assert record["force_calls"] >= 3
    assert record["force_calls"] < settings["cap"] or record["failure"] == "cap"
    if record["success"]:
        assert record["failure"] is None
        assert record["hessian_index"] == 1
        assert record["force_calls"] < settings["cap"]
    else:
        assert record["failure"] in FAILURES
    json.dumps(record, allow_nan=False)
    scored = score_v2([record], settings["cap"])
    assert scored["n"] == 1
    trained = int(settings["n_interior"]) + 2
    assert record["force_calls"] <= trained + record["base_calls"] + 1
    assert record["surrogate_calls"] >= 1
    assert record["force_calls"] < trained + record["base_calls"] + record["surrogate_calls"]


def test_benchmark_classes_name_the_four_problems():
    import benchmarks.bench_gprcalc as bench

    assert bench.Lj38DeGprGpNeb.template_name == "gprcalc/gp-neb"
    assert bench.Lj38DeGprGpNeb.problem_name == "double-ended-saddle-search/lj38"
    assert bench.Lj38DeGprGpNeb.cap == 2500
    assert bench.HeptamerGprGpCiNeb.template_name == "gprcalc/gp-ci-neb"
    assert bench.HeptamerGprGpCiNeb.cap == 10000
    assert bench.PtIslandGprGpNeb.problem_name == "double-ended-saddle-search/pt-island"
    assert bench.Lj38TwoTsGprGpCiNeb.problem_name == "double-ended-saddle-search/lj38_2ts_separation"
    assert len(bench.TEMPLATES) == 2
    assert len(bench.PROBLEMS) == 4
