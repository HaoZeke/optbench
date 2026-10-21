"""GPR calculator templates, the import shim, and one counted band step."""

import json
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, load_template, resolve_problem
from optbench.failures import FAILURES
from optbench.gpr_neb import CODE_NAME, GPR_COMMIT, GPR_VERSION, _noise, _registry, _settings, _unit_scales

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


def test_paper_numbers_are_scaled_into_the_potential_units():
    lj = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    morse = resolve_problem(ROOT, "double-ended-saddle-search/pt-island")
    loaded = [load_template(ROOT, name) for name in ("gprcalc/gp-neb", "gprcalc/gp-ci-neb")]
    lj_scale = _unit_scales("lj")
    assert lj_scale["energy"] == 0.7102
    assert lj_scale["length"] == 2.897 / 2.0 ** (1.0 / 6.0)
    assert _unit_scales("morse_pt") == {"energy": 1.0, "length": 1.0}
    for template in loaded:
        band = template["band"]
        assert band["noise_f"] == 0.075
        assert band["noise_e_structure"] == 0.05
        assert band["rcut"] == 5.0
        assert band["sigma_max"] == 50.0
        assert band["length_min"] == 0.1
        assert "wall_seconds" not in band
        lj_settings = _settings(template, lj, GPR_COMMIT)
        morse_settings = _settings(template, morse, GPR_COMMIT)
        assert morse_settings["noise_f"] == 0.075
        assert morse_settings["rcut"] == 5.0
        assert morse_settings["sigma_max"] == 50.0
        assert morse_settings["optimizer_fmax"] == 0.075
        noise_e, noise_f = _noise(lj_settings, 38)
        assert noise_f == pytest.approx(0.075 * lj_scale["length"] / lj_scale["energy"])
        assert noise_e == pytest.approx(max(0.0002 / lj_scale["energy"], 0.05 / lj_scale["energy"] / 38))
        assert lj_settings["rcut"] == pytest.approx(5.0 / lj_scale["length"])
        assert lj_settings["sigma_max"] == pytest.approx(50.0 / lj_scale["energy"])
        assert lj_settings["length_min"] == pytest.approx(0.1 / lj_scale["length"])
        assert lj_settings["sigma_max"] > lj_settings["sigma0"] > lj_settings["sigma_min"]
        assert lj_settings["algo"] == "FIRE"
        assert lj_settings["initial_path"] == "idpp"
        assert lj_settings["kernel"] == "RBF"
        assert lj_settings["lmax"] == 4
        assert lj_settings["nmax"] == 3
        assert lj_settings["energy_uncertainty"] is False
    plain = _settings(loaded[0], lj, GPR_COMMIT)
    climbed = _settings(loaded[1], lj, GPR_COMMIT)
    assert plain["climbing"] is False
    assert plain["steps"] == 50
    assert plain["passes"] == 2
    assert climbed["climbing"] is True
    assert climbed["steps"] == 100
    assert climbed["passes"] == 3


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
        "noise_e_structure": 0.006,
        "noise_f": 0.1,
        "noise_e_floor": 0.0002,
        "optimizer_fmax": 0.05,
        "initial_path": "idpp",
        "passes": 1,
        "lmax": 2,
        "nmax": 2,
        "rcut": 3.0,
        "freq": 10,
        "neb_method": "improvedtangent",
        "time_limit_seconds": 3600.0,
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


def test_wall_clock_slice_is_not_a_scored_miss(tmp_path: Path):
    import numpy as np

    from optbench.gpr_neb import _scored, run_gp_neb

    reactant = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.0, 0.0]], dtype=np.float64)
    product = reactant + np.array([[0.15, 0.02, 0.0], [-0.08, 0.12, 0.0], [0.0, -0.1, 0.0]])
    settings = {
        "potential": "lj",
        "n_interior": 1,
        "max_move": 0.05,
        "force_norm": 0.05,
        "cap": 40,
        "climbing": False,
        "algo": "FIRE",
        "spring_k": 0.1,
        "steps": 50,
        "passes": 2,
        "kernel": "RBF",
        "zeta": 2.0,
        "noise_e_structure": 0.05,
        "noise_f": 0.075,
        "noise_e_floor": 0.0002,
        "optimizer_fmax": 0.075,
        "initial_path": "idpp",
        "lmax": 4,
        "nmax": 3,
        "rcut": 5.0,
        "freq": 10,
        "neb_method": "improvedtangent",
        "time_limit_seconds": 0.0,
    }
    record = run_gp_neb(
        name="ceiling",
        reactant=reactant,
        product=product,
        free=np.ones(3, dtype=bool),
        box=None,
        settings=settings,
        scratch=tmp_path,
    )
    assert record["incomplete"] is True
    assert record["time_limited"] is True
    assert record["failure"] is None
    assert record["success"] is False
    assert record["force_calls"] == 0
    assert record["counter"] == "harness"
    assert _scored(record) is False
    assert _scored({"failure": "not_converged", "time_limited": True, "incomplete": False}) is False
    assert _scored({"failure": "not_converged", "time_limited": False, "incomplete": False}) is True
    assert not (tmp_path / "resume" / "state.json").exists()


def test_resume_continues_a_wall_clock_slice(tmp_path: Path):
    import numpy as np

    from optbench.gpr_neb import _scored, run_gp_neb

    reactant = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.6, 1.0, 0.0]], dtype=np.float64)
    product = reactant + np.array([[0.15, 0.02, 0.0], [-0.08, 0.12, 0.0], [0.0, -0.1, 0.0]])
    free = np.ones(3, dtype=bool)
    settings = {
        "potential": "lj",
        "n_interior": 1,
        "max_move": 0.05,
        "force_norm": 0.05,
        "cap": 40,
        "climbing": False,
        "algo": "FIRE",
        "spring_k": 0.1,
        "steps": 5,
        "passes": 1,
        "kernel": "RBF",
        "zeta": 2.0,
        "noise_e_structure": 0.006,
        "noise_f": 0.1,
        "noise_e_floor": 0.0002,
        "optimizer_fmax": 0.05,
        "initial_path": "idpp",
        "lmax": 2,
        "nmax": 2,
        "rcut": 3.0,
        "freq": 10,
        "neb_method": "improvedtangent",
        "time_limit_seconds": 0.05,
    }
    first = run_gp_neb(
        name="slice",
        reactant=reactant,
        product=product,
        free=free,
        box=None,
        settings=settings,
        scratch=tmp_path,
    )
    if not first["incomplete"]:
        assert _scored(first)
        assert first["time_limited"] is False
        return
    assert first["failure"] is None
    assert first["force_calls"] >= 3
    assert (tmp_path / "resume" / "state.json").is_file()
    assert (tmp_path / "resume" / "model.json").is_file()
    settings = dict(settings)
    settings["time_limit_seconds"] = 3600.0
    settings["steps"] = 1
    second = run_gp_neb(
        name="slice",
        reactant=reactant,
        product=product,
        free=free,
        box=None,
        settings=settings,
        scratch=tmp_path,
    )
    assert _scored(second)
    assert second["time_limited"] is False
    assert second["force_calls"] >= first["force_calls"]
    assert second["failure"] in {None, "not_converged", "wrong_stationary", "divergence"}
    json.dumps(second, allow_nan=False)


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
