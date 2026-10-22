"""ASE global-optimisation settings are the documented example and the defaults."""

import inspect
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import geometry_archive, load_template
from optbench.ccd import lennard_jones_minimum
from optbench.problems import load_problem
from optbench.runners.ase_go import (
    BASIN_NOTE,
    CARTESIAN,
    METHODS,
    MINIMA_NOTE,
    SEED,
    _spec,
    _step_artifact,
    basin_settings,
    campaign_stem,
    choose_job,
    load_campaign_records,
    load_starts,
    minima_settings,
    require_cartesian,
    search_one,
    template_id,
    write_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def test_basin_example_is_the_documented_temperature_and_step():
    pytest.importorskip("ase")
    from ase import units
    from ase.optimize import FIRE, LBFGS
    from ase.optimize.basin import BasinHopping

    params = inspect.signature(BasinHopping.__init__).parameters
    assert params["temperature"].default == 100 * units.kB
    assert params["dr"].default == 0.1
    assert params["fmax"].default == 0.1
    assert params["optimizer"].default is FIRE
    settings = basin_settings()
    assert settings["temperature"] == 100 * units.kB
    assert settings["dr"] == 0.5
    assert settings["fmax"] == 0.1
    assert settings["optimizer"] is LBFGS
    assert settings["seed"] == SEED
    assert settings["temperature"] != 0.8


def test_minima_hopping_uses_the_class_defaults():
    pytest.importorskip("ase")
    from ase.optimize import BFGSLineSearch
    from ase.optimize.minimahopping import MinimaHopping

    settings = minima_settings()
    defaults = MinimaHopping._default_settings
    for key, value in settings.items():
        if key == "seed":
            continue
        assert defaults[key] == value
    assert settings["optimizer"] is BFGSLineSearch
    assert settings["rng"] is None
    assert settings["T0"] == 1000.0
    assert settings["Ediff0"] == 0.5
    assert settings["fmax"] == 0.05
    assert settings["timestep"] == 1.0
    assert settings["mdmin"] == 2


def test_templates_repeat_the_documented_numbers():
    pytest.importorskip("ase")
    for problem_id in CARTESIAN:
        for method in METHODS:
            template = load_template(ROOT, template_id(problem_id, method))
            spec = _spec(ROOT, problem_id, method)
            assert spec["template_id"] == template["id"]
            assert spec["cap"] > 0
            note = template["note"]
            if method == "basin":
                assert BASIN_NOTE in note
                assert template["optimizer"]["local"] == "LBFGS"
                assert template["optimizer"]["parameters"]["dr"] == 0.5
            else:
                assert MINIMA_NOTE in note
                assert template["optimizer"]["local"] == "QuasiNewton"
                assert template["optimizer"]["parameters"]["T0"] == 1000.0
                assert template["optimizer"]["parameters"]["Ediff0"] == 0.5


def test_lj38_cap_is_the_existing_eon_failure_cap():
    problem = load_problem(ROOT / "problems" / "globalopt" / "lj38.toml")
    assert int(problem["convergence"]["max_force_calls"]) == 2000000
    eon = load_template(ROOT, "eon-svn/bh-lj38")
    for method in METHODS:
        template = load_template(ROOT, template_id("globalopt/lj38", method))
        assert int(template["failure_cap"]) == int(eon["failure_cap"]) == 2000000
        assert int(template["stop"]["max_force_calls"]) == 2000000
        assert "1795901" in template["note"]


def test_rigid_tip4p_is_outside_these_optimizers():
    for problem_id in ("globalopt/tip4p10", "globalopt/tip4p15", "globalopt/tip4p20"):
        with pytest.raises(ValueError, match="rigid TIP4P"):
            require_cartesian(problem_id)


def test_step_logs_are_the_only_discarded_artifacts():
    assert _step_artifact("md00001.traj")
    assert _step_artifact("qn00002.log")
    assert not _step_artifact("minima.traj")
    assert not _step_artifact("hop.log")
    assert not _step_artifact("lowest.traj")


def test_archives_match_the_stated_start_counts():
    for problem_id in CARTESIAN:
        problem = load_problem(ROOT / "problems" / f"{problem_id}.toml")
        spec = _spec(ROOT, problem_id, "basin")
        starts = load_starts(geometry_archive(ROOT, problem), problem_id, spec["atoms"])
        assert len(starts) == int(problem["sample"]["starts"])


def test_binary_component_order_follows_the_lammps_pair_coeff():
    problem = load_problem(ROOT / "problems" / "globalopt" / "lj100-binary.toml")
    spec = _spec(ROOT, "globalopt/lj100-binary", "basin")
    starts = {
        item["index"]: item
        for item in load_starts(geometry_archive(ROOT, problem), "globalopt/lj100-binary", spec["atoms"])
    }
    species = starts[15]["species"]
    assert int(np.sum(species == 1)) == 58
    assert int(np.sum(species == 0)) == 42
    assert starts[15]["symbols"][0] == "H"
    assert starts[15]["symbols"][-1] == "C"
    assert float(starts[15]["masses"][0]) == pytest.approx(12.011)
    assert float(starts[15]["masses"][-1]) == pytest.approx(12.011)


def test_campaign_file_names_keep_the_ase_version(tmp_path: Path):
    spec = _spec(ROOT, "globalopt/lj75", "basin")
    record = {
        "schema": "optbench.start/v2",
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "cap",
        "reasons": ["energy", "cap"],
        "force_calls": 10,
        "energy": -1.0,
        "start": "coords.1",
    }
    dat, jsonl = write_campaign(tmp_path, [record], spec=spec, partial=True)
    assert dat.name == f"{campaign_stem('globalopt/lj75', 'basin')}.dat"
    assert jsonl.name == "ase-3.29.0-lj75-basin.starts.jsonl"
    assert "100 * kB" in dat.read_text(encoding="utf-8")
    loaded = load_campaign_records(jsonl)
    assert loaded[0]["start"] == "coords.1"
    assert load_campaign_records(tmp_path / "missing.jsonl") == []


def test_choose_job_starts_every_method_before_filling_a_long_cap():
    keys = [
        ("globalopt/lj75", "basin"),
        ("globalopt/lj75", "minima"),
        ("globalopt/lj38", "basin"),
        ("globalopt/lj38", "minima"),
        ("globalopt/lj100-binary", "basin"),
        ("globalopt/lj100-binary", "minima"),
        ("globalopt/lj150", "basin"),
        ("globalopt/lj150", "minima"),
        ("globalopt/lj98", "basin"),
        ("globalopt/lj98", "minima"),
    ]
    caps = {
        ("globalopt/lj75", "basin"): 1_000_000,
        ("globalopt/lj75", "minima"): 1_000_000,
        ("globalopt/lj38", "basin"): 2_000_000,
        ("globalopt/lj38", "minima"): 2_000_000,
        ("globalopt/lj100-binary", "basin"): 2_000_000,
        ("globalopt/lj100-binary", "minima"): 2_000_000,
        ("globalopt/lj150", "basin"): 10_000_000,
        ("globalopt/lj150", "minima"): 10_000_000,
        ("globalopt/lj98", "basin"): 1_000_000_000,
        ("globalopt/lj98", "minima"): 1_000_000_000,
    }
    queues = {key: deque([1, 2]) for key in keys}
    inflight = {key: 0 for key in keys}
    done = {key: 0 for key in keys}
    picked = []
    for _ in range(8):
        key = choose_job(queues, inflight, done, caps)
        picked.append(key)
        queues[key].popleft()
        inflight[key] += 1
    assert ("globalopt/lj98", "basin") not in picked
    assert ("globalopt/lj98", "minima") not in picked
    assert ("globalopt/lj75", "basin") in picked
    assert ("globalopt/lj150", "basin") in picked
    done[("globalopt/lj75", "basin")] = 1
    inflight[("globalopt/lj75", "basin")] = 0
    assert choose_job(queues, inflight, done, caps) == ("globalopt/lj98", "basin")


def test_stock_basin_hopping_accepts_the_displaced_point():
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import LBFGS
    from ase.optimize.basin import BasinHopping

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = np.asarray(self.atoms.get_positions(), dtype=float)
            self.results["energy"] = float(np.sum(pos * pos))
            self.results["forces"] = -2.0 * pos

    origins = []

    class Recording(BasinHopping):
        def move(self, ro):
            origins.append(np.array(ro, dtype=float, copy=True))
            return super().move(ro)

    atoms = Atoms("H", positions=[[1.5, -0.4, 0.2]], pbc=False)
    atoms.calc = Harmonic()
    np.random.seed(1)
    hopper = Recording(
        atoms,
        temperature=10.0,
        optimizer=LBFGS,
        fmax=1.0e-8,
        dr=0.5,
        logfile=None,
        trajectory=None,
        optimizer_logfile=None,
        local_minima_trajectory=None,
        adjust_cm=False,
    )
    hopper.run(6)
    assert len(origins) == 6
    assert float(np.linalg.norm(origins[0])) < 1.0e-4
    assert max(float(np.linalg.norm(origin)) for origin in origins[1:]) > 0.05


def test_basin_hopping_accepts_the_lj75_cambridge_minimum():
    pytest.importorskip("ase")
    record = lennard_jones_minimum(75)
    result = search_one(
        {
            "index": 1,
            "label": "coords.1",
            "positions": record["positions"],
            "symbols": ["C"] * 75,
            "masses": None,
            "species": None,
        },
        problem_id="globalopt/lj75",
        method="basin",
        root=ROOT,
        hops=0,
    )
    assert result["success"]
    assert result["failure"] is None
    assert result["force_calls"] < 20
    assert result["template"] == "ase-globalopt/lj75-basin"
    assert result["energy"] == pytest.approx(-397.492331, abs=1.0e-3)


def test_minima_hopping_accepts_the_lj75_cambridge_minimum():
    pytest.importorskip("ase")
    record = lennard_jones_minimum(75)
    result = search_one(
        {
            "index": 1,
            "label": "coords.1",
            "positions": record["positions"],
            "symbols": ["C"] * 75,
            "masses": None,
            "species": None,
        },
        problem_id="globalopt/lj75",
        method="minima",
        root=ROOT,
        cap=50,
    )
    assert result["success"]
    assert result["failure"] is None
    assert result["force_calls"] < 20
    assert result["template"] == "ase-globalopt/lj75-minima"


def test_cap_stops_an_unrelaxed_lj75_start():
    pytest.importorskip("ase")
    problem = load_problem(ROOT / "problems" / "globalopt" / "lj75.toml")
    spec = _spec(ROOT, "globalopt/lj75", "basin")
    starts = {
        item["index"]: item
        for item in load_starts(geometry_archive(ROOT, problem), "globalopt/lj75", spec["atoms"])
    }
    result = search_one(starts[1], problem_id="globalopt/lj75", method="basin", root=ROOT, cap=40)
    assert result["failure"] == "cap"
    assert result["force_calls"] == 40
    assert result["success"] is False
