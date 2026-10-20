"""LAMMPS climbing-image NEB templates, call counts, and the spec potential."""

import os
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import geometry_archive, list_templates, load_template, resolve_problem
from optbench.failures import classify_outcome, classify_stored
from optbench.lammps_neb import (
    COMMIT,
    band_reasons,
    calls_from_blocks,
    cap_budget,
    climb_budget,
    ended_on_step_limit,
    iteration_budget,
    lammps_executable,
    minimization_blocks,
    probe_pair,
    render_neb_input,
    search_named,
)
from optbench.pes import evaluate, load_pairs

ROOT = Path(__file__).resolve().parents[1]
BINARY = lammps_executable()


def _settings(min_style: str, kind: str, images: int, cap: int, force_norm: float, max_move: float) -> dict:
    return {
        "potential": kind,
        "n_interior": images,
        "n_images": images + 2,
        "max_move": max_move,
        "force_norm": force_norm,
        "cap": cap,
        "min_style": min_style,
        "spring_k": 1.0,
        "parallel": "neigh",
        "timestep": 0.005 if kind == "lj" else 0.001,
        "units": "lj" if kind == "lj" else "metal",
    }


def test_registry_commit_matches_both_templates():
    registry = __import__("tomllib").loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "lammpsneb")
    assert row["commit"] == COMMIT
    assert row["name"] == "LAMMPS"
    assert row["ref"] == "stable_30Sep2026"
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "lammps-neb":
            continue
        found.append(template["id"])
        loaded = load_template(ROOT, template["id"])
        assert loaded["commit"] == row["commit"]
        assert loaded["code"] == "LAMMPS"
        assert loaded["code_version"] == "30 Sep 2026"
        assert loaded["tuning"] == "default"
        assert loaded["band"]["climbing"] is True
        assert loaded["band"]["spring_k"] == 1.0
        assert loaded["band"]["parallel"] == "neigh"
        assert "replicas" not in loaded["band"]
        assert "timestep_scale" not in loaded["band"]
    assert load_template(ROOT, "lammpsneb/ci-fire")["band"]["call_reserve"] == 0.2
    assert load_template(ROOT, "lammpsneb/ci-quickmin")["band"]["call_reserve"] == 0.0
    assert found == [
        "lammpsneb/ci-fire",
        "lammpsneb/ci-quickmin",
    ]


def test_iteration_budget_stays_under_each_problem_cap():
    expected = {
        "double-ended-saddle-search/lj38": (7, 2500),
        "double-ended-saddle-search/lj38_2ts_separation": (7, 10000),
        "double-ended-saddle-search/pt-island": (5, 3000),
        "neb/Pt-heptamer": (10, 10000),
    }
    for name, (images, cap) in expected.items():
        problem = resolve_problem(ROOT, name)
        assert int(problem["band"]["interior_images"]) + 2 == images
        assert int(problem["convergence"]["max_force_calls"]) == cap
        n1, n2 = iteration_budget(images, cap, 0.0)
        assert n1 >= 1 and n2 >= 1
        assert images * (n1 + n2 + 4) <= cap - 1
        fire_n1, fire_n2 = iteration_budget(images, cap, 0.2)
        assert fire_n1 + fire_n2 < n1 + n2
        assert images * (fire_n1 + fire_n2 + 4) <= int((cap - 1) * 0.8)


def test_climb_budget_splits_the_slack_total():
    expected = {
        (7, 2500, 0.2): ((140, 141), (93, 188)),
        (7, 10000, 0.2): ((569, 569), (379, 759)),
        (5, 3000, 0.2): ((237, 238), (158, 317)),
        (10, 10000, 0.2): ((397, 398), (265, 530)),
        (7, 2500, 0.0): ((176, 177), (117, 236)),
        (7, 10000, 0.0): ((712, 712), (474, 950)),
        (5, 3000, 0.0): ((297, 298), (198, 397)),
        (10, 10000, 0.0): ((497, 498), (331, 664)),
    }
    for (images, cap, reserve), (slack, climb) in expected.items():
        assert iteration_budget(images, cap, reserve) == slack
        assert climb_budget(images, cap, reserve) == climb
        assert sum(climb) == sum(slack)
        assert climb[0] == sum(slack) // 3


def test_cap_budget_reaches_the_problem_cap():
    for images, cap in ((5, 2500), (5, 3000), (5, 10000), (7, 2500), (10, 10000)):
        n1, n2 = cap_budget(images, cap)
        assert n1 >= 1 and n2 >= 1
        total = images * (n1 + n2 + 2)
        assert total <= cap
        assert total + images > cap


def test_step_limit_under_the_harness_cap_is_not_converged():
    limited = [
        [("force tolerance", 4), ("max iterations", 9)],
        [("force tolerance", 4), ("max iterations", 9)],
    ]
    assert ended_on_step_limit(limited)
    assert not ended_on_step_limit([[("max iterations", 4), ("force tolerance", 2)]])
    assert not ended_on_step_limit(
        [
            [("max iterations", 9)],
            [("force tolerance", 3)],
        ]
    )
    reasons = band_reasons(
        complete=True,
        calls=2485,
        cap=2500,
        force=0.016,
        force_norm=0.001,
        hessian_index=None,
    )
    assert reasons == ["force_norm"]
    assert (
        classify_outcome(
            energy=-1.0,
            force=0.016,
            calls=2485,
            cap=2500,
            force_tol=0.001,
            step_limit=True,
        )
        == "not_converged"
    )
    stored = classify_stored(
        {
            "success": False,
            "failure": "cap",
            "reasons": ["cap", "force_norm"],
            "energy": -1.0,
            "force_norm": 0.016,
            "force_calls": 2485,
            "lammps_stop": "max iterations",
            "step_limit": True,
        },
        2500,
        0.001,
    )
    assert stored == "not_converged"
    at_cap = band_reasons(
        complete=True,
        calls=2500,
        cap=2500,
        force=0.016,
        force_norm=0.001,
        hessian_index=None,
    )
    assert at_cap == ["cap", "force_norm"]
    assert (
        classify_outcome(
            energy=-1.0,
            force=0.016,
            calls=2500,
            cap=2500,
            force_tol=0.001,
            step_limit=True,
        )
        == "cap"
    )
    early = band_reasons(
        complete=True,
        calls=400,
        cap=2500,
        force=0.016,
        force_norm=0.001,
        hessian_index=None,
    )
    assert early == ["force_norm"]
    assert (
        classify_outcome(energy=-1.0, force=0.016, calls=400, cap=2500, force_tol=0.001, step_limit=False)
        == "not_converged"
    )
    saddle = band_reasons(
        complete=True,
        calls=2485,
        cap=2500,
        force=7.3e-4,
        force_norm=0.001,
        hessian_index=1,
    )
    assert saddle == []
    assert (
        classify_outcome(
            energy=-1.0,
            force=7.3e-4,
            calls=2485,
            cap=2500,
            force_tol=0.001,
            hessian_index=1,
            expected_index=1,
            step_limit=True,
        )
        is None
    )
    wrong = band_reasons(
        complete=True,
        calls=2485,
        cap=2500,
        force=1.0e-7,
        force_norm=0.001,
        hessian_index=0,
    )
    assert wrong == ["hessian_index"]
    assert (
        classify_outcome(
            energy=-1.0,
            force=1.0e-7,
            calls=2485,
            cap=2500,
            force_tol=0.001,
            hessian_index=0,
            expected_index=1,
            step_limit=True,
        )
        == "wrong_stationary"
    )


def test_timestep_scale_keeps_the_problem_image_count():
    settings = _settings("fire", "morse_pt", 8, 10000, 0.01, 0.2)
    settings["timestep_scale"] = 4.0
    settings["timestep"] = 0.004
    script = render_neb_input(settings, n1=10, n2=20, frozen=True)
    assert "timestep 0.004" in script
    assert "partition yes 10 fix hold neb setforce 0.0 0.0 0.0" in script
    assert "neb 0.0 0.01 10 20 1 final final.neb" in script
    assert "timestep" not in render_neb_input(_settings("quickmin", "lj", 5, 2500, 0.001, 0.05), n1=1, n2=1, frozen=False)


def test_minimization_stats_count_setup_and_the_early_extra():
    text = "\n".join(
        [
            "Minimization stats:",
            "  Stopping criterion = max iterations",
            "  Iterations, force evaluations = 20 21",
            "Minimization stats:",
            "  Stopping criterion = force tolerance",
            "  Iterations, force evaluations = 3 3",
        ]
    )
    blocks = minimization_blocks(text)
    assert blocks == [("max iterations", 21), ("force tolerance", 3)]
    assert calls_from_blocks(blocks) == (1 + 21) + (1 + 3 + 1)


def test_input_uses_the_spec_pair_style_and_holds_the_endpoints():
    quick = render_neb_input(_settings("quickmin", "lj", 5, 2500, 0.001, 0.05), n1=176, n2=177, frozen=False)
    assert quick.index("neb ") < quick.index("write_dump")
    assert "pair_style lj/cut 40.0" in quick
    assert "pair_modify shift no" in quick
    assert "neighbor 0.3 bin" in quick
    assert "boundary f f f" in quick
    assert "min_style quickmin" in quick
    assert "min_modify dmax 0.05 norm two" in quick
    assert "fix neb1 neb neb 1 parallel neigh" in quick
    assert "partition yes 1 fix hold neb setforce 0.0 0.0 0.0" in quick
    assert "partition yes 7 fix hold neb setforce 0.0 0.0 0.0" in quick
    assert "neb 0.0 0.001 176 177 1 final final.neb" in quick
    assert "timestep" not in quick
    fire = render_neb_input(_settings("fire", "morse_pt", 8, 10000, 0.01, 0.2), n1=497, n2=498, frozen=True)
    assert "pair_style morse 9.5" in fire
    assert "pair_coeff * * 0.7102 1.6047 2.897" in fire
    assert "pair_modify shift yes" in fire
    assert "neighbor 0.0 multi" in fire
    assert "comm_modify mode multi" in fire
    assert "boundary p p p" in fire
    assert "min_style fire" in fire
    assert "fix freeze frozen setforce 0.0 0.0 0.0" in fire
    assert "partition yes 10 fix hold neb setforce" in fire


def test_board_index_names_the_lammps_classes():
    from optbench.board import _benchmark_index

    index = _benchmark_index()
    quick = index["bench_lammpsneb.HeptamerLammpsnebCiQuickmin"]
    fire = index["bench_lammpsneb.Lj38DeLammpsnebCiFire"]
    assert quick == {"template": "lammpsneb/ci-quickmin", "problem": "neb/Pt-heptamer", "cap": 10000}
    assert "bench_lammpsneb.Lj38DeLammpsnebCiFireTuned" not in index
    assert fire["template"] == "lammpsneb/ci-fire"
    assert fire["problem"] == "double-ended-saddle-search/lj38"
    assert fire["cap"] == 2500


def test_default_asv_grid_is_one_start():
    from benchmarks.bench_lammpsneb import HeptamerLammpsnebCiFire, Lj38DeLammpsnebCiQuickmin

    full = os.environ.get("OPTBENCH_ASV_FULL") == "1"
    assert len(HeptamerLammpsnebCiFire.params[1]) == (13 if full else 1)
    assert len(Lj38DeLammpsnebCiQuickmin.params[1]) == (50 if full else 1)
    assert HeptamerLammpsnebCiFire.param_names == ["seed", "start"]
    assert HeptamerLammpsnebCiFire.track_force_calls.unit == "force calls"


@pytest.mark.skipif(BINARY is None, reason="LAMMPS binary is not available")
def test_lammps_pair_matches_the_harness_on_lj_and_morse():
    lj = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    pair = load_pairs(geometry_archive(ROOT, lj), lj["geometry"]["kind"], 50)[0]
    energy, grad = probe_pair("lj", pair.reactant, None, binary=BINARY)
    harness_energy, harness_grad = evaluate("lj", pair.reactant, None)
    assert energy == pytest.approx(harness_energy, abs=1e-8)
    assert np.allclose(grad, harness_grad, atol=1e-8)

    hept = resolve_problem(ROOT, "neb/Pt-heptamer")
    slab = load_pairs(geometry_archive(ROOT, hept), hept["geometry"]["kind"], 13)[0]
    moved = np.array(slab.reactant, copy=True)
    moved[slab.free][0, 0] += 0.2
    morse_energy, morse_grad = probe_pair("morse_pt", moved, slab.box, binary=BINARY)
    harness_energy, harness_grad = evaluate("morse_pt", moved, slab.box)
    assert morse_energy == pytest.approx(harness_energy, abs=1e-8)
    assert np.allclose(morse_grad, harness_grad, atol=1e-8)


@pytest.mark.skipif(BINARY is None, reason="LAMMPS binary is not available")
def test_short_quickmin_band_counts_pair_computes_and_holds_the_reactant(tmp_path: Path):
    record = search_named(
        ROOT,
        "double-ended-saddle-search/lj38",
        "lammpsneb/ci-quickmin",
        "pair_00",
        n1=2,
        n2=2,
        work=tmp_path,
    )
    assert record["lammps_stop"] == "max iterations"
    assert record["step_limit"] is True
    assert record["force_calls"] < 2500
    assert "cap" not in record["reasons"]
    assert record["failure"] == "not_converged"
    assert record["success"] is False
    assert record["n_images"] == 7
    logs = [tmp_path.joinpath(f"log.lammps.{index}").read_text(encoding="utf-8") for index in range(7)]
    assert record["force_calls"] == sum(calls_from_blocks(minimization_blocks(text)) for text in logs)
    assert record["force_calls"] == 7 * (2 + 2 + 2)
    from optbench.lammps_neb import _coords_from_dump

    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    pair = load_pairs(geometry_archive(ROOT, problem), problem["geometry"]["kind"], 50)[0]
    held = _coords_from_dump(tmp_path / "final.0.dump", pair.reactant.shape[0])
    assert np.allclose(held, pair.reactant, atol=1e-8)
