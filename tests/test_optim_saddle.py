"""OPTIM Pt-island search matches the problem potential and the full-row kick."""

import math
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import load_template
from optbench.connection import ProcessTable
from optbench.failures import classify_failure
from optbench.optim_saddle import (
    _record,
    climb_odata,
    climb_reasons,
    draw_island_kick,
    link_reasons,
    minimize_odata,
    rms_stop,
)
from optbench.pes import evaluate
from optbench.rgsaddle_minmode import _stored
from optbench.surfaces import potential_from_problem, read_con

ROOT = Path(__file__).resolve().parents[1]


def test_smooth_morse_matches_the_problem_and_the_finite_difference():
    text = _stored(ROOT, "website/saddle-search/pt-island/pt-island.con").read_text(encoding="utf-8")
    positions, _fixed, cell = read_con(text)
    problem = {
        "potential": {
            "kind": "morse",
            "D": 0.7102,
            "alpha": 1.6047,
            "r0": 2.897,
            "cutoff": 9.5,
            "pbc": [True, True, False],
        }
    }
    expected_energy, expected_gradient = potential_from_problem(problem)(positions, cell)
    energy, gradient = evaluate(
        "morse_smooth",
        positions,
        cell,
        pbc=(True, True, False),
        morse=(0.7102, 1.6047, 2.897, 9.5),
    )
    assert energy == expected_energy
    assert np.allclose(gradient, expected_gradient)
    trial = np.array(positions, copy=True)
    trial[10, 1] += 0.3
    step = 1e-5
    plus = np.array(trial, copy=True)
    minus = np.array(trial, copy=True)
    plus[10, 1] += step
    minus[10, 1] -= step
    plus_energy, _plus_gradient = evaluate("morse_smooth", plus, cell, pbc=(True, True, False))
    minus_energy, _minus_gradient = evaluate("morse_smooth", minus, cell, pbc=(True, True, False))
    _energy, gradient = evaluate("morse_smooth", trial, cell, pbc=(True, True, False))
    finite = (plus_energy - minus_energy) / (2 * step)
    assert abs(finite - gradient[10, 1]) < 1e-7


def test_a_relative_socket_hook_is_stored_as_an_absolute_path(tmp_path, monkeypatch):
    from optbench import optim_saddle as saddle

    work = Path("relative-socket-hook")
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_popen(cmd, **kwargs):
        seen["cwd"] = Path(kwargs["cwd"])
        seen["odata"] = (Path(kwargs["cwd"]) / "odata").read_text(encoding="utf-8")

        class Proc:
            pid = 1

            def poll(self):
                return 0

        return Proc()

    monkeypatch.setattr(saddle.subprocess, "Popen", fake_popen)
    spec = {
        "work": work / "climb",
        "positions": np.zeros((2, 3)),
        "reference": np.zeros((2, 3)),
        "fixed": np.array([True, False]),
        "cell": np.array([10.0, 10.0, 10.0]),
        "cap": 1,
        "method": "minimize",
        "vector": None,
        "gmax": 1e-4,
        "seed": 2014,
        "freeze": ["FREEZE 1"],
        "section": {},
        "pbc": (True, True, False),
        "morse": (0.7102, 1.6047, 2.897, 9.5),
        "python": "python",
        "binary": "optim",
        "repo": ".",
    }
    saddle._launch(spec)
    hook = tmp_path / "relative-socket-hook" / "climb" / "hook"
    assert seen["cwd"].is_absolute()
    assert str(hook) in seen["odata"]
    assert "relative-socket-hook/climb/relative-socket-hook" not in seen["odata"]


def test_relay_sends_the_negative_gradient():
    source = (ROOT / "optbench" / "engines" / "fmlip_relay.py").read_text(encoding="utf-8")
    assert "force = (-np.asarray(gradient, dtype=np.float64) / scale_l) * scale_e" in source
    assert "OPTBENCH_PBC" in source
    assert "TCP_NODELAY" in source


def test_one_atom_kick_matches_the_shared_draw():
    listed = [0, 1, 2, 3, 4, 5, 6]
    free = np.arange(7)
    minimum = np.zeros(21)
    left = np.random.default_rng(20261016)
    right = np.random.default_rng(20261016)
    point, kick, atom = draw_island_kick(left, listed, free, 0.1, minimum)
    expected_atom = int(listed[int(right.integers(0, len(listed)))])
    direction = right.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    assert atom == expected_atom
    assert np.allclose(kick.reshape(-1, 3)[atom], direction)
    assert np.allclose(point.reshape(-1, 3)[atom], 0.1 * direction)
    assert np.count_nonzero(np.linalg.norm(kick.reshape(-1, 3), axis=1)) == 1


def test_stock_odata_uses_the_problem_stop_and_one_freeze_line_per_atom():
    template = load_template(ROOT, "optim/saddle-hef-ffd-pt")
    section = template["ini"]["Optim"]
    fixed = np.array([True, False, True])
    freeze = [f"FREEZE {index}" for index, flag in enumerate(fixed, start=1) if flag]
    gmax = rms_stop(1e-3, 343)
    climb = climb_odata(python="python", hook=Path("/tmp/hook"), section=section, gmax=gmax, steps=1000, seed=2014, freeze=freeze)
    quench = minimize_odata(python="python", hook=Path("/tmp/hook"), gmax=gmax, steps=2000, seed=2014, freeze=freeze)
    assert "BFGSTS 3000 10 100" in climb
    assert "USEDIAG 2" in climb
    assert "STEPS 1000" in climb
    assert "BFGSSTEPS 1000" in climb
    assert "READVEC" in climb
    assert "DUMPVECTOR" in climb
    assert "FREEZE 1" in climb
    assert "FREEZE 3" in climb
    assert "FREEZE 2" not in climb
    assert "4.3895" not in climb
    assert "BFGSMIN" in quench
    assert "STEPS 2000" in quench
    assert "BFGSSTEPS 2000" in quench
    assert math.isclose(gmax, 1e-3 / math.sqrt(3 * 343))
    assert f"{gmax:.16E}".replace("E", "D") in climb


def test_cap_is_not_a_divergence_and_the_link_needs_one_endpoint():
    capped = climb_reasons(
        hang=False,
        error="exit 1",
        finite=False,
        capped=True,
        calls=1000,
        climb_cap=1000,
        force_norm=None,
        force_tol=1e-3,
        index=None,
    )
    assert classify_failure(capped) == "cap"
    assert classify_failure(["not_connected"]) == "divergence"
    assert classify_failure(["hessian_index"]) == "wrong_stationary"
    assert link_reasons(linked=True, novel=True, barrier=0.4, barrier_max=1.5) == []
    assert link_reasons(linked=False, novel=True, barrier=0.4, barrier_max=1.5) == ["not_connected"]
    assert link_reasons(linked=False, novel=False, barrier=0.4, barrier_max=1.5) == ["not_connected"]
    assert link_reasons(linked=True, novel=False, barrier=0.4, barrier_max=1.5) == ["duplicate"]


def _judged(name, saddle, plus, minus, energy):
    return {
        "job": {"name": name, "atom": 0},
        "climb": {"calls": 10},
        "coords": saddle,
        "energy": energy,
        "force": 1e-4,
        "reasons": None,
        "index": 1,
        "lowest": -0.2,
        "verify_calls": 0,
        "plus": {"calls": 4, "coords": plus},
        "minus": {"calls": 4, "coords": minus},
    }


def test_a_one_atom_hop_is_connected_and_a_second_copy_is_a_repeat():
    reactant = np.zeros((175, 3))
    product = reactant.copy()
    product[3, 0] = 2.77
    rmsd = float(np.sqrt(np.mean(np.sum((product - reactant) ** 2, axis=1))))
    assert rmsd < 0.5
    free = np.arange(175)
    table = ProcessTable(np.zeros(3), 0.1, 0.01)
    first = _record(
        _judged("rep1-search1", reactant, reactant, product, -1.4),
        reactant,
        free,
        table,
        1.5,
        -2.0,
        1e-3,
    )
    assert first["success"] is True
    assert first["reasons"] == []
    assert first["per_atom_max_minus"] == pytest.approx(2.77)
    table.add(first["saddle_x"], first["barrier"])
    second = _record(
        _judged("rep1-search2", reactant, product, reactant, -1.4),
        reactant,
        free,
        table,
        1.5,
        -2.0,
        1e-3,
    )
    assert second["success"] is False
    assert second["reasons"] == ["duplicate"]
    both = _record(
        _judged("rep1-search3", reactant, reactant, reactant, -1.4),
        reactant,
        free,
        table,
        1.5,
        -2.0,
        1e-3,
    )
    assert both["reasons"] == ["not_connected"]
