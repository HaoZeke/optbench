"""saddle-search/pt-island uses eOn Morse::force, not smooth-linear."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_eon_svn import _morse_fast
from optbench.ase_sella import _morse_arrays
from optbench.campaign import load_template, resolve_problem
from optbench.engines.geometries import load_starts
from optbench.pes import morse_pt
from optbench.surfaces import morse_eon, morse_smooth_linear, potential_from_problem

ROOT = Path(__file__).resolve().parents[1]


def _island():
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    start = load_starts(ROOT, problem, Path("/tmp/morse-eon-test"))[0]
    positions = np.asarray(start["positions"], dtype=np.float64)
    cell = np.asarray(start["box"], dtype=np.float64)
    return problem, positions, cell


def _reference(positions, cell):
    box = np.zeros(9, dtype=np.float64)
    box[0], box[4], box[8] = cell
    energy, forces = _morse_fast(positions.reshape(-1), box=box)
    return float(energy), np.asarray(forces, dtype=np.float64).reshape(-1, 3)


def test_the_saddle_problem_matches_morse_force_and_smooth_does_not():
    problem, positions, cell = _island()
    assert problem["potential"]["pair_style"] == "eon-morse"
    reference_energy, reference_force = _reference(positions, cell)
    energy, gradient = potential_from_problem(problem)(positions, cell)
    direct_energy, direct_gradient = morse_eon(
        positions, depth=0.7102, alpha=1.6047, r0=2.897, cutoff=9.5, cell=cell
    )
    assert energy == direct_energy
    assert np.array_equal(gradient, direct_gradient)
    assert abs(energy - reference_energy) < 1e-8
    assert np.max(np.abs(-gradient - reference_force)) < 1e-8

    smooth_energy, smooth_gradient = morse_smooth_linear(
        positions,
        depth=0.7102,
        alpha=1.6047,
        r0=2.897,
        cutoff=9.5,
        cell=cell,
        pbc=(True, True, False),
    )
    assert smooth_energy - reference_energy == pytest.approx(4.43284153, abs=1e-6)
    assert np.max(np.abs(-smooth_gradient - reference_force)) == pytest.approx(3.28533419e-3, abs=1e-8)


def test_pes_and_cutoff_9_5_match_and_cutoff_7_does_not():
    _problem, positions, cell = _island()
    reference_energy, reference_force = _reference(positions, cell)
    pes_energy, pes_gradient = morse_pt(positions, cell)
    assert abs(pes_energy - reference_energy) < 1e-8
    assert np.max(np.abs(-pes_gradient - reference_force)) < 1e-8
    matrix = np.diag(cell)
    shifted_energy, shifted_force = _morse_arrays(
        positions, matrix, [True, True, False], 0.7102, 1.6047, 2.897, 9.5, "shift"
    )
    assert abs(shifted_energy - reference_energy) < 1e-8
    assert np.max(np.abs(shifted_force - reference_force)) < 1e-8
    short_energy, short_force = _morse_arrays(
        positions, matrix, [True, True, False], 0.7102, 1.6047, 2.897, 7.0, "shift"
    )
    assert abs(short_energy - reference_energy - 32.5911541) < 1e-5
    assert abs(np.max(np.abs(short_force - reference_force)) - 3.17271076e-2) < 1e-6


def test_saddle_templates_name_the_eon_kernel():
    for name in (
        "eon-client/saddle-pt-dimer",
        "eon-client/saddle-pt-lanczos",
        "eon-client/saddle-pt-md",
        "eon-svn/saddle-pt-dimer-bounded",
        "eon-svn/saddle-pt-lanczos-bounded",
        "eon-svn/saddle-pt-md",
    ):
        assert load_template(ROOT, name)["kernel"] == "morse_shift"
    assert load_template(ROOT, "eon/develop-saddle-pt-dimer")["ini"]["Potential"]["potential"] == "morse_pt"
    assert load_template(ROOT, "eon/develop-saddle-pt-lanczos")["ini"]["Potential"]["potential"] == "morse_eon"
    for name in (
        "sella/pt-island-search-default",
        "sella/pt-island-search-tuned",
        "ase/dimer-pt-island-default",
        "ase/dimer-pt-island-tuned",
    ):
        potential = load_template(ROOT, name)["potential"]
        assert potential["rc"] == 9.5
        assert potential["form"] == "shift"
    tsopt = resolve_problem(ROOT, "tsopt/pt-island")
    assert tsopt["potential"]["pair_style"] == "morse/smooth/linear"
