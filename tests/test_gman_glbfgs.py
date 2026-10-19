"""GL-BFGS Hessian CI-NEB against the archived Pt-heptamer counting rule."""

import json
from pathlib import Path

import numpy as np

from optbench.campaign import geometry_archive, load_template, resolve_problem
from optbench.gman_glbfgs import (
    DEFAULT_OUTPUT,
    INVCURV,
    MAXMOVE,
    MEMORY,
    lbfgs_step,
    paper_report,
    run_pair,
)
from optbench.pes import load_pairs
from optbench.reproduction import within_mean

ROOT = Path(__file__).resolve().parents[1]


def test_the_first_step_caps_the_global_force():
    force = np.ones(6)
    state = {"fold": np.zeros(6), "rold": None, "s": [], "y": [], "rho": []}
    updated = lbfgs_step(np.zeros(6), force, state, memory=MEMORY, invcurv=INVCURV, maxmove=MAXMOVE, n_images=2)
    assert np.isclose(np.linalg.norm(updated), MAXMOVE * np.sqrt(2.0))


def test_paper_truncation_matches_opt_py():
    report = paper_report([210, 434], 8)
    assert report["force_calls"] == int((210 / 8 + 434 / 8) / 2)
    assert report["force_calls_min"] == 26
    assert report["force_calls_max"] == 54


def test_campaign_dat_is_the_truncated_mean_of_the_starts():
    dat = (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8")
    starts = [
        json.loads(line)
        for line in DEFAULT_OUTPUT.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(starts) == 13
    assert all(row["success"] for row in starts)
    report = paper_report([int(row["force_calls"]) for row in starts], 8)
    assert f"force_calls {report['force_calls']}\n" in dat
    assert report["force_calls"] == 44
    assert report["force_calls_min"] == 26
    assert report["force_calls_max"] == 54
    assert within_mean(48.0, float(report["force_calls"]), None)
    template = load_template(ROOT, "gman/neb-pt-glbfgs-hess")
    assert template["commit"] == "vtst-lbfgs-r128"
    assert template["code"] == "Gman"


def test_min01_rerun_matches_the_stored_start():
    problem = resolve_problem(ROOT, "neb/Pt-heptamer")
    pairs = load_pairs(geometry_archive(ROOT, problem), "con-minima-pairs", 13)
    row = run_pair(next(pair for pair in pairs if pair.name == "min01"))
    stored = json.loads(DEFAULT_OUTPUT.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert stored["start"] == "min01"
    assert row["calls"] == stored["force_calls"] == 210
    assert row["converged"] is True
    assert row["force_norm"] < 0.01
