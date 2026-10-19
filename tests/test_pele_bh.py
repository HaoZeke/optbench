"""Archived pele basin hopping for Lennard-Jones 38."""

import tarfile
from pathlib import Path

import pytest

from optbench.campaign import load_template, resolve_problem, run
from optbench.runners.pele_bh import (
    _LINESEARCH_CALLS,
    _LINESEARCH_ENERGY,
    _options,
    _require_a68ec5_linesearch,
    _rule,
)

ROOT = Path(__file__).resolve().parents[1]


def test_template_uses_the_archived_driver():
    template = load_template(ROOT, "pele/lj38-basinhopping")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    options = _options(template)
    minimum, tolerance, partial = _rule(problem, template)
    assert template["tuning"] == "archive"
    assert template["commit"] == "b7081b932a060be11342fbe8287ac35da8ada879"
    assert options == {
        "temperature": 1.0,
        "stepsize": 0.41,
        "quench_tol": 1e-3,
        "quench_maxstep": 0.1,
        "quench_m": 4,
        "quench_nsteps": 10000,
        "acc_ratio": 0.5,
        "factor": 0.9,
        "interval": 100,
        "takestep": "adaptive",
    }
    assert minimum == -173.9284
    assert tolerance == 0.001
    assert partial == ""
    assert template["stop"]["max_force_calls"] == 4000000
    patch = (ROOT / "codes" / "pele-a68ec5-lbfgs-linesearch.patch").read_text(encoding="utf-8")
    assert "factor /= 10." in patch
    assert "factor *= 0.5" in patch


def test_campaign_dispatches_pele(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, *, geometries, output, workers, work, limit, resume):
        captured["problem"] = problem["id"]
        captured["template"] = template["id"]
        captured["workers"] = workers
        captured["resume"] = resume
        captured["output"] = output
        return {"starts": 0}

    monkeypatch.setattr("optbench.runners.pele_bh.run_template", fake)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "globalopt/lj38",
        "pele/lj38-basinhopping",
        output=output,
        workers=3,
        resume=True,
    )
    assert summary == {"starts": 0}
    assert captured["problem"] == "globalopt/lj38"
    assert captured["template"] == "pele/lj38-basinhopping"
    assert captured["workers"] == 3
    assert captured["resume"] is True
    assert captured["output"] == output


def _start_one():
    archive = (
        ROOT
        / "inputs"
        / "sha256"
        / "9e5bb818adfad67a5afd53d083a6c8afd3addc0fe81a0f57413a73c80eb74e8f"
        / "100_lj38.tar.gz"
    )
    if not archive.is_file():
        return None
    import numpy as np

    with tarfile.open(archive) as packed:
        member = packed.extractfile("100_lj38/1.xyz")
        if member is None:
            return None
        text = member.read().decode("utf-8")
    rows = []
    for line in text.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 4:
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(rows) != 38:
        return None
    return np.asarray(rows, dtype=np.float64).reshape(-1)


def test_a68ec5_linesearch_quench_is_327_calls():
    pytest.importorskip("pele")
    coords = _start_one()
    if coords is None:
        pytest.skip("LJ38 geometry archive is not in the checkout")
    _require_a68ec5_linesearch(coords)
    assert _LINESEARCH_CALLS == 327
    assert _LINESEARCH_ENERGY == -163.20543198540588
