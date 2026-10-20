"""Pt-island OPTIM uses central-difference hybrid EF and the problem cap."""

from pathlib import Path

import numpy as np

import math

from optbench.engines.optim_hef import SVNREV, TARBALL_SHA256, log_counters, odata_text, rms_stop

ROOT = Path(__file__).resolve().parents[1]


def test_odata_uses_central_differences_and_the_problem_cap():
    frozen = np.zeros(16, dtype=bool)
    frozen[:3] = True
    text = odata_text("/usr/bin/python", Path("/tmp/hook"), True, 1.0e-5, 10000, frozen)
    assert "\nBFGSTS 10000 10 100 1.0D-10 10000\n" in text
    assert "bfgsts 30" not in text.lower()
    assert "0.063" not in text
    assert "USEDIAG 2" in text
    assert "NOHESS" in text
    assert "BFGSCONV 1.0000000000000001D-05" in text
    assert "BFGSSTEPS 10000" in text
    assert "STEPS 10000" in text
    assert "READVEC" in text
    assert "FREEZE 1 2 3" in text
    assert "PARAMS" not in text
    assert SVNREV.startswith("df8da599")
    assert len(TARBALL_SHA256) == 64
    template = (ROOT / "templates/optim/hef-ts-pt.toml").read_text(encoding="utf-8")
    assert 'method = "hef"' in template
    assert "df8da599" in template
    assert TARBALL_SHA256 in template


def test_bfgsconv_counts_free_atoms_only():
    # Pt island: 343 atoms, 168 frozen, 175 free. OPTIM RMSDOF = 3*175.
    free_rms = rms_stop(0.001, 175)
    all_atom_rms = 0.001 / math.sqrt(3 * 343)
    assert free_rms == 0.001 / math.sqrt(3 * 175)
    assert free_rms > all_atom_rms
    assert abs(0.001 * math.sqrt(175 / 343) - 0.0007142857142857143) < 1e-15


def test_harness_count_is_the_energy_gradient_tally_not_rayleigh_ritz_steps():
    log = (
        " xmylbfgs> Eigenvalue and RMS=  1.0  0.1 after      0 steps\n"
        " xmylbfgs> Eigenvalue, RMS, % change=  1.0  0.1  1.0 after      1 steps, step: 0.1\n"
        " xmylbfgs> Eigenvalue, RMS, % changes=  1.0  0.1  1.0  1.0 after      2 steps, step: 0.1\n"
        " xmylbfgs> Smallest eigenvalue converged in    2 steps.\n"
        "OPTIM> # of energy+gradient calls=               52 time=           0.06 %= 93.4\n"
    )
    assert log_counters(log) == (52, 2)
    assert log_counters("no calls") == (None, 0)
