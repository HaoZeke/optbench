"""The 2014 OPTIM Morse gradient omits rho. The preambles are the archive files."""

from pathlib import Path

import numpy as np

from optbench.engines.fmlip_relay import scaled_force
from optbench.engines.optim_paper import R0, RHO, energy_gradient_calls
from optbench.pes import MORSE_D, morse_pt

ROOT = Path(__file__).resolve().parents[1]


def test_morse_bulk_gradient_is_the_true_reduced_gradient_over_rho():
    coords = np.array([[0.0, 0.0, 0.0], [2.2, 0.1, -0.2]], dtype=np.float64)
    energy, grad = morse_pt(coords, None)
    reduced = coords / R0
    dist = float(np.linalg.norm(reduced[0] - reduced[1]))
    amplitude = float(np.exp(RHO * (1.0 - dist)))
    shift = (1.0 - np.exp(RHO * (1.0 - 9.5 / R0))) ** 2 - 1.0
    fortran_energy = amplitude * (amplitude - 2.0) - shift
    assert abs(energy / MORSE_D - fortran_energy) < 1e-9
    direction = (reduced[0] - reduced[1]) / dist
    # morse_bulk.f90: xmul2 = 2*R*(R-1)/DIST, then V -= xmul2 * dx. rho is absent.
    pair = 2.0 * amplitude * (amplitude - 1.0)
    fortran = np.zeros_like(reduced)
    fortran[0] = -pair * direction
    fortran[1] = pair * direction
    force = scaled_force(grad, 1.0 / R0, 1.0 / MORSE_D, RHO)
    assert np.allclose(-force, fortran, rtol=1e-8, atol=1e-10)


def test_the_2014_awk_field_is_the_energy_gradient_count():
    text = "\n".join(
        [
            "OPTIM> # of energy calls=                          3 time=           0.01 %=  1.0",
            "OPTIM> # of energy+gradient calls=              131 time=           1.20 %= 90.0",
            "OPTIM> # of energy+gradient+Hessian calls=        0 time=           0.00 %=  0.0",
        ]
    )
    assert energy_gradient_calls(text) == 131


def test_archived_preambles_keep_the_paper_keywords():
    folder = ROOT / "templates" / "optim" / "preambles"
    rr = (folder / "rr-first.txt").read_text(encoding="utf-8")
    assert "USEDIAG 3" in rr
    assert "BFGSCONV 1D10" in rr
    assert "eigenonly 1D-13 0.99" in rr
    assert "updates 4 30" in rr
    ts = (folder / "ts-lj38.txt").read_text(encoding="utf-8")
    assert "bfgsts 40 14 16 0.33" in ts
    assert "updates 82 10" in ts
    morse = (folder / "morse-min.txt").read_text(encoding="utf-8")
    assert "updates 100" in morse
    assert "maxbfgs 0.690369347601" in morse
    assert "bfgsmin 5.08087747456D-05" in morse
