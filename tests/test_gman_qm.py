"""Gman r235 QuickMin on the Pt-heptamer band matches the 2013 archive integers."""

import tempfile
from pathlib import Path

import numpy as np

from optbench.gman_qm import campaign_text, collect, loadcon, morse_force, unpack_minima
from optbench.pes import morse_pt

ROOT = Path(__file__).resolve().parents[1]


def test_morse_matches_the_harness_on_min01():
    with tempfile.TemporaryDirectory() as tmp:
        minima = unpack_minima(ROOT, Path(tmp))
        point = loadcon(minima / "min01" / "a.con")
    count = [0]
    morse_force(point, count, False)
    energy, gradient = morse_pt(point.R, np.diag(point.Box))
    assert count[0] == 0
    assert abs(energy - point.U) < 1e-10
    free = point.FreeD[:, 0] > 0
    assert free.sum() == 175
    assert float(np.max(np.abs(point.F[free] + gradient[free]))) < 1e-10


def test_quickmin_ci_neb_writes_the_archive_integers():
    calls = collect(ROOT)
    text = campaign_text(calls)
    stored = (ROOT / "results" / "campaign" / "gman-r235-neb-pt-qm.dat").read_text(encoding="utf-8")
    assert text == stored
    fields = dict(line.split(maxsplit=1) for line in text.splitlines())
    assert fields["force_calls"] == "190"
    assert fields["force_calls_max"] == "316"
    assert fields["force_calls_min"] == "103"
    assert fields["nfailed"] == "0"
    assert fields["code"] == "Gman"
    assert fields["code_version"] == "r235"
