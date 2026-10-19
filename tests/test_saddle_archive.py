"""Published Pt-island saddle counts.

The 2014 eOn rows store ``done.py`` output. The nested run tarballs
hold ``benchmark.dat``, ``done.py``, ``config.ini``, ``pos.con``, and
``run.sh``. They do not hold ``search_results.txt`` or endpoint
minima, so ``Matter::compare`` cannot be reapplied to those searches.
``calc_benchmarks.py`` is the per-run list that averages to the
published row.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import pytest

from optbench.dat import parse_lenient

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "entries" / "saddle-search" / "pt-island"


def _assignments(path: Path) -> dict[str, list[float]]:
    text = path.read_text(encoding="utf-8")
    found: dict[str, list[float]] = {}
    for key in ("force_calls", "unique_saddles", "fraction_connected", "total_searches"):
        match = re.search(rf"^{key} = \[([^\]]+)\]", text, re.M)
        assert match is not None, key
        found[key] = [float(part) for part in match.group(1).split(",")]
    return found


def _fields(path: Path) -> dict[str, str]:
    fields, _duplicates = parse_lenient(path.read_text(encoding="utf-8"))
    return fields


def test_done_py_counts_good_and_not_connected_only():
    text = (ARCHIVE / "eon-dimer" / "done.py").read_text(encoding="utf-8")
    assert "if 'good' in result: total_found += 1" in text
    assert "if 'Connected' in result: not_connected += 1" in text
    assert "(float(total_searches-not_connected)/float(total_searches))" in text
    assert "barrier < cutoff" in text
    assert "ntofind = 27" in text
    assert "cutoff = 1.5" in text
    assert "print 'unique_saddles %i' % total_found" in text


def test_published_eon_r2205_and_softsaddle_rows():
    dimer = _fields(ARCHIVE / "eon-dimer" / "benchmark.dat")
    assert dimer["code_version"] == "r2205"
    assert dimer["unique_saddles"] == "109.222"
    assert dimer["fraction_connected"] == "0.746017"
    assert dimer["total_searches"] == "11366.8"
    assert dimer["force_calls"] == "3.77888e+06"
    assert dimer["unique_saddles"] != "27"
    md = _fields(ARCHIVE / "eon-md" / "benchmark.dat")
    assert md["unique_saddles"] == "51.6667"
    assert md["fraction_connected"] == "0.886966"
    assert md["total_searches"] == "800.222"
    soft = {
        "adaptive_HSphere-lbfgs-davidson-sm50": ("114.9", "0.8", "1737.5"),
        "adaptive_HSphere-lbfgs-lanczos-sm0": ("130.4", "0.9", "2752.5"),
        "adaptive_HSphere-lbfgs-lanczos-sm50": ("142.6", "0.9", "3072.5"),
    }
    for name, (unique, fraction, searches) in soft.items():
        fields = _fields(ARCHIVE / name / "benchmark.dat")
        assert fields["unique_saddles"] == unique
        assert fields["fraction_connected"] == fraction
        assert fields["total_searches"] == searches


def test_r2213_per_run_lists_average_to_the_published_row():
    directories = sorted(path for path in ARCHIVE.iterdir() if (path / "calc_benchmarks.py").is_file())
    assert len(directories) == 7
    for directory in directories:
        listed = _assignments(directory / "calc_benchmarks.py")
        published = _fields(directory / "benchmark.dat")
        assert len(listed["unique_saddles"]) == 9
        for key in ("force_calls", "unique_saddles", "fraction_connected", "total_searches"):
            assert statistics.fmean(listed[key]) == pytest.approx(float(published[key]))

