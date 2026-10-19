"""Published Pt-island saddle counts, and the stored pARTn endpoints.

The 2014 eOn rows store ``done.py`` output. The nested run tarballs
hold ``benchmark.dat``, ``done.py``, ``config.ini``, ``pos.con``, and
``run.sh``. They do not hold ``search_results.txt`` or endpoint
minima, so ``Matter::compare`` cannot be reapplied to those searches.
``calc_benchmarks.py`` is the per-run list that averages to the
published row.

The pARTn campaign stored both endpoint minima. Cartesian RMSD 0.5
over the 175 free atoms merges distinct products. The rerun applies
``find_repeat`` to the saddle. A product rejudge of the same archive
agrees: 48 good saddles, 18 inside the barrier window, fraction 0.755.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import numpy as np
import pytest

from optbench.connection import rescore_endpoints
from optbench.dat import parse_lenient
from optbench.endpoints import load_endpoints
from optbench.score import score_v2

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "entries" / "saddle-search" / "pt-island"
PARTN = "artn-1.1.0-saddle-search-pt-island--partn-saddle-pt"


def _partn(suffix: str) -> Path:
    # The campaign stem contains dots, so Path.with_suffix would cut at 1.1.
    return ROOT / "results" / "campaign" / f"{PARTN}{suffix}"


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


def test_partn_endpoints_match_find_repeat_not_rmsd():
    records = [json.loads(line) for line in _partn(".starts.jsonl").read_text().splitlines() if line.strip()]
    for record in records:
        record["reasons"] = list(record.get("reasons") or [])
    reactant, plus, minus, cell, _free_index = load_endpoints(_partn(".endpoints.npz"))
    catalog = rescore_endpoints(records, reactant, plus, minus, cell)
    assert catalog.searches == 200
    assert catalog.not_connected == 49
    assert catalog.unique == 48
    assert catalog.qualifying == 18
    assert catalog.fraction_connected == 151 / 200
    assert sum(bool(record["success"]) for record in records) == 18
    dat = _partn(".dat").read_text(encoding="utf-8")
    assert "unique_saddles 48" in dat
    assert "fraction_connected 0.755" in dat
    assert "connected 18" in dat
    assert "force_calls 154413" in dat
    assert "par2_mean 10980.2" in dat
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in records
        ],
        6000,
    )
    assert metrics["successes"] == 18
    assert metrics["nfailed"] == 182
    assert metrics["n_crash"] == 13
    assert metrics["n_divergence"] == 169
    assert f"par2_mean {metrics['par2_mean']}" in dat

    def rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
        delta = moved - reference
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))

    clusters: list[np.ndarray] = []
    linked_low = 0
    for index, record in enumerate(records):
        if not record.get("barrier") or not (0.0 < float(record["barrier"]) < 1.5):
            continue
        row_plus = plus[index]
        row_minus = minus[index]
        if not (np.isfinite(row_plus).all() and np.isfinite(row_minus).all()):
            continue
        row_reactant = reactant[index]
        # The nearer endpoint is the reactant under the r2205 test for a linked search.
        product = row_plus if rmsd(row_minus, row_reactant) < rmsd(row_plus, row_reactant) else row_minus
        if "not_connected" in record["reasons"]:
            continue
        linked_low += 1
        if all(rmsd(product, old) > 0.5 for old in clusters):
            clusters.append(product)
    assert linked_low == 65
    assert len(clusters) == 2
