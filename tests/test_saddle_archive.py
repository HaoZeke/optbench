"""Published Pt-island saddle counts, and the stored pARTn endpoints.

The 2014 eOn rows store ``done.py`` output. The nested run tarballs
hold ``benchmark.dat``, ``done.py``, ``config.ini``, ``pos.con``, and
``run.sh``. They do not hold ``search_results.txt`` or endpoint
minima, so ``Matter::compare`` cannot be reapplied to those searches.
``calc_benchmarks.py`` is the per-run list that averages to the
published row.

The pARTn campaign stored both endpoint minima. Cartesian RMSD 0.5
over the 175 free atoms merges distinct products. ``find_repeat`` on
the product minimum does not.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import numpy as np
import pytest

from optbench.check import load_oracle
from optbench.connection import rescore_endpoints
from optbench.dat import parse_lenient
from optbench.endpoints import load_endpoints
from optbench.saddle_pt import DEVELOP_LABEL, SVN_LABEL, comparison_rows, saddle_pt_html
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


def test_pt_island_archive_has_no_develop_row():
    versions = set()
    for path in ARCHIVE.glob("*/benchmark.dat"):
        versions.add(_fields(path).get("code_version", ""))
    assert versions == {"", "r2205", "r2213"}


def _comment_reps(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# rep "):
            continue
        parts = line.split()

        def grab(key: str) -> float:
            return float(parts[parts.index(key) + 1])

        rows.append(
            {
                "force_calls": grab("force_calls"),
                "unique_saddles": grab("unique_saddles"),
                "fraction_connected": grab("fraction_connected"),
                "total_searches": grab("total_searches"),
            }
        )
    return rows


def test_r2549_paper_rerun_matches_report_sh_mean():
    campaign = ROOT / "results" / "campaign"
    expected = {
        "eon-svn-r2549-saddle-pt-paper-dimer.dat": {
            "fraction_connected_9": "0.7531125773",
            "total_searches_9": "13823.33333",
            "unique_saddles_9": "117.2222222",
            "force_calls_9": "4933128",
            "force_calls": "5075453.5",
            "fraction_connected": "0.753475785",
            "total_searches": "14227.6",
            "unique_saddles": "118.2",
        },
        "eon-svn-r2549-saddle-pt-paper-md.dat": {
            "fraction_connected_9": "0.8888963112",
            "total_searches_9": "834.2222222",
            "unique_saddles_9": "51.66666667",
            "force_calls_9": "1530551.222",
            "force_calls": "1550616.8",
            "fraction_connected": "0.890407524",
            "total_searches": "845.6",
            "unique_saddles": "51.7",
        },
    }
    for name, locked in expected.items():
        path = campaign / name
        assert not path.with_suffix(".starts").exists()
        fields = _fields(path)
        assert fields["code"] == "EON (SVN r2549)"
        assert fields["code_version"] == "r2549"
        assert fields["problem"] == "saddle-search/pt-island"
        assert fields["reducer"] == "akmc-until-27"
        assert fields["report_runs"] == "9"
        assert fields["nfailed"] == "0"
        assert fields["repetitions"] == "10"
        reps = _comment_reps(path)
        assert len(reps) == 10
        for key, value in locked.items():
            assert fields[key] == value
        sample = reps[:9]
        assert statistics.fmean(row["fraction_connected"] for row in sample) == pytest.approx(
            float(fields["fraction_connected_9"])
        )
        assert statistics.fmean(row["total_searches"] for row in sample) == pytest.approx(
            float(fields["total_searches_9"])
        )


def test_develop_500350d9_dimer_matches_report_sh_mean():
    path = ROOT / "results" / "campaign" / "eon-develop-500350d9-saddle-pt-dimer.dat"
    assert not path.with_suffix(".starts").exists()
    fields = _fields(path)
    assert fields["code"] == "eOn (develop 500350d9)"
    assert fields["code_version"] == "500350d9"
    assert fields["commit"] == "500350d9fae881f06c7ef194e5868da079c4be5a"
    assert fields["problem"] == "saddle-search/pt-island"
    assert fields["reducer"] == "akmc-until-27"
    assert fields["algorithm"] == "Dimer"
    assert fields["report_runs"] == "9"
    assert fields["nfailed"] == "0"
    assert fields["repetitions"] == "10"
    locked = {
        "fraction_connected_9": "0.7519646115",
        "total_searches_9": "21601.11111",
        "unique_saddles_9": "78.88888889",
        "force_calls_9": "5682183.778",
        "force_calls_mean_error_9": "822578.5012",
        "force_calls": "5426864.8",
        "fraction_connected": "0.7522965245",
        "total_searches": "20642.8",
        "unique_saddles": "78.4",
        "force_calls_mean_error": "778937.5256",
    }
    for key, value in locked.items():
        assert fields[key] == value
    reps = _comment_reps(path)
    assert len(reps) == 10
    numbers = [
        int(line.split()[2])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# rep ")
    ]
    assert numbers == list(range(1, 11))
    sample = reps[:9]
    assert statistics.fmean(row["fraction_connected"] for row in sample) == pytest.approx(
        float(fields["fraction_connected_9"])
    )
    assert statistics.fmean(row["force_calls"] for row in sample) == pytest.approx(float(fields["force_calls_9"]))
    assert statistics.fmean(row["force_calls"] for row in reps) == pytest.approx(float(fields["force_calls"]))


def test_develop_500350d9_dynamics_matches_report_sh_mean():
    path = ROOT / "results" / "campaign" / "eon-develop-500350d9-saddle-pt-md.dat"
    assert not path.with_suffix(".starts").exists()
    fields = _fields(path)
    assert fields["code"] == "eOn (develop 500350d9)"
    assert fields["code_version"] == "500350d9"
    assert fields["commit"] == "500350d9fae881f06c7ef194e5868da079c4be5a"
    assert fields["problem"] == "saddle-search/pt-island"
    assert fields["reducer"] == "akmc-until-27"
    assert fields["algorithm"] == "Dynamics"
    assert fields["report_runs"] == "9"
    assert fields["nfailed"] == "0"
    assert fields["repetitions"] == "10"
    locked = {
        "fraction_connected_9": "0.8476676664",
        "total_searches_9": "12051.33333",
        "unique_saddles_9": "84",
        "force_calls_9": "22526117",
        "force_calls_mean_error_9": "2148944.936",
        "force_calls": "23663658.6",
        "fraction_connected": "0.8473331647",
        "total_searches": "12652.5",
        "unique_saddles": "85.7",
        "force_calls_mean_error": "2214757.756",
    }
    for key, value in locked.items():
        assert fields[key] == value
    reps = _comment_reps(path)
    assert len(reps) == 10
    numbers = [
        int(line.split()[2])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# rep ")
    ]
    assert numbers == list(range(1, 11))
    sample = reps[:9]
    assert statistics.fmean(row["fraction_connected"] for row in sample) == pytest.approx(
        float(fields["fraction_connected_9"])
    )
    assert statistics.fmean(row["force_calls"] for row in sample) == pytest.approx(float(fields["force_calls_9"]))
    assert statistics.fmean(row["force_calls"] for row in reps) == pytest.approx(float(fields["force_calls"]))


def test_pt_island_ranks_one_row_per_code():
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    rows = comparison_rows(ROOT, oracle)
    fragment = saddle_pt_html(ROOT, oracle)
    folded = fragment.casefold()
    for banned in (
        "reproduce",
        "reproduces",
        "does not reproduce",
        "legacy",
        "current",
        "paper baseline",
        "10%",
        "wilson",
    ):
        assert banned.casefold() not in folded
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    assert set(by_method) == {"Dimer", "Dynamics"}
    for group in by_method.values():
        labels = [row["label"] for row in group]
        assert labels.count(SVN_LABEL) == 1
        assert labels.count(DEVELOP_LABEL) == 1
        assert sum(bool(row["archive"]) for row in group) == 1
        assert [row["rank"] for row in group] == list(range(1, len(group) + 1))
        calls = [float(row["metrics"]["force_calls"]) for row in group]
        assert calls == sorted(calls)
        archive = next(row for row in group if row["archive"])
        for row in group:
            if row["archive"]:
                assert row["difference"]["force_calls"] == 0
                continue
            for key, gap in row["difference"].items():
                assert gap == pytest.approx(float(row["metrics"][key]) - float(archive["metrics"][key]))
                assert f"{gap:+.10g}" in fragment
    assert "5075453.5" not in fragment
    assert "1550616.8" not in fragment
    assert "5426864.8" not in fragment
    assert "23663658.6" not in fragment
    assert "4933128" in fragment
    assert "1530551.222" in fragment
    assert "5682183.778" in fragment
    assert "22526117" in fragment


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
