import json
from pathlib import Path

from benchmarks.bench_eon_develop import _specs
from optbench.asv_campaign import STEMS, _reason, write_one
from optbench.dat import parse

ROOT = Path(__file__).resolve().parents[1]


def test_reason_codes_round_trip():
    assert _reason(1, 0) == "0"
    assert _reason(0, 5) == "5"
    assert _reason(0, 100) == "no-overlap"
    assert _reason(0, 101) == "incomplete"
    assert _reason(0, 102) is None


def test_every_develop_benchmark_has_a_campaign_stem():
    classes = {spec["cls"] for spec in _specs()}
    assert classes == set(STEMS)


def test_client_tracks_use_the_cap_rule(tmp_path: Path):
    spec = {
        "cls": "EonDevelopTsLj38Lanczos",
        "template": "eon/develop-ts-lj38-lanczos",
        "kind": "ts-lj38",
        "seeds": [1],
        "starts": ["0000", "0001", "0002"],
    }
    tracks = {
        "force_calls": [100, 40, 20000],
        "success": [1, 0, 1],
        "failure_cause": [0, 5, 0],
        "seconds": [0.05, 0.04, 0.2],
    }
    output = tmp_path / "bench.dat"
    summary = write_one(ROOT, spec, tracks, output, asv_commit="abc", workers=1)
    fields = parse(output.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "70"
    assert fields["nfailed"] == "1"
    assert fields["force_calls_min"] == "40"
    assert fields["force_calls_max"] == "100"
    assert fields["commit"] == "7095b23ac17d49f434a5e08ccb072897ceca698e"
    assert fields["counter"] == "total_force_calls"
    assert fields["asv_commit"] == "abc"
    assert summary["nfailed"] == 1
    starts = [json.loads(line) for line in output.with_suffix(".starts").read_text().splitlines()]
    assert starts[1]["reason"] == "5"
    assert starts[1]["failed"] is True
    assert starts[2]["failed"] is True
    assert starts[0]["seconds"] == 0.05


def test_akmc_mean_keeps_only_repetitions_that_reach_27(tmp_path: Path):
    spec = {
        "cls": "EonDevelopSaddlePtDimer",
        "template": "eon/develop-saddle-pt-dimer",
        "kind": "akmc",
        "seeds": [1000, 1001],
        "starts": ["pt-island"],
    }
    tracks = {
        "force_calls": [1000, 500],
        "success": [1, 0],
        "failure_cause": [0, 101],
        "unique_saddles": [30, 20],
        "fraction_connected": [0.75, 0.5],
        "total_searches": [4000, 2000],
        "seconds": [10.0, 20.0],
    }
    output = tmp_path / "saddle.dat"
    write_one(ROOT, spec, tracks, output, asv_commit="abc", workers=8)
    fields = parse(output.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "1000"
    assert fields["nfailed"] == "1"
    assert fields["repetitions"] == "2"
    assert fields["counter"] == "client-search-results"
    assert fields["total_searches"] == "4000"
    assert "7095b23ac17d49f434a5e08ccb072897ceca698e" in fields["commit"]
