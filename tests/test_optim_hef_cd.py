"""2014 OPTIM LJ38 central differences. The score is OPTIM's own counter."""

import tomllib
from pathlib import Path

from optbench.dat import parse
from optbench.engines.optim_search import (
    WALES_27058_SHA256,
    energy_gradient_calls,
    odata_text,
    paper_counts,
)
from optbench.reproduction import within_mean

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = ROOT / "archive" / "entries" / "tsopt" / "LJ38" / "optim_cd" / "odata.preamble"
DAT = ROOT / "results" / "campaign" / "optim-27058-lj38-hef-cd.dat"


def test_preamble_is_the_archived_central_difference_odata():
    text = PREAMBLE.read_text(encoding="utf-8")
    keywords = [
        line.split()[0]
        for line in text.splitlines()
        if line.strip() and not line.startswith("comment") and line.split()[0] != "debug"
    ]
    assert "bfgsts" in keywords
    assert "45 14 16 0.68 100" in text
    assert "bfgsconv" in keywords
    assert "9.36585D-05" in text
    assert "bfgssteps" in keywords
    assert "nohess" in keywords
    assert "usediag" not in keywords
    template = tomllib.loads((ROOT / "templates" / "optim" / "lj38-hef-cd.toml").read_text(encoding="utf-8"))
    assert template["id"] == "optim/lj38-hef-cd"
    assert template["commit"] == WALES_27058_SHA256
    assert template["optim"]["method"] == "hef-cd"
    assert "paper_baseline" not in template


def test_odata_matches_the_awk_spacing():
    text = odata_text("debug\nnohess\n", [("1.0", "-2.5", "3")])
    assert text.endswith("points\nAX   1.0 -2.5 3\n")
    assert "AX   1.0" in text


def test_energy_gradient_field_is_the_sixth_token():
    log = " OPTIM> # of energy+gradient calls=               223 time=       0.01\n"
    assert energy_gradient_calls(log) == 223
    assert energy_gradient_calls("no counter\n") is None


def test_paper_round_leaves_failures_out_of_the_mean():
    counts = paper_counts([1, 2, 10000], [True, True, False])
    assert counts["force_calls"] == 2
    assert counts["nfailed"] == 1
    half = paper_counts([182, 183], [True, True])
    assert half["unrounded_median"] == 182.5
    assert half["force_calls_median"] == 182


def test_committed_cd_row_matches_the_archive_integer():
    fields = parse(DAT.read_text(encoding="utf-8"))
    assert fields["counter"] == "optim"
    assert fields["template"] == "optim/lj38-hef-cd"
    assert fields["commit"] == WALES_27058_SHA256
    assert int(fields["nfailed"]) == 0
    assert int(fields["n"]) == 200
    assert within_mean(223.0, float(fields["force_calls"]), None)
    assert int(fields["force_calls"]) == 223
    assert int(fields["force_calls_min"]) == 77
    assert int(fields["force_calls_max"]) == 772
    starts = (DAT.with_suffix(".starts.jsonl")).read_text(encoding="utf-8").splitlines()
    assert len(starts) == 200
