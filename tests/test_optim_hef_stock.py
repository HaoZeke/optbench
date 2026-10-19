"""Latest OPTIM, stock hybrid eigenvector following, on the LJ38 starts."""

import json
import tomllib
from pathlib import Path

from optbench.dat import parse
from optbench.engines.optim_stock import SVNREV, TARBALL_SHA256
from optbench.reproduction import holdback, load_ledger, row_id, within_mean

ROOT = Path(__file__).resolve().parents[1]
DAT = ROOT / "results" / "campaign" / "optim-df8da599-lj38-hef-stock.dat"


def test_stock_template_is_the_current_tarball():
    template = tomllib.loads((ROOT / "templates" / "optim" / "lj38-hef-stock.toml").read_text(encoding="utf-8"))
    assert template["id"] == "optim/lj38-hef-stock"
    assert template["code"] == "optim"
    assert template["code_version"] == SVNREV
    assert template["commit"] == TARBALL_SHA256
    assert template["optim"]["method"] == "hef-stock"
    assert template["optim"]["geometry"].endswith("coords-xyz.tar.gz")
    assert template["optim"]["starts"] == 200
    assert "paper_baseline" not in template
    assert "ranked" not in template


def test_committed_stock_row_is_a_full_harness_score():
    fields = parse(DAT.read_text(encoding="utf-8"))
    assert fields["counter"] == "harness"
    assert "ranked" not in fields
    assert "partial" not in fields
    assert fields["template"] == "optim/lj38-hef-stock"
    assert fields["code_version"] == SVNREV
    assert fields["commit"] == TARBALL_SHA256
    assert fields["tuning"] == "default"
    assert int(fields["n"]) == 200
    assert int(fields["nfailed"]) == 0
    assert int(fields["force_calls"]) == 95
    assert float(fields["force_calls_min"]) == 24
    assert float(fields["force_calls_max"]) == 297
    assert float(fields["force_calls_median"]) == 82.5
    assert not within_mean(223.0, float(fields["force_calls"]), None)
    starts = [json.loads(line) for line in DAT.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(starts) == 200
    assert {row["counter"] for row in starts} == {"harness"}
    assert sum(row["success"] for row in starts) == 200


def test_ledger_records_the_stock_version_beside_the_archive():
    ledger = {row_id(row): row for row in load_ledger(ROOT / "oracle" / "reproduction.jsonl")}
    row = ledger["tsopt/LJ38/optim_cd"]
    assert row["status"] == "not_reproduced"
    assert row["cause"] == "optim_lj38_stock"
    assert row["gap"] == "force_calls 95 vs 223 (57.4%)"
    assert row["checks"][0]["archive"] == 223.0
    assert row["checks"][0]["rerun"] == 95.0
    assert row["checks"][0]["ok"] is False
    assert row["failure_check"]["ok"] is True
    rerun = row["rerun"]
    assert rerun["template"] == "optim/lj38-hef-stock"
    assert rerun["path"] == "results/campaign/optim-df8da599-lj38-hef-stock.dat"
    assert rerun["sha"] == "92150d4357e901855df7bd8b4c1039fc3245a106"
    assert rerun["paper_baseline"] is False
    assert rerun["unrounded_mean"] == 94.575
    assert SVNREV in row["cause_detail"]
    assert TARBALL_SHA256 in row["cause_detail"]
    held = holdback(ROOT)
    assert ("tsopt/LJ38", "optim/lj38-hef-stock") not in held.templates
    assert rerun["path"] not in held.paths
