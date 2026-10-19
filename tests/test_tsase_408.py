"""TSASE r408 dimer BFGS reproduces the archived LJ38 lowest-eigenvector row."""

import json
from pathlib import Path

from optbench.reproduction import load_ledger, row_id, within_mean

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "campaign" / "tsase-408-lj38-dimer-bfgs.dat"


def test_the_dimer_bfgs_row_matches_the_archive():
    ledger = {row_id(row): row for row in load_ledger(ROOT / "oracle" / "reproduction.jsonl")}
    row = ledger["lowest-eigenvector/lj38/tsase-dimer-bfgs"]
    assert row["status"] == "reproduced"
    assert row["rerun"]["paper_baseline"] is True
    assert row["rerun"]["template"] == "tsase-408/lj38-dimer-bfgs"
    assert row["checks"] == [
        {"archive": 27.0, "metric": "force_calls", "ok": True, "rerun": 27.0, "se": None}
    ]
    assert row["failure_check"]["ok"] is True
    assert row["failure_check"]["rerun"] == 0.0
    text = CAMPAIGN.read_text(encoding="utf-8")
    fields = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        fields[key] = value
    assert fields["force_calls"] == "27"
    assert fields["force_calls_min"] == "13"
    assert fields["nfailed"] == "0"
    assert within_mean(65.0, float(fields["force_calls_max"]), None)
    starts = [
        json.loads(line)
        for line in (CAMPAIGN.with_suffix(".starts.jsonl")).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(starts) == 200
    assert all(row["success"] for row in starts)
    assert sum(row["force_calls"] for row in starts) == 5425
