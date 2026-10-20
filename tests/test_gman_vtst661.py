"""Stock vtstcode 6.6.1 on the archived Pt-heptamer band."""

import json
from pathlib import Path

from optbench.reproduction import within_mean

ROOT = Path(__file__).resolve().parents[1]
DAT = ROOT / "results" / "campaign" / "vtstcode-6.6.1-neb-pt-glbfgs-hess.dat"


def test_stock_vtst_661_is_its_own_code():
    text = DAT.read_text(encoding="utf-8")
    assert "force_calls 51\n" in text
    assert "force_calls_min 27\n" in text
    assert "force_calls_max 81\n" in text
    assert "nfailed 0\n" in text
    assert "code vtstcode\n" in text
    assert "code_version 6.6.1\n" in text
    assert "template vtstcode/neb-pt-glbfgs-hess\n" in text
    assert "counter harness\n" in text
    assert "ranked false" not in text
    assert "partial true" not in text
    assert "code Gman\n" not in text
    assert within_mean(48.0, 51.0, None)
    starts = [
        json.loads(line)
        for line in DAT.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(starts) == 13
    assert all(
        row["success"] and row["counter"] == "harness" and row["code"] == "vtstcode" for row in starts
    )
    assert sum(row["force_calls"] for row in starts) / 8 / 13 == 51
