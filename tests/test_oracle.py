import json
from pathlib import Path

from optbench.check import check

ROOT = Path(__file__).resolve().parents[1]


def test_archived_entries_match_the_contracts():
    assert check(ROOT / "problems", ROOT / "oracle" / "entries.jsonl") == 0


def test_main_oracle_has_every_archived_score():
    rows = [
        json.loads(line)
        for line in (ROOT / "oracle" / "entries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 151
    assert all(row["commit"] == "0f26b796" for row in rows)


def test_stated_inputs_are_in_the_geometry_store():
    manifest = {
        json.loads(line)["path"]
        for line in (ROOT / "inputs" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert "website/minimization/lj38/lj38-clusters.tgz" in manifest
    assert "website/saddle-search/pt-island/pt-island.con" in manifest
    assert "website/saddle-search/pt-island/POSCAR_pt_island" in manifest
