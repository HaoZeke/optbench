import json
from pathlib import Path

from optbench.check import check, tuned_templates

ROOT = Path(__file__).resolve().parents[1]


def test_a_template_ending_in_tuned_is_rejected(tmp_path: Path):
    folder = tmp_path / "templates" / "demo"
    folder.mkdir(parents=True)
    (folder / "search-tuned.toml").write_text('id = "demo/search-tuned"\n', encoding="utf-8")
    (folder / "search.toml").write_text('id = "demo/search"\n', encoding="utf-8")
    (folder / "alias.toml").write_text('id = "demo/alias-tuned"\n', encoding="utf-8")
    assert tuned_templates(tmp_path) == [
        "demo/alias-tuned",
        "templates/demo/search-tuned.toml",
    ]
    assert tuned_templates(ROOT) == []


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
