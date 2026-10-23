"""Lossless gzip of the archived rgsaddle 0.2.2 saddle starts."""

import gzip
import json
import subprocess
from pathlib import Path

from optbench.board import _cap_of, _force_tol, _image_count, _load_group_records, _score_records
from optbench.campaign import list_templates
from optbench.dat import parse_lenient
from optbench.problems import load_all

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "084c547ad24f4c987539dc47797297d8fbe9c1a4"
STEMS = (
    "rgsaddle-0.2.2-saddle-dimer-fire",
    "rgsaddle-0.2.2-saddle-dimer-lbfgs",
    "rgsaddle-0.2.2-saddle-dimer-rotskip",
    "rgsaddle-0.2.2-saddle-lanczos",
)


def _rows(blob: bytes) -> list[dict]:
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


def _show(path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{ARCHIVE}:{path}"])


def _without_starts(document: dict) -> dict:
    copy = json.loads(json.dumps(document))
    for run in copy["runs"]:
        run.pop("starts")
    return copy


def test_recompressed_rgsaddle_022_starts_match_the_jsonl(tmp_path: Path):
    """Each gzip is the archived JSONL, row for row, and the board does not move."""
    problems = load_all(ROOT / "problems")
    templates = {item["id"]: item for item in list_templates(ROOT)}
    campaign = ROOT / "results" / "campaign"
    for stem in STEMS:
        plain = campaign / f"{stem}.starts.jsonl"
        packed = campaign / f"{stem}.starts.jsonl.gz"
        assert not plain.exists()
        old = _show(f"results/campaign/{stem}.starts.jsonl")
        assert _rows(gzip.decompress(packed.read_bytes())) == _rows(old)

        fields, _duplicates = parse_lenient((campaign / f"{stem}.dat").read_text(encoding="utf-8"))
        template = templates[fields["template"]]
        problem = problems[fields["problem"]]
        cap = _cap_of(problem, template, fields)
        force_tol = _force_tol(problem, template)
        images = _image_count(problem, template, fields)
        gz_records, gz_label = _load_group_records({"starts_gz": packed}, cap, force_tol, images)
        old_path = tmp_path / f"{stem}.starts.jsonl"
        old_path.write_bytes(old)
        plain_records, plain_label = _load_group_records({"starts_jsonl": old_path}, cap, force_tol, images)
        assert gz_label == "starts.jsonl.gz"
        assert plain_label == "starts.jsonl"
        assert gz_records == plain_records
        assert _score_records(gz_records, cap) == _score_records(plain_records, cap)

        old_doc = json.loads(_show(f"results/campaign/{stem}.v2.json"))
        new_doc = json.loads((campaign / f"{stem}.v2.json").read_text(encoding="utf-8"))
        assert [run["starts"] for run in new_doc["runs"]] == [[] for _run in new_doc["runs"]]
        assert _without_starts(new_doc) == _without_starts(old_doc)

    board = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", ARCHIVE, "--", "results/board"],
        check=False,
    )
    assert board.returncode == 0
