"""The fd8fb3d1 saddle sample replaced the archived rgsaddle 0.2.2 JSONL.

The parent commit checks that the gzip decompresses to that archived JSONL.
These four campaigns are the later sample, so the check here is the sample
the board scores.
"""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from optbench.board import _cap_of, _force_tol, _image_count, _load_group_records, _score_records
from optbench.campaign import list_templates
from optbench.dat import parse_lenient
from optbench.problems import load_all

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("failure", "force_calls", "start", "success")
SAMPLE = {
    "rgsaddle-0.2.2-saddle-dimer-fire": (
        "de9cb220f8f181ffe390a8c3c4b5e523000e6179f13af8b1559beba3aace2850",
        {"cap": 10337, "wrong_stationary": 9663},
    ),
    "rgsaddle-0.2.2-saddle-dimer-lbfgs": (
        "d854f4048f0651066afd099a025e035689b73f29a74e10f40dad432f2d3cbe94",
        {"cap": 4, "wrong_stationary": 19996},
    ),
    "rgsaddle-0.2.2-saddle-dimer-rotskip": (
        "e00766c1cac493862ec01fd163b2b8062c9641d59ae9baff7650451d8a016a63",
        {"cap": 5, "wrong_stationary": 19995},
    ),
    "rgsaddle-0.2.2-saddle-lanczos": (
        "f7dc6b1d6cc32a096722188d2ea2859ce534862912c6b561ecf30d241fddde7e",
        {"cap": 1, "wrong_stationary": 19999},
    ),
}


def test_fd8fb3d1_saddle_gzip_is_the_sample_the_board_scores():
    problems = load_all(ROOT / "problems")
    templates = {item["id"]: item for item in list_templates(ROOT)}
    campaign = ROOT / "results" / "campaign"
    for stem, (digest, expected) in SAMPLE.items():
        plain = campaign / f"{stem}.starts.jsonl"
        packed = campaign / f"{stem}.starts.jsonl.gz"
        assert not plain.exists()
        blob = gzip.decompress(packed.read_bytes())
        assert hashlib.sha256(blob).hexdigest() == digest
        rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
        assert len(rows) == 20000
        assert all(tuple(row) == FIELDS for row in rows)
        assert dict(Counter(row["failure"] for row in rows)) == expected
        assert not any(row["success"] for row in rows)

        fields, _duplicates = parse_lenient((campaign / f"{stem}.dat").read_text(encoding="utf-8"))
        template = templates[fields["template"]]
        problem = problems[fields["problem"]]
        cap = _cap_of(problem, template, fields)
        records, label = _load_group_records(
            {"starts_gz": packed},
            cap,
            _force_tol(problem, template),
            _image_count(problem, template, fields),
        )
        assert label == "starts.jsonl.gz"
        assert [
            (record["failure"], record["force_calls"], record["start"], record["success"]) for record in records
        ] == [(row["failure"], row["force_calls"], row["start"], False) for row in rows]
        summary = _score_records(records, cap)
        assert summary["n"] == 20000
        assert summary["nfailed"] == 20000
        assert summary["par2_mean"] == 10128.0
        document = json.loads((campaign / f"{stem}.v2.json").read_text(encoding="utf-8"))
        assert [run["starts"] for run in document["runs"]] == [[]]
        metrics = document["runs"][0]["metrics"]
        assert metrics["n_cap"] == expected["cap"]
        assert metrics["n_wrong_stationary"] == expected["wrong_stationary"]
        assert metrics["n_divergence"] == 0
