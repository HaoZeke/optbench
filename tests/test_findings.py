"""Archive numbers stay the paper rows. A later file is a finding, not a gate."""

import json
from pathlib import Path

from optbench.check import held_back_errors, load_oracle
from optbench.findings import campaign_mismatch, findings_errors, load_ledger, row_id
from optbench.problems import load_problem
from optbench.site import problem_html

ROOT = Path(__file__).resolve().parents[1]


def test_findings_follow_the_oracle():
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    ledger = load_ledger(ROOT / "oracle" / "findings.jsonl")
    assert len(ledger) == len(oracle) == 151
    assert [row_id(row) for row in ledger] == [row_id(row) for row in oracle]
    assert findings_errors(ROOT) == []
    assert not (ROOT / "oracle" / "reproduction.jsonl").exists()
    assert not (ROOT / "docs" / "reproduction.md").exists()
    beside = [row for row in ledger if row.get("beside")]
    assert len(beside) == 86
    text = (ROOT / "docs" / "findings.md").read_text(encoding="utf-8")
    assert "There is no pass/fail window." in text


def test_pele_archive_stop_stays_beside_the_harness_stop():
    ledger = {row_id(row): row for row in load_ledger(ROOT / "oracle" / "findings.jsonl")}
    forward = ledger["tsopt/LJ38/pele_ffd"]
    central = ledger["tsopt/LJ38/pele_cd"]
    island = ledger["tsopt/pt-island/pele"]
    assert forward["beside"]["stop2014_calls"] == 207.51
    assert forward["beside"]["harness_stop_mean"] == 254.48
    assert central["beside"]["stop2014_calls"] == 263.52
    assert central["beside"]["harness_stop_mean"] == 335.06
    assert island["beside"]["stop2014_calls"] == 88.551
    assert "starting set" in forward["finding"]
    assert held_back_errors(ROOT) == []


def test_a_list_that_keeps_templates_off_the_table_is_rejected(tmp_path: Path):
    optbench = tmp_path / "optbench"
    optbench.mkdir()
    (optbench / "board.py").write_text("def holdback(root):\n    return None\n", encoding="utf-8")
    (optbench / "findings.py").write_text('ROWS = [{"hold": ["eon/neb-pt-lbfgs"]}]\n', encoding="utf-8")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "findings.jsonl").write_text(
        json.dumps({"beside": {"omitted": ["eon-client/neb-pt-lbfgs"]}}) + "\n",
        encoding="utf-8",
    )
    errors = held_back_errors(tmp_path)
    assert any(line.startswith("optbench/board.py") for line in errors)
    assert any(line.startswith("optbench/findings.py") for line in errors)
    assert any("oracle/findings.jsonl" in line for line in errors)


def test_a_campaign_file_that_disagrees_is_reported(tmp_path: Path):
    row = {
        "suite": "minimization",
        "problem": "lj38",
        "entry": "eon-lbfgs",
        "beside": {
            "branch": "main",
            "path": "results/campaign/bad.dat",
            "kind": "dat",
        },
        "values": [{"metric": "force_calls", "archive": 181, "harness": 181}],
    }
    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    (campaign / "bad.dat").write_text("force_calls 1000\nnfailed 0\n", encoding="utf-8")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "findings.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    errors = campaign_mismatch(tmp_path)
    assert errors
    assert any("force_calls" in line and "1000" in line for line in errors)


def test_archive_rows_stay_on_the_archive_table():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    html = problem_html(problem, oracle)
    assert ">eon-cg<" in html
    assert "not reproduced" not in html
    assert "2014 archive" in html
