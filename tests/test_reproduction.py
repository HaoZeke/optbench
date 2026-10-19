"""The 2014 ledger is the paper baseline, and a miss fails the check."""

import json
from pathlib import Path

from optbench.check import load_oracle
from optbench.problems import load_problem
from optbench.reproduction import (
    baseline_errors,
    diagnostics,
    holdback,
    judge_checks,
    load_ledger,
    reproduction_errors,
    row_id,
)
from optbench.site import problem_html

ROOT = Path(__file__).resolve().parents[1]


def test_the_ledger_covers_the_oracle_in_order():
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    ledger = load_ledger(ROOT / "oracle" / "reproduction.jsonl")
    assert len(ledger) == len(oracle) == 151
    assert [row_id(row) for row in ledger] == [row_id(row) for row in oracle]
    counts = {"reproduced": 0, "not_reproduced": 0, "not_rerun": 0}
    for row in ledger:
        failure = row.get("failure_check") if row.get("rerun") else None
        status = judge_checks(row.get("checks") or [], failure)
        if row.get("rerun") is None:
            status = "not_rerun"
        assert row["status"] == status
        counts[status] += 1
        if status == "not_reproduced":
            assert row["cause"]
        if (row.get("rerun") or {}).get("paper_baseline"):
            assert status == "reproduced"
    assert counts == {"reproduced": 46, "not_reproduced": 29, "not_rerun": 76}
    assert reproduction_errors(ROOT) == []


def test_pele_paper_baseline_is_the_2014_stop():
    ledger = {row_id(row): row for row in load_ledger(ROOT / "oracle" / "reproduction.jsonl")}
    forward = ledger["tsopt/LJ38/pele_ffd"]
    central = ledger["tsopt/LJ38/pele_cd"]
    island = ledger["tsopt/pt-island/pele"]
    assert forward["status"] == "reproduced"
    assert central["status"] == "reproduced"
    assert island["status"] == "reproduced"
    assert forward["rerun"]["baseline_metric"] == "stop2014_calls"
    assert forward["rerun"]["stop2014_calls"] == 207.51
    assert forward["rerun"]["harness_stop_mean"] == 254.48
    assert central["rerun"]["stop2014_calls"] == 263.52
    assert central["rerun"]["harness_stop_mean"] == 335.06
    assert island["rerun"]["stop2014_calls"] == 88.551
    assert island["rerun"]["paper_baseline"] is True
    assert "starting set" in forward["note"]
    held = holdback(ROOT)
    assert ("tsopt/LJ38", "sella/lj38-ts-tuned") in held.templates
    assert ("tsopt/LJ38", "sella/lj38-ts-default") not in held.templates
    assert ("double-ended-saddle-search/lj38", "eon-client/de-lj38-ci-swdneb") in held.templates
    assert "results/campaign/eon-3.5.0-saddle-pt-classic.dat" in held.paths
    morse = ledger["minimization/morse-bulk/ase-preconLBFGS"]
    assert morse["status"] == "not_reproduced"
    assert morse["cause"] == "ase_stock"
    assert morse["rerun"]["path"] == "results/campaign/ase-3.29.0-morse-preconlbfgs-default.dat"
    assert morse["rerun"]["code_version"] == "3.29.0"
    assert morse["rerun"]["rank"] is True
    assert morse["rerun"]["paper_baseline"] is False
    assert morse["checks"][0]["archive"] == 34.0
    assert morse["checks"][0]["rerun"] == 60.0
    assert ("minimization/morse-bulk", "ase/morse-preconlbfgs-default") not in held.templates
    assert "results/campaign/ase-3.29.0-morse-preconlbfgs-default.dat" not in held.paths
    assert ("minimization/morse-bulk", "ase/morse-preconlbfgs-tuned") not in held.templates
    assert ("minimization/morse-bulk", "ase/morse-preconlbfgs-tuned") in diagnostics(ROOT)


def test_a_paper_baseline_outside_the_archive_fails(tmp_path: Path):
    source = next(
        row
        for row in load_ledger(ROOT / "oracle" / "reproduction.jsonl")
        if row_id(row) == "minimization/lj38/eon-lbfgs"
    )
    row = json.loads(json.dumps(source))
    row["rerun"]["path"] = "results/campaign/bad.dat"
    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    (campaign / "bad.dat").write_text("force_calls 1000\nnfailed 0\n", encoding="utf-8")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "reproduction.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    errors = baseline_errors(tmp_path)
    assert errors
    assert any("force_calls" in line and "1000" in line for line in errors)


def test_legacy_rows_that_miss_are_marked_in_place():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    html = problem_html(problem, oracle)
    assert ">eon-cg<" in html
    assert "pele-fire · not reproduced" in html
    assert "outside the archive tolerance" in html
    assert "eon-cg · not reproduced" not in html
