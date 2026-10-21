"""Measured campaigns stay in the ledger. The 2014 table is the archive."""

import json
from pathlib import Path

from optbench.check import load_oracle
from optbench.ledger import (
    LEDGER_PATH,
    build_rows,
    holdback,
    ledger_errors,
    load_ledger,
    row_id,
)
from optbench.problems import load_problem
from optbench.site import problem_html

ROOT = Path(__file__).resolve().parents[1]


def _metric(item: dict, name: str) -> float:
    found = next(row for row in item["metrics"] if row["metric"] == name)
    return float(found["latest"])


def test_the_ledger_covers_the_oracle_in_order():
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    ledger = load_ledger(ROOT / LEDGER_PATH)
    assert len(ledger) == len(oracle) == 151
    assert [row_id(row) for row in ledger] == [row_id(row) for row in oracle]
    assert ledger == build_rows(ROOT)
    for row in ledger:
        assert "status" not in row
        assert "cause" not in row
        assert "paper_baseline" not in row
        blob = json.dumps(row)
        assert "reproduced" not in blob
        assert "paper_baseline" not in blob
    assert ledger_errors(ROOT) == []


def test_measured_campaigns_stay_in_the_ledger():
    ledger = {row_id(row): row for row in load_ledger(ROOT / LEDGER_PATH)}
    dimer = ledger["saddle-search/pt-island/eon-dimer"]["latest"]
    assert [item["label"] for item in dimer] == [
        "EON SVN r2549",
        "eOn (develop 500350d9)",
    ]
    eon = dimer[0]
    assert eon["path"] == "results/campaign/eon-svn-r2549-saddle-pt-dimer.dat"
    assert _metric(eon, "force_calls") == 5719004.8
    assert _metric(eon, "unique_saddles") == 112.6
    assert abs(_metric(eon, "fraction_connected") - 0.7018048496) < 1e-9
    develop = dimer[1]
    assert develop["path"] == "results/campaign/eon-develop-500350d9-saddle-pt-dimer.dat"
    assert develop["template"] == "eon/develop-saddle-pt-dimer"
    assert _metric(develop, "force_calls") > 0

    for entry in (
        "adaptive_HSphere-lbfgs-davidson-sm50",
        "adaptive_HSphere-lbfgs-lanczos-sm50",
        "adaptive_HSphere-lbfgs-lanczos-sm0",
    ):
        assert ledger[f"saddle-search/pt-island/{entry}"]["latest"] == []

    assert _metric(ledger["tsopt/LJ38/sella"]["latest"][0], "force_calls") == 69
    assert _metric(ledger["tsopt/pt-island/sella"]["latest"][0], "force_calls") == 59

    forward = ledger["tsopt/LJ38/pele_ffd"]["latest"][0]
    central = ledger["tsopt/LJ38/pele_cd"]["latest"][0]
    island = ledger["tsopt/pt-island/pele"]["latest"][0]
    assert _metric(forward, "force_calls") == 207.51
    assert _metric(central, "force_calls") == 263.52
    assert abs(_metric(island, "force_calls") - 88.551) < 1e-6


def test_a_different_counter_is_held_and_a_distant_mean_is_not():
    held = holdback(ROOT)
    assert ("double-ended-saddle-search/lj38", "eon-client/de-lj38-ci-swdneb") in held.templates
    assert ("tsopt/LJ38", "sella/lj38-ts-tuned") not in held.templates
    assert ("tsopt/LJ38", "sella/lj38-ts-default") not in held.templates
    assert "results/campaign/eon-svn-r2549-saddle-pt-dimer.dat" not in held.paths
    assert "results/campaign/eon-3.5.0-saddle-pt-classic.dat" not in held.paths
    assert ledger_errors(ROOT) == []


def test_archive_tables_have_no_pass_mark():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    html = problem_html(problem, oracle)
    assert ">eon-cg<" in html
    assert "pele-fire" in html
    assert "not reproduced" not in html
    assert "Wilson" not in html
    assert "legacy" not in html.lower()
    assert "paper baseline" not in html.lower()
    saddle = load_problem(ROOT / "problems" / "saddle-search" / "pt-island.toml")
    saddle_html = problem_html(saddle, oracle)
    assert "EON SVN r2549" not in saddle_html
    assert "eOn (develop 500350d9)" not in saddle_html
    assert "5719004.8" not in saddle_html
    assert "406030.3" not in saddle_html
    assert ">115.5<" not in saddle_html
    assert "not reproduced" not in saddle_html
