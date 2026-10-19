"""Parallel-replica time accounting and the Vineyard prefactor cost.

The rate is events divided by physical time. A batch that escaped
contributes every replica up to the earliest transition. The Vineyard
row adds the Hessian calls. Neither test starts eonclient.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from optbench.campaign import load_template
from optbench.dat import parse_lenient
from optbench.eon_amd import PREFACTOR_COUNT_KEY, search_cost
from optbench.eon_jobs import JOBS
from optbench.eon_pr import COUNT_KEY, batch_account, poisson_rate, read_batch
from optbench.score import eon_process_calls

ROOT = Path(__file__).resolve().parents[1]


def test_poisson_rate_matches_the_archive_percent():
    summary = poisson_rate(20280, 20280 / 2.542941e8)
    assert summary["true_rate"] == pytest.approx(2.542941e8)
    assert summary["percent_uncertainty"] == pytest.approx(100.0 / math.sqrt(20280))
    assert summary["uncertainty"] == pytest.approx(summary["true_rate"] / math.sqrt(20280))
    with pytest.raises(ValueError):
        poisson_rate(0, 1.0)
    with pytest.raises(ValueError):
        poisson_rate(1, 0.0)


def test_batch_account_stops_at_the_earliest_transition():
    quiet = [
        {"force_calls": 10, "simulation_time_s": 5.0e-11},
        {"force_calls": 12, "simulation_time_s": 5.0e-11},
    ]
    held = batch_account(quiet)
    assert held["escaped"] is False
    assert held["time_s"] == pytest.approx(1.0e-10)
    assert held["force_calls"] == 22
    earliest = 7.5e-13
    mixed = [
        {"force_calls": 100, "transition_time_s": earliest, "job": "a"},
        {"force_calls": 80, "simulation_time_s": 5.0e-11, "job": "b"},
        {"force_calls": 80, "simulation_time_s": 5.0e-11, "job": "c"},
        {"force_calls": 90, "transition_time_s": 2.0e-12, "job": "d"},
    ]
    escaped = batch_account(mixed)
    assert escaped["escaped"] is True
    assert escaped["t_min"] == pytest.approx(earliest)
    assert escaped["time_s"] == pytest.approx(4 * earliest)
    assert escaped["force_calls"] == 350
    assert len(escaped["earliest"]) == 1
    tied = [
        {"force_calls": 1, "transition_time_s": 1.0e-12},
        {"force_calls": 1, "transition_time_s": 1.0e-12},
    ]
    same = batch_account(tied)
    assert same["time_s"] == pytest.approx(2.0e-12)
    assert len(same["earliest"]) == 2
    with pytest.raises(ValueError):
        batch_account([])
    with pytest.raises(ValueError):
        batch_account([{"force_calls": 1, "error": "no results"}])


def test_read_batch_keeps_the_product_text(tmp_path: Path):
    job = tmp_path / "0_1"
    job.mkdir()
    (job / "results.dat").write_text(
        "\n".join(
            [
                "eam_al potential_type",
                "-470.239689 potential_energy_reactant",
                "1 transition_found",
                "7.500000e-13 transition_time_s",
                "-470.257806 potential_energy_product",
                "2721 total_force_calls",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (job / "product.con").write_text("product\n", encoding="utf-8")
    rows = read_batch(tmp_path)
    assert rows[0]["force_calls"] == 2721
    assert rows[0]["transition_time_s"] == pytest.approx(7.5e-13)
    assert rows[0]["reactant_energy"] == pytest.approx(-470.239689)
    assert rows[0]["product_text"] == "product\n"


def test_pr_and_vineyard_templates_keep_the_paper_setup():
    for name, temperature in (
        ("eon/aladatom-pr", 250),
        ("eon/standard-aladatom-pr", 150),
    ):
        template = load_template(ROOT, name)
        assert template["ini"]["Potential"]["potential"] == "eam_al"
        assert template["ini"]["Main"]["temperature"] == temperature
        assert template["ini"]["Main"]["job"] == "parallel_replica"
        assert template["ini"]["Dynamics"]["thermostat"] == "langevin"
        assert template["ini"]["Dynamics"]["langevin_friction"] == pytest.approx(0.001)
        assert template["ini"]["Dynamics"]["time_step"] == pytest.approx(1.0)
        assert template["ini"]["Communicator"]["jobs_per_bundle"] == 1
        assert template["ini"]["Optimizer"]["converged_force"] == pytest.approx(0.001)
        assert template["commit"] == "1d71c944501832bfafb955ffe13b9aae50c8efd4"
    assert JOBS["aladatom-pr"] == ("eon/aladatom-pr", "pr")
    assert JOBS["standard-aladatom-pr"] == ("eon/standard-aladatom-pr", "pr")
    for name, temperature in (
        ("eon/aladatom-akmc-vineyard", 250),
        ("eon/standard-aladatom-akmc-vineyard", 150),
    ):
        template = load_template(ROOT, name)
        assert template["count_prefactor"] is True
        assert template["ini"]["Potential"]["potential"] == "eam_al"
        assert template["ini"]["Main"]["temperature"] == temperature
        assert template["ini"]["Prefactor"]["default_value"] == 0
        assert template["ini"]["Prefactor"]["filter_scheme"] == "fraction"
        assert template["ini"]["Prefactor"]["filter_fraction"] == pytest.approx(0.25)
        assert template["ini"]["Prefactor"]["rate_estimation"] == "htst"
        assert template["ini"]["Saddle Search"]["displace_atom_list"] == "150"
        assert template["confidence_min_searches"] == 2336
        assert "all_free_atoms" not in template["ini"]["Prefactor"]
    assert JOBS["aladatom-akmc-vineyard"] == ("eon/aladatom-akmc-vineyard", "amd")
    assert JOBS["standard-aladatom-akmc-vineyard"] == (
        "eon/standard-aladatom-akmc-vineyard",
        "amd",
    )
    assert COUNT_KEY == "eon_dynamics_calls"
    assert PREFACTOR_COUNT_KEY == "eon_search_calls+pref-fcs"


def test_finished_campaigns_keep_the_measured_rates():
    from benchmarks.bench_eon_aladatom import (
        EonAladatomAkmcVineyard,
        EonAladatomPr,
        EonStandardAladatomAkmcVineyard,
        EonStandardAladatomPr,
    )

    expected = (
        (
            "eon-3.6.0-aladatom-pr",
            "eon/aladatom-pr",
            "accelerated-md/aladatom",
            "eon_dynamics_calls",
            7113198.0,
            1.371993049e10,
            1252452569.0,
            0.9228616047,
            EonAladatomPr,
            7395010,
            1.339512953e10,
        ),
        (
            "eon-3.6.0-aladatom-akmc-vineyard",
            "eon/aladatom-akmc-vineyard",
            "accelerated-md/aladatom",
            "eon_search_calls+pref-fcs",
            2794693.667,
            3160585420.0,
            733882.994,
            0.9322157031,
            EonAladatomAkmcVineyard,
            1926475,
            3162017222.0,
        ),
        (
            "eon-3.6.0-standard-aladatom-akmc-vineyard",
            "eon/standard-aladatom-akmc-vineyard",
            "accelerated-md/standard_aladatom",
            "eon_search_calls+pref-fcs",
            2794693.667,
            33590220.0,
            4353.527306,
            0.8110605786,
            EonStandardAladatomAkmcVineyard,
            1926475,
            33598760.0,
        ),
        (
            "eon-3.6.0-standard-aladatom-pr",
            "eon/standard-aladatom-pr",
            "accelerated-md/standard_aladatom",
            "eon_dynamics_calls",
            77290280.33,
            116827661.7,
            23847346.58,
            0.7564661853,
            EonStandardAladatomPr,
            107186854,
            84038839.39,
        ),
    )
    for (
        stem,
        template,
        problem,
        count_key,
        calls,
        rate,
        uncertainty,
        catalogue,
        bench,
        rep_calls,
        rep_rate,
    ) in expected:
        dat = ROOT / "results" / "campaign" / f"{stem}.dat"
        text = dat.read_text(encoding="utf-8")
        fields, _duplicates = parse_lenient(text)
        assert fields["counter"] == "harness"
        assert fields["count_keys"] == count_key
        assert fields["template"] == template
        assert fields["problem"] == problem
        assert fields["repetitions"] == "3"
        assert "partial" not in fields
        assert float(fields["nfailed"]) == 0.0
        assert float(fields["true_rate"]) == pytest.approx(rate)
        assert float(fields["uncertainty"]) == pytest.approx(uncertainty)
        assert float(fields["rate_accuracy"]) == 0.0
        assert float(fields["catalogue_accuracy"]) == pytest.approx(catalogue)
        assert float(fields["force_calls"]) == pytest.approx(calls)
        assert float(fields["reference_rate"]) == pytest.approx(
            254294100.0 if problem.endswith("aladatom") and "standard" not in problem else 835900.0
        )
        records = [
            json.loads(line)
            for line in dat.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(records) == 3
        assert all(record["success"] is True for record in records)
        assert records[0]["seed"] == 20261018
        assert records[0]["force_calls"] == rep_calls
        measured = bench().measure("20261018", "rep-1")
        assert measured["force_calls"] == rep_calls
        assert measured["rate"] == pytest.approx(rep_rate)
        assert measured["count_key"] == count_key
    vineyard = (ROOT / "results" / "campaign" / "eon-3.6.0-aladatom-akmc-vineyard.dat").read_text(
        encoding="utf-8"
    )
    assert "barrier 0.13915 prefactor 9.07549e+11 rate 1421480000 channel 1" in vineyard
    assert "barrier 0.15218 prefactor 8.48564e+11 rate 725794000 channel 4" in vineyard
    cold = (
        ROOT / "results" / "campaign" / "eon-3.6.0-standard-aladatom-akmc-vineyard.dat"
    ).read_text(encoding="utf-8")
    assert "barrier 0.13915 prefactor 9.07549e+11 rate 19171400 channel 1" in cold
    assert "barrier 0.15218 prefactor 8.48564e+11 rate 6539820 channel 4" in cold
    cold_pr = (ROOT / "results" / "campaign" / "eon-3.6.0-standard-aladatom-pr.dat").read_text(
        encoding="utf-8"
    )
    cold_fields, _duplicates = parse_lenient(cold_pr)
    assert cold_fields["events"] == "24"
    assert float(cold_fields["percent_uncertainty"]) == pytest.approx(20.41241452)
    assert float(cold_fields["reactant_energy"]) == pytest.approx(-470.239689)
    assert float(cold_fields["time_s"]) == pytest.approx(2.054308e-07)


def test_vineyard_cost_adds_the_hessian_calls():
    assert eon_process_calls(10, 4, 9) == 23
    text = "\n".join(
        [
            "header",
            "rule",
            "1 random 0.2 1.0 10 4 9 good-1",
            "2 random 0.2 1.0 3 1 7 repeat-1",
        ]
    )
    assert search_cost(text)["force_calls"] == 18
    assert search_cost(text, include_prefactor=True)["force_calls"] == 34
