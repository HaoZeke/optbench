"""Aluminum adatom scores against the archived rates and the product catalogue."""

from pathlib import Path

import pytest

from optbench.amd import (
    BOND_BOOST_250K,
    KELVIN_TO_EV,
    MD_STANDARD_250K,
    catalogue_accuracy,
    distribution,
    force_calls_per_simulated_time,
    match_product,
    mobile_indices,
    mobile_rmsd,
    old_scheme_confidence,
    parse_con,
    percent_uncertainty,
    rate_summary,
    select_thermal,
    simulated_time_seconds,
)
from optbench.campaign import _check_stop, geometry_archive, load_template, resolve_problem
from optbench.check import archive_gaps, load_oracle
from optbench.dat import normalize, parse_lenient
from optbench.eon_amd import catalogue_confidence, catalogue_reached, rate_relevant_repeats, search_cost
from optbench.eon_jobs import JOBS
from optbench.problems import load_all

ROOT = Path(__file__).resolve().parents[1]
ALADATOM = ROOT / "archive/entries/accelerated-md/aladatom/true_standard/eon-pr-langevin/benchmark.dat"
STANDARD = ROOT / "archive/entries/accelerated-md/standard_aladatom/eon-md-langevin/benchmark.dat"


def _archived(path: Path) -> dict[str, str]:
    fields, duplicates = parse_lenient(path.read_text(encoding="utf-8"))
    assert duplicates == []
    canonical, _renamed = normalize(fields)
    return canonical


def _products():
    reactant_path = geometry_archive(
        ROOT,
        {"id": "accelerated-md", "input": {"archive_path": "website/accelerated-md/reactant.con"}},
    )
    reactant, fixed = parse_con(reactant_path.read_text(encoding="utf-8"))
    mobile = mobile_indices(fixed)
    products = []
    for index in range(24):
        path = geometry_archive(
            ROOT,
            {
                "id": "accelerated-md",
                "input": {"archive_path": f"website/accelerated-md/products/product{index}.con"},
            },
        )
        points, mask = parse_con(path.read_text(encoding="utf-8"))
        assert mask.tolist() == fixed.tolist()
        products.append(points)
    return reactant, fixed, mobile, products


def test_archived_percent_uncertainty_matches_the_formula():
    aladatom = _archived(ALADATOM)
    standard = _archived(STANDARD)
    assert percent_uncertainty(
        float(aladatom["true_rate"]), float(aladatom["uncertainty"])
    ) == pytest.approx(float(aladatom["percent_uncertainty"]), rel=1e-6)
    assert percent_uncertainty(
        float(standard["true_rate"]), float(standard["uncertainty"])
    ) == pytest.approx(float(standard["percent_uncertainty"]), abs=0.001)


def test_problem_files_carry_the_archived_rates_and_the_catalogue():
    aladatom = resolve_problem(ROOT, "accelerated-md/aladatom")
    standard = resolve_problem(ROOT, "accelerated-md/standard_aladatom")
    archived_aladatom = _archived(ALADATOM)
    archived_standard = _archived(STANDARD)
    assert aladatom["contract"] == "stated"
    assert standard["contract"] == "stated"
    assert aladatom["metrics"]["required"] == [
        "percent_uncertainty",
        "thermostat",
        "true_rate",
        "uncertainty",
    ]
    assert aladatom["reference"]["true_rate"] == pytest.approx(float(archived_aladatom["true_rate"]))
    assert aladatom["reference"]["uncertainty"] == pytest.approx(float(archived_aladatom["uncertainty"]))
    assert aladatom["reference"]["percent_uncertainty"] == pytest.approx(
        float(archived_aladatom["percent_uncertainty"])
    )
    assert aladatom["reference"]["thermostat"] == archived_aladatom["thermostat"]
    assert standard["reference"]["true_rate"] == pytest.approx(float(archived_standard["true_rate"]))
    assert standard["reference"]["uncertainty"] == pytest.approx(float(archived_standard["uncertainty"]))
    assert standard["reference"]["percent_uncertainty"] == pytest.approx(
        float(archived_standard["percent_uncertainty"])
    )
    assert tuple(aladatom["catalogue"]["probabilities"]) == MD_STANDARD_250K
    assert tuple(standard["catalogue"]["probabilities"]) == MD_STANDARD_250K
    assert standard["catalogue"]["temperature_K"] == 250
    assert standard["thermal"]["temperature_K"] == 150
    assert aladatom["thermal"]["temperature_K"] == 250
    assert aladatom["sample"]["products"] == 5000
    assert aladatom["sample"]["repetitions"] == 3
    assert aladatom["sample"]["reference_products"] == 15000
    assert standard["sample"]["repetitions"] == 3
    assert aladatom["potential"]["kind"] == "eam_al"
    assert aladatom["input"]["reactant"] == "website/accelerated-md/reactant.con"


def test_stating_the_contract_does_not_add_archive_gaps():
    problems = load_all(ROOT / "problems")
    rows = load_oracle(ROOT / "oracle" / "entries.jsonl")
    gaps = archive_gaps(problems, rows)
    known = (ROOT / "oracle" / "archive_gaps.txt").read_text(encoding="utf-8").splitlines()
    assert gaps == known


def test_published_comparison_rows_and_the_bond_boost_catalogue():
    problem = resolve_problem(ROOT, "accelerated-md/aladatom")
    rows = {row["name"]: row for row in problem["reference"]["comparison"]}
    langevin = rows["eon-md-langevin"]
    anderson = rows["eon-md-anderson"]
    assert percent_uncertainty(langevin["true_rate"], langevin["uncertainty"]) == pytest.approx(
        langevin["percent_uncertainty"], abs=0.001
    )
    assert percent_uncertainty(anderson["true_rate"], anderson["uncertainty"]) == pytest.approx(
        anderson["percent_uncertainty"], abs=0.001
    )
    assert langevin["force_calls_per_second"] != pytest.approx(
        langevin["force_calls"] / langevin["true_rate"]
    )
    seconds = simulated_time_seconds(langevin["true_rate"], problem["sample"]["products"])
    per_time = force_calls_per_simulated_time(langevin["force_calls"], seconds)
    assert per_time == pytest.approx(langevin["force_calls"] * langevin["true_rate"] / 5000)
    assert per_time != pytest.approx(langevin["force_calls_per_second"])
    accuracy = catalogue_accuracy(distribution(BOND_BOOST_250K), distribution(MD_STANDARD_250K))
    assert accuracy == pytest.approx(0.9872155802060875)
    assert accuracy < 1.0
    assert catalogue_accuracy(distribution(MD_STANDARD_250K), distribution(MD_STANDARD_250K)) == 1.0


def test_one_rate_has_no_standard_error_and_confidence_needs_twenty_repeats():
    summary = rate_summary([2.5e8])
    assert summary["uncertainty"] is None
    assert summary["percent_uncertainty"] is None
    assert summary["n"] == 1
    two = rate_summary([1.0, 3.0])
    assert two["true_rate"] == pytest.approx(2.0)
    assert two["uncertainty"] == pytest.approx(1.0)
    assert old_scheme_confidence(0) == 0.0
    assert old_scheme_confidence(20) == pytest.approx(0.95)
    assert old_scheme_confidence(19) < 0.95
    text = "\n".join(
        [
            "header",
            "rule",
            "1 random 0.2 1.0 10 4 0 good-1",
            "2 random 0.2 1.0 3 1 0 repeat-1",
        ]
    )
    cost = search_cost(text)
    assert cost["force_calls"] == 18
    assert cost["repeat_lines"] == 1
    assert cost["confidence"] == 0.0


def test_thermal_window_uses_the_eon_kelvin_conversion():
    assert KELVIN_TO_EV == pytest.approx(1.0 / 11604.5)
    processes = [
        {"barrier": 0.001, "rate": 1.0},
        {"barrier": 0.20, "rate": 4.0},
        {"barrier": 0.80, "rate": 2.0},
    ]
    kept = select_thermal(processes, 250.0, 20.0, 0.01)
    assert [item["barrier"] for item in kept] == [0.20]


def test_current_eon_writes_a_fixed_mask_of_seven():
    text = "\n".join(
        [
            "Generated by eOn",
            '{"con_spec_version": 3}',
            "1 1 1",
            "90 90 90",
            "1",
            "2",
            "1",
            "Al",
            "Coordinates of Component 1",
            "0 0 0 7 0",
            "1 0 0 0 1",
        ]
    )
    _points, fixed = parse_con(text)
    assert fixed.tolist() == [True, False]
    success_only = "0 0 0 7 0\n"
    with pytest.raises(ValueError):
        parse_con(success_only)


def test_catalogue_geometries_stay_distinct_at_the_match_tolerance():
    reactant, fixed, mobile, products = _products()
    assert len(reactant) == 151
    assert int(fixed.sum()) == 50
    assert len(mobile) == 101
    closest = min(
        mobile_rmsd(products[left], products[right], mobile)
        for left in range(24)
        for right in range(left + 1, 24)
    )
    assert closest == pytest.approx(0.3026187386307368, rel=1e-9)
    assert closest > 0.15
    nearest = min(mobile_rmsd(reactant, product, mobile) for product in products)
    assert nearest > 5.0
    for index, product in enumerate(products):
        assert match_product(product, products, mobile, 0.15) == index
    assert match_product(reactant, products, mobile, 0.15) is None


def test_akmc_templates_follow_the_problem_stop():
    for name, problem_name in (
        ("eon/aladatom-akmc", "accelerated-md/aladatom"),
        ("eon/standard-aladatom-akmc", "accelerated-md/standard_aladatom"),
    ):
        template = load_template(ROOT, name)
        problem = resolve_problem(ROOT, problem_name)
        assert template["problem"] == problem["id"]
        assert template["commit"] == "1d71c944501832bfafb955ffe13b9aae50c8efd4"
        assert template["ini"]["Potential"]["potential"] == "eam_al"
        assert template["ini"]["AKMC"]["confidence"] == 1.1
        assert template["ini"]["AKMC"]["thermally_accessible_window"] == problem["convergence"][
            "thermally_accessible_window"
        ]
        assert template["ini"]["Main"]["temperature"] == problem["thermal"]["temperature_K"]
        assert template["ini"]["Prefactor"]["default_value"] == 1e12
        assert template["ini"]["Saddle Search"]["displace_atom_list"] == "150"
        assert template["pass_limit"] == 800
        assert template["confidence_min_searches"] == 2336
        assert _check_stop(problem, template) == template["failure_cap"]
    assert JOBS["aladatom-akmc"][0] == "eon/aladatom-akmc"
    assert JOBS["standard-aladatom-akmc"][0] == "eon/standard-aladatom-akmc"
    assert JOBS["aladatom-akmc"][1] == "amd"


def _search_text(rows: list[str]) -> str:
    return "\n".join(
        [
            "wuid type barrier max-dist sad-fcs mins-fcs pref-fcs result",
            "rule",
            *rows,
        ]
    )


def test_hop_repeats_do_not_raise_confidence_after_a_fast_process():
    """The 250 K table's hop is negligible once the 0.152 eV process exists.

    Those two rows are the printed processtable from the one-seed 250 K
    catalogue. Twenty-one hop repeats after the fast process must leave
    the streak at 0. eOn's own counter would already be past 0.95.
    """
    hop = {"id": 1, "barrier": 0.42971, "rate": 2175.02}
    fast = {"id": 2, "barrier": 0.15218, "rate": 8.55320e8}
    early = ["1 random 0.42971 1.0 153 122 0 good-1"]
    early += [f"{index} random 0.42971 1.0 10 4 0 repeat-1" for index in range(2, 12)]
    assert rate_relevant_repeats(_search_text(early), [hop, fast], 250.0, 20.0, 0.01) == 10
    rows = early + ["12 random 0.15218 1.3 201 127 0 good-2"]
    rows += [f"{index} random 0.42971 1.0 10 4 0 repeat-1" for index in range(13, 34)]
    text = _search_text(rows)
    progress = catalogue_confidence(text, [hop, fast], 250.0, 20.0, 0.01, 21)
    assert progress["repeats"] == 0
    assert progress["confidence"] == 0.0
    assert progress["eon_repeats"] == 21
    assert progress["repeat_lines"] == 31
    assert search_cost(text)["repeats"] == 31
    finished = rows + [f"{index} random 0.15218 1.3 10 4 0 repeat-2" for index in range(34, 54)]
    done = catalogue_confidence(_search_text(finished), [hop, fast], 250.0, 20.0, 0.01, None)
    assert done["repeats"] == 20
    assert done["confidence"] == pytest.approx(0.95)
    assert done["eon_repeats"] == 51


def test_a_new_fast_process_resets_the_streak_and_the_window_drops_hops():
    low = {"id": 1, "barrier": 0.13915, "rate": 2.11243e7}
    mid = {"id": 2, "barrier": 0.17340, "rate": 1.49232e6}
    hop = {"id": 3, "barrier": 0.42971, "rate": 3.65123e-3}
    thermal = select_thermal([low, mid, hop], 150.0, 20.0, 0.01)
    assert [item["id"] for item in thermal] == [1, 2]
    assert mid["rate"] / (low["rate"] + mid["rate"]) > 0.01
    rows = ["1 random 0.13915 1.0 10 4 0 good-1"]
    rows += [f"{index} random 0.13915 1.0 10 4 0 repeat-1" for index in range(2, 22)]
    assert rate_relevant_repeats(_search_text(rows), [low, mid, hop], 150.0, 20.0, 0.01) == 20
    rows.append("22 random 0.17340 1.0 10 4 0 good-2")
    rows.append("23 random 0.42971 1.0 10 4 0 repeat-3")
    rows.append("24 random 0.17340 1.0 10 4 0 repeat-2")
    assert rate_relevant_repeats(_search_text(rows), [low, mid, hop], 150.0, 20.0, 0.01) == 1


def test_twenty_hop_repeats_do_not_stop_before_the_search_floor():
    """Seeds 20261019 and 20261020 reached old-scheme 0.95 on hops alone.

    Rep 2 stopped at 112 searches with rate 9109.47. Rep 3 stopped at
    80 searches with rate 9110.29. Neither table contained a barrier
    under 0.35 eV. The floor keeps that confidence from ending the run.
    """
    early = {
        "confidence": 0.95,
        "rate": 9109.47,
        "total_searches": 112,
    }
    assert catalogue_reached(early, 0.95, 2336) is False
    done = {
        "confidence": 0.95,
        "rate": 3595857710.0,
        "total_searches": 2336,
    }
    assert catalogue_reached(done, 0.95, 2336) is True
    assert catalogue_reached({**done, "total_searches": 2335}, 0.95, 2336) is False
    assert catalogue_reached({**done, "rate": 0.0}, 0.95, 2336) is False
    with pytest.raises(ValueError):
        catalogue_reached(done, 0.95, -1)


def test_the_one_percent_share_is_the_boundary():
    dominant = {"id": 1, "barrier": 0.20, "rate": 99.0}
    included = {"id": 2, "barrier": 0.25, "rate": 1.0}
    excluded = {"id": 3, "barrier": 0.26, "rate": 0.9}
    rows = [
        "1 random 0.20 1.0 10 4 0 good-1",
        "2 random 0.25 1.0 10 4 0 good-2",
        "3 random 0.25 1.0 10 4 0 repeat-2",
    ]
    assert rate_relevant_repeats(_search_text(rows), [dominant, included], 250.0, 20.0, 0.01) == 1
    rows.append("4 random 0.26 1.0 10 4 0 good-3")
    rows.append("5 random 0.26 1.0 10 4 0 repeat-3")
    rows.append("6 random 0.20 1.0 10 4 0 repeat-1")
    assert (
        rate_relevant_repeats(
            _search_text(rows), [dominant, included, excluded], 250.0, 20.0, 0.01
        )
        == 2
    )
    with pytest.raises(ValueError):
        rate_relevant_repeats("header\nrule\n", [], 250.0, 20.0, 0.01, share=0.0)
