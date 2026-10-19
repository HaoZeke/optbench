"""Analytic curvature is charged as finite-difference force calls."""

from optbench.curvature import full_hessian_cost, product_cost, reprice_record


def test_a_product_costs_one_force_call_and_a_matrix_costs_the_dimension():
    assert product_cost() == 1
    assert full_hessian_cost(114) == 114
    assert full_hessian_cost(3) == 3


def test_reprice_adds_the_charge_once_and_a_total_at_the_cap_fails():
    start = {
        "force_calls": 66,
        "hessian_calls": 65,
        "success": True,
        "failure": None,
        "converged": True,
        "reasons": [],
    }
    priced = reprice_record(start, unit_cost=114, cap=10000)
    assert priced["gradient_calls"] == 66
    assert priced["hessian_force_calls"] == 65 * 114
    assert priced["force_calls"] == 66 + 65 * 114
    assert priced["success"] is True
    again = reprice_record(priced, unit_cost=114, cap=10000)
    assert again["force_calls"] == priced["force_calls"]
    over = reprice_record(
        {"force_calls": 128, "hessian_calls": 127, "success": True, "failure": None, "reasons": []},
        unit_cost=114,
        cap=10000,
    )
    assert over["force_calls"] == 128 + 127 * 114
    assert over["success"] is False
    assert over["failure"] == "cap"
    assert over["converged"] is False


def test_a_crash_stays_a_crash_when_the_charge_meets_the_cap():
    crashed = reprice_record(
        {
            "force_calls": 10,
            "hessian_calls": 100,
            "success": False,
            "failure": "crash",
            "reasons": ["crash"],
        },
        unit_cost=114,
        cap=10000,
    )
    assert crashed["failure"] == "crash"
    assert crashed["force_calls"] == 10 + 100 * 114
