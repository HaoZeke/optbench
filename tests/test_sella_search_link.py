"""Sella saddle searches use the eOn r2205 link and repeat test."""

from optbench.ase_sella import _apply_search_catalog


def _row(plus, minus, saddle, barrier, reasons=None, judged=True):
    return {
        "stationary_reasons": list(reasons or []),
        "endpoint_judged": judged,
        "reactant_minimum": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "endpoint_plus": plus,
        "endpoint_minus": minus,
        "saddle_positions": saddle,
        "barrier": barrier,
        "cell": [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]],
        "link_tolerance": 0.1,
        "energy_difference": 0.01,
        "barrier_max": 1.5,
        "force_calls": 10,
    }


def test_catalog_keeps_one_endpoint_and_rejects_a_repeat():
    reactant = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    product = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    saddle = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    repeat = [[0.0, 0.0, 0.0], [2.05, 0.0, 0.0]]
    rows = [
        _row(reactant, product, saddle, 0.4),
        _row(reactant, product, repeat, 0.4),
        _row(reactant, reactant, saddle, 0.4),
        _row(product, product, saddle, 0.4),
        _row(reactant, product, saddle, 0.4, reasons=["hessian_index"], judged=False),
    ]
    catalog = _apply_search_catalog(rows)
    assert rows[0]["success"] is True
    assert rows[0]["failure"] is None
    assert rows[1]["success"] is False
    assert rows[1]["reasons"] == ["duplicate"]
    assert rows[1]["failure"] == "divergence"
    assert "not_connected" in rows[2]["reasons"]
    assert "not_connected" in rows[3]["reasons"]
    assert rows[4]["reasons"] == ["hessian_index"]
    assert rows[4]["failure"] == "wrong_stationary"
    assert catalog.unique == 1
    assert catalog.qualifying == 1
    assert catalog.fraction_connected == 3 / 5
