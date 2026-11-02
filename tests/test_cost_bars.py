"""Log10 cost bars. These tests do not build the board."""

import math

from optbench.site_charts import (
    AXIS_TITLE,
    cost_bars_svg,
    cost_decades,
    decade_ceil,
    decade_floor,
    log_cost_fraction,
    spans_a_decade,
)


def test_a_power_of_ten_is_its_own_decade():
    assert math.floor(math.log10(1000.0)) in {2, 3}
    assert decade_floor(1000.0) == 3
    assert decade_ceil(1000.0) == 3
    assert cost_decades([1000.0, 1000.0]) == (3, 3)
    assert log_cost_fraction(1000.0, 3, 3) == 0.0


def test_sella_is_shorter_than_lammps_on_one_axis():
    costs = [502.784, 520.333, 2885, 25022, 31447.3, 100831, 266216, 412102, 444244, 451753, 451959]
    span = cost_decades(costs)
    assert span is not None
    k_lo, k_hi = span
    assert 10.0 ** k_lo <= min(costs)
    assert 10.0 ** k_hi >= max(costs)
    assert k_hi > k_lo
    sella = log_cost_fraction(31447.3, k_lo, k_hi)
    lammps = log_cost_fraction(451959, k_lo, k_hi)
    assert sella is not None and lammps is not None
    assert 0.0 <= sella < lammps <= 1.0
    assert sella != lammps
    chart = cost_bars_svg(
        [
            (
                "Lennard-Jones 38 transition-state separation",
                [{"code": "Sella", "best": 31447.3}, {"code": "LAMMPS", "best": 451959}],
            )
        ]
    )
    assert AXIS_TITLE in chart
    assert "A shorter bar is a lower cost on a log scale shared by the codes of this problem." in chart
    assert "16.0%" not in chart
    assert "is-best" in chart
    widths = [float(part.split('width="')[1].split('"')[0]) for part in chart.split("<rect ")[1:]]
    assert widths[0] < widths[1]
    assert min(widths) >= 0.0


def test_negative_costs_stay_off_the_log_axis():
    assert cost_decades([-589.507, -584.579]) is None
    chart = cost_bars_svg(
        [("Binary Lennard-Jones 100", [{"code": "GMIN", "best": -589.507}, {"code": "Eon svn", "best": -584.579}])]
    )
    assert "left off the log scale" in chart
    assert "<rect " not in chart
    assert "-589.507" in chart


def test_a_mixed_problem_draws_only_the_positive_costs():
    chart = cost_bars_svg(
        [("mixed", [{"code": "low", "best": -2.0}, {"code": "high", "best": 50.0}])]
    )
    assert chart.count("<rect ") == 1
    assert "left off the axis" in chart
    assert AXIS_TITLE in chart


def test_a_log_scale_needs_a_full_decade():
    assert spans_a_decade([100.0, 1000.0])
    assert not spans_a_decade([100.0, 999.0])
    assert not spans_a_decade([0.0, 1000.0])
    assert not spans_a_decade([1000.0])


def test_the_ends_of_the_axis_are_exact():
    assert decade_floor(999.0) == 2
    assert decade_ceil(1001.0) == 4
    assert log_cost_fraction(100.0, 2, 4) == 0.0
    assert log_cost_fraction(10000.0, 2, 4) == 1.0
    assert 0.0 < log_cost_fraction(1000.0, 2, 4) < 1.0
