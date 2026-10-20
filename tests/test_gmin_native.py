"""The LJ98 symmetrise parser uses the target-hit line."""

from optbench.engines.gmin_native import comparison_fields, paper_calls, record_from_log


def test_paper_call_parser_uses_the_target_hit_line():
    log = (
        "saveit> Target hit after 10 quenches 20 function calls.\n"
        "gsaveit> Target hit after            3160 quenches          658449 function calls.\n"
        "finalio> time elapsed              31.36 seconds\n"
    )
    assert paper_calls(log) == (3160, 658449, 31.36)
    assert paper_calls("no hit") == (None, None, None)


def test_a_target_hit_under_the_cap_is_one_success():
    log = (
        "gsaveit> Target hit after            14 quenches            1456 function calls.\n"
        "finalio> time elapsed              1.0 seconds\n"
    )
    record = record_from_log("coords.1", log, 100_000_000)
    assert record["success"] is True
    assert record["failure"] is None
    assert record["force_calls"] == 1456
    assert record["basin_hopping_steps"] == 14
    assert record["counter"] == "harness"
    other = record_from_log(
        "coords.2",
        "gsaveit> Target hit after 16 quenches 1600 function calls.\nfinalio> time elapsed 3.0 seconds\n",
        100_000_000,
    )
    fields = comparison_fields([record, other])
    assert fields["basin_hopping_steps_mean"] == "15"
    assert fields["energy_calls_mean"] == "1528"
    assert fields["nfailed"] == "0"
    assert fields["cpu_time_mean"] == "2"
