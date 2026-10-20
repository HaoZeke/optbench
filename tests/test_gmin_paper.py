"""The archived GMIN log line is the reproduction counter."""

from optbench.runners.gmin_paper import parse_log


def test_target_hit_line_matches_run_gmin():
    text = (
        "Qu       1768 E=    -173.928416337074     steps=  112\n"
        "gsaveit> Target hit after            1768 quenches          254024 function calls.\n"
        "finalio> time elapsed               2.45 seconds\n"
    )
    parsed = parse_log(text)
    assert parsed["quenches"] == 1768
    assert parsed["ncalls"] == 254024
    assert parsed["success"] == 1
    assert parsed["seconds"] == 2.45


def test_a_miss_is_a_failure():
    parsed = parse_log("mc> run finished without the target\n")
    assert parsed["success"] == 0
    assert parsed["ncalls"] is None
