"""Barrier-window parsing and the binary potential socket."""

from optbench.eon_htst import assign_delta, parse_searches
from optbench.failures import classify_outcome
from optbench.potserver import PotServer, remote_bcalc
from optbench.pykmc_kmc import HTST, reduce_htst


def test_parse_searches_keeps_good_and_repeat_ids():
    text = "\n".join(
        [
            "wuid type barrier max-dist sad-fcs mins-fcs pref-fcs result",
            "----",
            "0 random 0.60100 1 10 2 0 good-4",
            "1 random 0.00000 1 3 1 0 Not",
            "2 random 0.60100 1 8 2 0 repeat-4",
        ]
    )
    rows = parse_searches(text)
    assert rows[0]["process_id"] == 4
    assert rows[0]["barrier"] == 0.601
    assert rows[1]["process_id"] is None
    assert rows[2]["process_id"] == 4
    assert [row["sad_fcs"] + row["mins_fcs"] for row in rows] == [12, 4, 10]


def test_assign_delta_sums_to_the_harness_tally():
    rows = [
        {"sad_fcs": 10, "mins_fcs": 0},
        {"sad_fcs": 30, "mins_fcs": 0},
    ]
    assign_delta(rows, 7)
    assert [row["force_calls"] for row in rows] == [1, 6]
    assert sum(row["force_calls"] for row in rows) == 7


def test_one_search_gets_the_whole_delta():
    rows = [{"sad_fcs": 4, "mins_fcs": 1}]
    assign_delta(rows, 19)
    assert rows[0]["force_calls"] == 19


def test_blank_tuning_is_left_out_of_the_dat():
    from optbench.dat import parse
    from optbench.pykmc_kmc import htst_dat

    summary = {
        "barrier_jobs": 1.5,
        "barrier_force_calls": 12,
        "barrier_cycles": 1,
        "rate_jobs": None,
        "matched_barriers": 15,
        "reference_barriers": 15,
        "processes": 1,
    }
    bare = {
        "id": "eon-client/htst-pt-dimer-lbfgs",
        "code": "eOn",
        "code_version": "3.6.0",
        "commit": "4c1b814f62f1dcf5fcd0cf18ba0b6df3bcff7a28",
        "algorithm": "Dimer L-BFGS",
        "reducer": "htst-cycles",
    }
    text = htst_dat(
        summary,
        template=bare,
        problem={"id": "hTST/Pt", "convergence": {"max_force_calls": 20000000}},
        spec=HTST["hTST/Pt"],
        searches=3,
        minimize_calls=0,
        partial=False,
        comments=["# counter harness"],
    )
    fields = parse(text)
    assert "tuning" not in fields
    assert fields["counter"] == "harness"
    assert fields["cap"] == "20000000"
    tuned = htst_dat(
        summary,
        template={**bare, "tuning": "beland-2015-pt111"},
        problem={"id": "hTST/Pt", "convergence": {"max_force_calls": 20000000}},
        spec=HTST["hTST/Pt"],
        searches=3,
        minimize_calls=0,
        partial=False,
        comments=["# counter harness"],
    )
    assert parse(tuned)["tuning"] == "beland-2015-pt111"


def test_window_cost_is_the_harness_sum():
    spec = HTST["hTST/Pt"]
    barriers = list(spec["barriers"])
    rows = []
    for index, barrier in enumerate(barriers):
        rows.append(
            {
                "force_calls": 100 + index,
                "barrier": barrier,
                "process_id": index + 1,
            }
        )
    summary = reduce_htst(rows, spec)
    assert summary["barrier_cycles"] == 1
    assert summary["barrier_force_calls"] == sum(100 + index for index in range(len(barriers)))
    assert classify_outcome(calls=float(summary["barrier_force_calls"]), cap=20000000) is None


def test_bcalc_matches_shifted_morse_and_keeps_the_socket_count():
    from optbench.eon_harness import morse_shift

    def evaluate(positions, numbers, box):
        return morse_shift(positions, numbers, box)

    positions = [0.0, 0.0, 0.0, 2.8, 0.0, 0.0]
    numbers = [78, 78]
    box = [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0]
    server = PotServer(evaluate).start()
    try:
        first = remote_bcalc(server.host, server.port, positions, numbers, box)
        second = remote_bcalc(server.host, server.port, positions, numbers, box)
    finally:
        server.stop()
    energy, forces = morse_shift(positions, numbers, box)
    assert first[2] == 1
    assert second[2] == 2
    assert abs(first[0] - energy) < 1e-9
    assert list(first[1]) == list(forces)
