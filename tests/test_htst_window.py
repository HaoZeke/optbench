"""The hTST window closes only after every reference barrier is in the table."""

from optbench.htst import Process, Search, reduce_htst, spec_from_problem


def _process(process_id: int, barrier: float) -> Process:
    return Process(
        id=process_id,
        saddle_energy=-1.0,
        prefactor=1.2e12,
        barrier=barrier,
        barrier_raw=barrier,
        rate=1.0,
        repeats=0,
    )


def test_a_missing_reference_barrier_does_not_close_the_window():
    spec = spec_from_problem(
        {
            "id": "hTST/Pt",
            "htst": {
                "barriers": [0.601, 0.620],
                "rate_ref": 10.0,
                "temperature": 700,
                "prefactor": 1.2e12,
                "barrier_ceiling": 1.3,
            },
        }
    )
    processes = [_process(0, 0.601)]
    searches = [
        Search(force_calls=10, barrier=0.601, status="good", process_id=0),
        Search(force_calls=10, barrier=0.601, status="repeat", process_id=0),
    ]
    reduction = reduce_htst(processes, searches, spec)
    assert reduction.missed_barriers == [0.620]
    assert reduction.barrier_windows == 0
    assert reduction.searches == 2
