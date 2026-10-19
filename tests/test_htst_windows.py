"""benchmark.dat lines for an hTST window sample."""

from optbench.dat import parse
from optbench.htst_windows import htst_dat

SUMMARY = {
    "barrier_jobs": 1.5,
    "barrier_force_calls": 12,
    "barrier_cycles": 1,
    "rate_jobs": None,
    "matched_barriers": 15,
    "reference_barriers": 15,
    "processes": 1,
}
PROBLEM = {"id": "hTST/Pt", "convergence": {"max_force_calls": 20000000}}
SPEC = {"temperature": 700.0, "prefactor": 1.2e12, "rate_ref": 155055901.244}


def _text(template: dict) -> str:
    return htst_dat(
        SUMMARY,
        template=template,
        problem=PROBLEM,
        spec=SPEC,
        searches=3,
        minimize_calls=0,
        unfinished=False,
        comments=["# counter harness"],
    )


def test_a_blank_tuning_is_omitted():
    fields = parse(
        _text(
            {
                "id": "eon-client/htst-pt-dimer-lbfgs",
                "code": "eOn",
                "code_version": "3.6.0",
                "commit": "4c1b814f62f1dcf5fcd0cf18ba0b6df3bcff7a28",
                "algorithm": "Dimer L-BFGS",
                "reducer": "htst-cycles",
            }
        )
    )
    assert "tuning" not in fields
    assert fields["counter"] == "harness"
    assert fields["cap"] == "20000000"
    assert fields["template"] == "eon-client/htst-pt-dimer-lbfgs"


def test_a_set_tuning_is_kept():
    fields = parse(
        _text(
            {
                "id": "eon-client/htst-pt-dimer-lbfgs",
                "code": "eOn",
                "code_version": "3.6.0",
                "commit": "abc",
                "algorithm": "Dimer L-BFGS",
                "tuning": "beland-2015-pt111",
                "reducer": "htst-cycles",
            }
        )
    )
    assert fields["tuning"] == "beland-2015-pt111"
