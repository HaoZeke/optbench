"""A full stock row is ranked. The archive table sits beside it."""

from pathlib import Path

from optbench.check import _ranked_stock_errors, load_oracle
from optbench.problems import load_problem
from optbench.site import problem_html

ROOT = Path(__file__).resolve().parents[1]

STOCK = (
    ("double-ended-saddle-search/lj38", "eon-client/de-lj38-ci-swdneb"),
    ("double-ended-saddle-search/lj38", "eon-svn/de-lj38-dimer"),
    ("double-ended-saddle-search/pt-island", "eon-svn/de-pt-lanczos"),
    ("double-ended-saddle-search/pt-island", "eon-svn/de-pt-lanczos-improved"),
    ("minimization/lj38", "pele/lj38-fire"),
    ("minimization/morse-bulk", "pele/morse-lbfgs"),
    ("neb/Pt-heptamer", "eon/neb-pt-lbfgs"),
    ("neb/Pt-heptamer", "eon-client/neb-pt-lbfgs"),
    ("saddle-search/pt-island", "softsaddle/saddle-davidson-sm50"),
    ("saddle-search/pt-island", "softsaddle/saddle-lanczos-sm0"),
    ("saddle-search/pt-island", "softsaddle/saddle-lanczos-sm50"),
    ("tsopt/LJ38", "eon-svn/ts-lj38-lanczos-improved"),
)


def test_full_stock_rows_are_ranked_once():
    from optbench.board import build_board

    board = build_board(ROOT)
    assert _ranked_stock_errors(board) == []
    for problem_id, template in STOCK:
        ranked = [row for row in board["problems"][problem_id]["ranked"] if row["template"] == template]
        assert len(ranked) == 1, template
        assert ranked[0]["newest"] is True
        assert "-tuned" not in template
    neb = board["problems"]["neb/Pt-heptamer"]["ranked"]
    for template in ("eon/neb-pt-lbfgs", "eon-client/neb-pt-lbfgs"):
        row = next(item for item in neb if item["template"] == template)
        assert row["call_unit"] == "total"
        assert row["mean_per_image"] is not None
        assert "campaign" in row["source"] or "asv" in row["source"]
    eon = next(item for item in neb if item["template"] == "eon/neb-pt-lbfgs")
    assert eon["source"] == "campaign, asv"
    assert abs(eon["mean_per_image"] - eon["mean_calls"] / 8) < 1e-6


def test_a_full_stock_partial_fails_the_check():
    board = {
        "problems": {
            "minimization/lj38": {
                "id": "minimization/lj38",
                "ranked": [],
                "partial": [
                    {
                        "template": "pele/lj38-fire",
                        "comparable": True,
                        "n": 1000,
                        "full_n": 1000,
                        "partial_reason": "held",
                    }
                ],
            }
        }
    }
    errors = _ranked_stock_errors(board)
    assert errors
    assert "pele/lj38-fire" in errors[0]


def test_a_smoke_row_can_stay_partial():
    board = {
        "problems": {
            "minimization/lj38": {
                "id": "minimization/lj38",
                "ranked": [
                    {
                        "template": "pele/lj38-fire",
                        "comparable": True,
                        "n": 1000,
                        "full_n": 1000,
                    }
                ],
                "partial": [
                    {
                        "template": "pele/lj38-fire",
                        "comparable": True,
                        "n": 1,
                        "full_n": 1000,
                        "partial_reason": "smoke run; scored 1 of 1000 starts",
                    }
                ],
            }
        }
    }
    assert _ranked_stock_errors(board) == []


def test_archive_rows_stay_beside_the_rank_without_the_old_labels():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    html = problem_html(problem, oracle)
    assert ">eon-cg<" in html
    assert "pele-fire" in html
    assert "not reproduced" not in html
    assert "2014 archive" in html
    assert "legacy" not in html.lower()
