"""A ranked row is one code, one version, and one stock method."""

from pathlib import Path

from optbench.board import build_board, rank_violations
from optbench.families import family_of, finite_difference_calls, method_name


def test_fd_calls_use_one_force_call_per_coordinate():
    assert finite_difference_calls(1.0, 114) == 114.0
    assert finite_difference_calls(80.881, 114) == 80.881 * 114
    assert finite_difference_calls(0.0, None) == 0.0
    assert finite_difference_calls(1.0, None) is None


def test_a_parameter_record_is_not_the_method_name():
    blob = '{"counter":"harness","parameters":{"memory":100}}'
    assert method_name({"algorithm": "LBFGS"}, {"method": blob, "algorithm": "LBFGS"}) == "LBFGS"
    assert method_name({"method": "CI-NEB lj/cut", "algorithm": "CI-NEB"}, {"method": blob}) == "CI-NEB lj/cut"

ROOT = Path(__file__).resolve().parents[1]

EON = "eOn (develop 500350d9)"
METHODS = {
    "Lanczos improved": "min-mode",
    "Lanczos": "min-mode",
    "dimer": "min-mode",
    "energy-weighted CI-NEB": "band",
    "CI-NEB lj/cut": "band",
    "CI-NEB ljcluster": "band",
    "doubly nudged CI-NEB": "band",
    "OCI-NEB": "band",
}


_DOCUMENT = None


def _document():
    global _DOCUMENT
    if _DOCUMENT is None:
        _DOCUMENT = build_board(ROOT)
    return _DOCUMENT


def test_lj38_2ts_ranks_each_eon_method_and_keeps_tsase_split():
    document = _document()
    problem = document["problems"]["double-ended-saddle-search/lj38_2ts_separation"]
    labels = [row["label"] for row in problem["ranked"]]
    for method, family in METHODS.items():
        label = f"{EON} · {method}"
        assert label in labels
        row = next(item for item in problem["ranked"] if item["label"] == label)
        assert row["family"] == family
        assert row["origin"] == "harness"
        assert row["hessian_calls"] == 0.0
        assert row["fd_calls"] == 0.0
        assert family_of(method, suite="double-ended-saddle-search", kind=row["kind"]) == family
    assert "eOn (3.5.0) · CI-NEB lj/cut" in labels
    tsase = [row for row in problem["ranked"] if row["label"].startswith("TSASE (1.0) ·")]
    assert {row["label"] for row in tsase} >= {"TSASE (1.0) · CI-NEB", "TSASE (1.0) · NEB"}
    families = {row["family"] for row in problem["ranked"]}
    assert "band" in families and "min-mode" in families
    archived = [row for row in problem["ranked"] if row["origin"] == "archive"]
    assert archived
    assert all(row["par2_mean"] is None and row["published"] is not None for row in archived)
    assert rank_violations(document) == []


def test_adjusted_and_short_rows_stay_off_the_rank():
    document = _document()
    for problem in document["problems"].values():
        for row in problem["ranked"]:
            assert row["short"] is False
            tuning = str(row.get("tuning") or row.get("profile") or "").lower()
            assert tuning not in {"tuned", "adjusted"}
            template = str(row.get("template") or "")
            assert not template.endswith("-tuned")
            assert not template.endswith("-adjusted")
            assert "ranked false" not in str(row.get("short_reason") or "")
    withheld = document["withheld"]
    assert any(str(row.get("template") or "").endswith("-tuned") for row in withheld)


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


def test_full_stock_rows_stay_ranked():
    document = _document()
    assert rank_violations(document) == []
    for problem_id, template in STOCK:
        ranked = [row for row in document["problems"][problem_id]["ranked"] if row["template"] == template]
        assert len(ranked) == 1, template
    neb = document["problems"]["neb/Pt-heptamer"]["ranked"]
    for template in ("eon/neb-pt-lbfgs", "eon-client/neb-pt-lbfgs"):
        row = next(item for item in neb if item["template"] == template)
        assert row["call_unit"] == "total"
        assert row["mean_per_image"] is not None
    eon = next(item for item in neb if item["template"] == "eon/neb-pt-lbfgs")
    assert "asv" in str(eon["source"])
    assert abs(eon["mean_per_image"] - eon["mean_calls"] / 8) < 1e-6


def test_omitted_saddle_searches_finish_the_catalog():
    from optbench.board import _charge_omitted_saddle_starts

    problem = {"id": "saddle-search/pt-island", "sample": {"searches": 20000}}
    two = [
        {"force_calls": 10.0, "success": True, "start": "0"},
        {"force_calls": 12.0, "success": False, "failure": "cap", "start": "1"},
    ]
    charged = _charge_omitted_saddle_starts(two, problem, 10000, skip=False)
    assert len(charged) == 20000
    assert charged[-1]["failure"] == "cap"
    assert charged[-1]["force_calls"] == 10000.0
    assert len(_charge_omitted_saddle_starts(two, problem, 10000, skip=True)) == 2
    sixteen = [{"force_calls": 1.0, "success": True, "start": str(index)} for index in range(16)]
    assert len(_charge_omitted_saddle_starts(sixteen, problem, 5000, skip=False)) == 20000
    three = [{"force_calls": 1.0, "success": True, "start": str(index)} for index in range(3)]
    assert len(_charge_omitted_saddle_starts(three, problem, 5000, skip=False)) == 3


def test_an_earlier_rgmin_ref_leaves_the_rank_when_the_child_is_present():
    from optbench.board import _demote_contained_rgmin

    rows = [
        {
            "code": "rgmin",
            "code_version": "0.4.1",
            "commit": "9a2e7ff1aaaa",
            "problem": "minimization/morse-bulk",
            "template": "rgmin/morse-dogleg",
            "comparable": True,
            "short": False,
            "tuning": "default",
        },
        {
            "code": "rgmin",
            "code_version": "0.4.1",
            "commit": "f15726f1bbbb",
            "problem": "minimization/morse-bulk",
            "template": "rgmin/morse-dogleg",
            "comparable": True,
            "short": False,
            "tuning": "default",
        },
        {
            "code": "rgmin",
            "code_version": "0.4.1",
            "commit": "9a2e7ff1cccc",
            "problem": "minimization/lj38-noise",
            "template": "noise/rgmin-dogleg-s0",
            "comparable": True,
            "short": False,
            "tuning": "default",
        },
    ]
    _demote_contained_rgmin(rows)
    assert rows[0]["short"] is True
    assert rows[0]["short_reason"] == "an earlier ref of the same method"
    assert rows[1]["short"] is False
    assert rows[2]["short"] is False
