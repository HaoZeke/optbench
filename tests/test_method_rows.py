"""A ranked row is one code, one version, and one stock method."""

from pathlib import Path

from optbench.board import build_board
from optbench.families import finite_difference_calls, method_name


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

_DOCUMENT = None


def _document():
    global _DOCUMENT
    if _DOCUMENT is None:
        _DOCUMENT = build_board(ROOT)
    return _DOCUMENT


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
