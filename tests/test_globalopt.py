"""Global optimisation problems share one force-call cap."""

from pathlib import Path

from optbench.problems import load_all

ROOT = Path(__file__).resolve().parents[1]


def test_globalopt_problems_declare_a_force_call_cap():
    """A missing cap lets each cell pick its own.

    Saddle search, accelerated MD, hTST, and mlopt omit the key on
    purpose. This gate covers globalopt only.
    """
    problems = load_all(ROOT / "problems")
    missing = [
        ident
        for ident, problem in sorted(problems.items())
        if problem["suite"] == "globalopt"
        and (problem.get("convergence") or {}).get("max_force_calls") in (None, "")
    ]
    assert missing == []
    caps = {
        ident: (problem.get("convergence") or {}).get("max_force_calls")
        for ident, problem in problems.items()
        if problem["suite"] == "globalopt"
    }
    assert caps == {
        "globalopt/lj100-binary": 2000000,
        "globalopt/lj150": 2000000,
        "globalopt/lj38": 2000000,
        "globalopt/lj75": 2000000,
        "globalopt/lj98": 2000000,
        "globalopt/tip4p10": 2000000,
        "globalopt/tip4p15": 2000000,
        "globalopt/tip4p20": 2000000,
    }
