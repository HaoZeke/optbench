"""Stock latest-release templates keep the constructor step."""

from pathlib import Path

from optbench.campaign import _check_stop, load_template, resolve_problem

ROOT = Path(__file__).resolve().parents[1]


def test_basinhopping_is_the_constructor():
    template = load_template(ROOT, "pele/lj38-basinhopping")
    assert template["pele"] == {"method": "basinhopping"}
    assert template["tuning"] == "default"
    assert template["code"] == "pele"
    assert template["code_version"] == "0.1.2"
    assert int(template["failure_cap"]) > 2533900
    _check_stop(resolve_problem(ROOT, template["problem"]), template)
    from benchmarks.bench_pele import _specs

    assert all(spec["template_id"] != "pele/lj38-basinhopping" for spec in _specs())


def test_morse_sd_omits_the_step():
    expected = {
        "eon-svn/morse-sd-harness": ("EON", "(SVN r2549)", "r2549", "svn"),
        "eon-develop/morse-sd-harness": (
            "eOn",
            "(develop a7aa57472fb7)",
            "a7aa57472fb7d823953fd36b78cd755fa4842d30",
            "develop",
        ),
    }
    for name, (code, version, commit, family) in expected.items():
        template = load_template(ROOT, name)
        optimizer = template["ini"]["Optimizer"]
        assert optimizer["opt_method"] == "sd"
        assert float(optimizer["converged_force"]) == 0.001
        assert int(optimizer["max_iterations"]) == 10000
        for key in ("sd_alpha", "sd_twopoint", "max_move"):
            assert key not in optimizer
        assert template["code"] == code
        assert template["code_version"] == version
        assert template["commit"] == commit
        assert template["client_family"] == family
        assert template["tuning"] == "default"
        _check_stop(resolve_problem(ROOT, template["problem"]), template)


def test_barrier_window_waits_for_every_reference_barrier():
    from optbench.runners.eon_htst import _reference_ids, _windows

    table = "\n".join(
        [
            "proc # saddle energy prefactor product product energy product prefactor barrier rate repeats",
            "0 -1 1 -1 -1 1 0.74619 1 0",
            "1 -1 1 -1 -1 1 0.23285 1 0",
        ]
    )
    found, missed = _reference_ids(table)
    assert found == [1]
    assert missed == [0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372]
    rows = [
        {"sad": 100, "barrier": 0.74619, "status": "good", "process_id": 0},
        {"sad": 100, "barrier": 0.23281, "status": "good", "process_id": 1},
        {"sad": 100, "barrier": 0.0, "status": "Not Connected", "process_id": None},
    ]
    barriers, rates = _windows(rows, [10.0, 20.0, 30.0], found, missed)
    assert barriers == []
    assert rates == []
    complete = [
        {"sad": 1, "barrier": barrier, "status": "good", "process_id": index}
        for index, barrier in enumerate([0.233, 0.233, 0.233, 0.233, 0.372, 0.372, 0.372, 0.372])
    ]
    closed, _rates = _windows(complete, [1.0] * 8, list(range(8)), [])
    assert len(closed) == 1
    assert closed[0]["jobs"] == 8
    assert closed[0]["force_calls"] == 8


def test_htst_stock_sets_only_the_algorithm():
    for name in ("eon-svn/htst-al-dimer-cg-stock", "eon-develop/htst-al-dimer-cg-stock"):
        template = load_template(ROOT, name)
        assert set(template["ini"]) == {"Main", "Saddle Search", "Dimer"}
        dimer = template["ini"]["Dimer"]
        assert dimer == {"rotations_max": 50, "converged_angle": 1, "opt_method": "cg"}
        assert template["algorithm"] == "Dimer-CG rotation: max=50 tol=1"
        assert template["tuning"] == "default"
        assert "partial" not in template
        _check_stop(resolve_problem(ROOT, template["problem"]), template)
