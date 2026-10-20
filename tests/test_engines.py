"""The saddle-code runner table.

Importing the table does not load libartn or TSASE. pyKMC and OPTIM
refuse with a concrete reason and write no file.
"""

from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.engines import ENGINES, find, reasons
from optbench.engines.common import band_settings, cached_pairs
from optbench.engines.optim_search import (
    SHA256,
    available as optim_available,
    clstring_calls,
    clstring_converged,
    stock_converged,
    stock_odata,
)
from optbench.engines.pykmc_search import available as pykmc_available

ROOT = Path(__file__).resolve().parents[1]


def test_named_engines_are_in_the_table():
    assert list(ENGINES) == [
        "partn",
        "tsase",
        "optim",
        "pykmc",
        "lammps",
        "dlfind",
        "compass",
        "pysisyphus",
    ]
    assert find("eon") is None
    assert find("rgsaddle-band") is None
    found = reasons()
    assert found["pykmc"]
    assert "sourceforge" in found["pykmc"]
    assert "1.3" in found["pykmc"]
    assert "LBOMD" in found["pykmc"]
    assert optim_available() == found["optim"]
    if found["optim"] is None:
        from optbench.engines.optim_search import binary_path

        assert binary_path().is_file()
    else:
        assert SHA256 in found["optim"]
    assert pykmc_available() == found["pykmc"]
    assert found["partn"] is None or "libartn" in found["partn"]
    assert found["tsase"] is None or "ssneb.py" in found["tsase"]
    assert found["lammps"] is None or "liblammps" in found["lammps"]
    assert found["dlfind"] is None or "libdlfind" in found["dlfind"]
    assert found["compass"] is None or "compass.py" in found["compass"]
    assert found["pysisyphus"] is None or "a4ce10dd" in found["pysisyphus"]


def test_new_templates_match_a_problem_stop():
    expected = {
        "partn/artn",
        "partn/mode-lj38",
        "partn/saddle-pt",
        "partn/ts-lj38",
        "partn/ts-pt",
        "tsase/ci-neb",
        "tsase/dimer-lj38",
        "tsase/dimer-pt",
        "tsase/lbfgs-lj38",
        "tsase/lbfgs-morse",
        "tsase/mdmin-lj38",
        "tsase/mode-lj38",
        "tsase/neb",
        "pykmc/lattice",
        "optim/unhooked",
        "optim/lbfgs-lj38",
        "optim/lbfgs-morse",
        "optim/de-pt-cl-string-latest",
        "lammps/cg-lj38",
        "lammps/cg-morse",
        "lammps/cg-lj38-noise-s0",
        "lammps/cg-lj38-noise-s1em4",
        "lammps/cg-lj38-noise-s1em3",
        "lammps/cg-lj38-noise-s1em2",
        "lammps/sd-lj38",
        "lammps/sd-morse",
        "lammps/sd-lj38-noise-s0",
        "lammps/sd-lj38-noise-s1em4",
        "lammps/sd-lj38-noise-s1em3",
        "lammps/sd-lj38-noise-s1em2",
        "lammps/hftn-lj38",
        "lammps/hftn-morse",
        "lammps/hftn-lj38-noise-s0",
        "lammps/hftn-lj38-noise-s1em4",
        "lammps/hftn-lj38-noise-s1em3",
        "lammps/hftn-lj38-noise-s1em2",
        "lammps/quickmin-lj38",
        "lammps/quickmin-morse",
        "lammps/quickmin-lj38-noise-s0",
        "lammps/quickmin-lj38-noise-s1em4",
        "lammps/quickmin-lj38-noise-s1em3",
        "lammps/quickmin-lj38-noise-s1em2",
        "lammps/fire-lj38",
        "lammps/fire-morse",
        "lammps/fire-lj38-noise-s0",
        "lammps/fire-lj38-noise-s1em4",
        "lammps/fire-lj38-noise-s1em3",
        "lammps/fire-lj38-noise-s1em2",
        "dlfind/lbfgs-lj38",
        "dlfind/lbfgs-morse",
        "dlfind/sd-lj38",
        "dlfind/sd-morse",
        "dlfind/cg-lj38",
        "dlfind/cg-morse",
        "dlfind/lbfgs-lj38-noise-s0",
        "dlfind/lbfgs-lj38-noise-s1em4",
        "dlfind/lbfgs-lj38-noise-s1em3",
        "dlfind/lbfgs-lj38-noise-s1em2",
        "dlfind/sd-lj38-noise-s0",
        "dlfind/sd-lj38-noise-s1em4",
        "dlfind/sd-lj38-noise-s1em3",
        "dlfind/sd-lj38-noise-s1em2",
        "dlfind/cg-lj38-noise-s0",
        "dlfind/cg-lj38-noise-s1em4",
        "dlfind/cg-lj38-noise-s1em3",
        "dlfind/cg-lj38-noise-s1em2",
        "compass/sqnm",
        "pysisyphus/lbfgs-lj38",
        "pysisyphus/cg-lj38",
        "pysisyphus/sd-lj38",
        "pysisyphus/lanczos-lj38",
    }
    found = []
    for template in list_templates(ROOT):
        if template["engine"] not in ENGINES:
            continue
        found.append(template["id"])
        problem_name = template.get("problem", "neb/Pt-heptamer")
        _check_stop(resolve_problem(ROOT, problem_name), template)
        assert template["commit"]
        assert template["code"]
    assert set(found) == expected


def test_pykmc_and_optim_write_nothing(tmp_path: Path):
    output = tmp_path / "benchmark.dat"
    with pytest.raises(RuntimeError, match="LBOMD"):
        run(ROOT, "neb/Pt-heptamer", "pykmc/lattice", output=output, workers=1)
    with pytest.raises(RuntimeError, match="sha256"):
        run(ROOT, "neb/Pt-heptamer", "optim/unhooked", output=output, workers=1)
    assert not output.exists()
    assert not output.with_suffix(".starts.jsonl").exists()


def test_clstring_keeps_the_last_energy_gradient_tally():
    log = (
        " OPTIM> # of energy+gradient calls=                 6 time=           0.01 %= 59.9\n"
        " Converged to TS\n"
        " OPTIM> # of energy+gradient calls=               211 time=           0.27 %= 78.8\n"
    )
    assert clstring_calls(log) == 211
    assert clstring_converged(log) is True
    assert clstring_calls("no tally\n") is None
    assert clstring_converged("OPTIM> not a transition state\n") is False


def test_stock_clstring_limits_follow_the_problem():
    problem = resolve_problem(ROOT, "double-ended-saddle-search/pt-island")
    settings = band_settings(problem)
    pair = cached_pairs(ROOT, problem, settings)["product_0"]
    odata, finish = stock_odata(
        pair.reactant,
        pair.free,
        pair.box,
        pair.product,
        images=settings["n_interior"],
        steps=settings["cap"],
    )
    head = odata.splitlines()[:6]
    assert head[:5] == [
        "BULK",
        "CLSTRING",
        f"GROWSTRING {settings['n_interior']} {settings['cap']}",
        f"STEPS {settings['cap']}",
        "DESMDEBUG",
    ]
    assert head[5].startswith("PARAMS ")
    assert "INE_NEW" not in odata
    assert "BFGSTS" not in odata
    assert "OPTIMIZETS" not in odata
    assert finish.count("\n") == len(pair.reactant)
    assert stock_converged("SUCCESS! - the last image converged!\n")
    assert stock_converged(" Converged to TS\n")
    assert not stock_converged("Number of final evolution steps exceeded maximum")


def test_stock_clstring_refuses_without_its_binary(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPTIM_LATEST_BIN", str(tmp_path / "missing-OPTIM"))
    output = tmp_path / "benchmark.dat"
    with pytest.raises(RuntimeError, match="738fafe3"):
        run(
            ROOT,
            "double-ended-saddle-search/pt-island",
            "optim/de-pt-cl-string-latest",
            output=output,
            workers=1,
        )
    assert not output.exists()


def test_partn_engine_is_dispatched(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem_name, template_name, *, output, workers):
        captured["problem"] = problem_name
        captured["template"] = template_name
        captured["workers"] = workers
        return {
            "force_calls": 3,
            "nfailed": 0,
            "starts": 1,
            "template": template_name,
            "problem": problem_name,
        }

    monkeypatch.setattr("optbench.engines.partn.run", fake)
    summary = run(
        ROOT,
        "neb/Pt-heptamer",
        "partn/artn",
        output=tmp_path / "benchmark.dat",
        workers=2,
    )
    assert captured == {"problem": "neb/Pt-heptamer", "template": "partn/artn", "workers": 2}
    assert summary["force_calls"] == 3
