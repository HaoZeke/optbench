"""Paper-baseline OPTIM saddle odata, displacement, and r2205 score."""

from pathlib import Path

import numpy as np

from optbench.connection import ProcessCatalog
from optbench.endpoints import EndpointArchive
from optbench.eon_displace import draw_search_start, search_seed
from optbench.optim_saddle import (
    KIND,
    PAPER_BFGSCONV,
    _score_one,
    climb_odata,
    climb_reasons,
    minimize_odata,
    optim_seed,
    paper_keyword_lines,
    read_vector_dump,
    search_displacement,
    write_vector,
)
from optbench.pes import MORSE_ALPHA, MORSE_CUTOFF, MORSE_D, MORSE_R0, evaluate
from optbench.rgsaddle_minmode import _stored
from optbench.surfaces import load_con, morse_eon

ROOT = Path(__file__).resolve().parents[1]
PREAMBLE = (ROOT / "templates/optim/preambles/saddle-pt.txt").read_text(encoding="utf-8")


def test_displacement_is_the_shared_generator():
    assert search_displacement is draw_search_start
    positions, fixed, cell = load_con(_stored(ROOT, "website/saddle-search/pt-island/pt-island.con"))
    listed = [0, 1, 2, 3, 4, 5, 6]
    drawn = search_displacement(
        positions,
        fixed,
        cell,
        (True, True, False),
        listed,
        rep=1,
        attempt=1,
        radius=3.3,
        magnitude=0.1,
    )
    again = draw_search_start(
        positions,
        fixed,
        cell,
        (True, True, False),
        listed,
        rep=1,
        attempt=1,
        radius=3.3,
        magnitude=0.1,
    )
    assert drawn["seed"] == search_seed(1, 1)
    assert drawn["epicenter"] in listed
    assert np.allclose(drawn["start"], again["start"])
    assert optim_seed(drawn["seed"]) == drawn["seed"] % 10000


def test_climb_odata_is_the_paper_preamble():
    keywords = paper_keyword_lines(PREAMBLE)
    assert "bfgsconv 4.3895D-5" in keywords
    assert "updates 100 " in keywords
    text = climb_odata(
        python="/usr/bin/python",
        hook=Path("/tmp/hook"),
        periodic=True,
        preamble=PREAMBLE,
        params="PARAMS 4.6488159D+00 1.0D+00 1.0D+00 1.0D+00 3.0D+00",
        seed=search_seed(1, 2),
        freeze=["FREEZE 8", "FREEZE 9"],
    )
    lines = text.splitlines()
    for keyword in keywords:
        assert keyword in lines
    assert f"RANSEED {optim_seed(search_seed(1, 2))}" in lines
    assert "FREEZE 8" in lines
    assert lines[-1] == "FREEZE 9"
    assert all(line.split()[:1] != ["MAXERISE"] for line in lines)
    assert all(line.split()[:1] != ["CONVERGE"] for line in lines)
    assert "BFGSTS" not in text.split()[0]


def test_minimize_odata_uses_the_paper_rms_token():
    text = minimize_odata(
        python="/usr/bin/python",
        hook=Path("/tmp/hook"),
        periodic=True,
        params="PARAMS 1.0D+00 1.0D+00 1.0D+00 1.0D+00 1.0D+00",
        seed=10001,
        freeze=["FREEZE 12"],
    )
    lines = text.splitlines()
    assert "BFGSMIN" in lines
    assert f"BFGSCONV {PAPER_BFGSCONV}" in lines
    assert "RANSEED 1" in lines
    assert "FREEZE 12" in lines
    assert "bfgsts" not in text.lower()


def test_freeze_lines_are_one_based_and_skip_the_island():
    from optbench.engines.optim_search import _freeze_lines

    _positions, fixed, _cell = load_con(_stored(ROOT, "website/saddle-search/pt-island/pt-island.con"))
    lines = _freeze_lines(~np.asarray(fixed, dtype=bool))
    frozen = {int(line.split()[1]) for line in lines}
    assert frozen.isdisjoint(set(range(1, 8)))
    assert len(lines) == int(np.count_nonzero(fixed))
    assert all(line.startswith("FREEZE ") for line in lines)


def test_vector_dump_keeps_the_last_complete_record(tmp_path: Path):
    path = tmp_path / "vector.dump"
    write_vector(path, np.arange(6, dtype=np.float64), eigenvalue=-1.0)
    first = path.read_text(encoding="utf-8")
    write_vector(path, np.arange(6, dtype=np.float64) + 10.0, eigenvalue=-2.0)
    path.write_text(first + path.read_text(encoding="utf-8") + "1.0 2.0\n", encoding="utf-8")
    vector = read_vector_dump(path, 6)
    assert vector is not None
    assert np.allclose(vector, np.arange(6) + 10.0)


def test_climb_reasons_match_the_socket_budget():
    assert climb_reasons(
        hang=True, error=None, finite=True, capped=False, calls=3, climb_cap=1000,
        force_norm=1e-4, force_tol=1e-3, index=1,
    ) == ["crash"]
    assert climb_reasons(
        hang=False, error=None, finite=True, capped=True, calls=1000, climb_cap=1000,
        force_norm=1e-4, force_tol=1e-3, index=1,
    ) == ["cap", "force_norm"]
    assert climb_reasons(
        hang=False, error=None, finite=True, capped=False, calls=10, climb_cap=1000,
        force_norm=1e-4, force_tol=1e-3, index=2,
    ) == ["hessian_index"]
    assert climb_reasons(
        hang=False, error=None, finite=True, capped=False, calls=10, climb_cap=1000,
        force_norm=1e-4, force_tol=1e-3, index=1,
    ) is None


def test_cap_stays_in_the_fraction_and_novelty_is_find_repeat(tmp_path: Path):
    free = np.array([0, 1])
    cell = np.array([20.0, 20.0, 30.0])
    archive = EndpointArchive(tmp_path / "saddle.endpoints.npz", 2)
    archive.reset(cell, free)
    catalog = ProcessCatalog(cell)
    minimum = np.zeros(6)
    cap = {
        "hang": False,
        "error": None,
        "finite": True,
        "capped": True,
        "climb": 1000,
        "force_norm": 0.2,
        "hessian_index": None,
        "endpoint_calls": 0,
        "verify_calls": 0,
        "converged": False,
        "plus": None,
        "minus": None,
        "energy": None,
        "coords": None,
        "lowest_eigenvalue": None,
    }
    record, stored = _score_one(
        catalog,
        cap,
        {"seed": 11, "epicenter": 2},
        minimum=minimum,
        min_energy=-10.0,
        free=free,
        fixed=np.array([False, False]),
        cell=cell,
        tolerance=0.1,
        climb_cap=1000,
        force_tol=1e-3,
        name="rep1-search1",
        archive=archive,
    )
    assert record["failure"] == "cap"
    assert catalog.not_connected == 0
    assert catalog.fraction_connected == 1.0
    assert record["endpoint_status"] == "not_minimized"
    assert "saddle_x" not in record
    assert "saddle_x" not in stored

    saddle = np.array([[0.2, 0.0, 0.4], [1.0, 0.0, 0.3]])
    plus = np.zeros((2, 3))
    minus = np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    linked = {
        "hang": False,
        "error": None,
        "finite": True,
        "capped": False,
        "climb": 40,
        "force_norm": 1e-4,
        "hessian_index": 1,
        "endpoint_calls": 30,
        "verify_calls": 4,
        "converged": True,
        "plus": plus,
        "minus": minus,
        "energy": -9.6,
        "coords": saddle,
        "lowest_eigenvalue": -0.2,
        "endpoint_error": None,
    }
    first, first_stored = _score_one(
        catalog,
        linked,
        {"seed": 12, "epicenter": 0},
        minimum=minimum,
        min_energy=-10.0,
        free=free,
        fixed=np.array([False, False]),
        cell=cell,
        tolerance=0.1,
        climb_cap=1000,
        force_tol=1e-3,
        name="rep1-search2",
        archive=archive,
    )
    second, _second_stored = _score_one(
        catalog,
        linked,
        {"seed": 13, "epicenter": 1},
        minimum=minimum,
        min_energy=-10.0,
        free=free,
        fixed=np.array([False, False]),
        cell=cell,
        tolerance=0.1,
        climb_cap=1000,
        force_tol=1e-3,
        name="rep1-search3",
        archive=archive,
    )
    assert first["success"] is True
    assert first["failure"] is None
    assert "saddle_x" in first_stored
    assert second["success"] is False
    assert second["reasons"] == ["duplicate"]
    assert catalog.unique == 1
    assert catalog.qualifying == 1
    assert catalog.not_connected == 0
    assert catalog.fraction_connected == 1.0


def test_socket_kind_is_morse_force():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(5, 3))
    box = np.array([12.0, 11.0, 15.0])
    energy, gradient = evaluate(KIND, coords, box)
    other_energy, other_gradient = morse_eon(
        coords,
        depth=MORSE_D,
        alpha=MORSE_ALPHA,
        r0=MORSE_R0,
        cutoff=MORSE_CUTOFF,
        cell=box,
    )
    assert energy == other_energy
    assert np.allclose(gradient, other_gradient)
