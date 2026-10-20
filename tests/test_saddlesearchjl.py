"""SaddleSearch.jl templates and the quadratic saddle."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "47b4f35953bb3fc0f0726c171adca8f5b628af5e"


def _available() -> bool:
    julia = os.environ.get("JULIA") or str(Path.home() / "julia" / "bin" / "julia")
    source = os.environ.get("SADDLESEARCHJL_SRC") or str(Path.home() / "src" / "SaddleSearch.jl")
    return Path(julia).is_file() and (Path(source) / "src" / "SaddleSearch.jl").is_file()


def test_hex_line_preserves_float_bits():
    from optbench.saddlesearchjl import _floats_from_hex_line, _hex, _hex_line

    values = np.array([1.0, -0.0, np.inf, np.nan, -1.5, 1e-320], dtype=np.float64)
    line = _hex_line(values)
    assert line == " ".join(_hex(float(value)) for value in values)
    back = _floats_from_hex_line(line, int(values.size))
    assert np.array_equal(values.view(np.uint64), back.view(np.uint64))


def test_catalog_partial_drops_only_a_torn_last_line(tmp_path):
    from optbench.saddlesearchjl import _load_catalog_partial

    path = tmp_path / "rep.partial.jsonl"
    path.write_text(
        '{"start": "rep1-search1"}\n{"start": "rep1-search2"}\n{"start":\n',
        encoding="utf-8",
    )
    rows = _load_catalog_partial(path)
    assert [row["start"] for row in rows] == ["rep1-search1", "rep1-search2"]
    path.write_text('{"start": "ok"}\n{"start":\n{"start": "later"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _load_catalog_partial(path)


def test_templates_pin_the_source_and_the_problem_cap():
    found = []
    for template in list_templates(ROOT):
        if not str(template["id"]).startswith("saddlesearchjl/"):
            continue
        found.append(template["id"])
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        assert template["commit"] == COMMIT
        assert template["engine"] == "saddlesearchjl"
        assert template["code"] == "saddlesearchjl"
        assert template["code_version"] == "0.0.2"
        if template["problem"] == "saddle-search/pt-island":
            assert cap == 1000
            assert int((template.get("search") or {})["max_searches"]) == 20000
        else:
            assert cap == 10000
    assert "saddlesearchjl/lj38-static" in found
    assert "saddlesearchjl/lj38-superlinear" in found
    assert "saddlesearchjl/pt-ode" in found
    assert "saddlesearchjl/mode-bb" in found
    assert "saddlesearchjl/saddle-dimer" not in found
    assert "saddlesearchjl/saddle-ode" in found
    assert len(found) == 15


@pytest.mark.skipif(not _available(), reason="Julia or SaddleSearch.jl is not on this host")
def test_dimers_find_the_quadratic_saddle():
    from optbench.saddlesearchjl import JuliaSession, execute

    def evaluate(coords):
        x, y, z = (float(value) for value in coords)
        return -x * x + y * y + 2.0 * z * z, (-2.0 * x, 2.0 * y, 4.0 * z)

    searches = [
        {"method": "static", "a_trans": 0.002, "a_rot": 0.002},
        {"method": "bb", "a0_trans": 0.001, "a0_rot": 0.001},
        {"method": "superlinear", "maximum_translation": 0.1, "max_num_rot": 3},
        {"method": "ode"},
    ]
    session = JuliaSession.start()
    try:
        for search in searches:
            spec = {
                "name": search["method"],
                "positions": np.array([[0.4, -0.3, 0.2]], dtype=np.float64),
                "fixed": np.array([False]),
                "cell": np.zeros(3),
                "evaluate": evaluate,
                "template": {"search": search},
                "task": "saddle",
                "mode_seed": "gradient",
                "cap": 4000,
                "force_tol": 1e-3,
                "seed": "0",
            }
            record = execute(spec, session)
            assert record["success"], record
            assert record["hessian_index"] == 1
            assert record["counter"] == "harness"
            assert record["force_calls"] < 4000
            assert np.linalg.norm(record["x"]) < 1e-2
        mode = {
            "name": "mode",
            "x": np.array([0.4, -0.3, 0.2], dtype=np.float64),
            "evaluate": evaluate,
            "template": {"search": {"method": "bb", "a0_trans": 0.001, "a0_rot": 0.001}},
            "task": "mode",
            "mode_seed": "gradient",
            "reference_mode": np.array([1.0, 0.0, 0.0]),
            "fixed": np.array([False]),
            "cap": 2000,
            "force_tol": 1e-3,
            "overlap": 0.99,
            "seed": "0",
        }
        found = execute(mode, session)
        assert found["success"], found
        assert found["overlap"] >= 0.99
        assert found["force_calls"] < 2000
    finally:
        session.close()
