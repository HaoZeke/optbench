"""i-PI band geometry, input, and one live socket exchange."""

from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import geometry_archive, load_template, resolve_problem
from optbench.ipi_band import CODE_COMMIT, CODE_VERSION, linear_frames, render_xml, run_search
from optbench.pes import load_pairs

ROOT = Path(__file__).resolve().parents[1]


def test_linear_frames_keep_the_endpoints_and_fixed_atoms():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    product = np.array([[0.2, 0.0, 0.0], [1.4, 0.0, 0.0], [0.5, 0.0, 0.0]])
    free = np.array([True, True, False])
    box = np.array([10.0, 10.0, 10.0])
    frames = linear_frames(reactant, product, free, 3, box)
    assert len(frames) == 5
    assert np.allclose(frames[0], reactant)
    assert np.allclose(frames[-1][2], reactant[2])
    # The free product is the minimum image, not the raw coordinate 0.5.
    assert np.allclose(frames[-1][0], [0.2, 0.0, 0.0])
    assert frames[-1][1, 0] == pytest.approx(1.4)


def test_xml_uses_atomic_units_and_leaves_pbc_off():
    text = render_xml(
        address="obtest",
        nbeads=7,
        method="neb",
        optimizer="fire",
        climb=True,
        tangent="improved",
        spring_k=1.0,
        max_move=0.05,
        force_norm=0.001,
        fixed=[2, 5],
        stage="climb",
    )
    assert "nbeads='7'" in text
    assert "units='atomic_unit'" in text
    assert "pbc='false'" in text
    assert "1e-8" in text
    assert "<stage> climb </stage>" in text
    assert "<fixcom> false </fixcom>" in text
    assert "[ 2, 5 ]" in text
    assert "mode='fire'" in text
    assert "<use_climb> true </use_climb>" in text
    string = render_xml(
        address="obtest",
        nbeads=7,
        method="string",
        optimizer="bfgstrm",
        climb=False,
        tangent="improved",
        spring_k=1.0,
        max_move=0.2,
        force_norm=0.01,
        fixed=[],
        stage="string",
    )
    assert "mode='string'" in string
    assert "mode='bfgstrm'" in string
    assert "[ 0.2 ]" in string
    assert "<stage> string </stage>" in string
    assert "fixatoms" not in string


def test_templates_name_the_registry_commit():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "ipi")
    assert pinned["commit"] == CODE_COMMIT
    assert pinned["name"] == "i-PI"
    for name in ("ipi/neb", "ipi/ci-neb", "ipi/string", "ipi/ci-string"):
        template = load_template(ROOT, name)
        assert template["engine"] == "ipi-band"
        assert template["commit"] == pinned["commit"]
        assert template["code_version"] == CODE_VERSION
        assert template["code"] == "i-PI"
        assert template["tuning"] == "default"
        assert template["ipi"]["stage"] == ("climb" if template["ipi"]["climb"] else template["ipi"]["method"])


@pytest.mark.parametrize(
    "method,optimizer,climb,stage",
    [
        ("neb", "fire", False, "neb"),
        ("string", "bfgstrm", False, "string"),
        ("neb", "damped_bfgs", True, "climb"),
    ],
)
def test_socket_sees_the_reactant_and_stops_at_the_cap(method, optimizer, climb, stage):
    pytest.importorskip("ipi")
    problem = resolve_problem(ROOT, "double-ended-saddle-search/lj38")
    pairs = load_pairs(
        geometry_archive(ROOT, problem),
        problem["geometry"]["kind"],
        int(problem["sample"]["pairs"]),
    )
    pair = pairs[0]
    record = run_search(
        name=pair.name,
        reactant=pair.reactant,
        product=pair.product,
        free=pair.free,
        box=pair.box,
        kind="lj",
        n_interior=5,
        force_norm=0.001,
        cap=12,
        max_move=0.05,
        method=method,
        optimizer=optimizer,
        climb=climb,
        tangent="improved",
        spring_k=1.0,
        stage=stage,
    )
    assert record["force_calls"] <= 12
    assert record["force_calls"] > 0
    assert record["counter"] == "harness"
    assert record["failure"] in {"cap", "divergence", "wrong_stationary", "crash", None}
    assert record["reactant_maxabs"] is not None
    assert record["reactant_maxabs"] < 1e-6, record.get("log_tail") or record.get("crash")
    if record["failure"] == "crash":
        pytest.fail(record.get("log_tail") or record.get("crash"))
