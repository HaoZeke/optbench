"""The aenet mlopt templates use the problem epoch caps."""

from __future__ import annotations

import hashlib
from pathlib import Path

import tomllib

from optbench.board import stated_full_n
from optbench.campaign import _check_stop, geometry_archive, list_templates, resolve_problem

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "8f35990f5e3dd6cb296b953448cb11694d0cff3d"


def _aenet():
    return [template for template in list_templates(ROOT) if template.get("engine") == "aenet"]


def test_twenty_templates_use_the_epoch_caps():
    templates = _aenet()
    assert len(templates) == 20
    ids = {template["id"] for template in templates}
    assert "aenet/emt-pt38-energy-bfgs" in ids
    assert "aenet/c2h4-force-ekf" in ids
    for template in templates:
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        target = template["aenet"]["target"]
        assert cap == (10000 if target == "energy" else 30000)
        assert stated_full_n(ROOT, problem) == 50
        assert template["commit"] == COMMIT
        assert template["code_version"] == "2.0.5"
        assert (ROOT / template["aenet"]["fp_parameters"]).is_file()


def test_registry_pins_the_built_revision():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "aenet")
    assert row["commit"] == COMMIT
    assert row["ref"] == "master"
    assert "2.0.5" in row["note"]


def test_force_fingerprints_match_the_energy_file():
    pairs = (
        (
            "archive/entries/mlopt/emt-pt38/pyamff-lbfgs10-7-fcoeff0/fpParas.dat",
            "archive/entries/mlopt/emt-pt38/pyamff-lbfgs10-7-fcoeff0_1/fpParas.dat",
        ),
        (
            "archive/entries/mlopt/C2H4/pyamff-lbfgs10-7-fcoeff0/fpParas.dat",
            "archive/entries/mlopt/C2H4/pyamff-lbfgs10-7-fcoeff0_1/fpParas.dat",
        ),
    )
    for energy, force in pairs:
        left = hashlib.sha256((ROOT / energy).read_bytes()).hexdigest()
        right = hashlib.sha256((ROOT / force).read_bytes()).hexdigest()
        assert left == right


def test_both_trajectories_are_in_the_manifest():
    platinum = resolve_problem(ROOT, "mlopt/emt-pt38")
    ethylene = resolve_problem(ROOT, "mlopt/C2H4")
    assert geometry_archive(ROOT, platinum).is_file()
    assert geometry_archive(ROOT, ethylene).is_file()
    assert ethylene["contract"] == "partial"
