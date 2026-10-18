"""eOn SVN templates stay on the r2549 pin and the problem grids."""

from pathlib import Path

import tomllib

from benchmarks.bench_eon_develop import lj_cut, morse_cluster
from benchmarks.bench_eon_svn import SPECS, _morse_fast
from optbench.campaign import _check_stop, load_template, resolve_problem
from optbench.potentials import lennard_jones

ROOT = Path(__file__).resolve().parents[1]


def test_c_morse_matches_the_develop_kernel():
    positions = []
    for i in range(12):
        positions.extend((0.4 * i, 0.15 * (i % 3), 0.2 * (i % 5)))
    energy, forces = morse_cluster(positions)
    other, fast = _morse_fast(positions)
    assert abs(energy - other) == 0.0
    assert fast == forces


def test_lj_cut_matches_the_noise_kernel():
    positions = [0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.2, 1.3, 0.4]
    energy, forces = lj_cut(positions, cutoff=40.0)
    other, gradient = lennard_jones(positions)
    assert abs(energy - other) < 1e-9
    assert len(forces) == len(gradient)
    for left, right in zip(forces, gradient, strict=True):
        assert abs(left + float(right)) < 1e-8


def test_svn_templates_match_the_problem_grids():
    by_template = {spec["template"]: spec for spec in SPECS}
    assert len(by_template) == 21
    expected = {
        "eon-svn/ts-lj38-dimer": 200,
        "eon-svn/ts-lj38-lanczos": 200,
        "eon-svn/ts-pt-dimer": 49,
        "eon-svn/ts-pt-lanczos": 49,
        "eon-svn/de-lj38-ci": 50,
        "eon-svn/de-lj38-neb": 50,
        "eon-svn/de-pt-ci": 59,
        "eon-svn/de-pt-neb": 59,
        "eon-svn/de-2ts-ci": 51,
        "eon-svn/de-2ts-neb": 51,
        "eon-svn/neb-pt-ci": 13,
        "eon-svn/neb-pt-neb": 13,
        "eon-svn/low-lj38-dimer": 200,
        "eon-svn/low-lj38-lanczos": 200,
        "eon-svn/lj38-noise-0": 1000,
        "eon-svn/lj38-noise-1e-4": 1000,
        "eon-svn/lj38-noise-1e-3": 1000,
        "eon-svn/lj38-noise-1e-2": 1000,
        "eon-svn/saddle-pt-dimer": 10,
        "eon-svn/saddle-pt-dimer-improved": 10,
        "eon-svn/saddle-pt-lanczos": 2,
    }
    assert set(by_template) == set(expected)
    for template_id, count in expected.items():
        spec = by_template[template_id]
        template = load_template(ROOT, template_id)
        assert template["commit"] == "r2549"
        assert template["code"] == "Eon svn"
        assert template["engine"] == "eon-client"
        assert template["problem"]
        _check_stop(resolve_problem(ROOT, template["problem"]), template)
        if spec["kind"] == "akmc":
            assert len(spec["campaign_seeds"]) == count
            assert spec["campaign_starts"] == ["pt-island"]
        else:
            assert spec["campaign_seeds"]
            assert len(spec["campaign_starts"]) == count
        optimizer = template["ini"].get("Optimizer") or {}
        if optimizer.get("opt_method") == "lbfgs":
            assert "LBFGS" not in template["ini"]
            assert optimizer["lbfgs_memory"] == 20
            assert "lbfgs_max_inverse_curvature" in optimizer
        assert "ci_mmf" not in template["ini"].get("Nudged Elastic Band", {})
        assert "energy_weighted" not in template["ini"].get("Nudged Elastic Band", {})


def test_registry_pins_r2549():
    data = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in data["code"] if item["id"] == "eon-svn")
    assert row["commit"] == "r2549"
    assert row["name"] == "Eon svn"
