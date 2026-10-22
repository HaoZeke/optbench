"""Templates and the minimum Hessian for eOn develop bbe4ffba.

These tests do not start eonclient.
"""

from pathlib import Path

import numpy as np

from benchmarks.bench_eon_develop import (
    _commit_matches,
    _lj_cluster_hessian,
    _morse_pt_hessian,
    campaign_starts,
    minimum_stationary,
    morse_cluster,
)
from optbench.campaign import _check_stop, load_template, resolve_problem
from optbench.lj import lj_cluster

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "bbe4ffbad32b8a479bf171cd7ff1497e363940e3"
NAMES = [
    "eon/develop-bbe4ffba-lj38-lbfgs",
    "eon/develop-bbe4ffba-lj38-fire",
    "eon/develop-bbe4ffba-lj38-cg",
    "eon/develop-bbe4ffba-lj38-sd",
    "eon/develop-bbe4ffba-lj38-qm",
    "eon/develop-bbe4ffba-morse-lbfgs",
    "eon/develop-bbe4ffba-neb-pt-lbfgs",
    "eon/develop-bbe4ffba-ts-pt-lanczos",
]
def test_templates_pin_the_merge_and_leave_the_backend_unset():
    kinds = {
        "eon/develop-bbe4ffba-lj38-lbfgs": "min-lj38",
        "eon/develop-bbe4ffba-lj38-fire": "min-lj38",
        "eon/develop-bbe4ffba-lj38-cg": "min-lj38",
        "eon/develop-bbe4ffba-lj38-sd": "min-lj38",
        "eon/develop-bbe4ffba-lj38-qm": "min-lj38",
        "eon/develop-bbe4ffba-morse-lbfgs": "min-morse",
        "eon/develop-bbe4ffba-neb-pt-lbfgs": "neb-pt",
        "eon/develop-bbe4ffba-ts-pt-lanczos": "ts-pt",
    }
    methods = {
        "eon/develop-bbe4ffba-lj38-lbfgs": "lbfgs",
        "eon/develop-bbe4ffba-lj38-fire": "fire",
        "eon/develop-bbe4ffba-lj38-cg": "cg",
        "eon/develop-bbe4ffba-lj38-sd": "sd",
        "eon/develop-bbe4ffba-lj38-qm": "qm",
        "eon/develop-bbe4ffba-morse-lbfgs": "lbfgs",
        "eon/develop-bbe4ffba-neb-pt-lbfgs": "lbfgs",
        "eon/develop-bbe4ffba-ts-pt-lanczos": "cg",
    }
    for name in NAMES:
        template = load_template(ROOT, name)
        assert template["commit"] == COMMIT
        assert template["code"] == "eOn"
        assert template["code_version"] == "(develop bbe4ffba)"
        assert template["kind"] == kinds[name]
        assert template["reducer"] == "harness"
        assert template["failure_cap"] == 10000
        assert template["ini"]["Optimizer"]["opt_method"] == methods[name]
        assert "opt_backend" not in template["ini"]["Optimizer"]
        assert _check_stop(resolve_problem(ROOT, template["problem"]), template) == 10000
    lbfgs = load_template(ROOT, "eon/develop-bbe4ffba-lj38-lbfgs")
    assert lbfgs["ini"]["LBFGS"]["lbfgs_accept"] == "energy"
    assert lbfgs["ini"]["LBFGS"]["lbfgs_memory"] == 20
    sd = load_template(ROOT, "eon/develop-bbe4ffba-lj38-sd")
    assert sd["ini"]["SD"]["sd_alpha"] == 0.001
    qm = load_template(ROOT, "eon/develop-bbe4ffba-lj38-qm")
    assert qm["ini"]["QuickMin"]["time_step"] == 0.01
    neb = load_template(ROOT, "eon/develop-bbe4ffba-neb-pt-lbfgs")
    assert "path_initialization" not in neb["ini"]["Nudged Elastic Band"]
    assert len(campaign_starts("min-lj38")) == 1000
    assert len(campaign_starts("min-morse")) == 100
    assert campaign_starts("neb-pt") == [f"min{i:02d}" for i in range(1, 14)]
    assert len(campaign_starts("ts-pt")) == 49


def test_version_token_matches_the_full_sha():
    assert _commit_matches("bbe4ffba", COMMIT)
    assert _commit_matches("7095b23", "7095b23ac17d49f434a5e08ccb072897ceca698e")
    assert _commit_matches("7095b23a", "7095b23ac17d49f434a5e08ccb072897ceca698e")
    assert not _commit_matches("bbe4ffba", "7095b23ac17d49f434a5e08ccb072897ceca698e")


def _gradient(potential: str, coords: np.ndarray, box) -> np.ndarray:
    flat = [float(value) for value in coords.reshape(-1)]
    if potential == "lj":
        _energy, forces = lj_cluster(flat)
    else:
        _energy, forces = morse_cluster(flat, None, box)
    return -np.asarray(forces, dtype=np.float64)


def _finite_difference(potential: str, coords: np.ndarray, box, step: float = 1e-6) -> np.ndarray:
    dim = coords.size
    matrix = np.zeros((dim, dim))
    for index in range(dim):
        plus = coords.copy()
        minus = coords.copy()
        plus.reshape(-1)[index] += step
        minus.reshape(-1)[index] -= step
        matrix[:, index] = (_gradient(potential, plus, box) - _gradient(potential, minus, box)) / (2.0 * step)
    return 0.5 * (matrix + matrix.T)


def test_lj_cluster_hessian_matches_forces_and_the_dimer_is_a_minimum():
    distance = 2.0 ** (1.0 / 6.0)
    coords = np.array([0.0, 0.0, 0.0, distance, 0.0, 0.0])
    analytic = _lj_cluster_hessian(coords)
    numeric = _finite_difference("lj", coords, None)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-5)
    lines = [
        "free",
        "",
        "100.0 100.0 100.0",
        "90.0 90.0 90.0",
        "",
        "",
        "1",
        "2",
        "1.0",
        "H",
        "Coordinates of Component 1",
        f"0.0 0.0 0.0 0 0",
        f"{distance:.16f} 0.0 0.0 0 0",
    ]
    found = minimum_stationary("ljcluster", "\n".join(lines) + "\n", 0.01)
    assert found["force_norm"] < 1e-8
    assert found["hessian_index"] == 0


def test_morse_pt_hessian_matches_forces():
    coords = np.array(
        [
            [0.2, 0.1, 0.0],
            [2.4, 0.3, 0.1],
            [1.1, 2.2, 0.2],
        ]
    )
    box = [8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]
    analytic = _morse_pt_hessian(coords.reshape(-1), box)
    numeric = _finite_difference("morse", coords.reshape(-1), box)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)
