"""pyGSM template pin and the dissociation gate."""

import tomllib
from pathlib import Path

import numpy as np

from optbench.campaign import load_template
from optbench.pygsm_band import _STEP_BUDGET, _both_signs, _sum_generators, largest_nearest_neighbour

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1b4d5a5557a61adc6a3df7405d6b7255b1fa4865"


def test_asv_runtime_requires_the_campaign_scipy():
    import inspect

    from benchmarks import bench_pygsm

    source = inspect.getsource(bench_pygsm.ensure_pygsm)
    assert "scipy==1.18.1" in source
    assert "threadpoolctl==3.7.0" in source
    assert bench_pygsm._runtime_ready()


def test_benchmark_classes_publish_the_problem_cap():
    from optbench.campaign import resolve_problem

    from benchmarks.bench_pygsm import PygsmLj38De, PygsmLj38TwoTs, problem_cap

    assert PygsmLj38De.problem_name == "double-ended-saddle-search/lj38"
    assert PygsmLj38De.template_id == "pygsm/de-gsm"
    assert PygsmLj38De.cap == 2500
    assert PygsmLj38De.timeout == 1800
    assert PygsmLj38De.params[1] == ["pair_00"]
    two_ts = "double-ended-saddle-search/lj38_2ts_separation"
    stated = int(resolve_problem(ROOT, two_ts)["convergence"]["max_force_calls"])
    assert stated == 250000
    assert problem_cap(two_ts) == stated
    assert PygsmLj38TwoTs.problem_name == two_ts
    assert PygsmLj38TwoTs.template_id == "pygsm/de-gsm"
    assert PygsmLj38TwoTs.cap == stated
    assert PygsmLj38TwoTs.timeout == 7200
    assert PygsmLj38TwoTs.params[1] == ["pair_00"]


def test_template_matches_the_registry_pin():
    from optbench.board import ranked_cells

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pygsm")
    template = load_template(ROOT, "pygsm/de-gsm")
    assert pinned["commit"] == COMMIT
    note = pinned["note"]
    lowered = note.lower()
    assert "ASELoT" in note
    assert "eV" in note
    assert "hybrid_coord_idx_file" in note
    assert "EV_PER_AU" in note
    assert "newMakePrimitives" in note
    assert "CartesianX" in note
    assert "write_xyzs_w_comments" in note
    assert "dirty" not in lowered
    assert "contact-distance" not in lowered
    assert "reproduced" not in lowered
    assert "legacy" not in lowered
    assert "current" not in lowered
    assert "de-gsm-climbclamp" not in note
    assert template["commit"] == pinned["commit"]
    assert template["code"] == "pygsm"
    assert template["engine"] == "pygsm-band"
    assert template["algorithm"] == "DE-GSM"
    assert template["tuning"] == "default"
    assert template["code_version"] == "1.0.0+692.g1b4d5a5"
    assert "dirty" not in template["code_version"]
    assert template["gsm"]["coordinate"] == "hybrid"
    assert template["gsm"]["nnodes"] == 11
    assert template["gsm"]["dmax"] == 0.1
    assert template["gsm"]["add_node_tol"] == 0.01
    assert "max_opt_steps" not in template["gsm"]
    assert _STEP_BUDGET >= 250000
    assert template["gsm"]["climb_clamp"] is False
    assert template["gsm"]["endpoints"] == "fixed"
    assert "problem" not in template
    assert not (ROOT / "templates" / "pygsm" / "de-gsm-climbclamp.toml").exists()
    cells = dict(
        ranked_cells(
            {
                "template": template["id"],
                "code": template["code"],
                "code_version": template["code_version"],
                "commit": template["commit"],
                "source": "campaign",
                "par2_mean": 1.0,
                "success_rate": 1.0,
                "wilson_low": 0.0,
                "wilson_high": 1.0,
                "median_calls": 1.0,
                "mean_calls": 1.0,
                "failures": {},
                "n": 50,
            },
            False,
        )
    )
    assert cells["template"] == "pygsm/de-gsm"
    assert cells["version"] == f"pygsm {template['code_version']}"
    assert "1b4d5a5" in cells["version"]
    assert cells["commit"] == COMMIT[:12]


def test_a_broken_delocalized_basis_is_divergence():
    from optbench.pygsm_band import _numeric_failure

    assert _numeric_failure(RuntimeError("error in orthonormality"))
    assert _numeric_failure(np.linalg.LinAlgError("Eigenvalues did not converge"))
    assert _numeric_failure(ValueError("array must not contain infs or NaNs"))
    assert _numeric_failure(RuntimeError("missing import")) is False


def test_largest_nearest_neighbour_is_the_worst_contact():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
    assert largest_nearest_neighbour(coords) == 3.0


def test_a_separated_cluster_is_divergence_before_the_hessian():
    from optbench.counter import HarnessCounter
    from optbench.pygsm_band import _judge

    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64)
    record = _judge(
        "pair",
        HarnessCounter(cache=True, cap=10),
        {"energy": -1.0e-5, "gradient": np.zeros(9), "coords": coords},
        force_norm=0.001,
        cap=10,
        project_rigid=False,
        crashed="",
        potential="lj",
    )
    assert record["success"] is False
    assert record["failure"] == "divergence"
    assert record["hessian_index"] is None
    assert record["max_nn"] == 3.0


def test_two_compact_fragments_are_divergence_under_the_cluster_rule():
    from optbench.counter import HarnessCounter
    from optbench.pygsm_band import _judge

    coords = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [10.0, 0.0, 0.0], [11.1, 0.0, 0.0]],
        dtype=np.float64,
    )
    record = _judge(
        "pair",
        HarnessCounter(cache=True, cap=10),
        {"energy": -2.0, "gradient": np.zeros(12), "coords": coords},
        force_norm=0.001,
        cap=10,
        project_rigid=False,
        crashed="",
        potential="lj",
    )
    assert record["failure"] == "divergence"
    assert record["hessian_index"] is None
    assert record["max_nn"] == 1.1


def test_a_capped_separated_cluster_stays_cap():
    from optbench.counter import HarnessCounter
    from optbench.pygsm_band import _judge

    counter = HarnessCounter(cache=True, cap=10)
    counter.calls = 10
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float64)
    record = _judge(
        "pair",
        counter,
        {"energy": -1.0e-5, "gradient": np.zeros(9), "coords": coords},
        force_norm=0.001,
        cap=10,
        project_rigid=False,
        crashed="",
        potential="lj",
    )
    assert record["failure"] == "cap"
    assert record["hessian_index"] is None


def test_a_connected_early_stop_is_not_converged():
    from optbench.counter import HarnessCounter
    from optbench.pygsm_band import _judge

    coords = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.55, 0.95, 0.0]], dtype=np.float64)
    record = _judge(
        "pair",
        HarnessCounter(cache=True, cap=10),
        {"energy": -3.0, "gradient": np.full(9, 0.2), "coords": coords},
        force_norm=0.001,
        cap=10,
        project_rigid=False,
        crashed="",
        potential="lj",
    )
    assert record["failure"] == "not_converged"
    assert record["hessian_index"] is None


def test_stock_climb_leaves_a_negative_projection_raw():
    import optbench.pygsm_band as pygsm_band
    from optbench.pygsm_band import ensure_runtime

    pygsm_band._CLIMB_CLAMP = False
    ensure_runtime()
    from pyGSM.optimizers.base_optimizer import base_optimizer

    class Optimizer:
        SCALE_CLIMB = 1.0

    class Molecule:
        constraints = np.eye(2)

    steps = base_optimizer.get_constraint_steps(Optimizer(), Molecule(), "CLIMB", np.array([-4.0, 0.2]))
    assert float(np.asarray(steps).reshape(-1)[0]) == -4.0
    positive = base_optimizer.get_constraint_steps(Optimizer(), Molecule(), "CLIMB", np.array([4.0, 0.0]))
    assert abs(float(np.asarray(positive).reshape(-1)[0])) == 0.05


def test_the_climb_step_limit_applies_when_the_tangent_gradient_is_negative():
    import optbench.pygsm_band as pygsm_band
    from pyGSM.optimizers.base_optimizer import base_optimizer

    pygsm_band._CLIMB_CLAMP = True
    wrapped = _both_signs(base_optimizer.get_constraint_steps)

    class Optimizer:
        SCALE_CLIMB = 1.0

    class Molecule:
        constraints = np.eye(2)

    try:
        steps = wrapped(Optimizer(), Molecule(), "CLIMB", np.array([-4.0, 0.2]))
        assert abs(float(np.asarray(steps).reshape(-1)[0])) == 0.05
        positive = wrapped(Optimizer(), Molecule(), "CLIMB", np.array([4.0, 0.0]))
        assert abs(float(np.asarray(positive).reshape(-1)[0])) == 0.05
        assert base_optimizer.get_constraint_steps is not wrapped
    finally:
        pygsm_band._CLIMB_CLAMP = False


def test_generator_sum_adds_vectors():
    total = _sum_generators(np.array([1.0, 2.0]) for _ in range(3))
    assert np.allclose(total, [3.0, 6.0])
    assert _sum_generators(float(i) for i in range(4)) == 6.0


def test_the_trajectory_rewrite_is_skipped():
    from optbench.pygsm_band import ensure_runtime

    ensure_runtime()
    from pyGSM.utilities import manage_xyz as namespaced
    from utilities import manage_xyz as flat

    assert namespaced.write_xyzs_w_comments(None, None, None) is None
    assert flat.write_xyzs_w_comments(None, None, None) is None
    import importlib

    optimizer = importlib.import_module("pyGSM.optimizers.eigenvector_follow")
    assert optimizer.manage_xyz.write_xyzs_w_comments(None, None, None) is None


def test_hybrid_primitives_are_cartesian_components():
    from optbench.pygsm_band import _stock_hybrid, ensure_runtime
    from pyGSM.utilities import elements

    ensure_runtime()
    reactant = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.5, 0.9, 0.1], [0.4, 0.2, 1.0]],
        dtype=np.float64,
    )
    product = reactant + np.array([0.2, -0.1, 0.05])
    atoms = [elements.ElementData().from_symbol("Ar") for _ in range(len(reactant))]
    coord = _stock_hybrid(reactant, product, atoms)
    names = [type(prim).__name__ for prim in coord.Prims.Internals]
    assert names == ["CartesianX", "CartesianY", "CartesianZ"] * len(reactant)
    assert coord.Vecs.shape == (12, 12)
