"""pyGSM template pin and the dissociation gate."""

import tomllib
from pathlib import Path

import numpy as np

from optbench.campaign import load_template
from optbench.pygsm_band import _stable_sorted_eigh, _sum_generators, largest_nearest_neighbour

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
    from benchmarks.bench_pygsm import PygsmLj38De, PygsmLj38DeClimb, PygsmLj38TwoTs, PygsmLj38TwoTsClimb

    assert PygsmLj38De.problem_name == "double-ended-saddle-search/lj38"
    assert PygsmLj38De.template_id == "pygsm/de-gsm"
    assert PygsmLj38De.cap == 2500
    assert PygsmLj38De.timeout == 1800
    assert PygsmLj38De.params[1] == ["pair_00"]
    assert PygsmLj38TwoTs.problem_name == "double-ended-saddle-search/lj38_2ts_separation"
    assert PygsmLj38TwoTs.cap == 10000
    assert PygsmLj38TwoTs.timeout == 7200
    assert PygsmLj38TwoTs.params[1] == ["pair_00"]
    assert PygsmLj38DeClimb.template_id == "pygsm/de-gsm-climbclamp"
    assert PygsmLj38DeClimb.problem_name == PygsmLj38De.problem_name
    assert PygsmLj38DeClimb.cap == 2500
    assert PygsmLj38TwoTsClimb.template_id == "pygsm/de-gsm-climbclamp"
    assert PygsmLj38TwoTsClimb.cap == 10000


def test_template_matches_the_registry_pin():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pygsm")
    template = load_template(ROOT, "pygsm/de-gsm")
    clamp = load_template(ROOT, "pygsm/de-gsm-climbclamp")
    assert pinned["commit"] == COMMIT
    assert "de-gsm-climbclamp" in pinned["note"]
    assert "stock climb step" in pinned["note"]
    assert template["commit"] == pinned["commit"]
    assert template["code"] == "pygsm"
    assert template["engine"] == "pygsm-band"
    assert template["algorithm"] == "DE-GSM"
    assert template["tuning"] == "default"
    assert template["code_version"] == "1.0.0+692.g1b4d5a5.dirty"
    assert template["gsm"]["coordinate"] == "cartesian"
    assert template["gsm"]["nnodes"] == 11
    assert template["gsm"]["dmax"] == 0.1
    assert template["gsm"]["add_node_tol"] == 0.01
    assert template["gsm"]["max_opt_steps"] == 3
    assert template["gsm"]["climb_clamp"] is False
    assert template["gsm"]["endpoints"] == "fixed"
    assert "problem" not in template
    assert clamp["id"] == "pygsm/de-gsm-climbclamp"
    assert clamp["tuning"] == "climbclamp"
    assert clamp["gsm"]["climb_clamp"] is True
    assert clamp["gsm"]["nnodes"] == template["gsm"]["nnodes"]
    assert clamp["gsm"]["dmax"] == template["gsm"]["dmax"]
    assert clamp["commit"] == template["commit"]
    diff = (ROOT / "templates" / "pygsm" / "de-gsm-climbclamp.diff").read_text(encoding="utf-8")
    assert "-            if gts > np.abs(max_step):" in diff
    assert "+            if np.abs(gts) > np.abs(max_step):" in diff


def test_a_broken_delocalized_basis_is_divergence():
    from optbench.pygsm_band import _numeric_failure

    assert _numeric_failure(RuntimeError("error in orthonormality"))
    assert _numeric_failure(np.linalg.LinAlgError("Eigenvalues did not converge"))
    assert _numeric_failure(ValueError("array must not contain infs or NaNs"))
    assert _numeric_failure(RuntimeError("missing import")) is False


def test_largest_nearest_neighbour_is_the_worst_contact():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
    assert largest_nearest_neighbour(coords) == 3.0


def test_rotation_eigensolve_symmetrizes_and_falls_back():
    def original(mat, b=None, asc=False):
        assert np.allclose(mat, mat.T)
        values = np.linalg.eigvalsh(mat)
        order = np.argsort(values)
        if not asc:
            order = order[::-1]
        return values[order], np.eye(len(values))

    matrix = np.array([[1.0, 2.0], [0.0, 3.0]])
    expected = np.linalg.eigvalsh(np.array([[1.0, 1.0], [1.0, 3.0]]))[::-1]
    values, _vectors = _stable_sorted_eigh(original)(matrix)
    assert np.allclose(values, expected)

    def failing(mat, b=None, asc=False):
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    values, vectors = _stable_sorted_eigh(failing)(matrix)
    assert np.allclose(values, expected)
    assert vectors.shape == (2, 2)


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
    from optbench.pygsm_band import ensure_runtime

    pygsm_band._CLIMB_CLAMP = True
    ensure_runtime()
    from pyGSM.optimizers.base_optimizer import base_optimizer

    class Optimizer:
        SCALE_CLIMB = 1.0

    class Molecule:
        constraints = np.eye(2)

    try:
        steps = base_optimizer.get_constraint_steps(Optimizer(), Molecule(), "CLIMB", np.array([-4.0, 0.2]))
        assert abs(float(np.asarray(steps).reshape(-1)[0])) == 0.05
        positive = base_optimizer.get_constraint_steps(Optimizer(), Molecule(), "CLIMB", np.array([4.0, 0.0]))
        assert abs(float(np.asarray(positive).reshape(-1)[0])) == 0.05
    finally:
        pygsm_band._CLIMB_CLAMP = False


def test_generator_sum_adds_vectors():
    total = _sum_generators(np.array([1.0, 2.0]) for _ in range(3))
    assert np.allclose(total, [3.0, 6.0])
    assert _sum_generators(float(i) for i in range(4)) == 6.0
