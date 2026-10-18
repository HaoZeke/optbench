"""pyGSM template pin and the contact coordinate helper."""

import tomllib
from pathlib import Path

import numpy as np

from optbench.campaign import load_template
from optbench.pygsm_band import _stable_sorted_eigh, _sum_generators, contact_pairs

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
    from benchmarks.bench_pygsm import PygsmLj38De, PygsmLj38TwoTs

    assert PygsmLj38De.problem_name == "double-ended-saddle-search/lj38"
    assert PygsmLj38De.template_id == "pygsm/de-gsm"
    assert PygsmLj38De.cap == 2500
    assert PygsmLj38De.timeout == 1800
    assert PygsmLj38De.params[1] == ["pair_00"]
    assert PygsmLj38TwoTs.problem_name == "double-ended-saddle-search/lj38_2ts_separation"
    assert PygsmLj38TwoTs.cap == 10000
    assert PygsmLj38TwoTs.timeout == 7200
    assert PygsmLj38TwoTs.params[1] == ["pair_00"]


def test_template_matches_the_registry_pin():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pygsm")
    template = load_template(ROOT, "pygsm/de-gsm")
    assert pinned["commit"] == COMMIT
    assert template["commit"] == pinned["commit"]
    assert template["code"] == "pygsm"
    assert template["engine"] == "pygsm-band"
    assert template["algorithm"] == "DE-GSM"
    assert template["tuning"] == "default"
    assert template["code_version"] == "1.0.0+692.g1b4d5a5.dirty"
    assert template["gsm"]["contact_cutoff"] == 1.35
    assert template["gsm"]["add_node_tol"] == 0.01
    assert template["gsm"]["max_opt_steps"] == 3
    assert template["gsm"]["endpoints"] == "fixed"
    assert "problem" not in template


def test_contact_pairs_are_the_upper_triangle_inside_the_cutoff():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    assert contact_pairs(coords, 1.35) == [(0, 1)]
    assert contact_pairs(coords, 3.1) == [(0, 1), (0, 2), (1, 2)]


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


def test_generator_sum_adds_vectors():
    total = _sum_generators(np.array([1.0, 2.0]) for _ in range(3))
    assert np.allclose(total, [3.0, 6.0])
    assert _sum_generators(float(i) for i in range(4)) == 6.0
