"""pyGSM template pin and the contact coordinate helper."""

import tomllib
from pathlib import Path

import numpy as np

from optbench.campaign import load_template
from optbench.pygsm_band import _sum_generators, contact_pairs

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1b4d5a5557a61adc6a3df7405d6b7255b1fa4865"


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


def test_generator_sum_adds_vectors():
    total = _sum_generators(np.array([1.0, 2.0]) for _ in range(3))
    assert np.allclose(total, [3.0, 6.0])
    assert _sum_generators(float(i) for i in range(4)) == 6.0
