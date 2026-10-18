"""ase-gpatom AIDNEB uses the harness counter and the problem cap."""

from pathlib import Path

import numpy as np

from optbench.campaign import geometry_archive, load_template, resolve_problem
from optbench.counter import HarnessCounter
from optbench.gpatom_band import _Spec, _cartesian_block, prepare, search_named
from optbench.pes import load_pairs

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "572bd64ef61ccf498e061162609bb5f16e39d8a4"


def test_template_matches_the_registry_pin():
    template = load_template(ROOT, "gpatom/aidneb")
    assert template["engine"] == "gpatom-band"
    assert template["commit"] == COMMIT
    assert template["code_version"] == "1.0.1"
    assert template["aid"]["interpolation"] == "idpp"
    assert template["aid"]["max_train_data"] == 25
    text = (ROOT / "codes" / "registry.toml").read_text(encoding="utf-8")
    assert f'id = "gpatom"' in text
    assert COMMIT in text


def test_archived_pair_counts():
    expected = {
        "neb/Pt-heptamer": (13, 175),
        "double-ended-saddle-search/lj38": (50, 38),
        "double-ended-saddle-search/lj38_2ts_separation": (51, 38),
        "double-ended-saddle-search/pt-island": (59, 175),
    }
    for name, (pairs, n_free) in expected.items():
        problem = resolve_problem(ROOT, name)
        loaded = load_pairs(
            geometry_archive(ROOT, problem),
            problem["geometry"]["kind"],
            int(problem["sample"]["pairs"]),
        )
        assert len(loaded) == pairs
        assert int(np.count_nonzero(loaded[0].free)) == n_free


def test_one_geometry_is_one_force_call():
    prepare()
    counter = HarnessCounter(cache=True, cap=10)
    template = np.arange(12, dtype=float).reshape(4, 3)
    mask = np.array([True, True, False, False])
    spec = _Spec(counter, "lj", template, mask, None, {"repeats": 0}, "cache")
    atoms = __import__("ase").Atoms(numbers=[1, 1], positions=template[mask])
    atoms.calc = spec
    atoms.get_potential_energy()
    atoms.get_forces()
    assert counter.calls == 1
    atoms.get_potential_energy()
    assert counter.calls == 1


def test_cartesian_kernel_matches_fpkernel():
    prepare()
    from ase import Atoms
    from gpatom.gpfp.fingerprint import CartesianCoordFP
    from gpatom.gpfp.kernel import FPKernel

    rng = np.random.default_rng(2)
    positions = rng.normal(size=(3, 5, 3))
    positions[2] = positions[0]
    fingerprints = [
        CartesianCoordFP(Atoms(numbers=[1] * 5, positions=row))
        for row in positions
    ]
    kernel = FPKernel("sqexp", params={"weight": 1.2, "scale": 0.4})
    for left in range(3):
        for right in range(3):
            reference = kernel.kernel(fingerprints[left], fingerprints[right])
            built = _cartesian_block(positions[left], positions[right], 1.2, 0.4)
            assert np.allclose(reference, built, rtol=1e-10, atol=1e-10)


def test_short_cap_stops_on_the_first_lj38_pair():
    record = search_named(
        ROOT,
        "double-ended-saddle-search/lj38",
        "gpatom/aidneb",
        "pair_00",
        cap=4,
    )
    assert record["counter"] == "harness"
    assert record["template"] == "gpatom/aidneb"
    assert record["commit"] == COMMIT
    assert 2 <= int(record["force_calls"]) <= 4
    assert record["failure"] in {"cap", "divergence", "wrong_stationary", "crash"}
    assert int(record["aid_force_calls"]) <= int(record["force_calls"])
    assert record["n_free"] == 38
    assert record["n_images"] == 7
