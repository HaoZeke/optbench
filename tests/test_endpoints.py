"""Free-atom float32 endpoint deltas stay small enough to store."""

from pathlib import Path

import numpy as np

from optbench.endpoints import FORMAT, large_endpoint_files, load_endpoints, save_endpoints


def test_float32_deltas_round_trip_and_beat_a_full_cartesian_archive(tmp_path: Path):
    rng = np.random.default_rng(1)
    searches = 80
    atoms = 343
    free = np.zeros(atoms, dtype=bool)
    free[-175:] = True
    free_index = np.flatnonzero(free)
    reactant = rng.normal(scale=4.0, size=(atoms, 3))
    plus = np.repeat(reactant[None, :, :], searches, axis=0)
    minus = np.repeat(reactant[None, :, :], searches, axis=0)
    plus[:, free] += rng.normal(scale=0.05, size=(searches, int(free.sum()), 3))
    minus[:, free] += rng.normal(scale=0.02, size=(searches, int(free.sum()), 3))
    plus[0] = np.nan
    minus[0] = np.nan
    fat = tmp_path / "results" / "campaign" / "fat.endpoints-plus.npz"
    fat.parent.mkdir(parents=True)
    flat = plus.reshape(searches, -1)
    minimum = np.repeat(reactant.reshape(1, -1), searches, axis=0)
    np.savez_compressed(fat, endpoint_plus=flat, minimum=minimum)
    path = save_endpoints(
        tmp_path / "results" / "campaign" / "run.endpoints.npz",
        reactant,
        plus,
        minus,
        free_index,
        cell=np.array([19.2, 19.0, 30.0]),
    )
    assert path.stat().st_size < 1_000_000
    assert path.stat().st_size < fat.stat().st_size
    loaded = load_endpoints(path)
    assert loaded["free_index"].tolist() == free_index.tolist()
    expected_plus = plus[:, free].astype(np.float32)
    expected_minus = minus[:, free].astype(np.float32)
    np.testing.assert_allclose(loaded["plus"][1:], expected_plus[1:], atol=2e-5, rtol=0)
    np.testing.assert_allclose(loaded["minus"][1:], expected_minus[1:], atol=2e-5, rtol=0)
    assert np.isnan(loaded["plus"][0]).all()
    assert np.isnan(loaded["minus"][0]).all()
    with np.load(path) as archive:
        assert str(archive["format"]) == FORMAT
        assert archive["reactant"].shape == (175, 3)
        assert archive["plus"].dtype == np.float32
        stored = archive["reactant"].nbytes + archive["plus"].nbytes + archive["minus"].nbytes
    full = plus.nbytes + minus.nbytes + np.repeat(reactant[None, :, :], searches, axis=0).nbytes
    assert stored * 4 < full


def test_distinct_reactants_keep_one_row_each(tmp_path: Path):
    rng = np.random.default_rng(2)
    reactant = rng.normal(size=(4, 6, 3))
    plus = reactant + 0.1
    minus = reactant - 0.2
    index = np.arange(6)
    path = save_endpoints(tmp_path / "varied.endpoints.npz", reactant, plus, minus, index)
    loaded = load_endpoints(path)
    np.testing.assert_allclose(loaded["reactant"], reactant.astype(np.float32), atol=2e-6, rtol=0)
    np.testing.assert_allclose(loaded["plus"], plus.astype(np.float32), atol=2e-5, rtol=0)
    with np.load(path) as archive:
        assert archive["reactant"].shape == (4, 6, 3)


def test_a_full_cartesian_endpoint_blob_fails_the_size_gate(tmp_path: Path):
    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    fat = campaign / "island.endpoints.npz"
    payload = np.random.default_rng(0).normal(size=(400, 525))
    np.savez_compressed(fat, endpoint_plus=payload, minimum=payload)
    assert fat.stat().st_size > 1_000_000
    errors = large_endpoint_files(tmp_path)
    assert errors
    assert "island.endpoints.npz" in errors[0]
