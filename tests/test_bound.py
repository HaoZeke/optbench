"""A minimization success is one connected cluster."""

from __future__ import annotations

import math

import numpy as np

from optbench.bound import (
    _sync_asv,
    archived_dissociated,
    bound_distance,
    connected_cluster,
    distance_for_kind,
    energy_proves_dissociated,
    note_unbound,
    rescore_record,
)
from optbench.failures import classify_failure
from optbench.potentials import MORSE_R0, lennard_jones, morse_bulk
from optbench.problems import load_problem

ROOT_PROBLEMS = {
    "lj38": "problems/minimization/lj38.toml",
    "noise": "problems/minimization/lj38-noise.toml",
    "morse": "problems/minimization/morse-bulk.toml",
}


def _problem(name: str) -> dict:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return load_problem(root / ROOT_PROBLEMS[name])


def _relax_lj38() -> np.ndarray:
    """Clipped descent from a compact 38-atom lattice. The step cannot leave the cutoff."""
    spacing = 1.12
    grid = math.ceil(38 ** (1.0 / 3.0))
    points = []
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                points.append((i * spacing, j * spacing, k * spacing))
                if len(points) == 38:
                    break
            if len(points) == 38:
                break
        if len(points) == 38:
            break
    pos = np.asarray(points, dtype=np.float64)
    for _ in range(4000):
        _energy, gradient = lennard_jones(pos)
        grad = gradient.reshape(-1, 3)
        if float(np.linalg.norm(grad)) <= 0.01:
            return pos
        step = -0.002 * grad
        norms = np.linalg.norm(step, axis=1)
        scale = np.minimum(1.0, 0.05 / np.maximum(norms, 1e-12))
        pos = pos + step * scale[:, None]
    raise AssertionError("the clipped descent did not reach the force stop")


def _morse_crystal() -> tuple[np.ndarray, np.ndarray]:
    a = MORSE_R0 * math.sqrt(2.0)
    basis = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        dtype=np.float64,
    )
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                origin = np.array([i, j, k], dtype=np.float64)
                points.extend((origin + basis) * a)
    pos = np.asarray(points, dtype=np.float64)
    box = np.array([4.0 * a, 4.0 * a, 4.0 * a])
    return pos, box


def test_problem_files_declare_the_neighbour_distance():
    lj = _problem("lj38")
    noise = _problem("noise")
    morse = _problem("morse")
    assert bound_distance(lj) == 1.5 * lj["potential"]["sigma"]
    assert bound_distance(noise) == distance_for_kind("lj")
    assert bound_distance(morse) == distance_for_kind("morse")
    assert lj["bound"]["atoms"] == 38
    assert morse["bound"]["atoms"] == 256
    assert morse["bound"]["periodic"] is True
    assert "one connected cluster" in lj["scoring"]["v2"]["success"]
    assert "per-start reference" in noise["scoring"]["v2"]["success"]
    assert "divergence" in morse["scoring"]["v2"]["success"]


def test_a_dissociated_lj38_end_state_is_divergence():
    pos = np.zeros((38, 3), dtype=np.float64)
    pos[:, 0] = np.arange(38) * 3.0
    assert connected_cluster(pos, distance_for_kind("lj")) is False
    reasons = ["hessian_index"]
    note_unbound(reasons, pos, kind="lj")
    assert classify_failure(reasons) == "divergence"
    problem = _problem("lj38")
    record = {"success": True, "failure": None, "reasons": [], "energy": 0.0, "force_calls": 2}
    assert archived_dissociated(problem, record)
    assert rescore_record(problem, record) is True
    assert record["success"] is False
    assert record["failure"] == "divergence"


def test_a_real_lj38_minimum_stays_one_cluster():
    pos = _relax_lj38()
    energy, gradient = lennard_jones(pos)
    assert float(np.linalg.norm(gradient)) <= 0.01
    assert energy < -11.9
    assert connected_cluster(pos, distance_for_kind("lj")) is True
    reasons: list[str] = []
    note_unbound(reasons, pos, kind="lj")
    assert reasons == []
    problem = _problem("lj38")
    assert energy_proves_dissociated("lj", energy, 38, bound_distance(problem)) is False
    record = {"success": True, "failure": None, "reasons": [], "energy": energy, "force_calls": 10}
    assert rescore_record(problem, record) is False
    assert record["success"] is True


def test_a_morse_minimum_is_one_cluster():
    pos, box = _morse_crystal()
    assert pos.shape == (256, 3)
    energy, gradient = morse_bulk(pos, box)
    assert float(np.linalg.norm(gradient)) < 1e-6
    assert energy < -1000.0
    distance = distance_for_kind("morse")
    assert connected_cluster(pos, distance, box) is True
    reasons: list[str] = []
    note_unbound(reasons, pos, kind="morse", box=box)
    assert reasons == []
    problem = _problem("morse")
    record = {"success": True, "failure": None, "reasons": [], "energy": energy, "force_calls": 20}
    assert rescore_record(problem, record) is False
    separated = {"success": True, "failure": None, "reasons": [], "energy": 0.0, "force_calls": 4}
    assert rescore_record(problem, separated) is True
    assert separated["failure"] == "divergence"


def _asv_document(success):
    params = [["'0'"], ["'0'", "'1'"]]
    def track(values):
        return [values, params, "v", 0, 0]
    return {
        "results": {
            "bench_torchsim.TorchsimLj38Gd.track_success": track(success),
            "bench_torchsim.TorchsimLj38Gd.track_force_calls": track([2.0, 3.0]),
            "bench_torchsim.TorchsimLj38Gd.track_failure_divergence": track([0.0, 0.0]),
            "bench_torchsim.TorchsimLj38Gd.track_failure_wrong_stationary": track([0.0, 0.0]),
            "bench_torchsim.TorchsimLj38Gd.track_failure_cap": track([0.0, 0.0]),
            "bench_torchsim.TorchsimLj38Gd.track_failure_crash": track([0.0, 0.0]),
        }
    }


def test_legacy_rows_without_success_do_not_rewrite_asv(tmp_path):
    import json

    path = tmp_path / "results" / "asv" / "cursor" / "sample.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_asv_document([1.0, 1.0])), encoding="utf-8")
    legacy = [
        {"index": 0, "force_calls": 2, "failed": True},
        {"index": 1, "force_calls": 3, "failed": True},
    ]
    _sync_asv(tmp_path, [{"template": "torchsim/lj38-gd", "rows": legacy}])
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["results"]["bench_torchsim.TorchsimLj38Gd.track_success"][0] == [1.0, 1.0]


def test_rescored_rows_rewrite_matching_asv_tracks(tmp_path):
    import json

    path = tmp_path / "results" / "asv" / "machine" / "sample.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_asv_document([1.0, 0.0])) + "\n", encoding="utf-8")
    rows = [
        {"index": 0, "force_calls": 2, "success": False, "failure": "divergence"},
        {"index": 1, "force_calls": 3, "success": False, "failure": "divergence"},
    ]
    _sync_asv(tmp_path, [{"template": "torchsim/lj38-gd", "rows": rows}])
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["results"]["bench_torchsim.TorchsimLj38Gd.track_success"][0] == [0.0, 0.0]
    assert stored["results"]["bench_torchsim.TorchsimLj38Gd.track_failure_divergence"][0] == [1.0, 1.0]
    assert stored["results"]["bench_torchsim.TorchsimLj38Gd.track_force_calls"][0] == [2.0, 3.0]
    assert path.read_text(encoding="utf-8").endswith("\n")
