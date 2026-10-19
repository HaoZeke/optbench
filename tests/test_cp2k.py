"""CP2K i-PI client, GEO_OPT input, and template pins.

These tests do not launch the binary. A campaign smoke is a separate run.
"""

import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.cp2k_ts import (
    BOHR_TO_ANG,
    CLIMB_CALLS,
    FORCE_SCALE,
    SEARCH_CAP,
    CountedGeometry,
    _commit_saddle_outcome,
    _judge,
    is_dimer_image,
    pad_header,
    render_input,
    serve_client,
)
from optbench.counter import ForceCallCap

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "702cc3d81422b92864ecdd7f544c21c56c04adca"


def test_header_is_twelve_bytes():
    assert pad_header("READY") == b"READY       "
    assert len(pad_header("FORCEREADY")) == 12
    with pytest.raises(ValueError):
        pad_header("FORCEREADY-EXTRA")


def test_image_is_the_dimer_half_step():
    assert is_dimer_image(0.01, 0.01)
    assert is_dimer_image(0.0100000004, 0.01)
    assert not is_dimer_image(0.02, 0.01)
    assert not is_dimer_image(0.01, 0.0)


def test_force_on_the_wire_is_hartree_per_bohr():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    fixed = np.array([True, False])
    grad = np.array([[0.3, -0.1, 0.2], [0.4, -0.2, 0.1]])

    def evaluate(pos, cell):
        return 1.5, grad

    geometry = CountedGeometry(
        evaluate,
        positions,
        fixed,
        np.zeros(3),
        cap=10,
        force_tol=0.001,
        force_match="l2",
        dr_ang=0.01,
    )
    energy, forces = geometry.evaluate_bohr(positions / BOHR_TO_ANG)
    assert energy == pytest.approx(1.5 * FORCE_SCALE)
    assert geometry.calls == 1
    expected = -grad * BOHR_TO_ANG * FORCE_SCALE
    expected[0] = 0.0
    np.testing.assert_allclose(forces.reshape(2, 3), expected)
    # The same center is a cache hit, including the post-search read.
    energy_again, _forces = geometry.evaluate_bohr(positions / BOHR_TO_ANG)
    assert energy_again == pytest.approx(1.5 * FORCE_SCALE)
    assert geometry.calls == 1
    image = positions.copy()
    image[1, 0] += 0.01
    geometry.evaluate_bohr(image / BOHR_TO_ANG)
    assert geometry.calls == 2
    assert geometry.center is not None
    np.testing.assert_allclose(geometry.center, positions)
    with geometry.verification():
        _energy, free = geometry.energy_forces(geometry.free_of(positions))
    assert geometry.calls == 2
    assert geometry.verify_calls == 1
    np.testing.assert_allclose(free, grad[1])


def test_l2_stop_fires_on_the_center_only():
    positions = np.zeros((1, 3))

    def evaluate(pos, cell):
        return 0.0, np.zeros_like(pos)

    geometry = CountedGeometry(
        evaluate,
        positions,
        np.array([False]),
        np.zeros(3),
        cap=10,
        force_tol=0.001,
        force_match="l2",
        dr_ang=0.01,
    )
    geometry.evaluate_bohr(positions)
    assert geometry.stop_now is True
    shifted = positions.copy()
    shifted[0, 0] = 0.01
    fresh = CountedGeometry(
        evaluate,
        positions,
        np.array([False]),
        np.zeros(3),
        cap=10,
        force_tol=0.001,
        force_match="l2",
        dr_ang=0.01,
    )
    fresh.center = positions.copy()
    fresh.evaluate_bohr(np.array([[0.01 / BOHR_TO_ANG, 0.0, 0.0]]))
    assert fresh.stop_now is False


def test_refused_call_does_not_move_the_center():
    positions = np.zeros((1, 3))
    calls = {"n": 0}

    def evaluate(pos, cell):
        calls["n"] += 1
        return 1.0, np.ones_like(pos)

    geometry = CountedGeometry(
        evaluate,
        positions,
        np.array([False]),
        np.zeros(3),
        cap=1,
        force_tol=0.001,
        force_match="library",
        dr_ang=0.0,
    )
    geometry.evaluate_bohr(positions)
    assert geometry.center is not None
    with pytest.raises(ForceCallCap):
        geometry.evaluate_bohr(positions + 1.0)
    np.testing.assert_allclose(geometry.center, positions)
    assert geometry.calls == 1
    assert calls["n"] == 1


def _exchange(sock: socket.socket, positions: np.ndarray) -> tuple[float, np.ndarray]:
    sock.sendall(pad_header("STATUS"))
    assert sock.recv(12).decode("ascii").strip() == "READY"
    natom = positions.shape[0]
    sock.sendall(pad_header("POSDATA"))
    sock.sendall(np.zeros(18, dtype="<f8").tobytes())
    sock.sendall(np.array([natom], dtype="<i4").tobytes())
    sock.sendall(np.ascontiguousarray(positions / BOHR_TO_ANG, dtype="<f8").tobytes())
    sock.sendall(pad_header("STATUS"))
    assert sock.recv(12).decode("ascii").strip() == "HAVEDATA"
    sock.sendall(pad_header("GETFORCE"))
    assert sock.recv(12).decode("ascii").strip() == "FORCEREADY"
    energy = np.frombuffer(sock.recv(8), dtype="<f8")[0]
    got = int(np.frombuffer(sock.recv(4), dtype="<i4")[0])
    forces = np.frombuffer(sock.recv(8 * 3 * got), dtype="<f8").copy()
    _virial = sock.recv(8 * 9)
    extra = int(np.frombuffer(sock.recv(4), dtype="<i4")[0])
    assert got == natom
    assert extra == 0
    return float(energy), forces


def test_socket_exchange_matches_the_ipi_server():
    positions = np.array([[0.2, 0.0, 0.0], [1.0, 0.3, -0.1]])
    grad = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, -0.5]])

    def evaluate(pos, cell):
        return -3.25, grad

    geometry = CountedGeometry(
        evaluate,
        positions,
        np.array([True, False]),
        np.array([10.0, 10.0, 10.0]),
        cap=5,
        force_tol=0.001,
        force_match="library",
        dr_ang=0.01,
    )
    server, client = socket.socketpair()
    holder: dict[str, str] = {}

    def run() -> None:
        holder["status"] = serve_client(client, geometry)

    thread = threading.Thread(target=run)
    thread.start()
    energy, forces = _exchange(server, positions)
    server.sendall(pad_header("EXIT"))
    thread.join(timeout=5)
    server.close()
    assert holder["status"] == "exit"
    assert energy == pytest.approx(-3.25 * FORCE_SCALE)
    np.testing.assert_allclose(forces.reshape(2, 3), -grad * BOHR_TO_ANG * FORCE_SCALE)
    assert geometry.calls == 1


def test_render_input_is_ipi_dimer_not_fist():
    positions = np.arange(6, dtype=np.float64).reshape(2, 3)
    fixed = np.array([True, False])
    mode = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    text = render_input(
        positions,
        fixed,
        np.array([12.0, 12.0, 30.0]),
        element="Pt",
        host="CP2KHOST",
        task="dimer",
        optimizer="BFGS",
        rot_optimizer="BFGS",
        force_match="l2",
        model_hessian=False,
        cg_line_search="2PNT",
        max_iter=100000,
        rot_max_iter=20,
        mode=mode,
    )
    assert "METHOD IPI" in text
    assert "FIST" not in text
    assert "TYPE TRANSITION_STATE" in text
    assert "&DIMER_VECTOR" in text
    assert "ANGLE_TOLERANCE [deg] 1.0" in text
    assert "USE_MODEL_HESSIAN FALSE" in text
    assert "MAX_FORCE 1.0E-8" in text
    assert "SEED 2000" in text
    assert "HOST CP2KHOST" in text
    assert "LIST 1" in text
    assert "PERIODIC NONE" in text
    library = render_input(
        positions,
        np.array([False, False]),
        np.zeros(3),
        element="H",
        host="h",
        task="dimer",
        optimizer="CG",
        rot_optimizer="CG",
        force_match="library",
        model_hessian=True,
        cg_line_search="GOLD",
        max_iter=200,
        rot_max_iter=200,
        mode=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert f"MAX_FORCE {0.00045 * FORCE_SCALE:.8e}" in library
    assert "MAX_FORCE 1.0E-8" not in library
    assert "TYPE GOLD" in library
    assert "MAX_ALLOWED_STEP" not in library
    assert "ABC 1.0000000000000000e+02" in library
    minimize = render_input(
        positions,
        np.array([False, False]),
        np.ones(3),
        element="Pt",
        host="h",
        task="minimize",
        optimizer="BFGS",
        rot_optimizer="BFGS",
        force_match="l2",
        model_hessian=False,
        cg_line_search="2PNT",
        max_iter=100000,
        rot_max_iter=20,
        mode=None,
    )
    assert "TYPE MINIMIZATION" in minimize
    assert "DIMER" not in minimize


def test_shipped_cp2k_templates_match_the_pin():
    found = [template for template in list_templates(ROOT) if template.get("engine") == "cp2k"]
    assert len(found) == 6
    saddle = []
    tsopt = []
    for template in found:
        assert template["engine"] == "cp2k"
        assert template["code"] == "CP2K"
        assert template["commit"] == COMMIT
        assert template["code_version"] == "2026.2"
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        if template["problem"] == "saddle-search/pt-island":
            assert cap is None
            assert "max_force_calls" not in (template.get("stop") or {})
            assert "failure_cap" not in template
            saddle.append(template["id"])
        else:
            assert cap == 10000
            assert float(template["stop"]["force_norm"]) == pytest.approx(0.001)
            tsopt.append(template["id"])
        block = template["cp2k"]
        if template["tuning"] == "tuned":
            assert block["force_match"] == "l2"
            assert block["model_hessian"] is False
        else:
            assert block["force_match"] == "library"
            assert block["max_iter"] == 200
    assert len(tsopt) == 4
    assert len(saddle) == 2
    assert SEARCH_CAP == 1000 + 64 + 4000


def test_asv_classes_import_and_follow_the_full_flag(monkeypatch):
    import importlib

    import benchmarks.bench_cp2k as bench

    monkeypatch.delenv("OPTBENCH_ASV_FULL", raising=False)
    bench = importlib.reload(bench)
    assert len(bench.Cp2kLj38DimerCg.params[1]) == 1
    assert bench.Cp2kLj38DimerCg.timeout == 3600
    assert bench.Cp2kPtDimerCg.timeout == 7200
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    bench = importlib.reload(bench)
    assert len(bench.Cp2kLj38DimerCg.params[1]) == 200
    assert len(bench.Cp2kPtDimerCgTuned.params[1]) == 49
    monkeypatch.delenv("OPTBENCH_ASV_FULL", raising=False)
    importlib.reload(bench)


def _counted(force: float, *, calls: int, cap: int = 10000, energy: float = -1.5) -> CountedGeometry:
    positions = np.zeros((1, 3))

    def evaluate(pos, cell):
        grad = np.zeros_like(pos)
        grad[0, 0] = force
        return energy, grad

    geometry = CountedGeometry(
        evaluate,
        positions,
        np.array([False]),
        np.zeros(3),
        cap=cap,
        force_tol=0.001,
        force_match="library",
        dr_ang=0.0,
    )
    geometry.center = positions.copy()
    geometry.counter.calls = calls
    return geometry


def _judge_spec(**overrides) -> dict:
    spec = {
        "cap": 10000,
        "name": "0001",
        "seed": "0",
        "code": "CP2K",
        "code_version": "2026.2",
        "commit": COMMIT,
        "tuning": "default",
        "template": "cp2k/lj38-dimer-cg",
        "problem": "tsopt/LJ38",
        "force_tol": 0.001,
        "hessian": False,
        "project_rigid": False,
        "task": "dimer",
    }
    spec.update(overrides)
    return spec


def test_clean_stop_above_the_force_threshold_is_not_converged():
    record = _judge(
        _judge_spec(),
        _counted(1.12, calls=1010),
        library_converged=False,
        status="exit",
        detail="",
    )
    assert record["success"] is False
    assert record["failure"] == "not_converged"
    assert record["force_norm"] == pytest.approx(1.12)
    assert record["energy"] == pytest.approx(-1.5)
    assert "force_norm" in record["reasons"]
    assert "divergence" not in record["reasons"]
    assert "cap" not in record["reasons"]


def test_force_equal_to_the_tolerance_is_not_above_it():
    record = _judge(
        _judge_spec(),
        _counted(0.001, calls=200),
        library_converged=True,
        status="exit",
        detail="",
    )
    assert record["failure"] is None
    assert record["success"] is True
    assert "force_norm" not in record["reasons"]


def test_harness_cap_wins_over_a_high_force():
    record = _judge(
        _judge_spec(),
        _counted(1.0, calls=10000),
        library_converged=False,
        status="cap",
        detail="",
    )
    assert record["failure"] == "cap"
    assert record["success"] is False


def test_finite_stall_above_the_stop_is_not_divergence():
    record = _judge(
        _judge_spec(),
        _counted(0.4, calls=305),
        library_converged=False,
        status="stall",
        detail="",
    )
    assert record["failure"] == "not_converged"
    assert "divergence" not in record["reasons"]


def test_nonfinite_energy_is_divergence():
    record = _judge(
        _judge_spec(),
        _counted(0.1, calls=10, energy=float("nan")),
        library_converged=False,
        status="exit",
        detail="",
    )
    assert record["failure"] == "divergence"


def test_process_error_under_the_cap_is_a_crash():
    record = _judge(
        _judge_spec(),
        _counted(0.0002, calls=4),
        library_converged=False,
        status="error",
        detail="CPABORT something broke",
    )
    assert record["failure"] == "crash"
    assert record["reasons"] == ["crash"]


def test_refused_call_at_the_cap_with_no_center_is_cap():
    geometry = _counted(1.0, calls=10000)
    geometry.center = None
    record = _judge(_judge_spec(), geometry, library_converged=False, status="error", detail="force-call cap")
    assert record["failure"] == "cap"
    assert record["reasons"] == ["cap"]


def _saddle_commit(tmp_path, outcome: dict):
    positions = np.zeros((2, 3))
    fixed = np.array([True, False])
    cell = np.ones(3)
    template = {"id": "cp2k/saddle-dimer-cg", "code": "CP2K", "code_version": "2026.2", "commit": COMMIT}
    base = {
        "code": "CP2K",
        "code_version": "2026.2",
        "commit": COMMIT,
        "tuning": "default",
        "potential": {},
        "force_tol": 0.001,
    }
    spec = {"name": "rep1-search1"}
    found = []
    records = []
    connected, _refused, failure = _commit_saddle_outcome(
        spec,
        outcome,
        {},
        found=found,
        all_records=records,
        connected=0,
        positions=positions,
        fixed=fixed,
        cell=cell,
        minimum=np.zeros(3),
        min_energy=-2.0,
        force_tol=0.001,
        connect_rmsd=0.5,
        tolerance=0.1,
        barrier_max=1.5,
        base=base,
        partial_path=tmp_path / "partial.jsonl",
        output=tmp_path / "campaign.dat",
        template=template,
        problem_id="saddle-search/pt-island",
    )
    return connected, failure, records[0]


def test_climb_budget_is_not_the_harness_cap(tmp_path):
    _connected, failure, record = _saddle_commit(
        tmp_path,
        {
            "force_calls": CLIMB_CALLS,
            "reasons": ["cap", "force_norm"],
            "converged": False,
            "hessian_index": None,
            "energy": -1.0,
            "force_norm": 1.2,
            "x": [0.1, 0.0, 0.0],
            "force": [1.2, 0.0, 0.0],
            "failure": "cap",
            "verify_calls": 0,
        },
    )
    assert failure == "not_converged"
    assert record["success"] is False
    assert record["failure"] == "not_converged"
    assert "cap" not in record["reasons"]
    assert record["force_calls"] == CLIMB_CALLS


def test_wrong_index_at_the_stop_is_wrong_stationary(tmp_path):
    _connected, failure, record = _saddle_commit(
        tmp_path,
        {
            "force_calls": 400,
            "reasons": ["hessian_index"],
            "converged": True,
            "hessian_index": 0,
            "energy": -1.0,
            "force_norm": 0.0004,
            "x": [0.2, 0.0, 0.0],
            "force": [0.0004, 0.0, 0.0],
            "failure": "wrong_stationary",
            "verify_calls": 12,
        },
    )
    assert failure == "wrong_stationary"
    assert record["success"] is False
    assert record["force_calls"] < SEARCH_CAP


def test_saddle_search_is_cap_only_at_the_harness_cap(tmp_path):
    _connected, failure, record = _saddle_commit(
        tmp_path,
        {
            "force_calls": SEARCH_CAP,
            "reasons": ["force_norm"],
            "converged": False,
            "hessian_index": None,
            "energy": -1.0,
            "force_norm": 0.5,
            "x": [0.3, 0.0, 0.0],
            "force": [0.5, 0.0, 0.0],
            "failure": "not_converged",
            "verify_calls": 0,
        },
    )
    assert failure == "cap"
    assert record["failure"] == "cap"
    assert record["force_calls"] == SEARCH_CAP
