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
    FORCE_SCALE,
    SEARCH_CAP,
    CountedGeometry,
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
    found = [template for template in list_templates(ROOT) if template["id"].startswith("cp2k/")]
    assert len(found) == 12
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
    assert len(tsopt) == 8
    assert len(saddle) == 4
    assert SEARCH_CAP == 1000 + 64 + 4000
