"""CP2K BAND client. The binary is optional; the socket tests use a socketpair."""

from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import list_templates
from optbench.counter import HarnessCounter
from optbench.cp2k_band import (
    ANGSTROM_PER_BOHR,
    EV_PER_HARTREE,
    IpiClient,
    _HEADER,
    _read_header,
    _recv_exact,
    binary_path,
    failure_reasons,
    initial_frames,
    max_force_internal,
    max_step_internal,
    open_replica_socket,
    render_input,
    select_candidate,
)
from optbench.failures import classify_failure

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "67b5da876dd6a76b8b021d5a04d1c81ba79a4c50"


def _pad(text: str) -> bytes:
    return text.encode("ascii").ljust(_HEADER)


def test_physcon_conversion_matches_the_component_bound():
    force_norm = 0.001
    n_free = 38
    internal = max_force_internal(force_norm, n_free)
    expected = (force_norm / np.sqrt(n_free * 3)) * ANGSTROM_PER_BOHR / EV_PER_HARTREE
    assert internal == pytest.approx(expected)
    assert EV_PER_HARTREE == pytest.approx(27.21138386, rel=1e-9)
    assert ANGSTROM_PER_BOHR == pytest.approx(0.52917720859)
    assert max_step_internal(0.05) == pytest.approx(0.05 / ANGSTROM_PER_BOHR)


def test_fixed_atoms_stay_on_the_reactant():
    reactant = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    product = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    free = np.array([False, True])
    frames = initial_frames(reactant, product, free, 1, None)
    assert len(frames) == 3
    for frame in frames:
        assert frame[0].tolist() == [0.0, 0.0, 0.0]
    assert frames[0][1, 0] == pytest.approx(1.0)
    assert frames[1][1, 0] == pytest.approx(2.0)
    assert frames[2][1, 0] == pytest.approx(3.0)


def test_input_records_the_band_and_the_mapped_force():
    internal = max_force_internal(0.01, 4)
    text = render_input(
        band_type="CI-NEB",
        n_replica=2,
        socket_path="/tmp/ob.sock",
        replica_paths=[Path("/tmp/r000.xyz"), Path("/tmp/r001.xyz")],
        positions=np.zeros((2, 3)),
        cell=(10.0, 10.0, 10.0),
        periodic=False,
        fixed=[1],
        max_force=internal,
        max_step=max_step_internal(0.2),
        max_steps=100,
        align=False,
    )
    assert "BAND_TYPE CI-NEB" in text
    assert "METHOD IPI" in text
    assert "NSTEPS_IT 5" in text
    assert "ALIGN_FRAMES F" in text
    assert f"MAX_FORCE {internal:.16e}" in text
    assert 'HOST "/tmp/ob.sock"' in text
    assert "LIST 1" in text
    assert "PERIODIC NONE" in text


def test_candidate_is_the_highest_interior_image_of_the_last_sweep():
    evals = []
    for energy in (0.0, 5.0, 1.0, 0.0, 4.0, 0.2, 9.0):
        evals.append({"energy": energy, "force": 1.0})
    chosen = select_candidate(evals, 3)
    assert chosen["energy"] == pytest.approx(4.0)


def test_cap_outranks_the_force_norm():
    reasons = failure_reasons(
        force=1.0,
        calls=10,
        cap=10,
        hit_cap=True,
        diverged=False,
        crashed=False,
        force_norm=0.001,
    )
    assert classify_failure(reasons) == "cap"


def test_socket_evaluation_uses_hartree_and_bohr():
    left, right = socket.socketpair()
    left.settimeout(5)
    right.settimeout(5)
    gradient = np.array([[0.5, -0.25, 0.0]], dtype=np.float64)

    def evaluate(_kind, coords, _box):
        assert coords.shape == (1, 3)
        return 2.5, gradient.copy()

    # The client calls pes.evaluate. Patch the name it uses.
    import optbench.cp2k_band as band

    original = band.evaluate
    band.evaluate = evaluate
    client = IpiClient(
        kind="lj",
        free=np.array([True]),
        box=None,
        n_images=3,
        force_norm=0.001,
        cap=4,
        band_type="IT-NEB",
        counter=HarnessCounter(cache=False, cap=4),
    )

    errors: list[BaseException] = []

    def server():
        try:
            _server_body()
        except BaseException as exc:
            errors.append(exc)

    def _server_body():
        right.sendall(_pad("STATUS"))
        assert _read_header(right) == "READY"
        right.sendall(_pad("POSDATA"))
        right.sendall(np.zeros(9, dtype=np.float64).tobytes())
        right.sendall(np.zeros(9, dtype=np.float64).tobytes())
        right.sendall(struct.pack("=i", 1))
        bohr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        right.sendall(bohr.tobytes())
        right.sendall(_pad("STATUS"))
        assert _read_header(right) == "HAVEDATA"
        right.sendall(_pad("GETFORCE"))
        assert _read_header(right) == "FORCEREADY"
        energy = struct.unpack("=d", _recv_exact(right, 8))[0]
        natom = struct.unpack("=i", _recv_exact(right, 4))[0]
        forces = np.frombuffer(_recv_exact(right, 24), dtype=np.float64).copy()
        _recv_exact(right, 8 * 9)
        extra = struct.unpack("=i", _recv_exact(right, 4))[0]
        assert natom == 1
        assert extra == 0
        assert energy == pytest.approx(2.5 / EV_PER_HARTREE)
        expected = -gradient.reshape(-1) * ANGSTROM_PER_BOHR / EV_PER_HARTREE
        assert forces == pytest.approx(expected)
        right.sendall(_pad("EXIT"))

    thread = threading.Thread(target=server)
    thread.start()
    try:
        client.serve(left)
    finally:
        band.evaluate = original
        thread.join(timeout=5)
        left.close()
        right.close()
    assert errors == []
    assert thread.is_alive() is False
    assert client.counter.calls == 1
    assert client.evals[0]["positions"][0].tolist() == pytest.approx(
        (np.array([1.0, 2.0, 3.0]) * ANGSTROM_PER_BOHR).tolist()
    )
    assert client.evals[0]["force"] == pytest.approx(float(np.linalg.norm(gradient)))


def test_the_cap_closes_before_the_next_evaluation():
    left, right = socket.socketpair()
    left.settimeout(5)
    right.settimeout(5)
    import optbench.cp2k_band as band

    original = band.evaluate
    band.evaluate = lambda *_args, **_kwargs: (1.0, np.zeros((1, 3)))
    client = IpiClient(
        kind="lj",
        free=np.array([True]),
        box=None,
        n_images=3,
        force_norm=0.001,
        cap=1,
        band_type="IT-NEB",
        counter=HarnessCounter(cache=False, cap=1),
    )

    def one_eval():
        right.sendall(_pad("STATUS"))
        assert _read_header(right) == "READY"
        right.sendall(_pad("POSDATA"))
        right.sendall(np.zeros(18, dtype=np.float64).tobytes())
        right.sendall(struct.pack("=i", 1))
        right.sendall(np.zeros(3, dtype=np.float64).tobytes())
        right.sendall(_pad("STATUS"))
        assert _read_header(right) == "HAVEDATA"
        right.sendall(_pad("GETFORCE"))
        assert _read_header(right) == "FORCEREADY"
        _recv_exact(right, 8 + 4 + 24 + 72 + 4)

    errors: list[BaseException] = []

    def server():
        try:
            one_eval()
            right.sendall(_pad("STATUS"))
            with pytest.raises(EOFError):
                _read_header(right)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=server)
    thread.start()
    try:
        client.serve(left)
    finally:
        band.evaluate = original
        thread.join(timeout=5)
        left.close()
        right.close()
    assert errors == []
    assert client.stop_reason == "cap"
    assert client.counter.calls == 1


def test_the_replica_socket_is_the_one_that_counts():
    import optbench.cp2k_band as band

    path = Path("/tmp") / f"obtest{threading.get_ident()}.sock"
    if path.exists():
        path.unlink()
    original = band.evaluate
    band.evaluate = lambda *_args, **_kwargs: (1.0, np.zeros((1, 3)))
    client = IpiClient(
        kind="lj",
        free=np.array([True]),
        box=None,
        n_images=3,
        force_norm=0.001,
        cap=4,
        band_type="IT-NEB",
        counter=HarnessCounter(cache=False, cap=4),
    )

    class _Alive:
        returncode = None

        def poll(self):
            return None

    errors: list[BaseException] = []

    def fake_cp2k():
        try:
            outer_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            outer_listener.bind(str(path))
            outer_listener.listen(1)
            outer, _addr = outer_listener.accept()
            outer_listener.close()
            path.unlink()
            outer.sendall(_pad("STATUS"))
            assert _read_header(outer) == "READY"
            replica_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            replica_listener.bind(str(path))
            replica_listener.listen(1)
            replica, _addr = replica_listener.accept()
            replica_listener.close()
            replica.sendall(_pad("STATUS"))
            assert _read_header(replica) == "READY"
            replica.sendall(_pad("POSDATA"))
            replica.sendall(np.zeros(18, dtype=np.float64).tobytes())
            replica.sendall(struct.pack("=i", 1))
            replica.sendall(np.zeros(3, dtype=np.float64).tobytes())
            replica.sendall(_pad("STATUS"))
            assert _read_header(replica) == "HAVEDATA"
            replica.sendall(_pad("GETFORCE"))
            assert _read_header(replica) == "FORCEREADY"
            _recv_exact(replica, 8 + 4 + 24 + 72 + 4)
            replica.sendall(_pad("EXIT"))
            outer.sendall(_pad("EXIT"))
            replica.close()
            outer.close()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=fake_cp2k, daemon=True)
    thread.start()
    hold = None
    replica = None
    try:
        replica, hold = open_replica_socket(path, _Alive(), timeout=10)
        client.serve(replica)
    finally:
        band.evaluate = original
        if replica is not None:
            replica.close()
        if hold is not None:
            hold.close()
        thread.join(timeout=5)
        if path.exists():
            path.unlink()
    assert errors == []
    assert hold is not None and hold.error == ""
    assert client.counter.calls == 1
    assert client.stop_reason == ""


def test_templates_name_the_registry_commit():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "cp2k")
    assert pinned["commit"] == COMMIT
    found = [row for row in list_templates(ROOT) if row.get("engine") == "cp2k-band"]
    assert [row["id"] for row in found] == ["cp2k/ci-neb", "cp2k/it-neb", "cp2k/string"]
    for row in found:
        assert row["commit"] == pinned["commit"]
        assert row["code"] == "CP2K"
        assert row["band"]["band_type"] in {"CI-NEB", "IT-NEB", "SM"}


@pytest.mark.skipif(binary_path() is None, reason="CP2K binary is not built on this machine")
def test_binary_answers_a_version_banner():
    import subprocess

    proc = subprocess.run([str(binary_path()), "--version"], check=False, capture_output=True, text=True, timeout=60)
    text = proc.stdout + proc.stderr
    assert "CP2K" in text
