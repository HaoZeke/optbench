"""The GMIN socket speaks the harness Lennard-Jones potential."""

from __future__ import annotations

import socket
import struct

from pathlib import Path

import numpy as np
import pytest

from optbench.counter import HarnessCounter
from optbench.gmin_pot import STATUS_CAP, STATUS_OK, GminServer
from optbench.potentials import lennard_jones


def _exchange(port: int, coords: np.ndarray) -> tuple[int, float, int, np.ndarray]:
    natoms = coords.size // 3
    payload = struct.pack("<ii", 1, natoms) + np.ascontiguousarray(coords, dtype="<f8").tobytes()
    with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
        conn.sendall(payload)
        status, energy, calls = struct.unpack("<idq", _recv(conn, 20))
        grad = np.frombuffer(_recv(conn, natoms * 24), dtype="<f8").copy()
    return int(status), float(energy), int(calls), grad


def _recv(conn: socket.socket, nbytes: int) -> bytes:
    buf = bytearray(nbytes)
    view = memoryview(buf)
    got = 0
    while got < nbytes:
        chunk = conn.recv_into(view[got:])
        if chunk == 0:
            raise ConnectionError("closed")
        got += chunk
    return bytes(buf)


def test_server_matches_lennard_jones_and_counts_every_request():
    root = Path(__file__).resolve().parents[1]
    pos = np.loadtxt(root / "codes" / "gmin" / "reference" / "lj38.points")
    counter = HarnessCounter(cache=False, cap=4)
    server = GminServer(lennard_jones, counter, sloppy_rms=5.0e-3)
    server.start()
    try:
        status, energy, calls, grad = _exchange(server.port, pos.reshape(-1))
        again, energy2, calls2, _grad2 = _exchange(server.port, pos.reshape(-1))
    finally:
        server.stop()
    expected, expected_grad = lennard_jones(pos)
    assert status == STATUS_OK
    assert again == STATUS_OK
    assert energy == pytest.approx(expected)
    assert energy2 == pytest.approx(expected)
    assert grad == pytest.approx(expected_grad)
    assert calls == 1
    assert calls2 == 2
    snap = server.snapshot()
    assert snap["calls"] == 2
    assert snap["quench_energy"] == pytest.approx(expected)
    assert snap["cap_hit"] is False


def test_cap_is_not_counted():
    pos = np.zeros(9, dtype=np.float64)
    pos[3:6] = (1.5, 0.0, 0.0)
    counter = HarnessCounter(cache=False, cap=1)
    server = GminServer(lennard_jones, counter, sloppy_rms=1.0)
    server.start()
    try:
        first = _exchange(server.port, pos)
        second = _exchange(server.port, pos)
    finally:
        server.stop()
    assert first[0] == STATUS_OK
    assert first[2] == 1
    assert second[0] == STATUS_CAP
    assert second[2] == 1
    assert server.snapshot()["calls"] == 1
    assert server.snapshot()["cap_hit"] is True
