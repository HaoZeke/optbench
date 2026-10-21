"""Socket potential for OPTIM's ASE relay.

OPTIM spawns ``pysocket/fmlip_relay.py`` and speaks the fmlip wire
protocol. This process is that server. Each COMPUTE request is one
harness force call. The counter is flushed before the reply, so a
client that dies still leaves the tally on disk. A request at the cap
is refused, not evaluated, and not counted.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
from pathlib import Path

import numpy as np

from .counter import ForceCallCap, HarnessCounter
from .surfaces import potential_from_problem

MSG_COMPUTE = 1
MSG_QUIT = 2
MSG_PING = 3
STATUS_OK = 0
STATUS_ERR = 1
_INT = "<i"
_F64 = "<d"


def _recv(conn: socket.socket, count: int) -> bytes:
    parts = bytearray()
    while len(parts) < count:
        chunk = conn.recv(count - len(parts))
        if not chunk:
            raise EOFError("relay client closed the socket")
        parts += chunk
    return bytes(parts)


def _save(
    path: Path,
    calls: int,
    capped: bool,
    energy: float | None,
    positions: list[list[float]] | None = None,
) -> None:
    payload = {"calls": int(calls), "capped": bool(capped), "energy": energy}
    if positions is not None:
        payload["positions"] = positions
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def _evaluator(job: dict):
    raw = potential_from_problem({"potential": job["potential"]})
    cell = np.asarray(job.get("cell") or [0.0, 0.0, 0.0], dtype=np.float64)

    def evaluate(positions: np.ndarray):
        return raw(positions, cell)

    return evaluate


def serve(job: dict, port: int = 0, ready=None) -> None:
    """Bind, announce ``READY <port>``, and answer one OPTIM client."""
    counter_path = Path(job["counter"])
    cap = None if job.get("cap") is None else int(job["cap"])
    expected = None if job.get("natoms") is None else int(job["natoms"])
    evaluate = _evaluator(job)
    counter = HarnessCounter(cache=False, cap=cap)
    _save(counter_path, 0, False, None)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", int(port)))
    server.listen(1)
    server.settimeout(120.0)
    actual = int(server.getsockname()[1])
    if ready is not None:
        ready.put(actual)
    sys.stdout.write(f"READY {actual}\n")
    sys.stdout.flush()
    try:
        conn, _addr = server.accept()
    except TimeoutError:
        return
    finally:
        server.close()
    conn.settimeout(None)
    last_energy: float | None = None
    last_positions: list[list[float]] | None = None
    try:
        while True:
            try:
                header = _recv(conn, 4)
            except EOFError:
                break
            message = struct.unpack(_INT, header)[0]
            if message == MSG_QUIT:
                break
            if message == MSG_PING:
                conn.sendall(struct.pack(_INT, STATUS_OK))
                continue
            if message != MSG_COMPUTE:
                conn.sendall(struct.pack(_INT, STATUS_ERR))
                break
            natoms = struct.unpack(_INT, _recv(conn, 4))[0]
            _recv(conn, 4 * natoms)
            positions = np.frombuffer(_recv(conn, 8 * 3 * natoms), dtype=np.float64).reshape(natoms, 3).copy()
            _recv(conn, 8 * 9)
            _recv(conn, 4 * 3)
            _recv(conn, 4)
            _recv(conn, 4)
            _recv(conn, 4)
            if expected is not None and natoms != expected:
                _save(counter_path, counter.calls, False, last_energy)
                conn.sendall(struct.pack(_INT, STATUS_ERR))
                break
            try:
                counter.charge(())
                energy, gradient = evaluate(positions)
            except ForceCallCap:
                _save(counter_path, counter.calls, True, last_energy)
                conn.sendall(struct.pack(_INT, STATUS_ERR))
                break
            except Exception:
                _save(counter_path, counter.calls, False, last_energy)
                conn.sendall(struct.pack(_INT, STATUS_ERR))
                break
            last_energy = float(energy)
            last_positions = positions.tolist()
            _save(counter_path, counter.calls, False, last_energy, last_positions)
            forces = -np.asarray(gradient, dtype=np.float64).reshape(-1)
            conn.sendall(struct.pack(_INT, STATUS_OK))
            conn.sendall(struct.pack(_F64, last_energy))
            conn.sendall(np.ascontiguousarray(forces, dtype=np.float64).tobytes())
            conn.sendall(np.zeros(9, dtype=np.float64).tobytes())
    finally:
        conn.close()
        _save(
            counter_path,
            counter.calls,
            counter.cap is not None and counter.calls >= counter.cap,
            last_energy,
            last_positions,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="OPTIM harness potential relay")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--backend", default="harness")
    parser.add_argument("--model", default="job")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--e-to-ev", default="1")
    parser.add_argument("--l-to-ang", default="1")
    args = parser.parse_args(argv)
    job_path = os.environ.get("OPTBENCH_OPTIM_JOB")
    if not job_path:
        raise SystemExit("OPTBENCH_OPTIM_JOB is not set")
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    serve(job, args.port)


if __name__ == "__main__":
    main()
