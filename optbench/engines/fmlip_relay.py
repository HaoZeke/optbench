"""Wales ``fmlip_relay`` server.

Speaks the socket in ``INTERFACES/pysocket/mlip_client.f90``. Each
``COMPUTE`` is one ``optbench.pes.evaluate`` call. The count file is
rewritten after every counted call. A call that would pass
``OPTBENCH_CAP`` is refused and not counted, and the last counted
coordinates stay on disk.
"""

from __future__ import annotations

import os
import socket
import struct
import sys

import numpy as np

MSG_COMPUTE = 1
MSG_QUIT = 2
MSG_PING = 3
STATUS_OK = 0
STATUS_ERR = 1


def _recv(conn: socket.socket, nbytes: int) -> bytes:
    chunks = []
    left = nbytes
    while left:
        piece = conn.recv(left)
        if not piece:
            raise ConnectionError("relay socket closed")
        chunks.append(piece)
        left -= len(piece)
    return b"".join(chunks)


def _kind_and_box():
    kind = os.environ.get("OPTBENCH_KIND", "lj")
    raw = os.environ.get("OPTBENCH_BOX", "").strip()
    if not raw:
        return kind, None
    values = np.asarray([float(item) for item in raw.split()], dtype=np.float64)
    return kind, values


def _write_count(calls: int) -> None:
    path = os.environ.get("OPTBENCH_COUNT_PATH")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{calls}\n")


def _write_last(coords: np.ndarray) -> None:
    path = os.environ.get("OPTBENCH_LAST_PATH")
    if path:
        np.savetxt(path, np.asarray(coords, dtype=np.float64), fmt="%.16e")


def _mark_capped() -> None:
    path = os.environ.get("OPTBENCH_CAPPED_PATH")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("cap\n")


def serve(port: int) -> None:
    from optbench.pes import evaluate

    kind, box = _kind_and_box()
    scale_e = float(os.environ.get("OPTBENCH_E_SCALE", "1"))
    scale_l = float(os.environ.get("OPTBENCH_L_SCALE", "1"))
    raw_cap = os.environ.get("OPTBENCH_CAP", "").strip()
    cap = int(raw_cap) if raw_cap else None
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    bound = listener.getsockname()[1]
    sys.stdout.write(f"READY {bound}\n")
    sys.stdout.flush()
    conn, _addr = listener.accept()
    calls = 0
    _write_count(calls)
    try:
        while True:
            try:
                header = _recv(conn, 8)
            except ConnectionError:
                break
            msg, natoms = struct.unpack("<ii", header)
            if msg == MSG_QUIT:
                break
            if msg == MSG_PING:
                conn.sendall(struct.pack("<i", STATUS_OK))
                continue
            if msg != MSG_COMPUTE:
                conn.sendall(struct.pack("<i", STATUS_ERR))
                break
            _recv(conn, 4 * natoms)
            pos = np.frombuffer(_recv(conn, 8 * 3 * natoms), dtype="<f8").reshape(natoms, 3).copy()
            _recv(conn, 8 * 9)
            _recv(conn, 4 * 3)
            _recv(conn, 4)
            _recv(conn, 4)
            _recv(conn, 4)
            if cap is not None and calls >= cap:
                _mark_capped()
                conn.sendall(struct.pack("<i", STATUS_ERR))
                break
            coords = pos / scale_l
            try:
                energy, gradient = evaluate(kind, coords, box)
            except Exception:
                conn.sendall(struct.pack("<i", STATUS_ERR))
                break
            force = (-np.asarray(gradient, dtype=np.float64) / scale_l) * scale_e
            calls += 1
            _write_last(coords)
            _write_count(calls)
            conn.sendall(struct.pack("<i", STATUS_OK))
            conn.sendall(struct.pack("<d", float(energy) * scale_e))
            conn.sendall(np.ascontiguousarray(force, dtype="<f8").tobytes())
            conn.sendall(struct.pack("<9d", *([0.0] * 9)))
    finally:
        _write_count(calls)
        conn.close()
        listener.close()


def main() -> None:
    argv = sys.argv[1:]
    port = 0
    index = 0
    while index < len(argv):
        flag = argv[index]
        value = argv[index + 1] if index + 1 < len(argv) else ""
        if flag == "--port":
            port = int(value)
        elif flag == "--e-to-ev":
            os.environ["OPTBENCH_E_SCALE"] = value
        elif flag == "--l-to-ang":
            os.environ["OPTBENCH_L_SCALE"] = value
        index += 2 if flag.startswith("--") else 1
    serve(port)


if __name__ == "__main__":
    main()
