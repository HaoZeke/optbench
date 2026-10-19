"""Counting potential server.

Compiled clients reach one ``HarnessCounter``. eonclient does it through
the ``ext_pot`` file exchange, which opens a short TCP connection per
force evaluation. rgmin and rgsaddle do it through ``RgpotCounter.calculate``,
the rgpot shape ``(positions, atomic numbers, box) -> (energy, forces)``.
The socket and that method charge the counter they are given.

The text dialogue is one command per connection. ``bcalc`` is the
binary form on a connection the client keeps open: atom count, nine
box floats, then ``Z x y z`` as little-endian int32 and float64. The
reply is ``ok``, two uint64 counts, the energy, and the forces.

``calculate`` then the atom count, nine box floats, and one
``Z x y z`` line per atom. The reply is ``ok``, the search-call count,
the verification-call count, the energy, and one ``fx fy fz`` line per
atom.

``reset`` zeroes the counter. ``count`` reports it. ``verify`` then
``on`` or ``off`` selects verification mode. ``bye`` closes.
"""

from __future__ import annotations

import socket
import struct
import threading
from collections.abc import Callable, Sequence

from .counter import ForceCallCap, HarnessCounter

Evaluate = Callable[
    [Sequence[float], Sequence[int] | None, Sequence[float] | None],
    tuple[float, Sequence[float]],
]


class RgpotCounter:
    """rgpot ``calculate`` on a harness counter.

    ``cache`` follows the counter. The server uses a counter that counts
    every request, which is what a compiled client does when it calls
    ``force`` twice at the same geometry.
    """

    def __init__(self, evaluate: Evaluate, counter: HarnessCounter | None = None):
        self.evaluate = evaluate
        self.counter = counter if counter is not None else HarnessCounter(cache=False)

    def calculate(
        self,
        positions: Sequence[float],
        atomic_numbers: Sequence[int] | None = None,
        box: Sequence[float] | None = None,
    ) -> tuple[float, tuple[float, ...]]:
        coords = tuple(float(value) for value in positions)
        fresh, cached = self.counter.charge(coords)
        if not fresh and cached is not None:
            return cached
        energy, forces = self.evaluate(coords, atomic_numbers, box)
        value = (float(energy), tuple(float(component) for component in forces))
        if len(value[1]) != len(coords):
            raise ValueError("forces must match positions")
        return self.counter.remember(coords, value)


def _read_floats(line: str, count: int) -> list[float]:
    parts = line.split()
    if len(parts) != count:
        raise ValueError(f"expected {count} floats, got {len(parts)}")
    return [float(part) for part in parts]


def handle_lines(lines: list[str], rgpot: RgpotCounter) -> list[str]:
    """Answer one command. ``lines`` does not include the trailing blanks."""
    if not lines:
        raise ValueError("empty command")
    command = lines[0]
    counter = rgpot.counter
    if command == "reset":
        counter.reset()
        return ["ok"]
    if command == "count":
        return ["count", str(counter.calls), str(counter.verify_calls)]
    if command == "verify":
        if len(lines) != 2 or lines[1] not in ("on", "off"):
            raise ValueError("verify needs on or off")
        counter._verifying = lines[1] == "on"
        return ["ok"]
    if command == "bye":
        return ["bye"]
    if command != "calculate":
        raise ValueError(f"unknown command {command}")
    if len(lines) < 3:
        raise ValueError("calculate needs a count and a box")
    atoms = int(lines[1])
    if atoms < 1:
        raise ValueError("atom count must be positive")
    box = _read_floats(lines[2], 9)
    body = lines[3:]
    if len(body) != atoms:
        raise ValueError(f"calculate announced {atoms} atoms and sent {len(body)}")
    positions: list[float] = []
    numbers: list[int] = []
    for row in body:
        parts = row.split()
        if len(parts) != 4:
            raise ValueError("an atom line is Z x y z")
        numbers.append(int(parts[0]))
        positions.extend(float(part) for part in parts[1:])
    energy, forces = rgpot.calculate(positions, numbers, box)
    reply = ["ok", str(counter.calls), str(counter.verify_calls), repr(energy)]
    for index in range(atoms):
        base = 3 * index
        reply.append(" ".join(repr(forces[base + axis]) for axis in range(3)))
    return reply


class PotServer:
    """One thread, one counter, one potential."""

    def __init__(
        self,
        evaluate: Evaluate,
        counter: HarnessCounter | None = None,
        host: str = "127.0.0.1",
        cap: int | None = None,
    ):
        if counter is None:
            counter = HarnessCounter(cache=False, cap=cap)
        elif cap is not None:
            counter.cap = int(cap)
        self.rgpot = RgpotCounter(evaluate, counter)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, 0))
        self._sock.listen(16)
        self._sock.settimeout(0.2)
        self.host = host
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="optbench-potserver", daemon=True)
        self._workers: list[threading.Thread] = []
        self._connections: list[socket.socket] = []
        self._worker_lock = threading.Lock()

    @property
    def counter(self) -> HarnessCounter:
        return self.rgpot.counter

    def start(self) -> PotServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._worker_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)
        with self._worker_lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=2.0)

    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _address = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(120.0)
            with self._worker_lock:
                self._connections.append(connection)
                worker = threading.Thread(
                    target=self._handle,
                    args=(connection,),
                    name="optbench-potserver-conn",
                    daemon=True,
                )
                self._workers.append(worker)
            worker.start()

    def _bcalc(self, reader: "_SocketReader") -> bytes:
        """One binary force call. The reply is ``ok`` plus the packed forces."""
        import numpy as np

        from .eon_harness import morse_shift, morse_shift_array

        count = struct.unpack("<I", reader.read(4))[0]
        if count < 1 or count > 100000:
            raise ValueError(f"bcalc atom count {count}")
        blob = reader.read(72 + count * 28)
        counter = self.rgpot.counter
        if self.rgpot.evaluate is morse_shift and not counter._cache_enabled:
            box = np.frombuffer(blob[:72], dtype="<f8")
            rows = np.frombuffer(blob[72:], dtype=np.uint8).reshape(count, 28)
            positions = rows[:, 4:].copy().view("<f8").reshape(-1)
            fresh, cached = counter.charge(())
            if not fresh and cached is not None:
                energy = float(cached[0])
                force_bytes = np.asarray(cached[1], dtype="<f8").tobytes()
            else:
                energy, forces = morse_shift_array(positions, box)
                force_bytes = np.ascontiguousarray(forces, dtype="<f8").tobytes()
            return b"ok\n" + struct.pack("<QQd", counter.calls, counter.verify_calls, float(energy)) + force_bytes
        box = struct.unpack("<9d", blob[:72])
        positions: list[float] = []
        numbers: list[int] = []
        for offset in range(72, len(blob), 28):
            number = struct.unpack_from("<i", blob, offset)[0]
            coords = struct.unpack_from("<3d", blob, offset + 4)
            numbers.append(int(number))
            positions.extend(coords)
        energy, forces = self.rgpot.calculate(positions, numbers, box)
        return b"ok\n" + struct.pack("<QQd", counter.calls, counter.verify_calls, float(energy)) + struct.pack(
            f"<{3 * count}d", *forces
        )

    def _handle(self, connection: socket.socket) -> None:
        reader = _SocketReader(connection)
        try:
            while not self._stop.is_set():
                try:
                    header = reader.readline()
                except EOFError:
                    return
                command = header.strip()
                if not command:
                    continue
                try:
                    if command == "bcalc":
                        try:
                            payload = self._bcalc(reader)
                        except ForceCallCap as exc:
                            connection.sendall(f"error\n{exc}\n".encode("utf-8"))
                            return
                        except Exception as exc:
                            connection.sendall(
                                f"error\n{exc.__class__.__name__}: {exc}\n".encode("utf-8")
                            )
                            return
                        connection.sendall(payload)
                        continue
                    lines = [command]
                    if command == "calculate":
                        count_line = reader.readline()
                        box_line = reader.readline()
                        lines.extend((count_line.strip(), box_line.strip()))
                        for _ in range(int(lines[1])):
                            lines.append(reader.readline().strip())
                    elif command == "verify":
                        lines.append(reader.readline().strip())
                    reply = handle_lines(lines, self.rgpot)
                except ForceCallCap as exc:
                    reply = ["error", str(exc)]
                except Exception as exc:
                    reply = ["error", exc.__class__.__name__ + ": " + str(exc)]
                connection.sendall(("\n".join(reply) + "\n").encode("utf-8"))
                if command == "bye":
                    return
        except (EOFError, OSError):
            return
        finally:
            try:
                connection.close()
            except OSError:
                pass


class _SocketReader:
    """Byte buffer over one socket. Text lines and binary bodies share it."""

    def __init__(self, connection: socket.socket):
        self._connection = connection
        self._data = b""

    def read(self, count: int) -> bytes:
        while len(self._data) < count:
            chunk = self._connection.recv(max(4096, count - len(self._data)))
            if not chunk:
                raise EOFError
            self._data += chunk
        body, self._data = self._data[:count], self._data[count:]
        return body

    def readline(self) -> str:
        while b"\n" not in self._data:
            chunk = self._connection.recv(4096)
            if not chunk:
                raise EOFError
            self._data += chunk
        index = self._data.index(b"\n")
        line, self._data = self._data[:index], self._data[index + 1 :]
        return line.decode("utf-8")


def _exchange(host: str, port: int, lines: list[str]) -> list[str]:
    with socket.create_connection((host, port), timeout=30.0) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        payload = "\n".join(lines) + "\n"
        connection.sendall(payload.encode("utf-8"))
        connection.shutdown(socket.SHUT_WR)
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        reply = []
        for row in reader:
            reply.append(row.rstrip("\n"))
        if not reply:
            raise RuntimeError("potential server closed without a reply")
        if reply[0] == "error":
            raise RuntimeError(reply[1] if len(reply) > 1 else "potential server error")
        return reply


def remote_reset(host: str, port: int) -> None:
    reply = _exchange(host, port, ["reset"])
    if reply != ["ok"]:
        raise RuntimeError(f"reset reply {reply}")


def remote_count(host: str, port: int) -> tuple[int, int]:
    reply = _exchange(host, port, ["count"])
    if len(reply) != 3 or reply[0] != "count":
        raise RuntimeError(f"count reply {reply}")
    return int(reply[1]), int(reply[2])


def remote_verify(host: str, port: int, enabled: bool) -> None:
    reply = _exchange(host, port, ["verify", "on" if enabled else "off"])
    if reply != ["ok"]:
        raise RuntimeError(f"verify reply {reply}")


def remote_bcalc(
    host: str,
    port: int,
    positions: Sequence[float],
    atomic_numbers: Sequence[int],
    box: Sequence[float],
) -> tuple[float, tuple[float, ...], int]:
    """One binary force call on a kept connection is not required.

    A fresh socket is enough for a check. The reply count is the
    server's search tally after the call.
    """
    atoms = len(atomic_numbers)
    if len(positions) != 3 * atoms:
        raise ValueError("positions and atomic numbers disagree")
    if len(box) != 9:
        raise ValueError("box is nine floats")
    payload = b"bcalc\n" + struct.pack("<I", atoms) + struct.pack("<9d", *[float(value) for value in box])
    for index, number in enumerate(atomic_numbers):
        base = 3 * index
        payload += struct.pack(
            "<i3d",
            int(number),
            float(positions[base]),
            float(positions[base + 1]),
            float(positions[base + 2]),
        )
    with socket.create_connection((host, port), timeout=30.0) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.sendall(payload)
        reader = _SocketReader(connection)
        header = reader.readline()
        if header == "error":
            raise RuntimeError(reader.readline())
        if header != "ok":
            raise RuntimeError(f"bcalc reply {header}")
        calls, _verify, energy = struct.unpack("<QQd", reader.read(24))
        forces = struct.unpack(f"<{3 * atoms}d", reader.read(24 * atoms))
    return float(energy), tuple(forces), int(calls)


def remote_calculate(
    host: str,
    port: int,
    positions: Sequence[float],
    atomic_numbers: Sequence[int],
    box: Sequence[float],
) -> tuple[float, tuple[float, ...], int]:
    """One force evaluation. The returned count is the server's search tally."""
    atoms = len(atomic_numbers)
    if len(positions) != 3 * atoms:
        raise ValueError("positions and atomic numbers disagree")
    if len(box) != 9:
        raise ValueError("box is nine floats")
    lines = [
        "calculate",
        str(atoms),
        " ".join(repr(float(value)) for value in box),
    ]
    for index, number in enumerate(atomic_numbers):
        base = 3 * index
        triple = " ".join(repr(float(positions[base + axis])) for axis in range(3))
        lines.append(f"{int(number)} {triple}")
    reply = _exchange(host, port, lines)
    if len(reply) != 4 + atoms or reply[0] != "ok":
        raise RuntimeError(f"calculate reply has {len(reply)} lines")
    calls = int(reply[1])
    energy = float(reply[3])
    forces: list[float] = []
    for row in reply[4:]:
        forces.extend(float(part) for part in row.split())
    if len(forces) != 3 * atoms:
        raise RuntimeError("force reply is short")
    return energy, tuple(forces), calls
