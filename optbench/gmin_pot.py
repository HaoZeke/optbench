"""Binary potential server for the GMIN harness hook.

One TCP connection stays open for the life of a GMIN process. Each
request is one entry into ``optbench.potentials.lennard_jones`` and one
``HarnessCounter`` charge. The client does not repeat a coordinate for
free, so the counter cache stays off.

The wire format is little-endian and has no padding. A request is an
int32 kind. Kind 1 is followed by an int32 atom count and that many
float64 coordinates in atom-major order. Kind 2 closes the socket. The
reply is an int32 status, a float64 energy, an int64 call count, and
the float64 gradient. Status 0 is a result, 1 is the force-call cap
(the refused request is not counted), and 2 is an error.
"""

from __future__ import annotations

import ctypes
import math
import socket
import struct
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter

KIND_COMPUTE = 1
KIND_QUIT = 2
STATUS_OK = 0
STATUS_CAP = 1
STATUS_ERROR = 2

_HEADER = struct.Struct("<idq")


def _recv(conn: socket.socket, nbytes: int) -> bytes:
    buf = bytearray(nbytes)
    view = memoryview(buf)
    got = 0
    while got < nbytes:
        chunk = conn.recv_into(view[got:])
        if chunk == 0:
            raise ConnectionError("socket closed")
        got += chunk
    return buf


def _rms(grad: np.ndarray) -> float:
    if grad.size == 0:
        return math.inf
    return math.sqrt(float(np.dot(grad, grad)) / float(grad.size))


class GminServer:
    """Listen on 127.0.0.1 and serve one harness potential.

    ``evaluate`` returns ``(energy, gradient)``. The gradient is dE/dx
    in atom-major order. A converged quench is an evaluation whose RMS
    gradient, ``sqrt(sum(g^2)/(3N))``, is at or below ``sloppy_rms``.
    The tracked quench energy is the lowest such value.

    ``target`` is the problem reference energy plus its tolerance. The
    first evaluation at or below it records the call count and energy.
    That is the success count, the same stop the anneal runner uses.
    """

    def __init__(
        self,
        evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
        counter: HarnessCounter,
        *,
        sloppy_rms: float,
        target: float | None = None,
    ):
        self._evaluate = evaluate
        self._target = None if target is None else float(target)
        self._hit_calls: int | None = None
        self._hit_energy: float | None = None
        self._counter = counter
        self._sloppy = float(sloppy_rms)
        self._lock = threading.Lock()
        self._cap_hit = False
        self._errors = 0
        self._quench: float | None = None
        self._quench_rms: float | None = None
        self._quench_calls: int | None = None
        self._lowest: float | None = None
        self._listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen.bind(("127.0.0.1", 0))
        self._listen.listen(16)
        self._listen.settimeout(0.2)
        self._port = int(self._listen.getsockname()[1])
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        if self._accept_thread is not None:
            return
        self._accept_thread = threading.Thread(target=self._accept, name="gmin-pot", daemon=True)
        self._accept_thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._listen.close()
        except OSError:
            pass
        thread = self._accept_thread
        if thread is not None:
            thread.join(timeout=2.0)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": int(self._counter.calls),
                "cap_hit": bool(self._cap_hit),
                "errors": int(self._errors),
                "quench_energy": self._quench,
                "quench_rms": self._quench_rms,
                "quench_calls": self._quench_calls,
                "lowest_energy": self._lowest,
                "hit_calls": self._hit_calls,
                "hit_energy": self._hit_energy,
            }

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._listen.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                kind = struct.unpack("<i", _recv(conn, 4))[0]
                if kind == KIND_QUIT:
                    break
                if kind != KIND_COMPUTE:
                    break
                natoms = struct.unpack("<i", _recv(conn, 4))[0]
                if natoms <= 0 or natoms > 100_000:
                    self._send(conn, STATUS_ERROR, 0.0, np.zeros(0, dtype=np.float64))
                    break
                coords = np.frombuffer(_recv(conn, natoms * 24), dtype="<f8")
                self._one(conn, natoms, coords)
        except ConnectionError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _one(self, conn: socket.socket, natoms: int, coords: np.ndarray) -> None:
        try:
            self._counter.charge(())
        except ForceCallCap:
            with self._lock:
                self._cap_hit = True
            self._send(conn, STATUS_CAP, 0.0, np.zeros(natoms * 3, dtype=np.float64))
            return
        try:
            energy, grad = self._evaluate(np.asarray(coords, dtype=np.float64))
            grad = np.asarray(grad, dtype=np.float64).reshape(-1)
            energy = float(energy)
        except Exception:
            with self._lock:
                self._errors += 1
            self._send(conn, STATUS_ERROR, 0.0, np.zeros(natoms * 3, dtype=np.float64))
            return
        if grad.size != natoms * 3:
            with self._lock:
                self._errors += 1
            self._send(conn, STATUS_ERROR, 0.0, np.zeros(natoms * 3, dtype=np.float64))
            return
        rms = _rms(grad)
        with self._lock:
            if math.isfinite(energy) and (self._lowest is None or energy < self._lowest):
                self._lowest = energy
            if (
                self._target is not None
                and self._hit_calls is None
                and math.isfinite(energy)
                and energy <= self._target
            ):
                self._hit_calls = int(self._counter.calls)
                self._hit_energy = energy
            if math.isfinite(energy) and math.isfinite(rms) and rms <= self._sloppy:
                if self._quench is None or energy < self._quench:
                    self._quench = energy
                    self._quench_rms = rms
                    self._quench_calls = int(self._counter.calls)
        self._send(conn, STATUS_OK, energy, grad)

    def _send(self, conn: socket.socket, status: int, energy: float, grad: np.ndarray) -> None:
        body = np.ascontiguousarray(grad, dtype="<f8")
        conn.sendall(_HEADER.pack(int(status), float(energy), int(self._counter.calls)) + body.tobytes())


_LJ40 = None


def lennard_jones_compiled(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Compiled unshifted LJ with the harness parameters.

    Epsilon 1, sigma 1, cutoff 40, no shift. This is the same function as
    ``optbench.potentials.lennard_jones``. The campaign server calls it so a
    multi-million-call basin-hopping run is not bound to the Python pair loop.
    """
    global _LJ40
    if _LJ40 is None:
        source = Path(__file__).resolve().parents[1] / "codes" / "gmin" / "lj40.c"
        library = Path(tempfile.gettempdir()) / "optbench-lj40.so"
        if not library.is_file() or library.stat().st_mtime < source.stat().st_mtime:
            subprocess.check_call(
                ["gcc", "-O3", "-shared", "-fPIC", str(source), "-o", str(library)],
            )
        loaded = ctypes.CDLL(str(library))
        loaded.optbench_lj40.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _LJ40 = loaded
    pos = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    natoms = int(pos.shape[0])
    energy = ctypes.c_double()
    grad = np.zeros(natoms * 3, dtype=np.float64)
    _LJ40.optbench_lj40(
        natoms,
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(energy),
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return float(energy.value), grad


_BLJ = None


def binary_lennard_jones_compiled(
    coords: np.ndarray,
    species: np.ndarray,
    *,
    sigma_aa: float,
    sigma_bb: float,
    epsilon_aa: float,
    epsilon_bb: float,
    epsilon_ab: float,
) -> tuple[float, np.ndarray]:
    """Compiled ``optbench.potentials.binary_lennard_jones`` with no cutoff.

    Species 0 is A and 1 is B. The unlike sigma is the arithmetic mean.
    """
    global _BLJ
    if _BLJ is None:
        source = Path(__file__).resolve().parents[1] / "codes" / "gmin" / "blj.c"
        library = Path(tempfile.gettempdir()) / "optbench-blj.so"
        if not library.is_file() or library.stat().st_mtime < source.stat().st_mtime:
            subprocess.check_call(
                ["gcc", "-O3", "-shared", "-fPIC", str(source), "-o", str(library)],
            )
        loaded = ctypes.CDLL(str(library))
        loaded.optbench_blj.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _BLJ = loaded
    pos = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
    natoms = int(pos.shape[0])
    labels = np.ascontiguousarray(species, dtype=np.int32)
    if labels.shape != (natoms,):
        raise ValueError("species must name one label per atom")
    energy = ctypes.c_double()
    grad = np.zeros(natoms * 3, dtype=np.float64)
    _BLJ.optbench_blj(
        natoms,
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        float(sigma_aa),
        float(sigma_bb),
        float(epsilon_aa),
        float(epsilon_bb),
        float(epsilon_ab),
        ctypes.byref(energy),
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return float(energy.value), grad
