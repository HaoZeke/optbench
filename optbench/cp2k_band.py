"""CP2K BAND campaigns through the i-PI server.

CP2K 2026.2 is the serial ``cp2k.ssmp`` binary (``CP2K_BIN``, or the
Release build under ``/tmp/cp2k-build``). BAND runs one image at a time
on one force environment. Python is the i-PI client. Each POSDATA is
one ``HarnessCounter`` call into ``pes.evaluate``, with the socket
cache off, so a repeated geometry still counts.

The potential is the problem pair potential. Lennard-Jones frames are
shifted into a non-periodic cell; the value depends on distances.
Morse platinum keeps the archived orthorhombic cell and periodic
boundaries. Fixed atoms stay on the reactant. Their socket forces are
zero. Alignment and replica rotation stay at the CP2K default when
every atom is free and the cell is open. A fixed atom or a periodic
cell turns both off, so the archived endpoints stay the start.

A BAND run builds two i-PI force environments on the same unix path.
The outer one handshakes and then sits idle. The replica environment
binds the path again and sends the positions that the harness counts.

The DIIS optimizer is the CP2K default. ``MAX_STEPS`` is the force
cap. ``MAX_STEPSIZE`` is the problem ``max_move`` in bohr. ``MAX_FORCE``
is the problem force norm changed to hartree/bohr and divided by
``sqrt(n_free * 3)``, so a band force at that infinity-norm has a free
L2 at or below the problem stop when the band force matches the true
force. ``MAX_DR``, ``RMS_DR``, and ``RMS_FORCE`` are loose, and the
force criterion is the one that can stop the band. CI-NEB also stops
when the highest interior image already meets the true force norm.
The recorded cost is the harness total across every image.

A clean stop under the cap is a success when the free-atom Cartesian
gradient L2 is within the problem force norm and the vibrational
index is 1. Hessian finite differences go through ``verification``.

The harness classes are crash, divergence, cap, and wrong_stationary.
There is no stalled class. Divergence is a non-finite energy or
geometry. A ``BAND TASK COMPLETED`` line with a finite geometry and a
free-atom force still above the stop is wrong_stationary: the nudged
band force can meet ``MAX_FORCE`` while the true gradient does not.
An exit at ``MAX_STEPS`` without that banner, under the force cap, is
the cap.

Replica files are named in the run directory. ``COORD_FILE_NAME`` is a
long string, and CP2K reads a quoted value in 80-character pieces. A
closing quote that lands on that boundary aborts the input.
"""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import statistics
import struct
import threading
import time
import tomllib
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .counter import ForceCallCap, HarnessCounter
from .failures import classify_failure
from .pes import evaluate, interpolate_free, load_pairs
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance, write_starts

# physcon.F at CP2K 2026.2. Positions and forces on the socket are
# hartree and bohr. The harness potential is eV and angstrom.
_C_LIGHT = 299792458.0
_H_PLANCK = 6.62606896e-34
_RYDBERG = 10973731.568527
_E_CHARGE = 1.602176487e-19
_A_BOHR = 0.52917720859e-10
_JOULE = 2.0 * _RYDBERG * _H_PLANCK * _C_LIGHT
EV_PER_HARTREE = _JOULE / _E_CHARGE
ANGSTROM_PER_BOHR = 1.0e10 * _A_BOHR
_HEADER = 12
_FD_STEP = 1e-5
_LOOSE = 10.0
_K_SPRING = 0.02
_COMMIT = "67b5da876dd6a76b8b021d5a04d1c81ba79a4c50"

_BINARY: Path | None = None


def binary_path() -> Path | None:
    """The serial CP2K executable, or None when this machine has no build."""
    global _BINARY
    env = os.environ.get("CP2K_BIN")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            _BINARY = path
            return path
    if _BINARY is not None and _BINARY.is_file():
        return _BINARY
    roots = (Path("/tmp/cp2k-build"), Path("/tmp/cp2k"))
    for root in roots:
        if not root.is_dir():
            continue
        for name in ("cp2k.ssmp", "cp2k.sopt"):
            for path in root.rglob(name):
                if path.is_file() and os.access(path, os.X_OK):
                    _BINARY = path
                    return path
    return None


def max_force_internal(force_norm: float, n_free: int) -> float:
    """Problem L2 force norm as a CP2K band infinity-norm in hartree/bohr."""
    if n_free < 1:
        raise ValueError("a band needs a free atom")
    per_component = float(force_norm) / math.sqrt(n_free * 3)
    return per_component * ANGSTROM_PER_BOHR / EV_PER_HARTREE


def max_step_internal(max_move: float) -> float:
    """Problem ``max_move`` in angstrom, as bohr."""
    return float(max_move) / ANGSTROM_PER_BOHR


def initial_frames(
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    n_interior: int,
    box: np.ndarray | None,
) -> list[np.ndarray]:
    """Full-atom linear path. Fixed atoms stay on the reactant."""
    free_path = interpolate_free(reactant, product, free, n_interior, box)
    frames = []
    mask = np.asarray(free, dtype=bool)
    for row in free_path:
        full = np.array(reactant, dtype=np.float64, copy=True)
        full[mask] = np.asarray(row, dtype=np.float64).reshape((-1, 3))
        frames.append(full)
    return frames


def place_frames(
    frames: list[np.ndarray],
    box: np.ndarray | None,
) -> tuple[list[np.ndarray], tuple[float, float, float], bool]:
    """Cell that holds the path.

    An open cell is shifted so every coordinate is at least 1 angstrom.
    A periodic cell is the archived box, unshifted.
    """
    copied = [np.array(frame, dtype=np.float64, copy=True) for frame in frames]
    if box is not None:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)
        return copied, (float(lengths[0]), float(lengths[1]), float(lengths[2])), True
    stacked = np.concatenate(copied, axis=0)
    low = float(stacked.min())
    shift = 0.0 if low >= 1.0 else 1.0 - low
    if shift:
        copied = [frame + shift for frame in copied]
    high = max(float(frame.max()) for frame in copied)
    length = max(20.0, high + 5.0)
    return copied, (length, length, length), False


def select_candidate(evals: list[dict], n_images: int) -> dict | None:
    """Highest-energy interior image of the last complete sweep."""
    if not evals:
        return None
    complete = (len(evals) // n_images) * n_images
    window = evals[complete - n_images : complete] if complete >= n_images else list(evals)
    interior = window[1:-1] if len(window) >= 3 else window
    if not interior:
        return None

    def energy_of(row: dict) -> float:
        energy = float(row["energy"])
        return energy if math.isfinite(energy) else -math.inf

    return max(interior, key=energy_of)


def failure_reasons(
    *,
    force: float,
    calls: int,
    cap: int,
    hit_cap: bool,
    diverged: bool,
    crashed: bool,
    force_norm: float,
) -> list[str]:
    """Harness failure reasons before the Hessian test."""
    reasons: list[str] = []
    if crashed:
        reasons.append("crash")
    if hit_cap or calls >= cap:
        reasons.append("cap")
    if diverged or (not math.isfinite(force)) or force > force_norm:
        reasons.append("force_norm")
    return reasons


def label_band_failure(
    reasons: list[str],
    *,
    band_completed: bool,
    step_limit: bool,
    finite: bool,
) -> tuple[str | None, list[str]]:
    """One class for a CP2K band.

    ``step_limit`` is a clean exit at ``MAX_STEPS`` with no
    ``BAND TASK COMPLETED`` line and with harness calls still under
    the cap. A finished finite band above the force stop is
    ``wrong_stationary``. A non-finite result stays ``divergence``.
    """
    labeled = list(reasons)
    # A step-limit exit that already meets the force stop can still be a
    # saddle. The cap applies when that exit is still above the stop.
    if (
        step_limit
        and "crash" not in labeled
        and "cap" not in labeled
        and "force_norm" in labeled
    ):
        labeled.append("cap")
    failure = classify_failure(labeled)
    if (
        failure == "divergence"
        and band_completed
        and finite
        and "divergence" not in labeled
    ):
        failure = "wrong_stationary"
    return failure, labeled


def _real(value: float) -> str:
    return format(float(value), ".12g")


def _optimize_lines(profile: dict, *, max_step: float, max_steps: int) -> list[str]:
    """``OPTIMIZE_BAND`` block. Missing keys keep the CP2K DIIS default."""
    opt_type = str(profile.get("opt_type") or "DIIS").upper()
    if opt_type not in {"DIIS", "MD"}:
        raise ValueError(f"unknown band optimizer {opt_type}")
    lines = ["    &OPTIMIZE_BAND", f"      OPT_TYPE {opt_type}"]
    if opt_type == "MD":
        lines.extend(
            [
                "      &MD",
                f"        MAX_STEPS {max_steps}",
                f"        TIMESTEP {_real(profile['timestep'])}",
                f"        TEMPERATURE {_real(profile['temperature'])}",
                "        &VEL_CONTROL",
                f"          ANNEALING {_real(profile['annealing'])}",
                "          PROJ_VELOCITY_VERLET T",
                "        &END VEL_CONTROL",
                "      &END MD",
            ]
        )
    else:
        lines.extend(
            [
                "      &DIIS",
                f"        MAX_STEPS {max_steps}",
                f"        MAX_STEPSIZE {max_step:.16e}",
            ]
        )
        if profile.get("stepsize") is not None:
            lines.append(f"        STEPSIZE {_real(profile['stepsize'])}")
        if profile.get("n_diis") is not None:
            lines.append(f"        N_DIIS {int(profile['n_diis'])}")
        if profile.get("no_ls"):
            lines.append("        NO_LS")
        lines.append("      &END DIIS")
    lines.append("    &END OPTIMIZE_BAND")
    return lines


def render_input(
    *,
    band_type: str,
    n_replica: int,
    socket_path: str,
    replica_paths: list[Path],
    positions: np.ndarray,
    cell: tuple[float, float, float],
    periodic: bool,
    fixed: list[int],
    max_force: float,
    max_step: float,
    max_steps: int,
    align: bool,
    k_spring: float = _K_SPRING,
    profile: dict | None = None,
) -> str:
    """One BAND input. Coordinates in the replica files are angstrom."""
    if band_type not in {"CI-NEB", "IT-NEB", "SM"}:
        raise ValueError(f"unknown band type {band_type}")
    if len(replica_paths) != n_replica:
        raise ValueError("one replica file per image, endpoints included")
    lines = [
        "&GLOBAL",
        "  PROJECT band",
        "  RUN_TYPE BAND",
        "  PRINT_LEVEL LOW",
        "&END GLOBAL",
        "&MOTION",
        "  &DRIVER",
        f'    HOST "{socket_path}"',
        "    PORT 12345",
        "    UNIX",
        "  &END DRIVER",
        "  &BAND",
        f"    BAND_TYPE {band_type}",
        f"    NUMBER_OF_REPLICA {n_replica}",
        "    NPROC_REP 1",
        f"    K_SPRING {float(k_spring)}",
        f"    ROTATE_FRAMES {'T' if align else 'F'}",
        f"    ALIGN_FRAMES {'T' if align else 'F'}",
        "    &CONVERGENCE_CONTROL",
        f"      MAX_FORCE {max_force:.16e}",
        f"      MAX_DR {_LOOSE}",
        f"      RMS_DR {_LOOSE}",
        f"      RMS_FORCE {_LOOSE}",
        "    &END CONVERGENCE_CONTROL",
        *_optimize_lines(profile or {}, max_step=max_step, max_steps=max_steps),
    ]
    if band_type == "CI-NEB":
        lines.extend(
            [
                "    &CI_NEB",
                "      NSTEPS_IT 5",
                "    &END CI_NEB",
            ]
        )
    for path in replica_paths:
        # The process starts in the replica directory, so the file name
        # is the path CP2K opens. A long quoted path can abort the parser.
        lines.extend(
            [
                "    &REPLICA",
                f'      COORD_FILE_NAME "{Path(path).name}"',
                "    &END REPLICA",
            ]
        )
    lines.append("  &END BAND")
    if fixed:
        lines.extend(["  &CONSTRAINT", "    &FIXED_ATOMS", "      COMPONENTS_TO_FIX xyz"])
        for start in range(0, len(fixed), 16):
            chunk = " ".join(str(index) for index in fixed[start : start + 16])
            lines.append(f"      LIST {chunk}")
        lines.extend(["    &END FIXED_ATOMS", "  &END CONSTRAINT"])
    lines.append("&END MOTION")
    periodic_line = "XYZ" if periodic else "NONE"
    lines.extend(
        [
            "&FORCE_EVAL",
            "  METHOD IPI",
            "  &SUBSYS",
            "    &CELL",
            f"      ABC {cell[0]:.10g} {cell[1]:.10g} {cell[2]:.10g}",
            f"      PERIODIC {periodic_line}",
            "    &END CELL",
            "    &COORD",
        ]
    )
    for x_ang, y_ang, z_ang in np.asarray(positions, dtype=np.float64):
        lines.append(f"      H {x_ang:.12f} {y_ang:.12f} {z_ang:.12f}")
    lines.extend(["    &END COORD", "  &END SUBSYS", "&END FORCE_EVAL", ""])
    return "\n".join(lines)


def _pad(text: str) -> bytes:
    raw = text.encode("ascii")
    if len(raw) > _HEADER:
        raise ValueError(f"i-PI header {text!r} is longer than {_HEADER}")
    return raw.ljust(_HEADER)


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    parts = []
    got = 0
    while got < count:
        chunk = sock.recv(count - got)
        if not chunk:
            raise EOFError(f"socket closed after {got} of {count} bytes")
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def _read_header(sock: socket.socket) -> str:
    return _recv_exact(sock, _HEADER).decode("ascii").strip()


def _read_i32(sock: socket.socket) -> int:
    return struct.unpack("=i", _recv_exact(sock, 4))[0]


def _read_f64(sock: socket.socket, count: int) -> np.ndarray:
    raw = _recv_exact(sock, 8 * count)
    return np.frombuffer(raw, dtype=np.float64).copy()


def _send_header(sock: socket.socket, text: str) -> None:
    sock.sendall(_pad(text))


class IpiClient:
    """One i-PI client for one BAND run.

    ``stop_reason`` is ``cap``, ``candidate``, ``diverge``, ``crash``,
    or ``""`` when CP2K closes the socket itself.
    """

    def __init__(
        self,
        *,
        kind: str,
        free: np.ndarray,
        box: np.ndarray | None,
        n_images: int,
        force_norm: float,
        cap: int,
        band_type: str,
        counter: HarnessCounter,
    ):
        self.kind = kind
        self.free = np.asarray(free, dtype=bool)
        self.box = None if box is None else np.asarray(box, dtype=np.float64)
        self.n_images = int(n_images)
        self.force_norm = float(force_norm)
        self.cap = int(cap)
        self.band_type = band_type
        self.counter = counter
        self.evals: list[dict] = []
        self.pending: tuple[float, np.ndarray] | None = None
        self.stop_reason = ""
        self.peer_error = ""
        self._closing = False

    def serve(self, sock: socket.socket) -> None:
        sock.settimeout(180.0)
        try:
            while not self._closing:
                header = _read_header(sock)
                if header == "STATUS":
                    self._status(sock)
                elif header == "POSDATA":
                    self._positions(sock)
                elif header == "GETFORCE":
                    self._forces(sock)
                elif header == "EXIT":
                    return
                else:
                    self.stop_reason = "crash"
                    self.peer_error = f"unexpected header {header}"
                    self._closing = True
                    sock.close()
                    return
        except (EOFError, ConnectionError, TimeoutError, OSError) as exc:
            self.peer_error = f"{type(exc).__name__}: {exc}"

    def _status(self, sock: socket.socket) -> None:
        if self.pending is not None:
            _send_header(sock, "HAVEDATA")
            return
        if self.counter.calls >= self.cap:
            self._stop(sock, "cap")
            return
        if self._candidate_ready():
            self._stop(sock, "candidate")
            return
        _send_header(sock, "READY")

    def _stop(self, sock: socket.socket, reason: str) -> None:
        self.stop_reason = reason
        self._closing = True
        try:
            sock.close()
        except OSError:
            pass

    def _positions(self, sock: socket.socket) -> None:
        _read_f64(sock, 9)
        _read_f64(sock, 9)
        n_atoms = _read_i32(sock)
        if n_atoms < 1:
            self.peer_error = f"natom {n_atoms}"
            self._stop(sock, "crash")
            return
        coords_bohr = _read_f64(sock, n_atoms * 3)
        if self.counter.calls >= self.cap:
            self._stop(sock, "cap")
            return
        try:
            self.counter.charge(())
        except ForceCallCap:
            self._stop(sock, "cap")
            return
        coords = (coords_bohr * ANGSTROM_PER_BOHR).reshape((n_atoms, 3))
        try:
            energy, gradient = evaluate(self.kind, coords, self.box)
        except Exception as exc:
            self.peer_error = type(exc).__name__
            self.evals.append(
                {
                    "positions": coords,
                    "energy": float("nan"),
                    "force": float("nan"),
                    "gradient": np.zeros_like(coords),
                }
            )
            self._stop(sock, "crash")
            return
        gradient = np.asarray(gradient, dtype=np.float64)
        free_force = float(np.linalg.norm(gradient[self.free]))
        self.evals.append(
            {
                "positions": coords,
                "energy": float(energy),
                "force": free_force,
                "gradient": gradient,
            }
        )
        if not math.isfinite(float(energy)) or not np.all(np.isfinite(gradient)):
            self._stop(sock, "diverge")
            return
        sent = -gradient * ANGSTROM_PER_BOHR / EV_PER_HARTREE
        sent[~self.free] = 0.0
        self.pending = (float(energy) / EV_PER_HARTREE, sent.reshape(-1))

    def _forces(self, sock: socket.socket) -> None:
        if self.pending is None:
            self.peer_error = "GETFORCE without a position"
            self._stop(sock, "crash")
            return
        energy, forces = self.pending
        self.pending = None
        _send_header(sock, "FORCEREADY")
        sock.sendall(struct.pack("=d", energy))
        n_atoms = forces.size // 3
        sock.sendall(struct.pack("=i", n_atoms))
        sock.sendall(np.ascontiguousarray(forces, dtype=np.float64).tobytes())
        sock.sendall(np.zeros(9, dtype=np.float64).tobytes())
        sock.sendall(struct.pack("=i", 0))

    def _candidate_ready(self) -> bool:
        if self.band_type != "CI-NEB":
            return False
        if len(self.evals) < self.n_images or len(self.evals) % self.n_images != 0:
            return False
        chosen = select_candidate(self.evals, self.n_images)
        if chosen is None:
            return False
        force = float(chosen["force"])
        return math.isfinite(force) and force <= self.force_norm


def _write_xyz(path: Path, coords: np.ndarray) -> None:
    lines = [str(len(coords)), "cp2k band image"]
    for x_ang, y_ang, z_ang in np.asarray(coords, dtype=np.float64):
        lines.append(f"H {x_ang:.12f} {y_ang:.12f} {z_ang:.12f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _OuterHold:
    """Keep the unused outer i-PI socket open so the replica server can bind."""

    def __init__(self) -> None:
        self.sock: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.error = ""

    def start(self, sock: socket.socket) -> None:
        self.sock = sock
        self.thread = threading.Thread(target=self._run, name="cp2k-outer-ipi", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self.thread
        if thread is not None:
            thread.join(timeout=5)

    def _run(self) -> None:
        sock = self.sock
        assert sock is not None
        sock.settimeout(0.5)
        try:
            while not self.stop.is_set():
                try:
                    header = _read_header(sock)
                except TimeoutError:
                    continue
                if header == "STATUS":
                    _send_header(sock, "READY")
                elif header == "EXIT":
                    return
                elif header == "POSDATA":
                    self.error = "outer force env requested a position"
                    return
                else:
                    self.error = f"outer header {header}"
                    return
        except (EOFError, ConnectionError, OSError):
            return


def _wait_socket(path: Path, proc, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if proc.poll() is not None:
            raise RuntimeError(f"cp2k exited {proc.returncode} before the socket was bound")
        time.sleep(0.05)
    raise TimeoutError(f"no socket at {path}")


def _connect(path: Path, proc, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"cp2k exited {proc.returncode} before the client connected")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(path))
            return sock
        except OSError as exc:
            last = str(exc)
            sock.close()
            time.sleep(0.05)
    raise TimeoutError(last or f"connect {path}")


def _socket_inode(path: Path) -> int | None:
    try:
        return path.stat().st_ino
    except FileNotFoundError:
        return None


def open_replica_socket(path: Path, proc, timeout: float = 180.0) -> tuple[socket.socket, _OuterHold]:
    """Connect to the replica i-PI server.

    The outer force environment accepts first and deletes the socket
    file. The replica environment binds the same path, which gets a
    new inode. The returned hold keeps the outer connection open until
    the run ends.
    """
    _wait_socket(path, proc, timeout)
    outer_inode = _socket_inode(path)
    outer = _connect(path, proc, timeout)
    hold = _OuterHold()
    hold.start(outer)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            hold.close()
            raise RuntimeError(f"cp2k exited {proc.returncode} before the replica socket")
        current = _socket_inode(path)
        if current is not None and current != outer_inode:
            try:
                replica = _connect(path, proc, max(1.0, deadline - time.monotonic()))
            except TimeoutError:
                hold.close()
                raise
            return replica, hold
        time.sleep(0.01)
    hold.close()
    raise TimeoutError(f"replica socket did not rebind at {path}")


def _kill(proc) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    try:
        proc.wait(timeout=15)
    except Exception:
        pass


_LOG_MARKERS = ("BAND TASK COMPLETED", "Band task not yet converged")


def _log_tail(path: Path, limit: int = 4000) -> str:
    """Ending of a CP2K log, plus the band-completion lines.

    The citation dump after a finished band is longer than a short
    tail, so the completion line is copied from the whole file.
    """
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = [needle for needle in _LOG_MARKERS if needle in text]
    tail = text[-limit:]
    if not markers:
        return tail
    return "\n".join(markers) + "\n" + tail


def _judge(client: IpiClient, *, returncode: int | None, log_text: str) -> dict:
    from .rgsaddle_band import vibrational_index

    chosen = select_candidate(client.evals, client.n_images)
    if chosen is None:
        energy = float("nan")
        force = float("nan")
        positions = None
    else:
        energy = float(chosen["energy"])
        force = float(chosen["force"])
        positions = np.asarray(chosen["positions"], dtype=np.float64)
    band_completed = "BAND TASK COMPLETED" in log_text
    intentional = client.stop_reason in {"cap", "candidate", "diverge"}
    crashed = (not intentional) and (
        client.stop_reason == "crash" or (returncode not in (0, None) and not band_completed)
    )
    if client.peer_error and client.stop_reason == "" and returncode not in (0, None):
        crashed = True
    calls = int(client.counter.calls)
    reasons = failure_reasons(
        force=force,
        calls=calls,
        cap=client.cap,
        hit_cap=client.stop_reason == "cap",
        diverged=client.stop_reason == "diverge",
        crashed=crashed,
        force_norm=client.force_norm,
    )
    step_limit = (
        returncode == 0
        and not band_completed
        and client.stop_reason == ""
        and not crashed
        and calls < client.cap
    )
    hessian_index = None
    project_rigid = bool(np.all(client.free) and int(np.count_nonzero(client.free)) >= 2 and client.kind == "lj")
    if "crash" not in reasons and "cap" not in reasons and "force_norm" not in reasons and positions is not None:
        point = positions[client.free].reshape(-1)
        template = np.array(positions, dtype=np.float64, copy=True)
        mask = client.free
        kind = client.kind
        box = client.box

        def free_gradient(flat):
            full = np.array(template, copy=True)
            full[mask] = np.asarray(flat, dtype=np.float64).reshape((-1, 3))
            client.counter.charge(())
            _energy, gradient = evaluate(kind, full, box)
            return np.asarray(gradient, dtype=np.float64)[mask].reshape(-1)

        with client.counter.verification():
            hessian = numerical_hessian(free_gradient, point.tolist(), step=_FD_STEP)
        hessian_index = vibrational_index(
            np.asarray(hessian, dtype=np.float64),
            point.reshape((-1, 3)),
            project_rigid,
        )
        if hessian_index != 1:
            reasons.append("hessian_index")
    finite = math.isfinite(energy) and math.isfinite(force)
    failure, reasons = label_band_failure(
        reasons,
        band_completed=band_completed,
        step_limit=step_limit,
        finite=finite,
    )
    finite_energy = energy if math.isfinite(energy) else None
    finite_force = force if math.isfinite(force) else None
    return {
        "force_calls": int(client.counter.calls),
        "verify_calls": int(client.counter.verify_calls),
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": finite_energy,
        "force_norm": finite_force,
        "hessian_index": hessian_index,
        "converged": bool(band_completed and failure is None),
        "cp2k_converged": bool(band_completed),
        "stop_reason": client.stop_reason,
        "returncode": returncode,
    }


def run_search(job: dict) -> dict:
    """One pair. ``job`` is plain data so a spawn worker can unpickle it."""
    _threads_one()
    name = str(job["name"])
    try:
        return _run_search(job)
    except Exception as exc:
        return _crash_record(name, 0, 0, f"{type(exc).__name__}: {exc}")


def _run_search(job: dict) -> dict:
    import subprocess

    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = np.asarray(job["product"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    settings = job["settings"]
    binary = Path(job["binary"])
    n_interior = int(settings["n_interior"])
    n_images = n_interior + 2
    n_free = int(np.count_nonzero(free))
    frames = initial_frames(reactant, product, free, n_interior, box)
    frames, cell, periodic = place_frames(frames, box)
    align = bool(box is None and np.all(free))
    cap = int(settings["cap"])
    force_norm = float(settings["force_norm"])
    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=True)
    replica_paths = []
    for index, frame in enumerate(frames):
        path = work / f"r{index:03d}.xyz"
        _write_xyz(path, frame)
        replica_paths.append(path)
    socket_path = Path("/tmp") / f"ob{os.getpid()}{uuid.uuid4().hex[:8]}.sock"
    if socket_path.exists():
        socket_path.unlink()
    max_step = settings.get("max_stepsize")
    if max_step is None:
        max_step = max_step_internal(float(settings["max_move"]))
    text = render_input(
        band_type=str(settings["band_type"]),
        n_replica=n_images,
        socket_path=str(socket_path),
        replica_paths=replica_paths,
        positions=frames[0],
        cell=cell,
        periodic=periodic,
        fixed=[index + 1 for index, flag in enumerate(free) if not flag],
        max_force=max_force_internal(force_norm, n_free),
        max_step=float(max_step),
        max_steps=cap,
        align=align,
        k_spring=float(settings["k_spring"]),
        profile=settings,
    )
    inp = work / "band.inp"
    log_path = work / "band.out"
    inp.write_text(text, encoding="utf-8")
    counter = HarnessCounter(cache=False, cap=cap)
    client = IpiClient(
        kind=str(settings["potential"]),
        free=free,
        box=box,
        n_images=n_images,
        force_norm=force_norm,
        cap=cap,
        band_type=str(settings["band_type"]),
        counter=counter,
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    began = time.perf_counter()
    proc = subprocess.Popen(
        [str(binary), "-i", str(inp), "-o", str(log_path)],
        cwd=work,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    sock = None
    hold = _OuterHold()
    try:
        sock, hold = open_replica_socket(socket_path, proc, 180.0)
        client.serve(sock)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _kill(proc)
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        hold.close()
        _kill(proc)
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError:
                pass
    if hold.error and client.stop_reason == "":
        client.stop_reason = "crash"
        client.peer_error = hold.error
    log_text = _log_tail(log_path, 4000)
    replica_log = work / "band-r-0.out"
    if replica_log.is_file():
        log_text = log_text + "\n" + _log_tail(replica_log, 4000)
    judged = _judge(client, returncode=proc.returncode, log_text=log_text)
    judged["search_seconds"] = time.perf_counter() - began
    record = {
        "schema": "optbench.start/v2",
        "start": job["name"],
        "seed": "0",
        "counter": "harness",
        "comparable": True,
        "call_unit": "total",
        "problem": settings["problem"],
        "template": settings["template"],
        "code": settings["code"],
        "code_version": settings["code_version"],
        "commit": settings["commit"],
        "algorithm": settings["algorithm"],
        "tuning": settings["tuning"],
        **judged,
    }
    if record["failure"] == "crash":
        record["log"] = log_text[-1500:]
        if client.peer_error:
            record["error"] = client.peer_error
    return record


def _crash_record(name: str, calls: int, verify: int, error: str) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "seed": "0",
        "force_calls": calls,
        "verify_calls": verify,
        "counter": "harness",
        "comparable": True,
        "call_unit": "total",
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "error": error,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "converged": False,
    }


def _threads_one() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _registry(root: Path) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == "cp2k":
            return row
    raise ValueError("codes/registry.toml has no cp2k entry")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the cp2k registry")
    band = template["band"]
    problem_band = problem["band"]
    convergence = problem["convergence"]
    def optional_float(key: str):
        if key not in band:
            return None
        return float(band[key])

    def optional_int(key: str):
        if key not in band:
            return None
        return int(band[key])

    return {
        "potential": problem_band["potential"],
        "n_interior": int(problem_band["interior_images"]),
        "max_move": float(problem_band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": problem["geometry"]["kind"],
        "band_type": str(band["band_type"]),
        "k_spring": float(band.get("k_spring", _K_SPRING)),
        "opt_type": str(band.get("opt_type", "DIIS")),
        "no_ls": bool(band.get("no_ls", False)),
        "stepsize": optional_float("stepsize"),
        "max_stepsize": optional_float("max_stepsize"),
        "n_diis": optional_int("n_diis"),
        "timestep": optional_float("timestep"),
        "temperature": optional_float("temperature"),
        "annealing": optional_float("annealing"),
        "problem": problem["id"],
        "template": template["id"],
        "code": template["code"],
        "code_version": template["code_version"],
        "commit": template["commit"],
        "algorithm": template["algorithm"],
        "tuning": template["tuning"],
    }


def _job(pair, settings: dict, binary: Path, work: Path) -> dict:
    box = None if pair.box is None else np.asarray(pair.box, dtype=np.float64).tolist()
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64).tolist(),
        "product": np.asarray(pair.product, dtype=np.float64).tolist(),
        "free": np.asarray(pair.free, dtype=bool).tolist(),
        "box": box,
        "settings": settings,
        "binary": str(binary),
        "work": str(work / pair.name),
    }


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, partial: bool) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness i-PI POSDATA",
        "# force_calls count socket potential evaluations and are not divided by the image count",
        "# the mean is round-half-even over successes; cap, force, or Hessian index other than 1 is a failure",
    ]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        images = int(settings["n_interior"])
        if images > 0:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row.get('hessian_index') == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {settings['problem']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append("call_unit total")
    lines.append(f"images {settings['n_interior']}")
    lines.append(f"cap {settings['cap']}")
    if partial:
        lines.append("partial true")
    return "\n".join(lines) + "\n"


def _legacy(records: list[dict]) -> dict:
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary = {
        "successes": len(successes),
        "nfailed": len(records) - len(successes),
        "n": len(records),
        "transition_states": sum(1 for row in records if row.get("hessian_index") == 1),
    }
    if successes:
        summary["force_calls"] = int(round(statistics.fmean(successes)))
        summary["force_calls_min"] = min(successes)
        summary["force_calls_max"] = max(successes)
        summary["force_calls_median"] = statistics.median(successes)
    return summary


def write_outputs(
    output: Path,
    records: list[dict],
    template: dict,
    settings: dict,
    problem: dict,
    *,
    partial: bool,
) -> dict:
    """Write ``benchmark.dat``, the per-start file, and the v2 score."""
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(records)
    output.write_text(_dat(ordered, template, settings, partial), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), ordered)
    parameters = {
        "band_type": settings["band_type"],
        "k_spring": settings["k_spring"],
        "opt_type": settings["opt_type"],
        "no_ls": bool(settings["no_ls"]),
        "max_steps": settings["cap"],
        "max_step_angstrom": settings["max_move"],
        "max_force": "problem L2 changed to hartree/bohr and divided by sqrt(n_free*3)",
        "max_dr": _LOOSE,
        "rms_dr": _LOOSE,
        "rms_force": _LOOSE,
        "potential": settings["potential"],
        "counter": "harness",
        "seed": "0",
    }
    if settings.get("stepsize") is not None:
        parameters["stepsize"] = settings["stepsize"]
    if settings.get("max_stepsize") is not None:
        parameters["max_stepsize_bohr"] = settings["max_stepsize"]
    if settings.get("n_diis") is not None:
        parameters["n_diis"] = settings["n_diis"]
    if settings.get("timestep") is not None:
        parameters["timestep_fs"] = settings["timestep"]
    if settings.get("temperature") is not None:
        parameters["temperature_k"] = settings["temperature"]
    if settings.get("annealing") is not None:
        parameters["annealing"] = settings["annealing"]
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score_v2(
            [{"force_calls": row["force_calls"], "success": row["success"], "failure": row.get("failure")} for row in ordered],
            settings["cap"],
        ),
        "legacy": _legacy(ordered),
    }
    if partial:
        document["partial"] = True
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy = document["legacy"]
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": legacy.get("force_calls"),
        "nfailed": legacy["nfailed"],
        "starts": legacy["n"],
        "successes": legacy["successes"],
        "output": str(output),
        "partial": partial,
    }


def _load_done(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def search_named(
    root: Path,
    problem_name: str,
    template_name: str,
    start: str,
    seed: str = "0",
) -> dict:
    """One archived pair. ``seed`` is recorded and does not move the band."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    binary = binary_path()
    if binary is None:
        raise RuntimeError("CP2K_BIN is unset and no cp2k.ssmp binary was found")
    pairs = load_pairs(geometry_archive(root, problem), settings["geometry"], settings["pairs"])
    pair = next((item for item in pairs if item.name == start), None)
    if pair is None:
        raise KeyError(f"{problem['id']} has no start {start}")
    work = Path(os.environ.get("OPTBENCH_CP2K_WORK", "/tmp/optbench-cp2k")) / problem["id"].replace("/", "-")
    record = run_search(_job(pair, settings, binary, work / template["id"].replace("/", "-")))
    record["seed"] = str(seed)
    return record


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Run the pairs in one problem and write the dat, starts, and v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != "cp2k-band":
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    binary = binary_path()
    if binary is None:
        raise RuntimeError("CP2K_BIN is unset and no cp2k.ssmp binary was found")
    pairs = load_pairs(geometry_archive(root, problem), settings["geometry"], settings["pairs"])
    if limit > 0:
        pairs = pairs[:limit]
    work = Path(os.environ.get("OPTBENCH_CP2K_WORK", "/tmp/optbench-cp2k"))
    work = work / problem["id"].replace("/", "-") / template["id"].replace("/", "-")
    jobs = [_job(pair, settings, binary, work) for pair in pairs]
    done: list[dict] = []
    if resume:
        done = _load_done(output.with_suffix(".starts.jsonl"))
    have = {row["start"] for row in done}
    pending = [job for job in jobs if job["name"] not in have]
    order = {job["name"]: index for index, job in enumerate(jobs)}
    _threads_one()
    partial_n = len(jobs) < int(problem["sample"]["pairs"])

    def publish(rows: list[dict]) -> dict:
        rows = [row for row in rows if row["start"] in order]
        rows.sort(key=lambda row: order[row["start"]])
        partial = partial_n or len(rows) < int(problem["sample"]["pairs"])
        return write_outputs(output, rows, template, settings, problem, partial=partial)

    if not pending and done:
        return publish(done)
    if workers <= 1:
        for job in pending:
            done.append(run_search(job))
            publish(done)
    else:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_search, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    done.append(future.result())
                except Exception as exc:
                    done.append(_crash_record(job["name"], 0, 0, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"))
                publish(done)
    return publish(done)
