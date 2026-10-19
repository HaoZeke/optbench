"""CP2K GEO_OPT dimer searches on the harness potential.

CP2K 2026.2 is the i-PI server (``FORCE_EVAL METHOD IPI``). GEO_OPT is
either ``MINIMIZATION`` or ``TRANSITION_STATE`` with ``METHOD DIMER``.
The translation optimizer is BFGS or CG, and ``ROT_OPT`` uses the same
method. FIST is not the scored potential: its Lennard-Jones pairs are
splined and its Morse form is not ``morse/smooth/linear``.

One socket request is one ``HarnessCounter`` entry, cached on the free
coordinates. Image evaluations of the dimer sit ``DR`` away from the
center in the 3N norm and are counted. The judged geometry is the last
center. A tuned row closes the socket when that center's Euclidean
force norm meets the problem stop, because CP2K converges the projected
dimer gradient rather than that norm. A default row leaves the library
thresholds in place and is judged afterwards.

Positions on the wire are Bohr. The harness potential is in angstroms.
``physcon`` takes one bohr as ``0.52917720859`` angstroms. The force
sent back is ``-dE/d(angstrom)`` times that factor times
``FORCE_SCALE``, and the energy is scaled by the same factor, so the
derivative stays consistent. The TS CG line search steps by the norm
of that gradient in bohr. A harness force of order 1 is a few bohr
and the two-point search rejects it, so the scale brings the step
back to a fraction of a bohr. Library force thresholds are multiplied
by the same factor. The harness stop is always the unscaled norm.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .counter import ForceCallCap, HarnessCounter
from .failures import FAILURES, classify_failure, failure_tally
from .rgsaddle_minmode import (
    _SADDLE_CLIMB_OVERSHOOT,
    _SADDLE_ENDPOINT_BUDGET,
    _cartesian_rmsd,
    _geometries,
    _hessian_spectrum,
    _outcome,
    _rng_restore,
    _rng_state,
    _unit,
    _vibrational_eigenvalues,
)
from .score import half_even, penalised_cost, score_v2
from .stationary import judge_stationary, negative_index
from .surfaces import load_con, potential_from_problem
from .v2 import collect_provenance, write_starts

# src/common/physcon.F: a_bohr = 0.52917720859e-10 m.
BOHR_TO_ANG = 0.52917720859
# See the module note. 0.05 turns an initial LJ38 gradient norm of a
# few bohr into a step of about 0.2 bohr.
FORCE_SCALE = 0.05
HEADER = 12
DIMER_DR_ANG = 0.01
CLIMB_CALLS = 1000
INITIAL_MIN_CALLS = 5000
SEARCH_CAP = CLIMB_CALLS + _SADDLE_CLIMB_OVERSHOOT + 2 * _SADDLE_ENDPOINT_BUDGET
_COMMIT = "702cc3d81422b92864ecdd7f544c21c56c04adca"


class StopAtForce(Exception):
    """The center is at the harness force stop. The process can end."""


class Stalled(Exception):
    """The socket is replaying cached geometries and the center is not moving."""


def cp2k_binary() -> Path:
    """The unmodified 2026.2 binary. ``CP2K_BIN`` wins over the build path."""
    env = os.environ.get("CP2K_BIN")
    if env:
        path = Path(env)
        if path.is_file():
            return path
        raise RuntimeError(f"CP2K_BIN is not a file: {env}")
    for name in ("cp2k.ssmp", "cp2k.sopt", "cp2k"):
        candidate = Path("/tmp/cp2k-install/bin") / name
        if candidate.is_file():
            return candidate
    raise RuntimeError("set CP2K_BIN to the CP2K 2026.2 binary")


def pad_header(text: str) -> bytes:
    """12-byte i-PI header, left aligned and space padded."""
    raw = text.encode("ascii")
    if len(raw) > HEADER:
        raise ValueError(text)
    return raw.ljust(HEADER)


def is_dimer_image(distance: float, dr_ang: float, tol: float = 1e-6) -> bool:
    """True when ``distance`` is the dimer half-length, not a new center."""
    if dr_ang <= 0.0:
        return False
    return abs(float(distance) - float(dr_ang)) <= tol


class CountedGeometry:
    """Harness potential behind one CP2K process.

    ``energy_forces`` takes the free coordinates and is what the Hessian
    uses. ``evaluate_bohr`` takes the full Cartesian geometry from the
    socket. Both share one cache, so the judgment of the last center does
    not add a call.
    """

    def __init__(
        self,
        evaluate,
        positions: np.ndarray,
        fixed: np.ndarray,
        cell: np.ndarray,
        *,
        cap: int | None,
        force_tol: float,
        force_match: str,
        dr_ang: float,
    ):
        self.evaluate = evaluate
        self.base = np.array(positions, dtype=np.float64, copy=True)
        self.fixed = np.asarray(fixed, dtype=bool)
        self.free = np.flatnonzero(~self.fixed)
        if len(self.free) == 0:
            raise ValueError("every atom is fixed")
        self.cell = np.asarray(cell, dtype=np.float64)
        self.counter = HarnessCounter(cache=True, cap=cap)
        self.force_tol = float(force_tol)
        self.force_match = force_match
        self.dr_ang = float(dr_ang)
        self.center: np.ndarray | None = None
        self.stop_now = False
        self.stopped_on_l2 = False
        self.stalled = False
        self.capped = False
        self._cache_streak = 0

    @property
    def calls(self) -> int:
        return self.counter.calls

    @property
    def verify_calls(self) -> int:
        return self.counter.verify_calls

    def verification(self):
        return self.counter.verification()

    def free_of(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions, dtype=np.float64)[self.free].reshape(-1).copy()

    def energy_forces(self, flat) -> tuple[float, np.ndarray]:
        pos = self.base.copy()
        pos[self.free] = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        energy, grad = self._charged(pos)
        free = np.ascontiguousarray(grad[self.free].reshape(-1), dtype=np.float64)
        return float(energy), free

    def _charged(self, pos_ang: np.ndarray) -> tuple[float, np.ndarray]:
        key = tuple(float(value) for value in pos_ang[self.free].reshape(-1))
        fresh, cached = self.counter.charge(key)
        if not fresh:
            energy, grad = cached
            return float(energy), grad
        energy, grad = self.evaluate(pos_ang, self.cell)
        grad = np.array(grad, dtype=np.float64, copy=True)
        return self.counter.remember(key, (float(energy), grad))

    def evaluate_bohr(self, xyz: np.ndarray) -> tuple[float, np.ndarray]:
        natom = int(self.base.shape[0])
        pos = np.asarray(xyz, dtype=np.float64).reshape(natom, 3) * BOHR_TO_ANG
        if self.center is None:
            image = False
        else:
            image = is_dimer_image(float(np.linalg.norm(pos - self.center)), self.dr_ang)
        before = self.counter.calls
        try:
            energy, grad = self._charged(pos)
        except ForceCallCap:
            self.capped = True
            raise
        if self.counter.calls == before:
            self._cache_streak += 1
        else:
            self._cache_streak = 0
        if self._cache_streak >= 80:
            self.stalled = True
            raise Stalled()
        if not image:
            self.center = pos.copy()
            if self.force_match == "l2" and self._under_cap() and self._at_stop(grad):
                self.stop_now = True
                self.stopped_on_l2 = True
        forces = np.ascontiguousarray((-grad * BOHR_TO_ANG * FORCE_SCALE).reshape(-1), dtype=np.float64)
        # Fixed atoms are not degrees of freedom. Their forces would enter
        # the dimer's rigid-mode projection and change the free-atom step.
        forces.reshape(-1, 3)[self.fixed] = 0.0
        return float(energy) * FORCE_SCALE, forces

    def _under_cap(self) -> bool:
        cap = self.counter.cap
        return cap is None or self.counter.calls < cap

    def _at_stop(self, grad: np.ndarray) -> bool:
        free = grad[self.free].reshape(-1)
        norm = float(np.linalg.norm(free))
        return math.isfinite(norm) and norm <= self.force_tol + 1e-10


def _readexact(sock: socket.socket, count: int) -> bytes:
    buf = bytearray()
    while len(buf) < count:
        chunk = sock.recv(count - len(buf))
        if not chunk:
            raise EOFError("i-PI socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _read_header(sock: socket.socket) -> str:
    return _readexact(sock, HEADER).decode("ascii").strip()


def _write_header(sock: socket.socket, text: str) -> None:
    sock.sendall(pad_header(text))


def _read_int(sock: socket.socket) -> int:
    return struct.unpack("<i", _readexact(sock, 4))[0]


def _write_int(sock: socket.socket, value: int) -> None:
    sock.sendall(struct.pack("<i", int(value)))


def _read_doubles(sock: socket.socket, count: int) -> np.ndarray:
    raw = _readexact(sock, 8 * count)
    return np.frombuffer(raw, dtype="<f8").copy()


def _write_doubles(sock: socket.socket, values) -> None:
    sock.sendall(np.ascontiguousarray(values, dtype="<f8").tobytes())


def serve_client(sock: socket.socket, geometry: CountedGeometry) -> str:
    """Answer one CP2K i-PI server until EXIT, the cap, or the l2 stop."""
    pending: tuple[float, np.ndarray, int] | None = None
    while True:
        header = _read_header(sock)
        if header == "STATUS":
            if geometry.stop_now:
                raise StopAtForce()
            _write_header(sock, "READY" if pending is None else "HAVEDATA")
        elif header == "POSDATA":
            _read_doubles(sock, 9)
            _read_doubles(sock, 9)
            natom = _read_int(sock)
            xyz = _read_doubles(sock, 3 * natom)
            if natom != int(geometry.base.shape[0]):
                raise RuntimeError(f"i-PI sent {natom} atoms, the geometry has {len(geometry.base)}")
            energy, forces = geometry.evaluate_bohr(xyz)
            pending = (energy, forces, natom)
        elif header == "GETFORCE":
            if pending is None:
                raise RuntimeError("GETFORCE before POSDATA")
            energy, forces, natom = pending
            _write_header(sock, "FORCEREADY")
            _write_doubles(sock, [energy])
            _write_int(sock, natom)
            _write_doubles(sock, forces)
            _write_doubles(sock, np.zeros(9, dtype=np.float64))
            _write_int(sock, 0)
            pending = None
        elif header == "EXIT":
            return "exit"
        else:
            raise RuntimeError(f"unexpected i-PI header {header!r}")


def render_input(
    positions: np.ndarray,
    fixed: np.ndarray,
    cell: np.ndarray,
    *,
    element: str,
    host: str,
    task: str,
    optimizer: str,
    rot_optimizer: str,
    force_match: str,
    model_hessian: bool,
    cg_line_search: str,
    max_iter: int,
    rot_max_iter: int,
    mode: np.ndarray | None,
) -> str:
    """One GEO_OPT input. The potential is the i-PI client, not FIST."""
    pos = np.asarray(positions, dtype=np.float64)
    lengths = np.asarray(cell, dtype=np.float64).reshape(3).copy()
    if float(np.min(lengths)) <= 0.0:
        lengths[:] = 100.0
    lines = [
        "&GLOBAL",
        "  PROJECT optbench",
        "  RUN_TYPE GEO_OPT",
        "  PRINT_LEVEL LOW",
        "  SEED 2000",
        "&END GLOBAL",
        "&FORCE_EVAL",
        "  METHOD IPI",
        "  &SUBSYS",
        "    &CELL",
        f"      ABC {lengths[0]:.16e} {lengths[1]:.16e} {lengths[2]:.16e}",
        "      PERIODIC NONE",
        "    &END CELL",
        "    &COORD",
        "      SCALED F",
        "      UNIT angstrom",
    ]
    for row in pos:
        lines.append(f"      {element} {row[0]:.16e} {row[1]:.16e} {row[2]:.16e}")
    lines.extend(
        [
            "    &END COORD",
            "  &END SUBSYS",
            "&END FORCE_EVAL",
            "&MOTION",
            "  &DRIVER",
            "    UNIX",
            f"    HOST {host}",
            "    PORT 12345",
            "    PREFIX ipi",
            "  &END DRIVER",
        ]
    )
    fixed_ids = [str(index + 1) for index in np.flatnonzero(np.asarray(fixed, dtype=bool))]
    if fixed_ids:
        lines.append("  &CONSTRAINT")
        lines.append("    &FIXED_ATOMS")
        lines.append("      COMPONENTS_TO_FIX XYZ")
        for start in range(0, len(fixed_ids), 16):
            lines.append("      LIST " + " ".join(fixed_ids[start : start + 16]))
        lines.append("    &END FIXED_ATOMS")
        lines.append("  &END CONSTRAINT")
    kind = "TRANSITION_STATE" if task == "dimer" else "MINIMIZATION"
    lines.extend(
        [
            "  &GEO_OPT",
            f"    TYPE {kind}",
            f"    OPTIMIZER {optimizer}",
            f"    MAX_ITER {int(max_iter)}",
        ]
    )
    if force_match == "l2":
        lines.extend(
            [
                "    MAX_FORCE 1.0E-8",
                "    RMS_FORCE 1.0E-8",
                "    MAX_DR 10.0",
                "    RMS_DR 10.0",
            ]
        )
    else:
        # Same physical threshold as the library 4.5e-4 and 3.0e-4 Ha/bohr
        # once the socket force has been multiplied by FORCE_SCALE.
        lines.extend(
            [
                f"    MAX_FORCE {0.00045 * FORCE_SCALE:.8e}",
                f"    RMS_FORCE {0.00030 * FORCE_SCALE:.8e}",
            ]
        )
    if optimizer == "BFGS":
        flag = "TRUE" if model_hessian else "FALSE"
        lines.extend(["    &BFGS", f"      USE_MODEL_HESSIAN {flag}", "    &END BFGS"])
    else:
        lines.extend(
            [
                "    &CG",
                "      MAX_STEEP_STEPS 0",
                "      &LINE_SEARCH",
                f"        TYPE {cg_line_search}",
            ]
        )
        if cg_line_search == "2PNT":
            lines.extend(
                [
                    "        &2PNT",
                    "          MAX_ALLOWED_STEP 0.05",
                    "        &END 2PNT",
                ]
            )
        lines.extend(["      &END LINE_SEARCH", "    &END CG"])
    if task == "dimer":
        if mode is None:
            raise ValueError("a dimer input needs a mode")
        lines.extend(
            [
                "    &TRANSITION_STATE",
                "      METHOD DIMER",
                "      &DIMER",
                "        DR 0.01",
                "        ANGLE_TOLERANCE [deg] 1.0",
                "        &DIMER_VECTOR",
            ]
        )
        vector = np.asarray(mode, dtype=np.float64).reshape(-1)
        if vector.size != pos.size:
            raise ValueError("dimer vector length is not 3N")
        for atom in vector.reshape(-1, 3):
            lines.append(f"          {atom[0]:.16e} {atom[1]:.16e} {atom[2]:.16e}")
        lines.extend(
            [
                "        &END DIMER_VECTOR",
                "        &ROT_OPT",
                f"          OPTIMIZER {rot_optimizer}",
                f"          MAX_ITER {int(rot_max_iter)}",
            ]
        )
        if rot_optimizer == "CG":
            lines.extend(
                [
                    "          &CG",
                    "            MAX_STEEP_STEPS 0",
                    "            &LINE_SEARCH",
                    f"              TYPE {cg_line_search}",
                ]
            )
            if cg_line_search == "2PNT":
                lines.extend(
                    [
                        "              &2PNT",
                        "                MAX_ALLOWED_STEP 0.05",
                        "              &END 2PNT",
                    ]
                )
            lines.extend(["            &END LINE_SEARCH", "          &END CG"])
        lines.extend(
            [
                "        &END ROT_OPT",
                "      &END DIMER",
                "    &END TRANSITION_STATE",
            ]
        )
    lines.extend(
        [
            "  &END GEO_OPT",
            "  &PRINT",
            "    &RESTART OFF",
            "    &END RESTART",
            "    &RESTART_HISTORY OFF",
            "    &END RESTART_HISTORY",
            "  &END PRINT",
            "&END MOTION",
            "",
        ]
    )
    return "\n".join(lines)


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return env


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=5)


def _library_converged(text: str) -> bool:
    return "GEOMETRY OPTIMIZATION COMPLETED" in text and "EXITING GEOMETRY OPTIMIZATION" not in text


def _connect(proc: subprocess.Popen, path: str) -> socket.socket:
    deadline = time.monotonic() + 180.0
    last = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"cp2k exited {proc.returncode} before the i-PI socket")
        if os.path.exists(path):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(path)
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                last = exc
                sock.close()
                time.sleep(0.05)
                continue
            sock.settimeout(300.0)
            return sock
        time.sleep(0.05)
    raise RuntimeError(f"i-PI socket {path} did not accept a client ({last})")


def run_process(geometry: CountedGeometry, inp: Path, binary: Path) -> tuple[str, str, int]:
    """Run one GEO_OPT. Returns status, the output tail, and the return code."""
    work = inp.parent
    host = f"c{os.getpid()}{work.name}"
    # The host is also written into the input. Rebuild is the caller's job;
    # the placeholder token CP2KHOST is replaced here so the path is unique.
    text = inp.read_text(encoding="utf-8").replace("CP2KHOST", host)
    inp.write_text(text, encoding="utf-8")
    out_path = work / "cp2k.out"
    socket_path = f"/tmp/ipi_{host}"
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    proc = subprocess.Popen(
        [str(binary), "-i", inp.name, "-o", out_path.name],
        cwd=work,
        env=_child_env(),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    status = "error"
    try:
        sock = _connect(proc, socket_path)
        try:
            status = serve_client(sock, geometry)
        except StopAtForce:
            status = "l2"
        except Stalled:
            status = "stall"
        except ForceCallCap:
            geometry.capped = True
            status = "cap"
        except (EOFError, TimeoutError, OSError):
            if geometry.capped:
                status = "cap"
            elif geometry.stopped_on_l2:
                status = "l2"
            elif geometry.stalled:
                status = "stall"
            else:
                status = "error"
        finally:
            sock.close()
    except Exception:
        _kill(proc)
        raise
    if status != "exit":
        _kill(proc)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _kill(proc)
    full = ""
    if out_path.is_file():
        full = out_path.read_text(encoding="utf-8", errors="replace")
    # A rotation prints the same completion line. The outer failure is
    # the EXITING GEOMETRY line, which can sit far above a long tail.
    if _library_converged(full) and status == "error":
        status = "exit"
    return status, full, int(proc.returncode or 0)


def _full_mode(positions: np.ndarray, fixed: np.ndarray, free_mode: np.ndarray) -> np.ndarray:
    mode = np.zeros(positions.shape, dtype=np.float64)
    mode[~np.asarray(fixed, dtype=bool)] = np.asarray(free_mode, dtype=np.float64).reshape(-1, 3)
    flat = mode.reshape(-1)
    scale = float(np.linalg.norm(flat))
    if not math.isfinite(scale) or scale < 1e-14:
        raise ValueError("dimer vector has no direction")
    return flat / scale


def _prepare_mode(geometry: CountedGeometry, spec: dict) -> np.ndarray:
    if spec["mode_policy"] == "gradient":
        _energy, grad = geometry.energy_forces(geometry.free_of(geometry.base))
        return _full_mode(geometry.base, geometry.fixed, _unit(grad))
    given = spec.get("mode")
    if given is None:
        raise ValueError("dimer mode is missing")
    vector = np.asarray(given, dtype=np.float64)
    if vector.size == geometry.base.size:
        return _full_mode(geometry.base, geometry.fixed, vector.reshape(-1, 3)[~geometry.fixed])
    return _full_mode(geometry.base, geometry.fixed, vector)


def _judge(spec: dict, geometry: CountedGeometry, *, library_converged: bool, status: str, detail: str) -> dict:
    cap = int(spec["cap"])
    name = str(spec["name"])
    base = {
        "schema": "optbench.start/v2",
        "start": name,
        "seed": str(spec.get("seed", "0")),
        "counter": "harness",
        "comparable": True,
        "code": spec["code"],
        "code_version": spec["code_version"],
        "commit": spec["commit"],
        "tuning": spec["tuning"],
        "template": spec["template"],
        "problem": spec["problem"],
        "verify_calls": int(geometry.verify_calls),
        "library_converged": bool(library_converged),
        "cp2k_status": status,
    }
    point = None if geometry.center is None else geometry.free_of(geometry.center)
    if point is None or status == "error" and geometry.calls == 0:
        picked = [line.strip() for line in detail.splitlines() if "not yet" in line or "[ABORT]" in line]
        base.update(
            {
                "force_calls": int(geometry.calls),
                "success": False,
                "failure": "crash",
                "reasons": ["crash"],
                "error": (picked[-1] if picked else detail[-500:])[:500],
                "energy": None,
                "force_norm": None,
                "hessian_index": None,
                "converged": False,
            }
        )
        if "not yet working" in detail or "not yet implemented" in detail:
            base["optimizer_refused"] = True
        return base
    # The center was already charged. A later image may have replaced the
    # one-slot cache, so this read is a verification and not a search call.
    with geometry.verification():
        energy, gradient = geometry.energy_forces(point)
    force_norm = float(np.linalg.norm(gradient))
    reasons: list[str] = []
    index = None
    lowest = None
    removed = 0
    cartesian = None
    if geometry.calls >= cap or status == "cap":
        reasons.append("cap")
    finite = math.isfinite(force_norm) and math.isfinite(float(energy))
    if not finite or force_norm > float(spec["force_tol"]) + 1e-10:
        reasons.append("force_norm" if finite else "divergence")
    elif geometry.calls < cap and bool(spec.get("hessian", True)):
        raw_eigenvalues, vectors = _hessian_spectrum(geometry, point)
        cartesian = negative_index(raw_eigenvalues, 1e-6)
        if spec["project_rigid"]:
            eigenvalues, removed = _vibrational_eigenvalues(point, raw_eigenvalues, vectors)
        else:
            eigenvalues, removed = list(raw_eigenvalues), 0
        index = negative_index(eigenvalues, 1e-6)
        lowest = [float(value) for value in eigenvalues[:8]]
        base["mode_axis"] = [float(value) for value in np.asarray(vectors[:, 0], dtype=np.float64).reshape(-1)]
        coords = np.asarray(point, dtype=np.float64).reshape(-1, 3).tolist()
        judged = judge_stationary(
            kind="saddle" if spec["task"] == "dimer" else "minimum",
            eigenvalues=eigenvalues,
            energy=float(energy),
            coords=coords,
            reference_energy=float(energy),
            reference_coords=coords,
            energy_tol=0.0,
            rmsd_tol=0.0,
        )
        reasons.extend(reason for reason in judged["reasons"] if reason not in {"rmsd", "energy"})
        base["expected_index"] = judged["expected_index"]
        base["rmsd"] = float(judged["rmsd"])
    if status == "stall" and "cap" not in reasons:
        reasons.append("divergence")
    if status == "error" and "cap" not in reasons and geometry.calls < cap and not geometry.stopped_on_l2:
        reasons.append("crash")
    reasons = list(dict.fromkeys(reasons))
    success, failure = _outcome(reasons)
    base.update(
        {
            "force_calls": int(geometry.calls),
            "verify_calls": int(geometry.verify_calls),
            "success": success,
            "failure": failure,
            "reasons": reasons,
            "energy": float(energy),
            "force_norm": force_norm,
            "force": [float(value) for value in np.asarray(gradient, dtype=np.float64).reshape(-1)],
            "hessian_index": None if index is None else int(index),
            "cartesian_index": None if cartesian is None else int(cartesian),
            "rigid_modes_removed": int(removed),
            "lowest_eigenvalue": None if not lowest else lowest[0],
            "lowest_eigenvalues": lowest or [],
            "converged": bool(library_converged or geometry.stopped_on_l2) and force_norm <= float(spec["force_tol"]) + 1e-10,
            "reference": "converged-point",
            "x": [float(value) for value in point],
        }
    )
    if "not yet working" in detail or "not yet implemented" in detail:
        base["optimizer_refused"] = True
    if detail and failure == "crash":
        picked = [line.strip() for line in detail.splitlines() if "not yet" in line or "[ABORT]" in line]
        base["error"] = (picked[-1] if picked else detail[-500:])[:500]
    return base


def run_start(spec: dict) -> dict:
    """One GEO_OPT, in this process. Safe to send to a process pool."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    raw = potential_from_problem({"potential": spec["potential"]})
    dr = DIMER_DR_ANG if spec["task"] == "dimer" else 0.0
    geometry = CountedGeometry(
        raw,
        positions,
        fixed,
        cell,
        cap=int(spec["cap"]),
        force_tol=float(spec["force_tol"]),
        force_match=str(spec["force_match"]),
        dr_ang=dr,
    )
    work = Path(spec["work"]) / str(spec["name"]).replace("/", "-")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        mode = _prepare_mode(geometry, spec) if spec["task"] == "dimer" else None
        text = render_input(
            positions,
            fixed,
            cell,
            element=str(spec["element"]),
            host="CP2KHOST",
            task=str(spec["task"]),
            optimizer=str(spec["optimizer"]),
            rot_optimizer=str(spec["rot_optimizer"]),
            force_match=str(spec["force_match"]),
            model_hessian=bool(spec["model_hessian"]),
            cg_line_search=str(spec["cg_line_search"]),
            max_iter=int(spec["max_iter"]),
            rot_max_iter=int(spec["rot_max_iter"]),
            mode=mode,
        )
        inp = work / "input.inp"
        inp.write_text(text, encoding="utf-8")
        try:
            status, output_text, code = run_process(geometry, inp, Path(spec["binary"]))
            detail = output_text if status == "error" or code not in {0, -15} else ""
        except Exception as exc:
            record = _judge(spec, geometry, library_converged=False, status="error", detail=f"{type(exc).__name__}: {exc}")
            return record
        library = _library_converged(output_text)
        return _judge(spec, geometry, library_converged=library, status=status, detail=detail)
    finally:
        if os.environ.get("CP2K_KEEP_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


def _section(template: dict, name: str) -> dict:
    value = template.get(name) or template.get(name.lower()) or {}
    return value if isinstance(value, dict) else {}


def _element(problem: dict) -> str:
    kind = str((problem.get("potential") or {}).get("kind", "lennard-jones"))
    if kind == "morse":
        return "Pt"
    return "H"


def _project_rigid(problem: dict) -> bool:
    return problem["id"] == "tsopt/LJ38"


def _spec_common(problem: dict, template: dict, *, cap: int, force_tol: float, binary: str, work: str) -> dict:
    block = _section(template, "cp2k")
    tuned = str(template.get("tuning", "")) == "tuned" or str(block.get("force_match", "")) == "l2"
    return {
        "potential": problem.get("potential") or {},
        "cap": int(cap),
        "force_tol": float(force_tol),
        "task": str(block.get("task", "dimer")),
        "optimizer": str(block.get("optimizer", "BFGS")),
        "rot_optimizer": str(block.get("rot_optimizer", block.get("optimizer", "BFGS"))),
        "force_match": "l2" if tuned else "library",
        "model_hessian": bool(block.get("model_hessian", not tuned)),
        "cg_line_search": str(block.get("cg_line_search", "2PNT" if tuned else "GOLD")),
        "rot_max_iter": int(block.get("rot_max_iter", 20 if tuned else 200)),
        "max_iter": int(block.get("max_iter", 100000 if tuned else 200)),
        "element": _element(problem),
        "project_rigid": _project_rigid(problem),
        "binary": binary,
        "work": work,
        "code": str(template.get("code", "CP2K")),
        "code_version": str(template.get("code_version", "2026.2")),
        "commit": str(template.get("commit", _COMMIT)),
        "tuning": str(template.get("tuning", "")),
        "template": template["id"],
        "problem": problem["id"],
        "seed": "0",
        "mode_policy": "gradient",
        "mode": None,
    }


def _public(record: dict) -> dict:
    """Drop the endpoint mode. The judged geometry and raw force stay."""
    return {key: value for key, value in record.items() if key != "mode_axis"}


def _write_dat(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _summarize_starts(records: list[dict], template: dict, cap: int, problem_id: str) -> tuple[str, dict]:
    metrics = score_v2(records, cap)
    ok = [int(row["force_calls"]) for row in records if row["success"]]
    lines = []
    if ok:
        lines.append(f"force_calls {half_even(sum(ok) / len(ok))}")
        lines.append(f"force_calls_min {min(ok)}")
        lines.append(f"force_calls_median {statistics.median(ok)}")
        lines.append(f"force_calls_max {max(ok)}")
    lines.append(f"nfailed {metrics['nfailed']}")
    lines.append(f"starts {len(records)}")
    lines.append(f"par2_mean {metrics['par2_mean']:.10f}")
    lines.append(f"successes {metrics['successes']}")
    lines.append(f"wilson_low {metrics['wilson_low']:.10f}")
    lines.append(f"wilson_high {metrics['wilson_high']:.10f}")
    lines.append(f"par2_median {metrics['median']:.10f}")
    lines.append(f"n_cap {metrics['n_cap']}")
    lines.append(f"n_wrong_stationary {metrics['n_wrong_stationary']}")
    lines.append(f"n_divergence {metrics['n_divergence']}")
    lines.append(f"n_crash {metrics['n_crash']}")
    lines.append("counter harness")
    lines.append(f"code {template.get('code', 'CP2K')}")
    lines.append(f"code_version {template.get('code_version', '2026.2')}")
    lines.append(f"commit {template.get('commit', _COMMIT)}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem_id}")
    lines.append(f"harness_calls {sum(int(row['force_calls']) for row in records)}")
    return "\n".join(lines) + "\n", metrics


def _done_path(done_dir: Path, name: str) -> Path:
    return done_dir / f"{name}.json"


def _load_done(done_dir: Path) -> dict[str, dict]:
    found = {}
    if not done_dir.is_dir():
        return found
    for path in sorted(done_dir.glob("*.json")):
        found[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _save_done(work: Path, record: dict) -> None:
    path = _done_path(work, str(record["start"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_public(record), sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_starts(specs: list[dict], *, workers: int, done: dict[str, dict], done_dir: Path) -> list[dict]:
    pending = [spec for spec in specs if spec["name"] not in done]
    records = dict(done)
    if not pending:
        return [records[spec["name"]] for spec in specs if spec["name"] in records]

    def _take(record: dict) -> None:
        records[str(record["start"])] = record
        _save_done(done_dir, record)
        print(
            f"{record['template']} {record['start']} calls={record['force_calls']} "
            f"success={record['success']} failure={record['failure']}",
            flush=True,
        )
    if workers <= 1:
        for spec in pending:
            _take(run_start(spec))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_start, spec): spec["name"] for spec in pending}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    spec = next(item for item in pending if item["name"] == name)
                    record = {
                        "schema": "optbench.start/v2",
                        "start": name,
                        "seed": "0",
                        "force_calls": 0,
                        "verify_calls": 0,
                        "counter": "harness",
                        "comparable": True,
                        "success": False,
                        "failure": "crash",
                        "reasons": ["crash"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "code": spec["code"],
                        "code_version": spec["code_version"],
                        "commit": spec["commit"],
                        "tuning": spec["tuning"],
                        "template": spec["template"],
                        "problem": spec["problem"],
                        "converged": False,
                    }
                _take(record)
    return [records[spec["name"]] for spec in specs]


def _emit_starts(output: Path, records: list[dict], template: dict, cap: int, problem_id: str) -> dict:
    public = [_public(row) for row in records]
    text, metrics = _summarize_starts(public, template, cap, problem_id)
    _write_dat(output, text)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    return {
        "force_calls": half_even(statistics.fmean([row["force_calls"] for row in records if row["success"]]))
        if any(row["success"] for row in records)
        else None,
        "nfailed": metrics["nfailed"],
        "starts": len(records),
        "template": template["id"],
        "problem": problem_id,
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
        "records": records,
    }


def _ts_specs(root: Path, problem: dict, template: dict, *, cap: int, force_tol: float, work: Path, limit: int) -> list[dict]:
    binary = str(cp2k_binary())
    rows = _geometries(root, problem, work)
    if limit > 0:
        rows = rows[:limit]
    specs = []
    for row in rows:
        positions, fixed, cell = load_con(Path(row["path"]))
        spec = _spec_common(problem, template, cap=cap, force_tol=force_tol, binary=binary, work=str(work / "runs"))
        spec["name"] = row["name"]
        spec["positions"] = positions.tolist()
        spec["fixed"] = fixed.astype(bool).tolist()
        spec["cell"] = cell.tolist()
        spec["task"] = "dimer"
        if row.get("reactant"):
            other, other_fixed, _cell = load_con(Path(row["reactant"]))
            free = np.flatnonzero(~fixed)
            spec["mode_policy"] = "given"
            spec["mode"] = _full_mode(positions, fixed, other[free] - positions[free]).tolist()
        specs.append(spec)
    return specs


def _minimize_spec(base: dict, positions: np.ndarray, *, name: str, cap: int) -> dict:
    spec = dict(base)
    spec["name"] = name
    spec["positions"] = np.asarray(positions, dtype=np.float64).tolist()
    spec["task"] = "minimize"
    spec["mode_policy"] = "given"
    spec["mode"] = None
    spec["cap"] = int(cap)
    spec["force_match"] = "l2"
    spec["max_iter"] = 100000
    spec["project_rigid"] = False
    spec["hessian"] = False
    return spec


def _checkpoint(work: Path) -> tuple[Path, Path]:
    return work / "saddle-header.json", work / "saddle-partial.jsonl"


def _stop_requested() -> bool:
    return Path("/tmp/optbench-cp2k-stop").is_file()


def _draw_saddle_spec(rng, searches: int, rep: int, listed: list[int], free: np.ndarray, positions: np.ndarray, minimum: np.ndarray, magnitude: float, base: dict, fixed: np.ndarray) -> dict:
    """One displacement. The draw is the only use of the search RNG."""
    atom = int(listed[int(rng.integers(0, len(listed)))])
    slots = np.flatnonzero(free == atom)
    if len(slots) != 1:
        raise RuntimeError(f"displace atom {atom} is not one free atom")
    slot = int(slots[0])
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    kicked = positions.copy()
    free_pos = minimum.reshape(-1, 3).copy()
    free_pos[slot] = free_pos[slot] + magnitude * direction
    kicked[free] = free_pos
    kick = np.zeros(positions.shape, dtype=np.float64)
    kick[atom] = direction
    spec = dict(base)
    spec["name"] = f"rep{rep}-search{searches}"
    spec["positions"] = kicked.tolist()
    spec["task"] = "dimer"
    spec["mode_policy"] = "given"
    spec["mode"] = _full_mode(positions, fixed, kick[~fixed]).tolist()
    spec["cap"] = CLIMB_CALLS
    return spec


def _commit_saddle_outcome(spec: dict, outcome: dict, rng_state: dict, *, found: list, all_records: list, connected: int, positions, fixed, cell, minimum, min_energy: float, force_tol: float, connect_rmsd: float, barrier_max: float, base: dict, partial_path: Path, output: Path, template: dict, problem_id: str) -> tuple[int, bool, str]:
    """Score one climb in catalog order. Returns connected, refused, failure."""
    climb = int(outcome.get("force_calls") or 0)
    record = {
        "schema": "optbench.start/v2",
        "start": spec["name"],
        "seed": "0",
        "counter": "harness",
        "comparable": True,
        "success": False,
        "reasons": list(outcome.get("reasons") or ["force_norm"]),
        "converged": bool(outcome.get("converged")),
        "hessian_index": outcome.get("hessian_index"),
        "code": base["code"],
        "code_version": base["code_version"],
        "commit": base["commit"],
        "tuning": base["tuning"],
        "template": template["id"],
        "problem": problem_id,
        "climb_calls": climb,
    }
    if outcome.get("energy") is not None:
        record["energy"] = float(outcome["energy"])
    if outcome.get("force_norm") is not None:
        record["force_norm"] = float(outcome["force_norm"])
    if outcome.get("x") is not None:
        record["x"] = [float(value) for value in outcome["x"]]
    if outcome.get("force") is not None:
        record["force"] = [float(value) for value in outcome["force"]]
    saved_point = None
    endpoint_calls = 0
    usable = (
        outcome.get("x") is not None
        and bool(outcome.get("converged"))
        and outcome.get("hessian_index") == 1
        and float(outcome.get("force_norm") or 1e9) <= force_tol + 1e-10
        and climb < CLIMB_CALLS
    )
    if usable:
        saddle_x = np.asarray(outcome["x"], dtype=np.float64)
        if outcome.get("mode_axis") is not None:
            axis = _unit(np.asarray(outcome["mode_axis"], dtype=np.float64))
        else:
            axis = _axis_from_point(base, saddle_x, positions, fixed, cell)
        free = np.flatnonzero(~np.asarray(fixed, dtype=bool))
        plus = _minimize_spec(base, _place(positions, free, saddle_x + 0.1 * axis), name=f"{spec['name']}-plus", cap=_SADDLE_ENDPOINT_BUDGET)
        minus = _minimize_spec(base, _place(positions, free, saddle_x - 0.1 * axis), name=f"{spec['name']}-minus", cap=_SADDLE_ENDPOINT_BUDGET)
        plus_out = run_start(plus)
        minus_out = run_start(minus)
        endpoint_calls = int(plus_out.get("force_calls") or 0) + int(minus_out.get("force_calls") or 0)
        plus_x = np.asarray(plus_out["x"], dtype=np.float64) if plus_out.get("x") is not None else saddle_x + 0.1 * axis
        minus_x = np.asarray(minus_out["x"], dtype=np.float64) if minus_out.get("x") is not None else saddle_x - 0.1 * axis
        plus_rmsd = _cartesian_rmsd(plus_x.reshape(-1, 3), minimum.reshape(-1, 3))
        minus_rmsd = _cartesian_rmsd(minus_x.reshape(-1, 3), minimum.reshape(-1, 3))
        links = (plus_rmsd <= connect_rmsd) ^ (minus_rmsd <= connect_rmsd)
        novel = all(_cartesian_rmsd(saddle_x.reshape(-1, 3), old.reshape(-1, 3)) > connect_rmsd for old in found)
        barrier = float(outcome["energy"] - min_energy)
        record["barrier"] = barrier
        record["rmsd_plus"] = plus_rmsd
        record["rmsd_minus"] = minus_rmsd
        record["saddle_x"] = saddle_x.reshape(-1).tolist()
        record["endpoint_plus"] = np.asarray(plus_x, dtype=np.float64).reshape(-1).tolist()
        record["endpoint_minus"] = np.asarray(minus_x, dtype=np.float64).reshape(-1).tolist()
        record["minimum_x"] = np.asarray(minimum, dtype=np.float64).reshape(-1).tolist()
        record["reasons"] = []
        if novel:
            found.append(saddle_x)
            saved_point = saddle_x
        if links and novel and barrier < barrier_max:
            connected += 1
            record["success"] = True
        else:
            if not links:
                record["reasons"].append("not_connected")
            if not novel:
                record["reasons"].append("duplicate")
            if barrier >= barrier_max:
                record["reasons"].append("barrier")
    elif outcome.get("failure") == "crash" or outcome.get("x") is None:
        record["reasons"] = ["crash"]
    elif climb >= CLIMB_CALLS:
        record["reasons"] = ["cap", "force_norm"]
    else:
        record["reasons"] = [str(item) for item in (outcome.get("reasons") or ["force_norm"])]
    record["force_calls"] = climb + endpoint_calls
    record["endpoint_calls"] = endpoint_calls
    record["verify_calls"] = int(outcome.get("verify_calls") or 0)
    success, failure = _outcome(record["reasons"])
    record["success"] = success
    record["failure"] = failure
    all_records.append(record)
    stored = dict(record)
    stored["rng_state"] = rng_state
    if saved_point is not None:
        stored["saddle_x"] = saved_point.tolist()
    _append_jsonl(partial_path, stored)
    _emit_saddle(output, all_records, template, problem_id)
    return connected, bool(outcome.get("optimizer_refused")), failure


def _run_saddle(root: Path, problem: dict, template: dict, *, output: Path, force_tol: float, work: Path, workers: int = 1) -> dict:
    """One displacement is one search: the climb plus two minimizations.

    The initial minimization is outside the per-search count. The board
    cap is ``cap`` in the result file, climb 1000 plus 64 of overshoot
    plus two endpoint budgets of 2000.
    """
    binary = str(cp2k_binary())
    Path("/tmp/optbench-cp2k-stop").unlink(missing_ok=True)
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    from .rgsaddle_minmode import _stored

    positions, fixed, cell = load_con(_stored(root, con_path))
    search = _section(template, "search")
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    repetitions = int(search.get("repetitions", 1))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    magnitude = float(search.get("displace_magnitude", 0.1))
    connect_rmsd = float(search.get("rmsd_connect", 0.5))
    max_searches = int(search.get("max_searches", 20000))
    free = np.flatnonzero(~fixed)
    base = _spec_common(
        problem,
        template,
        cap=CLIMB_CALLS,
        force_tol=force_tol,
        binary=binary,
        work=str(work / "runs"),
    )
    base["positions"] = positions.tolist()
    base["fixed"] = fixed.astype(bool).tolist()
    base["cell"] = cell.tolist()
    base["task"] = "dimer"
    base["project_rigid"] = False
    comments = []
    all_records: list[dict] = []
    failed_reps = 0
    finished = True
    for rep in range(1, repetitions + 1):
        header_path, partial_path = _checkpoint(work)
        header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
        saved = []
        if header and header.get("template") == template["id"] and int(header.get("rep", 0)) == rep and partial_path.is_file():
            saved = []
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    saved.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        if saved or (header and header.get("template") == template["id"] and int(header.get("rep", 0)) == rep):
            minimum = np.asarray(header["minimum"], dtype=np.float64)
            min_energy = float(header["min_energy"])
            minimize_calls = int(header["minimize_calls"])
            min_ok = bool(header["min_ok"])
            found = [np.asarray(row["saddle_x"], dtype=np.float64) for row in saved if row.get("saddle_x") is not None]
            connected = sum(1 for row in saved if row.get("success"))
            searches = len(saved)
            rng = _rng_restore(saved[-1]["rng_state"]) if saved else np.random.default_rng(20261015 + rep)
            for row in saved:
                all_records.append({key: value for key, value in row.items() if key != "rng_state"})
            print(f"{template['id']} rep {rep} resume searches {searches} connected {connected}", flush=True)
        else:
            minimum_positions = positions.copy()
            spec = _minimize_spec(base, minimum_positions, name=f"rep{rep}-minimum", cap=INITIAL_MIN_CALLS)
            outcome = run_start(spec)
            if outcome.get("x") is None:
                raise RuntimeError(f"initial minimization crashed: {outcome.get('error', outcome.get('failure'))}")
            minimum = np.asarray(outcome["x"], dtype=np.float64)
            min_energy = float(outcome["energy"])
            minimize_calls = int(outcome["force_calls"])
            min_ok = bool(outcome.get("converged"))
            rng = np.random.default_rng(20261015 + rep)
            found = []
            connected = 0
            searches = 0
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(
                json.dumps(
                    {
                        "template": template["id"],
                        "rep": rep,
                        "minimum": minimum.tolist(),
                        "min_energy": min_energy,
                        "minimize_calls": minimize_calls,
                        "min_ok": bool(min_ok),
                    }
                ),
                encoding="utf-8",
            )
            partial_path.unlink(missing_ok=True)
        _sync_saddle_campaign(output, all_records, template, problem["id"])
        # Climbs in one batch are independent. They are drawn and scored in
        # order, so a resumed catalog matches a one-worker run.
        nworkers = max(1, int(workers))
        pool = ProcessPoolExecutor(max_workers=nworkers) if nworkers > 1 else None
        try:
            while connected < goal and searches < max_searches:
                if _stop_requested():
                    finished = False
                    break
                width = min(nworkers, max_searches - searches)
                batch = []
                for _ in range(width):
                    searches += 1
                    spec = _draw_saddle_spec(rng, searches, rep, listed, free, positions, minimum, magnitude, base, fixed)
                    batch.append((spec, _rng_state(rng)))
                specs = [spec for spec, _state in batch]
                outcomes = list(pool.map(run_start, specs)) if pool is not None else [run_start(spec) for spec in specs]
                halt = False
                for (spec, rng_state), outcome in zip(batch, outcomes):
                    connected, refused, failure = _commit_saddle_outcome(
                        spec,
                        outcome,
                        rng_state,
                        found=found,
                        all_records=all_records,
                        connected=connected,
                        positions=positions,
                        fixed=fixed,
                        cell=cell,
                        minimum=minimum,
                        min_energy=min_energy,
                        force_tol=force_tol,
                        connect_rmsd=connect_rmsd,
                        barrier_max=barrier_max,
                        base=base,
                        partial_path=partial_path,
                        output=output,
                        template=template,
                        problem_id=problem["id"],
                    )
                    index = len(all_records)
                    if index == 1 or index % 5 == 0 or connected >= goal:
                        print(
                            f"{template['id']} rep {rep} searches {index} connected {connected} "
                            f"calls {all_records[-1]['force_calls']} failure {failure}",
                            flush=True,
                        )
                    if refused:
                        finished = False
                        print(
                            f"{template['id']} rep {rep} stopped after {index} searches: the optimizer refuses TRANSITION_STATE",
                            flush=True,
                        )
                        halt = True
                        break
                    if connected >= goal:
                        halt = True
                        break
                if halt:
                    searches = len(all_records)
                    break
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        if _stop_requested():
            finished = False
        if connected < goal:
            failed_reps += 1
            if searches < max_searches:
                finished = False
        fraction = connected / searches if searches else 0.0
        total_calls = minimize_calls + sum(int(row["force_calls"]) for row in all_records if str(row["start"]).startswith(f"rep{rep}-"))
        comments.append(
            f"# rep {rep} force_calls {total_calls} unique_saddles {len(found)} "
            f"fraction_connected {fraction} total_searches {searches} "
            f"connected {connected} minimize_calls {minimize_calls} "
            f"minimum_converged {int(min_ok)} seed {20261015 + rep}"
        )
    text = _saddle_text(comments, all_records, template, problem["id"], goal, failed_reps, repetitions, partial=not finished)
    _write_dat(output, text)
    write_starts(output.with_suffix(".starts.jsonl"), all_records)
    if finished:
        header_path, partial_path = _checkpoint(work)
        partial_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
    return {
        "force_calls": None,
        "nfailed": failed_reps,
        "starts": len(all_records),
        "template": template["id"],
        "problem": problem["id"],
        "records": all_records,
    }


def _place(positions: np.ndarray, free: np.ndarray, flat: np.ndarray) -> np.ndarray:
    moved = np.array(positions, dtype=np.float64, copy=True)
    moved[free] = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
    return moved


def _axis_from_point(base: dict, saddle_x: np.ndarray, positions: np.ndarray, fixed: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Negative eigenvector at the saddle. The finite difference is a verification."""
    raw = potential_from_problem({"potential": base["potential"]})
    geometry = CountedGeometry(
        raw,
        positions,
        fixed,
        cell,
        cap=None,
        force_tol=float(base["force_tol"]),
        force_match="library",
        dr_ang=0.0,
    )
    _values, vectors = _hessian_spectrum(geometry, saddle_x)
    return _unit(np.asarray(vectors[:, 0], dtype=np.float64))


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _cheap_saddle_metrics(records: list[dict]) -> dict:
    """PAR-2 mean and failure counts. The running file skips the bootstrap."""
    scored = [
        {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
        for row in records
    ]
    tally = failure_tally(scored, SEARCH_CAP)
    successes = sum(1 for row in scored if row["success"])
    mean = statistics.fmean(
        penalised_cost(float(row["force_calls"]), bool(row["success"]), SEARCH_CAP) for row in scored
    )
    metrics = {
        "par2_mean": mean,
        "nfailed": len(records) - successes,
        "successes": successes,
    }
    metrics.update({f"n_{name}": tally[name] for name in FAILURES})
    return metrics


def _sync_saddle_campaign(output: Path, records: list[dict], template: dict, problem_id: str) -> None:
    """Rewrite the campaign pair from the checkpoint before new searches append."""
    path = output.with_suffix(".starts.jsonl")
    if not records:
        path.unlink(missing_ok=True)
        return
    text = _saddle_text([], records, template, problem_id, 27, 0, 1, partial=True, full_score=False)
    _write_dat(output, text)
    write_starts(path, [_public(row) for row in records])


def _emit_saddle(output: Path, records: list[dict], template: dict, problem_id: str) -> None:
    if not records:
        return
    # The running file stays labelled partial until the goal or the search limit.
    # Bootstrap stays on the final write. Repeating it every search dominates
    # the catalog once the sample is large.
    text = _saddle_text([], records, template, problem_id, 27, 0, 1, partial=True, full_score=False)
    _write_dat(output, text)
    path = output.with_suffix(".starts.jsonl")
    line = json.dumps(_public(records[-1]), sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _saddle_text(comments: list[str], records: list[dict], template: dict, problem_id: str, goal: int, failed_reps: int, repetitions: int, *, partial: bool, full_score: bool = True) -> str:
    lines = list(comments)
    totals = []
    for line in comments:
        parts = line.split()
        if "force_calls" in parts:
            totals.append(int(parts[parts.index("force_calls") + 1]))
    if totals:
        lines.append(f"force_calls {statistics.fmean(totals)}")
        lines.append(f"force_calls_min {min(totals)}")
        lines.append(f"force_calls_max {max(totals)}")
    if not records:
        metrics = None
    elif full_score:
        metrics = score_v2(
            [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records],
            SEARCH_CAP,
        )
    else:
        metrics = _cheap_saddle_metrics(records)
    lines.append(f"code {template.get('code', 'CP2K')}")
    lines.append(f"code_version {template.get('code_version', '2026.2')}")
    lines.append(f"commit {template.get('commit', _COMMIT)}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem_id}")
    lines.append("counter harness")
    lines.append(f"repetitions {repetitions}")
    lines.append(f"nfailed {failed_reps}")
    lines.append(f"goal {goal}")
    lines.append(f"cap {SEARCH_CAP}")
    lines.append(f"n {len(records)}")
    if partial:
        lines.append("partial true")
    if metrics is not None:
        lines.append(f"par2_mean {metrics['par2_mean']}")
        lines.append(f"v2_nfailed {metrics['nfailed']}")
        lines.append(f"successes {metrics['successes']}")
        lines.append(f"n_cap {metrics['n_cap']}")
        lines.append(f"n_wrong_stationary {metrics['n_wrong_stationary']}")
        lines.append(f"n_divergence {metrics['n_divergence']}")
        lines.append(f"n_crash {metrics['n_crash']}")
    return "\n".join(lines) + "\n"


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = True,
) -> dict:
    """Run one CP2K template and write the campaign files."""
    from .campaign import _check_stop

    if str(template.get("commit", "")) and not str(template.get("commit", "")).startswith(_COMMIT[:12]):
        raise ValueError(f"{template['id']} commit is not {_COMMIT}")
    convergence = problem.get("convergence") or {}
    force_tol = float(convergence.get("force_norm", 1e-3))
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    if problem["id"] == "saddle-search/pt-island":
        return _run_saddle(root, problem, template, output=output, force_tol=force_tol, work=scratch, workers=workers)
    cap = _check_stop(problem, template)
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    specs = _ts_specs(root, problem, template, cap=cap, force_tol=force_tol, work=scratch, limit=limit)
    expected = int((problem.get("sample") or {}).get("starts") or len(specs))
    if limit <= 0 and len(specs) != expected:
        raise RuntimeError(f"{problem['id']} has {len(specs)} geometries, the problem states {expected}")
    done_dir = scratch / "done"
    done = _load_done(done_dir) if resume else {}
    records = run_starts(specs, workers=workers, done=done, done_dir=done_dir)
    summary = _emit_starts(output, records, template, cap, problem["id"])
    collect_provenance(
        code=str(template.get("code", "CP2K")),
        code_version=str(template.get("code_version", "2026.2")),
        commit=str(template.get("commit", _COMMIT)),
        parameters={"template": template["id"], "problem": problem["id"], "cap": cap, "force_tol": force_tol},
    )
    return summary


def spec_for_start(root: Path, template_id: str, start: str, *, work: Path) -> dict:
    """One tsopt spec. The saddle catalog is a campaign, not one of these starts."""
    from .campaign import _check_stop, load_template, resolve_problem

    template = load_template(root, template_id)
    problem = resolve_problem(root, str(template["problem"]))
    if problem["id"] == "saddle-search/pt-island":
        raise ValueError("saddle-search is a displacement campaign")
    cap = _check_stop(problem, template)
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    force_tol = float((problem.get("convergence") or {}).get("force_norm", 1e-3))
    specs = _ts_specs(root, problem, template, cap=cap, force_tol=force_tol, work=work, limit=0)
    found = next((spec for spec in specs if spec["name"] == start), None)
    if found is None:
        raise RuntimeError(f"{problem['id']} has no start {start}")
    return found
