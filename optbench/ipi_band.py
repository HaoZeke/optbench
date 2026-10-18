"""i-PI NEB and string campaigns on the spec potential.

i-PI is the server. This module is the unix-socket client. Every
POSDATA/GETFORCE exchange is one harness force call, with the cache
off. The potential is the problem's Lennard-Jones or Morse function.
The numbers i-PI stores are the problem numbers: the input says
atomic_unit, whose conversion factor is 1, and the socket forcefield
has periodic boundaries off so a cluster is not folded into the cell.

The search stops when the highest-energy interior image has a
free-atom physical gradient at or below the problem force stop, or
when the force-call cap is reached. i-PI's own energy and position
tolerances are loose. Its force tolerance is 1e-8, the max absolute
component of the NEB or string gradient, so that stop does not fire
before the physical gradient. A return under the cap with the
physical gradient still above the stop is divergence. A climbing
template starts in the climb stage. After every interior image has
been seen once, a later force call on the current highest image can
end the search. A plain band waits for a full interior round.

The Hessian test matches the rgsaddle band. Finite differences go
through verification and are not search cost. A free Lennard-Jones
cluster drops the rigid modes. A slab with fixed atoms does not.

The NEB rows use damped BFGS. That optimizer reads biggest_step, which
is the problem max_move. FIRE is the library default and does not read
biggest_step; on this potential its velocity correction steps into the
repulsive core. The plain string uses BFGSTRM, the library default. It reads
biggest_step and the trust radius. On one climbing bead that loop
stops requesting forces once a rejected step does not move, so the
climbing string uses damped BFGS. i-PI 3.3.1 refuses the string
motion mode at input time and tells the reader to skip that guard.
The launcher skips that one message. The string climber's bind names
attributes the mover does not have, so the launcher binds it the way
the NEB climber does. The climb step indexes ``fixmask``, which the
mapper never sets, so the bind stores that name as the same mask. A
climb started immediately leaves the stored force array empty, and
the launcher copies the forces already computed to choose the climber.
The string step itself is the shipped one.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .counter import ForceCallCap, HarnessCounter
from .failures import classify_failure
from .pes import _mic, evaluate, load_pairs
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance, write_starts

CODE_ID = "ipi"
CODE_NAME = "i-PI"
CODE_VERSION = "3.3.1"
CODE_COMMIT = "a6b350327ce7cb8fdcc17192b1b9af88513b662f"
_HDRLEN = 12
_EIGEN_TOL = 1e-6
_FD_STEP = 1e-5
_LOOSE = 1.0e8
_CELL = 1.0e6
_SOCKET_PREFIX = "/tmp/ipi_"
_SIDECAR_ROOT = Path(os.environ.get("OPTBENCH_IPI_SIDECAR", "/tmp/optbench-ipi-sidecars"))

_MSG = {
    name: name.upper().ljust(_HDRLEN).encode("ascii")
    for name in ("STATUS", "NEEDINIT", "READY", "HAVEDATA", "FORCEREADY", "EXIT")
}


def ipi_executable() -> Path | None:
    """The i-PI console script on PATH, if one is installed."""
    found = shutil.which("i-pi")
    if found:
        return Path(found)
    sibling = Path(sys.executable).resolve().parent / "i-pi"
    return sibling if sibling.is_file() else None


def linear_frames(
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    n_interior: int,
    box: np.ndarray | None,
) -> list[np.ndarray]:
    """Reactant, interior images, and the minimum-image product.

    Fixed atoms stay on the reactant. The free atoms move in a straight
    line in the minimum-image displacement.
    """
    if n_interior < 1:
        raise ValueError("a band needs at least one interior image")
    start = np.asarray(reactant, dtype=np.float64)
    end = np.asarray(product, dtype=np.float64)
    mask = np.asarray(free, dtype=bool)
    lengths = None if box is None else np.asarray(box, dtype=np.float64)
    delta = _mic(end - start, lengths)
    total = n_interior + 2
    frames = []
    for index in range(total):
        moved = np.array(start, copy=True)
        moved[mask] = start[mask] + (index / (total - 1)) * delta[mask]
        frames.append(moved)
    return frames


def write_xyz(path: Path, frames: list[np.ndarray]) -> None:
    """Concatenated XYZ frames. The symbol is only a mass label."""
    lines: list[str] = []
    count = frames[0].shape[0]
    for frame in frames:
        lines.append(str(count))
        lines.append("optbench")
        for row in frame:
            coords = " ".join(format(float(value), ".16g") for value in row)
            lines.append(f"Ne {coords}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_xml(
    *,
    address: str,
    nbeads: int,
    method: str,
    optimizer: str,
    climb: bool,
    tangent: str,
    spring_k: float,
    max_move: float,
    force_norm: float,
    fixed: list[int],
    stage: str,
) -> str:
    """One i-PI input. Lengths are atomic_unit, so they are not converted.

    ``force_norm`` is the harness stop. It is not written into the
    input. The library force tolerance stays at 1e-8.
    """
    if method not in {"neb", "string"}:
        raise ValueError(f"unknown i-PI motion {method}")
    if force_norm <= 0:
        raise ValueError("the problem force stop must be positive")
    allowed = {"neb", "climb"} if method == "neb" else {"string", "climb"}
    if stage not in allowed:
        raise ValueError(f"{method} has no stage {stage}")
    climb_text = "true" if climb else "false"
    fixed_xml = ""
    if fixed:
        body = ", ".join(str(index) for index in fixed)
        fixed_xml = f"      <fixatoms> [ {body} ] </fixatoms>\n"
    if method == "neb":
        motion = f"""    <motion mode='neb'>
      <fixcom> false </fixcom>
{fixed_xml}      <neb_optimizer mode='{optimizer}'>
        <tolerances>
          <energy units='atomic_unit'> {_LOOSE:.1e} </energy>
          <force units='atomic_unit'> 1e-8 </force>
          <position units='atomic_unit'> {_LOOSE:.1e} </position>
        </tolerances>
        <biggest_step units='atomic_unit'> {max_move:.16g} </biggest_step>
        <spring>
          <varsprings> false </varsprings>
          <kappa> {spring_k:.16g} </kappa>
          <kappamax> 1.5 </kappamax>
          <kappamin> 0.5 </kappamin>
        </spring>
        <tangent> {tangent} </tangent>
        <stage> {stage} </stage>
        <use_climb> {climb_text} </use_climb>
      </neb_optimizer>
    </motion>"""
    else:
        motion = f"""    <motion mode='string'>
      <fixcom> false </fixcom>
{fixed_xml}      <string_optimizer mode='{optimizer}'>
        <tolerances>
          <energy units='atomic_unit'> {_LOOSE:.1e} </energy>
          <force units='atomic_unit'> 1e-8 </force>
          <position units='atomic_unit'> {_LOOSE:.1e} </position>
        </tolerances>
        <biggest_step units='atomic_unit'> {max_move:.16g} </biggest_step>
        <tr_trm units='atomic_unit'> [ {max_move:.16g} ] </tr_trm>
        <stage> {stage} </stage>
        <use_climb> {climb_text} </use_climb>
      </string_optimizer>
    </motion>"""
    cell = ", ".join(
        [
            format(_CELL, ".1f"),
            "0",
            "0",
            "0",
            format(_CELL, ".1f"),
            "0",
            "0",
            "0",
            format(_CELL, ".1f"),
        ]
    )
    return f"""<simulation mode='static' verbosity='quiet'>
  <total_steps> 1000000 </total_steps>
  <prng>
    <seed> 12345 </seed>
  </prng>
  <output prefix='simulation'>
    <properties stride='1000000' filename='out'> [ step ] </properties>
  </output>
  <ffsocket name='spec' mode='unix' pbc='false' threaded='true'>
    <address> {address} </address>
    <slots> 1 </slots>
    <latency> 1e-4 </latency>
    <timeout> 60 </timeout>
    <exit_on_disconnect> true </exit_on_disconnect>
    <consolidate_messages> true </consolidate_messages>
    <batch_size> 1 </batch_size>
  </ffsocket>
  <system>
    <initialize nbeads='{nbeads}'>
      <file mode='xyz' units='atomic_unit'> init.xyz </file>
      <cell units='atomic_unit'> [ {cell} ] </cell>
    </initialize>
    <forces>
      <force forcefield='spec'/>
    </forces>
{motion}
  </system>
</simulation>
"""


class _Stream:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = bytearray()

    def exact(self, count: int) -> bytes:
        while len(self.buf) < count:
            chunk = self.sock.recv(max(4096, count - len(self.buf)))
            if not chunk:
                raise EOFError("i-PI closed the socket")
            self.buf.extend(chunk)
        out = bytes(self.buf[:count])
        del self.buf[:count]
        return out

    def header(self) -> bytes:
        return self.exact(_HDRLEN)

    def f64(self, count: int) -> np.ndarray:
        return np.frombuffer(self.exact(8 * count), dtype=np.float64).copy()

    def i32(self) -> int:
        return int(np.frombuffer(self.exact(4), dtype=np.int32)[0])


def _header_is(raw: bytes, name: str) -> bool:
    return raw.strip().upper() == name.encode("ascii")


class _Client:
    """One band. The socket is the only place a search call is counted."""

    def __init__(
        self,
        frames: list[np.ndarray],
        free: np.ndarray,
        box: np.ndarray | None,
        kind: str,
        force_norm: float,
        cap: int,
        single_image: bool = False,
    ):
        self.slots = [np.array(frame, dtype=np.float64, copy=True) for frame in frames]
        self.reactant = self.slots[0]
        self.free = np.asarray(free, dtype=bool)
        self.box = None if box is None else np.asarray(box, dtype=np.float64)
        self.kind = kind
        self.force_norm = float(force_norm)
        self.cap = int(cap)
        self.single_image = bool(single_image)
        self.counter = HarnessCounter(cache=False, cap=self.cap)
        self.energies = [float("nan")] * len(frames)
        self.grads: list[np.ndarray | None] = [None] * len(frames)
        self.met = False
        self.hit_cap = False
        self.stop = False
        self.crashed = ""
        self.reactant_maxabs = float("inf")
        self.fresh = [False] * len(frames)
        self.seen = [False] * len(frames)
        self.band_ready = False
        self._pending: np.ndarray | None = None

    def serve(self, sock: socket.socket) -> None:
        stream = _Stream(sock)
        inited = False
        while not self.stop:
            try:
                header = stream.header()
            except EOFError:
                break
            except (socket.timeout, OSError) as exc:
                if not self.met and not self.hit_cap:
                    self.crashed = type(exc).__name__
                break
            if _header_is(header, "STATUS"):
                if self._pending is not None:
                    sock.sendall(_MSG["HAVEDATA"])
                elif not inited:
                    sock.sendall(_MSG["NEEDINIT"])
                else:
                    sock.sendall(_MSG["READY"])
            elif _header_is(header, "INIT"):
                stream.i32()
                length = stream.i32()
                if length:
                    stream.exact(length)
                inited = True
            elif _header_is(header, "POSDATA"):
                stream.f64(9)
                stream.f64(9)
                nat = stream.i32()
                flat = stream.f64(nat * 3)
                self._pending = flat.reshape((nat, 3))
            elif _header_is(header, "GETFORCE"):
                pos = self._pending
                self._pending = None
                if pos is None:
                    self.crashed = "GETFORCE without POSDATA"
                    self.stop = True
                    break
                energy, forces = self._evaluate(pos)
                _send_force(sock, energy, forces)
            elif _header_is(header, "EXIT"):
                break
            else:
                self.crashed = f"unexpected header {header!r}"
                self.stop = True
                break

    def _evaluate(self, pos: np.ndarray) -> tuple[float, np.ndarray]:
        held = np.array(pos, dtype=np.float64, copy=True)
        delta = float(np.max(np.abs(held - self.reactant)))
        if delta < self.reactant_maxabs:
            self.reactant_maxabs = delta
        if held.shape != self.reactant.shape:
            self.crashed = f"atom count {held.shape[0]} is not {self.reactant.shape[0]}"
            self.stop = True
            return 0.0, np.zeros_like(self.reactant)
        try:
            self.counter.charge((self.counter.calls,))
        except ForceCallCap:
            self.hit_cap = True
            self.stop = True
            return 0.0, np.zeros_like(held)
        try:
            energy, grad = evaluate(self.kind, held, self.box)
        except Exception as exc:
            self.crashed = type(exc).__name__
            self.stop = True
            return 0.0, np.zeros_like(held)
        grad = np.asarray(grad, dtype=np.float64)
        index = self._assign(held)
        self.slots[index] = held
        self.energies[index] = float(energy)
        self.grads[index] = grad
        self._consider(index)
        return float(energy), -grad

    def _assign(self, pos: np.ndarray) -> int:
        best = 0
        best_distance = float("inf")
        for index, slot in enumerate(self.slots):
            distance = float(np.sum((slot - pos) ** 2))
            if distance < best_distance:
                best_distance = distance
                best = index
        return best

    def _consider(self, index: int) -> None:
        if index == 0 or index == len(self.slots) - 1:
            return
        self.seen[index] = True
        self.fresh[index] = True
        if not all(self.seen[1:-1]):
            return
        if self.single_image:
            first = not self.band_ready
            self.band_ready = True
            chosen = self.candidate_index()
            if chosen is not None and (first or index == chosen):
                self._maybe_stop(chosen)
            return
        if not all(self.fresh[1:-1]):
            return
        self.fresh = [False] * len(self.slots)
        chosen = self.candidate_index()
        if chosen is None:
            return
        self._maybe_stop(chosen)

    def _maybe_stop(self, chosen: int) -> None:
        norm = self.force_of(chosen)
        if np.isfinite(norm) and norm <= self.force_norm:
            self.met = True
            self.stop = True

    def candidate_index(self) -> int | None:
        best = None
        best_energy = float("-inf")
        for index in range(1, len(self.slots) - 1):
            energy = self.energies[index]
            if not np.isfinite(energy):
                continue
            if energy > best_energy:
                best_energy = energy
                best = index
        return best

    def force_of(self, index: int) -> float:
        grad = self.grads[index]
        if grad is None:
            return float("nan")
        return float(np.linalg.norm(np.asarray(grad, dtype=np.float64)[self.free]))


def _send_force(sock: socket.socket, energy: float, forces: np.ndarray) -> None:
    flat = np.ascontiguousarray(forces, dtype=np.float64).reshape(-1)
    nat = int(forces.shape[0])
    payload = b"".join(
        (
            _MSG["FORCEREADY"],
            np.float64(energy).tobytes(),
            np.int32(nat).tobytes(),
            flat.tobytes(),
            np.zeros(9, dtype=np.float64).tobytes(),
            np.int32(0).tobytes(),
        )
    )
    sock.sendall(payload)


def _patch_string_climb() -> None:
    """Replace the string climber's bind and fill an empty force store.

    StringClimbGrMapper.bind in this revision reads ``ens.fixatoms`` and
    writes ``ffixatoms_mask``. Neither name exists. NEB's climber bind
    uses ``fixatoms_dof``. The climb step indexes ``fixmask``, which the
    mapper never sets, so the bind stores the mask under both names.
    A climb that starts immediately leaves ``full_f`` empty. NEB fills
    that array in ``init_climb``. The string step does not, and then
    indexes it as a bead array. The wrapper copies the forces already
    computed to choose the climber. The step and the force projection
    stay the shipped methods.
    """
    from ipi.engine.beads import Beads
    from ipi.engine.motion.stringmep import StringClimbGrMapper, StringMover
    from ipi.utils.depend import dstrip

    def bind(self, ens):
        self.fixatoms_dof = ens.fixatoms_dof.copy()
        self.fixatoms_mask = np.ones(3 * ens.beads.natoms, dtype=bool)
        if len(ens.fixatoms_dof) > 0:
            self.fixatoms_mask[ens.fixatoms_dof] = 0
        self.rbeads = Beads(ens.beads.natoms, 1)
        self.rcell = ens.cell.clone()
        active = 3 * ens.beads.natoms - len(ens.fixatoms_dof)
        self.q_prev = np.zeros(active)
        self.q_next = np.zeros(active)
        self.rforces = ens.forces.clone(self.rbeads, self.rcell)
        self.fixmask = self.fixatoms_mask

    StringClimbGrMapper.bind = bind
    original_init = StringMover.init_climb

    def init_climb(self):
        index = original_init(self)
        if np.shape(self.full_f) == (0,):
            self.full_f = dstrip(self.forces.f).copy()
        if np.shape(self.full_v) == (0,):
            self.full_v = dstrip(self.forces.pots).copy()
        return index

    StringMover.init_climb = init_climb


def main_server() -> None:
    """Run one i-PI input. The string guard is skipped; the mover is not."""
    from ipi.engine.simulation import Simulation
    from ipi.utils.softexit import softexit

    xml_path = sys.argv[1]
    prefix = sys.argv[2]
    _patch_string_climb()
    original = softexit.trigger

    def trigger(status: str = "restartable", message: str = ""):
        if "String method is experimental" in str(message):
            print("optbench: skipping the i-PI string input guard", flush=True)
            return None
        return original(status=status, message=message)

    softexit.trigger = trigger
    with open(xml_path, encoding="utf-8") as handle:
        simulation = Simulation.load_from_xml(
            handle,
            custom_verbosity="quiet",
            sockets_prefix=prefix,
            request_banner=False,
        )
    softexit.hard_exit = True
    simulation.run()
    if not softexit.triggered:
        softexit.trigger(status=simulation.exit_status, message=simulation.exit_message)


def _threads_one() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _tail(path: Path, limit: int = 800) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=5)


def _project_rigid(free: np.ndarray, kind: str) -> bool:
    mask = np.asarray(free, dtype=bool)
    return bool(np.all(mask) and int(np.count_nonzero(mask)) >= 2 and kind == "lj")


def _judge(client: _Client, *, name: str, project_rigid: bool, log_text: str) -> dict:
    reasons: list[str] = []
    if client.crashed:
        reasons.append("crash")
    index = client.candidate_index()
    if index is None:
        energy = float("nan")
        force = float("nan")
        point = client.slots[1]
        grad = None
    else:
        energy = float(client.energies[index])
        force = client.force_of(index)
        point = client.slots[index]
        grad = client.grads[index]
    if client.counter.calls >= client.cap or client.hit_cap:
        reasons.append("cap")
    if not np.isfinite(force) or force > client.force_norm:
        reasons.append("force_norm")
    hessian_index = None
    if "crash" not in reasons and "cap" not in reasons and "force_norm" not in reasons and grad is not None:
        mask = client.free
        template = np.array(point, dtype=np.float64, copy=True)

        def free_gradient(flat):
            full = np.array(template, copy=True)
            full[mask] = np.asarray(flat, dtype=np.float64).reshape((-1, 3))
            _energy, full_grad = evaluate(client.kind, full, client.box)
            del _energy
            return np.asarray(full_grad, dtype=np.float64)[mask].reshape(-1)

        free_point = np.asarray(point, dtype=np.float64)[mask].reshape(-1)
        with client.counter.verification():
            hessian = numerical_hessian(free_gradient, free_point.tolist(), step=_FD_STEP)
        hessian_index = vibrational_index(
            np.asarray(hessian, dtype=np.float64),
            free_point.reshape((-1, 3)),
            project_rigid,
        )
        if hessian_index != 1:
            reasons.append("hessian_index")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(client.counter.calls),
        "verify_calls": int(client.counter.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if not np.isfinite(energy) else float(energy),
        "force_norm": None if not np.isfinite(force) else float(force),
        "hessian_index": hessian_index,
        "converged": bool(client.met),
        "candidate": "highest",
        "ci_index": None,
        "refined": False,
        "reactant_maxabs": None
        if not np.isfinite(client.reactant_maxabs)
        else float(client.reactant_maxabs),
        "call_unit": "total",
    }
    if client.crashed:
        record["crash"] = client.crashed
    if failure == "crash" and log_text:
        record["log_tail"] = log_text
    return record


def run_search(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    kind: str,
    n_interior: int,
    force_norm: float,
    cap: int,
    max_move: float,
    method: str,
    optimizer: str,
    climb: bool,
    tangent: str,
    spring_k: float,
    stage: str,
) -> dict:
    """One reactant/product pair. The record uses the harness counter."""
    _threads_one()
    frames = linear_frames(reactant, product, free, n_interior, box)
    project = _project_rigid(free, kind)
    client = _Client(frames, free, box, kind, force_norm, cap, single_image=stage == "climb")
    work = Path(tempfile.mkdtemp(prefix="ipi-ob-"))
    address = f"ob{os.getpid()}{uuid.uuid4().hex[:8]}"
    socket_path = Path(_SOCKET_PREFIX + address)
    log_path = work / "ipi.log"
    proc = None
    log_handle = None
    try:
        write_xyz(work / "init.xyz", frames)
        fixed = [int(index) for index in np.flatnonzero(~np.asarray(free, dtype=bool))]
        xml = render_xml(
            address=address,
            nbeads=len(frames),
            method=method,
            optimizer=optimizer,
            climb=climb,
            tangent=tangent,
            spring_k=spring_k,
            max_move=max_move,
            force_norm=force_norm,
            fixed=fixed,
            stage=stage,
        )
        (work / "input.xml").write_text(xml, encoding="utf-8")
        if socket_path.exists():
            socket_path.unlink()
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-m", "optbench.ipi_band", str(work / "input.xml"), _SOCKET_PREFIX],
            cwd=work,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(60)
        deadline = time.monotonic() + 60
        connected = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                sock.connect(str(socket_path))
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.05)
                continue
            connected = True
            break
        if not connected:
            client.crashed = "connect"
            client.stop = True
        else:
            try:
                client.serve(sock)
            except (EOFError, socket.timeout, OSError) as exc:
                if client.counter.calls == 0 and not client.hit_cap and not client.crashed:
                    client.crashed = type(exc).__name__
            finally:
                sock.close()
        log_handle.flush()
        text = _tail(log_path, limit=2000)
        if (
            "Traceback (most recent call last)" in text
            and not client.met
            and not client.hit_cap
            and not client.crashed
        ):
            client.crashed = "traceback"
        elif not client.crashed and client.counter.calls > 0:
            text = ""
        if client.counter.calls == 0 and not client.hit_cap and not client.crashed:
            client.crashed = "no force call"
        return _judge(client, name=name, project_rigid=project, log_text=text)
    except Exception as exc:
        client.crashed = client.crashed or type(exc).__name__
        text = _tail(log_path)
        return _judge(client, name=name, project_rigid=project, log_text=text or type(exc).__name__)
    finally:
        if proc is not None:
            _stop_process(proc)
        if log_handle is not None:
            log_handle.close()
        if socket_path.exists():
            socket_path.unlink()
        if os.environ.get("OPTBENCH_IPI_KEEP"):
            print(f"kept {work}", flush=True)
        else:
            shutil.rmtree(work, ignore_errors=True)


def _registry(root: Path) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == CODE_ID:
            return row
    raise ValueError("codes/registry.toml has no ipi entry")


def _stage(band: dict) -> str:
    """Climbing rows start on the climbing image. A plain row stays on the band."""
    method = str(band["method"])
    requested = band.get("stage")
    if requested:
        return str(requested)
    if bool(band["climb"]):
        return "climb"
    return method


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the i-PI registry")
    if template.get("code_version") != CODE_VERSION:
        raise ValueError(f"{template['id']} version is not {CODE_VERSION}")
    band = template["ipi"]
    problem_band = problem["band"]
    convergence = problem["convergence"]
    return {
        "potential": str(problem_band["potential"]),
        "n_interior": int(problem_band["interior_images"]),
        "max_move": float(problem_band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": problem["geometry"]["kind"],
        "method": str(band["method"]),
        "optimizer": str(band["optimizer"]),
        "climb": bool(band["climb"]),
        "tangent": str(band.get("tangent", "improved")),
        "spring_k": float(band.get("spring_k", 1.0)),
        "stage": _stage(band),
    }


def _job(pair, settings: dict) -> dict:
    box = None if pair.box is None else np.asarray(pair.box, dtype=np.float64)
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64),
        "product": np.asarray(pair.product, dtype=np.float64),
        "free": np.asarray(pair.free, dtype=bool),
        "box": box,
        "settings": settings,
    }


def _execute(job: dict) -> dict:
    _threads_one()
    settings = job["settings"]
    try:
        return run_search(
            name=job["name"],
            reactant=job["reactant"],
            product=job["product"],
            free=job["free"],
            box=job["box"],
            kind=settings["potential"],
            n_interior=settings["n_interior"],
            force_norm=settings["force_norm"],
            cap=settings["cap"],
            max_move=settings["max_move"],
            method=settings["method"],
            optimizer=settings["optimizer"],
            climb=settings["climb"],
            tangent=settings["tangent"],
            spring_k=settings["spring_k"],
            stage=settings["stage"],
        )
    except Exception as exc:
        return {
            "schema": "optbench.start/v2",
            "start": job["name"],
            "force_calls": 0,
            "verify_calls": 0,
            "counter": "harness",
            "comparable": True,
            "success": False,
            "failure": "crash",
            "reasons": ["crash"],
            "crash": type(exc).__name__,
            "energy": None,
            "force_norm": None,
            "hessian_index": None,
            "converged": False,
            "candidate": "crash",
            "refined": False,
            "call_unit": "total",
        }


_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    key = (str(Path(root).resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        loaded = load_pairs(
            geometry_archive(root, problem),
            settings["geometry"],
            settings["pairs"],
        )
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def _stamp(record: dict, *, seed: str, problem: str, template: dict) -> dict:
    record["seed"] = str(seed)
    record["problem"] = problem
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def search_named(
    root: Path,
    problem_name: str,
    template_name: str,
    start: str,
    seed: str = "0",
) -> dict:
    """One archived pair. The seed is recorded. The band does not draw it."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(pairs[start], settings))
    _stamp(record, seed=seed, problem=problem["id"], template=template)
    _write_sidecar(record)
    return record


def sidecar_path(problem_name: str, template_name: str, seed: str, start: str) -> Path:
    slug = problem_name.replace("/", "-") + "--" + template_name.replace("/", "-")
    return _SIDECAR_ROOT / slug / str(seed) / f"{start}.json"


def _write_sidecar(record: dict) -> None:
    path = sidecar_path(record["problem"], record["template"], str(record.get("seed", "0")), record["start"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, problem: dict) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness socket POSDATA",
        "# force_calls count raw potential evaluations and are not divided by the image count",
        "# the mean is round-half-even over successes; cap, force, or Hessian index other than 1 is a failure",
        f"# problem {problem['id']}",
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
            lines.append("call_unit total")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row['hessian_index'] == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append(f"problem {problem['id']}")
    lines.append(f"images {settings['n_interior']}")
    return "\n".join(lines) + "\n"


def _legacy(records: list[dict]) -> dict:
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary = {
        "successes": len(successes),
        "nfailed": len(records) - len(successes),
        "n": len(records),
        "transition_states": sum(1 for row in records if row["hessian_index"] == 1),
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
) -> dict:
    """Write benchmark.dat, the per-start file, and the v2 score."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(records, template, settings, problem), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    parameters = dict(settings)
    parameters["counter"] = "harness"
    parameters["seed"] = "0"
    parameters["ipi_prng_seed"] = 12345
    parameters["units"] = "atomic_unit"
    parameters["pbc"] = False
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
            [{"force_calls": row["force_calls"], "success": row["success"]} for row in records],
            settings["cap"],
        ),
        "legacy": _legacy(records),
    }
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
    }


def _read_starts(output: Path) -> dict[str, dict]:
    path = output.with_suffix(".starts.jsonl")
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[str(row["start"])] = row
    return found


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
    if template["engine"] != "ipi-band":
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem, settings).values())
    if limit > 0:
        pairs = pairs[:limit]
    jobs = [_job(pair, settings) for pair in pairs]
    done = _read_starts(output) if resume else {}
    pending = [job for job in jobs if job["name"] not in done]
    records = {name: row for name, row in done.items() if any(job["name"] == name for job in jobs)}

    def publish() -> None:
        ordered = []
        for job in jobs:
            row = records.get(job["name"])
            if row is None:
                continue
            ordered.append(_stamp(dict(row), seed="0", problem=problem["id"], template=template))
        if not ordered:
            return
        write_outputs(output, ordered, template, settings, problem)
        for row in ordered:
            _write_sidecar(row)

    _threads_one()
    if pending and workers <= 1:
        for job in pending:
            records[job["name"]] = _execute(job)
            publish()
    elif pending:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_execute, job): job["name"] for job in pending}
            for future in as_completed(futures):
                name = futures[future]
                records[name] = future.result()
                publish()
    ordered = [_stamp(dict(records[job["name"]]), seed="0", problem=problem["id"], template=template) for job in jobs]
    for row in ordered:
        _write_sidecar(row)
    return write_outputs(output, ordered, template, settings, problem)


if __name__ == "__main__":
    main_server()
