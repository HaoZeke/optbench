"""i-PI force slave for FLAME minhocao's msock potential.

FLAME binds a TCP port and then writes the i-PI driver messages. This
process connects, answers ``STATUS``, and on ``POSDATA`` evaluates the
harness potential. Positions arrive in Bohr. Energy goes back in Hartree
and forces in Hartree per Bohr, using the Bohr and Hartree constants
compiled into FLAME (``defs_basis``).

One ``POSDATA`` is one entry into the potential. ``before_eval`` runs
before that entry is charged, so a stop decided from ``global.mon`` does
not count the next hop.
"""

from __future__ import annotations

import socket
import struct
from collections.abc import Callable

from optbench.counter import ForceCallCap

MSGLEN = 12
# defs_basis in FLAME src/constants_minhocao_mod.F90
BOHR_ANG = 0.52917720859
HA_EV = 27.21138386

_HEADER = {
    "READY": b"READY       ",
    "HAVEDATA": b"HAVEDATA    ",
    "FORCEREADY": b"FORCEREADY  ",
}


class ProtocolError(RuntimeError):
    """FLAME sent a header this slave does not answer."""


class SearchStopped(Exception):
    """``before_eval`` ended the search before the next potential entry."""

    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(kind)


def angstrom_from_bohr(values):
    """Cartesian Bohr coordinates as Angstrom."""
    return values * BOHR_ANG


def hartree_from_lj(energy: float) -> float:
    """Lennard-Jones energy, with epsilon taken as one eV, in Hartree."""
    return float(energy) / HA_EV


def lj_from_hartree(energy: float) -> float:
    """Hartree energy back in the Lennard-Jones unit."""
    return float(energy) * HA_EV


def ha_per_bohr_from_lj_gradient(gradient):
    """``dE/d(Angstrom)`` as a force in Ha/Bohr.

    The force is the negative gradient. ``r_Ang = r_Bohr * BOHR_ANG`` and
    ``E_Ha = E_LJ / HA_EV``.
    """
    return -gradient * (BOHR_ANG / HA_EV)


def _read_exact(conn: socket.socket, nbytes: int) -> bytes | None:
    chunks = []
    got = 0
    while got < nbytes:
        piece = conn.recv(nbytes - got)
        if not piece:
            if got == 0:
                return None
            raise ProtocolError(f"socket closed after {got} of {nbytes} bytes")
        chunks.append(piece)
        got += len(piece)
    return b"".join(chunks)


def _write(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(payload)


def _header(conn: socket.socket) -> str | None:
    raw = _read_exact(conn, MSGLEN)
    if raw is None:
        return None
    return raw.decode("ascii", "replace").strip()


def _read_ints(conn: socket.socket, count: int) -> list[int]:
    raw = _read_exact(conn, 4 * count)
    if raw is None:
        raise ProtocolError("socket closed while reading integers")
    return list(struct.unpack("<" + "i" * count, raw))


def _read_doubles(conn: socket.socket, count: int):
    import numpy as np

    raw = _read_exact(conn, 8 * count)
    if raw is None:
        raise ProtocolError("socket closed while reading doubles")
    return np.frombuffer(raw, dtype="<f8").copy()


def _write_int(conn: socket.socket, value: int) -> None:
    _write(conn, struct.pack("<i", int(value)))


def _write_doubles(conn: socket.socket, values) -> None:
    import numpy as np

    array = np.asarray(values, dtype="<f8").reshape(-1)
    _write(conn, array.tobytes())


def lowest_relaxed_lj(path) -> float | None:
    """Lowest enthalpy in ``global.mon``, converted to Lennard-Jones units.

    minhocao writes one line per relaxed hop. The third field is the
    enthalpy in Hartree. Pressure is zero in this campaign, so the
    enthalpy is the potential energy.
    """
    from pathlib import Path

    file = Path(path)
    if not file.is_file():
        return None
    best = None
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            enthalpy = float(parts[2])
        except ValueError:
            continue
        energy = lj_from_hartree(enthalpy)
        if best is None or energy < best:
            best = energy
    return best


def _guard(conn: socket.socket, before_eval: Callable[[], None] | None) -> None:
    if before_eval is None:
        return
    try:
        before_eval()
    except SearchStopped:
        conn.close()
        raise


def _nodelay(conn: socket.socket) -> None:
    """Do not hold the 12-byte i-PI headers for a delayed ACK."""
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        return


def serve(
    conn: socket.socket,
    evaluate: Callable,
    *,
    before_eval: Callable[[], None] | None = None,
) -> None:
    """Answer one FLAME msock session.

    ``evaluate`` receives Angstrom coordinates of shape ``(3 * nat,)`` and
    returns ``(energy, gradient)`` in Lennard-Jones units. It charges the
    harness counter. ``before_eval`` may raise ``SearchStopped`` to close
    the socket before that charge.
    """
    _nodelay(conn)
    pending: tuple[float, int, object] | None = None
    while True:
        token = _header(conn)
        if token is None:
            return
        if token == "STATUS":
            if pending is None:
                _guard(conn, before_eval)
                _write(conn, _HEADER["READY"])
            else:
                _write(conn, _HEADER["HAVEDATA"])
        elif token == "POSDATA":
            _read_doubles(conn, 9)
            _read_doubles(conn, 9)
            nat = _read_ints(conn, 1)[0]
            if nat < 1:
                raise ProtocolError(f"nat {nat}")
            raw = _read_doubles(conn, 3 * nat)
            _guard(conn, before_eval)
            try:
                energy, gradient = evaluate(angstrom_from_bohr(raw))
            except (ForceCallCap, SearchStopped):
                conn.close()
                raise
            pending = (hartree_from_lj(energy), nat, ha_per_bohr_from_lj_gradient(gradient))
        elif token == "GETFORCE":
            if pending is None:
                raise ProtocolError("GETFORCE before a force evaluation")
            energy_ha, nat, forces = pending
            pending = None
            _write(conn, _HEADER["FORCEREADY"])
            _write_doubles(conn, [energy_ha])
            _write_int(conn, nat)
            _write_doubles(conn, forces)
            _write_doubles(conn, [0.0] * 9)
            _write_int(conn, 0)
        elif token == "STOP":
            return
        else:
            raise ProtocolError(f"unexpected header {token!r}")
