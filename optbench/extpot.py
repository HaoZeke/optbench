"""eonclient ``ext_pot`` bridge onto the counting potential server.

eOn writes ``from_eon_to_extpot`` and runs this module. The file holds
three box rows and then ``Z x y z`` rows, tab-separated. The reply file
``from_extpot_to_eon`` is the energy and one force row per atom.

``OPTBENCH_POTSERVER`` is ``host:port``. One run of this module is one
force call on that server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .potserver import remote_calculate


def read_extpot(text: str) -> tuple[list[float], list[int], list[float]]:
    """Box, atomic numbers, and flat positions from ``from_eon_to_extpot``."""
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) < 4:
        raise ValueError("ext_pot request is shorter than a box plus one atom")
    box: list[float] = []
    for line in rows[:3]:
        parts = line.split()
        if len(parts) != 3:
            raise ValueError("box row must hold three floats")
        box.extend(float(part) for part in parts)
    numbers: list[int] = []
    positions: list[float] = []
    for line in rows[3:]:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError("atom row must be Z x y z")
        numbers.append(int(float(parts[0])))
        positions.extend(float(part) for part in parts[1:])
    return box, numbers, positions


def format_extpot(energy: float, forces: tuple[float, ...] | list[float]) -> str:
    """The ``from_extpot_to_eon`` body."""
    if len(forces) % 3 != 0:
        raise ValueError("forces must be x, y, z triples")
    lines = [repr(float(energy))]
    atoms = len(forces) // 3
    for index in range(atoms):
        base = 3 * index
        lines.append(" ".join(repr(float(forces[base + axis])) for axis in range(3)))
    return "\n".join(lines) + "\n"


def respond(
    directory: Path | None = None,
    endpoint: str | None = None,
) -> int:
    """Read the exchange files in ``directory`` and answer one evaluation."""
    work = directory or Path.cwd()
    address = endpoint or os.environ.get("OPTBENCH_POTSERVER", "")
    if ":" not in address:
        raise RuntimeError("OPTBENCH_POTSERVER must be host:port")
    host, port_text = address.rsplit(":", 1)
    request = (work / "from_eon_to_extpot").read_text(encoding="utf-8")
    box, numbers, positions = read_extpot(request)
    energy, forces, calls = remote_calculate(host, int(port_text), positions, numbers, box)
    (work / "from_extpot_to_eon").write_text(format_extpot(energy, forces), encoding="utf-8")
    return calls


def main() -> None:
    try:
        respond()
    except Exception as exc:
        print(f"optbench ext_pot: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
