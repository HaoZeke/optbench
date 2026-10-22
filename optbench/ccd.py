"""Putative global minima fetched from the Cambridge Cluster Database.

The coordinate files and the HTML tables under ``optbench/data/ccd`` are the
pages named below, saved in this checkout. Energies are read from those
files. The Lennard-Jones table is the N<=150 page linked from the cluster
index. The binary file is the improved N=100, size ratio 1.3 minimum linked
from the database homepage. TIP4P energies are the xyz headers, and the HTML
table is the same values printed to fewer digits.

Lennard-Jones clusters:
https://www-wales.ch.cam.ac.uk/CCD.html
http://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ.html
https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html

Binary Lennard-Jones:
http://www-wales.ch.cam.ac.uk/~wales/CCD/NEW_BINARY_LJ_MINIMA/
http://www-wales.ch.cam.ac.uk/~wales/CCD/NEW_BINARY_LJ_MINIMA/1.3/100

TIP4P water:
http://www-wales.ch.cam.ac.uk/~wales/CCD/TIP4P-water.html
https://www-wales.ch.cam.ac.uk/~wales/CCD/TIP4P/xyz/TIP4P-10.xyz
https://www-wales.ch.cam.ac.uk/~wales/CCD/TIP4P/xyz/TIP4P-15.xyz
https://www-wales.ch.cam.ac.uk/~wales/CCD/TIP4P/xyz/TIP4P-20.xyz
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "ccd"

LJ_TABLE_URL = "https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html"
LJ_INDEX_URL = "http://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/LJ.html"
CCD_HOME_URL = "https://www-wales.ch.cam.ac.uk/CCD.html"
BINARY_INDEX_URL = "http://www-wales.ch.cam.ac.uk/~wales/CCD/NEW_BINARY_LJ_MINIMA/"
BINARY_POINTS_URL = "http://www-wales.ch.cam.ac.uk/~wales/CCD/NEW_BINARY_LJ_MINIMA/1.3/100"
TIP4P_PAGE_URL = "http://www-wales.ch.cam.ac.uk/~wales/CCD/TIP4P-water.html"

LJ_SIZES = (38, 75, 98, 150)
TIP4P_SIZES = (10, 15, 20)


def _read(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8", errors="replace")


def lennard_jones_table(text: str | None = None) -> dict[int, dict[str, str | float]]:
    """Symmetry and energy for each N on the Lennard-Jones table page."""
    raw = _read("lj-tables.html") if text is None else text
    parts = re.split(r'<a href="points/(\d+)">\d+</a>', raw)
    found: dict[int, dict[str, str | float]] = {}
    for index in range(1, len(parts), 2):
        size = int(parts[index])
        cells = re.findall(r"<td>(.*?)</td>", parts[index + 1], flags=re.S)
        if len(cells) < 2:
            continue
        symmetry = re.sub(r"<[^>]+>", "", cells[0]).replace("&nbsp;", "").strip()
        energy_text = cells[1].strip()
        found[size] = {
            "symmetry": symmetry,
            "energy_text": energy_text,
            "energy": float(energy_text),
        }
    return found


def lennard_jones_minimum(size: int) -> dict[str, str | float | np.ndarray]:
    """Table row plus the point file for one Lennard-Jones size."""
    row = lennard_jones_table()[size]
    positions = _xyz_rows(_read(f"lj-{size}.xyz"))
    if len(positions) != size:
        raise ValueError(f"LJ {size} points file has {len(positions)} atoms")
    return {
        "size": size,
        "symmetry": row["symmetry"],
        "energy_text": row["energy_text"],
        "energy": row["energy"],
        "positions": positions,
        "page": LJ_TABLE_URL,
        "points": f"https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/{size}",
    }


def _xyz_rows(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
        except ValueError:
            continue
    if not rows:
        raise ValueError("coordinate file has no xyz rows")
    return np.asarray(rows, dtype=np.float64)


def binary_minimum() -> dict[str, str | float | np.ndarray]:
    """Improved binary Lennard-Jones N=100 minimum, size ratio 1.3.

    ``species`` is 0 for LA (42 atoms, sigma 1) and 1 for LB (58 atoms, sigma 1.3),
    in the order of the points file.
    """
    text = _read("blj-100.txt")
    match = re.search(r"Energy of minimum 1=\s+([-\d.]+)", text)
    if match is None:
        raise ValueError("binary points file has no energy header")
    energy_text = match.group(1)
    positions = []
    species = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] not in {"LA", "LB"}:
            continue
        species.append(0 if parts[0] == "LA" else 1)
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(positions) != 100:
        raise ValueError(f"binary points file has {len(positions)} atoms")
    labels = np.asarray(species, dtype=np.int64)
    if int(np.sum(labels == 0)) != 42 or int(np.sum(labels == 1)) != 58:
        raise ValueError("binary points file is not 42 LA and 58 LB")
    return {
        "energy_text": energy_text,
        "energy": float(energy_text),
        "positions": np.asarray(positions, dtype=np.float64),
        "species": labels,
        "page": CCD_HOME_URL,
        "index": BINARY_INDEX_URL,
        "points": BINARY_POINTS_URL,
    }


def tip4p_page(text: str | None = None) -> dict[int, str]:
    """Printed kJ/mol values on the TIP4P cluster page, keyed by N."""
    raw = _read("tip4p.html") if text is None else text
    plain = re.sub(r"<[^>]+>", " ", raw)
    plain = plain.replace("&nbsp;", " ")
    found: dict[int, str] = {}
    for match in re.finditer(r"\b(\d+)\s+(-?\d+\.\d+)\s*kJ/mol", plain):
        found[int(match.group(1))] = match.group(2)
    return found


def tip4p_minimum(size: int) -> dict[str, str | float | np.ndarray]:
    """TIP4P xyz file. Coordinates are O, H, H repeated. The header carries the energy."""
    text = _read(f"tip4p-{size}.xyz")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"TIP4P {size} xyz is too short")
    match = re.search(r"Energy =\s+([-\d.]+)", lines[1])
    if match is None:
        raise ValueError(f"TIP4P {size} xyz has no energy header")
    energy_text = match.group(1)
    positions = _xyz_rows("\n".join(lines[2:]))
    if len(positions) != 3 * size:
        raise ValueError(f"TIP4P {size} xyz has {len(positions)} atoms")
    page_text = tip4p_page()[size]
    return {
        "size": size,
        "energy_text": energy_text,
        "energy": float(energy_text),
        "page_energy_text": page_text,
        "page_energy": float(page_text),
        "positions": positions,
        "page": TIP4P_PAGE_URL,
        "points": f"https://www-wales.ch.cam.ac.uk/~wales/CCD/TIP4P/xyz/TIP4P-{size}.xyz",
    }
