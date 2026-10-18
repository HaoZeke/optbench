"""Start geometries for a harness campaign.

``.con`` files are eOn coordinates. ``coords.<digits>`` files are plain
Cartesian blocks. Angle-axis ``.aa`` files are refused: this harness has
no TIP4P potential and no rigid-body optimizer.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

import numpy as np

from optbench.campaign import geometry_archive
from optbench.potentials import read_con, read_con_components

_COORDS = re.compile(r"coords\.(\d+)$")
_AA = re.compile(r".+\.aa$")


class GeometryRefusal(RuntimeError):
    """The archive cannot be scored with this potential."""


def _layout(problem: dict) -> str:
    sample = problem.get("sample") or {}
    stated = str(sample.get("layout") or "")
    if stated:
        return stated
    if problem.get("suite") == "lowest-eigenvector":
        return "con-mode"
    return "con"


def load_starts(
    root: Path,
    problem: dict,
    *,
    work: Path | None,
    output: Path,
    limit: int = 0,
) -> list[dict]:
    layout = _layout(problem)
    if layout == "aa":
        raise GeometryRefusal(
            f"{problem['id']} is an angle-axis TIP4P archive. "
            "This harness has no TIP4P potential, so it does not emit a hit rate."
        )
    archive = geometry_archive(root, problem)
    dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    recorded = archive.name
    if marker.is_file():
        previous = marker.read_text(encoding="utf-8").strip()
        if previous != recorded:
            raise RuntimeError(f"{dest} was extracted from {previous}, not {recorded}")
    else:
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
        marker.write_text(recorded, encoding="utf-8")
    if layout == "con-mode":
        loaded = _load_modes(dest)
    elif layout == "coords":
        loaded = _load_coords(dest)
    elif layout == "con":
        loaded = _load_cons(dest, components=problem["id"] == "globalopt/lj100-binary")
    else:
        raise GeometryRefusal(f"{problem['id']} has no loader for layout {layout}")
    expected = (problem.get("sample") or {}).get("starts")
    if limit <= 0 and expected is not None and len(loaded) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(loaded)}")
    if limit > 0:
        loaded = loaded[: int(limit)]
    return loaded


def _load_cons(dest: Path, *, components: bool) -> list[dict]:
    paths = sorted(path for path in dest.rglob("*.con") if path.is_file())
    loaded = []
    for index, path in enumerate(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        if components:
            pos, box, species = read_con_components(text)
        else:
            pos, box = read_con(text)
            species = None
        loaded.append(_row(index, path.stem, pos, box, species, None))
    return loaded


def _load_coords(dest: Path) -> list[dict]:
    found = []
    for path in dest.rglob("*"):
        if path.is_file() and _COORDS.fullmatch(path.name):
            found.append(path)
    found.sort(key=lambda path: int(_COORDS.fullmatch(path.name).group(1)))
    loaded = []
    for index, path in enumerate(found):
        values = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for token in line.split():
                values.append(float(token))
        if len(values) % 3 != 0 or not values:
            raise ValueError(f"{path} is not a Cartesian block")
        pos = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        loaded.append(_row(index, path.name, pos, None, None, None))
    return loaded


def _load_modes(dest: Path) -> list[dict]:
    cons = sorted(path for path in dest.rglob("*.con") if path.is_file())
    loaded = []
    for index, path in enumerate(cons):
        mode_path = path.with_name(path.stem + "_mode")
        if not mode_path.is_file():
            # The archive pairs ``0024.con`` with ``0024_mode``.
            alt = path.with_name(path.name.replace(".con", "_mode"))
            mode_path = alt if alt.is_file() else mode_path
        if not mode_path.is_file():
            raise FileNotFoundError(f"{path.name} has no mode file")
        pos, box = read_con(path.read_text(encoding="utf-8", errors="replace"))
        mode = _read_mode(mode_path.read_text(encoding="utf-8", errors="replace"))
        if mode.size != pos.size:
            raise ValueError(f"{mode_path.name} has {mode.size} components, coordinates have {pos.size}")
        loaded.append(_row(index, path.stem, pos, box, None, mode))
    return loaded


def _read_mode(text: str) -> np.ndarray:
    values = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.append(float(stripped.split()[0]))
    if not values:
        raise ValueError("mode file has no components")
    return np.asarray(values, dtype=np.float64)


def _row(index, name, pos, box, species, mode) -> dict:
    return {
        "index": index,
        "start": name,
        "coords": np.asarray(pos, dtype=np.float64).reshape(-1),
        "box": None if box is None else np.asarray(box, dtype=np.float64).reshape(-1),
        "species": None if species is None else np.asarray(species, dtype=np.int64).reshape(-1),
        "mode": mode,
    }
