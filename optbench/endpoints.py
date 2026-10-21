"""Free-atom endpoint minima for a saddle-search campaign.

One compressed ``<stem>.endpoints.npz`` sits next to the starts file.
``reactant`` is the free-atom geometry. ``plus`` and ``minus`` are
float32 deltas against that reactant, in the order of ``free_index``.
A search that stopped before the two minimizations has NaN deltas.

When every finite row shares one reactant, that geometry is stored
once. A 20_000-search Pt-island log is still tens of megabytes,
because the deltas themselves are the bulk, so those files are
git-LFS pointers rather than loose blobs. The npz is the rescoring
geometry. It is not copied into the site.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FORMAT = "reactant-delta-f32"
MAGIC = b"OBEP"
VERSION = 1
FORMAT_ABSOLUTE = 1
FORMAT_DELTA = 2
STATUS_STORED = "stored"
STATUS_NOT_MINIMIZED = "not_minimized"


def save_endpoints(path: Path, reactant, plus, minus, free_index, cell=None) -> Path:
    """Pack absolute coordinates as compressed free-atom float32 deltas.

    ``reactant``, ``plus``, and ``minus`` share a layout: ``(n, atom, 3)``,
    ``(atom, 3)``, or flat ``(n, atom * 3)``. ``free_index`` selects the
    moving atoms. A missing minimization is a row of NaN.
    """
    path = Path(path)
    index = np.asarray(free_index, dtype=np.int32).reshape(-1)
    if index.size == 0:
        raise ValueError("free_index is empty")
    reactant_rows = _rows(reactant, index)
    plus_rows = _rows(plus, index)
    minus_rows = _rows(minus, index)
    count = max(reactant_rows.shape[0], plus_rows.shape[0], minus_rows.shape[0])
    reactant_rows = _broadcast(reactant_rows, count)
    plus_rows = _broadcast(plus_rows, count)
    minus_rows = _broadcast(minus_rows, count)
    shared = _shared_reactant(reactant_rows)
    if shared is None:
        stored_reactant = reactant_rows.astype(np.float32)
        base = reactant_rows
    else:
        stored_reactant = shared
        base = shared.astype(np.float64)
    plus_delta = (plus_rows - base).astype(np.float32)
    minus_delta = (minus_rows - base).astype(np.float32)
    box = np.zeros(3, dtype=np.float64) if cell is None else np.asarray(cell, dtype=np.float64).reshape(-1)[:3]
    if box.size != 3:
        raise ValueError("cell needs three lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".writing.npz")
    np.savez_compressed(
        temporary,
        reactant=stored_reactant,
        plus=plus_delta,
        minus=minus_delta,
        free_index=index,
        cell=box,
        format=np.asarray(FORMAT),
    )
    temporary.replace(path)
    return path


def _integer_format(archive) -> int | None:
    if "format" not in archive.files:
        return None
    value = archive["format"]
    if getattr(value, "dtype", None) is not None and value.dtype.kind in {"i", "u"}:
        return int(value)
    return None


def _from_delta(reactant: np.ndarray, delta) -> np.ndarray:
    shift = np.asarray(delta, dtype=np.float64)
    expanded = reactant + shift
    missing = ~np.isfinite(shift)
    if missing.any():
        expanded = np.array(expanded, copy=True)
        expanded[missing] = np.nan
    return expanded


def load_endpoints(path: Path):
    """Absolute free-atom coordinates.

    A campaign archive (integer format, or ``plus_delta``) returns
    ``(reactant, plus, minus, cell, free_index)`` as float64. The packed
    delta file written by ``save_endpoints`` returns a dict of float32
    arrays.
    """
    with np.load(path) as packed:
        names = set(packed.files)
        code = _integer_format(packed)
        if "plus_delta" in names or code in {FORMAT_ABSOLUTE, FORMAT_DELTA}:
            reactant = np.asarray(packed["reactant"], dtype=np.float64)
            cell = np.asarray(packed["cell"], dtype=np.float64)
            free_index = np.asarray(packed["free_index"])
            if "plus_delta" in names:
                plus = _from_delta(reactant, packed["plus_delta"])
                minus = _from_delta(reactant, packed["minus_delta"])
            else:
                plus = np.asarray(packed["plus"], dtype=np.float64)
                minus = np.asarray(packed["minus"], dtype=np.float64)
            return reactant, plus, minus, cell, free_index
        kind = str(packed["format"])
        if kind != FORMAT:
            raise ValueError(f"{path} format is {kind}")
        reactant = np.array(packed["reactant"], dtype=np.float32, copy=True)
        plus = np.array(packed["plus"], dtype=np.float32, copy=True)
        minus = np.array(packed["minus"], dtype=np.float32, copy=True)
        free_index = np.array(packed["free_index"], dtype=np.int32, copy=True)
        cell = np.array(packed["cell"], dtype=np.float64, copy=True)
    if reactant.ndim == 2:
        reactant_rows = np.broadcast_to(reactant, plus.shape).copy()
        absolute_plus = reactant + plus
        absolute_minus = reactant + minus
    else:
        reactant_rows = reactant
        absolute_plus = reactant + plus
        absolute_minus = reactant + minus
    return {
        "reactant": reactant_rows,
        "plus": absolute_plus,
        "minus": absolute_minus,
        "free_index": free_index,
        "cell": cell,
    }


def _rows(value, free_index: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        if array.size % 3 != 0:
            raise ValueError("a flat coordinate vector is not a multiple of 3")
        array = array.reshape(1, array.size // 3, 3)
    elif array.ndim == 2 and array.shape[-1] == 3:
        array = array.reshape(1, array.shape[0], 3)
    elif array.ndim == 2:
        if array.shape[-1] % 3 != 0:
            raise ValueError("a flat coordinate row is not a multiple of 3")
        array = array.reshape(array.shape[0], array.shape[-1] // 3, 3)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"coordinates have shape {array.shape}")
    if array.shape[1] == free_index.size:
        return array
    if int(free_index.max()) >= array.shape[1] or int(free_index.min()) < 0:
        raise ValueError("free_index does not fit the coordinate array")
    return array[:, free_index, :]


def _broadcast(rows: np.ndarray, count: int) -> np.ndarray:
    if rows.shape[0] == count:
        return rows
    if rows.shape[0] == 1:
        return np.repeat(rows, count, axis=0)
    raise ValueError(f"expected {count} endpoint rows, got {rows.shape[0]}")


def _git_blob_size(root: Path, path: Path) -> int | None:
    rel = path.relative_to(root).as_posix()
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "cat-file", "-s", f":{rel}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out.strip())


def _lfs_tracked(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "check-attr", "filter", "--", rel],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return out.strip().endswith("filter: lfs")


def large_endpoint_files(root: Path, limit: int = 1_000_000) -> list[str]:
    """Endpoint archives stored as git blobs instead of an LFS pointer.

    The committed size is the blob in the index. A path matched by
    git-LFS is allowed while it is still untracked, because ``git add``
    stores the pointer. A loose blob over ``limit`` fails.
    """
    errors = []
    campaign = root / "results" / "campaign"
    if not campaign.is_dir():
        return errors
    for path in sorted(campaign.rglob("*.npz")):
        if "endpoint" not in path.name:
            continue
        committed = _git_blob_size(root, path)
        if committed is not None:
            size = committed
        elif _lfs_tracked(root, path):
            continue
        else:
            size = path.stat().st_size
        if size > limit:
            errors.append(f"{path.relative_to(root)} is {size} bytes")
    return errors


def _shared_reactant(rows: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(rows).all(axis=(1, 2))
    if not bool(finite.any()):
        return np.zeros((rows.shape[1], 3), dtype=np.float32)
    base = rows[int(np.flatnonzero(finite)[0])]
    if not np.allclose(rows[finite], base, rtol=0.0, atol=1e-6):
        return None
    return base.astype(np.float32)


def stamp_endpoint(record: dict, archive_name: str, index: int, status: str) -> None:
    """Point one start at its row in the endpoint archive."""
    record["endpoints"] = archive_name
    record["endpoint_index"] = int(index)
    record["endpoint_status"] = status


class EndpointArchive:
    """Append-only endpoint log, packed into an npz when the run finishes.

    The ``.bin`` beside the npz is the crash log. ``save`` packs the npz
    and leaves the log in place so a crash while the starts file is
    written can still resume. ``drop_log`` removes it afterwards.
    ``load`` restores a resume from that log.
    """

    def __init__(self, path: Path, n_free: int):
        self.path = Path(path)
        self.bin_path = self.path.with_name(self.path.stem + ".bin")
        self.n_free = int(n_free)
        self.cell = np.zeros(3, dtype=np.float64)
        self.free_index = np.zeros(self.n_free, dtype=np.int32)
        self.rows: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = []

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def count(self) -> int:
        return len(self.rows)

    def reset(self, cell, free_index) -> None:
        """Start a new log. ``free_index`` is the free-atom order."""
        self.cell = np.asarray(cell, dtype=np.float64).reshape(-1)[:3].copy()
        self.free_index = np.asarray(free_index, dtype=np.int32).reshape(-1).copy()
        if self.free_index.size != self.n_free:
            raise ValueError(f"free_index has {self.free_index.size} atoms, the archive has {self.n_free}")
        self.rows.clear()
        self.bin_path.unlink(missing_ok=True)
        self._write_header()

    def add(self, reactant, plus, minus) -> str:
        """Append one search. ``plus`` and ``minus`` are ``None`` when it did not minimize."""
        reactant_row = self._row(reactant)
        stored = plus is not None and minus is not None
        plus_row = self._row(plus) if stored else None
        minus_row = self._row(minus) if stored else None
        self.rows.append((reactant_row, plus_row, minus_row))
        self._append_row(reactant_row, plus_row, minus_row)
        return STATUS_STORED if stored else STATUS_NOT_MINIMIZED

    def load(self) -> None:
        """Read the crash log back into memory."""
        if not self.bin_path.is_file():
            raise FileNotFoundError(f"{self.bin_path} is not beside {self.path.name}")
        blob = self.bin_path.read_bytes()
        self.cell, self.free_index, self.rows = _parse_bin(blob, self.n_free)

    def truncate(self, count: int) -> None:
        """Drop log rows past ``count`` and rewrite the crash log."""
        if count < 0 or count > len(self.rows):
            raise ValueError(f"cannot truncate {len(self.rows)} rows to {count}")
        self.rows = self.rows[:count]
        self.bin_path.unlink(missing_ok=True)
        self._write_header()
        for reactant, plus, minus in self.rows:
            self._append_row(reactant, plus, minus)

    def save(self) -> None:
        """Pack the log into a format-2 npz. The crash log stays until ``drop_log``."""
        count = len(self.rows)
        reactant = np.full((count, self.n_free, 3), np.nan, dtype=np.float32)
        plus_delta = np.full((count, self.n_free, 3), np.nan, dtype=np.float16)
        minus_delta = np.full((count, self.n_free, 3), np.nan, dtype=np.float16)
        for index, (reactant_row, plus_row, minus_row) in enumerate(self.rows):
            reactant[index] = reactant_row
            if plus_row is not None and minus_row is not None:
                plus_delta[index] = np.asarray(plus_row, dtype=np.float32) - reactant[index]
                minus_delta[index] = np.asarray(minus_row, dtype=np.float32) - reactant[index]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".writing.npz")
        np.savez_compressed(
            temporary,
            format=np.int32(FORMAT_DELTA),
            reactant=reactant,
            plus_delta=plus_delta,
            minus_delta=minus_delta,
            cell=self.cell.astype(np.float64, copy=False),
            free_index=self.free_index.astype(np.int32, copy=False),
        )
        temporary.replace(self.path)

    def drop_log(self) -> None:
        """Remove the crash log after the starts file points at the npz."""
        self.bin_path.unlink(missing_ok=True)

    def _row(self, value) -> np.ndarray:
        row = np.asarray(value, dtype=np.float32).reshape(self.n_free, 3)
        if not np.isfinite(row).all():
            raise ValueError("an endpoint coordinate is not finite")
        return row

    def _write_header(self) -> None:
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        with self.bin_path.open("wb") as handle:
            handle.write(MAGIC)
            handle.write(np.uint32(VERSION).tobytes())
            handle.write(np.uint32(self.n_free).tobytes())
            handle.write(self.cell.astype(np.float64).tobytes())
            handle.write(np.uint32(self.free_index.size).tobytes())
            handle.write(self.free_index.astype(np.int32).tobytes())

    def _append_row(self, reactant: np.ndarray, plus: np.ndarray | None, minus: np.ndarray | None) -> None:
        stored = plus is not None and minus is not None
        with self.bin_path.open("ab") as handle:
            handle.write(np.uint8(1 if stored else 0).tobytes())
            handle.write(reactant.astype(np.float32).tobytes())
            if stored:
                handle.write(plus.astype(np.float32).tobytes())
                handle.write(minus.astype(np.float32).tobytes())
            handle.flush()


def _parse_bin(blob: bytes, n_free: int):
    view = memoryview(blob)
    offset = 0

    def take(size: int) -> memoryview:
        nonlocal offset
        piece = view[offset : offset + size]
        if len(piece) != size:
            raise ValueError("endpoint log ended early")
        offset += size
        return piece

    magic = bytes(take(4))
    if magic != MAGIC:
        raise ValueError("endpoint log has a bad header")
    version = int(np.frombuffer(take(4), dtype=np.uint32)[0])
    if version != VERSION:
        raise ValueError(f"endpoint log version is {version}")
    stored_free = int(np.frombuffer(take(4), dtype=np.uint32)[0])
    if stored_free != n_free:
        raise ValueError(f"endpoint log has {stored_free} free atoms, the run has {n_free}")
    cell = np.frombuffer(take(24), dtype=np.float64).copy()
    index_count = int(np.frombuffer(take(4), dtype=np.uint32)[0])
    free_index = np.frombuffer(take(4 * index_count), dtype=np.int32).copy()
    row_bytes = 4 * n_free * 3
    rows = []
    while offset < len(blob):
        status = int(np.frombuffer(take(1), dtype=np.uint8)[0])
        reactant = np.frombuffer(take(row_bytes), dtype=np.float32).reshape(n_free, 3).copy()
        plus = minus = None
        if status == 1:
            plus = np.frombuffer(take(row_bytes), dtype=np.float32).reshape(n_free, 3).copy()
            minus = np.frombuffer(take(row_bytes), dtype=np.float32).reshape(n_free, 3).copy()
        elif status != 0:
            raise ValueError(f"endpoint log status {status}")
        rows.append((reactant, plus, minus))
    return cell, free_index, rows


def stored_cell(cell) -> np.ndarray:
    """Three orthorhombic lengths stored beside the endpoint coordinates.

    Zeros mean the geometry has no positive cell.
    """
    from .connection import _lengths

    lengths = _lengths(cell)
    if lengths is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(lengths, dtype=np.float64).copy()


def align_resume(archive: EndpointArchive, expected: int) -> None:
    """Load the crash log and drop a row that missed the checkpoint line.

    ``expected`` is the number of searches already recorded. One extra
    log row is the search that appended coordinates and then stopped
    before that line. Any other length is a different run.
    """
    if expected < 0:
        raise RuntimeError(f"{archive.name} cannot resume {expected} searches")
    if not archive.bin_path.is_file():
        raise RuntimeError(f"{archive.bin_path.name} is missing; {expected} searches cannot resume")
    archive.load()
    if archive.count == expected + 1:
        archive.truncate(expected)
    elif archive.count != expected:
        raise RuntimeError(
            f"{archive.name} has {archive.count} endpoint rows and the resume has {expected}"
        )
