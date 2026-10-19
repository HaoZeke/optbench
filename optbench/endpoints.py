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


def load_endpoints(path: Path) -> dict[str, np.ndarray]:
    """Return absolute free-atom coordinates reconstructed from the deltas."""
    with np.load(path) as archive:
        kind = str(archive["format"])
        if kind != FORMAT:
            raise ValueError(f"{path} format is {kind}")
        reactant = np.array(archive["reactant"], dtype=np.float32, copy=True)
        plus = np.array(archive["plus"], dtype=np.float32, copy=True)
        minus = np.array(archive["minus"], dtype=np.float32, copy=True)
        free_index = np.array(archive["free_index"], dtype=np.int32, copy=True)
        cell = np.array(archive["cell"], dtype=np.float64, copy=True)
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


def _shared_reactant(rows: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(rows).all(axis=(1, 2))
    if not bool(finite.any()):
        return np.zeros((rows.shape[1], 3), dtype=np.float32)
    base = rows[int(np.flatnonzero(finite)[0])]
    if not np.allclose(rows[finite], base, rtol=0.0, atol=1e-6):
        return None
    return base.astype(np.float32)
