"""eOn r2205 saddle-search link test.

The archived Pt-island fraction connected comes from
``archive/entries/saddle-search/pt-island/eon-dimer``. That row is Eon
r2205, dated 15 Jul 2014. ``done.py`` counts a search as not connected
when the result token contains ``Connected``, which is how it matches
the label ``Not Connected``.

That label is termination reason 6 in ``eon/akmcstate.py``. The client
sets it in ``ProcessSearchJob::doProcessSearch`` (public tree
``8dc9bf61``, 2014-07-15, the same day as the archived row). After the
two endpoint minimizations the code swaps the endpoints when only the
second matches the reactant, then rejects the search when the first
does not match or when both match:

```
initial->compare(min1)
initial->compare(min2)
```

``Matter::compare`` defaults ``indistinguishable`` to false
(``Matter.h``). ``check_rotation`` defaults to false, so the call is

```
parameters->distanceDifference > perAtomNorm(*matter)
```

``Parameters.cpp`` sets ``distanceDifference`` to 0.1, and the archived
``config.ini`` does not override ``[Structure Comparison]``.
``perAtomNorm`` is the maximum per-atom distance after ``Matter::pbc``,
which folds every positive cell axis with

```
fmod(fmod(fraction, 1) + 1.5, 1) - 0.5
```

Two minima are the same when that maximum is strictly below 0.1 Å.
A search is connected when exactly one endpoint is the same minimum.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DISTANCE_DIFFERENCE = 0.1


def link_tolerance(problem: dict | None = None) -> float:
    """Threshold from the problem file, otherwise the r2205 default."""
    block = {} if problem is None else (problem.get("connection") or {})
    if "distance_difference" in block:
        return float(block["distance_difference"])
    return DISTANCE_DIFFERENCE


def _lengths(cell) -> np.ndarray | None:
    """Orthorhombic edge lengths.

    A length-3 box is used as stored. A 3×3 cell, or nine numbers in
    row-major order, uses the diagonal. ``Matter::pbc`` multiplies by
    that cell after folding the fractional coordinates.
    """
    if cell is None:
        return None
    array = np.asarray(cell, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] >= 3 and array.shape[1] >= 3:
        return np.array([array[0, 0], array[1, 1], array[2, 2]], dtype=np.float64)
    flat = array.reshape(-1)
    if flat.size == 9:
        matrix = flat.reshape(3, 3)
        return np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2]], dtype=np.float64)
    if flat.size < 3:
        return None
    return flat[:3]


def minimum_image(delta: np.ndarray, cell) -> np.ndarray:
    """``Matter::pbc`` for an orthorhombic cell. Every positive axis folds."""
    out = np.asarray(delta, dtype=np.float64, copy=True)
    lengths = _lengths(cell)
    if lengths is None:
        return out
    for axis, length in enumerate(lengths):
        if float(length) <= 0.0:
            continue
        fraction = out[..., axis] / float(length)
        folded = np.fmod(np.fmod(fraction, 1.0) + 1.5, 1.0) - 0.5
        out[..., axis] = folded * float(length)
    return out


def per_atom_max(moved, reference, cell) -> float:
    """``Matter::perAtomNorm``: maximum per-atom minimum-image distance."""
    left = np.asarray(moved, dtype=np.float64).reshape(-1, 3)
    right = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if left.shape != right.shape:
        raise ValueError(f"endpoint has shape {left.shape}, reactant has {right.shape}")
    if left.size == 0:
        return 0.0
    delta = minimum_image(left - right, cell)
    return float(np.max(np.linalg.norm(delta, axis=1)))


def same_minimum(moved, reference, cell, tolerance: float = DISTANCE_DIFFERENCE) -> bool:
    """True when ``distance_difference`` is strictly above ``perAtomNorm``."""
    return float(tolerance) > per_atom_max(moved, reference, cell)


def connected_endpoints(reactant, plus, minus, cell, tolerance: float = DISTANCE_DIFFERENCE) -> bool:
    """Exactly one minimized endpoint is the reactant minimum."""
    return same_minimum(plus, reactant, cell, tolerance) ^ same_minimum(minus, reactant, cell, tolerance)


def relink(record: dict, cell, tolerance: float = DISTANCE_DIFFERENCE) -> str:
    """Recompute the link from stored endpoint coordinates.

    Returns ``missing`` when ``minimum``, ``endpoint_plus``, and
    ``endpoint_minus`` are absent. A cap, crash, or wrong-stationary
    record is left alone. Otherwise the ``not_connected`` reason follows
    the r2205 test, and success still requires an empty reason list.
    """
    keys = ("minimum", "endpoint_plus", "endpoint_minus")
    if any(record.get(key) is None for key in keys):
        return "missing"
    if record.get("failure") in {"cap", "crash", "wrong_stationary"}:
        return "unchanged"
    reasons = [reason for reason in (record.get("reasons") or []) if reason != "not_connected"]
    linked = connected_endpoints(
        record["minimum"],
        record["endpoint_plus"],
        record["endpoint_minus"],
        cell,
        tolerance,
    )
    if not linked:
        reasons.append("not_connected")
    record["reasons"] = reasons
    record["per_atom_max_plus"] = per_atom_max(record["endpoint_plus"], record["minimum"], cell)
    record["per_atom_max_minus"] = per_atom_max(record["endpoint_minus"], record["minimum"], cell)
    record["success"] = not reasons
    record["failure"] = None if record["success"] else record.get("failure") or "divergence"
    return "updated"


def geometry_status(record: dict[str, Any]) -> str:
    """Which endpoint coordinates a start record actually carries."""
    keys = [key for key in ("minimum", "endpoint_plus", "endpoint_minus") if record.get(key) is not None]
    if len(keys) == 3:
        return "stored"
    scalars = [key for key in ("rmsd_plus", "rmsd_minus", "per_atom_max_plus", "per_atom_max_minus") if key in record]
    if scalars:
        return "scalars:" + ",".join(scalars)
    return "absent"
