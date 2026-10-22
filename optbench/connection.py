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

Uniqueness is the server test in ``AKMCState.find_repeat``
(``eon/akmcstate.py``). A later saddle is a repeat when the absolute
barrier difference is not greater than ``energy_difference`` (default
0.01 eV, ``eon/config.yaml``) and ``atoms.match`` accepts the pair.
``find_repeat`` passes ``indistinguishable=False``. With
``check_rotation`` false, ``match`` is

```
max(sum(pbc(diff) ** 2, axis=1)) < distance_difference ** 2
```

which is the same strict per-atom maximum as ``Matter::compare``.
The comparison is the saddle geometry. A stored campaign that kept
only the endpoint minima uses the product minimum, the endpoint that
is not the reactant.

``done.py`` counts ``unique_saddles`` as result tokens containing
``good``. It increments ``not_connected`` when the token contains
``Connected``. ``fraction_connected`` is

```
(total_searches - not_connected) / total_searches
```

A repeat and any other failure that is not ``Not Connected`` stay in
that fraction. The goal counter is the number of good saddles whose
barrier is strictly below 1.5 eV.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DISTANCE_DIFFERENCE = 0.1
ENERGY_DIFFERENCE = 0.01
BARRIER_MAX = 1.5


def result_is_not_connected(result: str) -> bool:
    """``done.py`` matches ``Not Connected`` by the substring ``Connected``.

    ``ProcessSearchJob`` writes that label when ``Matter::compare`` rejects
    the search. The comparison is the r2205 rule: ``distance_difference``
    strictly above ``perAtomNorm``, with ``indistinguishable`` left false.
    """
    return "Connected" in result


def done_fraction(searches: int, not_connected: int) -> float:
    """``(total_searches - not_connected) / total_searches`` from ``done.py``."""
    if searches <= 0:
        raise ValueError("searches must be positive")
    if not_connected < 0 or not_connected > searches:
        raise ValueError("not_connected is outside the search count")
    return (searches - not_connected) / searches


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
    out = np.array(delta, dtype=np.float64, copy=True)
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


def same_process(
    moved,
    reference,
    cell,
    barrier: float,
    other_barrier: float,
    tolerance: float = DISTANCE_DIFFERENCE,
    energy_difference: float = ENERGY_DIFFERENCE,
) -> bool:
    """``AKMCState.find_repeat`` for one earlier good saddle.

    The energy gate is ``abs(barrier difference) > energy_difference``,
    so a difference of exactly 0.01 eV still compares the geometries.
    The geometry gate is ``Matter::compare`` with atoms distinguished
    by index.
    """
    if abs(float(barrier) - float(other_barrier)) > float(energy_difference):
        return False
    return same_minimum(moved, reference, cell, tolerance)


def product_endpoint(reactant, plus, minus, cell, tolerance: float = DISTANCE_DIFFERENCE):
    """The minimized endpoint that is not the reactant.

    ``None`` when the search is not connected. eOn compares
    ``saddle.con``. This is the stand-in when that file was not stored.
    """
    plus_same = same_minimum(plus, reactant, cell, tolerance)
    minus_same = same_minimum(minus, reactant, cell, tolerance)
    if plus_same == minus_same:
        return None
    chosen = minus if plus_same else plus
    return np.array(np.asarray(chosen, dtype=np.float64).reshape(-1, 3), copy=True)


class ProcessCatalog:
    """Walk searches in order, as the 2014 AKMC server did.

    ``unique`` is the number of good saddles, including barriers at or
    above the goal. ``qualifying`` is good saddles strictly inside
    ``(0, barrier_max)``. ``fraction_connected`` leaves out only
    searches judged not connected. A cap, crash, or search that never
    minimized its endpoints is judged false and stays in the fraction.
    """

    def __init__(
        self,
        cell,
        *,
        tolerance: float = DISTANCE_DIFFERENCE,
        energy_difference: float = ENERGY_DIFFERENCE,
        barrier_max: float = BARRIER_MAX,
    ):
        self.cell = cell
        self.tolerance = float(tolerance)
        self.energy_difference = float(energy_difference)
        self.barrier_max = float(barrier_max)
        self._goods: list[tuple[np.ndarray, float]] = []
        self.searches = 0
        self.not_connected = 0
        self.qualifying = 0

    @property
    def unique(self) -> int:
        return len(self._goods)

    @property
    def fraction_connected(self) -> float:
        if not self.searches:
            return 0.0
        return done_fraction(self.searches, self.not_connected)

    def observe(
        self,
        *,
        judged: bool,
        linked: bool = False,
        structure=None,
        barrier: float | None = None,
    ) -> dict[str, Any]:
        """Record one search and return ``reasons``, ``good``, and ``success``."""
        self.searches += 1
        reasons: list[str] = []
        if not judged:
            return {"reasons": reasons, "good": False, "success": False}
        if not linked:
            self.not_connected += 1
            reasons.append("not_connected")
            return {"reasons": reasons, "good": False, "success": False}
        height = _barrier(barrier)
        registered = False
        if structure is not None and height is not None and height > 0.0:
            moved = np.array(np.asarray(structure, dtype=np.float64).reshape(-1, 3), copy=True)
            repeat = any(
                same_process(
                    moved,
                    old,
                    self.cell,
                    height,
                    old_height,
                    self.tolerance,
                    self.energy_difference,
                )
                for old, old_height in self._goods
            )
            if repeat:
                reasons.append("duplicate")
            else:
                self._goods.append((moved, height))
                registered = True
        if height is None or height <= 0.0 or height >= self.barrier_max:
            reasons.append("barrier")
        success = registered and height is not None and 0.0 < height < self.barrier_max
        if success:
            self.qualifying += 1
        return {"reasons": reasons, "good": registered, "success": success}


def _barrier(value) -> float | None:
    if value is None:
        return None
    try:
        height = float(value)
    except (TypeError, ValueError):
        return None
    if height != height:  # NaN
        return None
    return height


def rescore_endpoints(
    records: list[dict],
    reactant,
    plus,
    minus,
    cell,
    *,
    tolerance: float = DISTANCE_DIFFERENCE,
    energy_difference: float = ENERGY_DIFFERENCE,
    barrier_max: float = BARRIER_MAX,
) -> ProcessCatalog:
    """Rejudge stored searches from endpoint minima.

    Rows line up with ``reactant``, ``plus``, and ``minus``. A row
    whose endpoints are missing is not ``Not Connected``. A connected
    row is unique when its product minimum is not a ``find_repeat`` of
    an earlier good product. Saddle coordinates are not in the archive,
    so the product is the geometry ``find_repeat`` would have applied
    to ``saddle.con``.
    """
    from .failures import classify_failure

    catalog = ProcessCatalog(
        cell,
        tolerance=tolerance,
        energy_difference=energy_difference,
        barrier_max=barrier_max,
    )
    reactant_rows = np.asarray(reactant, dtype=np.float64)
    plus_rows = np.asarray(plus, dtype=np.float64)
    minus_rows = np.asarray(minus, dtype=np.float64)
    if len(records) != len(reactant_rows):
        raise ValueError(f"{len(records)} searches and {len(reactant_rows)} endpoint rows")
    for index, record in enumerate(records):
        kept = [reason for reason in (record.get("reasons") or []) if reason not in {"not_connected", "duplicate", "barrier"}]
        pair_plus = plus_rows[index]
        pair_minus = minus_rows[index]
        finite = bool(np.isfinite(pair_plus).all() and np.isfinite(pair_minus).all())
        if not finite:
            catalog.observe(judged=False)
            continue
        origin = reactant_rows[index]
        linked = connected_endpoints(origin, pair_plus, pair_minus, cell, tolerance)
        structure = product_endpoint(origin, pair_plus, pair_minus, cell, tolerance) if linked else None
        decision = catalog.observe(
            judged=True,
            linked=linked,
            structure=structure,
            barrier=record.get("barrier"),
        )
        reasons = list(kept)
        for reason in decision["reasons"]:
            if reason not in reasons:
                reasons.append(reason)
        record["reasons"] = reasons
        record["per_atom_max_plus"] = per_atom_max(pair_plus, origin, cell)
        record["per_atom_max_minus"] = per_atom_max(pair_minus, origin, cell)
        record["success"] = bool(decision["success"] and not kept)
        record["failure"] = None if record["success"] else classify_failure(reasons)
    return catalog


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
