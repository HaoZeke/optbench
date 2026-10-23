"""One primary failure class for a start.

The order is fixed, so a start lands in one bin:

1. ``crash`` — the optimizer raised, or the record says so.
2. ``divergence`` — a non-finite result, an explicit blow-up, or a
   return that is still above the force stop and under the cap.
3. ``cap`` — the search used the whole force-call budget and did not
   blow up.
4. ``wrong_stationary`` — the force is at the stop, and the Hessian
   index, the energy, or the Kabsch RMSD rejects the point. The
   harness records that rejection as ``hessian_index``, ``energy``,
   ``rmsd``, ``not_connected``, ``duplicate``, or ``barrier``.

A success has no class. Every failure stays in the PAR-2 mean; the
class says why it is there.
"""

from __future__ import annotations

FAILURES = ("crash", "divergence", "cap", "wrong_stationary")

# Connection, novelty, and barrier are RMSD and energy rejections.
# They are not a blow-up and not a point still above the force stop.
_STATIONARY = ("hessian_index", "energy", "rmsd", "not_connected", "duplicate", "barrier")


def classify_failure(reasons: list[str]) -> str | None:
    """Primary class, or ``None`` when ``reasons`` is empty."""
    if not reasons:
        return None
    if "crash" in reasons:
        return "crash"
    if "divergence" in reasons:
        return "divergence"
    if "cap" in reasons:
        return "cap"
    if "force_norm" in reasons:
        return "divergence"
    if any(name in reasons for name in _STATIONARY):
        return "wrong_stationary"
    return "divergence"


def failure_of(record: dict, cap: int | None) -> str | None:
    """Class stored on the record, or inferred when an older row omits it."""
    if record.get("success"):
        label = record.get("failure")
        if label not in (None, "", "none"):
            raise ValueError("a success cannot carry a failure class")
        return None
    label = record.get("failure")
    if label in FAILURES:
        return str(label)
    reasons = [str(reason) for reason in record.get("reasons") or []]
    if reasons:
        return classify_failure(reasons)
    calls = float(record["force_calls"])
    if cap is not None and calls >= cap:
        return "cap"
    return "divergence"


def failure_tally(records: list[dict], cap: int | None) -> dict[str, int]:
    """Count each class. The four counts plus nothing else equal ``nfailed``."""
    counts = {name: 0 for name in FAILURES}
    for record in records:
        label = failure_of(record, cap)
        if label is None:
            continue
        counts[label] += 1
    return counts
