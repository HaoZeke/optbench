"""One primary failure class for a start.

The order is fixed, so a start lands in one bin:

1. ``crash`` — the optimizer raised, or the record says so.
2. ``divergence`` — a non-finite result, an explicit blow-up, or a
   return that is still above the force stop and under the cap.
3. ``cap`` — the search used the whole force-call budget and did not
   blow up.
4. ``wrong_stationary`` — the force is at the stop, and the Hessian
   index, the energy, or the Kabsch RMSD rejects the point.

A success has no class. Every failure stays in the PAR-2 mean; the
class says why it is there.
"""

from __future__ import annotations

FAILURES = ("crash", "divergence", "cap", "wrong_stationary")


def classify_outcome(
    *,
    energy: float | None = None,
    geometry=None,
    force: float | None = None,
    calls: float | None = None,
    cap: int | None = None,
    force_tol: float | None = None,
    hessian_index: int | None = None,
    expected_index: int | None = None,
    process_failed: bool = False,
    force_above: bool | None = None,
    stationary_rejected: bool = False,
    unmet: bool = False,
) -> str | None:
    """One failure class from a finished optimizer point.

    The order matches :func:`classify_failure`. A saddle catalog that has
    already applied the link test overwrites this class.
    """
    reasons: list[str] = []
    if process_failed:
        reasons.append("crash")
    if energy is not None:
        try:
            if float(energy) != float(energy):
                reasons.append("divergence")
        except (TypeError, ValueError):
            reasons.append("divergence")
    if geometry is not None:
        try:
            import numpy as np

            if not bool(np.all(np.isfinite(np.asarray(geometry, dtype=np.float64)))):
                reasons.append("divergence")
        except (TypeError, ValueError):
            reasons.append("divergence")
    if cap is not None and calls is not None and float(calls) >= float(cap):
        reasons.append("cap")
    above = force_above
    if above is None and force is not None and force_tol is not None:
        above = float(force) > float(force_tol)
    if above:
        reasons.append("force_norm")
    if (
        expected_index is not None
        and hessian_index is not None
        and int(hessian_index) != int(expected_index)
    ):
        reasons.append("hessian_index")
    if stationary_rejected or unmet:
        reasons.append("divergence")
    return classify_failure(reasons)


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
    if any(name in reasons for name in ("hessian_index", "energy", "rmsd")):
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
