"""Price search-time analytic curvature as force calls.

``problems/`` and ``docs/v2.md`` do not define a separate analytic-Hessian
score. A force call is an energy-and-gradient entry. A finite-difference
Hessian-vector product costs one extra gradient: SciPy's ``approx_fhess_p``
evaluates the gradient at one displaced point, and the gradient at the
current point is already a force call when the harness cache hits. An
analytic product is charged that one call.

A full analytic Hessian is the same product along each coordinate, so it
costs the dimension. ``numerical_hessian`` is a central difference and costs
two gradients per coordinate; that builder is the index check, which stays
on ``verify_calls`` and is not a search call.
"""

from __future__ import annotations


def product_cost() -> int:
    """Force calls charged for one analytic Hessian-vector product."""
    return 1


def full_hessian_cost(dimension: int) -> int:
    """Force calls charged for one full analytic Hessian."""
    size = int(dimension)
    if size < 1:
        raise ValueError("a Hessian needs a positive dimension")
    return size


def reprice_record(record: dict, *, unit_cost: int, cap: int) -> dict:
    """Add analytic curvature onto one stored start.

    ``unit_cost`` is 1 for a product and the dimension for a full matrix.
    ``hessian_calls`` on the stored record is the number of those units.
    A second call uses ``gradient_calls`` and does not charge twice.
    A total at or above the cap is a cap failure, unless the start already
    crashed. The component counts stay on the record.
    """
    row = dict(record)
    if "gradient_calls" in row:
        gradient = int(row["gradient_calls"])
    else:
        gradient = int(row["force_calls"])
    units = int(row.get("hessian_calls") or 0)
    charge = units * int(unit_cost)
    total = gradient + charge
    row["gradient_calls"] = gradient
    row["hessian_calls"] = units
    row["hessian_force_calls"] = charge
    row["force_calls"] = total
    reasons = [str(item) for item in (row.get("reasons") or [])]
    if total >= int(cap) and row.get("failure") != "crash" and "crash" not in reasons:
        row["success"] = False
        row["converged"] = False
        row["failure"] = "cap"
        row["reasons"] = ["cap", *[item for item in reasons if item != "cap"]]
    return row
