"""Energy success rule for a global-optimisation start.

The published minimum is the Cambridge Cluster Database value stored on the
problem. A start succeeds when the energy is finite and at or below that
minimum plus ``energy_above_minimum``, and the harness force-call count is
below the cap. A lower energy still succeeds. The class of a failure follows
``optbench.failures``: divergence, then cap, then a high energy.
"""

from __future__ import annotations

import math

from .failures import classify_failure


def reference_energy(problem: dict) -> float:
    reference = problem.get("reference") or {}
    if reference.get("energy") in (None, ""):
        raise ValueError(f"{problem.get('id', 'problem')} has no reference energy")
    return float(reference["energy"])


def energy_tolerance(problem: dict) -> float:
    convergence = problem.get("convergence") or {}
    if convergence.get("energy_above_minimum") in (None, ""):
        raise ValueError(f"{problem.get('id', 'problem')} has no energy_above_minimum")
    return float(convergence["energy_above_minimum"])


def force_cap(problem: dict) -> int:
    convergence = problem.get("convergence") or {}
    if convergence.get("max_force_calls") in (None, ""):
        raise ValueError(f"{problem.get('id', 'problem')} has no max_force_calls")
    return int(convergence["max_force_calls"])


def judge_globalopt(
    energy: float | None,
    force_calls: float,
    *,
    reference: float,
    tolerance: float,
    cap: int,
) -> dict:
    """Classify one global-optimisation start.

    ``energy`` may be missing when the search stopped before a finite value
    was returned. A call count at or above ``cap`` is the cap class when the
    energy is finite.
    """
    reasons: list[str] = []
    finite = energy is not None and math.isfinite(float(energy))
    if not finite:
        reasons.append("divergence")
    elif float(energy) > float(reference) + float(tolerance):
        reasons.append("energy")
    if force_calls >= cap:
        reasons.append("cap")
    failure = classify_failure(reasons)
    return {
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": float(energy) if finite else None,
        "force_calls": float(force_calls),
    }
