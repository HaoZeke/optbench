"""Energy window for one global-optimisation start.

The published minimum is the reference energy on the problem. A start
succeeds when that energy is finite and at or below the minimum plus
``energy_above_minimum``, and the harness call count is below the cap.
A lower energy still succeeds. The failure class follows
``optbench.failures``: divergence, then the cap, then a high energy.
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
    """Classify one global-optimisation start."""
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
