"""Repeated starts with a randomised displacement or dimer orientation.

A displacement is an independent normal shift of each coordinate, with
the given standard deviation. A dimer orientation is a normal vector
normalised to unit length. ``seed + index`` is the ``random.Random``
stream of start ``index``, so the sample can be recomputed.

The score of the sample is the v2 score of every start, including the
ones that fail.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from .v2 import record_start, summarise_run


def displace(point: Sequence[float], rng: random.Random, sigma: float) -> list[float]:
    """Add ``N(0, sigma^2)`` to each coordinate. ``sigma = 0`` returns a copy."""
    values = [float(value) for value in point]
    if sigma == 0.0:
        return values
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    return [value + rng.gauss(0.0, sigma) for value in values]


def random_orientation(rng: random.Random, size: int) -> list[float]:
    """Unit vector from independent normal components."""
    if size <= 0:
        raise ValueError("size must be positive")
    raw = [rng.gauss(0.0, 1.0) for _ in range(size)]
    norm = math.sqrt(sum(value * value for value in raw))
    if norm == 0.0:
        raw[0] = 1.0
        norm = 1.0
    return [value / norm for value in raw]


def _norm(gradient) -> float:
    return math.sqrt(sum(value * value for value in gradient))


def descend(potential, point: Sequence[float], step: float, cap: int, force_norm: float) -> dict:
    """Gradient descent. The first evaluation counts, as in the quadratic smoke."""
    current = [float(value) for value in point]
    last = None
    for _ in range(cap):
        energy, gradient = potential.energy_forces(current)
        size = _norm(gradient)
        last = (energy, gradient, current[:], size)
        if size <= force_norm:
            return {"converged": True, "energy": energy, "x": current[:], "force_norm": size}
        current = [value - step * component for value, component in zip(current, gradient, strict=True)]
    energy, gradient, current, size = last
    return {"converged": False, "energy": energy, "x": current, "force_norm": size}


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("orientation has length zero")
    return [value / norm for value in vector]


def _curvature(potential, point: Sequence[float], mode: Sequence[float], delta: float) -> float:
    plus = [value + delta * step for value, step in zip(point, mode, strict=True)]
    minus = [value - delta * step for value, step in zip(point, mode, strict=True)]
    _energy_plus, grad_plus = potential.energy_forces(plus)
    _energy_minus, grad_minus = potential.energy_forces(minus)
    return sum((left - right) * step for left, right, step in zip(grad_plus, grad_minus, mode, strict=True)) / (
        2.0 * delta
    )


def dimer_search(
    potential,
    point: Sequence[float],
    orientation: Sequence[float],
    *,
    delta: float = 1e-3,
    angle: float = 0.3,
    rot_cap: int = 12,
    trans_step: float = 0.25,
    trans_cap: int = 80,
    force_norm: float = 1e-6,
) -> dict:
    """Rotate a dimer mode onto the lowest curvature, then translate.

    Curvature uses two force evaluations. Each rotation tries both
    directions in the plane perpendicular to the mode, in two dimensions
    the single perpendicular line, and keeps going while a neighbour is
    lower. Stopping at the first negative sample leaves a sideways force.
    Translation follows the effective force that reverses the component
    along the mode.
    """
    if len(point) != 2 or len(orientation) != 2:
        raise ValueError("this dimer sample is a point in the plane")
    mode = _unit(orientation)
    current = [float(value) for value in point]
    aligned = False
    # A barely negative curvature still has a sideways force, and a fixed
    # step then leaves the saddle. Keep the rotation that lowers curvature
    # until neither neighbour is lower.
    for _ in range(rot_cap):
        curve = _curvature(potential, current, mode, delta)
        side = _unit([-mode[1], mode[0]])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        left = [cosine * mode[i] + sine * side[i] for i in range(2)]
        right = [cosine * mode[i] - sine * side[i] for i in range(2)]
        left_curve = _curvature(potential, current, left, delta)
        right_curve = _curvature(potential, current, right, delta)
        if min(left_curve, right_curve) >= curve:
            aligned = curve < 0.0
            break
        mode = left if left_curve <= right_curve else right
    else:
        curve = _curvature(potential, current, mode, delta)
        aligned = curve < 0.0
    last = None
    for _ in range(trans_cap):
        energy, gradient = potential.energy_forces(current)
        dot = sum(component * axis for component, axis in zip(gradient, mode, strict=True))
        effective = [-component + 2.0 * dot * axis for component, axis in zip(gradient, mode, strict=True)]
        true_norm = _norm(gradient)
        last = (energy, current[:], true_norm)
        if true_norm <= force_norm and aligned:
            return {"converged": True, "energy": energy, "x": current[:], "force_norm": true_norm}
        current = [value + trans_step * component for value, component in zip(current, effective, strict=True)]
    energy, current, true_norm = last
    return {
        "converged": False,
        "energy": energy,
        "x": current,
        "force_norm": true_norm,
        "diverged": true_norm > 1.0,
    }


def run_seeded(
    evaluate: Callable,
    *,
    x0: Sequence[float],
    kind: str,
    reference_energy: float,
    reference_coords: Sequence[Sequence[float]],
    cap: int,
    force_norm: float,
    energy_tol: float,
    rmsd_tol: float,
    samples: int,
    seed: int,
    displacement: float,
    step: float | None = None,
    mode: str = "displacement",
) -> tuple[list[dict], dict]:
    """``samples`` starts. ``mode`` is ``displacement`` or ``dimer``."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    starts = []
    for index in range(samples):
        rng = random.Random(seed + index)
        origin = displace(x0, rng, displacement)
        if mode == "displacement":
            if step is None:
                raise ValueError("displacement mode needs a step")

            def optimize(potential, origin=origin, step=step, cap=cap, force_norm=force_norm):
                return descend(potential, origin, step, cap, force_norm)

        elif mode == "dimer":
            orientation = random_orientation(rng, len(origin))

            def optimize(potential, origin=origin, orientation=orientation, force_norm=force_norm):
                return dimer_search(
                    potential,
                    origin,
                    orientation,
                    force_norm=force_norm,
                )

        else:
            raise ValueError("mode must be displacement or dimer")
        starts.append(
            record_start(
                f"seed-{index}",
                evaluate,
                optimize,
                cap=cap,
                kind=kind,
                reference_energy=reference_energy,
                reference_coords=reference_coords,
                force_norm=force_norm,
                energy_tol=energy_tol,
                rmsd_tol=rmsd_tol,
            )
        )
    return starts, summarise_run(starts, cap)
