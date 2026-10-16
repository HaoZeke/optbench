"""Additive Gaussian force noise and a stop that can be met under it.

The energy is the clean energy. Each force component gains an independent
normal sample of standard deviation ``sigma``. A clean tolerance below
the noise floor cannot be a convergence test: the observed Euclidean
norm of that noise is typically ``sigma * sqrt(components)``.

The stop is the larger of the clean force tolerance and
``kappa * sigma * sqrt(components)``, and, when ``sigma`` is positive,
the range of the clean energy over the last ``window`` evaluations is
at most the energy tolerance. One lucky force sample does not stop the
search. ``sigma = 0`` is the clean force stop.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

SIGMA_LEVELS = (0.0, 1e-4, 1e-3, 1e-2)
KAPPA = 3.0
WINDOW = 5


def noise_floor(sigma: float, components: int, kappa: float = KAPPA) -> float:
    """RMS scale of the Euclidean norm of isotropic Gaussian force noise."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if components <= 0:
        raise ValueError("components must be positive")
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    return kappa * sigma * math.sqrt(components)


def stop_threshold(clean_tol: float, sigma: float, components: int, kappa: float = KAPPA) -> float:
    """Force-norm stop. The clean tolerance remains in force when it is larger."""
    return max(float(clean_tol), noise_floor(sigma, components, kappa))


def energy_span(energies: Sequence[float]) -> float:
    if not energies:
        raise ValueError("energies is empty")
    return max(energies) - min(energies)


def converged_under_noise(
    *,
    force_norm: float,
    energies: Sequence[float],
    clean_tol: float,
    sigma: float,
    components: int,
    energy_tol: float,
    kappa: float = KAPPA,
    window: int = WINDOW,
) -> bool:
    """Whether the latest evaluation may stop.

    Positive ``sigma`` requires ``window`` clean energies whose range is
    within ``energy_tol``, and a force norm at or below the noise-aware
    threshold. Zero ``sigma`` ignores the window and uses ``clean_tol``.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if sigma == 0.0:
        return float(force_norm) <= float(clean_tol)
    if len(energies) < window:
        return False
    recent = list(energies[-window:])
    threshold = stop_threshold(clean_tol, sigma, components, kappa)
    return float(force_norm) <= threshold and energy_span(recent) <= float(energy_tol)


def gaussian_forces(evaluate: Callable, sigma: float, rng: random.Random) -> Callable:
    """Return ``evaluate`` with ``N(0, sigma^2)`` added to each force component.

    ``sigma = 0`` does not draw, so the clean path is unchanged.
    """

    def wrapped(coords):
        energy, gradient = evaluate(coords)
        if sigma == 0.0:
            return energy, tuple(float(value) for value in gradient)
        noisy = tuple(float(value) + rng.gauss(0.0, sigma) for value in gradient)
        return energy, noisy

    return wrapped


def _norm(gradient) -> float:
    return math.sqrt(sum(value * value for value in gradient))


def noisy_descent(
    evaluate: Callable,
    x0: Sequence[float],
    step: float,
    cap: int,
    *,
    sigma: float,
    rng: random.Random,
    clean_tol: float,
    energy_tol: float,
    kappa: float = KAPPA,
    window: int = WINDOW,
):
    """Gradient descent that stops with :func:`converged_under_noise`.

    The harness potential stays the clean ``evaluate``. Noise is added to
    the force the stepper sees, after the call is counted. The Hessian
    check then differentiates the clean force, which is the stationary
    point. Differentiating the noise would invent an index.
    """

    def optimize(potential):
        point = [float(value) for value in x0]
        energies: list[float] = []
        last = None
        components = len(point)
        for _ in range(cap):
            energy, gradient = potential.energy_forces(point)
            gradient = tuple(float(value) for value in gradient)
            if sigma != 0.0:
                gradient = tuple(value + rng.gauss(0.0, sigma) for value in gradient)
            energies.append(float(energy))
            size = _norm(gradient)
            last = (float(energy), gradient, point[:], size)
            if converged_under_noise(
                force_norm=size,
                energies=energies,
                clean_tol=clean_tol,
                sigma=sigma,
                components=components,
                energy_tol=energy_tol,
                kappa=kappa,
                window=window,
            ):
                return {"converged": True, "energy": float(energy), "x": point[:], "force_norm": size}
            point = [value - step * component for value, component in zip(point, gradient, strict=True)]
        energy, gradient, point, size = last
        return {"converged": False, "energy": energy, "x": point, "force_norm": size}

    optimize.evaluate = evaluate
    return optimize
