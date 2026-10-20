"""Harness budget shared by the lane-B searches.

One fresh coordinate is one force call. A repeated coordinate is free.
A call at the cap is refused and is not counted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter


class Converged(Exception):
    """The Euclidean force stop has been met under the cap."""


class Hit(Exception):
    """A converged quench landed inside the global-minimum window."""


class Budget:
    """Counted ``(energy, gradient)`` with the last point kept."""

    def __init__(
        self,
        evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
        cap: int,
        *,
        target: float | None = None,
        window: float = 0.0,
    ):
        self.counter = HarnessCounter(cache=True, cap=int(cap))
        self._evaluate = evaluate
        self.cap = int(cap)
        self.target = None if target is None else float(target)
        self.window = float(window)
        self.energy: float | None = None
        self.gradient: np.ndarray | None = None
        self.point: np.ndarray | None = None
        self.hit = False
        self.quenched = False
        self.quench_energy: float | None = None
        self.quench_point: np.ndarray | None = None
        self.clean_energies: list[float] = []

    @property
    def calls(self) -> int:
        return int(self.counter.calls)

    def fg(self, coords: Sequence[float] | np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(coords, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in x)
        fresh, cached = self.counter.charge(key)
        if not fresh and cached is not None:
            energy, gradient = cached
            return float(energy), np.asarray(gradient, dtype=np.float64)
        energy, gradient = self._evaluate(x)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
        stored = (float(energy), tuple(float(value) for value in gradient))
        self.counter.remember(key, stored)
        self.energy = float(energy)
        self.gradient = gradient
        self.point = x.copy()
        self.clean_energies.append(float(energy))
        return float(energy), gradient

    def note_quench(
        self,
        point: Sequence[float] | np.ndarray,
        energy: float,
        gradient: Sequence[float] | np.ndarray,
        *,
        converged: bool,
    ) -> None:
        """Record a finished local minimization. A hit raises :class:`Hit`.

        The stored structure is the lowest converged quench. A later
        quench replaces it only when that quench is lower, or when it
        lands inside the global-minimum window.
        """
        if not converged:
            return
        point_arr = np.asarray(point, dtype=np.float64).reshape(-1).copy()
        grad_arr = np.asarray(gradient, dtype=np.float64).reshape(-1).copy()
        energy_f = float(energy)
        self.quenched = True
        in_window = (
            self.target is not None
            and energy_f <= self.target + self.window
            and self.calls < self.cap
        )
        better = self.quench_energy is None or energy_f < self.quench_energy
        if better or in_window:
            self.quench_energy = energy_f
            self.quench_point = point_arr
            self.point = point_arr.copy()
            self.energy = energy_f
            self.gradient = grad_arr
        if in_window:
            self.hit = True
            raise Hit()
