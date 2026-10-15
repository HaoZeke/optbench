"""Count potential evaluations at the harness boundary.

One force call is one entry into the wrapped potential. A later request
at the same coordinates returns the cached energy and forces and does
not add a call. Calls made while verifying a stationary point are
counted separately and are not part of the search cost.

A code's own counter is not this number. Two optimizers that call this
wrapper pay the same count for the same evaluations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def _key(coords: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in coords)


class CountedPotential:
    """Wrap ``evaluate(coords) -> (energy, gradient)``."""

    def __init__(self, evaluate: Callable[[Sequence[float]], tuple[float, Sequence[float]]]):
        self._evaluate = evaluate
        self.calls = 0
        self.verify_calls = 0
        self._verifying = False
        self._cache_key: tuple[float, ...] | None = None
        self._cached: tuple[float, tuple[float, ...]] | None = None

    def reset(self) -> None:
        self.calls = 0
        self.verify_calls = 0
        self._verifying = False
        self._cache_key = None
        self._cached = None

    def energy_forces(self, coords: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        key = _key(coords)
        if key == self._cache_key and self._cached is not None:
            return self._cached
        if self._verifying:
            self.verify_calls += 1
        else:
            self.calls += 1
        energy, gradient = self._evaluate(key)
        cached = (float(energy), tuple(float(value) for value in gradient))
        self._cache_key = key
        self._cached = cached
        return cached

    def verification(self) -> _Verification:
        return _Verification(self)


class _Verification:
    def __init__(self, potential: CountedPotential):
        self._potential = potential
        self._previous = False

    def __enter__(self) -> CountedPotential:
        self._previous = self._potential._verifying
        self._potential._verifying = True
        return self._potential

    def __exit__(self, *exc: object) -> None:
        self._potential._verifying = self._previous


def attach_ase_counter(calculator: object, cap: int | None = None) -> object:
    """Count ASE ``calculate`` entries on this calculator instance.

    ASE skips ``calculate`` when the atoms have not changed, so an energy
    request and a force request at the same geometry are one call. A
    call past ``cap`` raises ``RuntimeError`` before the potential runs.
    """
    original = calculator.calculate
    state = {"calls": 0, "verify_calls": 0, "verifying": False, "cap": cap}

    def calculate(atoms=None, properties=("energy", "forces"), system_changes=("positions",)):
        if state["verifying"]:
            state["verify_calls"] += 1
        else:
            state["calls"] += 1
            if cap is not None and state["calls"] > cap:
                raise RuntimeError("force-call cap")
        return original(atoms, properties, system_changes)

    calculator.calculate = calculate
    calculator.optbench_counter = state
    return calculator


class PeleCounter:
    """Count pele potential entries.

    ``getEnergy`` goes through ``getEnergyGradient``, so a caller that
    asks for the energy alone is charged one call, not two.
    """

    def __init__(self, inner: object):
        self._inner = inner
        self.calls = 0
        self.verify_calls = 0
        self._verifying = False

    def getEnergyGradient(self, coords):
        if self._verifying:
            self.verify_calls += 1
        else:
            self.calls += 1
        return self._inner.getEnergyGradient(coords)

    def getEnergy(self, coords):
        energy, _gradient = self.getEnergyGradient(coords)
        return energy

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
