"""Count potential evaluations at the harness boundary.

One force call is one entry into the wrapped potential. ``HarnessCounter``
is that tally. In-process wrappers cache a second request at the same
coordinates, because an energy read and a force read of one geometry are
one evaluation. A socket request is already one evaluation the client
chose to make, so the server counts every request.

Calls made while verifying a stationary point increment ``verify_calls``
and are not part of the search cost.

A code's own counter is not this number. Two optimizers that call this
wrapper pay the same count for the same evaluations.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence


def _key(coords: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in coords)


class ForceCallCap(RuntimeError):
    """A search call at the budget. It is not counted and does not run."""


class HarnessCounter:
    """The single force-call tally.

    ``cache`` is true for an in-process wrapper and false for the socket,
    where each request is one client entry. ``cap`` refuses a search call
    once ``calls`` has reached it. The refused call is not counted, and a
    verification call does not spend the budget.
    """

    def __init__(self, *, cache: bool = True, cap: int | None = None):
        self.calls = 0
        self.verify_calls = 0
        self.cap = None if cap is None else int(cap)
        self._verifying = False
        self._cache_enabled = cache
        self._cache_key: tuple[float, ...] | None = None
        self._cached: tuple[float, tuple[float, ...]] | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.verify_calls = 0
            self._verifying = False
            self._cache_key = None
            self._cached = None

    def charge(self, key: tuple[float, ...]) -> tuple[bool, tuple[float, tuple[float, ...]] | None]:
        """Count one entry unless it repeats the cached coordinates.

        Returns ``(fresh, cached)``. A cache hit is not a new call.
        """
        with self._lock:
            if self._cache_enabled and key == self._cache_key and self._cached is not None:
                return False, self._cached
            if self._verifying:
                self.verify_calls += 1
            else:
                if self.cap is not None and self.calls >= self.cap:
                    raise ForceCallCap(f"force-call cap {self.cap}")
                self.calls += 1
            return True, None

    def remember(self, key: tuple[float, ...], value: tuple[float, tuple[float, ...]]):
        with self._lock:
            if self._cache_enabled:
                self._cache_key = key
                self._cached = value
            return value

    def verification(self) -> _Verification:
        return _Verification(self)


class _Verification:
    def __init__(self, counter: HarnessCounter):
        self._counter = counter
        self._previous = False

    def __enter__(self) -> HarnessCounter:
        with self._counter._lock:
            self._previous = self._counter._verifying
            self._counter._verifying = True
        return self._counter

    def __exit__(self, *exc: object) -> None:
        with self._counter._lock:
            self._counter._verifying = self._previous


class CountedPotential:
    """Wrap ``evaluate(coords) -> (energy, gradient)``."""

    def __init__(self, evaluate: Callable[[Sequence[float]], tuple[float, Sequence[float]]]):
        self._evaluate = evaluate
        self._counter = HarnessCounter(cache=True)

    @property
    def calls(self) -> int:
        return self._counter.calls

    @property
    def verify_calls(self) -> int:
        return self._counter.verify_calls

    def reset(self) -> None:
        self._counter.reset()

    def energy_forces(self, coords: Sequence[float]) -> tuple[float, tuple[float, ...]]:
        key = _key(coords)
        fresh, cached = self._counter.charge(key)
        if not fresh and cached is not None:
            return cached
        energy, gradient = self._evaluate(key)
        value = (float(energy), tuple(float(item) for item in gradient))
        return self._counter.remember(key, value)

    def verification(self) -> _Verification:
        return self._counter.verification()


def attach_ase_counter(calculator: object, cap: int | None = None) -> object:
    """Count ASE ``calculate`` entries on this calculator instance.

    ASE skips ``calculate`` when the atoms have not changed, so an energy
    request and a force request at the same geometry are one call. A
    call past ``cap`` raises ``RuntimeError`` before the potential runs.
    """
    original = calculator.calculate
    counter = HarnessCounter(cache=False, cap=cap)
    state = {"calls": 0, "verify_calls": 0, "verifying": False, "cap": cap}

    def calculate(atoms=None, properties=("energy", "forces"), system_changes=("positions",)):
        counter._verifying = bool(state["verifying"])
        try:
            counter.charge((float(counter.calls + counter.verify_calls),))
        except ForceCallCap as exc:
            raise RuntimeError("force-call cap") from exc
        state["calls"] = counter.calls
        state["verify_calls"] = counter.verify_calls
        return original(atoms, properties, system_changes)

    calculator.calculate = calculate
    calculator.optbench_counter = state
    calculator.optbench_harness = counter
    return calculator


class PeleCounter:
    """Count pele potential entries.

    ``getEnergy`` goes through ``getEnergyGradient``, so a caller that
    asks for the energy alone is charged one call, not two.
    """

    def __init__(self, inner: object):
        self._inner = inner
        self._counter = HarnessCounter(cache=False)

    @property
    def calls(self) -> int:
        return self._counter.calls

    @property
    def verify_calls(self) -> int:
        return self._counter.verify_calls

    def getEnergyGradient(self, coords):
        key = tuple(float(value) for value in coords)
        self._counter.charge(key)
        return self._inner.getEnergyGradient(coords)

    def getEnergy(self, coords):
        energy, _gradient = self.getEnergyGradient(coords)
        return energy

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
