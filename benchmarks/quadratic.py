"""In-process v2 sample, one asv parameter pair per start.

The seed and the start index are the parameters. Each class calls the
same function as ``optbench.statistics``, so the track values are that
sample. These runs do not need eonclient.

``cap`` is the PAR-2 penalty for the class. asv result rows do not
store it. ``optbench.asv_score`` reads it from the class.
"""

from __future__ import annotations

from optbench.asv_tracks import TrackedSearch as _TrackedSearch
from optbench.statistics import (
    DIMER_CAP,
    DIMER_SAMPLES,
    DISPLACEMENT_CAP,
    DISPLACEMENT_SAMPLES,
    FAILURE_ORDER,
    KEPT_CAP,
    NOISE_CAP,
    PROFILE_SEED,
    PROFILE_STARTS,
    SEED,
    SIGMA_LEVELS,
    dimer_start,
    displacement_start,
    failure_start,
    kept_start,
    noise_start,
    profile_start,
)


def _seeds(count: int) -> list[str]:
    return [str(SEED + index) for index in range(count)]


class QuadraticDisplacement(_TrackedSearch):
    cap = DISPLACEMENT_CAP
    param_names = ["seed", "start"]
    params = [_seeds(DISPLACEMENT_SAMPLES), ["0"]]

    def measure(self, seed, start):
        if start != "0":
            raise ValueError("the displacement start is the origin")
        return displacement_start(int(seed))


class QuadraticDimer(_TrackedSearch):
    cap = DIMER_CAP
    param_names = ["seed", "start"]
    params = [_seeds(DIMER_SAMPLES), ["0"]]

    def measure(self, seed, start):
        if start != "0":
            raise ValueError("the dimer start is the fixed origin")
        return dimer_start(int(seed))


def _profile(name: str, code_name: str, step: float, cap: int):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [[str(PROFILE_SEED)], [str(index) for index in range(PROFILE_STARTS)]]

        def measure(self, seed, start):
            if int(seed) != PROFILE_SEED:
                raise ValueError("the profile draws use PROFILE_SEED")
            return profile_start(int(start), step, cap)

    _Benchmark.__name__ = name
    _Benchmark.__qualname__ = name
    _Benchmark.cap = cap
    _Benchmark.step = step
    _Benchmark.code_name = code_name
    return _Benchmark


QuadraticProfileStep005 = _profile("QuadraticProfileStep005", "gd-step-0.05", 0.05, 5000)
QuadraticProfileStep001 = _profile("QuadraticProfileStep001", "gd-step-0.01", 0.01, 20000)
QuadraticProfileCap30 = _profile("QuadraticProfileCap30", "gd-step-0.001-cap-30", 0.001, 30)


def _sigma_name(sigma: float) -> str:
    if sigma == 0.0:
        return "QuadraticNoise0"
    text = f"{sigma:.0e}".replace("+", "").replace("-", "m")
    return f"QuadraticNoise{text}"


def _noise(sigma: float):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [[str(SEED)], ["0"]]

        def measure(self, seed, start):
            if int(seed) != SEED or start != "0":
                raise ValueError("the noise sample is one seed and the origin")
            return noise_start(sigma)

    name = _sigma_name(sigma)
    _Benchmark.__name__ = name
    _Benchmark.__qualname__ = name
    _Benchmark.cap = NOISE_CAP
    _Benchmark.sigma = sigma
    return _Benchmark


for _sigma in SIGMA_LEVELS:
    _noise_cls = _noise(_sigma)
    globals()[_noise_cls.__name__] = _noise_cls


_FAILURE_CAPS = {
    "cap": 200,
    "wrong_stationary": 5000,
    "divergence": 100,
    "crash": 100,
}


def _failure(kind: str):
    class _Benchmark(_TrackedSearch):
        param_names = ["seed", "start"]
        params = [["0"], ["0"]]

        def measure(self, seed, start):
            record, cap = failure_start(kind)
            if cap != _FAILURE_CAPS[kind]:
                raise RuntimeError("failure cap drifted from the benchmark class")
            return record

    name = "QuadraticFailure" + "".join(part.capitalize() for part in kind.split("_"))
    _Benchmark.__name__ = name
    _Benchmark.__qualname__ = name
    _Benchmark.kind = kind
    _Benchmark.cap = _FAILURE_CAPS[kind]
    return _Benchmark


for _kind in FAILURE_ORDER:
    _failure_cls = _failure(_kind)
    globals()[_failure_cls.__name__] = _failure_cls


class QuadraticKept(_TrackedSearch):
    """Success under cap 200. Scored with QuadraticFailureCap."""

    cap = KEPT_CAP
    param_names = ["seed", "start"]
    params = [["0"], ["0"]]

    def measure(self, seed, start):
        return kept_start()
