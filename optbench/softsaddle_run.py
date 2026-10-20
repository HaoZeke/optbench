"""SoftSaddle on the harness counter.

The Lanczos and Davidson solvers are the 2017 libraries. Each shared
object exports one C entry, ``softsaddle_minimum_mode``, and the spec
potential supplies the gradient. The saddle campaign is the archived
adaptive-hypersphere LBFGS minimum-mode walk: memory 60, step 0.5,
eight eigen-iterations, and a batch of 25. Each repetition loads one
archived 5000-direction hypersphere, starts again at radius 0.1, and
keeps its own saddle list. A fresh library is loaded for every search,
which is what ``MyOwn_v10_morse`` does with ``unloadlibrary``. The
lowest-mode campaign repeats that eigen-solve until the overlap stop
or the call cap.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import statistics
from datetime import date
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .connection import ProcessCatalog, connected_endpoints, link_tolerance, per_atom_max
from .endpoints import EndpointArchive, stamp_endpoint, stored_cell
from .failures import classify_failure
from .rgsaddle_minmode import (
    HarnessSurface,
    _SADDLE_CLIMB_OVERSHOOT,
    _SADDLE_ENDPOINT_BUDGET,
    _bind,
    _cartesian_rmsd,
    _finite_point,
    _geometries,
    _hessian_spectrum,
    _lbfgs_minimize,
    _norm,
    _outcome,
    _unit,
    _vibrational_eigenvalues,
)
from .score import half_even, score_v2
from .stationary import negative_index
from .surfaces import load_con, potential_from_problem, read_mode
from .v2 import collect_provenance, write_starts

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_GRAD = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_void_p,
)

INITIAL_RANDOM = 0
INITIAL_GRADIENT = 1
INITIAL_USER = 2
INITIAL_PREVIOUS = 3

CLIMB_BUDGET = 1000
HYPER_SOURCE = "eb78f6d9cde1e9aae308e3f8ecec777fdb655e3432e356c9da339b950bd3267a"
_LIBS: dict[str, ctypes.CDLL] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def library_path(kind: str) -> Path | None:
    """Shared object for ``lanczos`` or ``davidson``, or None when it was not built."""
    env = "SOFTSADDLE_DAVIDSON" if kind == "davidson" else "SOFTSADDLE_LANCZOS"
    named = os.environ.get(env)
    if named:
        path = Path(named)
        return path if path.is_file() else None
    directory = Path(os.environ.get("SOFTSADDLE_LIB_DIR", str(_repo_root() / "softsaddle" / "lib")))
    path = directory / f"libsoftsaddle_{kind}.so"
    return path if path.is_file() else None


def libraries_ready() -> bool:
    return library_path("lanczos") is not None and library_path("davidson") is not None


def _load(kind: str) -> ctypes.CDLL:
    cached = _LIBS.get(kind)
    if cached is not None:
        return cached
    path = library_path(kind)
    if path is None:
        raise RuntimeError(
            f"softsaddle {kind} library is not built; run softsaddle/driver/build.sh"
        )
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.softsaddle_reset.argtypes = []
    lib.softsaddle_seed.argtypes = [ctypes.c_uint]
    lib.softsaddle_set_error.argtypes = [ctypes.c_int]
    lib.softsaddle_kk.restype = ctypes.c_int
    lib.softsaddle_minimum_mode.restype = ctypes.c_int
    lib.softsaddle_minimum_mode.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        _GRAD,
        ctypes.c_void_p,
    ]
    _LIBS[kind] = lib
    return lib


def _section(template: dict, name: str) -> dict:
    return ((template.get("ini") or {}).get(name) or {})


@contextmanager
def _quiet(path: Path):
    """Send C++ stdout and stderr to ``path`` for one search."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(handle, 1)
        os.dup2(handle, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(handle)


@contextmanager
def _at(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        with _quiet(path / "stdio.log"):
            yield
    finally:
        os.chdir(previous)


def _minimum_mode(
    lib: ctypes.CDLL,
    potential: HarnessSurface,
    x: np.ndarray,
    *,
    initial: int,
    iteration_limit: int,
    finite: float,
    tolerance: float,
    seed_vector: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, int, int, float]:
    """One solver call. The gradient at ``x`` is the vector returned last."""
    point = np.ascontiguousarray(x, dtype=np.float64)
    n = int(point.size)
    eigenvalue = ctypes.c_double(float("inf"))
    vector = np.zeros(n, dtype=np.float64)
    if seed_vector is not None and seed_vector.size == n:
        vector[:] = seed_vector
    gradient = np.zeros(n, dtype=np.float64)
    iterations = ctypes.c_int()
    convergence = ctypes.c_double()
    status = ctypes.c_int()
    holder = {"energy": None, "error": None}

    def invoke(x_ptr, g_ptr, length, _user):
        try:
            coords = np.ctypeslib.as_array(x_ptr, shape=(length,)).copy()
            _energy, grad = potential.energy_forces(coords)
            out = np.ctypeslib.as_array(g_ptr, shape=(length,))
            out[:] = np.asarray(grad, dtype=np.float64)
            if length == n and float(np.max(np.abs(coords - point))) == 0.0:
                holder["energy"] = float(_energy)
        except Exception as exc:
            holder["error"] = repr(exc)
            np.ctypeslib.as_array(g_ptr, shape=(length,))[:] = 0.0
            lib.softsaddle_set_error(1)

    callback = _GRAD(invoke)
    rc = lib.softsaddle_minimum_mode(
        point.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        ctypes.byref(eigenvalue),
        vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(initial),
        int(iteration_limit),
        float(finite),
        float(tolerance),
        ctypes.byref(iterations),
        ctypes.byref(convergence),
        ctypes.byref(status),
        callback,
        None,
    )
    del callback
    if holder["error"] or rc:
        raise RuntimeError(holder["error"] or "softsaddle gradient callback failed")
    if holder["energy"] is None:
        energy, _gradient = potential.energy_forces(point)
        holder["energy"] = float(energy)
    return (
        float(eigenvalue.value),
        vector,
        gradient,
        int(iterations.value),
        int(status.value),
        float(holder["energy"]),
    )


def _effective(gradient: np.ndarray, eigenvalue: float, mode: np.ndarray) -> np.ndarray:
    """Minimum-mode gradient from ``get_func_and_grad_v8_morse``.

    The positive-curvature branch is the projection onto the mode.
    The negative branch reflects the gradient through that mode.
    ``use_same_formula`` is false, so the positive branch stays on
    while the search has not left the starting basin.
    """
    direction = _unit(mode)
    product = float(np.dot(gradient, direction))
    if eigenvalue >= 0.0:
        return -product * direction
    return gradient - 2.0 * product * direction


def _max_motion(vector: np.ndarray, limit: float) -> np.ndarray:
    length = _norm(vector)
    if length > limit > 0.0:
        return vector * (limit / length)
    return vector


def _sign(value: float) -> float:
    """MATLAB ``sign``. Zero stays zero; ``math.copysign`` does not."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _lambda_zero(xpos: np.ndarray, xneg: np.ndarray, epos: float, eneg: float) -> np.ndarray | None:
    """Linear crossing from ``x_lamda_zero.m``. The caller passes the MATLAB argument order."""
    delta = np.asarray(xneg, dtype=np.float64) - np.asarray(xpos, dtype=np.float64)
    distance = _norm(delta)
    span = float(epos) - float(eneg)
    if distance < 1e-15 or span == 0.0 or not math.isfinite(span):
        return None
    return np.asarray(xpos, dtype=np.float64) + (float(epos) * distance / span) * (delta / distance)


def _analyze(mode, vector, step, accumulated, count: int, tolerance: float):
    """Reuse the mode while the running step stays inside ``tolerance``."""
    if accumulated is None:
        total = np.array(step, dtype=np.float64, copy=True)
    else:
        total = accumulated + step
    if _norm(total) > tolerance:
        return None, None, np.zeros_like(step), 0
    return mode, vector, total, count


class _LBFGS:
    """Two-loop memory from ``LBFGS_morse_v6`` ``updateLBFGS`` and ``getStep``."""

    def __init__(self, dimension: int, memory: int):
        self.memory = int(memory)
        self.s = np.zeros((dimension, self.memory), dtype=np.float64)
        self.y = np.zeros((dimension, self.memory), dtype=np.float64)
        self.rho = np.zeros(self.memory, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray, x_prev: np.ndarray, g: np.ndarray, g_prev: np.ndarray) -> None:
        if self.count == self.memory:
            self.s[:, :-1] = self.s[:, 1:]
            self.y[:, :-1] = self.y[:, 1:]
            self.rho[:-1] = self.rho[1:]
            self.count = self.memory - 1
        step = x - x_prev
        change = g - g_prev
        self.count += 1
        slot = self.count - 1
        self.s[:, slot] = step
        self.y[:, slot] = change
        denom = float(np.dot(change, step))
        self.rho[slot] = 1.0 / denom if denom != 0.0 else 0.0

    def step(
        self,
        x: np.ndarray,
        g: np.ndarray,
        *,
        iteration: int,
        finite: float,
        max_step: float,
        potential: HarnessSurface,
        eigenvalue: float | None,
        mode: np.ndarray | None,
        bandera: bool,
    ) -> tuple[np.ndarray, bool]:
        forced = bandera
        if iteration > 0:
            if forced or self.count < 1:
                curvature = -0.01
                forced = False
            else:
                slot = self.count - 1
                denom = float(np.dot(self.s[:, slot], self.y[:, slot]))
                curvature = float(np.dot(self.y[:, slot], self.y[:, slot]) / denom) if denom != 0.0 else float("nan")
            if not math.isfinite(curvature) or curvature < 0.0:
                self.count = 0
                return _max_motion(-g * 1000.0, max_step), False
        if iteration == 0:
            if _norm(g) < 1e-15:
                kick = np.zeros_like(g)
                kick[0] = 1.0
                self.count = 0
                return _max_motion(kick, max_step), False
            if forced:
                curvature = -0.01
            else:
                length = _norm(g)
                direction = -g / length if length > 0.0 else -g
                trial = x + finite * direction
                _energy, trial_gradient = potential.energy_forces(trial)
                if eigenvalue is None or mode is None:
                    trial_effective = trial_gradient
                else:
                    trial_effective = _effective(trial_gradient, eigenvalue, mode)
                curvature = float(np.dot(trial_effective - g, direction) / finite) if finite != 0.0 else float("nan")
            if not math.isfinite(curvature) or curvature < 0.0:
                self.count = 0
                return _max_motion(-g * 1000.0, max_step), False
        scale = 1.0 / curvature
        q = np.array(g, dtype=np.float64, copy=True)
        alpha = np.zeros(self.count, dtype=np.float64)
        for index in range(self.count - 1, -1, -1):
            alpha[index] = self.rho[index] * float(np.dot(self.s[:, index], q))
            q = q - alpha[index] * self.y[:, index]
        direction_h = scale * q
        for index in range(self.count):
            beta = self.rho[index] * float(np.dot(self.y[:, index], direction_h))
            direction_h = direction_h + (alpha[index] - beta) * self.s[:, index]
        proposed = -direction_h
        length = _norm(proposed)
        gradient_length = _norm(g)
        if length >= max_step or length < 1e-15 or gradient_length < 1e-15:
            self.count = 0
            return _max_motion(scale * -g, max_step), False
        descent = float(np.dot(proposed / length, -g / gradient_length))
        if descent < 0.0:
            self.count = 0
            return _max_motion(scale * -g, max_step), False
        return _max_motion(proposed, max_step), False


def _settings(template: dict) -> dict:
    optimizer = _section(template, "Optimizer")
    saddle = _section(template, "SaddleSearch")
    search = _section(template, "Search")
    main = _section(template, "Main")
    factor = float(saddle.get("save_lanczos_factor", 50.0))
    max_step = float(optimizer.get("max_step", 0.5))
    ls_err = float(optimizer.get("ls_err", 1e-2))
    return {
        "solver": str(main.get("solver", "lanczos")),
        "seed_mode": str(saddle.get("mode_seed", "previous")),
        "iteration_limit": int(saddle.get("lanc_iter", 8)),
        "tolerance": float(saddle.get("lanc_abs_tol", 0.01)),
        "finite": float(saddle.get("lanc_finite", optimizer.get("finite_diff", 1e-4))),
        "memory": int(optimizer.get("memory_size", 60)),
        "max_step": max_step,
        "cg_iter_max": int(optimizer.get("max_iterations", CLIMB_BUDGET)),
        "cg_err": float(optimizer.get("cg_err", 1e-3)),
        "ls_err": ls_err,
        "reuse_tolerance": ls_err + (max_step - ls_err) * factor / 100.0,
        "save_lanczos": bool(saddle.get("save_lanczos_call", True)),
        "dE_max": float(optimizer.get("dE_max", 480.0)),
        "batch": int(search.get("batch", 25)),
        "hyper_rad0": float(search.get("hyper_rad0", 0.1)),
        "displace_atoms": int(search.get("displace_atoms", 7)),
        "max_searches": int(search.get("max_searches", 5000)),
        "repetitions": int(search.get("repetitions", 1)),
        "mix_radius": bool(search.get("mix_prev_hyper_rad", True)),
        "keep_efficiency": float(search.get("keep_efficiency", 0.2)),
    }


def _mode_at(lib, potential, x, settings, eigenvalue, mode) -> tuple[float, np.ndarray, np.ndarray, float]:
    if eigenvalue is None or mode is None:
        initial = INITIAL_GRADIENT if settings["seed_mode"] == "gradient" else INITIAL_PREVIOUS
        seed_vector = None
        if initial == INITIAL_GRADIENT:
            _energy, gradient = potential.energy_forces(x)
            if _norm(gradient) < 1e-14:
                initial = INITIAL_USER
                seed_vector = np.zeros(x.size, dtype=np.float64)
                seed_vector[0] = 1.0
        eigenvalue, mode, gradient, _iterations, _status, energy = _minimum_mode(
            lib,
            potential,
            x,
            initial=initial,
            iteration_limit=settings["iteration_limit"],
            finite=settings["finite"],
            tolerance=settings["tolerance"],
            seed_vector=seed_vector,
        )
    else:
        energy, gradient = potential.energy_forces(x)
    if not np.all(np.isfinite(gradient)) or not math.isfinite(float(energy)) or not math.isfinite(eigenvalue):
        raise RuntimeError("softsaddle mode returned a non-finite value")
    return float(eigenvalue), np.asarray(mode, dtype=np.float64), np.asarray(gradient, dtype=np.float64), float(energy)


def climb(
    lib,
    potential: HarnessSurface,
    x0: np.ndarray,
    *,
    settings: dict,
    force_tol: float,
    minimum_energy: float,
    climb_budget: int,
) -> dict:
    """One LBFGS minimum-mode climb. Endpoint minimizations are separate."""
    x = np.array(x0, dtype=np.float64, copy=True)
    dimension = int(x.size)
    memory = _LBFGS(dimension, settings["memory"])
    eigenvalue = None
    mode = None
    effective = np.ones(dimension, dtype=np.float64) * 1000.0
    true_gradient = effective.copy()
    energy = float(minimum_energy)
    previous_x = x.copy()
    previous_g = effective.copy()
    previous_eigenvalue = None
    step = np.zeros(dimension, dtype=np.float64)
    accumulated = None
    reuse_count = 0
    bandera = False
    crossed = None
    iteration = 0
    converged = False
    true_norm = float("inf")
    while (
        iteration < int(settings["cg_iter_max"])
        and potential.calls < climb_budget
        and energy - minimum_energy < float(settings["dE_max"])
    ):
        if iteration == 0 or not settings["save_lanczos"]:
            eigenvalue, mode = None, None
        else:
            reuse_count += 1
            eigenvalue, mode, accumulated, reuse_count = _analyze(
                eigenvalue, mode, step, accumulated, reuse_count, settings["reuse_tolerance"]
            )
        try:
            eigenvalue, mode, true_gradient, energy = _mode_at(lib, potential, x, settings, eigenvalue, mode)
        except RuntimeError as exc:
            return {
                "x": x,
                "energy": energy,
                "eigenvalue": eigenvalue,
                "mode": mode,
                "force_norm": true_norm,
                "converged": False,
                "iterations": iteration,
                "x_lambda": crossed,
                "error": str(exc),
            }
        effective = _effective(true_gradient, eigenvalue, mode)
        true_norm = _norm(true_gradient)
        if eigenvalue < 0.0 and true_norm <= force_tol and np.all(np.isfinite(x)):
            converged = True
            break
        if iteration > 0:
            memory.update(x, previous_x, effective, previous_g)
        step, bandera = memory.step(
            x,
            effective,
            iteration=iteration,
            finite=settings["finite"],
            max_step=settings["max_step"],
            potential=potential,
            eigenvalue=eigenvalue,
            mode=mode,
            bandera=bandera,
        )
        if bandera:
            step = _max_motion(step, settings["max_step"])
            bandera = False
        previous_x = x.copy()
        previous_g = effective.copy()
        x = x + step
        if (
            iteration > 0
            and previous_eigenvalue is not None
            and eigenvalue is not None
            and crossed is None
            and _sign(float(previous_eigenvalue)) + _sign(float(eigenvalue)) == 0.0
        ):
            crossed = _lambda_zero(previous_x, x, previous_eigenvalue, eigenvalue)
        previous_eigenvalue = eigenvalue
        iteration += 1
        if eigenvalue is not None and eigenvalue >= 0.0 and float(np.max(np.abs(effective))) <= settings["cg_err"]:
            length = _norm(effective)
            if length > 0.0:
                effective = 1000.0 * effective / length
            bandera = True
    if converged and settings["save_lanczos"]:
        try:
            eigenvalue, mode, true_gradient, energy = _mode_at(lib, potential, x, settings, None, None)
            effective = _effective(true_gradient, eigenvalue, mode)
            true_norm = _norm(true_gradient)
            converged = bool(eigenvalue < 0.0 and true_norm <= force_tol)
        except RuntimeError as exc:
            converged = False
            return {
                "x": x,
                "energy": energy,
                "eigenvalue": eigenvalue,
                "mode": mode,
                "force_norm": true_norm,
                "converged": False,
                "iterations": iteration,
                "x_lambda": crossed,
                "error": str(exc),
            }
    return {
        "x": x,
        "energy": float(energy),
        "eigenvalue": None if eigenvalue is None else float(eigenvalue),
        "mode": None if mode is None else np.asarray(mode, dtype=np.float64),
        "force_norm": float(true_norm) if math.isfinite(true_norm) else float("inf"),
        "converged": bool(converged and _finite_point(x)),
        "iterations": int(iteration),
        "x_lambda": None if crossed is None else np.asarray(crossed, dtype=np.float64),
        "error": None,
    }


def _public(name: str, calls: int, success: bool, failure: str | None, reasons: list[str], **extra) -> dict:
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(extra.pop("verify_calls", 0)),
        "counter": "harness",
        "comparable": True,
        "success": bool(success),
        "failure": failure,
        "reasons": reasons,
    }
    record.update(extra)
    return record


def run_mode_point(spec: dict) -> dict:
    """Lowest mode of one geometry. ``spec`` is picklable."""
    os.environ["OMP_NUM_THREADS"] = "1"
    kind = str(spec["solver"])
    lib = _load(kind)
    evaluate = spec.get("evaluate")
    if evaluate is None:
        positions = np.asarray(spec["positions"], dtype=np.float64)
        fixed = np.asarray(spec["fixed"], dtype=bool)
        cell = np.asarray(spec["cell"], dtype=np.float64)
        free = np.flatnonzero(~fixed)
        potential = HarnessSurface(_bind(potential_from_problem({"potential": spec["potential"]}), positions, free, cell))
        x = positions[free].reshape(-1).copy()
        reference = _unit(np.asarray(spec["reference"], dtype=np.float64))
        fixed_any = bool(np.any(fixed))
    else:
        potential = HarnessSurface(evaluate)
        x = np.asarray(spec["x"], dtype=np.float64).copy()
        reference = _unit(np.asarray(spec["reference"], dtype=np.float64))
        fixed_any = False
    cap = int(spec["cap"])
    overlap_stop = float(spec["overlap"])
    work = Path(spec["work"])
    settings = spec["settings"]
    with _at(work):
        lib.softsaddle_seed(int(spec["seed"]) & 0xFFFFFFFF)
        lib.softsaddle_reset()
        initial = INITIAL_GRADIENT if settings["seed_mode"] == "gradient" else INITIAL_PREVIOUS
        seed_vector = None
        overlap = 0.0
        best_overlap = -1.0
        previous = None
        eigenvalue = float("inf")
        vector = np.zeros(x.size, dtype=np.float64)
        steps = 0
        error = None
        try:
            if initial == INITIAL_GRADIENT:
                _energy, gradient = potential.energy_forces(x)
                if _norm(gradient) < 1e-14:
                    initial = INITIAL_USER
                    seed_vector = np.zeros(x.size, dtype=np.float64)
                    seed_vector[0] = 1.0
            while potential.calls < cap:
                before = potential.calls
                eigenvalue, vector, _gradient, _iterations, _status, _energy = _minimum_mode(
                    lib,
                    potential,
                    x,
                    initial=initial,
                    iteration_limit=settings["iteration_limit"],
                    finite=settings["finite"],
                    tolerance=settings["tolerance"],
                    seed_vector=seed_vector,
                )
                steps += 1
                overlap = float(abs(np.dot(_unit(vector), reference)))
                if overlap >= overlap_stop:
                    break
                if previous is not None and _norm(vector - previous) < 1e-8:
                    break
                # A repeated estimate at one geometry that does not raise the
                # overlap is the same stall. Davidson rebuilds its Hessian
                # when the point has not moved, and the vector jitters.
                if overlap <= best_overlap + 1e-4:
                    break
                best_overlap = overlap
                if potential.calls == before:
                    break
                previous = vector.copy()
                initial = INITIAL_PREVIOUS
                seed_vector = None
        except RuntimeError as exc:
            error = str(exc)
        energy, gradient = potential.energy_forces(x)
        reasons: list[str] = []
        if error:
            reasons.append("crash")
        if not (overlap >= overlap_stop and potential.calls < cap and error is None):
            reasons.append("mode_overlap")
        if potential.calls >= cap:
            reasons.append("cap")
        success, failure = _outcome(reasons)
        record = _public(
            spec["name"],
            potential.calls,
            success,
            failure,
            reasons,
            energy=float(energy),
            force_norm=_norm(gradient),
            overlap=overlap,
            curvature=float(eigenvalue) if math.isfinite(eigenvalue) else None,
            converged=bool(overlap >= overlap_stop and potential.calls < cap and error is None),
            steps=steps,
            solver=kind,
            error=error,
        )
        if error is None and _finite_point(x):
            raw, vectors = _hessian_spectrum(potential, x)
            if fixed_any:
                eigenvalues, removed = list(raw), 0
            else:
                eigenvalues, removed = _vibrational_eigenvalues(x, raw, vectors)
            record["cartesian_index"] = negative_index(raw, 1e-6)
            record["hessian_index"] = negative_index(eigenvalues, 1e-6)
            record["rigid_modes_removed"] = int(removed)
            record["lowest_eigenvalue"] = float(eigenvalues[0]) if eigenvalues else None
            record["lowest_eigenvalues"] = [float(value) for value in eigenvalues[:8]]
            record["verify_calls"] = int(potential.verify_calls)
        return record


def _judge_saddle(payload: dict, climbed: dict, potential: HarnessSurface, calls_before_endpoints: int) -> dict:
    force_tol = float(payload["force_tol"])
    minimum = np.asarray(payload["minimum"], dtype=np.float64)
    min_energy = float(payload["min_energy"])
    tolerance = float(payload["link_tolerance"])
    barrier_max = float(payload["barrier_max"])
    cell = payload["cell"]
    search_cap = int(payload["search_cap"])
    reasons = ["force_norm"]
    record = {
        "converged": bool(climbed["converged"]),
        "curvature": climbed["eigenvalue"],
        "iterations": climbed["iterations"],
        "hessian_index": None,
        "barrier": None,
        "rmsd_plus": None,
        "rmsd_minus": None,
        "lambda_distance": None,
        "saddle_distance": None,
        "saddle_x": None,
        "novel_candidate": False,
        "error": climbed["error"],
    }
    point = np.asarray(climbed["x"], dtype=np.float64)
    finite = _finite_point(point)
    negative = (
        climbed["eigenvalue"] is not None
        and math.isfinite(float(climbed["eigenvalue"]))
        and float(climbed["eigenvalue"]) < 0.0
    )
    if climbed["error"]:
        reasons = ["crash"]
    elif climbed["converged"] and negative and finite:
        raw, _vectors = _hessian_spectrum(potential, point)
        index = negative_index(raw, 1e-6)
        record["hessian_index"] = int(index)
        record["lowest_eigenvalue"] = float(raw[0]) if raw else None
        record["verify_calls"] = int(potential.verify_calls)
        if index == 1 and float(climbed["force_norm"]) <= force_tol + 1e-10 and potential.calls < search_cap:
            mode = climbed["mode"] if climbed["mode"] is not None else np.zeros_like(point)
            axis = _unit(np.asarray(mode, dtype=np.float64))
            plus, _plus_ok = _lbfgs_minimize(potential, point + 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET)
            minus, _minus_ok = _lbfgs_minimize(potential, point - 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET)
            plus_rows = plus.reshape(-1, 3)
            minus_rows = minus.reshape(-1, 3)
            minimum_rows = minimum.reshape(-1, 3)
            plus_rmsd = _cartesian_rmsd(plus_rows, minimum_rows)
            minus_rmsd = _cartesian_rmsd(minus_rows, minimum_rows)
            links = connected_endpoints(minimum_rows, plus_rows, minus_rows, cell, tolerance)
            barrier = float(climbed["energy"] - min_energy)
            record["barrier"] = barrier
            record["rmsd_plus"] = plus_rmsd
            record["rmsd_minus"] = minus_rmsd
            record["per_atom_max_plus"] = per_atom_max(plus_rows, minimum_rows, cell)
            record["per_atom_max_minus"] = per_atom_max(minus_rows, minimum_rows, cell)
            record["saddle_distance"] = _norm(point - minimum)
            record["saddle_x"] = point.copy()
            record["endpoint_plus"] = plus_rows.reshape(-1).copy()
            record["endpoint_minus"] = minus_rows.reshape(-1).copy()
            record["linked"] = bool(links)
            record["novel_candidate"] = True
            if climbed["x_lambda"] is not None:
                distance = _norm(np.asarray(climbed["x_lambda"], dtype=np.float64) - minimum)
                if distance > 0.0:
                    record["lambda_distance"] = distance
            reasons = []
            if barrier >= barrier_max:
                record["above_barrier"] = True
        else:
            reasons = ["hessian_index"] if index != 1 else ["force_norm"]
    elif not finite:
        reasons = ["divergence"]
    elif calls_before_endpoints >= int(payload["climb_budget"]) or potential.calls >= search_cap:
        reasons = ["cap", "force_norm"]
    if potential.calls >= search_cap and "cap" not in reasons and reasons:
        reasons.append("cap")
    success, failure = _outcome(reasons)
    if success and potential.calls >= search_cap:
        success, failure = False, "cap"
        reasons = ["cap"]
    record["reasons"] = reasons
    record["success"] = success
    record["failure"] = failure
    record["force_calls"] = int(potential.calls)
    record["force_norm"] = float(climbed["force_norm"]) if math.isfinite(float(climbed["force_norm"])) else None
    record["energy"] = float(climbed["energy"]) if climbed["energy"] is not None else None
    return record


def _saddle_worker(payload: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    positions = np.asarray(payload["positions"], dtype=np.float64)
    fixed = np.asarray(payload["fixed"], dtype=bool)
    cell = np.asarray(payload["cell"], dtype=np.float64)
    free = np.flatnonzero(~fixed)
    potential = HarnessSurface(_bind(potential_from_problem({"potential": payload["potential"]}), positions, free, cell))
    lib = _load(str(payload["settings"]["solver"]))
    work = Path(payload["work"])
    with _at(work):
        lib.softsaddle_seed(int(payload["seed"]) & 0xFFFFFFFF)
        lib.softsaddle_reset()
        climbed = climb(
            lib,
            potential,
            np.asarray(payload["x0"], dtype=np.float64),
            settings=payload["settings"],
            force_tol=float(payload["force_tol"]),
            minimum_energy=float(payload["min_energy"]),
            climb_budget=int(payload["climb_budget"]),
        )
        judged = _judge_saddle(payload, climbed, potential, int(potential.calls))
    judged["verify_calls"] = int(potential.verify_calls)
    judged["name"] = payload["name"]
    judged["index"] = int(payload["index"])
    for key in ("saddle_x", "endpoint_plus", "endpoint_minus"):
        if judged.get(key) is not None:
            judged[key] = np.asarray(judged[key], dtype=np.float64).reshape(-1).tolist()
    return judged


def _apply_novelty(judged: dict, found: list[np.ndarray], connect: float) -> dict:
    reasons = list(judged["reasons"])
    success = bool(judged["success"])
    point = judged.get("saddle_x")
    if judged.get("novel_candidate") and point is not None:
        saddle = np.asarray(point, dtype=np.float64)
        novel = all(_cartesian_rmsd(saddle.reshape(-1, 3), old.reshape(-1, 3)) > connect for old in found)
        if novel:
            found.append(saddle)
            judged["accepted_novel"] = True
        else:
            if "duplicate" not in reasons:
                reasons.append("duplicate")
            success = False
        if success and reasons:
            success = False
    failure = None if success and not reasons else classify_failure(reasons or ["divergence"])
    if success:
        failure = None
        reasons = []
    judged = dict(judged)
    judged["reasons"] = reasons
    judged["success"] = bool(success and failure is None)
    judged["failure"] = failure
    return judged


def _radius_update(radius: float, batch: list[dict], lam: list[float], sad: list[float], settings: dict) -> float:
    connected = [row for row in batch if row.get("radius_hit")]
    efficiency = len(connected) / float(settings["batch"])
    if not connected and not sad:
        return radius
    for row in connected:
        if row.get("lambda_distance"):
            lam.append(float(row["lambda_distance"]))
        if row.get("saddle_distance"):
            sad.append(float(row["saddle_distance"]))
    if not lam:
        return radius
    proposed = float(np.mean(lam))
    if settings["mix_radius"]:
        proposed = 0.5 * (proposed + radius)
    if efficiency >= float(settings["keep_efficiency"]):
        return radius
    if not math.isfinite(proposed) or proposed <= 0.0:
        return radius
    return float(proposed)


def _replay_radius(records: list[dict], settings: dict) -> tuple[float, list[float], list[float]]:
    radius = float(settings["hyper_rad0"])
    lam: list[float] = []
    sad: list[float] = []
    batch = int(settings["batch"])
    complete = len(records) - (len(records) % batch)
    for start in range(0, complete, batch):
        radius = _radius_update(radius, records[start : start + batch], lam, sad, settings)
    return radius, lam, sad


_ARCHIVE_ENTRY = {
    "softsaddle/saddle-davidson-sm50": "adaptive_HSphere-lbfgs-davidson-sm50",
    "softsaddle/saddle-lanczos-sm0": "adaptive_HSphere-lbfgs-lanczos-sm0",
    "softsaddle/saddle-lanczos-sm50": "adaptive_HSphere-lbfgs-lanczos-sm50",
}


def hypersphere_path(repetition: int = 1) -> Path:
    return _repo_root() / "softsaddle" / "data" / f"hypersphere_5000_rep{int(repetition)}.npz"


def hypersphere(repetition: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Row-normal displacements and the archived farthest-point order."""
    path = hypersphere_path(repetition)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    packed = np.load(path)
    return np.asarray(packed["V"], dtype=np.float64), np.asarray(packed["order"], dtype=np.int64)


def hypersphere_sha(repetition: int = 1) -> str:
    packed = np.load(hypersphere_path(repetition))
    if "source_sha256" in packed.files:
        return str(packed["source_sha256"])
    if int(repetition) == 1:
        return HYPER_SOURCE
    raise KeyError(f"hypersphere repetition {repetition} has no source hash")


def _search_seed(repetition: int, index: int) -> int:
    """Library seed for one search.

    Repetition 1 keeps the published seeds: search 1 is 20261016.
    Later repetitions shift by 100000 so a direction index cannot collide.
    """
    if int(repetition) == 1:
        return 20261015 + int(index) + 1
    return 20261015 + (int(repetition) - 1) * 100_000 + int(index) + 1


def _wanted_repetitions(total: int) -> list[int]:
    raw = os.environ.get("SOFTSADDLE_REPETITIONS", "").strip()
    if not raw:
        return list(range(1, int(total) + 1))
    chosen: list[int] = []
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, stop_text = piece.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"repetition range {piece} is reversed")
            chosen.extend(range(start, stop + 1))
        else:
            chosen.append(int(piece))
    unique: list[int] = []
    for item in chosen:
        if item < 1 or item > int(total):
            raise ValueError(f"repetition {item} is outside 1..{total}")
        if item not in unique:
            unique.append(item)
    if not unique:
        raise ValueError("SOFTSADDLE_REPETITIONS did not select a repetition")
    return unique


def _rep_directory(work: Path, repetition: int) -> Path:
    return work / f"rep-{int(repetition):02d}"


def _parse_rep_comment(line: str) -> dict:
    parts = line.split()
    if len(parts) < 3 or parts[0] != "#" or parts[1] != "rep":
        raise ValueError(f"not a repetition comment: {line!r}")

    def grab(key: str, cast):
        if key not in parts:
            return None
        return cast(parts[parts.index(key) + 1])

    return {
        "repetition": int(parts[2]),
        "force_calls": grab("force_calls", int),
        "unique_saddles": grab("unique_saddles", int),
        "fraction_connected": grab("fraction_connected", float),
        "total_searches": grab("total_searches", int),
        "connected": grab("connected", int),
        "minimize_calls": grab("minimize_calls", int),
        "minimum_converged": grab("minimum_converged", int),
        "seed": grab("seed", int),
        "hypersphere": grab("hypersphere", str),
        "comment": line,
    }


def _published_comments(output: Path) -> dict[int, dict]:
    if not output.is_file():
        return {}
    found = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.startswith("# rep "):
            parsed = _parse_rep_comment(line)
            found[int(parsed["repetition"])] = parsed
    return found


def _published_records(output: Path, repetition: int) -> list[dict]:
    path = output.with_suffix(".starts.jsonl")
    if not path.is_file():
        return []
    prefix = f"rep{int(repetition)}-search"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("start", "")).startswith(prefix):
            rows.append(row)

    def _index(row: dict) -> int:
        return int(str(row["start"]).rsplit("search", 1)[1])

    rows.sort(key=_index)
    return rows


def _rep_comment(
    repetition: int,
    records: list[dict],
    unique: int,
    connected: int,
    minimize_calls: int,
    min_ok: bool,
    seed: int,
    sha: str,
    fraction: float,
) -> str:
    total = sum(int(row["force_calls"]) for row in records)
    return (
        f"# rep {repetition} force_calls {total} unique_saddles {unique} "
        f"fraction_connected {fraction} total_searches {len(records)} "
        f"connected {connected} minimize_calls {minimize_calls} "
        f"minimum_converged {int(bool(min_ok))} seed {seed} hypersphere {sha}"
    )


def matlab_fraction(unique_sp: int, connected: int, searches: int) -> float:
    """``1 - (unique_sp_sofar - n_connected_sp) / N_searches``."""
    if searches <= 0:
        raise ValueError("searches must be positive")
    return 1.0 - (int(unique_sp) - int(connected)) / float(searches)


def matlab_benchmark(reps: list[dict]) -> dict:
    """Means of the harness repetition comments.

    ``force_calls`` is the repetition total. ``unique_saddles`` is the
    catalog of linked novel saddles, including barriers at or above 1.5 eV
    (the archive's ``n_connected_sp``). ``fraction_connected`` is the done.py
    ratio stored on the comment. ``force_calls_mean_error`` is the sample
    standard deviation.
    """
    if len(reps) < 2:
        raise ValueError("the sample standard deviation needs at least two repetitions")
    totals = [float(row["force_calls"]) for row in reps]
    return {
        "force_calls": statistics.fmean(totals),
        "force_calls_mean_error": statistics.stdev(totals),
        "fraction_connected": statistics.fmean(float(row["fraction_connected"]) for row in reps),
        "total_searches": statistics.fmean(int(row["total_searches"]) for row in reps),
        "unique_saddles": statistics.fmean(float(row["unique_saddles"]) for row in reps),
    }


def _summary_complete(summary: dict, max_searches: int, goal: int) -> bool:
    searches = int(summary.get("total_searches") or 0)
    connected = int(summary.get("connected") or 0)
    return connected >= int(goal) or searches >= int(max_searches)


def _read_done(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_done(directory: Path, summary: dict, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_starts(directory / "records.jsonl", records)
    (directory / "done.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "partial.jsonl").unlink(missing_ok=True)


def _load_done_records(directory: Path) -> list[dict]:
    rows = []
    path = directory / "records.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _campaign_dat(campaign: Path, template_id: str) -> Path:
    """Result file for one saddle template, under either published stem."""
    short = campaign / f"{_campaign_stem(template_id)}.dat"
    if short.is_file():
        return short
    needle = str(template_id).split("/", 1)[1]
    found = sorted(campaign.glob(f"softsaddle-*{needle}.dat"))
    return found[0] if found else short


def _campaign_stem(template_id: str, version: str = "2019.04") -> str:
    short = str(template_id).split("/", 1)[1]
    return f"softsaddle-{version}-{short}"


def _fmt_metric(value: float) -> str:
    return f"{value:.10g}"


def write_archive_comparison(root: Path, campaign: Path | None = None) -> Path:
    """Write the harness-versus-archive table for the three saddle templates."""
    destination = campaign or (root / "results" / "campaign")
    destination.mkdir(parents=True, exist_ok=True)
    wanted = set(_ARCHIVE_ENTRY.values())
    oracle = {}
    for line in (root / "oracle" / "entries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("suite") == "saddle-search" and row.get("entry") in wanted:
            oracle[row["entry"]] = row
    missing = wanted.difference(oracle)
    if missing:
        raise FileNotFoundError(f"oracle is missing {sorted(missing)}")
    lines = [
        "SoftSaddle Pt-island repetitions, set beside the archived website entries.",
        "",
        "Archive rows are the benchmark.dat values in oracle/entries.jsonl for",
        "website/saddle-search/pt-island/adaptive_HSphere-*. The oracle dates those",
        "files April 2019. force_calls there counts MATLAB evaluations",
        "(m_minimizer_func_calls + sp_func_calls). force_calls_mean_error is MATLAB",
        "std, the sample standard deviation. fraction_connected is the mean of",
        "1 - (unique_sp_sofar - n_connected_sp) / N_searches. unique_saddles is the",
        "mean of n_connected_sp. total_searches is the mean of N_searches.",
        "The reduction is softsaddle/upstream/test_run/get_data_for_web_benchmark.m.",
        "",
        "Harness rows are the repetition comments. force_calls is the sum of harness",
        "calls over the searches. The initial minimization is outside the sum.",
        "unique_saddles counts linked novel saddles, including a barrier at or above",
        "1.5 eV. fraction_connected is the done.py ratio. The link is the eOn r2205",
        "per-atom test and novelty is find_repeat. A repetition stops at 27 saddles",
        "below 1.5 eV, or at the 5000 archived hypersphere directions.",
        "",
        f"{'template':<40} {'source':<8} {'force_calls':>16} {'mean_error':>16} "
        f"{'fraction':>12} {'searches':>12} {'unique_saddles':>16}",
    ]
    detail = []
    for template_id, entry in _ARCHIVE_ENTRY.items():
        archived = oracle[entry]["fields"]
        lines.append(
            f"{template_id:<40} {'archive':<8} "
            f"{archived['force_calls']:>16} {archived['force_calls_mean_error']:>16} "
            f"{archived['fraction_connected']:>12} {archived['total_searches']:>12} "
            f"{archived['unique_saddles']:>16}"
        )
        lines.append(f"{'':<40} {'sha256':<8} {oracle[entry]['sha256']}")
        dat = _campaign_dat(destination, template_id)
        comments = sorted(_published_comments(dat).values(), key=lambda row: int(row["repetition"]))
        ready = len(comments) >= 10 and all(row.get("connected") is not None for row in comments)
        if not ready:
            lines.append(f"{template_id:<40} {'harness':<8} {'pending':>16}")
            continue
        pooled = matlab_benchmark(comments)
        lines.append(
            f"{template_id:<40} {'harness':<8} "
            f"{_fmt_metric(pooled['force_calls']):>16} {_fmt_metric(pooled['force_calls_mean_error']):>16} "
            f"{_fmt_metric(pooled['fraction_connected']):>12} {_fmt_metric(pooled['total_searches']):>12} "
            f"{_fmt_metric(pooled['unique_saddles']):>16}"
        )
        detail.append((template_id, comments, pooled))
    lines.extend(
        [
            "",
            "Per repetition. unique_sp is the linked novel count, including barriers",
            "at or above 1.5 eV. connected is the count below 1.5 eV.",
            "matlab_fraction is 1 - (unique_sp - connected) / searches.",
            "The # rep fraction_connected field is the done.py ratio.",
        ]
    )
    for template_id, comments, _pooled in detail:
        lines.append("")
        lines.append(template_id)
        lines.append(
            f"{'rep':>4} {'force_calls':>12} {'unique_sp':>10} {'connected':>10} "
            f"{'searches':>10} {'matlab_fraction':>16}"
        )
        for row in comments:
            fraction = matlab_fraction(
                int(row["unique_saddles"]),
                int(row["connected"]),
                int(row["total_searches"]),
            )
            lines.append(
                f"{int(row['repetition']):4d} {int(row['force_calls']):12d} "
                f"{int(row['unique_saddles']):10d} {int(row['connected']):10d} "
                f"{int(row['total_searches']):10d} {fraction:16.6f}"
            )
    path = destination / "softsaddle-pt-island-repetitions.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _displace(minimum: np.ndarray, direction: np.ndarray, radius: float, atoms: int) -> np.ndarray:
    width = 3 * int(atoms)
    x0 = np.array(minimum, dtype=np.float64, copy=True)
    x0[:width] = minimum[:width] + float(radius) * np.asarray(direction, dtype=np.float64)
    return x0


def _load_partial(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_partial(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        handle.flush()


def _jsonable(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _run_saddle(root: Path, problem: dict, template: dict, *, output: Path, force_tol: float, work: Path, workers: int) -> dict:
    settings = _settings(template)
    template_budget = int(settings["max_searches"])
    override = os.environ.get("SOFTSADDLE_MAX_SEARCHES", "").strip()
    run_limit = int(override) if override else template_budget
    if override:
        print(f"{template['id']} SOFTSADDLE_MAX_SEARCHES {run_limit}", flush=True)
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    from .rgsaddle_minmode import _stored

    positions, fixed, cell = load_con(_stored(root, con_path))
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    tolerance = link_tolerance(problem)
    climb_budget = int(settings["cg_iter_max"])
    search_cap = climb_budget + _SADDLE_CLIMB_OVERSHOOT + 2 * _SADDLE_ENDPOINT_BUDGET
    total_reps = int(settings["repetitions"])
    selected = _wanted_repetitions(total_reps)
    free = np.flatnonzero(~fixed)
    raw = potential_from_problem(problem)
    work.mkdir(parents=True, exist_ok=True)
    minimum, min_energy, minimize_calls, min_ok = _campaign_minimum(
        work, raw, positions, free, cell, force_tol
    )
    print(
        f"{template['id']} repetitions {selected} budget {template_budget} "
        f"minimum_calls {minimize_calls} minimum_energy {min_energy:.6f}",
        flush=True,
    )

    def _done_summary(repetition: int) -> dict | None:
        done = _read_done(_rep_directory(work, repetition) / "done.json")
        if done is not None and _summary_complete(done, template_budget, goal):
            return done
        return None

    def _import_published(repetition: int) -> None:
        if _done_summary(repetition) is not None:
            return
        directory = _rep_directory(work, repetition)
        if _load_partial(directory / "partial.jsonl"):
            return
        published = _published_comments(output).get(repetition)
        if published is None or not _summary_complete(published, template_budget, goal):
            return
        records = _published_records(output, repetition)
        if len(records) != int(published["total_searches"]):
            print(
                f"{template['id']} rep {repetition} published "
                f"{len(records)} starts, comment says {published['total_searches']}",
                flush=True,
            )
            return
        summary = dict(published)
        summary["finished"] = True
        summary["reached"] = int(summary["connected"]) >= goal
        _write_done(directory, summary, records)
        print(f"{template['id']} imported published repetition {repetition}", flush=True)

    for repetition in range(1, total_reps + 1):
        _import_published(repetition)

    def _run_repetition(repetition: int, pool) -> None:
        directory = _rep_directory(work, repetition)
        directory.mkdir(parents=True, exist_ok=True)
        partial_path = directory / "partial.jsonl"
        saved = _load_partial(partial_path)
        prefix = f"rep{repetition}-search"
        for row in saved:
            if not str(row.get("start", "")).startswith(prefix):
                raise RuntimeError(f"{partial_path} holds {row.get('start')}, expected {prefix}")
        vectors, order = hypersphere(repetition)
        sha = hypersphere_sha(repetition)
        run_cap = min(run_limit, template_budget, int(order.size))
        radius, _lam, _sad = _replay_radius(saved, settings)
        catalog = ProcessCatalog(cell, tolerance=tolerance, barrier_max=barrier_max)
        for row in saved:
            judged_row = row.get("per_atom_max_plus") is not None
            catalog.observe(
                judged=judged_row,
                linked=judged_row and "not_connected" not in (row.get("reasons") or []),
                structure=row.get("saddle_x"),
                barrier=row.get("barrier"),
            )
        print(
            f"{template['id']} rep {repetition} resume searches {catalog.searches} "
            f"connected {catalog.qualifying} unique {catalog.unique} radius {radius:.4f} hypersphere {sha}",
            flush=True,
        )
        batch = int(settings["batch"])

        def _batch(start: int, stop: int, radius_now: float) -> list[dict]:
            payloads = []
            for index in range(start, stop):
                direction = vectors[int(order[index])]
                payloads.append(
                    {
                        "name": f"{prefix}{index + 1}",
                        "index": index,
                        "positions": positions,
                        "fixed": fixed,
                        "cell": cell,
                        "potential": problem["potential"],
                        "x0": _displace(minimum, direction, radius_now, settings["displace_atoms"]),
                        "minimum": minimum,
                        "min_energy": min_energy,
                        "settings": settings,
                        "force_tol": force_tol,
                        "link_tolerance": tolerance,
                        "barrier_max": barrier_max,
                        "search_cap": search_cap,
                        "climb_budget": climb_budget,
                        "seed": _search_seed(repetition, index),
                        "work": str(work / f"r{repetition}-s{index + 1:05d}"),
                    }
                )
            return payloads

        while catalog.qualifying < goal and len(saved) < run_cap:
            start = len(saved)
            into_batch = batch - (start % batch) if start % batch else batch
            stop = min(start + into_batch, run_cap)
            payloads = _batch(start, stop, radius)
            if pool is None or len(payloads) == 1:
                judged_rows = [_saddle_worker(item) for item in payloads]
            else:
                judged_rows = list(pool.map(_saddle_worker, payloads))
            judged_rows.sort(key=lambda row: int(row["index"]))
            for judged in judged_rows:
                if catalog.qualifying >= goal:
                    break
                was_judged = judged.get("per_atom_max_plus") is not None
                decision = catalog.observe(
                    judged=was_judged,
                    linked=bool(judged.get("linked")),
                    structure=judged.get("saddle_x"),
                    barrier=judged.get("barrier"),
                )
                if was_judged:
                    reasons = list(decision["reasons"])
                    success = bool(decision["success"])
                else:
                    reasons = list(judged.get("reasons") or ["divergence"])
                    success = False
                failure = None if success else classify_failure(reasons or ["divergence"])
                public = _public(
                    judged["name"],
                    judged["force_calls"],
                    success,
                    failure,
                    reasons,
                    verify_calls=int(judged.get("verify_calls") or 0),
                    energy=judged.get("energy"),
                    force_norm=judged.get("force_norm"),
                    hessian_index=judged.get("hessian_index"),
                    barrier=judged.get("barrier"),
                    rmsd_plus=judged.get("rmsd_plus"),
                    rmsd_minus=judged.get("rmsd_minus"),
                    per_atom_max_plus=judged.get("per_atom_max_plus"),
                    per_atom_max_minus=judged.get("per_atom_max_minus"),
                    curvature=judged.get("curvature"),
                    converged=bool(judged.get("converged")),
                    lambda_distance=judged.get("lambda_distance"),
                    saddle_distance=judged.get("saddle_distance"),
                    iterations=judged.get("iterations"),
                    solver=settings["solver"],
                    radius=radius,
                    repetition=repetition,
                    error=judged.get("error"),
                    radius_hit=bool(decision["good"]),
                    linked=bool(judged.get("linked")),
                )
                partial = dict(public)
                partial["saddle_x"] = judged.get("saddle_x") if was_judged else None
                partial["endpoint_plus"] = judged.get("endpoint_plus")
                partial["endpoint_minus"] = judged.get("endpoint_minus")
                saved.append(partial)
                _append_partial(partial_path, partial)
            if len(saved) % batch == 0 or len(saved) >= run_cap or catalog.qualifying >= goal:
                radius, _lam, _sad = _replay_radius(saved, settings)
            (work / "progress.txt").write_text(
                f"rep {repetition} searches {len(saved)} connected {catalog.qualifying} radius {radius:.4f}\n",
                encoding="utf-8",
            )
            print(
                f"{template['id']} rep {repetition} searches {len(saved)} "
                f"connected {catalog.qualifying} unique {catalog.unique} radius {radius:.4f} "
                f"last {saved[-1]['force_calls']} {saved[-1]['failure']}",
                flush=True,
            )
        reached = catalog.qualifying >= goal
        finished = reached or len(saved) >= run_cap
        if not finished:
            print(
                f"{template['id']} rep {repetition} stopped at {len(saved)} "
                f"before {template_budget}",
                flush=True,
            )
            return
        records = []
        for row in saved:
            public = {key: value for key, value in row.items() if key != "saddle_stored"}
            records.append(public)
        comment = _rep_comment(
            repetition,
            records,
            catalog.unique,
            catalog.qualifying,
            minimize_calls,
            bool(min_ok),
            _search_seed(repetition, 0),
            sha,
            catalog.fraction_connected,
        )
        summary = _parse_rep_comment(comment)
        summary["finished"] = True
        summary["reached"] = reached
        summary["comment"] = comment
        _write_done(directory, summary, records)
        print(f"{template['id']} finished {comment}", flush=True)

    pool = None
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        for repetition in selected:
            if _done_summary(repetition) is not None:
                print(f"{template['id']} repetition {repetition} already finished", flush=True)
                continue
            _run_repetition(repetition, pool)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)

    finished_rows = []
    for repetition in range(1, total_reps + 1):
        summary = _done_summary(repetition)
        if summary is None:
            done = [row["repetition"] for row in finished_rows]
            print(
                f"{template['id']} withholding {output} until repetitions 1..{total_reps} "
                f"finish; done {done}",
                flush=True,
            )
            return {
                "force_calls": None,
                "nfailed": total_reps - len(finished_rows),
                "starts": 0,
                "template": template["id"],
                "problem": problem["id"],
                "connected": None,
            }
        finished_rows.append(summary)
    return _publish_saddle(
        root,
        problem,
        template,
        output=output,
        settings=settings,
        search_cap=search_cap,
        goal=goal,
        summaries=finished_rows,
        work=work,
    )


def _campaign_minimum(work, raw, positions, free, cell, force_tol):
    path = work / "minimum.json"
    if path.is_file():
        header = json.loads(path.read_text(encoding="utf-8"))
        return (
            np.asarray(header["minimum"], dtype=np.float64),
            float(header["min_energy"]),
            int(header["minimize_calls"]),
            bool(header["min_ok"]),
        )
    potential = HarnessSurface(_bind(raw, positions, free, cell))
    minimum, min_ok = _lbfgs_minimize(potential, positions[free].reshape(-1).copy(), force_tol, 5000)
    min_energy, _gradient = potential.energy_forces(minimum)
    payload = {
        "minimum": minimum.tolist(),
        "min_energy": float(min_energy),
        "minimize_calls": int(potential.calls),
        "min_ok": bool(min_ok),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return minimum, float(min_energy), int(potential.calls), bool(min_ok)


def _publish_endpoints(output: Path, records: list[dict], work: Path, cell, free) -> list[dict]:
    """Pack endpoint minima and drop the coordinate arrays from the starts."""
    header = json.loads((work / "minimum.json").read_text(encoding="utf-8"))
    minimum = np.asarray(header["minimum"], dtype=np.float64).reshape(-1, 3)
    archive = EndpointArchive(output.with_name(output.stem + ".endpoints.npz"), int(np.asarray(free).size))
    archive.reset(stored_cell(cell), np.asarray(free, dtype=np.int32))
    cleaned = []
    for row in records:
        plus = row.get("endpoint_plus")
        minus = row.get("endpoint_minus")
        plus_rows = None if plus is None else np.asarray(plus, dtype=np.float64).reshape(-1, 3)
        minus_rows = None if minus is None else np.asarray(minus, dtype=np.float64).reshape(-1, 3)
        status = archive.add(minimum, plus_rows, minus_rows)
        public = {
            key: value
            for key, value in row.items()
            if key not in {"endpoint_plus", "endpoint_minus", "saddle_x", "saddle_stored"}
        }
        stamp_endpoint(public, archive.name, archive.count - 1, status)
        cleaned.append(public)
    archive.save()
    archive.drop_log()
    return cleaned


def _publish_saddle(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    settings: dict,
    search_cap: int,
    goal: int,
    summaries: list[dict],
    work: Path,
) -> dict:
    records = []
    comments = []
    failed_reps = 0
    connected = 0
    for summary in summaries:
        repetition = int(summary["repetition"])
        rows = _load_done_records(_rep_directory(work, repetition))
        if len(rows) != int(summary["total_searches"]):
            raise RuntimeError(
                f"{template['id']} rep {repetition} stored {len(rows)} starts, "
                f"summary says {summary['total_searches']}"
            )
        records.extend(rows)
        comments.append(str(summary["comment"]))
        connected += int(summary["connected"])
        if not summary.get("reached"):
            failed_reps += 1
    from .rgsaddle_minmode import _stored

    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    _positions, fixed, cell = load_con(_stored(root, con_path))
    records = _publish_endpoints(output, records, work, cell, np.flatnonzero(~fixed))
    totals = [int(summary["force_calls"]) for summary in summaries]
    pooled = matlab_benchmark(summaries)
    text = "\n".join(comments) + "\n"
    text += (
        f"force_calls {statistics.fmean(totals)}\n"
        f"force_calls_mean_error {statistics.stdev(totals)}\n"
        f"force_calls_min {min(totals)}\n"
        f"force_calls_max {max(totals)}\n"
        f"unique_saddles {statistics.fmean(int(row['unique_saddles']) for row in summaries)}\n"
        f"fraction_connected {statistics.fmean(float(row['fraction_connected']) for row in summaries)}\n"
        f"total_searches {statistics.fmean(int(row['total_searches']) for row in summaries)}\n"
        f"benchmark_force_calls {pooled['force_calls']}\n"
        f"benchmark_force_calls_mean_error {pooled['force_calls_mean_error']}\n"
        f"benchmark_fraction_connected {pooled['fraction_connected']}\n"
        f"benchmark_total_searches {pooled['total_searches']}\n"
        f"benchmark_unique_saddles {pooled['unique_saddles']}\n"
    )
    ok = [int(row["force_calls"]) for row in records if row["success"]]
    if ok:
        text += f"success_force_calls {half_even(sum(ok) / len(ok))}\n"
        text += f"success_force_calls_min {min(ok)}\n"
        text += f"success_force_calls_median {float(np.median(ok))}\n"
        text += f"success_force_calls_max {max(ok)}\n"
    metrics = score_v2(
        [
            {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
            for row in records
        ],
        search_cap,
    )
    text += (
        f"code {template.get('code', 'softsaddle')}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"repetitions {len(summaries)}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(records)}\n"
        f"par2_mean {metrics['par2_mean']}\n"
        f"successes {metrics['successes']}\n"
        f"wilson_low {metrics['wilson_low']}\n"
        f"wilson_high {metrics['wilson_high']}\n"
        f"par2_median {metrics['median']}\n"
        f"n_cap {metrics['n_cap']}\n"
        f"n_wrong_stationary {metrics['n_wrong_stationary']}\n"
        f"n_divergence {metrics['n_divergence']}\n"
        f"n_crash {metrics['n_crash']}\n"
        f"v2_nfailed {metrics['nfailed']}\n"
        f"contributor Rohit Goswami\n"
        f"date {date.today().isoformat()}\n"
        f"host {os.uname().nodename}\n"
    )
    if template.get("note"):
        text += f"note {template['note']}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    starts_body = "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in records)
    _atomic_write(output, text)
    _atomic_write(output.with_suffix(".starts.jsonl"), starts_body)
    provenance = collect_provenance(
        code=str(template.get("code", "softsaddle")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={
            "template": template["id"],
            "problem": problem["id"],
            "cap": search_cap,
            "solver": settings["solver"],
            "hypersphere_sha256": [hypersphere_sha(int(row["repetition"])) for row in summaries],
            "repetitions": len(summaries),
        },
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "measured",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "One row is one hypersphere displacement. Each repetition loads one "
            "archived 5000-direction set and starts again at radius 0.1. The cap "
            "is the climb budget, 64 calls of overshoot, and the two endpoint "
            "minimizations. The initial minimization is not in that count."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code", "softsaddle"),
                "algorithm": template.get("algorithm", ""),
                "code_version": template.get("code_version", ""),
                "commit": template.get("commit", ""),
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": search_cap,
                "provenance": provenance,
                "metrics": metrics,
                "starts": records,
            }
        ],
    }
    _atomic_write(
        output.with_suffix(".v2.json"),
        json.dumps(document, indent=2, sort_keys=True) + "\n",
    )
    table = write_archive_comparison(root, output.parent)
    print(f"wrote {table}", flush=True)
    return {
        "force_calls": statistics.fmean(totals),
        "nfailed": failed_reps,
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "connected": connected,
    }


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def mode_spec(root: Path, problem: dict, template: dict, row: dict, cap: int, work: Path, index: int) -> dict:
    """One lowest-mode start. The geometry index selects the library seed."""
    settings = _settings(template)
    positions, fixed, cell = load_con(Path(row["path"]))
    reference = None
    if row.get("mode"):
        full = read_mode(Path(row["mode"]).read_text(encoding="utf-8", errors="replace"))
        reference = full.reshape(-1, 3)[~fixed].reshape(-1)
    return {
        "name": row["name"],
        "positions": positions,
        "fixed": fixed,
        "cell": cell,
        "potential": problem.get("potential") or {"kind": "lennard-jones", "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0},
        "reference": reference,
        "cap": cap,
        "overlap": float((problem.get("sample") or {}).get("overlap", 0.99)),
        "solver": settings["solver"],
        "settings": settings,
        "seed": 20261015 + index,
        "work": str(work / f"mode-{row['name']}"),
    }


def _run_modes(root: Path, problem: dict, template: dict, *, output: Path, cap: int, work: Path, workers: int) -> dict:
    rows = _geometries(root, problem, work)
    expected = int((problem.get("sample") or {}).get("starts") or len(rows))
    if len(rows) != expected:
        raise RuntimeError(f"{problem['id']} has {len(rows)} geometries, the problem states {expected}")
    specs = [mode_spec(root, problem, template, row, cap, work, index) for index, row in enumerate(rows)]
    if workers <= 1:
        records = [run_mode_point(spec) for spec in specs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            records = list(pool.map(run_mode_point, specs))
    ok = [int(row["force_calls"]) for row in records if row["success"]]
    lines = []
    if ok:
        lines.append(f"force_calls {half_even(sum(ok) / len(ok))}")
        lines.append(f"force_calls_min {min(ok)}")
        lines.append(f"force_calls_median {float(np.median(ok))}")
        lines.append(f"force_calls_max {max(ok)}")
    metrics = score_v2(
        [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records],
        cap,
    )
    lines.append(f"nfailed {metrics['nfailed']}")
    lines.append(f"par2_mean {metrics['par2_mean']:.10f}")
    lines.append(f"successes {metrics['successes']}")
    lines.append(f"wilson_low {metrics['wilson_low']:.10f}")
    lines.append(f"wilson_high {metrics['wilson_high']:.10f}")
    lines.append(f"par2_median {metrics['median']:.10f}")
    lines.append(f"n_cap {metrics['n_cap']}")
    lines.append(f"n_wrong_stationary {metrics['n_wrong_stationary']}")
    lines.append(f"n_divergence {metrics['n_divergence']}")
    lines.append(f"n_crash {metrics['n_crash']}")
    lines.append("counter harness")
    lines.append(f"code {template.get('code', 'softsaddle')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append(f"harness_calls {sum(int(row['force_calls']) for row in records)}")
    if len(records) < expected:
        lines.append("partial true")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = collect_provenance(
        code=str(template.get("code", "softsaddle")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={
            "template": template["id"],
            "problem": problem["id"],
            "cap": cap,
            "solver": str(_section(template, "Main").get("solver", "lanczos")),
        },
    )
    public = []
    for record in records:
        copied = _jsonable(record)
        copied["code"] = provenance["code"]
        copied["code_version"] = provenance["code_version"]
        copied["commit"] = provenance["commit"]
        copied["tuning"] = template.get("tuning", "")
        copied["template"] = template["id"]
        public.append(copied)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    return {
        "force_calls": half_even(sum(ok) / len(ok)) if ok else None,
        "nfailed": metrics["nfailed"],
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
        "records": records,
    }


def run_template(root: Path, problem: dict, template: dict, *, output: Path, workers: int = 1, work: Path | None = None) -> dict:
    """Run one SoftSaddle template."""
    from .campaign import _check_stop

    solver = str(_section(template, "Main").get("solver", "lanczos"))
    if library_path(solver) is None:
        raise RuntimeError(f"softsaddle {solver} library is not built")
    cap = _check_stop(problem, template)
    optimizer = _section(template, "Optimizer")
    convergence = problem.get("convergence") or {}
    force_tol = float(optimizer.get("converged_force", convergence.get("force_norm", 1e-3)))
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    if problem["id"] == "saddle-search/pt-island":
        return _run_saddle(root, problem, template, output=output, force_tol=force_tol, work=scratch, workers=workers)
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    return _run_modes(root, problem, template, output=output, cap=cap, work=scratch, workers=workers)
