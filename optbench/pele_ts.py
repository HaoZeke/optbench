"""pele transition-state campaigns on the problem potential.

``findTransitionState`` is hybrid eigenvector following.
``findLowestEigenVector`` is the Rayleigh-Ritz lowest mode.
The potential is the problem pair. ``getEnergy`` and
``getEnergyGradient`` share one ``HarnessCounter`` that caches a
geometry, so the two reads are one force call.

A tsopt success is a point under the force cap, at the Euclidean force
stop, whose Cartesian Hessian has index 1. Free Lennard-Jones clusters
drop numerical rigid modes the same way the rgsaddle rows do. A fixed
substrate keeps every free-atom eigenvalue. The reference geometry
passed to ``judge_stationary`` is the converged point, so the energy
and Kabsch tests do not add a second target.

A lowest-eigenvector success is calls under the cap and absolute
overlap of at least 0.99 with the reference mode. The Hessian index is
recorded and does not decide that score.

The failure class is ``classify_outcome`` on the final energy, the
geometry, the true force, and the call count.

A saddle-search row is one ``draw_search_start`` displacement. The kick
is the initial eigenvector. The climb stops at 1000 force calls. Each
endpoint minimization has a budget of 2000. The PAR-2 cap is 5064. The
initial minimization is outside that count. The link is the eOn r2205
test and novelty is ``AKMCState.find_repeat``. Endpoint minima are
float16 deltas in the sibling ``.endpoints.npz``.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from .connection import connected_endpoints, link_tolerance, per_atom_max
from .counter import ForceCallCap, HarnessCounter
from .failures import classify_outcome
from .rgsaddle_minmode import _cartesian_rmsd, _geometries, _stored, _vibrational_eigenvalues
from .score import half_even, score_v2
from .stationary import judge_stationary, negative_index, numerical_hessian
from .surfaces import load_con, potential_from_problem, read_mode
from .v2 import collect_provenance, write_starts

PELE_VERSION = "0.1.2"
_SADDLE_ENDPOINT_BUDGET = 2000
_SADDLE_CLIMB_OVERSHOOT = 64
_SADDLE_MINIMIZE_BUDGET = 5000
_SADDLE_SEARCH_CAP = 5064


class HarnessPotential:
    """Problem pair behind the pele potential interface.

    ``cache`` is on. A second read of the same coordinates is not a new
    force call. ``cap`` raises ``ForceCallCap`` before the call that
    would pass it, and that call is not counted.
    """

    def __init__(self, evaluate, cap: int | None):
        self._evaluate = evaluate
        self.counter = HarnessCounter(cache=True, cap=cap)
        self.last: np.ndarray | None = None

    @property
    def calls(self) -> int:
        return self.counter.calls

    @property
    def verify_calls(self) -> int:
        return self.counter.verify_calls

    def verification(self):
        return self.counter.verification()

    def getEnergyGradient(self, coords):
        flat = np.asarray(coords, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in flat)
        fresh, cached = self.counter.charge(key)
        if not fresh:
            energy, gradient = cached
            return float(energy), np.asarray(gradient, dtype=np.float64)
        energy, gradient = self._evaluate(flat)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
        stored = (float(energy), tuple(float(value) for value in gradient))
        self.counter.remember(key, stored)
        self.last = flat.copy()
        return stored[0], gradient.copy()

    def getEnergy(self, coords) -> float:
        energy, _gradient = self.getEnergyGradient(coords)
        return float(energy)

    def energy_forces(self, coords):
        return self.getEnergyGradient(coords)


class OverlapStop:
    """Stop a Rayleigh-Ritz search when the absolute overlap exceeds ``tol``."""

    def __init__(self, mode: np.ndarray):
        vec = np.asarray(mode, dtype=np.float64).reshape(-1)
        self.vec = vec / np.linalg.norm(vec)

    def __call__(self, energy=None, gradient=None, tol=None, coords=None, **kwargs) -> bool:
        if coords is None or tol is None:
            return False
        vector = np.asarray(coords, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm < 1e-14:
            return False
        overlap = abs(float(np.dot(self.vec, vector) / norm))
        if not math.isfinite(overlap):
            return False
        return bool(overlap > float(tol))


def _silent_stdout():
    return contextlib.redirect_stdout(io.StringIO())


def _quiet() -> None:
    import logging

    for name in ("pele", "pele.optimize", "pele.connect", "pele.connect.findTS"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _assert_pele(expected: str) -> None:
    from importlib.metadata import version

    found = version("pele")
    if found != PELE_VERSION:
        raise RuntimeError(f"pele {found} is not {PELE_VERSION}")
    src = os.environ.get("PELE_SRC")
    if not src or not expected:
        return
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src, text=True).strip()
    if head != expected:
        raise RuntimeError(f"pele HEAD {head} does not match the template commit {expected}")


def _import_pele():
    _quiet()
    import pele.optimize  # noqa: F401
    import pele.transition_states  # noqa: F401

    return pele


def _mean_free_rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
    """RMSD averaged over every free atom.

    A one-atom hop is diluted by the rest of the island, so both
    endpoints of a real hop can fall inside 0.5.
    """
    delta = np.asarray(moved, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _unit(values: np.ndarray) -> np.ndarray:
    scale = float(np.linalg.norm(values))
    if not math.isfinite(scale) or scale < 1e-14:
        out = np.zeros_like(values, dtype=np.float64)
        out.reshape(-1)[0] = 1.0
        return out
    return np.asarray(values, dtype=np.float64) / scale


def _finite(point: np.ndarray) -> bool:
    array = np.asarray(point, dtype=np.float64).reshape(-1)
    return bool(array.size) and bool(np.all(np.isfinite(array)))


def _bind(raw, positions: np.ndarray, free: np.ndarray, cell: np.ndarray):
    base = np.array(positions, dtype=np.float64, copy=True)

    def evaluate(flat):
        pos = base.copy()
        pos[free] = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        energy, gradient = raw(pos, cell)
        return float(energy), np.ascontiguousarray(gradient[free].reshape(-1), dtype=np.float64)

    return evaluate


def _orthog_value(name: str):
    if name == "none":
        return None
    if name == "translations-rotations":
        return 0
    raise ValueError(f"unknown orthog {name}")


def _l2_stop(energy, gradient, tol, coords=None, **kwargs) -> bool:
    del energy, coords, kwargs
    grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(grad)):
        return False
    return float(np.linalg.norm(grad)) <= float(tol)


def _minimize(evaluate, x0: np.ndarray, force_tol: float, budget: int) -> tuple[np.ndarray, bool, int, float]:
    """L-BFGS to the Euclidean force stop. The budget is force calls."""
    from pele.optimize import mylbfgs
    from pele.optimize.optimization_exceptions import LineSearchError

    pot = HarnessPotential(evaluate, budget)
    point = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    converged = False
    try:
        with _silent_stdout():
            result = mylbfgs(
                point,
                pot,
                tol=float(force_tol),
                maxstep=0.2,
                nsteps=int(budget),
                alternate_stop_criterion=_l2_stop,
                iprint=-1,
            )
        point = np.asarray(result.coords, dtype=np.float64).reshape(-1)
        converged = bool(result.success)
    except (ForceCallCap, LineSearchError):
        if pot.last is not None:
            point = np.asarray(pot.last, dtype=np.float64).reshape(-1)
        converged = False
    with pot.verification():
        energy, gradient = pot.energy_forces(point)
    force = float(np.linalg.norm(gradient))
    converged = bool(converged and math.isfinite(force) and force <= force_tol and pot.calls < budget)
    return point, converged, int(pot.calls), float(energy)


def _spectrum(evaluate, point: np.ndarray) -> tuple[list[float], np.ndarray, int]:
    pot = HarnessPotential(evaluate, None)
    flat = [float(value) for value in np.asarray(point, dtype=np.float64).reshape(-1)]
    with pot.verification():
        hessian = numerical_hessian(lambda coords: pot.energy_forces(coords)[1], flat, step=1e-5)
    matrix = np.asarray(hessian, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, vectors = np.linalg.eigh(matrix)
    return [float(value) for value in eigenvalues], vectors, int(pot.verify_calls)


def _index(point: np.ndarray, eigenvalues: list[float], vectors, project_rigid: bool) -> tuple[int, int, int, list[float]]:
    cartesian = negative_index(eigenvalues, 1e-6)
    if project_rigid:
        adjusted, removed = _vibrational_eigenvalues(point, eigenvalues, vectors)
    else:
        adjusted, removed = list(eigenvalues), 0
    return negative_index(adjusted, 1e-6), cartesian, int(removed), adjusted


def _transition_kwargs(hef: dict, dof: int, force_tol: float, eigenvec0: np.ndarray | None) -> dict:
    lowest = {key: value for key, value in dict(hef.get("lowest") or {}).items()}
    tangent = {key: value for key, value in dict(hef.get("tangent") or {}).items()}
    lowest["iprint"] = -1
    tangent["iprint"] = -1
    kwargs = {
        "tol": float(force_tol) / math.sqrt(float(dof)),
        "nsteps": int(hef.get("nsteps", 100)),
        "nfail_max": int(hef.get("nfail_max", 200)),
        "nsteps_tangent1": int(hef.get("nsteps_tangent1", 10)),
        "nsteps_tangent2": int(hef.get("nsteps_tangent2", 100)),
        "max_uphill_step": float(hef.get("max_uphill_step", 0.5)),
        "max_uphill_step_initial": float(hef.get("max_uphill_step_initial", 0.2)),
        "demand_initial_negative_vec": bool(hef.get("demand_initial_negative_vec", False)),
        "check_negative": bool(hef.get("check_negative", True)),
        "lowestEigenvectorQuenchParams": lowest,
        "tangentSpaceQuenchParams": tangent,
        "orthogZeroEigs": _orthog_value(str(hef.get("orthog", "translations-rotations"))),
        "verbosity": 0,
        "iprint": -1,
        "index_seed": 0 if hef.get("index_seed") is None else int(hef["index_seed"]),
        "hessian_diagonalization": False,
    }
    if eigenvec0 is not None:
        kwargs["eigenvec0"] = np.asarray(eigenvec0, dtype=np.float64).reshape(-1).copy()
    return kwargs


def _eigenvec0(kind: str, x: np.ndarray, reactant, file_mode) -> np.ndarray | None:
    if kind == "random":
        return None
    if kind == "reactant":
        if reactant is None:
            raise ValueError("reactant eigenvector requested and the reactant is missing")
        return _unit(x - np.asarray(reactant, dtype=np.float64).reshape(-1))
    if kind in {"kick", "initial-mode", "file"}:
        if file_mode is None:
            raise ValueError(f"{kind} eigenvector requested and the mode is missing")
        return _unit(np.asarray(file_mode, dtype=np.float64).reshape(-1))
    raise ValueError(f"unknown eigenvec0 {kind}")


def _crash(name: str, calls: int, verify_calls: int, error: str, cap: int | None = None) -> dict:
    failure = classify_outcome(calls=int(calls), cap=cap, process_failed=True)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": ["crash"],
        "error": error,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": 1,
        "converged": False,
    }


def _judge_ts(
    name: str,
    *,
    point: np.ndarray,
    energy: float,
    gradient: np.ndarray,
    evaluate,
    cap: int,
    calls: int,
    force_tol: float,
    project_rigid: bool,
    library_converged: bool,
    steps: int,
    eigenval: float | None,
    library_evaluations: int | None,
) -> dict:
    force_norm = float(np.linalg.norm(gradient))
    raw: list[float] = []
    vectors = np.zeros((0, 0))
    verify_calls = 0
    index = None
    cartesian = None
    removed = 0
    adjusted: list[float] = []
    reasons: list[str] = []
    if not (_finite(point) and math.isfinite(energy) and math.isfinite(force_norm)):
        reasons.append("divergence")
    else:
        raw, vectors, verify_calls = _spectrum(evaluate, point)
        index, cartesian, removed, adjusted = _index(point, raw, vectors, project_rigid)
        coords = np.asarray(point, dtype=np.float64).reshape(-1, 3).tolist()
        judged = judge_stationary(
            kind="saddle",
            eigenvalues=adjusted,
            energy=float(energy),
            coords=coords,
            reference_energy=float(energy),
            reference_coords=coords,
            energy_tol=0.0,
            rmsd_tol=0.0,
        )
        reasons = [reason for reason in judged["reasons"] if reason not in {"rmsd", "energy"}]
        if force_norm > force_tol + 1e-10:
            reasons.append("force_norm")
    if calls >= cap:
        reasons.append("cap")
    reasons = list(dict.fromkeys(reasons))
    failure = classify_outcome(
        energy=float(energy),
        geometry=point,
        force=force_norm,
        calls=int(calls),
        cap=cap,
        force_tol=force_tol,
        hessian_index=None if index is None else int(index),
        expected_index=1 if index is not None else None,
    )
    success = failure is None
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": float(energy),
        "force_norm": float(force_norm),
        "hessian_index": None if index is None else int(index),
        "cartesian_index": None if cartesian is None else int(cartesian),
        "rigid_modes_removed": int(removed),
        "expected_index": 1,
        "lowest_eigenvalue": float(adjusted[0]) if adjusted else None,
        "converged": bool(library_converged) and math.isfinite(force_norm) and force_norm <= force_tol,
        "steps": int(steps),
        "eigenval": None if eigenval is None or not math.isfinite(float(eigenval)) else float(eigenval),
        "library_evaluations": None if library_evaluations is None else int(library_evaluations),
        "x": [float(value) for value in np.asarray(point, dtype=np.float64).reshape(-1)],
        "mode": None,
    }


def _climb(spec: dict) -> dict:
    """One hybrid-eigenvector search. The return includes the final point."""
    _assert_pele(str(spec.get("commit") or ""))
    _import_pele()
    from pele.optimize.optimization_exceptions import LineSearchError
    from pele.transition_states import findTransitionState

    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    free = np.flatnonzero(~fixed)
    x = positions[free].reshape(-1).copy()
    raw = potential_from_problem({"potential": spec["potential"]})
    evaluate = _bind(raw, positions, free, cell)
    cap = int(spec["cap"])
    force_tol = float(spec["force_tol"])
    pot = HarnessPotential(evaluate, cap)
    guess = _eigenvec0(str(spec["eigenvec0"]), x, spec.get("reactant"), spec.get("file_mode"))
    np.random.seed(0)
    kwargs = _transition_kwargs(spec["hef"], int(x.size), force_tol, guess)
    library_converged = False
    steps = 0
    eigenval = None
    eigenvec = None
    nfev = None
    screen_calls = 0
    screen_passes = 0
    from pele.transition_states._transition_state_refinement import FindTransitionState

    original_screen = FindTransitionState._get_lowest_transverse_eigenvector

    def _counting_screen(self, coords, gradient):
        nonlocal screen_calls, screen_passes
        before = int(pot.calls)
        try:
            return original_screen(self, coords, gradient)
        finally:
            screen_passes += 1
            screen_calls += int(pot.calls) - before

    FindTransitionState._get_lowest_transverse_eigenvector = _counting_screen
    try:
        try:
            with _silent_stdout():
                result = findTransitionState(x, pot, **kwargs)
            point = np.asarray(result.coords, dtype=np.float64).reshape(-1)
            library_converged = bool(result.success)
            steps = int(result.nsteps)
            eigenval = float(result.eigenval)
            eigenvec = np.asarray(result.eigenvec, dtype=np.float64).reshape(-1).copy()
            nfev = int(getattr(result, "nfev", pot.calls))
        except ForceCallCap:
            point = pot.last.copy() if pot.last is not None else x
        except LineSearchError:
            point = pot.last.copy() if pot.last is not None else x
        except Exception as exc:
            return _crash(spec["name"], pot.calls, pot.verify_calls, exc.__class__.__name__, cap)
    finally:
        FindTransitionState._get_lowest_transverse_eigenvector = original_screen
    with pot.verification():
        energy, gradient = pot.energy_forces(point)
    if spec.get("defer_hessian"):
        force_norm = float(np.linalg.norm(gradient))
        return {
            "name": spec["name"],
            "x": [float(value) for value in point],
            "mode": None if eigenvec is None else [float(value) for value in eigenvec],
            "force_calls": int(pot.calls),
            "verify_calls": int(pot.verify_calls),
            "energy": float(energy) if math.isfinite(float(energy)) else None,
            "force_norm": force_norm if math.isfinite(force_norm) else None,
            "converged": bool(library_converged) and math.isfinite(force_norm) and force_norm <= force_tol,
            "steps": int(steps),
            "eigenval": None if eigenval is None or not math.isfinite(float(eigenval)) else float(eigenval),
            "library_evaluations": nfev,
            "failure": None,
            "screen_calls": int(screen_calls),
            "screen_passes": int(screen_passes),
            "unscreened_calls": int(pot.calls) - int(screen_calls),
        }
    record = _judge_ts(
        spec["name"],
        point=point,
        energy=float(energy),
        gradient=np.asarray(gradient, dtype=np.float64),
        evaluate=evaluate,
        cap=cap,
        calls=int(pot.calls),
        force_tol=force_tol,
        project_rigid=not bool(np.any(fixed)),
        library_converged=library_converged,
        steps=steps,
        eigenval=eigenval,
        library_evaluations=nfev,
    )
    record["mode"] = None if eigenvec is None else [float(value) for value in eigenvec]
    record["screen_calls"] = int(screen_calls)
    record["screen_passes"] = int(screen_passes)
    record["unscreened_calls"] = int(pot.calls) - int(screen_calls)
    return record


def _run_ts(spec: dict) -> dict:
    record = _climb(spec)
    record.pop("evaluate", None)
    return record


def _run_mode(spec: dict) -> dict:
    _assert_pele(str(spec.get("commit") or ""))
    _import_pele()
    from pele.optimize.optimization_exceptions import LineSearchError
    from pele.transition_states import findLowestEigenVector

    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    free = np.flatnonzero(~fixed)
    x = positions[free].reshape(-1).copy()
    reference = _unit(np.asarray(spec["reference_mode"], dtype=np.float64).reshape(-1))
    guess = _eigenvec0("initial-mode", x, None, spec.get("file_mode"))
    raw = potential_from_problem({"potential": spec["potential"]})
    evaluate = _bind(raw, positions, free, cell)
    cap = int(spec["cap"])
    pot = HarnessPotential(evaluate, cap)
    mode = dict(spec["mode"])
    kwargs = {
        "eigenvec0": guess,
        "orthogZeroEigs": _orthog_value(str(mode.get("orthog", "translations-rotations"))),
        "dx": float(mode.get("dx", 1e-3)),
        "first_order": bool(mode.get("first_order", True)),
        "iprint": -1,
        "tol": float(mode.get("tol", 1e-6)),
        "nsteps": int(mode.get("nsteps", 500)),
    }
    for key in ("M", "maxstep", "H0"):
        if key in mode:
            kwargs[key] = int(mode[key]) if key == "M" else float(mode[key])
    if mode.get("overlap_stop"):
        kwargs["alternate_stop_criterion"] = OverlapStop(reference)
    np.random.seed(0)
    overlap = 0.0
    steps = 0
    eigenval = None
    try:
        with _silent_stdout():
            result = findLowestEigenVector(x, pot, **kwargs)
        eigenvec = np.asarray(result.eigenvec, dtype=np.float64).reshape(-1)
        overlap = float(abs(np.dot(_unit(eigenvec), reference)))
        steps = int(getattr(result, "nsteps", 0))
        eigenval = float(result.eigenval)
        point = x
    except ForceCallCap:
        point = x
        overlap = 0.0
    except LineSearchError:
        point = x
    except Exception as exc:
        return _crash(spec["name"], pot.calls, pot.verify_calls, exc.__class__.__name__, cap)
    threshold = float(spec["overlap"])
    reasons: list[str] = []
    if not (math.isfinite(overlap) and overlap >= threshold and pot.calls < cap):
        reasons.append("mode_overlap")
    if pot.calls >= cap:
        reasons.append("cap")
    failure = classify_outcome(
        calls=int(pot.calls),
        cap=cap,
        unmet="mode_overlap" in reasons,
    )
    success = failure is None
    with pot.verification():
        energy, gradient = pot.energy_forces(point)
    record = {
        "schema": "optbench.start/v2",
        "start": spec["name"],
        "force_calls": int(pot.calls),
        "verify_calls": int(pot.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": float(energy),
        "force_norm": float(np.linalg.norm(gradient)),
        "hessian_index": None,
        "expected_index": 1,
        "overlap": float(overlap) if math.isfinite(overlap) else None,
        "converged": bool(success),
        "steps": steps,
        "eigenval": eigenval,
    }
    if _finite(point):
        raw_values, vectors, verify_calls = _spectrum(evaluate, point)
        index, cartesian, removed, adjusted = _index(point, raw_values, vectors, not bool(np.any(fixed)))
        record["hessian_index"] = int(index)
        record["cartesian_index"] = int(cartesian)
        record["rigid_modes_removed"] = int(removed)
        record["lowest_eigenvalue"] = float(adjusted[0]) if adjusted else None
        record["verify_calls"] = int(verify_calls)
    return record


def _dispatch(spec: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if spec["task"] == "lowest-eigenvector":
        return _run_mode(spec)
    if spec["task"] == "transition-state":
        record = _climb(spec)
        record.pop("mode", None)
        return record
    raise ValueError(f"unknown task {spec['task']}")


def _spec_from_geometry(row: dict, problem: dict, template: dict, cap: int, force_tol: float) -> dict:
    positions, fixed, cell = load_con(Path(row["path"]))
    reactant = None
    if row.get("reactant"):
        other, other_fixed, _cell = load_con(Path(row["reactant"]))
        reactant = other[~other_fixed].reshape(-1)
    file_mode = None
    reference = None
    if row.get("initial_mode"):
        full = read_mode(Path(row["initial_mode"]).read_text(encoding="utf-8", errors="replace"))
        file_mode = full.reshape(-1, 3)[~fixed].reshape(-1)
    if row.get("mode"):
        full = read_mode(Path(row["mode"]).read_text(encoding="utf-8", errors="replace"))
        reference = full.reshape(-1, 3)[~fixed].reshape(-1)
    task = str(template.get("task") or "")
    hef = template.get("hef") or {}
    kind = str(hef.get("eigenvec0", "random"))
    if reactant is not None and kind == "random":
        kind = "reactant"
    return {
        "name": row["name"],
        "positions": positions,
        "fixed": fixed,
        "cell": cell,
        "potential": problem.get("potential") or {},
        "task": task,
        "hef": hef,
        "mode": template.get("mode") or {},
        "cap": cap,
        "force_tol": force_tol,
        "reactant": reactant,
        "file_mode": file_mode,
        "reference_mode": reference,
        "overlap": float((problem.get("sample") or {}).get("overlap", 0.99)),
        "eigenvec0": kind,
        "commit": str(template.get("commit") or ""),
    }


def _public(record: dict, template: dict, problem_id: str) -> dict:
    dropped = {"x", "mode", "evaluate", "saddle_x", "rng_state"}
    copied = {}
    for key, value in record.items():
        if key in dropped:
            continue
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, np.generic):
            copied[key] = value.item()
        else:
            copied[key] = value
    copied["code"] = template.get("code", "pele")
    copied["code_version"] = template.get("code_version", PELE_VERSION)
    copied["commit"] = template.get("commit", "")
    copied["tuning"] = template.get("tuning", "")
    copied["template"] = template["id"]
    copied["problem"] = problem_id
    return copied


def _summarize(records: list[dict], template: dict, cap: int, problem_id: str, *, short: bool) -> tuple[str, dict]:
    ok = [row for row in records if row["success"]]
    counts = [int(row["force_calls"]) for row in ok]
    lines = []
    if counts:
        lines.append(f"force_calls {half_even(sum(counts) / len(counts))}")
        lines.append(f"force_calls_min {min(counts)}")
        lines.append(f"force_calls_median {statistics.median(counts)}")
        lines.append(f"force_calls_max {max(counts)}")
    lines.append(f"nfailed {len(records) - len(ok)}")
    lines.append(f"starts {len(records)}")
    lines.append(f"n {len(records)}")
    metrics = score_v2(records, cap)
    lines.append(f"par2_mean {metrics['par2_mean']:.10f}")
    lines.append(f"successes {metrics['successes']}")
    lines.append(f"rate_low {metrics['rate_low']:.10f}")
    lines.append(f"rate_high {metrics['rate_high']:.10f}")
    lines.append(f"par2_median {metrics['median']:.10f}")
    lines.append(f"median {metrics['median']:.10f}")
    lines.append(f"median_low {metrics['median_low']:.10f}")
    lines.append(f"median_high {metrics['median_high']:.10f}")
    lines.append(f"n_cap {metrics['n_cap']}")
    lines.append(f"n_wrong_stationary {metrics['n_wrong_stationary']}")
    lines.append(f"n_divergence {metrics['n_divergence']}")
    lines.append(f"n_not_converged {metrics['n_not_converged']}")
    lines.append(f"n_crash {metrics['n_crash']}")
    lines.append("counter harness")
    lines.append(f"cap {cap}")
    lines.append(f"code {template.get('code', 'pele')}")
    lines.append(f"code_version {template.get('code_version', PELE_VERSION)}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem_id}")
    if short:
        lines.append("short true")
    if template.get("note"):
        lines.append(f"note {template['note']}")
    lines.append("contributor Rohit Goswami")
    lines.append(f"date {date.today().isoformat()}")
    lines.append(f"host {os.uname().nodename}")
    return "\n".join(lines) + "\n", metrics


def _store_endpoints(output: Path, records: list[dict]) -> None:
    """Write endpoint minima beside ``output`` and drop them from ``records``.

    One row of each array is one search, in starts-file order. A search
    that never reached the endpoints is a row of NaN.
    """
    if not any(row.get("endpoint_plus") is not None or row.get("endpoint_minus") is not None for row in records):
        return
    width = 0
    for row in records:
        for key in ("endpoint_plus", "endpoint_minus", "minimum"):
            value = row.get(key)
            if value is not None:
                width = len(value)
                break
        if width:
            break
    count = len(records)
    plus = np.full((count, width), np.nan, dtype=np.float64)
    minus = np.full((count, width), np.nan, dtype=np.float64)
    minimum = np.full((count, width), np.nan, dtype=np.float64)
    for index, row in enumerate(records):
        if row.get("endpoint_plus") is not None:
            plus[index] = row["endpoint_plus"]
        if row.get("endpoint_minus") is not None:
            minus[index] = row["endpoint_minus"]
        if row.get("minimum") is not None:
            minimum[index] = row["minimum"]
        row.pop("endpoint_plus", None)
        row.pop("endpoint_minus", None)
        row.pop("minimum", None)
    plus_path = output.with_name(output.stem + ".endpoints-plus.npz")
    minus_path = output.with_name(output.stem + ".endpoints-minus.npz")
    np.savez_compressed(plus_path, endpoint_plus=plus, minimum=minimum)
    np.savez_compressed(minus_path, endpoint_minus=minus)


def _write_campaign(output: Path, text: str, records: list[dict], template: dict, problem_id: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)
    public = [_public(row, template, problem_id) for row in records]
    _store_endpoints(output, public)
    starts = output.with_suffix(".starts.jsonl")
    temporary_starts = starts.with_suffix(".jsonl.tmp")
    write_starts(temporary_starts, public)
    os.replace(temporary_starts, starts)


def _checkpoint_paths(output: Path) -> tuple[Path, Path]:
    return output.with_name(output.stem + ".header.json"), output.with_name(output.stem + ".journal.jsonl")


def _load_journal(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
    return rows


def _append_journal(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _rng_state(rng) -> dict:
    return json.loads(json.dumps(rng.bit_generator.state))


def _rng_restore(payload: dict):
    rng = np.random.default_rng()
    rng.bit_generator.state = payload
    return rng


def _saddle_limits(template: dict) -> dict:
    search = template.get("search") or {}
    return {
        "goal": int(search.get("connected", 27)),
        "barrier_max": float(search.get("barrier_eV", 1.5)),
        "repetitions": int(search.get("repetitions", 1)),
        "atoms": [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()],
        "magnitude": float(search.get("displace_magnitude", 0.1)),
        "connect_rmsd": float(search.get("rmsd_connect", 0.5)),
        "max_searches": int(search.get("max_searches", 20000)),
        "endpoint_budget": int(search.get("endpoint_budget", _SADDLE_ENDPOINT_BUDGET)),
        "minimize_budget": int(search.get("minimize_budget", _SADDLE_MINIMIZE_BUDGET)),
    }


def _one_displacement(spec: dict, *, evaluate, minimum: np.ndarray, min_energy: float, kick: np.ndarray, limits: dict, climb_cap: int, force_tol: float) -> dict:
    climbed = dict(spec)
    climbed["defer_hessian"] = True
    outcome = _climb(climbed)
    if outcome.get("failure") == "crash" and outcome.get("hessian_index") is None and "x" not in outcome:
        outcome["force_calls"] = int(outcome.get("force_calls") or 0)
        outcome["endpoint_calls"] = 0
        outcome["climb_calls"] = int(outcome["force_calls"])
        return outcome
    climb_calls = int(outcome["force_calls"])
    point = np.asarray(outcome.get("x"), dtype=np.float64) if outcome.get("x") is not None else None
    force_norm = outcome.get("force_norm")
    finite = point is not None and _finite(point) and force_norm is not None and math.isfinite(float(force_norm))
    record = {
        "schema": "optbench.start/v2",
        "start": spec["name"],
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": None,
        "reasons": [],
        "converged": bool(outcome.get("converged")),
        "steps": int(outcome.get("steps") or 0),
        "eigenval": outcome.get("eigenval"),
        "library_evaluations": outcome.get("library_evaluations"),
        "climb_calls": climb_calls,
        "endpoint_calls": 0,
        "force_calls": climb_calls,
        "verify_calls": 0,
        "hessian_index": None,
        "energy": outcome.get("energy"),
        "force_norm": None if force_norm is None else float(force_norm),
    }
    energy_ok = outcome.get("energy") is not None and math.isfinite(float(outcome["energy"]))
    if not finite or not energy_ok:
        record["reasons"] = ["divergence"] if outcome.get("failure") != "crash" else ["crash"]
    elif float(force_norm) > force_tol + 1e-10:
        record["reasons"] = ["cap", "force_norm"] if climb_calls >= climb_cap else ["force_norm"]
    else:
        raw, vectors, verify_calls = _spectrum(evaluate, point)
        index, cartesian, removed, _adjusted = _index(point, raw, vectors, False)
        record["hessian_index"] = int(index)
        record["cartesian_index"] = int(cartesian)
        record["rigid_modes_removed"] = int(removed)
        record["verify_calls"] = int(verify_calls)
        record["lowest_eigenvalue"] = float(raw[0]) if raw else None
        if index != 1:
            record["reasons"] = ["hessian_index"]
        else:
            mode = outcome.get("mode")
            axis = _unit(np.asarray(mode if mode is not None else kick, dtype=np.float64))
            plus, _plus_ok, plus_calls, _plus_energy = _minimize(
                evaluate, point + 0.1 * axis, force_tol, limits["endpoint_budget"]
            )
            minus, _minus_ok, minus_calls, _minus_energy = _minimize(
                evaluate, point - 0.1 * axis, force_tol, limits["endpoint_budget"]
            )
            record["endpoint_calls"] = int(plus_calls + minus_calls)
            record["force_calls"] = climb_calls + record["endpoint_calls"]
            record["endpoint_plus"] = [float(value) for value in plus]
            record["endpoint_minus"] = [float(value) for value in minus]
            record["minimum"] = [float(value) for value in minimum]
            plus_atoms = plus.reshape(-1, 3)
            minus_atoms = minus.reshape(-1, 3)
            minimum_atoms = minimum.reshape(-1, 3)
            cell = spec["cell"]
            tolerance = float(limits.get("link_tolerance", link_tolerance(None)))
            record["rmsd_plus"] = _cartesian_rmsd(plus_atoms, minimum_atoms)
            record["rmsd_minus"] = _cartesian_rmsd(minus_atoms, minimum_atoms)
            record["rmsd_anchor_plus"] = _mean_free_rmsd(plus_atoms, minimum_atoms)
            record["rmsd_anchor_minus"] = _mean_free_rmsd(minus_atoms, minimum_atoms)
            record["per_atom_max_plus"] = per_atom_max(plus_atoms, minimum_atoms, cell)
            record["per_atom_max_minus"] = per_atom_max(minus_atoms, minimum_atoms, cell)
            record["barrier"] = float(outcome["energy"] - min_energy)
            record["_point"] = point
            record["_axis_links"] = connected_endpoints(
                minimum_atoms, plus_atoms, minus_atoms, cell, tolerance
            )
            record["_rmsd_anchor_links"] = (record["rmsd_anchor_plus"] <= limits["connect_rmsd"]) ^ (
                record["rmsd_anchor_minus"] <= limits["connect_rmsd"]
            )
    names = set(record["reasons"])
    energy = record.get("energy")
    if "divergence" in names and (energy is None or not math.isfinite(float(energy))):
        energy = float("nan")
        record["energy"] = energy
    failure = classify_outcome(
        energy=None if energy is None else float(energy),
        geometry=record.get("_point"),
        force=None if record.get("force_norm") is None else float(record["force_norm"]),
        calls=float(record["force_calls"]),
        cap=climb_cap,
        force_tol=force_tol,
        hessian_index=None if record.get("hessian_index") is None else int(record["hessian_index"]),
        expected_index=1 if record.get("hessian_index") is not None else None,
        process_failed="crash" in names,
        force_above=True if "force_norm" in names and record.get("force_norm") is None else None,
    )
    record["success"] = failure is None
    record["failure"] = failure
    return record


def _saddle_once(root: Path, problem: dict, template: dict) -> dict:
    """The first displacement of repetition 1. The campaign checkpoint is left alone."""
    from .campaign import _check_stop

    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = load_con(_stored(root, con_path))
    limits = _saddle_limits(template)
    limits["link_tolerance"] = link_tolerance(problem)
    climb_cap = int(_check_stop(problem, template) or 1000)
    force_tol = float((problem.get("convergence") or {}).get("force_norm", 1e-3))
    free = np.flatnonzero(~fixed)
    raw = potential_from_problem(problem)
    evaluate = _bind(raw, positions, free, cell)
    minimum, _min_ok, _min_calls, min_energy = _minimize(
        evaluate, positions[free].reshape(-1).copy(), force_tol, limits["minimize_budget"]
    )
    rng = np.random.default_rng(20261016)
    atom = int(limits["atoms"][int(rng.integers(0, len(limits["atoms"])))])
    slots = np.flatnonzero(free == atom)
    if len(slots) != 1:
        raise RuntimeError(f"displace atom {atom} is not one free atom")
    slot = int(slots[0])
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    x0 = minimum.copy()
    x0[3 * slot : 3 * slot + 3] += limits["magnitude"] * direction
    kick = np.zeros_like(x0)
    kick[3 * slot : 3 * slot + 3] = direction
    moved = positions.copy()
    moved[free] = x0.reshape(-1, 3)
    spec = _spec_from_geometry(
        {"name": "rep1-search1", "path": _stored(root, con_path)},
        problem,
        template,
        climb_cap,
        force_tol,
    )
    spec["positions"] = moved
    spec["file_mode"] = kick
    spec["eigenvec0"] = "kick"
    spec["name"] = "rep1-search1"
    found: list[np.ndarray] = []
    found_rmsd: list[np.ndarray] = []
    record = _finish_saddle_record(
        _one_displacement(
            spec,
            evaluate=evaluate,
            minimum=minimum,
            min_energy=min_energy,
            kick=kick,
            limits=limits,
            climb_cap=climb_cap,
            force_tol=force_tol,
        ),
        found=found,
        found_rmsd=found_rmsd,
        limits=limits,
        cap=climb_cap,
        force_tol=force_tol,
    )
    return record


def _restore_saddles(saved: list[dict], limits: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    found: list[np.ndarray] = []
    found_rmsd: list[np.ndarray] = []
    tol = float(limits["connect_rmsd"])
    for row in saved:
        raw = row.get("saddle_x")
        if raw is None:
            continue
        saddle = np.asarray(raw, dtype=np.float64)
        atoms = saddle.reshape(-1, 3)
        if all(_cartesian_rmsd(atoms, old.reshape(-1, 3)) > tol for old in found):
            found.append(saddle)
        if all(_mean_free_rmsd(atoms, old.reshape(-1, 3)) > tol for old in found_rmsd):
            found_rmsd.append(saddle.copy())
    return found, found_rmsd


def _finish_saddle_record(
    record: dict,
    *,
    found: list[np.ndarray],
    found_rmsd: list[np.ndarray],
    limits: dict,
    cap: int,
    force_tol: float,
) -> dict:
    point = record.pop("_point", None)
    links = record.pop("_axis_links", False)
    rmsd_links = record.pop("_rmsd_anchor_links", False)
    if point is None:
        record["rmsd_success"] = False
        return record
    saddle = np.asarray(point, dtype=np.float64)
    record["saddle_x"] = [float(value) for value in saddle]
    novel = all(
        _cartesian_rmsd(saddle.reshape(-1, 3), old.reshape(-1, 3)) > limits["connect_rmsd"] for old in found
    )
    rmsd_novel = all(
        _mean_free_rmsd(saddle.reshape(-1, 3), old.reshape(-1, 3)) > limits["connect_rmsd"] for old in found_rmsd
    )
    barrier = float(record.get("barrier") or 0.0)
    reasons: list[str] = []
    if novel:
        found.append(saddle.copy())
    if rmsd_novel:
        found_rmsd.append(saddle.copy())
    if links and novel and barrier < limits["barrier_max"]:
        record["reasons"] = []
    else:
        if not links:
            reasons.append("not_connected")
        if not novel:
            reasons.append("duplicate")
        if barrier >= limits["barrier_max"]:
            reasons.append("barrier")
        record["reasons"] = reasons
    failure = classify_outcome(
        energy=None if record.get("energy") is None else float(record["energy"]),
        geometry=point,
        force=None if record.get("force_norm") is None else float(record["force_norm"]),
        calls=float(record["force_calls"]),
        cap=cap,
        force_tol=force_tol,
        hessian_index=None if record.get("hessian_index") is None else int(record["hessian_index"]),
        expected_index=1 if record.get("hessian_index") is not None else None,
        stationary_rejected=bool(set(reasons) & {"not_connected", "duplicate", "barrier"}),
    )
    record["success"] = failure is None
    record["failure"] = failure
    rmsd_reasons: list[str] = []
    if not rmsd_links:
        rmsd_reasons.append("not_connected")
    if not rmsd_novel:
        rmsd_reasons.append("duplicate")
    if barrier >= limits["barrier_max"]:
        rmsd_reasons.append("barrier")
    record["rmsd_success"] = not rmsd_reasons
    record["rmsd_reasons"] = rmsd_reasons
    return record


def _saddle_job(payload: dict) -> dict:
    """One displacement. The parent applies the connection rules in search order."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    spec = payload["spec"]
    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    free = np.flatnonzero(~fixed)
    raw = potential_from_problem({"potential": spec["potential"]})
    evaluate = _bind(raw, positions, free, cell)
    return _one_displacement(
        spec,
        evaluate=evaluate,
        minimum=np.asarray(payload["minimum"], dtype=np.float64),
        min_energy=float(payload["min_energy"]),
        kick=np.asarray(payload["kick"], dtype=np.float64),
        limits=payload["limits"],
        climb_cap=int(payload["climb_cap"]),
        force_tol=float(payload["force_tol"]),
    )


def _run_saddle_search(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    cap: int,
    force_tol: float,
    limit: int,
    workers: int = 1,
) -> dict:
    """Hybrid eigenvector following from ``draw_search_start`` until 27 qualifying saddles.

    The climb budget is 1000 force calls. Each endpoint minimization has
    its own budget of 2000. The PAR-2 cap is 5064. The initial
    minimization is outside that count. ``cap`` from the template is the
    climb budget when it is 1000. A larger template value is not used as
    the climb: the problem statement fixes that budget.
    """
    del cap
    from .connection import ProcessCatalog
    from .eon_displace import draw_search_start, problem_pbc
    from .endpoints import EndpointArchive, align_resume, stamp_endpoint, stored_cell
    from .failures import classify_failure
    from .saddle_publish import write_saddle_campaign

    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = load_con(_stored(root, con_path))
    limits = _saddle_limits(template)
    goal = int((problem.get("goal") or {}).get("connected_saddles", limits["goal"]))
    limits["barrier_max"] = float((problem.get("goal") or {}).get("barrier_below_eV", limits["barrier_max"]))
    limits["link_tolerance"] = link_tolerance(problem)
    climb_cap = 1000
    pbc = problem_pbc(problem)
    radius = float((template.get("search") or {}).get("displace_radius", 3.3))
    max_searches = int(limits["max_searches"])
    if limit:
        max_searches = min(max_searches, int(limit))
    free = np.flatnonzero(~fixed)
    archive = EndpointArchive(output.with_name(output.stem + ".endpoints.npz"), int(free.size))
    raw_potential = potential_from_problem(problem)
    comments: list[str] = []
    all_records: list[dict] = []
    failed_reps = 0
    executor = None
    if int(workers) > 1:
        import multiprocessing

        executor = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"))
    try:
        for rep in range(1, limits["repetitions"] + 1):
            header_path, journal_path = _checkpoint_paths(output)
            header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
            same = bool(header and header.get("template") == template["id"] and int(header.get("rep", 0)) == rep)
            saved = _load_journal(journal_path) if same else []
            if same:
                align_resume(archive, len(all_records) + len(saved))
                minimum = np.asarray(header["minimum"], dtype=np.float64)
                min_energy = float(header["min_energy"])
                minimize_calls = int(header["minimize_calls"])
                min_ok = bool(header["min_ok"])
                catalog = ProcessCatalog(cell, tolerance=limits["link_tolerance"], barrier_max=limits["barrier_max"])
                for row in saved:
                    judged = row.get("per_atom_max_plus") is not None
                    catalog.observe(
                        judged=judged,
                        linked=judged and "not_connected" not in (row.get("reasons") or []),
                        structure=row.get("saddle_x"),
                        barrier=row.get("barrier"),
                    )
                for row in saved:
                    all_records.append(
                        {key: value for key, value in row.items() if key not in {"saddle_x", "endpoint_plus", "endpoint_minus", "minimum"}}
                    )
                print(f"{template['id']} rep {rep} resume searches {catalog.searches} connected {catalog.qualifying}", flush=True)
            else:
                if rep == 1:
                    archive.reset(stored_cell(cell), free)
                evaluate = _bind(raw_potential, positions, free, cell)
                minimum, min_ok, minimize_calls, min_energy = _minimize(
                    evaluate, positions[free].reshape(-1).copy(), force_tol, limits["minimize_budget"]
                )
                catalog = ProcessCatalog(cell, tolerance=limits["link_tolerance"], barrier_max=limits["barrier_max"])
                header_path.parent.mkdir(parents=True, exist_ok=True)
                header_path.write_text(
                    json.dumps(
                        {
                            "template": template["id"],
                            "rep": rep,
                            "minimum": minimum.tolist(),
                            "min_energy": min_energy,
                            "minimize_calls": minimize_calls,
                            "min_ok": bool(min_ok),
                        }
                    ),
                    encoding="utf-8",
                )
                journal_path.unlink(missing_ok=True)
            reactant = positions.copy()
            reactant[free] = minimum.reshape(-1, 3)
            width = max(1, int(workers))
            while catalog.qualifying < goal and catalog.searches < max_searches:
                batch = []
                while len(batch) < width and catalog.searches + len(batch) < max_searches:
                    attempt = catalog.searches + len(batch) + 1
                    drawn = draw_search_start(
                        reactant,
                        fixed,
                        cell,
                        pbc,
                        limits["atoms"],
                        rep=rep,
                        attempt=attempt,
                        radius=radius,
                        magnitude=limits["magnitude"],
                    )
                    spec = {
                        "name": f"rep{rep}-search{attempt}",
                        "positions": drawn["start"],
                        "fixed": fixed,
                        "cell": cell,
                        "potential": problem.get("potential") or {},
                        "task": "transition-state",
                        "hef": template.get("hef") or {},
                        "mode": {},
                        "cap": climb_cap,
                        "force_tol": force_tol,
                        "reactant": None,
                        "file_mode": np.asarray(drawn["kick"], dtype=np.float64).copy(),
                        "reference_mode": None,
                        "overlap": 0.99,
                        "eigenvec0": "kick",
                        "commit": str(template.get("commit") or ""),
                    }
                    batch.append(
                        {
                            "spec": spec,
                            "minimum": minimum,
                            "min_energy": min_energy,
                            "kick": np.asarray(drawn["kick"], dtype=np.float64),
                            "limits": limits,
                            "climb_cap": climb_cap,
                            "force_tol": force_tol,
                            "seed": int(drawn["seed"]),
                        }
                    )
                if executor is None:
                    outcomes = [_saddle_job(item) for item in batch]
                else:
                    outcomes = list(executor.map(_saddle_job, batch))
                for item, record in zip(batch, outcomes):
                    if catalog.qualifying >= goal:
                        break
                    point = record.pop("_point", None)
                    links = bool(record.pop("_axis_links", False))
                    record.pop("_rmsd_anchor_links", None)
                    judged = record.get("per_atom_max_plus") is not None
                    decision = catalog.observe(
                        judged=judged,
                        linked=links,
                        structure=None if point is None else np.asarray(point, dtype=np.float64),
                        barrier=record.get("barrier"),
                    )
                    if judged:
                        record["reasons"] = list(decision["reasons"])
                        record["success"] = bool(decision["success"])
                        record["failure"] = None if record["success"] else classify_failure(record["reasons"] or ["divergence"])
                    elif record.get("success"):
                        record["success"] = False
                        record["reasons"] = list(record.get("reasons") or ["divergence"])
                        record["failure"] = classify_failure(record["reasons"])
                    plus = record.pop("endpoint_plus", None)
                    minus = record.pop("endpoint_minus", None)
                    record.pop("minimum", None)
                    plus_rows = None if plus is None else np.asarray(plus, dtype=np.float64).reshape(-1, 3)
                    minus_rows = None if minus is None else np.asarray(minus, dtype=np.float64).reshape(-1, 3)
                    status = archive.add(minimum.reshape(-1, 3), plus_rows, minus_rows)
                    stamp_endpoint(record, archive.name, archive.count - 1, status)
                    record["code"] = template.get("code", "pele")
                    record["code_version"] = template.get("code_version", PELE_VERSION)
                    record["commit"] = template.get("commit", "")
                    record["tuning"] = template.get("tuning", "")
                    record["template"] = template["id"]
                    record["problem"] = problem["id"]
                    stored = {key: value for key, value in record.items() if key != "saddle_x"}
                    if judged and point is not None:
                        stored["saddle_x"] = [float(value) for value in np.asarray(point, dtype=np.float64).reshape(-1)]
                    stored["seed"] = int(item["seed"])
                    _append_journal(journal_path, stored)
                    all_records.append({key: value for key, value in stored.items() if key != "saddle_x"})
                    if catalog.searches == 1 or catalog.searches % 10 == 0 or catalog.qualifying >= goal:
                        print(
                            f"{template['id']} rep {rep} searches {catalog.searches} "
                            f"connected {catalog.qualifying} unique {catalog.unique} "
                            f"calls {record['force_calls']} failure {record['failure']}",
                            flush=True,
                        )
            if catalog.qualifying < goal:
                failed_reps += 1
            rep_calls = sum(int(row["force_calls"]) for row in all_records if str(row["start"]).startswith(f"rep{rep}-"))
            comments.append(
                f"# rep {rep} force_calls {rep_calls} unique_saddles {catalog.unique} "
                f"fraction_connected {catalog.fraction_connected} total_searches {catalog.searches} "
                f"connected {catalog.qualifying} minimize_calls {minimize_calls} "
                f"minimum_converged {int(min_ok)} search_seed (20261015+{rep})*1000003+attempt"
            )
        metrics = write_saddle_campaign(
            output,
            template,
            problem,
            all_records,
            comments,
            archive,
            search_cap=_SADDLE_SEARCH_CAP,
            goal=goal,
            repetitions=limits["repetitions"],
            failed_reps=failed_reps,
            note=(
                "One row is one draw_search_start displacement followed by hybrid "
                "eigenvector following. The Rayleigh-Ritz step is first order. "
                "The cap is the climb budget of 1000, 64 calls of overshoot, and "
                "the two endpoint minimizations. The initial minimization is not "
                "in that count."
            ),
        )
        header_path, journal_path = _checkpoint_paths(output)
        journal_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
        return {
            "force_calls": None if metrics is None else metrics.get("success_only_mean"),
            "nfailed": failed_reps,
            "starts": len(all_records),
            "template": template["id"],
            "problem": problem["id"],
            "par2_mean": None if metrics is None else metrics.get("par2_mean"),
            "median": None if metrics is None else metrics.get("median"),
            "connected": None if not comments else int(comments[-1].split()[comments[-1].split().index("connected") + 1]),
            "short": False,
            "records": all_records,
            "failed_reps": failed_reps,
        }
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
) -> dict:
    """Run one pele template and write ``benchmark.dat`` plus the starts."""
    from .campaign import _check_stop

    _assert_pele(str(template.get("commit") or ""))
    cap = _check_stop(problem, template)
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    convergence = problem.get("convergence") or {}
    force_tol = float(convergence.get("force_norm", 1e-3))
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    if str(template.get("task")) == "saddle-search" or problem["id"] == "saddle-search/pt-island":
        return _run_saddle_search(
            root,
            problem,
            template,
            output=output,
            cap=cap,
            force_tol=force_tol,
            limit=limit,
            workers=workers,
        )
    rows = _geometries(root, problem, scratch)
    expected = int((problem.get("sample") or {}).get("starts") or len(rows))
    if limit:
        rows = rows[: int(limit)]
    elif len(rows) != expected:
        raise RuntimeError(f"{problem['id']} has {len(rows)} geometries, the problem states {expected}")
    specs = [_spec_from_geometry(row, problem, template, cap, force_tol) for row in rows]
    if workers <= 1:
        records = [_dispatch(spec) for spec in specs]
    else:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            records = []
            for index, record in enumerate(pool.map(_dispatch, specs), start=1):
                records.append(record)
                if index == 1 or index % 25 == 0 or index == len(specs):
                    print(
                        f"{template['id']} {index}/{len(specs)} calls {record['force_calls']} "
                        f"success {record['success']} failure {record['failure']}",
                        flush=True,
                    )
    short = bool(limit) and len(records) < expected
    text, metrics = _summarize(records, template, cap, problem["id"], short=short)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = collect_provenance(
        code=str(template.get("code", "pele")),
        code_version=str(template.get("code_version", PELE_VERSION)),
        commit=str(template.get("commit", "")),
        parameters={"template": template["id"], "problem": problem["id"], "cap": cap},
    )
    del provenance
    _write_campaign(output, text, records, template, problem["id"])
    return {
        "force_calls": metrics["success_only_mean"],
        "nfailed": metrics["nfailed"],
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
        "short": short,
        "records": records,
    }


def measure_start(root: Path, template_name: str, start: str) -> dict:
    """One archived start, or the first Pt-island displacement."""
    from .campaign import _check_stop, load_template, resolve_problem

    template = load_template(root, template_name)
    problem = resolve_problem(root, template["problem"])
    if str(template.get("task")) == "saddle-search":
        return _saddle_once(root, problem, template)
    cap = _check_stop(problem, template)
    force_tol = float((problem.get("convergence") or {}).get("force_norm", 1e-3))
    work = root / "results" / "pele-asv" / "work" / template["id"].replace("/", "-")
    rows = _geometries(root, problem, work)
    row = next(item for item in rows if item["name"] == start)
    return _dispatch(_spec_from_geometry(row, problem, template, int(cap), force_tol))
