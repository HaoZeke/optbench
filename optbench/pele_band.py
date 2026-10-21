"""pele NEB, DNEB, and double-ended connect on the harness potential.

pele 0.1.2 supplies the band and the transition-state refinement.
The energy and the gradient come from ``optbench.pes.evaluate``, so the
row uses the same Lennard-Jones or Morse platinum potential as every
other code on the problem. Coordinates are the free atoms. Frozen atoms
stay on the reactant. Endpoints stay in the given order: the interpolant
is the minimum-image line from ``interpolate_free``, and the connect
distance is that Cartesian distance with no permutation and no rotation.

A start succeeds when the search stops under the force-call cap, the
true Cartesian force of the candidate is within the problem norm, and
the vibrational Hessian index is 1. The stock NEB residual is only the
band stop. A tuned band uses the problem image count and the linear
interpolant, and it is given the iteration budget of connect's NEB
stage. Hybrid eigenvector following then refines an interior maximum
until the true force meets the problem norm. Climbing stays off:
``NEBForce`` returns the climbing step before the doubly nudged term.
The candidate for a plain band is the highest-energy interior image.
The candidate for a refined band or for double-ended connect is a
transition state from that search, or the highest interior image when
the refinement stored none. Hessian finite differences go through
``verification`` and are not search cost.

Each start stores the final energy, the full geometry, and the force.
Frozen atoms keep the reactant coordinate. Their stored force components
are zero. The free block is the gradient the search evaluated, and
``force_norm`` is its Cartesian norm.

Double-ended connect runs one cycle. That cycle is one local connect:
a doubly nudged band at the problem image count, refinement of the
climbing images, and the pushoff to the two adjacent minima. The problem
asks for a refined saddle, so a later cycle is not run.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .counter import ForceCallCap, PeleCounter
from .failures import classify_outcome
from .pes import evaluate, interpolate_free, load_pairs
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance, write_starts

_FD_STEP = 1e-5
_KINDS = ("neb", "dneb", "connect")

_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def _threads_one() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _quiet() -> None:
    """pele's package init attaches a debug handler. Silence it after import."""
    log = logging.getLogger("pele")
    log.setLevel(logging.ERROR)
    log.propagate = False
    logging.getLogger("sqlalchemy").setLevel(logging.ERROR)


def _require_pele() -> None:
    import pele  # noqa: F401
    from importlib.metadata import version

    _quiet()
    found = version("pele")
    if found != "0.1.2":
        raise RuntimeError(f"pele {found} is installed, the registry pin is 0.1.2")


def _registry(root: Path) -> dict:
    import tomllib

    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == "pele":
            return row
    raise ValueError("codes/registry.toml has no pele entry")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the pele registry")
    band = template["band"]
    kind = str(band["kind"])
    if kind not in _KINDS:
        raise ValueError(f"unknown pele band kind {kind}")
    problem_band = problem["band"]
    convergence = problem["convergence"]
    dneb = kind == "dneb" or (kind == "connect" and bool(band["dneb"]))
    return {
        "potential": str(problem_band["potential"]),
        "n_interior": int(problem_band["interior_images"]),
        "max_move": float(problem_band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": str(problem["geometry"]["kind"]),
        "kind": kind,
        "dneb": bool(dneb),
        "spring_k": float(band["spring_k"]),
        "max_erise": float(band["max_erise"]),
        "memory": int(band["memory"]),
        "neb_tol": float(band["neb_tol"]),
        "niter": int(band["niter"]),
        "iter_density": float(band["iter_density"]),
        "quench_tol": float(band["quench_tol"]),
        "climbing": bool(band.get("climbing", False)),
        "stop": str(band.get("stop", "residual")),
        "h0": float(band.get("h0", 0.1)),
        "seed": "0",
    }


def _key(coords: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(np.asarray(coords, dtype=np.float64).reshape(-1))
    return np.round(flat, decimals=8).tobytes()


class _Spec:
    """Free-atom view of one harness pair potential."""

    def __init__(self, kind: str, template: np.ndarray, free: np.ndarray, box: np.ndarray | None):
        self.kind = kind
        self.template = np.array(template, dtype=np.float64, copy=True)
        self.free = np.asarray(free, dtype=bool)
        self.box = None if box is None else np.asarray(box, dtype=np.float64)
        self.memory: dict[bytes, tuple[float, np.ndarray]] = {}

    def getEnergyGradient(self, coords):
        flat = np.asarray(coords, dtype=np.float64).reshape(-1)
        full = np.array(self.template, copy=True)
        full[self.free] = flat.reshape((-1, 3))
        energy, grad = evaluate(self.kind, full, self.box)
        free_grad = np.ascontiguousarray(np.asarray(grad, dtype=np.float64)[self.free].reshape(-1))
        self.memory[_key(flat)] = (float(energy), free_grad.copy())
        return float(energy), free_grad

    def recall(self, coords) -> tuple[float, np.ndarray] | None:
        return self.memory.get(_key(coords))


def _crash_record(name: str, calls: int, verify: int, crash: str | None, seed: str, cap: int) -> dict:
    """An exception before a judged point. The class comes from the call count."""
    at_cap = int(calls) >= int(cap)
    failure = classify_outcome(
        calls=int(calls),
        cap=int(cap),
        process_failed=not at_cap,
    )
    reasons: list[str] = ["cap"] if at_cap else ["crash"]
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "seed": seed,
        "force_calls": int(calls),
        "verify_calls": int(verify),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "converged": False,
        "candidate": "cap" if at_cap else "crash",
        "connected": False,
        "geometry": None,
        "force": None,
    }
    if crash and not at_cap:
        record["crash"] = crash
    return record


def _judge(
    *,
    name: str,
    seed: str,
    coords: np.ndarray | None,
    candidate: str,
    converged: bool,
    connected: bool,
    counter: PeleCounter,
    spec: _Spec,
    force_norm: float,
    cap: int,
    project_rigid: bool,
    crash: str | None,
) -> dict:
    reasons: list[str] = []
    if crash:
        reasons.append("crash")
    energy = None
    grad = None
    if coords is not None and crash is None:
        found = spec.recall(coords)
        if found is not None:
            energy, grad = found
        elif counter.calls < cap:
            try:
                energy, grad = counter.getEnergyGradient(np.asarray(coords, dtype=np.float64).reshape(-1))
                grad = np.asarray(grad, dtype=np.float64)
                energy = float(energy)
            except ForceCallCap:
                reasons.append("cap")
    if counter.calls >= cap and "cap" not in reasons:
        reasons.append("cap")
    force = None
    if grad is not None:
        force = float(np.linalg.norm(np.asarray(grad, dtype=np.float64)))
    if crash is None and (force is None or not math.isfinite(force) or force > force_norm):
        reasons.append("force_norm")
    hessian_index = None
    if crash is None and "cap" not in reasons and "force_norm" not in reasons and coords is not None:
        point = np.asarray(coords, dtype=np.float64).reshape(-1)

        def gradient(values):
            _energy, vector = counter.getEnergyGradient(values)
            return [float(item) for item in np.asarray(vector, dtype=np.float64).reshape(-1)]

        with counter.verification():
            hessian = numerical_hessian(gradient, point.tolist(), step=_FD_STEP)
        hessian_index = int(
            vibrational_index(
                np.asarray(hessian, dtype=np.float64),
                point.reshape((-1, 3)),
                project_rigid,
            )
        )
        if hessian_index != 1:
            reasons.append("hessian_index")
    geometry = None
    stored_force = None
    if coords is not None and crash is None:
        geometry, stored_force = _placed(spec, coords, grad)
    # A missing gradient is not a non-finite force. Passing NaN here would
    # be divergence, and that check runs before the harness cap.
    measured = None if force is None else float(force)
    # No stored gradient, still under the cap: the stop was not met.
    # A NaN force is a measured blow-up and stays NaN so it is divergence.
    above = crash is None and measured is None and int(counter.calls) < int(cap)
    failure = classify_outcome(
        energy=None if energy is None else float(energy),
        geometry=geometry,
        force=measured,
        calls=int(counter.calls),
        cap=int(cap),
        force_tol=float(force_norm),
        process_failed=bool(crash) and int(counter.calls) < int(cap),
        hessian_index=hessian_index,
        expected_index=1 if hessian_index is not None else None,
        force_above=True if above else None,
    )
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "seed": seed,
        "force_calls": int(counter.calls),
        "verify_calls": int(counter.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": measured,
        "hessian_index": hessian_index,
        "converged": bool(converged),
        "candidate": candidate,
        "connected": bool(connected),
        "geometry": geometry,
        "force": stored_force,
    }
    if hessian_index is not None:
        record["expected_index"] = 1
    if crash and int(counter.calls) < int(cap):
        record["crash"] = crash
    return record


def _placed(spec: _Spec, coords, grad) -> tuple[list, list | None]:
    """Full geometry and the free gradient on that geometry."""
    full = np.array(spec.template, copy=True)
    full[spec.free] = np.asarray(coords, dtype=np.float64).reshape((-1, 3))
    geometry = [[float(value) for value in row] for row in full]
    if grad is None:
        return geometry, None
    placed = np.zeros_like(full)
    placed[spec.free] = np.asarray(grad, dtype=np.float64).reshape((-1, 3))
    force = [[float(value) for value in row] for row in placed]
    return geometry, force


def _highest_index(energies) -> int:
    best = 1
    best_energy = -math.inf
    last = len(energies) - 1
    for index in range(1, max(last, 1)):
        energy = float(energies[index])
        if math.isfinite(energy) and energy > best_energy:
            best = index
            best_energy = energy
    return best


def _highest_interior(path: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, int]:
    best = _highest_index(energies)
    return np.asarray(path[best], dtype=np.float64).reshape(-1).copy(), best


def _linear(initial, final, t):
    start = np.asarray(initial, dtype=np.float64)
    stop = np.asarray(final, dtype=np.float64)
    return start + float(t) * (stop - start)


def _same_minimum(left, right) -> bool:
    first = np.asarray(left.coords, dtype=np.float64).reshape(-1)
    second = np.asarray(right.coords, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        return False
    return float(np.max(np.abs(first - second))) <= 1e-6


def _identity_mindist(coords1, coords2):
    first = np.asarray(coords1, dtype=np.float64).reshape(-1).copy()
    second = np.asarray(coords2, dtype=np.float64).reshape(-1).copy()
    return float(np.linalg.norm(first - second)), first, second


def _run_band(pot: PeleCounter, path: np.ndarray, settings: dict) -> dict:
    from pele.transition_states import NEB

    images = [np.asarray(row, dtype=np.float64).copy() for row in path]
    quench = {
        "tol": float(settings["neb_tol"]),
        "maxErise": float(settings["max_erise"]),
        "maxstep": float(settings["max_move"]),
        "nsteps": int(settings["cap"]),
        "M": int(settings["memory"]),
        "iprint": -1,
    }
    neb = None
    converged = False
    try:
        neb = NEB(
            images,
            pot,
            k=float(settings["spring_k"]),
            dneb=bool(settings["dneb"]),
            verbose=-1,
            quenchParams=quench,
            use_minimizer_callback=True,
        )
        result = neb.optimize()
        converged = bool(result.success)
    except ForceCallCap:
        converged = False
    if neb is None:
        coords, _index = _highest_interior(path, np.full(path.shape[0], np.nan))
        return {"coords": coords, "candidate": "highest", "converged": False, "connected": False}
    coords, _index = _highest_interior(neb.coords, neb.energies)
    return {"coords": coords, "candidate": "highest", "converged": converged, "connected": False}


def _candidates(neb) -> list[int]:
    """Interior maxima, highest energy first. Otherwise the highest interior image."""
    maxima = []
    last = int(neb.nimages) - 1
    for index in range(1, last):
        energy = float(neb.energies[index])
        if not math.isfinite(energy):
            continue
        left = float(neb.energies[index - 1])
        right = float(neb.energies[index + 1])
        if math.isfinite(left) and math.isfinite(right) and energy > left and energy > right:
            maxima.append(index)
    if not maxima:
        maxima = [_highest_index(neb.energies)]
    maxima.sort(key=lambda index: float(neb.energies[index]), reverse=True)
    return maxima


def _run_refined(pot: PeleCounter, path: np.ndarray, settings: dict, ndof: int, project_rigid: bool) -> dict:
    """Band, then hybrid eigenvector following on an interior maximum.

    The band uses the problem image count and the linear interpolant.
    Its step limit is ``iter_density`` times the image count, and its
    rms tolerance is ``quench_tol``. That is the first NEB pass in
    ``LocalConnect``: ``NEBDriver.run`` sets ``nsteps`` from
    ``iter_density * nimages`` and stops early when the rms is below
    the quench tolerance. The problem cap is the total number of
    potential calls, not a per-image budget. One band gradient
    evaluates every interior image.

    The band gradient is not the gradient of the energy ``NEB`` returns.
    ``NEB.optimize`` therefore asks for a large ``maxErise`` when the
    quench is ``mylbfgs``. Spring 100, memory 4, and an initial
    inverse-Hessian diagonal of 0.1 are the L-BFGS window in Trygubenko
    and Wales, J. Chem. Phys. 120, 2082 (2004), section III.3. That
    paper refines a DNEB maximum with eigenvector following. Climbing
    stays off because ``NEBForce`` returns the climbing step before the
    doubly nudged spring.

    The refinement is ``findTransitionState``. Its rms tolerance is the
    problem force norm divided by ``sqrt(ndof)``, so the Cartesian
    force meets the problem norm. Rigid modes are projected for a free
    Lennard-Jones cluster, the same switch connect uses.
    """
    from pele.optimize import mylbfgs
    from pele.optimize.optimization_exceptions import LineSearchError
    from pele.transition_states import NEB, findTransitionState

    images = [np.asarray(row, dtype=np.float64).copy() for row in path]
    n_images = int(path.shape[0])
    nsteps = int(float(settings["iter_density"]) * n_images)
    max_move = float(settings["max_move"])
    memory = int(settings["memory"])
    max_erise = float(settings["max_erise"])
    force_norm = float(settings["force_norm"])
    cap = int(settings["cap"])
    quench = {
        "tol": float(settings["quench_tol"]),
        "maxErise": max_erise,
        "maxstep": max_move,
        "nsteps": nsteps,
        "M": memory,
        "H0": float(settings["h0"]),
        "iprint": -1,
    }
    holder: dict = {}

    def _accept(coords=None, energy=None, rms=None):
        neb = holder.get("neb")
        if coords is None or neb is None:
            return
        neb.active[:, :] = np.asarray(coords, dtype=np.float64).reshape(neb.active.shape)
        for index in range(1, neb.nimages - 1):
            found = spec_recall(pot, neb.coords[index])
            if found is not None:
                neb.energies[index] = float(found[0])

    def _quench(coords, potential, **kwargs):
        events = list(kwargs.get("events") or [])
        events.append(_accept)
        kwargs["events"] = events
        kwargs["H0"] = float(settings["h0"])
        return mylbfgs(coords, potential, **kwargs)

    neb = None
    try:
        neb = NEB(
            images,
            pot,
            k=float(settings["spring_k"]),
            dneb=bool(settings["dneb"]),
            verbose=-1,
            quenchParams=quench,
            use_minimizer_callback=True,
        )
        holder["neb"] = neb
        if pot.calls < cap:
            neb.optimize(quenchRoutine=_quench)
    except (ForceCallCap, LineSearchError):
        pass
    if neb is None:
        coords, _index = _highest_interior(path, np.full(path.shape[0], np.nan))
        return {"coords": coords, "candidate": "highest", "converged": False, "connected": False}
    for index in range(neb.nimages):
        found = spec_recall(pot, neb.coords[index])
        if found is not None:
            neb.energies[index] = float(found[0])
    fallback, _index = _highest_interior(neb.coords, neb.energies)
    best_force = math.inf
    recalled = spec_recall(pot, fallback)
    if recalled is not None:
        best_force = float(np.linalg.norm(recalled[1]))
    if pot.calls >= cap or (math.isfinite(best_force) and best_force <= force_norm):
        converged = math.isfinite(best_force) and best_force <= force_norm
        return {"coords": fallback, "candidate": "highest", "converged": converged, "connected": False}

    rms_tol = force_norm / math.sqrt(float(ndof))
    chosen = fallback
    for index in _candidates(neb):
        if pot.calls >= cap:
            break
        eigenvec = neb.tangent(
            neb.energies[index],
            neb.energies[index - 1],
            neb.energies[index + 1],
            neb.distance(neb.coords[index, :], neb.coords[index - 1, :])[1],
            neb.distance(neb.coords[index, :], neb.coords[index + 1, :])[1],
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                found = findTransitionState(
                    np.asarray(neb.coords[index, :], dtype=np.float64).copy(),
                    pot,
                    eigenvec0=eigenvec,
                    tol=rms_tol,
                    nsteps=cap,
                    orthogZeroEigs=0 if project_rigid else None,
                    verbosity=0,
                    iprint=-1,
                    index_seed=int(settings["seed"]),
                    max_uphill_step=max_move,
                    max_uphill_step_initial=max_move,
                    tangentSpaceQuenchParams={
                        "maxstep": max_move,
                        "maxErise": max_erise,
                        "M": memory,
                        "iprint": -1,
                    },
                )
        except ForceCallCap:
            break
        except LineSearchError:
            continue
        coords = np.asarray(found.coords, dtype=np.float64).reshape(-1)
        recalled = spec_recall(pot, coords)
        force = math.inf if recalled is None else float(np.linalg.norm(recalled[1]))
        if math.isfinite(force) and force < best_force:
            best_force = force
            chosen = coords
        eigenval = getattr(found, "eigenval", None)
        negative = eigenval is not None and math.isfinite(float(eigenval)) and float(eigenval) < 0.0
        if bool(found.success) and negative and math.isfinite(force) and force <= force_norm:
            return {"coords": coords, "candidate": "transition-state", "converged": True, "connected": False}
    converged = math.isfinite(best_force) and best_force <= force_norm
    return {"coords": chosen, "candidate": "highest", "converged": converged, "connected": False}


def _run_connect(pot: PeleCounter, path: np.ndarray, settings: dict, ndof: int) -> dict:
    from pele.landscape import DoubleEndedConnect
    from pele.landscape.local_connect import LocalConnect
    from pele.optimize import Result, mylbfgs
    from pele.optimize.optimization_exceptions import LineSearchError
    from pele.storage import Database

    coords1 = np.asarray(path[0], dtype=np.float64).copy()
    coords2 = np.asarray(path[-1], dtype=np.float64).copy()
    energy1 = float(pot.getEnergy(coords1))
    energy2 = float(pot.getEnergy(coords2))
    database = Database(db=":memory:", compareMinima=_same_minimum)
    minimum1 = database.addMinimum(energy1, coords1)
    minimum2 = database.addMinimum(energy2, coords2)
    rms_tol = float(settings["force_norm"]) / math.sqrt(float(ndof))
    max_move = float(settings["max_move"])
    memory = int(settings["memory"])
    n_images = int(path.shape[0])
    project_rigid = bool(settings["project_rigid"])

    def quench(coords):
        try:
            return mylbfgs(
                np.asarray(coords, dtype=np.float64),
                pot,
                tol=rms_tol,
                maxstep=max_move,
                maxErise=1e-4,
                nsteps=int(settings["cap"]),
                M=memory,
                iprint=-1,
            )
        except ForceCallCap:
            result = Result()
            result.coords = np.asarray(coords, dtype=np.float64).copy()
            result.energy = float("nan")
            result.rms = float("inf")
            result.success = False
            return result

    local_params = {
        "verbosity": 0,
        "nrefine_max": n_images,
        "reoptimize_climbing": 0,
        "pushoff_params": {"quench": quench, "stepmin": min(0.01, max_move)},
        "tsSearchParams": {
            "tol": rms_tol,
            "nsteps": int(settings["cap"]),
            "orthogZeroEigs": 0 if project_rigid else None,
            "verbosity": 0,
            "iprint": -1,
            "index_seed": int(settings["seed"]),
            "max_uphill_step": max_move,
            "max_uphill_step_initial": max_move,
            "tangentSpaceQuenchParams": {
                "maxstep": max_move,
                "maxErise": float(settings["max_erise"]),
                "M": memory,
                "iprint": -1,
            },
        },
        "NEBparams": {
            "k": float(settings["spring_k"]),
            "max_images": n_images,
            "image_density": float(n_images),
            "iter_density": float(settings["iter_density"]),
            "dneb": True,
            "verbose": -1,
            "reinterpolate": 0,
            "adaptive_nimages": False,
            "adaptive_niter": False,
            "interpolator": _linear,
            "NEBquenchParams": {
                "tol": float(settings["quench_tol"]),
                "maxErise": float(settings["max_erise"]),
                "maxstep": max_move,
                "M": memory,
                "iprint": -1,
            },
        },
    }

    class _Local(LocalConnect):
        def connect(self, min1, min2):
            try:
                return super().connect(min1, min2)
            except ForceCallCap:
                return self.res

    class _Connect(DoubleEndedConnect):
        def _getLocalConnectObject(self):
            self.local = _Local(self.pot, self.mindist, **self.local_connect_params)
            return self.local

    connect = _Connect(
        minimum1,
        minimum2,
        pot,
        _identity_mindist,
        database,
        verbosity=0,
        niter=int(settings["niter"]),
        local_connect_params=local_params,
        fresh_connect=True,
        merge_minima=False,
    )
    crash = None
    try:
        connect.connect()
    except ForceCallCap:
        pass
    except LineSearchError:
        pass
    except Exception as exc:
        crash = type(exc).__name__
    found: list[tuple[float, np.ndarray]] = []
    local = getattr(connect, "local", None)
    if local is not None:
        for ts_result, _min1, _min2 in local.res.new_transition_states:
            coords = np.asarray(ts_result.coords, dtype=np.float64).reshape(-1)
            energy = float(getattr(ts_result, "energy", float("nan")))
            found.append((energy, coords))
    for transition in database.transition_states():
        coords = np.asarray(transition.coords, dtype=np.float64).reshape(-1)
        found.append((float(transition.energy), coords))
    candidate = "highest"
    chosen = None
    if found:
        finite = [item for item in found if math.isfinite(item[0])]
        pool = finite or found
        under = []
        for energy, coords in pool:
            recalled = spec_recall(pot, coords)
            if recalled is None:
                continue
            _energy, grad = recalled
            if float(np.linalg.norm(grad)) <= float(settings["force_norm"]):
                under.append((energy, coords))
        pick = under or pool
        pick.sort(key=lambda item: item[0], reverse=True)
        chosen = pick[0][1]
        candidate = "transition-state"
    elif local is not None and getattr(local, "neb", None) is not None:
        chosen, _index = _highest_interior(local.neb.coords, local.neb.energies)
        climbing = [index for index, flag in enumerate(local.neb.isclimbing) if flag]
        if climbing:
            best = max(climbing, key=lambda index: float(local.neb.energies[index]))
            chosen = np.asarray(local.neb.coords[best], dtype=np.float64).reshape(-1).copy()
            candidate = "climbing"
    if chosen is None:
        energies = []
        for row in path:
            recalled = spec_recall(pot, row)
            energies.append(float("nan") if recalled is None else float(recalled[0]))
        chosen, _index = _highest_interior(path, np.asarray(energies, dtype=np.float64))
        candidate = "highest"
    if found:
        crash = None
    connected = False
    try:
        connected = bool(connect.success())
    except Exception:
        connected = False
    return {
        "coords": chosen,
        "candidate": candidate,
        "converged": bool(found),
        "connected": connected,
        "crash": crash,
    }


def spec_recall(pot: PeleCounter, coords):
    inner = pot._inner
    recall = getattr(inner, "recall", None)
    if recall is None:
        return None
    return recall(coords)


def _search(job: dict, state: dict) -> dict:
    _require_pele()
    np.random.seed(int(job["settings"]["seed"]))
    settings = job["settings"]
    free = np.asarray(job["free"], dtype=bool)
    n_free = int(np.count_nonzero(free))
    kind = settings["potential"]
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = np.asarray(job["product"], dtype=np.float64)
    project_rigid = bool(np.all(free) and n_free >= 2 and kind == "lj")
    spec = _Spec(kind, reactant, free, box)
    counter = PeleCounter(spec, cap=int(settings["cap"]))
    state["counter"] = counter
    path = interpolate_free(reactant, product, free, int(settings["n_interior"]), box)
    ndof = int(path.shape[1])
    crash = None
    if settings["kind"] == "connect":
        tuned = dict(settings)
        tuned["project_rigid"] = project_rigid
        outcome = _run_connect(counter, path, tuned, ndof)
        crash = outcome.get("crash")
    elif settings.get("stop") == "true_force":
        outcome = _run_refined(counter, path, settings, ndof, project_rigid)
    else:
        outcome = _run_band(counter, path, settings)
    if crash:
        return _crash_record(
            job["name"],
            counter.calls,
            counter.verify_calls,
            crash,
            str(settings["seed"]),
            int(settings["cap"]),
        )
    return _judge(
        name=job["name"],
        seed=str(settings["seed"]),
        coords=outcome["coords"],
        candidate=str(outcome["candidate"]),
        converged=bool(outcome["converged"]),
        connected=bool(outcome["connected"]),
        counter=counter,
        spec=spec,
        force_norm=float(settings["force_norm"]),
        cap=int(settings["cap"]),
        project_rigid=project_rigid,
        crash=None,
    )


def _execute(job: dict) -> dict:
    _threads_one()
    _quiet()
    state: dict = {"counter": None}
    try:
        return _search(job, state)
    except ForceCallCap:
        counter = state["counter"]
        calls = 0 if counter is None else int(counter.calls)
        verify = 0 if counter is None else int(counter.verify_calls)
        seed = str(job.get("settings", {}).get("seed", "0"))
        cap = int(job.get("settings", {}).get("cap", 0))
        return _crash_record(job["name"], calls, verify, None, seed, cap)
    except Exception as exc:
        counter = state["counter"]
        calls = 0 if counter is None else int(counter.calls)
        verify = 0 if counter is None else int(counter.verify_calls)
        seed = str(job.get("settings", {}).get("seed", "0"))
        cap = int(job.get("settings", {}).get("cap", 0))
        return _crash_record(job["name"], calls, verify, type(exc).__name__, seed, cap)


def _job(pair, settings: dict) -> dict:
    return {
        "name": pair.name,
        "reactant": pair.reactant,
        "product": pair.product,
        "free": pair.free,
        "box": pair.box,
        "settings": settings,
    }


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    key = (str(Path(root).resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        loaded = load_pairs(
            geometry_archive(root, problem),
            settings["geometry"],
            settings["pairs"],
        )
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def _stamp(record: dict, problem: dict, template: dict) -> dict:
    stamped = dict(record)
    stamped["problem"] = problem["id"]
    stamped["template"] = template["id"]
    stamped["code"] = template["code"]
    stamped["code_version"] = template["code_version"]
    stamped["commit"] = template["commit"]
    stamped["algorithm"] = template["algorithm"]
    stamped["tuning"] = template["tuning"]
    return stamped


def search_named(
    root: Path,
    problem_name: str,
    template_name: str,
    start: str,
    seed: str = "0",
) -> dict:
    """One archived pair. ``seed`` seeds the eigenvector screen."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    settings["seed"] = str(seed)
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(pairs[start], settings))
    return _stamp(record, problem, template)


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, problem: dict) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness PeleCounter.getEnergyGradient",
        "# force_calls count raw potential evaluations and are not divided by the image count",
        "# the mean is round-half-even over successes; cap, force, or Hessian index other than 1 is a failure",
    ]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        images = int(settings["n_interior"])
        if images > 0:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
            lines.append("call_unit total")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row['hessian_index'] == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append(f"problem {problem['id']}")
    lines.append(f"images {settings['n_interior']}")
    return "\n".join(lines) + "\n"


def _legacy(records: list[dict]) -> dict:
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary = {
        "successes": len(successes),
        "nfailed": len(records) - len(successes),
        "n": len(records),
        "transition_states": sum(1 for row in records if row["hessian_index"] == 1),
    }
    if successes:
        summary["force_calls"] = int(round(statistics.fmean(successes)))
        summary["force_calls_min"] = min(successes)
        summary["force_calls_max"] = max(successes)
        summary["force_calls_median"] = statistics.median(successes)
    return summary


def write_outputs(
    output: Path,
    records: list[dict],
    template: dict,
    settings: dict,
    problem: dict,
) -> dict:
    """Write ``benchmark.dat``, the per-start file, and the v2 score."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(records, template, settings, problem), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    parameters = dict(settings)
    parameters["counter"] = "harness"
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score_v2(records, settings["cap"]),
        "legacy": _legacy(records),
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy = document["legacy"]
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": legacy.get("force_calls"),
        "nfailed": legacy["nfailed"],
        "starts": legacy["n"],
        "successes": legacy["successes"],
        "output": str(output),
    }


def _load_resume(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[str(row["start"])] = row
    return found


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    resume: bool = False,
    limit: int = 0,
) -> dict:
    """Run every pair in one problem and write the dat, starts, and v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != "pele-band":
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_job(pair, settings) for pair in pairs]
    if limit > 0:
        jobs = jobs[: int(limit)]
    _threads_one()
    _quiet()
    starts_path = output.with_suffix(".starts.jsonl")
    done = _load_resume(starts_path) if resume else {}
    pending = [job for job in jobs if job["name"] not in done]
    records = [done[job["name"]] for job in jobs if job["name"] in done]
    output.parent.mkdir(parents=True, exist_ok=True)

    def _keep(record: dict) -> None:
        stamped = _stamp(record, problem, template)
        records.append(stamped)
        with starts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, sort_keys=True) + "\n")
        print(
            f"{stamped['template']} {stamped['start']} calls={stamped['force_calls']} "
            f"success={stamped['success']} failure={stamped['failure']}",
            flush=True,
        )

    if pending and workers <= 1:
        for job in pending:
            _keep(_execute(job))
    elif pending:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_execute, job) for job in pending]
            for future in as_completed(futures):
                _keep(future.result())
    order = {job["name"]: index for index, job in enumerate(jobs)}
    records.sort(key=lambda row: order.get(row["start"], len(order)))
    return write_outputs(output, records, template, settings, problem)


def publish_sidecars(root: Path, directory: Path, output_dir: Path) -> list[dict]:
    """Reduce a complete asv sidecar tree into one dat per problem and template."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    paths = sorted(Path(directory).rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no asv sidecar records in {directory}")
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault((row["problem"], row["template"]), []).append(row)
    summaries = []
    commit = _registry(root)["commit"]
    for problem_name, template_name in sorted(grouped):
        rows = grouped[(problem_name, template_name)]
        problem = resolve_problem(root, problem_name)
        template = load_template(root, template_name)
        settings = _settings(template, problem, commit)
        order = list(cached_pairs(root, problem, settings))
        found = {row["start"] for row in rows}
        if found != set(order) or len(rows) != len(order):
            raise ValueError(
                f"{template_name} on {problem_name} has {len(rows)} records, the problem has {len(order)} pairs"
            )
        rank = {name: index for index, name in enumerate(order)}
        rows.sort(key=lambda row: rank[row["start"]])
        slug = f"pele-0.1.2-{template['id'].split('/')[-1]}-{problem['id'].replace('/', '-')}"
        summaries.append(write_outputs(output_dir / f"{slug}.dat", rows, template, settings, problem))
    return summaries
