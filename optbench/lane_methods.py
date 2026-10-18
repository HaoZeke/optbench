"""One start for a lane-B harness template.

Imports of the client libraries stay inside the job so a missing
package fails that method and leaves the others importable.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile

import numpy as np

from optbench.counter import ForceCallCap
from optbench.failures import classify_failure
from optbench.lane_budget import Budget, Converged, Hit
from optbench.noise import converged_under_noise, gaussian_forces
from optbench.potentials import (
    binary_lennard_jones,
    lennard_jones,
    lennard_jones_hessian,
    morse_bulk,
    morse_bulk_hessian,
)
from optbench.stationary import curvature_tolerance, kabsch_rmsd, negative_index, vibrational_eigenvalues

_HESSIAN_METHODS = frozenset({"Newton-CG", "trust-exact", "trust-krylov", "trust-ncg"})
_EIGEN_TOL = 1e-6


def run_job(job: dict) -> dict:
    """Score one start. The returned dict is an ``optbench.start/v2`` record."""
    sigma = float(job.get("sigma") or 0.0)
    reference = None
    if sigma > 0.0 and not job.get("is_reference"):
        ref_job = dict(job)
        ref_job["sigma"] = 0.0
        ref_job["is_reference"] = True
        reference = run_job(ref_job)
    error = ""
    budget = None
    overlap = None
    try:
        _silence()
        evaluate = _evaluate(job)
        if sigma > 0.0:
            rng = random.Random(int(job["noise_seed"]))
            evaluate = gaussian_forces(evaluate, sigma, rng)
        target = job.get("target")
        budget = Budget(
            evaluate,
            int(job["cap"]),
            target=None if target is None else float(target),
            window=float(job.get("energy_window") or 0.0),
        )
        _dispatch(budget, np.asarray(job["coords"], dtype=np.float64), job)
    except Hit:
        budget.hit = True
    except Converged:
        pass
    except ForceCallCap as exc:
        error = f"ForceCallCap: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _record(job, budget, error, reference, overlap=_overlap_from(budget, job) if budget is not None else None)


def _overlap_from(budget: Budget, job: dict) -> float | None:
    mode = job.get("mode")
    if mode is None or budget.point is None or budget.gradient is None:
        return None
    # The eigenvector methods store the vector on the budget point only when
    # they replace it. Overlap is attached by those methods via budget.hit
    # and a side channel on the job dict, which is local to this process.
    return job.get("_overlap")


def _silence() -> None:
    import logging

    logging.getLogger("pele").setLevel(logging.ERROR)
    logging.getLogger("lbfgs").setLevel(logging.ERROR)
    logging.getLogger("geometric").setLevel(logging.ERROR)
    logging.getLogger("optking").setLevel(logging.ERROR)


def _evaluate(job: dict):
    kind = job["kind"]
    box = None if job.get("box") is None else np.asarray(job["box"], dtype=np.float64)
    species = None if job.get("species") is None else np.asarray(job["species"], dtype=np.int64)

    def clean(coords):
        x = np.asarray(coords, dtype=np.float64).reshape(-1)
        if kind == "lj":
            return lennard_jones(x)
        if kind == "morse":
            return morse_bulk(x, box)
        if kind == "binary":
            return binary_lennard_jones(x, species)
        raise ValueError(f"no potential {kind}")

    return clean


def _dispatch(budget: Budget, x0: np.ndarray, job: dict) -> None:
    method = str(job["method"])
    if method == "scipy-minimize":
        _scipy_minimize(budget, x0, job)
    elif method == "scipy-bh":
        _scipy_basinhopping(budget, x0, job)
    elif method == "scipy-da":
        _scipy_dual_annealing(budget, x0, job)
    elif method == "scipy-de":
        _scipy_de(budget, x0, job)
    elif method == "scipy-shgo":
        _scipy_shgo(budget, x0, job)
    elif method == "scipy-eigsh":
        _scipy_eigs(budget, x0, job, which="eigsh")
    elif method == "scipy-lobpcg":
        _scipy_eigs(budget, x0, job, which="lobpcg")
    elif method == "pele-lbfgs":
        _pele_lbfgs(budget, x0, job)
    elif method == "pele-fire":
        _pele_fire(budget, x0, job)
    elif method == "pele-bh":
        _pele_bh(budget, x0, job)
    elif method == "pele-lowest":
        _pele_lowest(budget, x0, job)
    elif method == "pele-ts":
        _pele_ts(budget, x0, job)
    elif method == "sqnm":
        _sqnm(budget, x0, job)
    elif method == "geometric":
        _geometric(budget, x0, job)
    elif method == "ase-bh":
        _ase_bh(budget, x0, job)
    elif method == "ase-mh":
        _ase_mh(budget, x0, job)
    elif method == "goedecker-mh":
        _goedecker_mh(budget, x0, job)
    elif method == "jaxmd-fire":
        _jaxmd_fire(budget, x0, job)
    elif method == "scgo-bh":
        _scgo_bh(budget, x0, job)
    elif method == "torchsim-fire":
        _torchsim_fire(budget, x0, job)
    elif method == "ezga":
        _ezga(budget, x0, job)
    elif method == "agox-bh":
        _agox_bh(budget, x0, job)
    elif method == "ase-ga":
        _ase_ga(budget, x0, job)
    elif method == "nlopt-minimize":
        _nlopt_minimize(budget, x0, job)
    elif method == "optking":
        _optking(budget, x0, job)
    elif method == "pybcga":
        _pybcga(budget, x0, job)
    elif method == "gmin":
        raise RuntimeError(_gmin_reason())
    else:
        raise ValueError(f"unknown harness method {method}")


def _gmin_reason() -> str:
    return (
        "GMIN sha256 02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f "
        "builds with gfortran, and its Lennard-Jones potential is evaluated inside the "
        "executable. The PASE socket is the MLIP relay in INTERFACES/pysocket. It does "
        "not evaluate the harness potential."
    )


def _maybe_stop(budget: Budget, job: dict) -> None:
    if budget.gradient is None or budget.calls >= budget.cap:
        return
    gtol = float(job["gtol"])
    sigma = float(job.get("sigma") or 0.0)
    norm = float(np.linalg.norm(budget.gradient))
    if sigma == 0.0:
        if norm <= gtol:
            raise Converged()
        return
    components = int(budget.gradient.size)
    if converged_under_noise(
        force_norm=norm,
        energies=budget.clean_energies,
        clean_tol=gtol,
        sigma=sigma,
        components=components,
        energy_tol=float(job["energy_tol"]),
        kappa=float(job["kappa"]),
        window=int(job["noise_window"]),
    ):
        raise Converged()


def _scipy_options(method: str, cap: int) -> dict:
    if method == "TNC":
        # SciPy 1.18 warns that TNC does not accept maxiter. maxfun is the cap.
        return {"maxfun": cap, "gtol": 1e-14}
    if method == "L-BFGS-B":
        # The library ftol stops on a relative energy change near 2e-9.
        # On Morse bulk that fires while the Euclidean force is still above
        # the problem stop. Machine epsilon leaves the harness stop in charge.
        return {"maxfun": cap, "maxiter": cap, "gtol": 1e-14, "ftol": float(np.finfo(float).eps)}
    if method == "Newton-CG":
        return {"maxiter": cap}
    return {"maxiter": cap, "gtol": 1e-14}


def _scipy_minimize(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from scipy.optimize import minimize

    method = str(job["scipy_method"])
    kind = job["kind"]
    box = job.get("box")

    def fun(x):
        budget.fg(x)
        _maybe_stop(budget, job)
        return float(budget.energy)

    def jac(x):
        _energy, gradient = budget.fg(x)
        _maybe_stop(budget, job)
        return gradient

    def hess(x):
        if kind == "lj":
            return lennard_jones_hessian(x)
        return morse_bulk_hessian(x, box)

    kwargs = {
        "fun": fun,
        "x0": x0,
        "method": method,
        "jac": jac,
        "options": _scipy_options(method, int(job["cap"])),
    }
    if method in _HESSIAN_METHODS:
        kwargs["hess"] = hess
    minimize(**kwargs)


def _wrap_minimize(budget: Budget):
    import scipy.optimize as so

    real = so.minimize

    def wrapped(fun, x0, *args, **kwargs):
        result = real(fun, x0, *args, **kwargs)
        energy, gradient = budget.fg(result.x)
        budget.note_quench(result.x, float(energy), gradient, converged=bool(result.success))
        return result

    return real, wrapped


def _scipy_basinhopping(budget: Budget, x0: np.ndarray, job: dict) -> None:
    import scipy.optimize as so
    from scipy.optimize import basinhopping

    real, wrapped = _wrap_minimize(budget)
    so.minimize = wrapped
    try:
        def fun(x):
            energy, _gradient = budget.fg(x)
            return float(energy)

        basinhopping(
            fun,
            x0,
            niter=int(job["cap"]),
            rng=np.random.default_rng(int(job["seed"])),
            disp=False,
        )
    finally:
        so.minimize = real


def _scipy_dual_annealing(budget: Budget, x0: np.ndarray, job: dict) -> None:
    import scipy.optimize._dual_annealing as mod
    from scipy.optimize import dual_annealing

    real = mod.minimize
    _scipy_real, wrapped = _wrap_minimize(budget)
    mod.minimize = wrapped
    try:
        def fun(x):
            energy, _gradient = budget.fg(x)
            return float(energy)

        dual_annealing(
            fun,
            _bounds(x0, job),
            maxiter=int(job["cap"]),
            maxfun=int(job["cap"]),
            x0=x0,
            rng=np.random.default_rng(int(job["seed"])),
        )
    finally:
        mod.minimize = real


def _scipy_de(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from scipy.optimize import differential_evolution, minimize

    def fun(x):
        energy, _gradient = budget.fg(x)
        return float(energy)

    def polish(func, guess, **kwargs):
        result = minimize(func, guess, method="L-BFGS-B", bounds=_bounds(x0, job))
        energy, gradient = budget.fg(result.x)
        budget.note_quench(result.x, float(energy), gradient, converged=bool(result.success))
        return result

    differential_evolution(
        fun,
        _bounds(x0, job),
        maxiter=int(job["cap"]),
        rng=np.random.default_rng(int(job["seed"])),
        polish=polish,
        workers=1,
        x0=x0,
        disp=False,
    )


def _scipy_shgo(budget: Budget, x0: np.ndarray, job: dict) -> None:
    import scipy.optimize._shgo as mod
    from scipy.optimize import shgo

    real = mod.minimize
    _scipy_real, wrapped = _wrap_minimize(budget)
    mod.minimize = wrapped
    try:
        def fun(x):
            energy, _gradient = budget.fg(x)
            return float(energy)

        # n=100 with the default simplicial sampler is replaced by
        # 2**dimension + 1. On LJ38 that is 2**114 + 1 and the call does not
        # return. Sobol sampling uses the stated n.
        shgo(
            fun,
            _bounds(x0, job),
            n=100,
            iters=int(job["cap"]),
            sampling_method="sobol",
        )
    finally:
        mod.minimize = real


def _bounds(x0: np.ndarray, job: dict) -> list[tuple[float, float]]:
    """Box bounds when the con file has a cell, otherwise the start expanded by 5."""
    pos = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    box = job.get("box")
    if box is not None and str(job.get("layout") or "") == "con":
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)
        lower = np.zeros(3)
        upper = lengths[:3]
    else:
        lower = pos.min(axis=0) - 5.0
        upper = pos.max(axis=0) + 5.0
    return [(float(lower[axis]), float(upper[axis])) for _atom in range(pos.shape[0]) for axis in range(3)]


def _scipy_eigs(budget: Budget, x0: np.ndarray, job: dict, *, which: str) -> None:
    from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

    dimension = int(x0.size)
    eps = 1e-4

    def matvec(vector):
        step = eps * np.asarray(vector, dtype=np.float64).reshape(-1)
        _energy_p, grad_p = budget.fg(x0 + step)
        _energy_m, grad_m = budget.fg(x0 - step)
        return (grad_p - grad_m) / (2.0 * eps)

    operator = LinearOperator((dimension, dimension), matvec=matvec, dtype=np.float64)
    if which == "eigsh":
        _values, vectors = eigsh(operator, k=1, which="SA", tol=1e-4)
        vector = np.asarray(vectors[:, 0], dtype=np.float64)
    else:
        guess = np.random.default_rng(int(job["seed"])).normal(size=(dimension, 1))
        guess /= np.linalg.norm(guess)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _values, vectors = lobpcg(operator, guess, largest=False, tol=1e-4, maxiter=200)
        vector = np.asarray(vectors[:, 0], dtype=np.float64)
    mode = np.asarray(job["mode"], dtype=np.float64)
    denom = float(np.linalg.norm(vector) * np.linalg.norm(mode))
    overlap = 0.0 if denom == 0.0 else abs(float(np.dot(vector, mode))) / denom
    job["_overlap"] = float(overlap)
    budget.point = x0.copy()
    budget.energy = float(budget.energy if budget.energy is not None else 0.0)
    budget.gradient = np.zeros(dimension)
    if overlap >= float(job.get("overlap") or 0.99) and budget.calls < budget.cap:
        budget.hit = True


def _pele_potential(budget: Budget, job: dict):
    class _Potential:
        def getEnergy(self, coords):
            energy, _gradient = budget.fg(coords)
            _maybe_stop(budget, job)
            return float(energy)

        def getEnergyGradient(self, coords):
            energy, gradient = budget.fg(coords)
            _maybe_stop(budget, job)
            return float(energy), np.asarray(gradient, dtype=np.float64)

    return _Potential()


def _pele_lbfgs(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from pele.optimize import LBFGS

    potential = _pele_potential(budget, job)
    size = max(int(x0.size), 1)
    tol = float(job["gtol"]) / math.sqrt(size)
    profile = str(job.get("profile") or "default")
    memory = 100 if profile == "tuned" else 4
    maxstep = 0.2 if profile == "tuned" else 0.1
    optimizer = LBFGS(
        x0,
        potential,
        M=memory,
        maxstep=maxstep,
        tol=tol,
        nsteps=int(job["cap"]),
        iprint=0,
        fortran=False,
    )
    optimizer.run()
    result = optimizer.get_result()
    budget.point = np.asarray(result.coords, dtype=np.float64)
    budget.energy = float(result.energy)
    budget.gradient = np.asarray(result.grad, dtype=np.float64)


def _pele_fire(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from pele.optimize._fire import Fire

    potential = _pele_potential(budget, job)
    profile = str(job.get("profile") or "default")
    maxstep = 0.2 if profile == "tuned" else 0.5

    def stop(energy, gradient, tol):
        return float(np.linalg.norm(gradient)) <= float(job["gtol"])

    optimizer = Fire(x0, potential, maxstep=maxstep, alternate_stop_criterion=stop)
    optimizer.run(fmax=float(job["gtol"]), steps=int(job["cap"]))
    budget.point = np.asarray(optimizer.coords, dtype=np.float64)


def _pele_bh(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from pele.basinhopping import BasinHopping
    from pele.optimize import LBFGS, mylbfgs
    from pele.takestep import RandomDisplacement

    # Quench prints are suppressed. The potential used by the quench is the harness.
    class _Potential:
        def getEnergy(self, coords):
            energy, _gradient = budget.fg(coords)
            return float(energy)

        def getEnergyGradient(self, coords):
            energy, gradient = budget.fg(coords)
            return float(energy), np.asarray(gradient, dtype=np.float64)

    potential = _Potential()
    profile = str(job.get("profile") or "default")
    size = max(int(x0.size), 1)

    def quench(coords):
        if profile == "tuned":
            optimizer = LBFGS(
                coords,
                potential,
                M=100,
                maxstep=0.2,
                tol=0.001 / math.sqrt(size),
                nsteps=int(job["cap"]),
                iprint=0,
                fortran=False,
            )
            result = optimizer.run()
        else:
            result = mylbfgs(coords, potential, iprint=0, nsteps=int(job["cap"]))
        budget.note_quench(result.coords, float(result.energy), result.grad, converged=bool(result.success))
        return result

    BasinHopping(
        x0,
        potential,
        RandomDisplacement(stepsize=0.5 if profile == "tuned" else 1.0),
        temperature=1.0,
        quench=quench,
        outstream=None,
    ).run(int(job["cap"]))


def _pele_lowest(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """Pele Rayleigh-Ritz lowest eigenvector. The potential is the harness."""
    from pele.transition_states import findLowestEigenVector

    class _Potential:
        def getEnergy(self, coords):
            energy, _gradient = budget.fg(coords)
            return float(energy)

        def getEnergyGradient(self, coords):
            energy, gradient = budget.fg(coords)
            return float(energy), np.asarray(gradient, dtype=np.float64)

    np.random.seed(int(job["seed"]))
    result = findLowestEigenVector(np.asarray(x0, dtype=np.float64), _Potential(), nsteps=int(job["cap"]))
    vector = np.asarray(result.eigenvec, dtype=np.float64).reshape(-1)
    mode = np.asarray(job["mode"], dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(vector) * np.linalg.norm(mode))
    overlap = 0.0 if denom == 0.0 else abs(float(np.dot(vector, mode))) / denom
    job["_overlap"] = float(overlap)
    budget.point = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    budget.energy = float(budget.energy if budget.energy is not None else 0.0)
    budget.gradient = np.zeros(budget.point.size)
    if overlap >= float(job.get("overlap") or 0.99) and budget.calls < budget.cap:
        budget.hit = True


def _pele_ts(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """Pele hybrid eigenvector following. The potential is the harness.

    The library stops on the RMS gradient. That tolerance is the Euclidean
    stop divided by sqrt(3N), and nsteps is the force-call cap.
    """
    from pele.transition_states import findTransitionState

    class _Potential:
        def getEnergy(self, coords):
            energy, _gradient = budget.fg(coords)
            return float(energy)

        def getEnergyGradient(self, coords):
            energy, gradient = budget.fg(coords)
            return float(energy), np.asarray(gradient, dtype=np.float64)

    size = max(int(np.asarray(x0).size), 1)
    result = findTransitionState(
        np.asarray(x0, dtype=np.float64),
        _Potential(),
        tol=float(job["gtol"]) / math.sqrt(size),
        nsteps=int(job["cap"]),
        iprint=-1,
        verbosity=0,
        index_seed=int(job["seed"]),
    )
    budget.point = np.asarray(result.coords, dtype=np.float64).reshape(-1)
    budget.energy = float(result.energy)
    budget.gradient = np.asarray(result.grad, dtype=np.float64).reshape(-1)


def _sqnm(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from sqnm.free_or_fixed_cell_sqnm import free_sqnm

    nat = int(x0.size // 3)
    point = np.asarray(x0, dtype=np.float64).copy()
    optimizer = None
    for _step in range(int(job["cap"])):
        if budget.calls >= budget.cap:
            break
        energy, gradient = budget.fg(point)
        _maybe_stop(budget, job)
        if optimizer is None:
            # The ASE wrapper default is initial_step_size -0.01, so the first
            # displacement is 0.01 times the force. These LJ starts have force
            # norms near 1e6, and that step dissociates the cluster. The tuned
            # profile makes the first displacement length 0.01, then SQNM
            # estimates the next step because the size is negative.
            if str(job.get("profile") or "default") == "tuned":
                step = -0.01 / max(float(np.linalg.norm(gradient)), 1e-30)
            else:
                step = -0.01
            optimizer = free_sqnm(nat, step, min(10, nat * 3), 1e-3, 1e-3)
        pos = point.reshape(-1, 3).T.copy()
        forces = (-gradient).reshape(-1, 3).T.copy()
        updated = optimizer.optimizer_step(pos, float(energy), forces)
        point = np.asarray(updated, dtype=np.float64).T.reshape(-1).copy()
    budget.point = point


def _geometric(budget: Budget, x0: np.ndarray, job: dict) -> None:
    import logging.config

    import geometric.optimize as geoopt
    from geometric.engine import Engine
    from geometric.errors import EngineError, GeomOptNotConvergedError, GeomOptStructureError
    from geometric.internal import PrimitiveInternalCoordinates
    from geometric.molecule import Molecule
    from geometric.nifty import bohr2ang

    nat = int(x0.size // 3)
    pos = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    molecule = Molecule()
    molecule.elem = ["C"] * nat
    # geomeTRIC converts Angstrom coordinates to Bohr before calc_new.
    # Scaling by bohr2ang makes those Bohr numbers equal the potential coordinates.
    molecule.xyzs = [pos * float(bohr2ang)]
    molecule.build_topology()

    class _Engine(Engine):
        def calc_new(self, coords, dirname):
            energy, gradient = budget.fg(np.asarray(coords, dtype=np.float64).reshape(-1))
            _maybe_stop(budget, job)
            return {"energy": float(energy), "gradient": np.asarray(gradient, dtype=np.float64)}

    def _skip_primitives(self, molecule, connect, addcart):
        # Carbon labels on an LJ cluster make a nearly complete bond graph.
        # Primitive-internal setup then does not return. Cartesian coordinates
        # only need one fragment.
        self.frags = [list(range(molecule.na))]

    engine = _Engine(molecule)
    gtol = float(job["gtol"])
    rms = gtol / math.sqrt(max(int(x0.size), 1))
    cwd = os.getcwd()
    original_primitives = PrimitiveInternalCoordinates.makePrimitives
    original_file_config = logging.config.fileConfig

    def _quiet_file_config(*args, **kwargs):
        original_file_config(*args, **kwargs)
        for handler in logging.getLogger().handlers:
            if type(handler).__name__ == "RawStreamHandler":
                handler.setLevel(logging.ERROR)

    PrimitiveInternalCoordinates.makePrimitives = _skip_primitives
    logging.config.fileConfig = _quiet_file_config
    with tempfile.TemporaryDirectory(prefix="geometric-") as tmp:
        os.chdir(tmp)
        try:
            empty = os.path.join(tmp, "empty.in")
            open(empty, "w", encoding="utf-8").close()
            geoopt.run_optimizer(
                customengine=engine,
                input=empty,
                prefix=os.path.join(tmp, "run"),
                maxiter=int(job["cap"]),
                coordsys="cart",
                convergence_grms=rms,
                convergence_gmax=rms,
                convergence_energy=1.0e6,
                convergence_drms=1.0e6,
                convergence_dmax=1.0e6,
            )
        except (GeomOptNotConvergedError, GeomOptStructureError, EngineError, SystemExit):
            pass
        finally:
            PrimitiveInternalCoordinates.makePrimitives = original_primitives
            logging.config.fileConfig = original_file_config
            os.chdir(cwd)


def _nlopt_minimize(budget: Budget, x0: np.ndarray, job: dict) -> None:
    import nlopt

    algorithm = getattr(nlopt, str(job.get("nlopt_algorithm") or "LD_TNEWTON"))
    opt = nlopt.opt(algorithm, int(x0.size))
    # ForcedStop is its own Exception, not RuntimeError and not nlopt.exception.
    stopped = False

    def objective(x, grad):
        nonlocal stopped
        if stopped:
            if grad.size and budget.gradient is not None:
                grad[:] = budget.gradient
            opt.force_stop()
            return float(budget.energy)
        _energy, gradient = budget.fg(x)
        if grad.size:
            grad[:] = gradient
        # Raising out of the callback becomes nlopt.runtime_error and is stored
        # as a crash. force_stop keeps the point that met the Euclidean stop.
        try:
            _maybe_stop(budget, job)
        except Converged:
            stopped = True
            opt.force_stop()
        return float(budget.energy)

    opt.set_min_objective(objective)
    opt.set_maxeval(int(job["cap"]))
    try:
        opt.optimize(np.asarray(x0, dtype=np.float64).copy())
    except (nlopt.ForcedStop, nlopt.RoundoffLimited, nlopt.exception, RuntimeError, Converged):
        return


def _optking(budget: Budget, x0: np.ndarray, job: dict) -> None:
    import optking
    from optking.exceptions import AlgError, OptError

    nat = int(x0.size // 3)
    molecule = {
        "symbols": ["C"] * nat,
        "geometry": np.asarray(x0, dtype=np.float64).reshape(-1).tolist(),
        "fix_com": False,
        "fix_orientation": False,
    }
    # Redundant internals on the first LJ38 start do not finish setup.
    # Cartesian coordinates use the harness numbers as the coordinate values.
    params = {
        "opt_coordinates": "cartesian",
        "g_convergence": "gau_loose",
        "print_lvl": 0,
        "maxiter": int(job["cap"]),
    }
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="optking-") as tmp:
        os.chdir(tmp)
        try:
            helper = optking.CustomHelper(molecule, params)
            for _step in range(int(job["cap"])):
                if budget.calls >= budget.cap:
                    break
                point = np.asarray(helper.geom, dtype=np.float64).reshape(-1)
                energy, gradient = budget.fg(point)
                _maybe_stop(budget, job)
                helper.E = float(energy)
                helper.gX = gradient
                helper.compute()
                helper.take_step()
        except (OptError, AlgError, TypeError, ValueError):
            return
        finally:
            if "helper" in locals():
                helper.close()
            os.chdir(cwd)


def _pybcga(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from pele.optimize import mylbfgs

    from bcga.cluster import Cluster
    from bcga.cluster_factory import ClusterFactory
    from bcga.crossover import DeavenHo
    from bcga.mutator import MutateReplace
    from bcga.population import PopulationList
    from bcga.selector import TournamentSelector

    nat = int(x0.size // 3)

    class _Potential:
        def getEnergy(self, coords):
            energy, _gradient = budget.fg(coords)
            return float(energy)

        def getEnergyGradient(self, coords):
            energy, gradient = budget.fg(coords)
            return float(energy), np.asarray(gradient, dtype=np.float64)

    class _Minimiser:
        def __init__(self, potential):
            self.potential = potential

        def minimise(self, cluster):
            result = mylbfgs(
                cluster._coords.reshape(-1),
                self.potential,
                iprint=0,
                nsteps=int(job["cap"]),
            )
            cluster.energy = float(result.energy)
            cluster._coords = np.reshape(result.coords, (-1, 3))
            cluster.quenched = True
            budget.note_quench(
                result.coords,
                float(result.energy),
                result.grad,
                converged=bool(result.success),
            )
            return cluster

    np.random.seed(int(job["seed"]))
    minimiser = _Minimiser(_Potential())
    factory = ClusterFactory(nat, minimiser)
    selector = TournamentSelector(3)
    crossover = DeavenHo()
    mutator = MutateReplace()
    population = PopulationList(nat, factory, 10)
    population.append(Cluster(nat, np.asarray(x0, dtype=np.float64).reshape(-1, 3).copy(), minimiser))
    while len(population) < population.max_size and budget.calls < budget.cap:
        population.append(factory.get_random_cluster())
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="pybcga-") as tmp:
        os.chdir(tmp)
        try:
            population.sort_energy()
            for _generation in range(10):
                if budget.calls >= budget.cap or budget.hit:
                    break
                for _offspring in range(8):
                    indices = selector.select(population)
                    child = crossover.get_offspring(population[indices[0]], population[indices[1]])
                    population.append(child)
                for member in list(population):
                    if np.random.uniform(0, 1) < 0.1:
                        population.append(mutator.get_mutant(member))
                population.truncate()
                population.sort_energy()
        finally:
            os.chdir(cwd)


def _atoms(x0: np.ndarray, job: dict, budget: Budget):
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    pos = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    box = job.get("box")
    if box is None:
        span = pos.max(axis=0) - pos.min(axis=0)
        cell = np.maximum(span + 10.0, 20.0)
    else:
        cell = np.asarray(box, dtype=np.float64).reshape(-1)[:3]

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)
            energy, gradient = budget.fg(atoms.positions.reshape(-1))
            self.results["energy"] = float(energy)
            self.results["forces"] = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)

    atoms = Atoms(numbers=[6] * pos.shape[0], positions=pos, cell=cell, pbc=False)
    atoms.calc = _Calc()
    return atoms


def _ase_quench_hit(atoms, budget: Budget, fmax: float) -> None:
    forces = atoms.get_forces()
    energy = float(atoms.get_potential_energy())
    atomic = float(np.linalg.norm(forces, axis=1).max()) if len(atoms) else 0.0
    budget.note_quench(
        atoms.positions.reshape(-1),
        energy,
        -forces.reshape(-1),
        converged=atomic < float(fmax),
    )


def _ase_bh(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from ase import units
    from ase.optimize import FIRE, LBFGS
    from ase.optimize.basin import BasinHopping

    np.random.seed(int(job["seed"]))
    atoms = _atoms(x0, job, budget)
    profile = str(job.get("profile") or "default")
    if profile == "tuned":
        temperature = 1.0
        optimizer = LBFGS
        fmax = 0.001
        dr = 0.5
    else:
        temperature = 100 * units.kB
        optimizer = FIRE
        fmax = 0.1
        dr = 0.1
    with tempfile.TemporaryDirectory(prefix="ase-bh-") as tmp:
        hopper = BasinHopping(
            atoms,
            temperature=temperature,
            optimizer=optimizer,
            fmax=fmax,
            dr=dr,
            logfile=os.path.join(tmp, "bh.log"),
            trajectory=os.path.join(tmp, "lowest.traj"),
            optimizer_logfile=os.path.join(tmp, "opt.log"),
            local_minima_trajectory=os.path.join(tmp, "local.traj"),
        )
        _ase_quench_hit(atoms, budget, fmax)
        while budget.calls < budget.cap:
            hopper.run(1)
            _ase_quench_hit(atoms, budget, fmax)


def _uniform_translated_dmax(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """MinimaHopping's translated pairing when every atom has one symbol.

    One row is one anchor. The earliest remaining atom wins a tie, which
    is what ``argmin`` returns.
    """
    standard = pos1 - pos1[0]
    count = pos2.shape[0]
    shifted = pos2[None, :, :] - pos2[:, None, :]
    alive = np.ones((count, count), dtype=bool)
    furthest = np.zeros(count, dtype=np.float64)
    rows = np.arange(count)
    for position in standard:
        dist = np.linalg.norm(shifted - position, axis=2)
        dist = np.where(alive, dist, np.inf)
        chosen = np.argmin(dist, axis=1)
        np.maximum(furthest, dist[rows, chosen], out=furthest)
        alive[rows, chosen] = False
    return float(furthest.min())


def _ase_compare_dmax(atoms1, atoms2, *, translate: bool) -> float:
    """Same pairing as ``ComparePositions`` for an unperiodic cluster.

    The library walks atoms in Python and copies the cluster once per
    anchor. The distances and the tie break (the earliest atom) match
    that walk. Periodic cells stay on the library routine.
    """
    pos1 = np.asarray(atoms1.positions, dtype=np.float64)
    pos2 = np.asarray(atoms2.positions, dtype=np.float64)
    sym1 = np.asarray(atoms1.get_chemical_symbols())
    sym2 = np.asarray(atoms2.get_chemical_symbols())
    uniform = (
        translate
        and sym1.size > 0
        and sym1.shape == sym2.shape
        and bool(np.all(sym1 == sym1[0]))
        and bool(np.all(sym2 == sym1[0]))
    )
    if uniform:
        return _uniform_translated_dmax(pos1, pos2)

    def paired(left: np.ndarray, right: np.ndarray, left_symbols: np.ndarray, right_symbols: np.ndarray) -> float:
        alive = np.ones(right.shape[0], dtype=bool)
        dmax = 0.0
        for position, symbol in zip(left, left_symbols, strict=True):
            dist = np.linalg.norm(right - position, axis=1)
            dist = np.where(alive & (right_symbols == symbol), dist, np.inf)
            index = int(np.argmin(dist))
            if dist[index] > dmax:
                dmax = float(dist[index])
            alive[index] = False
        return dmax

    if not translate:
        return paired(pos1, pos2, sym1, sym2)
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for index, symbol in enumerate(sym1.tolist()):
        counts[symbol] = counts.get(symbol, 0) + 1
        first.setdefault(symbol, index)
    # set() iteration order is the library's choice for the anchor element.
    least = ""
    least_count = 10**9
    for element in set(sym1.tolist()):
        if counts[element] < least_count:
            least = element
            least_count = counts[element]
    anchor = first[least]
    standard = pos1 - pos1[anchor]
    indices2 = [index for index, symbol in enumerate(sym2.tolist()) if symbol == least]
    dmax = 10**9
    for index in indices2:
        shifted = pos2 - pos2[index]
        dmax = min(dmax, paired(standard, shifted, sym1, sym2))
    return float(dmax)


def _ase_mh(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """ASE MinimaHopping. The hop limit is the force cap.

    The library writes a trajectory and a log on every MD and quench step.
    Those files do not change the search, and on LJ38 they dominate the
    runtime, so this run keeps the minima list and drops the per-step files.
    Comparing two carbon clusters uses the same nearest-atom pairing as
    ``ComparePositions`` without the per-anchor copies.
    A force call at the cap still raises and ends the start.
    """
    from ase import units
    from ase.md import VelocityVerlet, thermalize_momenta
    from ase.optimize.minimahopping import ComparePositions, MinimaHopping

    def _fast_call(self, left, right):
        left = left.copy()
        right = right.copy()
        if np.any(left.pbc) or np.any(right.pbc):
            if not self._translate:
                return ComparePositions._optbench_indistinguishable(self, left, right)
            return ComparePositions._optbench_translated(self, left, right)
        return _ase_compare_dmax(left, right, translate=bool(self._translate))

    if not getattr(ComparePositions, "_optbench_fast", False):
        ComparePositions._optbench_indistinguishable = ComparePositions._indistinguishable_compare
        ComparePositions._optbench_translated = ComparePositions._translated_compare
        ComparePositions.__call__ = _fast_call
        ComparePositions._optbench_fast = True

    np.random.seed(int(job["seed"]))
    atoms = _atoms(x0, job, budget)
    profile = str(job.get("profile") or "default")
    fmax = 0.001 if profile == "tuned" else 0.05
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="ase-mh-") as tmp:
        os.chdir(tmp)
        try:
            class _Stop(MinimaHopping):
                def _optimize(self):
                    if budget.calls >= budget.cap:
                        raise ForceCallCap(f"force-call cap {budget.cap}")
                    self._atoms.set_momenta(np.zeros(self._atoms.get_momenta().shape))
                    with self._optimizer(self._atoms, trajectory=None, logfile=None) as opt:
                        opt.run(fmax=self._fmax)

                def _molecular_dynamics(self, resume=None):
                    if budget.calls >= budget.cap:
                        raise ForceCallCap(f"force-call cap {budget.cap}")
                    mincount = 0
                    energies: list[float] = []
                    oldpositions: list[np.ndarray] = []
                    thermalize_momenta(
                        self._atoms,
                        self._temperature,
                        exact_temperature=True,
                        rng=self._rng,
                    )
                    dyn = VelocityVerlet(self._atoms, timestep=self._timestep * units.fs)
                    while mincount < self._mdmin:
                        dyn.run(1)
                        energies.append(float(self._atoms.get_potential_energy()))
                        passedmin = self._passedminimum(energies)
                        if passedmin:
                            mincount += 1
                        oldpositions.append(self._atoms.positions.copy())
                    self._atoms.positions = oldpositions[passedmin[0]]

                def _check_results(self):
                    super()._check_results()
                    _ase_quench_hit(self._atoms, budget, fmax)

            hopper = _Stop(atoms, T0=1000.0, Ediff0=0.5, fmax=fmax, rng=np.random.default_rng(int(job["seed"])))
            hopper(totalsteps=int(job["cap"]))
        finally:
            os.chdir(cwd)


def _ase_ga(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from ase.optimize import LBFGS
    from ase_ga.particle_mutations import RandomMutation

    np.random.seed(int(job["seed"]))
    atoms = _atoms(x0, job, budget)
    fmax = 0.001
    rng = np.random.RandomState(int(job["seed"]))
    mutation = RandomMutation(length=2.0, rng=rng)

    def relax(candidate) -> None:
        opt = LBFGS(candidate, logfile=None)
        opt.run(fmax=fmax, steps=int(job["cap"]))
        _ase_quench_hit(candidate, budget, fmax)

    atoms.info["confid"] = "start"
    relax(atoms)
    while budget.calls < budget.cap:
        parent = atoms.copy()
        parent.info["confid"] = "start"
        parent.calc = atoms.calc
        child, _desc = mutation.get_new_individual([parent])
        if child is None:
            continue
        child.calc = atoms.calc
        relax(child)
        if budget.quench_energy is not None and (
            budget.energy is None or budget.quench_energy < float(atoms.get_potential_energy())
        ):
            atoms.set_positions(child.get_positions())


def _goedecker_mh(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """Goedecker minima hopping through its ASE calculator.

    ASE 3.29 removed the communicator argument and rejects the old serial
    string. NumPy 2 eig returns a complex dtype for the real inertia tensor,
    which the torque removal cannot subtract.
    """
    import logging
    import warnings

    import minimahopping.md.soften as soften
    import minimahopping.minhop as mh
    import minimahopping.opt.optim as optim
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.parallel import world
    from minimahopping.minhop import Minimahopping

    real_velocity = mh.MaxwellBoltzmannDistribution
    real_inertia = soften.moment_of_inertia
    real_optimization = optim.optimization

    def velocity(atoms, *args, **kwargs):
        comm = kwargs.pop("communicator", None)
        if comm == "serial":
            kwargs.setdefault("comm", world)
        elif comm is not None:
            kwargs.setdefault("comm", comm)
        return real_velocity(atoms, *args, **kwargs)

    def inertia(positions, masses):
        values, vectors = real_inertia(positions, masses)
        return np.real(np.asarray(values)), np.real(np.asarray(vectors))

    def optimization(*args, **kwargs):
        positions, lattice, noise, trajectory, nsteps, epot_max = real_optimization(*args, **kwargs)
        energy, gradient = budget.fg(np.asarray(positions, dtype=np.float64).reshape(-1))
        forces = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
        atomic = float(np.max(np.abs(forces))) if forces.size else 0.0
        threshold = float(kwargs.get("max_force_threshold", 0.005))
        budget.note_quench(positions, float(energy), gradient, converged=atomic <= threshold)
        return positions, lattice, noise, trajectory, nsteps, epot_max

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)
            energy, gradient = budget.fg(atoms.positions.reshape(-1))
            self.results["energy"] = float(energy)
            self.results["forces"] = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)

    pos = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    atoms = Atoms(numbers=[6] * pos.shape[0], positions=pos, cell=[40.0, 40.0, 40.0], pbc=False)
    atoms.calc = _Calc()
    np.random.seed(int(job["seed"]))
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    cwd = os.getcwd()
    mh.MaxwellBoltzmannDistribution = velocity
    soften.moment_of_inertia = inertia
    optim.optimization = optimization
    try:
        with tempfile.TemporaryDirectory(prefix="ase-mh-") as tmp:
            os.chdir(tmp)
            with Minimahopping(
                atoms,
                use_MPI=False,
                logLevel=logging.ERROR,
                verbose_output=False,
                collect_md_data=False,
                write_graph_output=False,
            ) as hopper:
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    saved = sys.stdout
                    sys.stdout = devnull
                    try:
                        hopper(totalsteps=int(job["cap"]))
                    finally:
                        sys.stdout = saved
    finally:
        mh.MaxwellBoltzmannDistribution = real_velocity
        soften.moment_of_inertia = real_inertia
        optim.optimization = real_optimization
        os.chdir(cwd)


def _agox_bh(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """AGOX basin hopping. The ASE calculator is the harness potential.

    BasinHoppingConfig defaults are Metropolis temperature 0.25 eV, a rattle
    of a quarter of the atoms by at most 3 angstrom, and BFGS at fmax 0.05
    for 100 steps. The archived start is quenched and seeded into the sampler.
    Without that seed the rattle falls back to a random cluster.
    """
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    fmax = 0.05
    steps = 100

    class _Halt(BaseException):
        """Leave AGOX. Its evaluator catches Exception, including Hit."""

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)
            if budget.hit or budget.calls >= budget.cap:
                raise _Halt()
            point = np.asarray(atoms.positions, dtype=np.float64).reshape(-1)
            try:
                energy, gradient = budget.fg(point)
            except ForceCallCap:
                raise _Halt() from None
            forces = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
            self.results["energy"] = float(energy)
            self.results["forces"] = forces
            atomic = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
            try:
                budget.note_quench(point, float(energy), gradient, converged=atomic < fmax)
            except Hit:
                raise _Halt() from None

    pos = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    nat = int(pos.shape[0])
    margin = 15.0
    lo = pos.min(axis=0) - margin
    span = np.maximum(pos.max(axis=0) + margin - lo, 20.0)
    template = Atoms(cell=np.diag(span + 6.0), pbc=False)
    calc = _Calc()
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="agox-") as tmp:
            os.chdir(tmp)
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                saved_out, saved_err = sys.stdout, sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    _agox_search(job, pos, nat, lo, span, template, calc, fmax, steps, tmp)
                except _Halt:
                    return
                finally:
                    sys.stdout = saved_out
                    sys.stderr = saved_err
    finally:
        os.chdir(cwd)


def _agox_search(job, pos, nat, lo, span, template, calc, fmax, steps, tmp) -> None:
    from uuid import uuid4

    from agox import AGOX
    from agox.candidates import StandardCandidate
    from agox.databases import Database
    from agox.environments import Environment
    from agox.evaluators import LocalOptimizationEvaluator
    from agox.generators import RattleGenerator
    from agox.samplers import MetropolisSampler

    environment = Environment(
        template=template,
        numbers=[6] * nat,
        confinement_cell=np.diag(span),
        confinement_corner=lo,
        print_report=False,
    )
    seed = StandardCandidate(
        template=template,
        positions=pos,
        numbers=[6] * nat,
        cell=template.cell,
        pbc=False,
    )
    seed.calc = calc
    identity = str(uuid4())
    seed.add_meta_information("uuid", identity)
    seed.add_meta_information("spawn_uuids", [identity])
    seed.add_meta_information("search_iteration", 0)
    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "candidates"},
        store_trajectory=False,
        optimizer_run_kwargs={"fmax": fmax, "steps": steps},
        order=2,
        constraints=environment.get_constraints(),
    )
    evaluator.evaluate_candidate(seed)
    sampler = MetropolisSampler(temperature=0.25, order=3)
    sampler.set_sample([seed])
    generator = RattleGenerator(
        **environment.get_confinement(),
        environment=environment,
        sampler=sampler,
        order=1,
        n_rattle=max(int(nat * 0.25), 1),
        rattle_amplitude=3,
    )
    database = Database(filename=os.path.join(tmp, "db.db"), order=4)
    search = AGOX(generator, database, sampler, evaluator, seed=int(job["seed"]), report=False)
    search.run(N_iterations=int(job["cap"]))


def _ezga(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """EZGA from the archived start.

    The relaxer defaults are ase_calculator's: FIRE, fmax 0.05, 100 steps,
    no MD. The population is 16 copies of the start, and the run is 10
    generations of rattle. A quench is a point whose maximum atomic force
    is inside fmax.
    """
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.io import write
    from ezga.evaluator.objective import objective_energy
    from ezga.factory import build_default_engine, load_config
    from ezga.simulator.ase_calculator import ase_calculator
    from ezga.variation.mutation import mutation_rattle

    fmax = 0.05
    population = int(job.get("ezga_population") or 16)
    generations = int(job.get("ezga_generations") or 10)

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)
            point = np.asarray(atoms.positions, dtype=np.float64).reshape(-1)
            energy, gradient = budget.fg(point)
            forces = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
            self.results["energy"] = float(energy)
            self.results["forces"] = forces
            atomic = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
            budget.note_quench(point, float(energy), gradient, converged=atomic < fmax)

    pos = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    frames = [
        Atoms(numbers=[6] * pos.shape[0], positions=pos, cell=[40.0, 40.0, 40.0], pbc=False)
        for _copy in range(population)
    ]
    runner = ase_calculator(
        calculator=_Calc(),
        nvt_steps=0,
        fmax=fmax,
        steps_max=100,
        device="cpu",
        log_interval=0,
        write_interval=0,
    )
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="ezga-") as tmp:
        os.chdir(tmp)
        try:
            path = os.path.join(tmp, "population.xyz")
            write(path, frames)
            config = {
                "output_path": os.path.join(tmp, "engine"),
                "max_generations": generations,
                "resume": False,
                "rng": int(job["seed"]),
                "population": {
                    "dataset_path": path,
                    "db_path": os.path.join(tmp, "db"),
                    "filter_duplicates": False,
                    "collision_factor": 0.0,
                },
                "multiobjective": {"size": population},
                "variation": {"initial_mutation_rate": 1.0},
                "evaluator": {"objectives_funcs": [objective_energy()]},
                "mutation_funcs": [mutation_rattle(std=0.2, seed=int(job["seed"]))],
                "simulator": {"calculator": runner},
            }
            build_default_engine(load_config(config)).run()
        finally:
            os.chdir(cwd)


def _torchsim_fire(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """TorchSim FIRE. The model forward call is the harness potential."""
    import torch
    import torch_sim as ts
    from ase import Atoms
    from torch_sim.models.interface import ModelInterface

    stopped = False

    class _Model(ModelInterface):
        def __init__(self):
            super().__init__()
            self._device = torch.device("cpu")
            self._dtype = torch.float64
            self._compute_forces = True
            self._compute_stress = False

        def forward(self, state, **kwargs):
            nonlocal stopped
            point = np.asarray(state.positions.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            if stopped or budget.calls >= budget.cap:
                stopped = True
                energy = 0.0 if budget.energy is None else float(budget.energy)
                gradient = np.zeros(point.size) if budget.gradient is None else budget.gradient
            else:
                try:
                    energy, gradient = budget.fg(point)
                    _maybe_stop(budget, job)
                except Converged:
                    stopped = True
                    energy = float(budget.energy)
                    gradient = budget.gradient
                except ForceCallCap:
                    stopped = True
                    energy = 0.0 if budget.energy is None else float(budget.energy)
                    gradient = np.zeros(point.size) if budget.gradient is None else budget.gradient
            return {
                "energy": torch.tensor([float(energy)], dtype=self.dtype),
                "forces": torch.tensor(
                    -np.asarray(gradient, dtype=np.float64).reshape(-1, 3),
                    dtype=self.dtype,
                ),
            }

    def converged(state, last_energy=None):
        return torch.tensor([stopped])

    pos = np.asarray(x0, dtype=np.float64).reshape(-1, 3)
    atoms = Atoms(numbers=[6] * pos.shape[0], positions=pos, pbc=False)
    ts.optimize(
        atoms,
        _Model(),
        optimizer=ts.Optimizer.fire,
        max_steps=int(job["cap"]),
        convergence_fn=converged,
        steps_between_swaps=1,
        pbar=False,
    )


def _scgo_bh(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """SCGO basin hopping. The attached ASE calculator is the harness potential.

    The library's covalent-connectivity gate treats a reduced Lennard-Jones
    cluster as fragmented, so fragmentation is allowed. Hits are the quenches.
    """
    import scgo.algorithms.basinhopping_go as bhmod
    from scgo.algorithms.basinhopping_go import bh_go

    real = bhmod.perform_local_relaxation

    def relaxation(atoms, calculator, optimizer, fmax, steps, *args, **kwargs):
        energy = real(atoms, calculator, optimizer, fmax, steps, *args, **kwargs)
        point = np.asarray(atoms.positions, dtype=np.float64).reshape(-1)
        energy, gradient = budget.fg(point)
        forces = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
        atomic = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
        budget.note_quench(point, float(energy), gradient, converged=atomic < float(fmax))
        return energy

    bhmod.perform_local_relaxation = relaxation
    cwd = os.getcwd()
    try:
        atoms = _atoms(x0, job, budget)
        with tempfile.TemporaryDirectory(prefix="scgo-") as tmp:
            os.chdir(tmp)
            bh_go(
                atoms,
                output_dir=tmp,
                verbosity=0,
                rng=np.random.default_rng(int(job["seed"])),
                allow_cluster_fragmentation=True,
            )
    finally:
        bhmod.perform_local_relaxation = real
        os.chdir(cwd)


def _jaxmd_fire(budget: Budget, x0: np.ndarray, job: dict) -> None:
    """JAX MD FIRE. The harness gradient is the force via pure_callback."""
    import jax
    import jax.numpy as jnp
    from jax import pure_callback
    from jax_md import minimize, space

    jax.config.update("jax_enable_x64", True)
    stop = False

    def force_fn(positions):
        def host(x):
            nonlocal stop
            if stop or budget.calls >= budget.cap:
                stop = True
                return np.zeros(np.shape(x), dtype=np.float64)
            try:
                _energy, gradient = budget.fg(np.asarray(x, dtype=np.float64).reshape(-1))
                _maybe_stop(budget, job)
            except Converged:
                stop = True
            except ForceCallCap:
                stop = True
                return np.zeros(np.shape(x), dtype=np.float64)
            return -np.asarray(gradient, dtype=np.float64).reshape(np.shape(x))

        return pure_callback(host, jax.ShapeDtypeStruct(positions.shape, jnp.float64), positions)

    _displacement, shift = space.free()
    init_fn, apply_fn = minimize.fire_descent(force_fn, shift)
    state = init_fn(jnp.asarray(np.asarray(x0, dtype=np.float64).reshape(-1, 3)))
    for _step in range(int(job["cap"])):
        if stop or budget.calls >= budget.cap:
            return
        state = apply_fn(state)


def _record(job: dict, budget: Budget | None, error: str, reference: dict | None, overlap: float | None) -> dict:
    calls = 0 if budget is None else budget.calls
    verify = 0
    energy = None if budget is None else budget.energy
    gradient = None if budget is None else budget.gradient
    point = None if budget is None else budget.point
    cap = int(job["cap"])
    gtol = float(job.get("gtol") or 0.0)
    family = str(job["family"])
    sigma = float(job.get("sigma") or 0.0)
    if budget is not None and point is not None:
        try:
            with budget.counter.verification():
                energy, gradient = budget.fg(point)
            verify = int(budget.counter.verify_calls)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    force_norm = None if gradient is None else float(np.linalg.norm(gradient))
    finite = (
        point is not None
        and all(math.isfinite(float(value)) for value in point)
        and energy is not None
        and math.isfinite(float(energy))
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    hessian_index = None
    if family in {"minimize", "tsopt"} and point is not None and finite and job["kind"] in {"lj", "morse"}:
        try:
            hessian_index = _index(job["kind"], point, job.get("box"), gtol)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    at_force = False
    if finite and calls < cap and not (error and "ForceCallCap" not in error and "force-call cap" not in error):
        if sigma == 0.0:
            at_force = force_norm is not None and force_norm <= gtol
        else:
            at_force = converged_under_noise(
                force_norm=float(force_norm),
                energies=budget.clean_energies if budget is not None else [],
                clean_tol=gtol,
                sigma=sigma,
                components=int(point.size),
                energy_tol=float(job["energy_tol"]),
                kappa=float(job["kappa"]),
                window=int(job["noise_window"]),
            )
    reasons: list[str] = []
    crash = bool(error) and "ForceCallCap" not in error and "force-call cap" not in error
    hit = bool(budget and budget.hit and calls < cap and finite)
    if family == "eigen":
        hit = bool(overlap is not None and overlap >= float(job.get("overlap") or 0.99) and calls < cap and not crash)
    if crash:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif calls >= cap or (error and ("ForceCallCap" in error or "force-call cap" in error)):
        reasons.append("cap")
    elif family == "minimize":
        if not at_force:
            reasons.append("force_norm")
        else:
            if hessian_index not in (None, 0):
                reasons.append("hessian_index")
            if sigma > 0.0:
                reasons.extend(_noise_reasons(job, point, energy, reference))
    elif family == "tsopt":
        if not at_force:
            reasons.append("force_norm")
        elif hessian_index not in (None, 1):
            reasons.append("hessian_index")
    elif family == "globalopt":
        if not hit:
            if budget is not None and budget.quenched:
                reasons.append("energy")
            else:
                reasons.append("force_norm")
    elif family == "eigen":
        if not hit:
            reasons.append("energy")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": int(calls),
        "verify_calls": int(verify),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": force_norm,
        "hessian_index": hessian_index,
        "converged": bool(at_force) if family in {"minimize", "tsopt"} else bool(hit),
    }
    if overlap is not None:
        record["overlap"] = float(overlap)
    if sigma > 0.0:
        record["sigma"] = sigma
    if error:
        record["error"] = error
    return record


def _noise_reasons(job: dict, point: np.ndarray, energy: float, reference: dict | None) -> list[str]:
    reasons = []
    tol = float(job["energy_tol"])
    if reference is None or not reference.get("converged") or reference.get("energy") is None:
        reasons.append("energy")
        reasons.append("rmsd")
        return reasons
    if abs(float(energy) - float(reference["energy"])) > tol:
        reasons.append("energy")
    # The reference coordinates are not on the public record. Compare through
    # a private field set only for this in-process call.
    ref_point = reference.get("_point")
    if ref_point is None:
        reasons.append("rmsd")
        return reasons
    moved = np.asarray(point, dtype=np.float64).reshape(-1, 3).tolist()
    origin = np.asarray(ref_point, dtype=np.float64).reshape(-1, 3).tolist()
    if kabsch_rmsd(moved, origin) > tol:
        reasons.append("rmsd")
    return reasons


def _index(kind: str, point: np.ndarray, box, gtol: float) -> int:
    if kind == "lj":
        matrix = lennard_jones_hessian(point)
        values = vibrational_eigenvalues(matrix, np.asarray(point, dtype=np.float64).reshape(-1, 3), rotations=True)
        tolerance = curvature_tolerance(gtol, float(np.linalg.norm(matrix)))
        return negative_index(values, tolerance)
    matrix = morse_bulk_hessian(point, box)
    values = np.linalg.eigvalsh(matrix).tolist()
    return negative_index(values, _EIGEN_TOL)


# The reference minimization has to expose its final coordinates to the
# noisy comparison without writing them into the campaign record. The
# public record builder drops unknown keys, so stash the point here by
# patching _record's return for in-process reference jobs.
_record_impl = _record


def _record(job: dict, budget: Budget | None, error: str, reference: dict | None, overlap: float | None) -> dict:
    record = _record_impl(job, budget, error, reference, overlap)
    if job.get("is_reference") and budget is not None and budget.point is not None:
        record["_point"] = budget.point.tolist()
    return record
