"""One start for a lane-B harness template.

Imports of the client libraries stay inside the job so a missing
package fails that method and leaves the others importable.
"""

from __future__ import annotations

import math
import os
import random
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
    elif method == "sqnm":
        _sqnm(budget, x0, job)
    elif method == "geometric":
        _geometric(budget, x0, job)
    elif method == "ase-bh":
        _ase_bh(budget, x0, job)
    elif method == "ase-mh":
        _ase_mh(budget, x0, job)
    elif method == "ase-ga":
        _ase_ga(budget, x0, job)
    elif method == "nlopt-minimize":
        _nlopt_minimize(budget, x0, job)
    elif method == "optking":
        _optking(budget, x0, job)
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
        return {"maxfun": cap, "maxiter": cap, "gtol": 1e-14}
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


def _ase_mh(budget: Budget, x0: np.ndarray, job: dict) -> None:
    from ase.optimize.minimahopping import MinimaHopping

    np.random.seed(int(job["seed"]))
    atoms = _atoms(x0, job, budget)
    profile = str(job.get("profile") or "default")
    fmax = 0.001 if profile == "tuned" else 0.05
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="ase-mh-") as tmp:
        os.chdir(tmp)
        try:
            class _Stop(MinimaHopping):
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
    if family == "minimize" and point is not None and finite and job["kind"] in {"lj", "morse"}:
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
        "converged": bool(at_force) if family == "minimize" else bool(hit),
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
