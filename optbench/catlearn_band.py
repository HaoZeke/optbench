"""CatLearn ML-NEB on the double-ended band problems.

A force call is one ``calculate`` of the spec potential. The Gaussian
process that proposes the next image is not a force call. Endpoint
evaluations are calls. Finite-difference checks of the Hessian go
through ``HarnessCounter.verification`` and stay out of the search cost.

The iteration budget, the acquisition function, the spring, and the
surrogate step are the library defaults unless a template sets them.
The force stop is the problem L2 norm on the free atoms. Fixed atoms
contribute a zero residual so the library's per-atom maximum sees the
same constrained gradient.

Each template declares a wall-clock ceiling. A start that reaches it
stops as a cap failure and keeps the harness call count at that stop.
A later geometry within 1e-6 of an earlier acquisition is recorded as
cycling. The library step budget stays 500.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem, write_starts
from .counter import HarnessCounter
from .failures import classify_failure
from .pes import evaluate, load_pairs
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance

_CODE_ID = "catlearn"
_PIN = "6a307f32dbef18da9cd848f632d082b8e4f117b1"
_VERSION = "0.7.0"
_FD_STEP = 1e-5
_CYCLE_ABS = 1e-6
_INSTALLED = False
_STATE: dict = {"path": None, "counter": None, "name": "", "deadline": None}


class _WallClock(RuntimeError):
    """A start reached the wall-clock ceiling declared on its template."""
_PAIRS: dict[tuple[str, str], dict] = {}


def _threads(count: str) -> None:
    """Pin BLAS threads for this process and for children spawned later.

    OpenBLAS reads the environment at load time. A later assignment does
    not move an already loaded library, so the scipy and numpy copies are
    updated through their C entry points when those libraries are present.
    """
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(count)
    import ctypes
    import glob
    import sys

    wanted = int(count)
    roots = [Path(entry) for entry in sys.path if entry]
    patterns = (
        "scipy.libs/libscipy_openblas*.so",
        "numpy.libs/libscipy_openblas*.so",
        "scipy.libs/libopenblas*.so",
        "numpy.libs/libopenblas*.so",
    )
    symbols = (
        "scipy_openblas_set_num_threads",
        "openblas_set_num_threads",
        "mkl_set_num_threads",
    )
    for root in roots:
        site = root / "site-packages"
        base = site if site.is_dir() else root
        for pattern in patterns:
            for library in glob.glob(str(base / pattern)):
                try:
                    loaded = ctypes.CDLL(library)
                except OSError:
                    continue
                for symbol in symbols:
                    setter = getattr(loaded, symbol, None)
                    if setter is None:
                        continue
                    setter.argtypes = [ctypes.c_int]
                    setter(wanted)
                    break


def gaussian_xx_gradients(m1, kwidth, k):
    """Covariance block for ``k(x, x)`` with feature derivatives.

    Feature counts above 30 follow the library's loop, evaluated a
    block at a time. Smaller counts use the library's broadcast
    expression. Both paths return the same matrix as CatLearn 0.7.0.
    """
    features = np.asarray(m1, dtype=np.float64)
    width = np.asarray(kwidth, dtype=np.float64)
    gram = np.asarray(k, dtype=np.float64)
    count, dimension = features.shape
    if dimension <= 30:
        return _xx_broadcast(features, width, gram)
    return _xx_blocks(features, width, gram, count, dimension)


def gaussian_xxp_gradients(m1, m2, kwidth, k):
    """Covariance block for ``k(x, x')`` with feature derivatives."""
    left = np.asarray(m1, dtype=np.float64)
    right = np.asarray(m2, dtype=np.float64)
    width = np.asarray(kwidth, dtype=np.float64)
    gram = np.asarray(k, dtype=np.float64)
    invsq = width ** (-2)
    diff = right[None, :, :] - left[:, None, :]
    tiled = (-(diff * invsq) * gram[:, :, None]).reshape(left.shape[0], -1)
    return np.concatenate([gram, tiled], axis=1)


def _xx_broadcast(features: np.ndarray, width: np.ndarray, gram: np.ndarray) -> np.ndarray:
    count, dimension = features.shape
    cross = np.zeros((count, count * dimension))
    curvature = np.zeros((count * dimension, count * dimension))
    invsq = width ** (-2)
    identity = np.identity(dimension) * invsq
    for row in range(count):
        delta = invsq * (features - features[row])
        block = (delta.T * gram[row]).T
        start = dimension * row
        cross[:, start : start + dimension] = block
        outer = identity - (delta[:, None, :] * delta[:, :, None])
        curvature[:, start : start + dimension] = (outer * (gram[row, None, None].T)).reshape(-1, dimension)
    return np.block([[gram, cross], [cross.T, curvature]])


def _xx_blocks(
    features: np.ndarray,
    width: np.ndarray,
    gram: np.ndarray,
    count: int,
    dimension: int,
) -> np.ndarray:
    invsq = np.atleast_1d(width ** (-2))
    if invsq.size == 1:
        eye = np.eye(dimension) * float(invsq[0])
    else:
        eye = np.diag(invsq)
    width_full = count * (1 + dimension)
    out = np.zeros((width_full, width_full))
    out[:count, :count] = gram
    for row in range(count):
        delta = invsq * (features - features[row])
        cross = delta * gram[row, :, None]
        col = count + row * dimension
        out[:count, col : col + dimension] = cross
        out[col : col + dimension, :count] = cross.T
        blocks = (eye - delta[:, :, None] * delta[:, None, :]) * gram[row, :, None, None]
        for col_index in range(row, count):
            dest = count + col_index * dimension
            block = blocks[col_index]
            out[col : col + dimension, dest : dest + dimension] = block
            if col_index != row:
                out[dest : dest + dimension, col : col + dimension] = block.T
    return out


def _scaled_gaussian_block(kdict: dict, matrix: np.ndarray) -> np.ndarray:
    """Gaussian block with gradients, multiplied by the kernel scale."""
    from catlearn.regression.gpfunctions.kernel_setup import kdict2list
    from catlearn.regression.gpfunctions.kernels import gaussian_kernel

    scaling, theta = kdict2list(kdict, int(matrix.shape[1]))
    if len(scaling) == 0:
        scale = 1.0
    else:
        scale = float(np.asarray(scaling, dtype=np.float64).reshape(-1)[0])
    block = gaussian_kernel(
        theta=np.asarray(theta, dtype=np.float64),
        log_scale=False,
        m1=matrix,
        m2=None,
        eval_gradients=True,
    )
    scaled = np.array(block, dtype=np.float64, copy=True)
    scaled *= scale
    return scaled


def _add_regularization(cov: np.ndarray, regularization) -> None:
    if regularization is None:
        return
    shift = float(np.asarray(regularization, dtype=np.float64).reshape(-1)[0])
    if shift == 0.0:
        return
    index = np.arange(cov.shape[0])
    cov[index, index] += shift * shift


def _covariance_with_diagonal_noise(original):
    """Covariance matching CatLearn on a training set.

    The Gaussian block is scaled in place. Noise and regularization sit on
    the diagonal. The matrix matches the dense sum.
    """

    def get_covariance(
        kernel_list,
        log_scale,
        matrix1,
        matrix2=None,
        regularization=None,
        eval_gradients=False,
    ):
        noises = [item for item in kernel_list if item.get("type") == "noise_multi"]
        if (
            len(noises) != 1
            or matrix2 is not None
            or not eval_gradients
            or "features" in noises[0]
            or "operation" in noises[0]
        ):
            return original(
                kernel_list,
                log_scale,
                matrix1,
                matrix2=matrix2,
                regularization=regularization,
                eval_gradients=eval_gradients,
            )
        others = [item for item in kernel_list if item.get("type") != "noise_multi"]
        direct = (
            not log_scale
            and len(others) == 1
            and others[0].get("type") == "gaussian"
            and "features" not in others[0]
            and "operation" not in others[0]
        )
        if direct:
            cov = _scaled_gaussian_block(others[0], np.asarray(matrix1, dtype=np.float64))
            _add_regularization(cov, regularization)
        else:
            cov = original(
                others,
                log_scale,
                matrix1,
                matrix2=None,
                regularization=regularization,
                eval_gradients=True,
            )
        from catlearn.regression.gpfunctions.kernel_setup import kdict2list

        scaling, hyperparameters = kdict2list(noises[0], np.shape(matrix1)[1])
        theta = np.asarray(hyperparameters, dtype=np.float64)
        if len(scaling) == 0:
            scale = 1.0
        else:
            scale = float(np.asarray(scaling, dtype=np.float64).reshape(-1)[0])
        if log_scale:
            theta = np.exp(theta)
            scale = float(np.exp(scale))
        count = int(np.shape(matrix1)[0])
        dimension = int(np.shape(matrix1)[1])
        energy = np.arange(count)
        cov[energy, energy] += scale * float(theta[0]) ** 2
        gradient = np.arange(count, count + count * dimension)
        cov[gradient, gradient] += scale * float(theta[1]) ** 2
        return cov

    get_covariance.__wrapped_noise__ = True
    return get_covariance


def _install_likelihood(likelihood_module, gp_module) -> None:
    """Same log marginal likelihood, with a direct triangular solve.

    The factor comes from ``cho_factor``. The solve uses the lower triangle
    through LAPACK ``dpotrs``. Hyperparameter optimization sees the same value.
    """
    original = likelihood_module.log_marginal_likelihood
    if getattr(original, "_optbench_fast", False):
        return

    def log_marginal_likelihood(
        theta,
        train_matrix,
        targets,
        kernel_list,
        scale_optimizer,
        eval_gradients,
        cinv=None,
        eval_jac=False,
    ):
        if eval_jac or cinv is not None:
            return original(
                theta,
                train_matrix,
                targets,
                kernel_list,
                scale_optimizer,
                eval_gradients,
                cinv,
                eval_jac,
            )
        from catlearn.regression.gpfunctions.kernel_setup import list2kdict
        from scipy.linalg import LinAlgError, cho_factor
        from scipy.linalg.lapack import dpotrs

        listed = list2kdict(theta, kernel_list)
        count = len(targets)
        rhs = np.asfortranarray(np.asarray(targets, dtype=np.float64).reshape(count, 1))
        gram = likelihood_module.get_covariance(
            kernel_list=listed,
            matrix1=train_matrix,
            regularization=theta[-1],
            log_scale=scale_optimizer,
            eval_gradients=eval_gradients,
        )
        factor, _lower = cho_factor(gram, overwrite_a=False, lower=True, check_finite=True)
        solved, info = dpotrs(np.asfortranarray(factor), rhs, lower=1, overwrite_b=0)
        if info != 0:
            raise LinAlgError(f"dpotrs failed with info {info}")
        datafit = -0.5 * np.dot(rhs.T, solved)
        complexity = -np.log(np.diag(factor)).sum()
        normalization = -count * np.log(2 * np.pi) / 2.0
        value = (datafit + complexity + normalization).sum(-1)
        return -value

    log_marginal_likelihood._optbench_fast = True
    likelihood_module.log_marginal_likelihood = log_marginal_likelihood
    gp_module.log_marginal_likelihood = log_marginal_likelihood


def install_runtime() -> None:
    """Load CatLearn 0.7.0 and apply the ASE 3.29 and NumPy 2 shims."""
    global _INSTALLED
    if _INSTALLED:
        return
    import ase.mep as mep

    sys.modules["ase.neb"] = mep
    import catlearn
    import catlearn.optimize.constraints as constraints
    import catlearn.optimize.mlneb as mlneb
    import catlearn.regression.gpfunctions.kernels as kernels

    if catlearn.__version__ != _VERSION:
        raise RuntimeError(f"catlearn version {catlearn.__version__} is not {_VERSION}")
    source = Path(catlearn.__file__).resolve().parents[1]
    if (source / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if not revision.startswith(_PIN):
            raise RuntimeError(f"catlearn checkout {revision} is not {_PIN}")
    _patch_mask(constraints)
    kernels.gaussian_xx_gradients = gaussian_xx_gradients
    kernels.gaussian_xxp_gradients = gaussian_xxp_gradients
    import catlearn.regression.gaussian_process as gp_module
    import catlearn.regression.gpfunctions.covariance as covariance
    import catlearn.regression.gpfunctions.log_marginal_likelihood as likelihood
    import catlearn.regression.gpfunctions.uncertainty as uncertainty

    fast_covariance = _covariance_with_diagonal_noise(covariance.get_covariance)
    covariance.get_covariance = fast_covariance
    gp_module.get_covariance = fast_covariance
    likelihood.get_covariance = fast_covariance
    uncertainty.get_covariance = fast_covariance
    _install_likelihood(likelihood, gp_module)
    mlneb.ASECalc.calculate = _surrogate_calculate
    mlneb.ASECalc._predict_scalar = _predict_scalar
    mlneb.ASECalc._predict_vector = _predict_vector
    mlneb.ASECalc._masked_row = _masked_row
    _wrap_progress(mlneb)
    _INSTALLED = True


def _patch_mask(constraints) -> None:
    original = constraints.create_mask

    def create_mask(ini, constraint_list):
        index = original(ini, constraint_list)
        return np.asarray(index, dtype=int).reshape(-1)

    constraints.create_mask = create_mask


def _surrogate_calculate(self, atoms=None, properties=None, system_changes=None):
    """Finite-difference forces of the Gaussian process.

    The stencil matches ``ASECalc.calculate``. Prediction components are
    scalars, and the mask index is one integer per free coordinate, so
    NumPy 2 can assign them.
    """
    from ase.calculators.calculator import Calculator, all_changes

    if properties is None:
        properties = ["energy", "forces"]
    if system_changes is None:
        system_changes = all_changes
    Calculator.calculate(self, atoms, properties, system_changes)
    flat = np.asarray(self.atoms.get_positions(), dtype=np.float64).reshape(-1)
    index = np.asarray(self.ind_constraints, dtype=int).reshape(-1)
    masked = self._masked_row(flat)
    energy = self._predict_scalar(masked.reshape(1, -1)) + float(self.scaling)
    trial = np.repeat(masked.reshape(1, -1), len(index), axis=0)
    values = flat[index]
    diagonal = np.arange(len(index))
    plus = trial.copy()
    minus = trial.copy()
    plus[diagonal, diagonal] = values + self.fs
    minus[diagonal, diagonal] = values - self.fs
    high = self._predict_vector(plus)
    low = self._predict_vector(minus)
    slope = (high - low) / (2.0 * self.fs)
    gradient = np.zeros(flat.shape[0], dtype=np.float64)
    gradient[index] = slope
    self.results["energy"] = float(energy)
    self.results["forces"] = -gradient.reshape((-1, 3))


def _predict_scalar(self, row: np.ndarray) -> float:
    prediction = self.gp.predict(test_fp=row, uncertainty=False)["prediction"]
    return float(np.asarray(prediction, dtype=np.float64).reshape(-1)[0])


def _predict_vector(self, rows: np.ndarray) -> np.ndarray:
    prediction = self.gp.predict(test_fp=rows)["prediction"]
    return np.asarray(prediction, dtype=np.float64).reshape(-1)


def _masked_row(self, flat: np.ndarray) -> np.ndarray:
    from catlearn.optimize.constraints import apply_mask

    _original, masked = apply_mask(list_to_mask=[flat], mask_index=self.ind_constraints)
    return np.asarray(masked[0], dtype=np.float64).reshape(-1)


def _wrap_progress(mlneb) -> None:
    train = mlneb.train_gp_model
    evaluate_point = mlneb.eval_and_append

    def train_gp_model(*args, **kwargs):
        _raise_if_due()
        began = time.perf_counter()
        trained = train(*args, **kwargs)
        _heartbeat(f"train {time.perf_counter() - began:.3f}s")
        _raise_if_due()
        return trained

    def eval_and_append(self, interesting_point):
        _raise_if_due()
        began = time.perf_counter()
        evaluate_point(self, interesting_point)
        gradient = np.asarray(self.list_gradients[-1], dtype=np.float64)
        force = float(np.linalg.norm(gradient))
        _heartbeat(f"eval {time.perf_counter() - began:.3f}s force {force:.6g}")

    mlneb.train_gp_model = train_gp_model
    mlneb.eval_and_append = eval_and_append


def _heartbeat(stage: str) -> None:
    path = _STATE.get("path")
    counter = _STATE.get("counter")
    if not path or counter is None:
        return
    payload = {
        "start": _STATE.get("name") or "",
        "force_calls": int(counter.calls),
        "verify_calls": int(counter.verify_calls),
        "stage": stage,
    }
    target = Path(path)
    temporary = target.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(target)
    except OSError:
        return
    print(
        f"{payload['start']} calls={payload['force_calls']} {stage}",
        file=sys.stderr,
        flush=True,
    )


def _registry(root: Path) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == _CODE_ID:
            return row
    raise ValueError("codes/registry.toml has no catlearn entry")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if str(template.get("commit") or "") != commit:
        raise ValueError(f"{template['id']} commit does not match the catlearn registry")
    if template.get("code_version") != _VERSION:
        raise ValueError(f"{template['id']} version is not {_VERSION}")
    band = problem["band"]
    convergence = problem["convergence"]
    controls = template.get("mlneb") or {}
    interior = int(band["interior_images"])
    spring = controls.get("spring", None)
    if "wall_seconds" not in controls:
        raise ValueError(f"{template['id']} declares no wall-clock ceiling")
    wall_seconds = float(controls["wall_seconds"])
    if not np.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError(f"{template['id']} wall-clock ceiling must be positive")
    return {
        "potential": str(band["potential"]),
        "n_interior": interior,
        "n_images": interior + 2,
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "full_n": int(problem["sample"]["pairs"]),
        "interpolation": str(controls.get("interpolation", "linear")),
        "mic": bool(controls.get("mic", False)),
        "acquisition": str(controls.get("acquisition", "acq_5")),
        "unc_convergence": float(controls.get("unc_convergence", 0.05)),
        "dt": float(controls.get("dt", 0.025)),
        "ml_steps": int(controls.get("ml_steps", 750)),
        "max_step": float(controls.get("max_step", 0.25)),
        "sequential": bool(controls.get("sequential", False)),
        "neb_method": str(controls.get("neb_method", "improvedtangent")),
        "steps": int(controls.get("steps", 500)),
        "wall_seconds": wall_seconds,
        "spring": None if spring in (None, "") else float(spring),
        "problem": problem["id"],
        "template": template["id"],
        "code": template["code"],
        "code_version": template["code_version"],
        "commit": template["commit"],
        "algorithm": template["algorithm"],
        "tuning": template["tuning"],
    }


def cached_pairs(root: Path, problem: dict) -> dict:
    """Archived pairs for ``problem``, in archive order, keyed by name."""
    key = (str(Path(root).resolve()), str(problem["id"]))
    found = _PAIRS.get(key)
    if found is None:
        loaded = load_pairs(
            geometry_archive(root, problem),
            problem["geometry"]["kind"],
            int(problem["sample"]["pairs"]),
        )
        found = {pair.name: pair for pair in loaded}
        _PAIRS[key] = found
    return found


def _symbol(kind: str) -> str:
    if kind == "morse_pt":
        return "Pt"
    return "H"


def _atoms(positions: np.ndarray, free: np.ndarray, box: np.ndarray | None, symbol: str):
    from ase import Atoms
    from ase.constraints import FixAtoms

    atoms = Atoms(symbols=[symbol] * len(positions), positions=np.asarray(positions, dtype=np.float64))
    if box is not None:
        atoms.set_cell(np.asarray(box, dtype=np.float64))
        atoms.set_pbc(True)
    fixed = np.flatnonzero(~np.asarray(free, dtype=bool))
    if len(fixed):
        atoms.set_constraint(FixAtoms(indices=[int(index) for index in fixed]))
    return atoms


def _spec_calculator(kind: str, box: np.ndarray | None, free: np.ndarray):
    """One ASE calculator for the spec potential.

    Fixed atoms keep a zero force. ``calculate`` stores energy and forces
    together, so the following force read is the same evaluation.
    """
    from ase.calculators.calculator import Calculator, all_changes

    free_mask = np.asarray(free, dtype=bool)
    lengths = None if box is None else np.asarray(box, dtype=np.float64)

    class _Spec(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            if properties is None:
                properties = ["energy", "forces"]
            if system_changes is None:
                system_changes = all_changes
            Calculator.calculate(self, atoms, properties, system_changes)
            coords = np.asarray(self.atoms.get_positions(), dtype=np.float64)
            energy, gradient = evaluate(kind, coords, lengths)
            forces = -np.asarray(gradient, dtype=np.float64)
            forces[~free_mask] = 0.0
            self.results["energy"] = float(energy)
            self.results["forces"] = forces

    return _Spec()


def _attach(kind: str, box, free, counter: HarnessCounter):
    from .counter import attach_ase_counter

    calculator = _spec_calculator(kind, box, free)
    return attach_ase_counter(calculator, cap=counter.cap, counter=counter)


def _is_cap(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "force-call cap" in str(exc)


def _due() -> bool:
    deadline = _STATE.get("deadline")
    return deadline is not None and time.perf_counter() >= float(deadline)


def _raise_if_due() -> None:
    if _due():
        raise _WallClock("wall-clock ceiling")


def _project_rigid(free: np.ndarray, box: np.ndarray | None) -> bool:
    return box is None and bool(np.all(free))


def _complete_rows(ml) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if ml is None:
        return None
    trains = getattr(ml, "list_train", None)
    targets = getattr(ml, "list_targets", None)
    gradients = getattr(ml, "list_gradients", None)
    if trains is None or targets is None or gradients is None:
        return None
    trains = np.asarray(trains, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    gradients = np.asarray(gradients, dtype=np.float64)
    count = min(len(trains), len(targets), len(gradients))
    if count < 1:
        return None
    return trains[:count], targets[:count], gradients[:count]


def _cycle_report(trains) -> tuple[bool, int]:
    """Count later geometries within ``_CYCLE_ABS`` of an earlier acquisition."""
    if trains is None:
        return False, 0
    points = np.asarray(trains, dtype=np.float64)
    if points.ndim != 2 or len(points) < 2:
        return False, 0
    repeats = 0
    for index in range(1, len(points)):
        distances = np.linalg.norm(points[:index] - points[index], axis=1)
        if np.any(distances <= _CYCLE_ABS):
            repeats += 1
    return repeats > 0, repeats


def _judge(job: dict, counter: HarnessCounter, ml, cap_hit: bool, wall_hit: bool = False) -> dict:
    settings = job["settings"]
    name = job["name"]
    free = np.asarray(job["free"], dtype=bool)
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    kind = settings["potential"]
    force_norm = float(settings["force_norm"])
    cap = int(settings["cap"])
    wall_limit = float(settings["wall_seconds"])
    rows = _complete_rows(ml)
    reasons: list[str] = []
    energy = float("nan")
    force = float("nan")
    index = None
    if wall_hit or cap_hit or counter.calls >= cap:
        reasons.append("cap")
    if wall_hit:
        reasons.append("wall_clock")
    if rows is None:
        reasons.append("crash")
        return _record(
            name,
            counter,
            cap_hit,
            wall_hit,
            reasons,
            energy,
            force,
            index,
            "missing",
            (False, 0),
            wall_limit,
        )
    trains, targets, gradients = rows
    coords = trains[-1].reshape((-1, 3))
    gradient = gradients[-1].reshape((-1, 3))
    energy = float(targets[-1])
    free_gradient = gradient[free].reshape(-1)
    force = float(np.linalg.norm(free_gradient))
    if not np.isfinite(energy) or not np.isfinite(force):
        reasons.append("force_norm")
    elif force > force_norm:
        reasons.append("force_norm")
    if "crash" not in reasons and "cap" not in reasons and "force_norm" not in reasons:
        free_point = coords[free].reshape(-1)
        template = np.array(coords, dtype=np.float64, copy=True)

        def gradient_of(free_flat):
            counter.charge(tuple(float(value) for value in free_flat))
            full = np.array(template, dtype=np.float64, copy=True)
            full[free] = np.asarray(free_flat, dtype=np.float64).reshape((-1, 3))
            _energy, full_gradient = evaluate(kind, full, box)
            return full_gradient[free].reshape(-1)

        with counter.verification():
            hessian = numerical_hessian(gradient_of, free_point.tolist(), step=_FD_STEP)
        index = vibrational_index(
            np.asarray(hessian, dtype=np.float64),
            coords[free],
            _project_rigid(free, box),
        )
        if index != 1:
            reasons.append("hessian_index")
    return _record(
        name,
        counter,
        cap_hit,
        wall_hit,
        reasons,
        energy,
        force,
        index,
        "last-acquired",
        _cycle_report(trains),
        wall_limit,
    )


def _record(
    name,
    counter,
    cap_hit,
    wall_hit,
    reasons,
    energy,
    force,
    index,
    candidate,
    cycle,
    wall_limit,
) -> dict:
    failure = classify_failure(reasons)
    finite_energy = float(energy) if np.isfinite(energy) else None
    finite_force = float(force) if np.isfinite(force) else None
    cycling, repeats = cycle
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(counter.calls),
        "verify_calls": int(counter.verify_calls),
        "call_unit": "total",
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": list(reasons),
        "energy": finite_energy,
        "force_norm": finite_force,
        "hessian_index": None if index is None else int(index),
        "converged": failure is None,
        "candidate": candidate,
        "cap_hit": bool(cap_hit or wall_hit),
        "wall_hit": bool(wall_hit),
        "wall_limit": float(wall_limit),
        "cycling": bool(cycling),
        "cycle_repeats": int(repeats),
    }


def _crash_record(name: str, calls: int, verify: int, exc: BaseException) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify),
        "call_unit": "total",
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "crash": type(exc).__name__,
        "crash_detail": traceback.format_exception_only(type(exc), exc)[-1].strip(),
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "converged": False,
        "candidate": "crash",
        "cap_hit": False,
        "wall_hit": False,
        "wall_limit": None,
        "cycling": False,
        "cycle_repeats": 0,
    }


def _freeze_endpoint(atoms, kind: str, box, free, counter: HarnessCounter) -> None:
    """Count one spec evaluation and store it on a single-point calculator.

    ML-NEB writes the endpoints and reads them back. The single-point
    calculator keeps that read off the harness tally.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms.calc = _attach(kind, box, free, counter)
    energy = float(atoms.get_potential_energy())
    forces = np.array(atoms.get_forces(), dtype=np.float64, copy=True)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)


def _search(job: dict, counter: HarnessCounter) -> dict:
    from ase.io import read

    install_runtime()
    from catlearn.optimize.mlneb import MLNEB

    settings = job["settings"]
    free = np.asarray(job["free"], dtype=bool)
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    kind = settings["potential"]
    symbol = _symbol(kind)
    reactant = _atoms(job["reactant"], free, box, symbol)
    product = _atoms(job["product"], free, box, symbol)
    _freeze_endpoint(reactant, kind, box, free, counter)
    _freeze_endpoint(product, kind, box, free, counter)
    calculator = _attach(kind, box, free, counter)
    _heartbeat("endpoints")
    ml = None
    with tempfile.TemporaryDirectory(prefix="catlearn-") as tmp:
        previous = os.getcwd()
        os.chdir(tmp)
        try:
            with open("mlneb.log", "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                ml = MLNEB(
                    reactant,
                    product,
                    n_images=int(settings["n_images"]),
                    k=settings["spring"],
                    interpolation=settings["interpolation"],
                    mic=bool(settings["mic"]),
                    neb_method=settings["neb_method"],
                    ase_calc=calculator,
                    restart=False,
                )
                stored = read("initial.traj", "-1:")[-1]
                if stored.calc is None or "energy" not in stored.calc.results:
                    raise RuntimeError("endpoint trajectory dropped the spec evaluation")
                try:
                    ml.run(
                        fmax=float(settings["force_norm"]),
                        unc_convergence=float(settings["unc_convergence"]),
                        steps=int(settings["steps"]),
                        acquisition=settings["acquisition"],
                        dt=float(settings["dt"]),
                        ml_steps=int(settings["ml_steps"]),
                        max_step=float(settings["max_step"]),
                        sequential=bool(settings["sequential"]),
                        full_output=False,
                    )
                except _WallClock:
                    return _judge(job, counter, ml, False, True)
                except Exception as exc:
                    if _is_cap(exc):
                        return _judge(job, counter, ml, True, False)
                    raise
        finally:
            os.chdir(previous)
    if _due():
        return _judge(job, counter, ml, False, True)
    return _judge(job, counter, ml, counter.calls >= int(settings["cap"]), False)


def search_one(job: dict) -> dict:
    """One pair. The returned record is one harness tally."""
    settings = job["settings"]
    name = str(job["name"])
    counter = HarnessCounter(cache=False, cap=int(settings["cap"]))
    _STATE["counter"] = counter
    _STATE["name"] = name
    _STATE["path"] = job.get("heartbeat")
    began = time.perf_counter()
    _STATE["deadline"] = began + float(settings["wall_seconds"])
    try:
        record = _search(job, counter)
    except Exception as exc:
        record = _crash_record(name, counter.calls, counter.verify_calls, exc)
    finally:
        _STATE["deadline"] = None
    record["wall_seconds"] = time.perf_counter() - began
    record["wall_limit"] = float(settings["wall_seconds"])
    record.setdefault("wall_hit", False)
    record.setdefault("cycling", False)
    record.setdefault("cycle_repeats", 0)
    record["seed"] = "0"
    record["problem"] = settings["problem"]
    record["template"] = settings["template"]
    record["code"] = settings["code"]
    record["code_version"] = settings["code_version"]
    record["commit"] = settings["commit"]
    record["algorithm"] = settings["algorithm"]
    record["tuning"] = settings["tuning"]
    _heartbeat("done")
    return record


def _job(pair, settings: dict, heartbeat: str | None) -> dict:
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64),
        "product": np.asarray(pair.product, dtype=np.float64),
        "free": np.asarray(pair.free, dtype=bool),
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64),
        "settings": settings,
        "heartbeat": heartbeat,
    }


def _execute(job: dict) -> dict:
    return search_one(job)


def _load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _number(value: float) -> str:
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, partial: bool) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness spec-potential calculate",
        "# force_calls count spec-potential evaluations and are not divided by the image count",
        "# Gaussian-process predictions stay outside the tally",
        "# the mean is round-half-even over successes; a Hessian index other than 1 is a failure",
        "# a start at the wall-clock ceiling is a cap failure and keeps its harness call count",
        "# cycling is a later geometry within 1e-6 of an earlier acquisition",
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
    lines.append(f"transition_states {sum(1 for row in records if row.get('hessian_index') == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append(f"problem {settings['problem']}")
    lines.append("counter harness")
    lines.append(f"images {settings['n_interior']}")
    lines.append(f"wall_limit {_number(settings['wall_seconds'])}")
    if partial:
        lines.append("partial true")
    note = str(template.get("note") or "").strip()
    if note:
        lines.append(f"note {note}")
    return "\n".join(lines) + "\n"


def write_outputs(output: Path, records: list[dict], template: dict, settings: dict, problem: dict) -> dict:
    """Write ``benchmark.dat``, the per-start file, and the v2 score."""
    output.parent.mkdir(parents=True, exist_ok=True)
    full = len(records) >= int(settings["full_n"])
    output.write_text(_dat(records, template, settings, not full), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    parameters = dict(settings)
    parameters["counter"] = "harness"
    parameters["seed"] = "0"
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
        "score": score_v2(
            [
                {
                    "force_calls": row["force_calls"],
                    "success": row["success"],
                    "failure": row.get("failure"),
                    "reasons": list(row.get("reasons") or []),
                }
                for row in records
            ],
            settings["cap"],
        ),
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    mean = int(round(statistics.fmean(successes))) if successes else None
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": mean,
        "nfailed": int(document["score"]["nfailed"]),
        "starts": len(records),
        "successes": len(successes),
        "output": str(output),
    }


def _order(records: list[dict], names: list[str]) -> list[dict]:
    rank = {name: index for index, name in enumerate(names)}
    return sorted(records, key=lambda row: rank.get(row["start"], len(rank)))


def _dead_worker(job: dict, exc: BaseException) -> dict:
    calls = 0
    verify = 0
    path = job.get("heartbeat")
    if path and Path(path).is_file():
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            calls = int(payload.get("force_calls") or 0)
            verify = int(payload.get("verify_calls") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            calls = 0
    record = _crash_record(str(job["name"]), calls, verify, exc)
    settings = job["settings"]
    record["seed"] = "0"
    record["problem"] = settings["problem"]
    record["template"] = settings["template"]
    record["code"] = settings["code"]
    record["code_version"] = settings["code_version"]
    record["commit"] = settings["commit"]
    record["algorithm"] = settings["algorithm"]
    record["tuning"] = settings["tuning"]
    record["wall_seconds"] = None
    record["wall_limit"] = float(settings["wall_seconds"])
    record["wall_hit"] = False
    record["cycling"] = False
    record["cycle_repeats"] = 0
    return record


def search_named(root: Path, problem_name: str, template_name: str, start: str, seed: str = "0") -> dict:
    """One archived pair. ``seed`` is recorded and does not change the search."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = cached_pairs(root, problem)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = search_one(_job(pairs[start], settings, None))
    record["seed"] = str(seed)
    return record


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Run the archived pairs and write the dat, the starts, and the v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != "catlearn-mlneb":
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem).values())
    names = [pair.name for pair in pairs]
    starts_path = output.with_suffix(".starts.jsonl")
    done = _load_records(starts_path) if resume else []
    finished = {row["start"] for row in done}
    pending = [pair for pair in pairs if pair.name not in finished]
    if limit > 0:
        pending = pending[:limit]
    heartbeat_dir = Path(tempfile.mkdtemp(prefix="catlearn-hb-"))
    jobs = [_job(pair, settings, str(heartbeat_dir / f"{pair.name}.json")) for pair in pending]
    threads = os.environ.get("OPTBENCH_BLAS_THREADS")
    if threads is None:
        threads = "1" if workers > 1 else "8"
    _threads(threads)
    records = list(done)
    if workers <= 1:
        for job in jobs:
            records.append(_execute(job))
            write_outputs(output, _order(records, names), template, settings, problem)
    else:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_execute, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = _dead_worker(job, exc)
                records.append(record)
                write_outputs(output, _order(records, names), template, settings, problem)
    ordered = _order(records, names)
    return write_outputs(output, ordered, template, settings, problem)


def asv_document(commit: str, benchmarks: list[dict], machine: dict, when_ms: int) -> dict:
    """One asv 0.6 result file for these benchmark classes.

    Each benchmark dict carries ``name`` and ``starts``. A start carries
    ``seed``, ``start``, ``force_calls``, ``success``, ``failure``, and
    ``wall_seconds``. Parameter axes use ``repr``, in archive order.
    """
    columns = [
        "result",
        "params",
        "version",
        "started_at",
        "duration",
        "stats_ci_99_a",
        "stats_ci_99_b",
        "stats_q_25",
        "stats_q_75",
        "stats_number",
        "stats_repeat",
        "samples",
        "profile",
    ]
    results = {}
    for benchmark in benchmarks:
        starts = list(benchmark["starts"])
        seeds = [repr(str(row["seed"])) for row in starts]
        names = [repr(str(row["start"])) for row in starts]
        if len(set(seeds)) != 1:
            raise ValueError(f"{benchmark['name']} has more than one seed")
        params = [[seeds[0]], names] if starts else [[], []]
        version = commit

        def _track(method: str) -> list:
            values = []
            for row in starts:
                failure = row.get("failure")
                if method == "track_force_calls":
                    values.append(float(row["force_calls"]))
                elif method == "track_success":
                    values.append(1.0 if row["success"] else 0.0)
                else:
                    label = method.removeprefix("track_failure_")
                    values.append(1.0 if failure == label else 0.0)
            return [values, params, version, when_ms, 0.0]

        prefix = benchmark["name"]
        for method in (
            "track_force_calls",
            "track_success",
            "track_failure_cap",
            "track_failure_wrong_stationary",
            "track_failure_divergence",
            "track_failure_crash",
        ):
            results[f"{prefix}.{method}"] = _track(method)
        walls = [float(row["wall_seconds"]) for row in starts]
        results[f"{prefix}.time_search"] = [
            walls,
            params,
            version,
            when_ms,
            sum(walls),
            [float("-inf")],
            [float("inf")],
            walls,
            walls,
            [1] * len(walls),
            [1] * len(walls),
        ]
    return {
        "commit_hash": commit,
        "env_name": "uv-pip-py3.12",
        "date": when_ms,
        "params": {
            "arch": machine.get("arch", ""),
            "cpu": machine.get("cpu", ""),
            "machine": machine.get("machine", ""),
            "num_cpu": str(machine.get("num_cpu", "")),
            "os": machine.get("os", ""),
            "ram": str(machine.get("ram", "")),
            "python": "3.12",
        },
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }
