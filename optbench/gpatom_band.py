"""AIDNEB campaigns through ase-gpatom on the spec potential.

ase-gpatom 1.0.1 (572bd64) is the active-learning nudged elastic band.
A force call is one ``HarnessCounter`` entry on the spec potential.
Gaussian-process predictions, the IDPP interpolation, and Hessian
finite differences are not search cost. The band total is the raw
count across images.

The published loop supplies the search. Four gaps in this revision are
filled here so that loop can run on ASE 3.29: ``ase.neb`` is ``ase.mep.NEB``;
``Database.add`` stores forces and ``get_all_energyforces`` returns
them; ``ConstantPrior.let_update`` is a no-op because the ``fit``
prior branch calls it and this revision does not define it; the
surrogate calculator returns an energy, forces, and an energy
standard deviation. ``GPCalculator`` in this revision stores the
prediction vector as an energy and never sets ``uncertainty``.

The Gaussian process sees the free atoms. Fixed atoms stay at the
reactant. The potential is the full spec potential, minimum-imaged
with the archived box. The product is unwrapped onto the reactant
before interpolation. AIDNEB's own stop is the maximum per-atom force
and the path uncertainty. The score is the problem rule: the free
gradient L2 at the highest true-energy interior image, then a Hessian
index of 1, under the problem cap.

MDMin reads the surrogate force. The variance is computed when AIDNEB
reads the path uncertainty, which is the same number the library
returns. For the Cartesian fingerprint the kernel block is the closed
form of ``FPKernel``: same energy, gradients, and Hessian. The mean
prediction is that block contracted with the solved weights. The energy
variance is the same corner of ``k K^{-1} k``, solved once for the
whole band. After the prefactor update the dense kernel matrix is
dropped and its pages are returned. Predictions keep the scaled
Cholesky factor. A factor that would exceed 12 GiB raises before
the allocation, and that start is scored from the path so far.

Every fiftieth step the loop returns to the initial interpolation.
Inside one of those blocks the path and the acquired geometry can
repeat. The potential is not evaluated again. The remaining
iterations of that block still append the same observation, once per
step, and the next reset reads that training set. Those copies are
written together and the step moves to the reset.
"""

from __future__ import annotations

import contextvars
import copy
import ctypes
import gc
import json
import os
import statistics
import sys
import tempfile
import tomllib
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .campaign import geometry_archive, load_template, resolve_problem
from .counter import ForceCallCap, HarnessCounter
from .failures import classify_failure
from .pes import _mic, free_gradient, load_pairs
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance, write_starts

_COMMIT = "572bd64ef61ccf498e061162609bb5f16e39d8a4"
_FD_STEP = 1e-5
_STALL_HITS = 3
_KERNEL_BYTES = 12 * 1024**3
_READY = False
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("gpatom_active", default=None)

_SLUGS = {
    "neb/Pt-heptamer": "neb-pt-heptamer",
    "double-ended-saddle-search/lj38": "de-lj38",
    "double-ended-saddle-search/lj38_2ts_separation": "de-lj38-2ts",
    "double-ended-saddle-search/pt-island": "de-pt-island",
}


class RepeatedGeometry(RuntimeError):
    """The search asked for a geometry it had already evaluated."""


class _MemoryBudget(RuntimeError):
    """The next Cartesian factor would not fit in the kernel budget."""


def _kernel_pair_bytes(n_train: int, n_atoms: int) -> int:
    """Bytes for the kernel matrix and its Cholesky factor."""
    width = 1 + 3 * int(n_atoms)
    length = int(n_train) * width
    return length * length * 16


def _threads_one() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _cartesian_positions(fingerprint):
    if fingerprint.__class__.__name__ != "CartesianCoordFP":
        return None
    return np.asarray(fingerprint.vector, dtype=np.float64).reshape((-1, 3))


def _cartesian_block(left: np.ndarray, right: np.ndarray, weight: float, scale: float) -> np.ndarray:
    """Squared-exponential Cartesian block.

    Matches ``FPKernel.kernel`` for ``CartesianCoordFP``: energy, both
    gradients, and the coordinate Hessian in atom-major order.
    """
    delta = left - right
    atoms = int(left.shape[0])
    scale_sq = scale * scale
    inverse = 1.0 / scale_sq
    distance_sq = float(np.dot(delta.ravel(), delta.ravel()))
    value = weight * weight * np.exp(-distance_sq / (2.0 * scale_sq))
    gradient = (-value * inverse) * delta
    hessian = (value * inverse) * np.eye(atoms)[:, :, None, None] * np.eye(3)[None, None, :, :]
    hessian = hessian - (value * inverse * inverse) * (
        delta[:, None, :, None] * delta[None, :, None, :]
    )
    flat = np.swapaxes(hessian, 1, 2).reshape((3 * atoms, 3 * atoms))
    size = 3 * atoms + 1
    block = np.empty((size, size), dtype=np.float64)
    block[0, 0] = value
    block[1:, 0] = gradient.reshape(-1)
    block[0, 1:] = (-gradient).reshape(-1)
    block[1:, 1:] = flat
    return block


def _install_cartesian_kernel(kernel_cls) -> None:
    if getattr(kernel_cls, "_optbench_cartesian", False):
        return
    original_vector = kernel_cls.kernel_vector
    original_matrix = kernel_cls.kernel_matrix

    def kernel_vector(self, fingerprint, training):
        rows = [_cartesian_positions(fingerprint)]
        rows.extend(_cartesian_positions(item) for item in training)
        if any(row is None for row in rows):
            return original_vector(self, fingerprint, training)
        weight = float(self.kerneltype.weight)
        scale = float(self.kerneltype.scale)
        blocks = [_cartesian_block(rows[0], row, weight, scale) for row in rows[1:]]
        return np.hstack(blocks)

    def kernel_matrix(self, training):
        rows = [_cartesian_positions(item) for item in training]
        if any(row is None for row in rows):
            return original_matrix(self, training)
        weight = float(self.kerneltype.weight)
        scale = float(self.kerneltype.scale)
        count = len(rows)
        size = rows[0].size + 1
        matrix = np.empty((count * size, count * size), dtype=np.float64)
        for i in range(count):
            for j in range(i, count):
                block = _cartesian_block(rows[i], rows[j], weight, scale)
                start_i = i * size
                start_j = j * size
                matrix[start_i : start_i + size, start_j : start_j + size] = block
                if i != j:
                    matrix[start_j : start_j + size, start_i : start_i + size] = block.T
        return matrix

    kernel_cls.kernel_vector = kernel_vector
    kernel_cls.kernel_matrix = kernel_matrix
    kernel_cls._optbench_cartesian = True


def _cartesian_mean(
    query: np.ndarray,
    training: np.ndarray,
    alpha: np.ndarray,
    weight: float,
    scale: float,
) -> tuple[float, np.ndarray]:
    """Contract the Cartesian blocks with one weight vector per structure.

    ``training`` is ``(structures, atoms, 3)`` and ``alpha`` is
    ``(structures, 1 + 3 * atoms)``, in the same order as
    ``kernel_vector``.
    """
    delta = query[None, :, :] - training
    scale_sq = scale * scale
    inverse = 1.0 / scale_sq
    distance_sq = np.einsum("ijk,ijk->i", delta, delta)
    value = (weight * weight) * np.exp(-distance_sq / (2.0 * scale_sq))
    coefficient = value * inverse
    energy_weight = alpha[:, 0]
    force_weight = alpha[:, 1:].reshape((alpha.shape[0], query.shape[0], 3))
    overlap = np.einsum("ijk,ijk->i", delta, force_weight)
    energy = float(np.sum(value * energy_weight + coefficient * overlap))
    forces = np.sum(
        coefficient[:, None, None] * force_weight
        - (coefficient * inverse * overlap)[:, None, None] * delta
        - (coefficient * energy_weight)[:, None, None] * delta,
        axis=0,
    )
    return energy, forces


def _cartesian_energy_row(query: np.ndarray, training: np.ndarray, weight: float, scale: float) -> np.ndarray:
    """Energy row of ``kernel_vector`` for Cartesian squared-exponential blocks."""
    delta = query[None, :, :] - training
    scale_sq = scale * scale
    distance_sq = np.einsum("ijk,ijk->i", delta, delta)
    value = (weight * weight) * np.exp(-distance_sq / (2.0 * scale_sq))
    width = query.size + 1
    blocks = np.empty((training.shape[0], width), dtype=np.float64)
    blocks[:, 0] = value
    blocks[:, 1:] = ((value / scale_sq)[:, None, None] * delta).reshape((training.shape[0], query.size))
    return blocks.reshape(-1)


def _cartesian_energy_variances(gp, queries) -> np.ndarray:
    """Energy variances for several Cartesian fingerprints, one triangular solve."""
    from scipy.linalg import cho_solve

    query_pos = [_cartesian_positions(fingerprint) for fingerprint in queries]
    train_pos = [_cartesian_positions(item) for item in gp.X]
    if any(row is None for row in query_pos) or any(row is None for row in train_pos):
        variances = []
        for fingerprint in queries:
            vector = gp.kernel.kernel_vector(fingerprint, gp.X)
            variances.append(float(gp.calculate_variance(True, vector, fingerprint)[0, 0]))
        return np.asarray(variances, dtype=np.float64)
    training = np.stack(train_pos)
    weight = float(gp.kernel.kerneltype.weight)
    scale = float(gp.kernel.kerneltype.scale)
    rows = np.stack([_cartesian_energy_row(pos, training, weight, scale) for pos in query_pos])
    solved = cho_solve((gp.L, gp.lower), rows.T, overwrite_b=False, check_finite=False)
    return weight * weight - np.einsum("ij,ji->i", rows, solved)


def _install_cartesian_predict(gp_cls) -> None:
    if getattr(gp_cls, "_optbench_cartesian_predict", False):
        return
    original = gp_cls.predict

    def predict(self, fingerprint, get_variance=False, return_dkdx=False):
        if return_dkdx or not self.use_forces or self.use_stress:
            return original(self, fingerprint, get_variance=get_variance, return_dkdx=return_dkdx)
        query = _cartesian_positions(fingerprint)
        rows = [] if query is None else [_cartesian_positions(item) for item in self.X]
        if query is None or any(row is None for row in rows):
            return original(self, fingerprint, get_variance=get_variance, return_dkdx=return_dkdx)
        training = np.stack(rows)
        width = query.size + 1
        alpha = np.asarray(self.model_vector, dtype=np.float64).reshape((len(rows), width))
        energy, forces = _cartesian_mean(
            query,
            training,
            alpha,
            float(self.kernel.kerneltype.weight),
            float(self.kernel.kerneltype.scale),
        )
        prediction = np.empty(width, dtype=np.float64)
        prediction[0] = energy
        prediction[1:] = forces.reshape(-1)
        prediction += np.asarray(
            self.calculate_prior_array([fingerprint], get_forces=True, get_stress=False),
            dtype=np.float64,
        )
        if not get_variance:
            return prediction, None
        variance = float(_cartesian_energy_variances(self, [fingerprint])[0])
        return prediction, np.array([[variance]], dtype=np.float64)

    gp_cls.predict = predict
    gp_cls._optbench_cartesian_predict = True


def _install_memory_guard(gp_cls) -> None:
    if getattr(gp_cls, "_optbench_memory_guard", False):
        return
    original = gp_cls.train

    def train(self, features, targets):
        if len(features):
            atoms = len(features[0].atoms)
            if _kernel_pair_bytes(len(features), atoms) > _KERNEL_BYTES:
                raise _MemoryBudget(
                    f"Cartesian kernel for {len(features)} structures of {atoms} atoms "
                    f"exceeds {_KERNEL_BYTES} bytes"
                )
        return original(self, features, targets)

    gp_cls.train = train
    gp_cls._optbench_memory_guard = True


def _release_kernel_matrix(model) -> None:
    """Drop ``K_reg`` after the prefactor update has scaled ``L``.

    ``predict`` reads the scaled factor and the scaled weights. The next
    AIDNEB iteration trains a new model.
    """
    gp = getattr(model, "gp", None)
    if gp is not None and getattr(gp, "K_reg", None) is not None:
        gp.K_reg = None


def _return_freed_pages() -> None:
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)


def _install_path_reset_copy() -> None:
    """Reset the band without copying the surrogate.

    Every fiftieth step AIDNEB deep-copies the initial images. Those
    images carry the current surrogate, so the copy would duplicate the
    Cholesky factor. The reset only needs the positions. The next
    iteration trains a new model.
    """
    if getattr(copy, "_optbench_image_copy", False):
        return
    original = copy.deepcopy

    def deepcopy(obj, memo=None):
        calculators = []
        if isinstance(obj, list):
            for image in obj:
                calc = getattr(image, "calc", None)
                if isinstance(calc, _Surrogate):
                    calculators.append((image, calc))
        if not calculators:
            if memo is None:
                return original(obj)
            return original(obj, memo)
        for image, _calc in calculators:
            image.calc = None
        try:
            if memo is None:
                return original(obj)
            return original(obj, memo)
        finally:
            for image, calc in calculators:
                image.calc = calc

    copy.deepcopy = deepcopy
    copy._optbench_image_copy = True


def prepare() -> None:
    """Import ase-gpatom and install the ASE 3.29 compatibility shims."""
    global _READY
    if _READY:
        return
    import types

    import ase.mep.neb as mep

    neb = types.ModuleType("ase.neb")
    neb.NEB = mep.NEB
    sys.modules["ase.neb"] = neb

    from ase.optimize import MDMin
    from gpatom.gpfp.database import Database
    from gpatom.gpfp.fingerprint import CartesianCoordFP
    from gpatom.gpfp.prior import ConstantPrior
    from gpatom.io import TrainingSet

    if not hasattr(Database, "get_all_energyforces"):

        def add(self, fingerprint, energy, forces=None):
            self.fingerprintlist.append(fingerprint)
            self.energylist.append(energy)
            if forces is not None:
                self.forceslist.append(np.asarray(forces, dtype=float))

        def get_all_energyforces(self, negative_forces=False):
            rows = []
            for energy, forces in zip(self.energylist, self.forceslist, strict=True):
                signed = np.asarray(forces, dtype=float).reshape(-1)
                if negative_forces:
                    signed = -signed
                rows.append(np.concatenate([[float(energy)], signed]))
            return rows

        Database.add = add
        Database.get_all_energyforces = get_all_energyforces

    if not hasattr(ConstantPrior, "let_update"):
        ConstantPrior.let_update = lambda self: None

    original_fp = CartesianCoordFP.__init__

    def quiet_fp(self, atoms, **kwargs):
        saved = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            original_fp(self, atoms, **kwargs)
        finally:
            sys.stdout.close()
            sys.stdout = saved

    CartesianCoordFP.__init__ = quiet_fp

    class QuietMDMin(MDMin):
        def __init__(self, atoms, restart=None, logfile="-", trajectory=None, dt=None, maxstep=None, **kwargs):
            super().__init__(
                atoms,
                restart=restart,
                logfile=None,
                trajectory=None,
                dt=dt,
                maxstep=maxstep,
                **kwargs,
            )

    import gpatom.aidneb as aidneb
    from gpatom.gpfp.kernel import FPKernel

    aidneb.MDMin = QuietMDMin
    _install_cartesian_kernel(FPKernel)
    from gpatom.gpfp.gp import GaussianProcess

    _install_cartesian_predict(GaussianProcess)
    _install_memory_guard(GaussianProcess)
    _install_path_reset_copy()

    def predictions(images):
        _return_freed_pages()
        energy = []
        for image in images:
            image.calc.calculate(
                image,
                properties=["energy", "forces"],
                system_changes=["positions"],
            )
            energy.append(float(image.calc.results["energy"]))
        model = images[0].calc.model
        _release_kernel_matrix(model)
        queries = [model.new_fingerprint(image) for image in images]
        raw = _cartesian_energy_variances(model.gp, queries)
        uncertainty = []
        for image, value in zip(images, raw, strict=True):
            standard = float(np.sqrt(max(float(value), 0.0)))
            image.calc.results["uncertainty"] = standard
            uncertainty.append(2.0 * standard)
        uncertainty[0] = 0.0
        uncertainty[-1] = 0.0
        _remember_path(images)
        _return_freed_pages()
        return {"energy": energy, "uncertainty": uncertainty}

    aidneb.get_neb_predictions = predictions

    original_dump = TrainingSet.dump

    def dump(self, atoms, method):
        from ase.calculators.singlepoint import SinglePointCalculator

        results = getattr(getattr(atoms, "calc", None), "results", None) or {}
        if "energy" in results and "forces" in results:
            energy = float(results["energy"])
            forces = np.array(results["forces"], dtype=float, copy=True)
        else:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
        snap = atoms.copy()
        snap.calc = SinglePointCalculator(snap, energy=energy, forces=forces)
        return original_dump(self, snap, method)

    TrainingSet.dump = dump
    _READY = True


def _remember_path(images) -> None:
    active = _ACTIVE.get()
    if active is None:
        return
    signature = np.concatenate(
        [np.asarray(image.positions, dtype=np.float64).reshape(-1) for image in images]
    ).tobytes()
    state = active["state"]
    state["path_same"] = state.get("path_sig") == signature
    state["path_sig"] = signature


def _advance_reset_block(band, atoms, label: str) -> None:
    """Append the repeated observation through the next reset.

    The published loop stores one copy per step and resets when the
    step is a multiple of 50. Writing those copies here leaves the
    training set at that reset unchanged.
    """
    step = int(band.step)
    reset = (step // 50 + 1) * 50
    copies = reset - step - 1
    if copies <= 0:
        return
    for _ in range(copies):
        band.train.dump(atoms, "neb")
    band.step = reset - 1
    print(
        f"{label} spin step={step} reset={reset} copies={copies}",
        file=sys.stderr,
        flush=True,
    )


class _Spec(Calculator):
    """ASE calculator for free atoms on the spec potential."""

    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(self, counter, kind, template, mask, box, state, label):
        Calculator.__init__(self)
        self.counter = counter
        self.kind = kind
        self.template = np.array(template, dtype=np.float64, copy=True)
        self.mask = np.asarray(mask, dtype=bool)
        self.box = None if box is None else np.asarray(box, dtype=np.float64)
        self.state = state
        self.label = label

    def get_potential_energy(self, atoms=None, force_consistent=False):
        active = _ACTIVE.get()
        band = None if active is None else active.get("band")
        target = self.atoms if atoms is None else atoms
        if band is not None and target is band.atoms:
            self.results = {}
        return Calculator.get_potential_energy(self, atoms, force_consistent=force_consistent)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        free = np.asarray(atoms.get_positions(), dtype=np.float64)
        full = np.array(self.template, copy=True)
        full[self.mask] = free
        key = tuple(float(value) for value in full.reshape(-1))
        fresh, cached = self.counter.charge(key)
        advance = None
        if not fresh:
            if not self.counter._verifying:
                active = _ACTIVE.get()
                band = None if active is None else active.get("band")
                if band is not None and atoms is band.atoms and self.state.get("path_same"):
                    advance = band
                elif band is None or atoms is band.atoms:
                    self.state["repeats"] = int(self.state["repeats"]) + 1
                    if self.state["repeats"] >= _STALL_HITS:
                        raise RepeatedGeometry(self.label)
            energy, grad = cached
        else:
            self.state["repeats"] = 0
            energy, grad = free_gradient(self.kind, free.reshape(-1), self.template, self.mask, self.box)
            grad = tuple(float(value) for value in grad)
            self.counter.remember(key, (float(energy), grad))
            calls = int(self.counter.calls)
            if calls == 1 or calls % 25 == 0:
                print(f"{self.label} calls={calls}", file=sys.stderr, flush=True)
        gradient = np.asarray(grad, dtype=np.float64).reshape((-1, 3))
        self.results["energy"] = float(energy)
        self.results["free_energy"] = float(energy)
        self.results["forces"] = -gradient
        if advance is not None:
            _advance_reset_block(advance, atoms, self.label)

    def __copy__(self):
        return _Spec(self.counter, self.kind, self.template, self.mask, self.box, self.state, self.label)

    def __deepcopy__(self, memo):
        cloned = self.__copy__()
        memo[id(self)] = cloned
        return cloned


class _Surrogate(Calculator):
    """Energy, forces, and energy standard deviation from one AIDModel."""

    implemented_properties = ["energy", "forces"]
    nolabel = True

    def __init__(self, model):
        Calculator.__init__(self)
        self.model = model

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        wants_uncertainty = "uncertainty" in properties
        prediction, variance = self.model.calculate(atoms, get_variance=wants_uncertainty)
        values = np.asarray(prediction, dtype=np.float64).reshape(-1)
        energy = float(values[0])
        grads = values[1:].reshape((-1, 3))
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = -grads
        if wants_uncertainty:
            raw = 0.0 if variance is None else float(np.asarray(variance, dtype=np.float64)[0, 0])
            self.results["uncertainty"] = float(np.sqrt(max(raw, 0.0)))

    def __copy__(self):
        return _Surrogate(self.model)

    def __deepcopy__(self, memo):
        cloned = _Surrogate(copy.deepcopy(self.model, memo))
        memo[id(self)] = cloned
        return cloned


def _model(aid: dict):
    from gpatom.aidmin import AIDModel

    return AIDModel(
        train_images=[],
        prior=None,
        fit_weight=aid["fit_weight"],
        update_prior_strategy=aid["update_prior_strategy"],
        params={"weight": aid["weight"], "scale": aid["scale"], "noise": aid["noise"]},
        params_to_update=None,
        batch_size=5,
        bounds=None,
        kerneltype=aid["kerneltype"],
        use_forces=bool(aid["use_forces"]),
        max_train_data_strategy=aid["max_train_data_strategy"],
        max_train_data=int(aid["max_train_data"]),
    )


def _atoms(positions: np.ndarray):
    from ase import Atoms

    count = int(positions.shape[0])
    return Atoms(numbers=[1] * count, positions=np.asarray(positions, dtype=np.float64))


def _unwrapped_product(reactant: np.ndarray, product: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    return np.asarray(reactant, dtype=np.float64) + _mic(
        np.asarray(product, dtype=np.float64) - np.asarray(reactant, dtype=np.float64),
        box,
    )


def _last_observation(aid) -> tuple[np.ndarray, float, np.ndarray] | None:
    try:
        atoms = aid.train.load_last()
    except Exception:
        return None
    if atoms is None or atoms.calc is None:
        return None
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    except Exception:
        return None
    return np.asarray(atoms.get_positions(), dtype=np.float64), energy, forces


def _score_candidate(
    spec: _Spec,
    positions: np.ndarray,
    energy: float,
    forces: np.ndarray,
    *,
    force_norm: float,
    cap: int,
    hit_cap: bool,
    project_rigid: bool,
) -> dict:
    force = float(np.linalg.norm(forces))
    reasons: list[str] = []
    if hit_cap or spec.counter.calls >= cap:
        reasons.append("cap")
    if not np.isfinite(force) or force > force_norm:
        reasons.append("force_norm")
    hessian_index = None
    if "cap" not in reasons and "force_norm" not in reasons:
        point = np.asarray(positions, dtype=np.float64).reshape(-1)

        def gradient(coords):
            probe = _atoms(np.asarray(coords, dtype=np.float64).reshape((-1, 3)))
            probe.calc = copy.copy(spec)
            observed = np.asarray(probe.get_forces(), dtype=np.float64)
            return (-observed).reshape(-1)

        with spec.counter.verification():
            hessian = numerical_hessian(gradient, point.tolist(), step=_FD_STEP)
        hessian_index = vibrational_index(
            np.asarray(hessian, dtype=np.float64),
            np.asarray(positions, dtype=np.float64).reshape((-1, 3)),
            project_rigid,
        )
        if hessian_index != 1:
            reasons.append("hessian_index")
    failure = classify_failure(reasons)
    return {
        "energy": float(energy),
        "force_norm": force,
        "hessian_index": hessian_index,
        "reasons": reasons,
        "failure": failure,
        "success": failure is None,
    }


def _select_candidate(aid, spec: _Spec, cap: int) -> tuple[np.ndarray, float, np.ndarray, str]:
    """Highest true-energy interior image, while the cap still allows it."""
    images = list(getattr(aid, "images", []) or [])
    interior = images[1:-1] if len(images) >= 3 else []
    best = None
    for image in interior:
        if spec.counter.calls >= cap:
            break
        try:
            probe = _atoms(image.get_positions())
            probe.calc = copy.copy(spec)
            energy = float(probe.get_potential_energy())
            forces = np.asarray(probe.get_forces(), dtype=np.float64)
        except (ForceCallCap, RepeatedGeometry):
            break
        if best is None or energy > best[1]:
            best = (np.asarray(probe.get_positions(), dtype=np.float64), energy, forces, "highest")
    if best is not None:
        return best
    last = _last_observation(aid)
    if last is None:
        raise RuntimeError("AIDNEB stored no true evaluation")
    positions, energy, forces = last
    return positions, energy, forces, "last"


def run_band_search(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    kind: str,
    n_interior: int,
    force_norm: float,
    cap: int,
    aid: dict,
) -> dict:
    """One reactant/product pair. The record uses the harness counter."""
    state: dict = {"counter": None}
    try:
        return _run_band_search(
            name=name,
            reactant=reactant,
            product=product,
            free=free,
            box=box,
            kind=kind,
            n_interior=n_interior,
            force_norm=force_norm,
            cap=cap,
            aid=aid,
            state=state,
        )
    except Exception as exc:
        counter = state["counter"]
        calls = 0 if counter is None else int(counter.calls)
        verify = 0 if counter is None else int(counter.verify_calls)
        return _crash_record(name, calls, verify, f"{type(exc).__name__}: {exc}")


def _run_band_search(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    kind: str,
    n_interior: int,
    force_norm: float,
    cap: int,
    aid: dict,
    state: dict,
) -> dict:
    _threads_one()
    prepare()
    from gpatom.aidneb import AIDNEB

    warnings.filterwarnings(
        "ignore",
        message="Please use NEB's interpolate",
        category=FutureWarning,
    )
    mask = np.asarray(free, dtype=bool)
    reactant = np.asarray(reactant, dtype=np.float64)
    product = _unwrapped_product(reactant, product, box)
    n_free = int(np.count_nonzero(mask))
    counter = HarnessCounter(cache=True, cap=int(cap))
    state["counter"] = counter
    tally = {"repeats": 0}
    spec = _Spec(counter, kind, reactant, mask, box, tally, name)
    previous = Path.cwd()
    hit_cap = False
    repeated = False
    memory_stop = False
    aid_converged = False
    library_calls = 0
    spring = None
    n_images = int(n_interior) + 2
    active = {"state": tally, "band": None}
    token = _ACTIVE.set(active)
    try:
        with tempfile.TemporaryDirectory(prefix="gpatom-") as work:
            os.chdir(work)
            try:
                band = AIDNEB(
                    _atoms(reactant[mask]),
                    _atoms(product[mask]),
                    calculator=spec,
                    model_calculator=_Surrogate(_model(aid)),
                    interpolation=str(aid["interpolation"]),
                    n_images=n_images,
                    k=None,
                    mic=False,
                    neb_method=str(aid["neb_method"]),
                    remove_rotation_and_translation=bool(aid["remove_rotation_and_translation"]),
                    max_train_data=int(aid["max_train_data"]),
                    max_train_data_strategy=str(aid["max_train_data_strategy"]),
                    trainingset="observations.traj",
                    trajectory="AIDNEB.traj",
                    trajectory_surrogate="surrogate.traj",
                    use_previous_observations=False,
                )
                spring = float(band.spring)
                n_images = int(band.n_images)
                active["band"] = band
                try:
                    band.run(
                        fmax=float(force_norm),
                        unc_convergence=float(aid["unc_convergence"]),
                        dt=float(aid["dt"]),
                        ml_steps=int(aid["ml_steps"]),
                        max_step=float(aid["max_step"]),
                    )
                    aid_converged = True
                except ForceCallCap:
                    hit_cap = True
                except RepeatedGeometry:
                    repeated = True
                except _MemoryBudget:
                    memory_stop = True
                library_calls = int(band.force_calls)
                positions, energy, forces, candidate = _select_candidate(band, spec, cap)
            finally:
                os.chdir(previous)
    finally:
        _ACTIVE.reset(token)
    if counter.calls >= cap:
        hit_cap = True
    scored = _score_candidate(
        spec,
        positions,
        energy,
        forces,
        force_norm=float(force_norm),
        cap=int(cap),
        hit_cap=hit_cap,
        project_rigid=bool(np.all(mask) and n_free >= 2 and kind == "lj"),
    )
    success = bool(scored["success"])
    failure = scored["failure"]
    reasons = list(scored["reasons"])
    if memory_stop and not success:
        if "crash" not in reasons:
            reasons.insert(0, "crash")
        failure = "crash"
    print(
        f"{name} done calls={counter.calls} success={success} "
        f"failure={failure} force={scored['force_norm']}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(counter.calls),
        "verify_calls": int(counter.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": scored["energy"],
        "force_norm": scored["force_norm"],
        "hessian_index": scored["hessian_index"],
        "converged": bool(aid_converged),
        "candidate": candidate,
        "repeated": bool(repeated),
        "memory_stop": bool(memory_stop),
        "aid_force_calls": library_calls,
        "n_images": n_images,
        "spring": spring,
        "n_free": n_free,
    }


def _crash_record(name: str, calls: int, verify: int, crash: str) -> dict:
    print(f"{name} crash {crash} calls={calls}", file=sys.stderr, flush=True)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify),
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "crash": crash,
        "energy": float("nan"),
        "force_norm": float("nan"),
        "hessian_index": None,
        "converged": False,
        "candidate": "crash",
        "repeated": False,
        "memory_stop": False,
        "aid_force_calls": 0,
        "n_images": None,
        "spring": None,
        "n_free": None,
    }


def _registry(root: Path) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == "gpatom":
            return row
    raise ValueError("codes/registry.toml has no gpatom entry")


def _aid(template: dict) -> dict:
    aid = dict(template["aid"])
    required = (
        "interpolation",
        "neb_method",
        "max_train_data",
        "max_train_data_strategy",
        "unc_convergence",
        "dt",
        "ml_steps",
        "max_step",
        "fit_weight",
        "update_prior_strategy",
        "kerneltype",
        "weight",
        "scale",
        "noise",
        "use_forces",
        "remove_rotation_and_translation",
    )
    missing = [name for name in required if name not in aid]
    if missing:
        raise ValueError(f"{template['id']} aid block lacks {', '.join(missing)}")
    return aid


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the gpatom registry")
    if not str(commit).startswith(_COMMIT) and not _COMMIT.startswith(str(commit)):
        raise ValueError(f"{template['id']} is not pinned to {_COMMIT}")
    band = problem["band"]
    convergence = problem["convergence"]
    return {
        "potential": band["potential"],
        "n_interior": int(band["interior_images"]),
        "max_move": float(band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": problem["geometry"]["kind"],
        "aid": _aid(template),
    }


def _job(pair, settings: dict) -> dict:
    box = None if pair.box is None else np.asarray(pair.box, dtype=np.float64)
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64),
        "product": np.asarray(pair.product, dtype=np.float64),
        "free": np.asarray(pair.free, dtype=bool),
        "box": box,
        "settings": settings,
    }


def _execute(job: dict) -> dict:
    _threads_one()
    settings = job["settings"]
    try:
        return run_band_search(
            name=job["name"],
            reactant=job["reactant"],
            product=job["product"],
            free=job["free"],
            box=job["box"],
            kind=settings["potential"],
            n_interior=settings["n_interior"],
            force_norm=settings["force_norm"],
            cap=settings["cap"],
            aid=settings["aid"],
        )
    except Exception:
        return _crash_record(job["name"], 0, 0, traceback.format_exc().splitlines()[-1])


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, problem: dict, *, partial: bool) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness HarnessCounter on the spec potential",
        "# force_calls count raw potential evaluations and are not divided by the image count",
        "# surrogate evaluations are not force calls",
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
    lines.append(f"problem {problem['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append(f"images {settings['n_interior']}")
    if partial:
        lines.append("partial true")
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


def _read_records(path: Path) -> list[dict]:
    starts = path.with_suffix(".starts.jsonl")
    if not starts.is_file():
        return []
    rows = []
    for line in starts.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_outputs(
    output: Path,
    records: list[dict],
    template: dict,
    settings: dict,
    problem: dict,
) -> dict:
    """Write ``benchmark.dat``, the per-start file, and the v2 score."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = len(records) < int(settings["pairs"])
    output.write_text(_dat(records, template, settings, problem, partial=partial), encoding="utf-8")
    stamped = []
    for row in records:
        item = dict(row)
        item["seed"] = "0"
        item["problem"] = problem["id"]
        item["template"] = template["id"]
        item["code"] = template["code"]
        item["code_version"] = template["code_version"]
        item["commit"] = template["commit"]
        item["algorithm"] = template["algorithm"]
        item["tuning"] = template["tuning"]
        stamped.append(item)
    write_starts(output.with_suffix(".starts.jsonl"), stamped)
    parameters = {
        "potential": settings["potential"],
        "n_interior": settings["n_interior"],
        "force_norm": settings["force_norm"],
        "cap": settings["cap"],
        "counter": "harness",
        "seed": "0",
        "aid": settings["aid"],
    }
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "partial": partial,
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
                    "reasons": row.get("reasons") or [],
                }
                for row in stamped
            ],
            settings["cap"],
        ),
        "legacy": _legacy(stamped),
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
        "partial": partial,
        "output": str(output),
    }


def default_output(root: Path, problem_id: str) -> Path:
    slug = _SLUGS.get(problem_id, problem_id.replace("/", "-"))
    return Path(root) / "results" / "campaign" / f"gpatom-1.0.1-aidneb-{slug}.dat"


def search_named(
    root: Path,
    problem_name: str,
    template_name: str,
    start: str,
    seed: str = "0",
    cap: int | None = None,
) -> dict:
    """One archived pair.

    ``seed`` is recorded and does not change the search. ``cap`` replaces
    the problem cap when a test needs a short run.
    """
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    if cap is not None:
        settings = dict(settings)
        settings["cap"] = int(cap)
    pairs = load_pairs(geometry_archive(root, problem), settings["geometry"], settings["pairs"])
    found = {pair.name: pair for pair in pairs}
    if start not in found:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(found[start], settings))
    record["seed"] = str(seed)
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
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
    """Run the pairs in one problem and write the dat, starts, and v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != "gpatom-band":
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = load_pairs(geometry_archive(root, problem), settings["geometry"], settings["pairs"])
    jobs = [_job(pair, settings) for pair in pairs]
    order = {job["name"]: index for index, job in enumerate(jobs)}
    kept = _read_records(output) if resume else []
    done = {row["start"] for row in kept}
    pending = [job for job in jobs if job["name"] not in done]
    if limit > 0:
        pending = pending[: int(limit)]
    records = [row for row in kept if row["start"] in order]
    _threads_one()

    def publish() -> dict:
        ranked = sorted(records, key=lambda row: order.get(row["start"], 10**9))
        return write_outputs(output, ranked, template, settings, problem)

    if pending:
        if workers <= 1:
            for job in pending:
                records.append(_execute(job))
                publish()
        else:
            import multiprocessing

            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [pool.submit(_execute, job) for job in pending]
                for future in as_completed(futures):
                    records.append(future.result())
                    publish()
    return publish()
