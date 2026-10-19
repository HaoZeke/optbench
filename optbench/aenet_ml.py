"""Behler-Parrinello loss and the aenet 2.0.5 trainers.

The dataset, fingerprints, architecture, and loss are the mlopt problem.
aenet supplies L-BFGS-B (``setulb`` from the bundled Lbfgsb.3.0) and the
other trainers in ``src/optimize.f90``. One harness call is one evaluation
of the full-dataset loss and its gradient. That is the epoch the problem
counts. A repeated weight vector is not counted again.

The stop is the Frobenius norm of each parameter-tensor gradient. The
public problem states that L2 test. pyamff 8037eb98 stopped on the max
absolute entry of each tensor instead. A start is a success only when
every tensor is at or below the tolerance and the call count is under
the cap. There is no weight-space Hessian and no ``wrong_stationary`` test.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .counter import ForceCallCap, HarnessCounter

HIDDEN = (10, 7)
NEIGHBOR_CUTOFF = 6.0
INIT_SEED = 20261015
FINGERPRINT_REVISION = 1
LBFGS_M = 20
LBFGS_FACTR = 0.0
LBFGS_PGTOL = 1.0e-6

# Task codes returned by the setulb shim. NEW_X is consumed inside the shim.
TASK_START = 0
TASK_FG = 1
TASK_NEW_X = 2
TASK_CONV = 3
TASK_ERROR = 4
TASK_WARN = 5


class OptimizerCrash(RuntimeError):
    """The trainer stopped the way aenet stops: Cholesky or setulb failed."""


@dataclass(frozen=True)
class Spec:
    kind: str
    center: str
    neigh1: str
    neigh2: str
    eta: float
    rs: float
    rc: float
    zeta: float
    lam: float


@dataclass(frozen=True)
class Group:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


class Layout:
    """Flat weight vector in PyTorch ``parameters()`` order.

    Each element, in fingerprint-file order, stores ``W1, b1, W2, b2, W3, b3``.
    ``Linear`` weights are ``(out, in)`` and are C-contiguous. ``y = x @ W.T + b``.
    """

    def __init__(self, elements: list[str], n_fp: dict[str, int], hidden: tuple[int, int] = HIDDEN):
        self.elements = list(elements)
        self.n_fp = dict(n_fp)
        self.h1, self.h2 = hidden
        groups: list[Group] = []
        cursor = 0
        for element in self.elements:
            width = int(n_fp[element])
            specs = (
                (f"{element}.W1", (self.h1, width)),
                (f"{element}.b1", (self.h1,)),
                (f"{element}.W2", (self.h2, self.h1)),
                (f"{element}.b2", (self.h2,)),
                (f"{element}.W3", (1, self.h2)),
                (f"{element}.b3", (1,)),
            )
            for name, shape in specs:
                size = int(math.prod(shape))
                groups.append(Group(name, shape, cursor, cursor + size))
                cursor += size
        self.groups = tuple(groups)
        self.n = cursor
        self._by_element: dict[str, tuple[Group, ...]] = {}
        for element in self.elements:
            self._by_element[element] = tuple(group for group in groups if group.name.startswith(element + "."))

    def split(self, weights: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {element: {} for element in self.elements}
        flat = np.asarray(weights, dtype=np.float64).reshape(-1)
        for group in self.groups:
            element, key = group.name.split(".")
            out[element][key] = flat[group.start : group.stop].reshape(group.shape)
        return out

    def element_span(self, element: str) -> tuple[int, int]:
        groups = self._by_element[element]
        return groups[0].start, groups[-1].stop


def parse_fpparas(text: str) -> tuple[list[str], dict[str, list[Spec]]]:
    """Read one archived ``fpParas.dat``.

    The angle column on a G2 line is stored by pyamff and is not used.
    ``calculate_G2`` takes the angle from the neighbor geometry.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    if not lines or lines[0] != "BP":
        raise ValueError("fpParas is not a Behler-Parrinello file")
    elements = lines[1].split()
    specs: dict[str, list[Spec]] = {element: [] for element in elements}
    for line in lines[4:]:
        parts = line.split()
        kind = parts[0]
        if kind == "G1":
            center, neigh = parts[1], parts[2]
            eta, rs, rc = (float(parts[3]), float(parts[4]), float(parts[5]))
            specs[center].append(Spec("G1", center, neigh, "", eta, rs, rc, 0.0, 0.0))
        elif kind == "G2":
            center, neigh1, neigh2 = parts[1], parts[2], parts[3]
            eta, zeta, lam, _theta, rc = (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
                float(parts[8]),
            )
            specs[center].append(Spec("G2", center, neigh1, neigh2, eta, 0.0, rc, zeta, lam))
        else:
            raise ValueError(f"unknown fingerprint line {line!r}")
    return elements, specs


def initial_weights(layout: Layout, start_index: int, seed: int = INIT_SEED) -> np.ndarray:
    """Uniform ``[-1, 1)`` from ``PCG64(seed + start_index)``.

    Each parameter tensor is drawn in layout order, C-order. The problem
    states the distribution and not the seed.
    """
    rng = np.random.Generator(np.random.PCG64(int(seed) + int(start_index)))
    weights = np.empty(layout.n, dtype=np.float64)
    for group in layout.groups:
        weights[group.start : group.stop] = rng.random(group.stop - group.start) * 2.0 - 1.0
    return weights


def _mic(pos: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(cell)
    frac = pos @ inv
    # vec[i, j] is the minimum-image vector from i to j.
    delta = frac[None, :, :] - frac[:, None, :]
    delta -= np.round(delta)
    vec = delta @ cell
    dist = np.linalg.norm(vec, axis=-1)
    return vec, dist


def neighbor_indices(pos: np.ndarray, cell: np.ndarray, cutoff: float = NEIGHBOR_CUTOFF) -> list[np.ndarray]:
    """Neighbors with minimum-image distance strictly below ``cutoff``.

    Indices are sorted. ASE ``NeighborList`` with both-way cutoffs of
    ``cutoff / 2`` and ``np.sort`` produces this list when no second
    periodic image lies inside the cutoff. These cells are large enough
    that the second image does not.
    """
    _vec, dist = _mic(pos, cell)
    found = []
    count = pos.shape[0]
    eye = np.arange(count)
    for center in range(count):
        found.append(np.flatnonzero((dist[center] < cutoff) & (eye != center)))
    return found


def _accumulate_image(
    pos: np.ndarray,
    symbols: np.ndarray,
    cell: np.ndarray,
    specs: dict[str, list[Spec]],
    atoms_of: dict[str, np.ndarray],
    local_of: np.ndarray,
    fp: dict[str, np.ndarray],
    dG: dict[str, np.ndarray] | None,
    cutoff: float,
) -> None:
    vec, dist = _mic(pos, cell)
    count = pos.shape[0]
    eye = np.arange(count)
    for center in range(count):
        js = np.flatnonzero((dist[center] < cutoff) & (eye != center))
        element = str(symbols[center])
        local = int(local_of[center])
        rows = specs[element]
        if js.size:
            rij = dist[center, js]
            vij = vec[center, js]
            unit = vij / rij[:, None]
            sym = symbols[js]
        else:
            rij = unit = sym = None
        for g_index, spec in enumerate(rows):
            if spec.kind == "G1":
                if js.size == 0:
                    continue
                mask = sym == spec.neigh1
                if not np.any(mask):
                    continue
                radius = rij[mask]
                uv = unit[mask]
                targets = js[mask]
                rc = spec.rc
                arg = (radius - spec.rs) / rc
                gauss = np.exp(-spec.eta * arg * arg)
                fc = 0.5 * (np.cos(np.pi * radius / rc) + 1.0)
                fp[element][local, g_index] += float(np.sum(gauss * fc))
                if dG is not None:
                    dgauss = gauss * (-2.0 * spec.eta * (radius - spec.rs) / rc**2)
                    dfc = 0.5 * (-np.pi / rc) * np.sin(np.pi * radius / rc)
                    dval = dgauss * fc + gauss * dfc
                    slope = (dval[:, None] * uv).astype(np.float64, copy=False)
                    dG[element][local, g_index, center, :] += -slope.sum(axis=0)
                    np.add.at(dG[element][local, g_index], targets, slope)
            else:
                if js.size < 2:
                    continue
                y, z = np.triu_indices(js.size, k=1)
                j = js[y]
                k = js[z]
                sym_j = sym[y]
                sym_k = sym[z]
                mask = (sym_j == spec.neigh1) & (sym_k == spec.neigh2)
                if not np.any(mask):
                    continue
                j = j[mask]
                k = k[mask]
                radius_ij = dist[center, j]
                radius_ik = dist[center, k]
                vij = vec[center, j]
                vik = vec[center, k]
                cos = np.sum(vij * vik, axis=1) / (radius_ij * radius_ik)
                cos = np.clip(cos, -1.0, 1.0)
                rjk_vec = pos[k] - pos[j]
                rjk = np.linalg.norm(rjk_vec, axis=1)
                alive = (rjk > 1e-12) & (radius_ij > 1e-12) & (radius_ik > 1e-12)
                if not np.any(alive):
                    continue
                j = j[alive]
                k = k[alive]
                radius_ij = radius_ij[alive]
                radius_ik = radius_ik[alive]
                vij = vij[alive]
                vik = vik[alive]
                cos = cos[alive]
                rjk_vec = rjk_vec[alive]
                rjk = rjk[alive]
                rc = spec.rc
                eta = spec.eta
                zeta = spec.zeta
                lam = spec.lam
                pref = 2.0 ** (1.0 - zeta)
                base = 1.0 + lam * cos
                ang = np.power(base, zeta)
                gauss = np.exp(-eta * (radius_ij**2 + radius_ik**2 + rjk**2) / rc**2)
                fcij = 0.5 * (np.cos(np.pi * radius_ij / rc) + 1.0)
                fcik = 0.5 * (np.cos(np.pi * radius_ik / rc) + 1.0)
                fcjk = 0.5 * (np.cos(np.pi * rjk / rc) + 1.0)
                fc = fcij * fcik * fcjk
                fp[element][local, g_index] += float(np.sum(pref * ang * gauss * fc))
                if dG is None:
                    continue
                dang = zeta * np.power(base, zeta - 1.0) * lam
                inv_rc2 = 1.0 / rc**2
                dg_ij = gauss * (-2.0 * eta * radius_ij * inv_rc2)
                dg_ik = gauss * (-2.0 * eta * radius_ik * inv_rc2)
                dg_jk = gauss * (-2.0 * eta * rjk * inv_rc2)
                dfc_ij = 0.5 * (-np.pi / rc) * np.sin(np.pi * radius_ij / rc)
                dfc_ik = 0.5 * (-np.pi / rc) * np.sin(np.pi * radius_ik / rc)
                dfc_jk = 0.5 * (-np.pi / rc) * np.sin(np.pi * rjk / rc)
                dval_ij = pref * ang * (dg_ij * fc + gauss * dfc_ij * fcik * fcjk)
                dval_ik = pref * ang * (dg_ik * fc + gauss * fcij * dfc_ik * fcjk)
                dval_jk = pref * ang * (dg_jk * fc + gauss * fcij * fcik * dfc_jk)
                dval_cos = pref * dang * gauss * fc
                unit_ij = vij / radius_ij[:, None]
                unit_ik = vik / radius_ik[:, None]
                unit_jk = rjk_vec / rjk[:, None]
                denom = radius_ij * radius_ik
                dcos_i = -(vij + vik) / denom[:, None] + cos[:, None] * unit_ij / radius_ij[:, None]
                dcos_i = dcos_i + cos[:, None] * unit_ik / radius_ik[:, None]
                dcos_j = vik / denom[:, None] - cos[:, None] * unit_ij / radius_ij[:, None]
                dcos_k = vij / denom[:, None] - cos[:, None] * unit_ik / radius_ik[:, None]
                di = dval_ij[:, None] * (-unit_ij) + dval_ik[:, None] * (-unit_ik) + dval_cos[:, None] * dcos_i
                dj = dval_ij[:, None] * unit_ij + dval_jk[:, None] * (-unit_jk) + dval_cos[:, None] * dcos_j
                dk = dval_ik[:, None] * unit_ik + dval_jk[:, None] * unit_jk + dval_cos[:, None] * dcos_k
                slot = dG[element][local, g_index]
                slot[center, :] += di.sum(axis=0)
                np.add.at(slot, j, dj)
                np.add.at(slot, k, dk)
        # g_index for G2 counted G1s from the spec list. The increment above
        # uses that. Nothing else to close.


def build_descriptors(
    positions: np.ndarray,
    cell: np.ndarray,
    symbols: list[str] | np.ndarray,
    elements: list[str],
    specs: dict[str, list[Spec]],
    *,
    derivatives: bool,
    cutoff: float = NEIGHBOR_CUTOFF,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fingerprints for every image.

    ``Rij`` and ``Rik`` use the minimum image. ``Rjk`` is the raw
    Cartesian difference, which is what pyamff's Python G2 uses. Pairs
    with ``R`` above the fingerprint ``Rc`` are kept: the neighbor list
    is the 6 Å list, and the cutoff function is still applied.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 3:
        raise ValueError("positions must be (images, atoms, 3)")
    n_images, n_atoms, _ = pos.shape
    symbol_array = np.asarray(list(symbols))
    if symbol_array.shape != (n_atoms,):
        raise ValueError("symbols must name each atom once; every image shares the order")
    atoms_of = {element: np.flatnonzero(symbol_array == element) for element in elements}
    local_of = np.full(n_atoms, -1, dtype=np.int64)
    for element, indexes in atoms_of.items():
        local_of[indexes] = np.arange(indexes.size)
    if np.any(local_of < 0):
        raise ValueError("an atom is missing from the fingerprint elements")
    fp = {
        element: np.zeros((n_images, atoms_of[element].size, len(specs[element])), dtype=np.float64)
        for element in elements
    }
    dG = {}
    if derivatives:
        dG = {
            element: np.zeros(
                (n_images, atoms_of[element].size, len(specs[element]), n_atoms, 3),
                dtype=np.float64,
            )
            for element in elements
        }
    cell_m = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    for image in range(n_images):
        fp_image = {element: fp[element][image] for element in elements}
        dG_image = {element: dG[element][image] for element in elements} if derivatives else None
        _accumulate_image(
            pos[image],
            symbol_array,
            cell_m,
            specs,
            atoms_of,
            local_of,
            fp_image,
            dG_image,
            cutoff,
        )
    return fp, dG, atoms_of


def _forward(G: np.ndarray, packed: dict[str, np.ndarray]):
    W1, b1, W2, b2, W3, b3 = packed["W1"], packed["b1"], packed["W2"], packed["b2"], packed["W3"], packed["b3"]
    h1 = np.tanh(G @ W1.T + b1)
    h2 = np.tanh(h1 @ W2.T + b2)
    w3 = W3.reshape(-1)
    energy = h2 @ w3 + float(b3.reshape(-1)[0])
    sech1 = 1.0 - h1 * h1
    sech2 = 1.0 - h2 * h2
    s2 = sech2 * w3
    v = s2 @ W2
    s1 = sech1 * v
    u = s1 @ W1
    return h1, h2, energy, u, s1, s2, v, sech1, sech2, w3


def _with_batch(tensor: np.ndarray, batch: tuple[int, ...]) -> np.ndarray:
    if not batch:
        return tensor
    return np.broadcast_to(tensor, batch + tensor.shape)


def _backward(G, packed, up_e, up_u, cache) -> list[np.ndarray]:
    """Vector-Jacobian product.

    ``up_e`` ends in the atom axis. Activations either already carry that
    leading batch (one axis per image) or are a single image, in which
    case an extra residual axis on ``up_e`` is broadcast across them.
    """
    h1, h2, _energy, _u, s1, s2, v, sech1, sech2, w3 = cache
    W1, W2 = packed["W1"], packed["W2"]
    extra = up_e.ndim - (h1.ndim - 1)
    if extra > 0:
        batch = up_e.shape[:extra]
        G = _with_batch(G, batch)
        h1 = _with_batch(h1, batch)
        h2 = _with_batch(h2, batch)
        s1 = _with_batch(s1, batch)
        s2 = _with_batch(s2, batch)
        v = _with_batch(v, batch)
        sech1 = _with_batch(sech1, batch)
        sech2 = _with_batch(sech2, batch)
    elif extra < 0:
        raise ValueError("upstream batch does not match the activations")
    batch = up_e.shape[:-1]
    # Energy path. Sum over atoms, keep the batch.
    dW3 = np.einsum("...a,...aq->...q", up_e, h2, optimize=True)
    db3 = up_e.sum(axis=-1)
    dz2 = up_e[..., None] * s2
    dW2 = np.einsum("...aq,...ap->...qp", dz2, h1, optimize=True)
    db2 = dz2.sum(axis=-2)
    dh1 = dz2 @ W2
    dz1 = dh1 * sech1
    dW1 = np.einsum("...ap,...af->...pf", dz1, G, optimize=True)
    db1 = dz1.sum(axis=-2)
    if up_u is not None:
        ds1 = np.einsum("...af,pf->...ap", up_u, W1, optimize=True)
        dW1 = dW1 + np.einsum("...ap,...af->...pf", s1, up_u, optimize=True)
        dv = ds1 * sech1
        dh1_u = ds1 * (-2.0 * h1) * v
        ds2 = dv @ W2.T
        dW2 = dW2 + np.einsum("...aq,...ap->...qp", s2, dv, optimize=True)
        dw3 = np.einsum("...aq,...aq->...q", ds2, sech2, optimize=True)
        dh2 = ds2 * (-2.0 * h2) * w3
        dW3 = dW3 + dw3
        dz2 = dh2 * sech2
        dW2 = dW2 + np.einsum("...aq,...ap->...qp", dz2, h1, optimize=True)
        db2 = db2 + dz2.sum(axis=-2)
        dh1 = dz2 @ W2 + dh1_u
        dz1 = dh1 * sech1
        dW1 = dW1 + np.einsum("...ap,...af->...pf", dz1, G, optimize=True)
        db1 = db1 + dz1.sum(axis=-2)
    return [dW1, db1, dW2, db2, dW3.reshape(dW3.shape[:-1] + (1, dW3.shape[-1])), db3[..., None]]


_RANKS = (2, 1, 2, 1, 2, 1)


def _reduce_sum(parts: list[np.ndarray]) -> list[np.ndarray]:
    reduced = []
    for part, rank in zip(parts, _RANKS, strict=True):
        while part.ndim > rank:
            part = part.sum(axis=0)
        reduced.append(part)
    return reduced


class LossModel:
    """Problem loss on a fixed fingerprint dataset.

    ``energyloss`` is ``sum_images ((E_pred - E_ref) / N)^2``.
    ``forceloss`` is ``sum ||F_pred - F_ref||^2 / (3 N)``.
    ``loss = cE * energyloss + cF * forceloss``. RMSE divides by the
    image count outside those sums and does not include the coefficients.
    Force RMSE is ``-1`` when ``cF`` is zero.
    """

    def __init__(
        self,
        *,
        fp: dict[str, np.ndarray],
        atoms: dict[str, np.ndarray],
        energy: np.ndarray,
        forces: np.ndarray | None,
        layout: Layout,
        c_energy: float,
        c_force: float,
        dG: dict[str, np.ndarray] | None = None,
    ):
        self.fp = fp
        self.atoms = atoms
        self.energy = np.asarray(energy, dtype=np.float64)
        self.forces = None if forces is None else np.asarray(forces, dtype=np.float64)
        self.layout = layout
        self.c_energy = float(c_energy)
        self.c_force = float(c_force)
        self.dG = dG
        self.n_images = int(self.energy.shape[0])
        self.n_atoms = int(sum(indexes.size for indexes in atoms.values()))
        if self.c_force and dG is None:
            raise ValueError("force training needs fingerprint derivatives")

    def loss_grad(self, weights: np.ndarray, *, jacobian: bool = False) -> tuple[float, np.ndarray, dict]:
        packed = self.layout.split(weights)
        caches = {}
        predicted = np.zeros(self.n_images, dtype=np.float64)
        for element in self.layout.elements:
            caches[element] = _forward(self.fp[element], packed[element])
            predicted += caches[element][2].sum(axis=1)
        residual = (predicted - self.energy) / self.n_atoms
        energy_loss = float(np.dot(residual, residual))
        loss = self.c_energy * energy_loss
        d_energy = self.c_energy * 2.0 * residual / self.n_atoms
        force_loss = 0.0
        delta_f = None
        if self.c_force:
            forces = np.zeros((self.n_images, self.n_atoms, 3), dtype=np.float64)
            for element in self.layout.elements:
                forces -= np.einsum("ieg,iegax->iax", caches[element][3], self.dG[element], optimize=True)
            delta = forces - self.forces
            force_loss = float(np.sum(delta * delta) / (3.0 * self.n_atoms))
            loss += self.c_force * force_loss
            delta_f = self.c_force * 2.0 * delta / (3.0 * self.n_atoms)
        gradient = np.zeros(self.layout.n, dtype=np.float64)
        for element in self.layout.elements:
            _h1, _h2, atom_energy, _u, *_rest = caches[element]
            up_e = np.broadcast_to(d_energy[:, None], atom_energy.shape)
            up_u = None
            if self.c_force:
                up_u = -np.einsum("iax,iegax->ieg", delta_f, self.dG[element], optimize=True)
            parts = _backward(self.fp[element], packed[element], up_e, up_u, caches[element])
            flat = _concat(_reduce_sum(parts))
            start, stop = self.layout.element_span(element)
            gradient[start:stop] = flat
        energy_rmse = math.sqrt(energy_loss / self.n_images)
        if self.c_force:
            force_rmse = math.sqrt(force_loss / self.n_images)
        else:
            force_rmse = -1.0
        aux = {
            "energy_rmse": energy_rmse,
            "force_rmse": force_rmse,
            "predicted": predicted,
            "finite": bool(np.isfinite(loss) and np.isfinite(gradient).all()),
        }
        if jacobian:
            residuals, jac = self._jacobian(packed, caches, predicted)
            aux["residuals"] = residuals
            aux["jacobian"] = jac
            aux["sse"] = 0.5 * float(np.dot(residuals, residuals))
        return float(loss), gradient, aux

    def _jacobian(self, packed, caches, predicted) -> tuple[np.ndarray, np.ndarray]:
        scale_e = -math.sqrt(2.0 * self.c_energy) / self.n_atoms
        energy_residual = math.sqrt(2.0 * self.c_energy) * (self.energy - predicted) / self.n_atoms
        image_jac = np.zeros((self.n_images, self.layout.n), dtype=np.float64)
        for element in self.layout.elements:
            atom_energy = caches[element][2]
            up_e = np.full(atom_energy.shape, scale_e, dtype=np.float64)
            parts = _backward(self.fp[element], packed[element], up_e, None, caches[element])
            per_image = _concat_batch(parts)
            start, stop = self.layout.element_span(element)
            image_jac[:, start:stop] = per_image
        blocks = [image_jac]
        residual_parts = [energy_residual]
        if self.c_force:
            scale_f = math.sqrt(2.0 * self.c_force / (3.0 * self.n_atoms))
            forces = np.zeros((self.n_images, self.n_atoms, 3), dtype=np.float64)
            for element in self.layout.elements:
                forces -= np.einsum("ieg,iegax->iax", caches[element][3], self.dG[element], optimize=True)
            delta = forces - self.forces
            residual_parts.append((scale_f * delta).reshape(-1))
            blocks.append(self._force_jacobian(packed, scale_f))
        residual = np.concatenate(residual_parts)
        jacobian = np.concatenate(blocks, axis=0)
        return residual, jacobian.T

    def _force_jacobian(self, packed, scale_f: float) -> np.ndarray:
        """Rows are ``d e_{i,a,x} / dW`` with atom-major ``x, y, z`` order."""
        n_rows = self.n_images * self.n_atoms * 3
        jac = np.zeros((n_rows, self.layout.n), dtype=np.float64)
        for element in self.layout.elements:
            block = _element_force_jacobian(self.fp[element], self.dG[element], packed[element], scale_f)
            start, stop = self.layout.element_span(element)
            jac[:, start:stop] = block
        return jac

    def image_energy_jacobian(self, weights: np.ndarray, image: int) -> tuple[float, np.ndarray]:
        """One structure, at the current weights. Used by the serial EKF."""
        packed = self.layout.split(weights)
        predicted = 0.0
        caches = {}
        for element in self.layout.elements:
            cache = _forward(self.fp[element][image], packed[element])
            caches[element] = cache
            predicted += float(cache[2].sum())
        scale = -math.sqrt(2.0 * self.c_energy) / self.n_atoms
        value = math.sqrt(2.0 * self.c_energy) * (float(self.energy[image]) - predicted) / self.n_atoms
        jac = np.zeros(self.layout.n, dtype=np.float64)
        for element in self.layout.elements:
            up_e = np.full(caches[element][2].shape, scale, dtype=np.float64)
            parts = _backward(self.fp[element][image], packed[element], up_e, None, caches[element])
            start, stop = self.layout.element_span(element)
            jac[start:stop] = _concat(parts)
        return value, jac

    def image_force_jacobian(self, weights: np.ndarray, image: int, atom: int, coord: int) -> tuple[float, np.ndarray]:
        packed = self.layout.split(weights)
        caches = {}
        for element in self.layout.elements:
            caches[element] = _forward(self.fp[element][image], packed[element])
        forces = np.zeros((self.n_atoms, 3), dtype=np.float64)
        for element in self.layout.elements:
            forces -= np.einsum("eg,egax->ax", caches[element][3], self.dG[element][image], optimize=True)
        scale = math.sqrt(2.0 * self.c_force / (3.0 * self.n_atoms))
        value = scale * float(forces[atom, coord] - self.forces[image, atom, coord])
        jac = np.zeros(self.layout.n, dtype=np.float64)
        for element in self.layout.elements:
            up_u = -scale * self.dG[element][image][:, :, atom, coord]
            up_e = np.zeros(caches[element][2].shape, dtype=np.float64)
            parts = _backward(self.fp[element][image], packed[element], up_e, up_u, caches[element])
            start, stop = self.layout.element_span(element)
            jac[start:stop] = _concat(parts)
        return value, jac


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([part.reshape(-1) for part in parts])


def _concat_batch(parts: list[np.ndarray]) -> np.ndarray:
    """Batch is every axis except the parameter axes of each tensor."""
    columns = []
    for part in parts:
        if part.ndim == 1:
            columns.append(part.reshape(-1, 1))
        else:
            columns.append(part.reshape(part.shape[0], -1))
    return np.concatenate(columns, axis=1)


def _element_force_jacobian(G, dG, packed, scale_f: float, chunk: int = 16) -> np.ndarray:
    """``(images * atoms * 3, n_parameters)`` for one element.

    Same vector-Jacobian product as ``_backward`` with a zero energy
    upstream and ``up_u = -scale_f * dG``. Images are chunked so the
    force-row batch stays in memory when several workers run at once.
    """
    n_images = int(G.shape[0])
    n_atoms = int(dG.shape[3])
    n_force = n_atoms * 3
    n_param = sum(int(np.size(packed[key])) for key in ("W1", "b1", "W2", "b2", "W3", "b3"))
    out = np.zeros((n_images, n_force, n_param), dtype=np.float64)
    for begin in range(0, n_images, chunk):
        end = min(begin + chunk, n_images)
        out[begin:end] = _element_force_chunk(G[begin:end], dG[begin:end], packed, scale_f)
    return out.reshape(n_images * n_force, n_param)


def _element_force_chunk(G, dG, packed, scale_f: float) -> np.ndarray:
    """One chunk. ``dG`` is ``(images, atoms_of_element, n_fp, atoms, 3)``."""
    W1, W2 = packed["W1"], packed["W2"]
    h1, h2, _energy, _u, s1, s2, v, sech1, sech2, w3 = _forward(G, packed)
    n_images, _n_e, n_fp, n_atoms, _ = dG.shape
    up_u = -scale_f * np.transpose(dG, (0, 3, 4, 1, 2)).reshape(n_images, n_atoms * 3, _n_e, n_fp)
    # Force path of ``_backward``. The energy upstream is zero.
    ds1 = np.einsum("iraf,pf->irap", up_u, W1, optimize=True)
    dW1 = np.einsum("iap,iraf->irpf", s1, up_u, optimize=True)
    dv = ds1 * sech1[:, None, :, :]
    dh1_u = ds1 * (-2.0 * h1[:, None, :, :]) * v[:, None, :, :]
    ds2 = dv @ W2.T
    dW2 = np.einsum("iaq,irap->irqp", s2, dv, optimize=True)
    dw3 = np.einsum("iraq,iaq->irq", ds2, sech2, optimize=True)
    dh2 = ds2 * (-2.0 * h2[:, None, :, :]) * w3
    dz2 = dh2 * sech2[:, None, :, :]
    dW2 = dW2 + np.einsum("iraq,iap->irqp", dz2, h1, optimize=True)
    db2 = dz2.sum(axis=-2)
    dh1 = dz2 @ W2 + dh1_u
    dz1 = dh1 * sech1[:, None, :, :]
    dW1 = dW1 + np.einsum("irap,iaf->irpf", dz1, G, optimize=True)
    db1 = dz1.sum(axis=-2)
    dW3 = dw3.reshape(n_images, n_atoms * 3, 1, w3.shape[0])
    db3 = np.zeros((n_images, n_atoms * 3, 1), dtype=np.float64)
    parts = (dW1, db1, dW2, db2, dW3, db3)
    columns = [part.reshape(n_images, n_atoms * 3, -1) for part in parts]
    return np.concatenate(columns, axis=2)


def group_norms(gradient: np.ndarray, layout: Layout) -> list[float]:
    norms = []
    flat = np.asarray(gradient, dtype=np.float64).reshape(-1)
    for group in layout.groups:
        block = flat[group.start : group.stop]
        norms.append(float(np.linalg.norm(block)))
    return norms


def gradients_ok(gradient: np.ndarray, layout: Layout, tolerance: float) -> bool:
    return all(norm <= tolerance for norm in group_norms(gradient, layout))


class _Adam:
    def __init__(self, n: int, lr: float, batch: int):
        self.lr = float(lr)
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1.0e-8
        self.batch = int(batch)
        self.t = 1
        self.m = np.zeros(n, dtype=np.float64)
        self.v = np.zeros(n, dtype=np.float64)

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        g = gradient / self.batch
        self.m = self.b1 * self.m + (1.0 - self.b1) * g
        self.v = self.b2 * self.v + (1.0 - self.b2) * (g * g)
        mhat = self.m / (1.0 - self.b1**self.t)
        vhat = self.v / (1.0 - self.b2**self.t)
        self.t += 1
        return weights - self.lr / (np.sqrt(vhat) + self.eps) * mhat


class _OnlineSD:
    def __init__(self, n: int, gamma: float):
        self.alpha = 0.01
        self.gamma = float(gamma)
        self.v = np.zeros(n, dtype=np.float64)

    def step(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.v = self.alpha * self.v - self.gamma * gradient
        return weights + self.v


class _LM:
    def __init__(self, lam: float):
        self.lam = float(lam)
        self.adjust = 10.0
        self.cycle = 0
        self.mode = "jacob"
        self.sse_prev = 0.0
        self.H: np.ndarray | None = None
        self.g: np.ndarray | None = None
        self.saved: np.ndarray | None = None

    def step(self, weights: np.ndarray, loss: float, aux: dict) -> np.ndarray:
        if self.mode == "jacob":
            jac = aux["jacobian"]
            residual = aux["residuals"]
            hessian = jac @ jac.T
            grad = jac @ residual
            delta = _cholesky_solve(hessian, grad, self.lam)
            self.H = hessian
            self.g = grad
            self.saved = np.array(weights, copy=True)
            self.sse_prev = float(loss)
            self.cycle += 1
            self.mode = "error"
            return weights - delta
        if loss > self.sse_prev and self.cycle <= 5:
            self.lam *= self.adjust
            delta = _cholesky_solve(self.H, self.g, self.lam)
            self.cycle += 1
            return self.saved - delta
        if loss <= self.sse_prev:
            self.lam /= self.adjust
        self.mode = "jacob"
        self.cycle = 1
        return weights


class _EKF:
    def __init__(self, n: int):
        self.lam = 0.999
        self.lambda0 = 0.97
        self.pnoise = 0.0
        self.adaptive = 0.6
        self.group = 100
        self.sizes = []
        left = n
        while left:
            size = min(self.group, left)
            self.sizes.append(size)
            left -= size
        self.P = [(1.0 / 10.0) * np.eye(size) for size in self.sizes]

    def update(self, weights: np.ndarray, residual: float, jac: np.ndarray) -> np.ndarray:
        """One serial residual. ``jac`` is ``de/dW`` and ``residual`` is ``e``."""
        updated = np.array(weights, copy=True)
        cursor = 0
        for index, size in enumerate(self.sizes):
            P = self.P[index]
            column = jac[cursor : cursor + size]
            # A = J^T P J + lambda, with J a column. B = P J.
            B = P @ column
            gram = float(column @ B)
            scale = gram + self.lam
            if not np.isfinite(scale) or scale <= 0.0:
                raise OptimizerCrash("Cholesky decomposition failed")
            gain = B / scale
            updated[cursor : cursor + size] += gain * residual
            # C = K J^T, P <- (P - C P) / lambda
            C = np.outer(gain, column)
            P[:, :] = (P - C @ P) / self.lam
            if self.pnoise > 0.0:
                np.fill_diagonal(P, np.diag(P) + self.pnoise)
            cursor += size
        self.lam = self.lambda0 * (self.lam - 1.0) + 1.0
        return updated


def _cholesky_solve(hessian: np.ndarray, grad: np.ndarray, lam: float) -> np.ndarray:
    from scipy.linalg import LinAlgError, cho_factor, cho_solve

    matrix = np.array(hessian, dtype=np.float64, copy=True)
    diag = np.arange(matrix.shape[0])
    matrix[diag, diag] += lam
    try:
        factor = cho_factor(matrix, lower=False, check_finite=False, overwrite_a=True)
        return cho_solve(factor, grad, check_finite=False)
    except LinAlgError as exc:
        raise OptimizerCrash("Cholesky decomposition failed") from exc


class _LBFGS:
    def __init__(self, weights: np.ndarray):
        self.lib = load_lbfgsb()
        self.x = np.array(weights, dtype=np.float64, copy=True)
        code = int(self.lib.lbfgsb_start(self.x.size, LBFGS_FACTR, LBFGS_PGTOL, self.x))
        if code == TASK_ERROR:
            raise OptimizerCrash(f"setulb start failed: {lbfgsb_task(self.lib)}")
        self.stopped = code in (TASK_CONV, TASK_WARN, TASK_ERROR)

    def step(self, loss: float, gradient: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.x)
        code = int(self.lib.lbfgsb_step(float(loss), np.ascontiguousarray(gradient, dtype=np.float64), out))
        self.x = out
        if code == TASK_ERROR:
            raise OptimizerCrash(f"setulb {lbfgsb_task(self.lib)}")
        self.stopped = code in (TASK_CONV, TASK_WARN)
        return self.x


def _key(weights: np.ndarray) -> tuple[float, ...]:
    return tuple(np.asarray(weights, dtype=np.float64).reshape(-1).tolist())


def train_start(
    model: LossModel,
    weights: np.ndarray,
    method: str,
    *,
    cap: int,
    tolerance: float,
) -> dict:
    """One initialization. ``force_calls`` is the harness count."""
    counter = HarnessCounter(cache=True, cap=int(cap))
    w = np.array(weights, dtype=np.float64, copy=True)
    name = method
    state = None
    try:
        if name == "bfgs":
            state = _LBFGS(w)
            w = state.x
        elif name == "adam":
            state = _Adam(w.size, 0.01, model.n_images)
        elif name == "online_sd":
            state = _OnlineSD(w.size, 0.01)
        elif name == "lm":
            state = _LM(0.01)
        elif name == "ekf":
            state = _EKF(w.size)
        else:
            raise ValueError(f"unknown aenet trainer {method}")
    except OptimizerCrash as exc:
        return _record(0, False, "crash", error=str(exc))
    last = None
    try:
        while True:
            # An accepted LM step leaves the weights in place and then asks
            # for the Jacobian. That request is a new evaluation. The cached
            # error-only pass did not store one. A repeat that already has
            # the Jacobian is a stall.
            need_j = name == "lm" and state.mode == "jacob"
            fresh, cached = counter.charge(_key(w))
            cache_ok = cached is not None and (not need_j or "jacobian" in cached[2])
            if not fresh and cache_ok:
                loss, gradient, aux = cached
                stalled = True
            else:
                if not fresh:
                    if counter.cap is not None and counter.calls >= counter.cap:
                        raise ForceCallCap(f"force-call cap {counter.cap}")
                    counter.calls += 1
                loss, gradient, aux = model.loss_grad(w, jacobian=need_j)
                counter.remember(_key(w), (float(loss), gradient, aux))
                stalled = False
            last = (loss, gradient, aux, counter.calls)
            lm_retry = name == "lm" and state.mode == "error" and counter.calls < cap
            if not aux["finite"]:
                if not lm_retry:
                    return _record(counter.calls, False, "divergence", aux=aux, gradient=gradient, layout=model.layout)
                loss = math.inf
            else:
                ok = gradients_ok(gradient, model.layout, tolerance)
                if ok and counter.calls < cap:
                    return _record(counter.calls, True, None, aux=aux, gradient=gradient, layout=model.layout)
                if counter.calls >= cap:
                    return _record(counter.calls, False, "cap", aux=aux, gradient=gradient, layout=model.layout)
            if stalled:
                return _record(counter.calls, False, "divergence", aux=aux, gradient=gradient, layout=model.layout)
            if name == "bfgs" and state.stopped:
                return _record(counter.calls, False, "divergence", aux=aux, gradient=gradient, layout=model.layout)
            if name == "bfgs":
                w = state.step(loss, gradient)
            elif name == "adam":
                w = state.step(w, gradient)
            elif name == "online_sd":
                w = state.step(w, gradient)
            elif name == "lm":
                w = state.step(w, loss, aux)
            else:
                w = _ekf_epoch(model, state, w)
    except ForceCallCap:
        aux = None if last is None else last[2]
        calls = cap if last is None else int(last[3])
        return _record(calls, False, "cap", aux=aux)
    except OptimizerCrash as exc:
        aux = None if last is None else last[2]
        calls = 0 if last is None else int(last[3])
        return _record(calls, False, "crash", aux=aux, error=str(exc))
    except Exception as exc:
        aux = None if last is None else last[2]
        calls = 0 if last is None else int(last[3])
        return _record(calls, False, "crash", aux=aux, error=str(exc))


def _ekf_epoch(model: LossModel, state: _EKF, weights: np.ndarray) -> np.ndarray:
    """One pass over residuals.

    aenet stores ``J = -dE/dW`` and ``E = e``, then adds ``K E``. The
    minus makes that step a descent step on ``0.5 e^2``.
    """
    w = weights
    for image in range(model.n_images):
        residual, jac = model.image_energy_jacobian(w, image)
        w = state.update(w, residual, -jac)
        if model.c_force:
            for atom in range(model.n_atoms):
                for coord in range(3):
                    residual, jac = model.image_force_jacobian(w, image, atom, coord)
                    w = state.update(w, residual, -jac)
    return w


def _record(
    calls: int,
    success: bool,
    failure: str | None,
    aux: dict | None = None,
    error: str | None = None,
    gradient: np.ndarray | None = None,
    layout: Layout | None = None,
) -> dict:
    record = {
        "force_calls": int(calls),
        "success": bool(success),
        "failure": failure,
        "energy_rmse": None if aux is None else aux.get("energy_rmse"),
        "force_rmse": None if aux is None else aux.get("force_rmse"),
    }
    if gradient is not None and layout is not None and np.isfinite(gradient).all():
        record["max_group_norm"] = max(group_norms(gradient, layout))
    if error:
        record["error"] = error
    return record


_LIB = None


def lbfgsb_library_path() -> Path | None:
    root = Path(os.environ.get("AENET_ROOT", "/tmp/aenet"))
    archive = root / "build" / "lib" / "liblbfgsb.a"
    if archive.is_file():
        return archive
    return None


def ensure_lbfgsb_built() -> Path:
    """Compile the shim. This does not load the shared library."""
    archive = lbfgsb_library_path()
    if archive is None:
        raise FileNotFoundError("liblbfgsb.a not found; set AENET_ROOT to an aenet build")
    source = Path(__file__).resolve().parent / "aenet" / "lbfgsb_api.f90"
    out_dir = Path(os.environ.get("AENET_LBFGSB_DIR", "/tmp/optbench-aenet"))
    out_dir.mkdir(parents=True, exist_ok=True)
    library = out_dir / "libaenet_lbfgsb.so"
    if not library.is_file() or library.stat().st_mtime < source.stat().st_mtime:
        subprocess.check_call(
            [
                "gfortran",
                "-shared",
                "-fPIC",
                "-O2",
                "-o",
                str(library),
                str(source),
                str(archive),
            ]
        )
    return library


def load_lbfgsb():
    """Compile the shim against aenet's ``liblbfgsb.a`` and load it."""
    global _LIB
    if _LIB is not None:
        return _LIB
    library = ensure_lbfgsb_built()
    lib = ctypes.CDLL(str(library))
    lib.lbfgsb_start.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    lib.lbfgsb_start.restype = ctypes.c_int
    lib.lbfgsb_step.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    lib.lbfgsb_step.restype = ctypes.c_int
    lib.lbfgsb_task.argtypes = [ctypes.c_char_p]
    lib.lbfgsb_task.restype = None
    _LIB = lib
    return lib


def lbfgsb_task(lib) -> str:
    buf = ctypes.create_string_buffer(61)
    lib.lbfgsb_task(buf)
    return buf.value.decode()


def lbfgsb_available() -> bool:
    if lbfgsb_library_path() is None:
        return False
    try:
        load_lbfgsb()
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True
