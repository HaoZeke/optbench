"""The mlopt squared-error loss on Behler-Parrinello fingerprints.

The network is the published one: hidden widths 10 and 7, tanh on the
hidden layers, a linear output, and one network per element. Energies are
the sum of the atomic networks. There is no feature scaler. The loss is
the archived sum of squares, with the energy term divided by the atom
count and the force term divided by three times the atom count.

Fingerprints follow the Fortran cutoff used for the published rows: a
pair contributes when its distance is inside the cutoff, and a triplet
contributes only when all three distances are inside the cutoff. G2 pairs
are ordered by increasing atom index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIDDEN = (10, 7)
INIT_SEED = 20261015
FP_SHA256 = {
    "emt-pt38": "48d7fd70360ab6ab547689b05ed1f9d71413a78f21c4cd2c0faf8dac14c4c8e2",
    "C2H4": "ec02cdb8bd5fab393ddefcb110c28328e20fe3483fc73bf9a21cd8e0f06e5753",
}


@dataclass(frozen=True)
class G1:
    center: str
    neighbor: str
    eta: float
    rs: float
    rc: float


@dataclass(frozen=True)
class G2:
    center: str
    neighbor1: str
    neighbor2: str
    eta: float
    zeta: float
    lam: float
    theta_s: float
    rc: float


@dataclass(frozen=True)
class Fingerprints:
    elements: tuple[str, ...]
    g1: tuple[G1, ...]
    g2: tuple[G2, ...]

    def counts(self) -> dict[str, int]:
        found = {element: 0 for element in self.elements}
        for term in (*self.g1, *self.g2):
            found[term.center] += 1
        return found


@dataclass(frozen=True)
class TensorSpec:
    species: str
    name: str
    shape: tuple[int, ...]
    offset: int

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=int))


@dataclass(frozen=True)
class Layout:
    specs: tuple[TensorSpec, ...]
    hidden: tuple[int, int]

    @property
    def size(self) -> int:
        if not self.specs:
            return 0
        last = self.specs[-1]
        return last.offset + last.size

    def species(self) -> tuple[str, ...]:
        found: list[str] = []
        for spec in self.specs:
            if spec.species not in found:
                found.append(spec.species)
        return tuple(found)


@dataclass(frozen=True)
class SpeciesBlock:
    symbol: str
    atom_index: np.ndarray
    fingerprints: np.ndarray
    d_fingerprints: np.ndarray | None


@dataclass(frozen=True)
class Landscape:
    """One problem after the fingerprints have been built."""

    species: tuple[SpeciesBlock, ...]
    energies: np.ndarray
    forces: np.ndarray
    n_atoms: np.ndarray
    layout: Layout
    symbols: tuple[str, ...]

    @property
    def n_images(self) -> int:
        return int(self.energies.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.forces.shape[1])


def parse_fingerprints(text: str) -> Fingerprints:
    """Read one archived ``fpParas.dat`` document."""
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    if len(lines) < 4 or lines[0] != "BP":
        raise ValueError("fingerprint file is not a BP parameter file")
    elements = tuple(lines[1].split())
    references = [float(value) for value in lines[2].split()]
    if len(references) != len(elements) or any(value != 0.0 for value in references):
        raise ValueError("fingerprint reference energies are not zero")
    n_g1, n_g2 = (int(value) for value in lines[3].split())
    g1: list[G1] = []
    g2: list[G2] = []
    for line in lines[4:]:
        fields = line.split()
        if fields[0] == "G1":
            if len(fields) != 6:
                raise ValueError(f"G1 line has {len(fields)} fields")
            g1.append(
                G1(
                    center=fields[1],
                    neighbor=fields[2],
                    eta=float(fields[3]),
                    rs=float(fields[4]),
                    rc=float(fields[5]),
                )
            )
        elif fields[0] == "G2":
            if len(fields) != 9:
                raise ValueError(f"G2 line has {len(fields)} fields")
            g2.append(
                G2(
                    center=fields[1],
                    neighbor1=fields[2],
                    neighbor2=fields[3],
                    eta=float(fields[4]),
                    zeta=float(fields[5]),
                    lam=float(fields[6]),
                    theta_s=float(fields[7]),
                    rc=float(fields[8]),
                )
            )
        else:
            raise ValueError(f"unknown fingerprint {fields[0]}")
    if len(g1) != n_g1 or len(g2) != n_g2:
        raise ValueError("fingerprint counts do not match the header")
    for term in (*g1, *g2):
        if term.center not in elements:
            raise ValueError(f"fingerprint center {term.center} is not an element")
    for term in g2:
        if term.theta_s != 0.0:
            raise ValueError("G2 theta shift is not zero")
    return Fingerprints(elements=elements, g1=tuple(g1), g2=tuple(g2))


def layout_for(counts: dict[str, int], elements: tuple[str, ...], hidden: tuple[int, int] = HIDDEN) -> Layout:
    """Parameter tensors in element order, weight then bias, PyTorch shapes."""
    if len(hidden) != 2 or hidden[0] < 1 or hidden[1] < 1:
        raise ValueError("the network has two hidden widths")
    specs: list[TensorSpec] = []
    offset = 0
    for element in elements:
        n_fp = int(counts[element])
        shapes = (
            ("W0", (hidden[0], n_fp)),
            ("b0", (hidden[0],)),
            ("W1", (hidden[1], hidden[0])),
            ("b1", (hidden[1],)),
            ("W2", (1, hidden[1])),
            ("b2", (1,)),
        )
        for name, shape in shapes:
            spec = TensorSpec(species=element, name=name, shape=shape, offset=offset)
            specs.append(spec)
            offset += spec.size
    return Layout(specs=tuple(specs), hidden=hidden)


def initial_parameters(layout: Layout, start: int, seed: int = INIT_SEED) -> np.ndarray:
    """One uniform[-1, 1) draw. ``start`` selects the stream."""
    rng = np.random.Generator(np.random.PCG64(int(seed) + int(start)))
    return rng.random(layout.size) * 2.0 - 1.0


def _pair_vectors(positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    disp = positions[:, None, :, :] - positions[:, :, None, :]
    if np.any(pbc):
        inv = np.linalg.inv(cell)
        frac = disp @ inv.T
        wrapped = np.round(frac) * np.asarray(pbc, dtype=np.float64)
        disp = disp - wrapped @ cell
    dist = np.linalg.norm(disp, axis=-1)
    return disp, dist


def _g1_value_derivative(dist: np.ndarray, eta: float, rs: float, rc: float) -> tuple[np.ndarray, np.ndarray]:
    fc = 0.5 * (np.cos(np.pi * dist / rc) + 1.0)
    dfc = -0.5 * (np.pi / rc) * np.sin(np.pi * dist / rc)
    arg = (dist - rs) / rc
    expo = np.exp(-eta * arg * arg)
    dexpo = expo * (-2.0 * eta * (dist - rs) / rc**2)
    return expo * fc, dexpo * fc + expo * dfc


def _accumulate_g1(
    fingerprints: np.ndarray,
    derivatives: np.ndarray | None,
    disp: np.ndarray,
    dist: np.ndarray,
    symbols: tuple[str, ...],
    term: G1,
    slot: int,
) -> None:
    symbols_arr = np.asarray(symbols)
    mask = (
        (symbols_arr[:, None] == term.center)
        & (symbols_arr[None, :] == term.neighbor)
        & (dist > 1e-12)
        & (dist < term.rc)
    )
    value, slope = _g1_value_derivative(dist, term.eta, term.rs, term.rc)
    value = np.where(mask, value, 0.0)
    slope = np.where(mask, slope, 0.0)
    fingerprints[:, :, slot] += value.sum(axis=2)
    if derivatives is None:
        return
    unit = np.zeros_like(disp)
    np.divide(disp, dist[..., None], out=unit, where=mask[..., None])
    # disp[:, i, j] is the minimum-image vector from i to j.
    grad = slope[..., None] * unit
    derivatives[:, :, slot, :, :] += grad
    summed = grad.sum(axis=2)
    for atom in range(dist.shape[1]):
        derivatives[:, atom, slot, atom, :] -= summed[:, atom, :]


def _g2_chunk(
    disp: np.ndarray,
    triplets: np.ndarray,
    symbols: tuple[str, ...],
    term: G2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = triplets[:, 0]
    left = triplets[:, 1]
    right = triplets[:, 2]
    images = np.arange(disp.shape[0])[:, None]
    rij_v = disp[images, centers, left, :]
    rik_v = disp[images, centers, right, :]
    rjk_v = disp[images, left, right, :]
    rij = np.linalg.norm(rij_v, axis=-1)
    rik = np.linalg.norm(rik_v, axis=-1)
    rjk = np.linalg.norm(rjk_v, axis=-1)
    species = np.asarray(symbols)
    mask = (
        (species[centers] == term.center)
        & (species[left] == term.neighbor1)
        & (species[right] == term.neighbor2)
        & (rij > 1e-12)
        & (rik > 1e-12)
        & (rjk > 1e-12)
        & (rij < term.rc)
        & (rik < term.rc)
        & (rjk < term.rc)
    )
    safe_ij = np.where(mask, rij, 1.0)
    safe_ik = np.where(mask, rik, 1.0)
    safe_jk = np.where(mask, rjk, 1.0)
    uij = rij_v / safe_ij[..., None]
    uik = rik_v / safe_ik[..., None]
    ujk = rjk_v / safe_jk[..., None]
    cos = np.clip(np.sum(uij * uik, axis=-1), -1.0, 1.0)
    base = 1.0 + term.lam * cos
    ang = base**term.zeta
    dang = np.where(base > 0.0, term.zeta * base ** (term.zeta - 1.0) * term.lam, 0.0)
    rc = term.rc
    fc_ij = 0.5 * (np.cos(np.pi * safe_ij / rc) + 1.0)
    fc_ik = 0.5 * (np.cos(np.pi * safe_ik / rc) + 1.0)
    fc_jk = 0.5 * (np.cos(np.pi * safe_jk / rc) + 1.0)
    dfc_ij = -0.5 * (np.pi / rc) * np.sin(np.pi * safe_ij / rc)
    dfc_ik = -0.5 * (np.pi / rc) * np.sin(np.pi * safe_ik / rc)
    dfc_jk = -0.5 * (np.pi / rc) * np.sin(np.pi * safe_jk / rc)
    expo = np.exp(-term.eta * (safe_ij**2 + safe_ik**2 + safe_jk**2) / rc**2)
    scale = -2.0 * term.eta / rc**2
    pref = 2.0 ** (1.0 - term.zeta)
    common = pref * ang * expo
    value = common * fc_ij * fc_ik * fc_jk
    d_rij = pref * ang * fc_ik * fc_jk * (expo * scale * safe_ij * fc_ij + expo * dfc_ij)
    d_rik = pref * ang * fc_ij * fc_jk * (expo * scale * safe_ik * fc_ik + expo * dfc_ik)
    d_rjk = pref * ang * fc_ij * fc_ik * (expo * scale * safe_jk * fc_jk + expo * dfc_jk)
    d_cos = pref * dang * expo * fc_ij * fc_ik * fc_jk
    value = np.where(mask, value, 0.0)
    d_rij = np.where(mask, d_rij, 0.0)
    d_rik = np.where(mask, d_rik, 0.0)
    d_rjk = np.where(mask, d_rjk, 0.0)
    d_cos = np.where(mask, d_cos, 0.0)
    dcos_vij = (uik - cos[..., None] * uij) / safe_ij[..., None]
    dcos_vik = (uij - cos[..., None] * uik) / safe_ik[..., None]
    d_i = d_rij[..., None] * (-uij) + d_rik[..., None] * (-uik) + d_cos[..., None] * (-dcos_vij - dcos_vik)
    d_j = d_rij[..., None] * uij + d_rjk[..., None] * (-ujk) + d_cos[..., None] * dcos_vij
    d_k = d_rik[..., None] * uik + d_rjk[..., None] * ujk + d_cos[..., None] * dcos_vik
    return value, d_i, d_j, d_k, triplets


def _triplets(n_atoms: int) -> np.ndarray:
    rows = []
    for center in range(n_atoms):
        others = [index for index in range(n_atoms) if index != center]
        for left in range(len(others)):
            for right in range(left + 1, len(others)):
                rows.append((center, others[left], others[right]))
    if not rows:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def _accumulate_g2(
    fingerprints: np.ndarray,
    derivatives: np.ndarray | None,
    disp: np.ndarray,
    symbols: tuple[str, ...],
    term: G2,
    slot: int,
) -> None:
    triplets = _triplets(disp.shape[1])
    if triplets.size == 0:
        return
    chunk = 256
    for start in range(0, len(triplets), chunk):
        value, d_i, d_j, d_k, used = _g2_chunk(disp, triplets[start : start + chunk], symbols, term)
        centers = used[:, 0]
        left = used[:, 1]
        right = used[:, 2]
        for local, center in enumerate(centers):
            fingerprints[:, center, slot] += value[:, local]
        if derivatives is None:
            continue
        images = np.arange(disp.shape[0])[:, None]
        center_b = np.broadcast_to(centers, value.shape)
        left_b = np.broadcast_to(left, value.shape)
        right_b = np.broadcast_to(right, value.shape)
        for axis in range(3):
            np.add.at(derivatives[:, :, slot, :, axis], (images, center_b, center_b), d_i[:, :, axis])
            np.add.at(derivatives[:, :, slot, :, axis], (images, center_b, left_b), d_j[:, :, axis])
            np.add.at(derivatives[:, :, slot, :, axis], (images, center_b, right_b), d_k[:, :, axis])


def compute_fingerprints(
    positions: np.ndarray,
    symbols: tuple[str, ...],
    cell: np.ndarray,
    pbc: np.ndarray,
    parameters: Fingerprints,
    *,
    with_derivatives: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fingerprints shaped ``(images, atoms, components)``."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3:
        raise ValueError("positions are (images, atoms, 3)")
    counts = parameters.counts()
    width = max(counts.values()) if counts else 0
    if width == 0:
        raise ValueError("fingerprint file has no components")
    n_images, n_atoms, _ = positions.shape
    fingerprints = np.zeros((n_images, n_atoms, width), dtype=np.float64)
    derivatives = np.zeros((n_images, n_atoms, width, n_atoms, 3), dtype=np.float64) if with_derivatives else None
    disp, dist = _pair_vectors(positions, np.asarray(cell, dtype=np.float64), np.asarray(pbc, dtype=bool))
    # Component order for one element is its own G1 lines followed by its own G2 lines.
    per_element_g1 = {element: [] for element in parameters.elements}
    per_element_g2 = {element: [] for element in parameters.elements}
    for term in parameters.g1:
        per_element_g1[term.center].append(term)
    for term in parameters.g2:
        per_element_g2[term.center].append(term)
    for element in parameters.elements:
        slot = 0
        for term in per_element_g1[element]:
            _accumulate_g1(fingerprints, derivatives, disp, dist, symbols, term, slot)
            slot += 1
        for term in per_element_g2[element]:
            _accumulate_g2(fingerprints, derivatives, disp, symbols, term, slot)
            slot += 1
        if slot != counts[element]:
            raise ValueError(f"{element} has {slot} fingerprints, expected {counts[element]}")
    return fingerprints, derivatives


def build_landscape(
    positions: np.ndarray,
    symbols: tuple[str, ...],
    cell: np.ndarray,
    pbc: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    parameters: Fingerprints,
    *,
    with_forces: bool,
    hidden: tuple[int, int] = HIDDEN,
) -> Landscape:
    fingerprints, derivatives = compute_fingerprints(
        positions, symbols, cell, pbc, parameters, with_derivatives=with_forces
    )
    counts = parameters.counts()
    blocks = []
    symbol_array = np.asarray(symbols)
    for element in parameters.elements:
        index = np.flatnonzero(symbol_array == element).astype(np.int32)
        width = counts[element]
        block_g = np.ascontiguousarray(fingerprints[:, index, :width])
        block_dg = None
        if derivatives is not None:
            block_dg = np.ascontiguousarray(derivatives[:, index, :width, :, :])
        blocks.append(SpeciesBlock(symbol=element, atom_index=index, fingerprints=block_g, d_fingerprints=block_dg))
    n_atoms = np.full(len(energies), positions.shape[1], dtype=np.float64)
    return Landscape(
        species=tuple(blocks),
        energies=np.asarray(energies, dtype=np.float64),
        forces=np.asarray(forces, dtype=np.float64),
        n_atoms=n_atoms,
        layout=layout_for(counts, parameters.elements, hidden),
        symbols=tuple(symbols),
    )


def squared_error(
    predicted_energy: np.ndarray,
    energy: np.ndarray,
    predicted_force: np.ndarray | None,
    force: np.ndarray | None,
    n_atoms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Archived energy and force sums of squares, one value per start."""
    residual = (predicted_energy - energy[:, None]) / n_atoms[:, None]
    energy_loss = np.sum(residual * residual, axis=0)
    if predicted_force is None:
        force_loss = np.zeros(predicted_energy.shape[1], dtype=np.float64)
        return energy_loss, force_loss
    diff = predicted_force - force[:, None, :, :]
    per_atom = np.sum(diff * diff, axis=-1)
    force_loss = np.sum(per_atom / (3.0 * n_atoms[:, None, None]), axis=(0, 2))
    return energy_loss, force_loss


def _view(flat: np.ndarray, spec: TensorSpec) -> np.ndarray:
    return flat[:, spec.offset : spec.offset + spec.size].reshape((flat.shape[0],) + spec.shape)


def _species_tensors(flat: np.ndarray, layout: Layout, species: str) -> dict[str, np.ndarray]:
    return {spec.name: _view(flat, spec) for spec in layout.specs if spec.species == species}


def _contract(dedg: np.ndarray, d_fingerprints: np.ndarray) -> np.ndarray:
    n_images, n_local, _batch, _width = dedg.shape
    n_atoms = d_fingerprints.shape[3]
    left = np.ascontiguousarray(dedg.transpose(0, 2, 1, 3)).reshape(n_images, dedg.shape[2], n_local * d_fingerprints.shape[2])
    right = d_fingerprints.reshape(n_images, n_local * d_fingerprints.shape[2], n_atoms * 3)
    contracted = np.einsum("ibp,ipq->ibq", left, right, optimize=True)
    return contracted.reshape(n_images, dedg.shape[2], n_atoms, 3)


def loss_and_grad(
    parameters: np.ndarray,
    landscape: Landscape,
    *,
    energy_coefficient: float = 1.0,
    force_coefficient: float = 0.0,
) -> dict[str, np.ndarray]:
    """Loss, parameter gradient, and the max parameter-set L2, per start."""
    flat = np.asarray(parameters, dtype=np.float64)
    if flat.ndim == 1:
        flat = flat[None, :]
    batch = flat.shape[0]
    predicted = np.zeros((landscape.n_images, batch), dtype=np.float64)
    activations = []
    dedgs = []
    force = np.zeros((landscape.n_images, batch, landscape.n_sites, 3), dtype=np.float64) if force_coefficient else None
    for block in landscape.species:
        weights = _species_tensors(flat, landscape.layout, block.symbol)
        hidden0, hidden1 = _forward(block.fingerprints, weights)
        predicted += hidden1["energy"].sum(axis=1)
        unit = _unit_sensitivity(hidden0, hidden1, weights)
        dedgs.append(unit)
        activations.append((block, weights, hidden0, hidden1, unit))
        if force is not None:
            if block.d_fingerprints is None:
                raise ValueError("force loss requested without fingerprint derivatives")
            force -= _contract(unit, block.d_fingerprints)
    energy_loss, force_loss = squared_error(
        predicted,
        landscape.energies,
        force,
        landscape.forces if force is not None else None,
        landscape.n_atoms,
    )
    loss = energy_coefficient * energy_loss + force_coefficient * force_loss
    gradient = np.zeros_like(flat)
    residual = predicted - landscape.energies[:, None]
    upstream_energy = 2.0 * energy_coefficient * residual / (landscape.n_atoms[:, None] ** 2)
    upstream_force = None
    if force is not None and force_coefficient:
        diff = force - landscape.forces[:, None, :, :]
        scale = (2.0 * force_coefficient) / (3.0 * landscape.n_atoms)
        upstream_force = diff * scale[:, None, None, None]
    for block, weights, hidden0, hidden1, unit in activations:
        upstream_g = None
        if upstream_force is not None:
            contracted = _contract_upstream(upstream_force, block.d_fingerprints)
            upstream_g = -contracted
        _accumulate_gradient(
            gradient,
            landscape.layout,
            block,
            weights,
            hidden0,
            hidden1,
            unit,
            upstream_energy,
            upstream_g,
        )
    norms = parameter_set_norms(gradient, landscape.layout)
    energy_rmse = np.sqrt(energy_loss / landscape.n_images)
    if force_coefficient:
        force_rmse = np.sqrt(force_loss / landscape.n_images)
    else:
        force_rmse = np.full(batch, -1.0, dtype=np.float64)
    return {
        "loss": loss,
        "gradient": gradient,
        "grad_norm": norms,
        "energy_rmse": energy_rmse,
        "force_rmse": force_rmse,
        "energy": predicted,
    }


def _forward(fingerprints: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    z0 = np.einsum("bhf,iaf->iabh", weights["W0"], fingerprints) + weights["b0"][None, None, :, :]
    h0 = np.tanh(z0)
    z1 = np.einsum("bqh,iabh->iabq", weights["W1"], h0) + weights["b1"][None, None, :, :]
    h1 = np.tanh(z1)
    energy = np.einsum("bq,iabq->iab", weights["W2"][:, 0, :], h1) + weights["b2"][None, None, :, 0]
    return {"z0": z0, "h0": h0}, {"z1": z1, "h1": h1, "energy": energy}


def _unit_sensitivity(hidden0: dict[str, np.ndarray], hidden1: dict[str, np.ndarray], weights: dict[str, np.ndarray]) -> np.ndarray:
    w2 = weights["W2"][:, 0, :]
    d1 = (1.0 - hidden1["h1"] ** 2) * w2[None, None, :, :]
    pre = np.einsum("bqh,iabq->iabh", weights["W1"], d1)
    d0 = (1.0 - hidden0["h0"] ** 2) * pre
    return np.einsum("bhf,iabh->iabf", weights["W0"], d0)


def _contract_upstream(upstream_force: np.ndarray, d_fingerprints: np.ndarray) -> np.ndarray:
    n_images, batch, n_atoms, _ = upstream_force.shape
    n_local = d_fingerprints.shape[1]
    width = d_fingerprints.shape[2]
    left = upstream_force.reshape(n_images, batch, n_atoms * 3)
    right = d_fingerprints.reshape(n_images, n_local * width, n_atoms * 3)
    contracted = np.einsum("ibq,ipq->ibp", left, right, optimize=True)
    return contracted.reshape(n_images, batch, n_local, width).transpose(0, 2, 1, 3)


def _accumulate_gradient(
    gradient: np.ndarray,
    layout: Layout,
    block: SpeciesBlock,
    weights: dict[str, np.ndarray],
    hidden0: dict[str, np.ndarray],
    hidden1: dict[str, np.ndarray],
    dedg: np.ndarray,
    upstream_energy: np.ndarray,
    upstream_g: np.ndarray | None,
) -> None:
    index = block.atom_index
    n_local = hidden0["h0"].shape[1]
    upstream = np.broadcast_to(upstream_energy[:, None, :], (upstream_energy.shape[0], n_local, upstream_energy.shape[1]))
    h0 = hidden0["h0"]
    h1 = hidden1["h1"]
    w2 = weights["W2"][:, 0, :]
    delta1 = (1.0 - h1**2) * w2[None, None, :, :] * upstream[..., None]
    d_w2 = np.einsum("iab,iabq->bq", upstream, h1)
    d_b2 = np.einsum("iab->b", upstream)
    d_w1 = np.einsum("iabq,iabh->bqh", delta1, h0)
    d_b1 = np.einsum("iabq->bq", delta1)
    back = np.einsum("bqh,iabq->iabh", weights["W1"], delta1)
    delta0 = (1.0 - h0**2) * back
    d_w0 = np.einsum("iabh,iaf->bhf", delta0, block.fingerprints)
    d_b0 = np.einsum("iabh->bh", delta0)
    if upstream_g is not None:
        d1 = (1.0 - h1**2) * w2[None, None, :, :]
        pre = np.einsum("bqh,iabq->iabh", weights["W1"], d1)
        d0 = (1.0 - h0**2) * pre
        v0 = np.einsum("bhf,iabf->iabh", weights["W0"], upstream_g)
        ds_dpre = v0 * (1.0 - h0**2)
        ds_dh0 = -2.0 * h0 * v0 * pre
        ds_dd1 = np.einsum("bqh,iabh->iabq", weights["W1"], ds_dpre)
        d_w1 += np.einsum("iabq,iabh->bqh", d1, ds_dpre)
        sech1 = 1.0 - h1**2
        d_w2 += np.einsum("iabq,iabq->bq", ds_dd1, sech1)
        ds_dh1 = -2.0 * h1 * w2[None, None, :, :] * ds_dd1
        ds_dz1 = ds_dh1 * sech1
        d_w1 += np.einsum("iabq,iabh->bqh", ds_dz1, h0)
        d_b1 += np.einsum("iabq->bq", ds_dz1)
        ds_dh0 += np.einsum("bqh,iabq->iabh", weights["W1"], ds_dz1)
        ds_dz0 = ds_dh0 * (1.0 - h0**2)
        d_w0 += np.einsum("iabh,iaf->bhf", ds_dz0, block.fingerprints)
        d_b0 += np.einsum("iabh->bh", ds_dz0)
        d_w0 += np.einsum("iabh,iabf->bhf", d0, upstream_g)
    pieces = {"W0": d_w0, "b0": d_b0, "W1": d_w1, "b1": d_b1, "W2": d_w2[:, None, :], "b2": d_b2[:, None]}
    del index, dedg
    for spec in layout.specs:
        if spec.species != block.symbol:
            continue
        gradient[:, spec.offset : spec.offset + spec.size] += pieces[spec.name].reshape(gradient.shape[0], spec.size)


def parameter_set_norms(gradient: np.ndarray, layout: Layout) -> np.ndarray:
    """Largest Euclidean norm of one weight or bias tensor, per start."""
    flat = np.asarray(gradient, dtype=np.float64)
    single = flat.ndim == 1
    if single:
        flat = flat[None, :]
    norms = np.zeros(flat.shape[0], dtype=np.float64)
    for spec in layout.specs:
        piece = flat[:, spec.offset : spec.offset + spec.size]
        norms = np.maximum(norms, np.sqrt(np.sum(piece * piece, axis=1)))
    return norms[0] if single else norms
