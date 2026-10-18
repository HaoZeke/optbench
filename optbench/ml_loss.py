"""Squared-error loss on Behler-Parrinello fingerprints.

The dataset is the problem trajectory. The loss is the pyamff SE sum at
commit 8037eb98 with ``use_cohesive_energy`` false and a NoScaler slope
of 1:

    energyloss = sum_images ((E_pred - E_ref) / N) ** 2
    forceloss = sum_images (sum of squared force-component errors) / (3 N)
    loss = energy_coefficient * energyloss + force_coefficient * forceloss

The energy RMSE is ``sqrt(energyloss / n_images)``. The force RMSE is
``sqrt(forceloss / n_images)``. A force coefficient of 0 leaves the force
term out of the graph and reports a force RMSE of -1.

Fingerprints follow ``calculate_G1`` and ``calculate_G2`` in that commit.
The cutoff is ``0.5 * (cos(pi R / Rc) + 1)``. G2 pairs are neighbors
sorted by atom index, and the lower-index symbol has to match
``neighbor1``. ``thetas`` is 0 in the shipped files and that commit does
not subtract it. The neighbor-neighbor distance is the raw Cartesian
difference. Neighbors are the ASE list inside ``Rc``.

The network is one 10-7 tanh net per element, linear output, bias on,
float64. Atomic energies sum to the image energy. Forces are ``-dE/dR``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HIDDEN = (10, 7)
ACTIVATION = "tanh"
ENERGY_COEFFICIENT = 1.0
SLOPE = 1.0
MISSING_FORCE_RMSE = -1.0
CACHE_VERSION = "1"

FP_FILES = {
    "mlopt/emt-pt38": "archive/entries/mlopt/emt-pt38/pyamff-lbfgs10-7-fcoeff0/fpParas.dat",
    "mlopt/C2H4": "archive/entries/mlopt/C2H4/pyamff-lbfgs10-7-fcoeff0/fpParas.dat",
}


@dataclass(frozen=True)
class G1:
    center: str
    neighbor: str
    eta: float
    rs: float
    rc: float
    slot: int


@dataclass(frozen=True)
class G2:
    center: str
    neighbor1: str
    neighbor2: str
    eta: float
    zeta: float
    lam: float
    thetas: float
    rc: float
    slot: int


@dataclass(frozen=True)
class Fingerprints:
    elements: tuple[str, ...]
    g1: tuple[G1, ...]
    g2: tuple[G2, ...]
    n_fp: dict[str, int]
    rc: float

    @property
    def width(self) -> int:
        widths = {count for count in self.n_fp.values()}
        if len(widths) != 1:
            raise ValueError("elements have different fingerprint widths")
        return next(iter(widths))


def parameter_count(n_fp: int, hidden: tuple[int, int] = HIDDEN) -> int:
    """Weights and biases of one element network."""
    first, second = hidden
    return n_fp * first + first + first * second + second + second + 1


def parse_fp(path: Path) -> Fingerprints:
    """Read a pyamff ``fpParas.dat`` file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 7 or lines[1].split()[0] != "BP":
        raise ValueError(f"{path} is not a BP fingerprint file")
    elements = tuple(lines[3].split())
    n_g1, n_g2 = (int(part) for part in lines[6].split()[:2])
    slots = {element: 0 for element in elements}
    g1: list[G1] = []
    g2: list[G2] = []
    for line in lines[7:]:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        center = fields[1]
        if center not in slots:
            raise ValueError(f"{path} names {center} outside the element line")
        slot = slots[center]
        slots[center] = slot + 1
        if fields[0] == "G1":
            g1.append(
                G1(
                    center=center,
                    neighbor=fields[2],
                    eta=float(fields[3]),
                    rs=float(fields[4]),
                    rc=float(fields[5]),
                    slot=slot,
                )
            )
        elif fields[0] == "G2":
            thetas = float(fields[7])
            if thetas != 0.0:
                raise ValueError(f"{path} has a nonzero G2 thetas")
            g2.append(
                G2(
                    center=center,
                    neighbor1=fields[2],
                    neighbor2=fields[3],
                    eta=float(fields[4]),
                    zeta=float(fields[5]),
                    lam=float(fields[6]),
                    thetas=thetas,
                    rc=float(fields[8]),
                    slot=slot,
                )
            )
        else:
            raise ValueError(f"{path} has fingerprint type {fields[0]}")
    if len(g1) != n_g1 or len(g2) != n_g2:
        raise ValueError(f"{path} count line does not match the body")
    radii = {row.rc for row in g1} | {row.rc for row in g2}
    if len(radii) != 1:
        raise ValueError(f"{path} mixes cutoff radii")
    return Fingerprints(
        elements=elements,
        g1=tuple(g1),
        g2=tuple(g2),
        n_fp=dict(slots),
        rc=next(iter(radii)),
    )


def cutoff(distance: float, rc: float) -> float:
    return 0.5 * (math.cos(math.pi * distance / rc) + 1.0)


def g1_value(symbols: list[str], distances: list[float], row: G1) -> float:
    """One G1, the pyamff loop."""
    total = 0.0
    for symbol, distance in zip(symbols, distances, strict=True):
        if symbol != row.neighbor:
            continue
        total += math.exp(-row.eta * ((distance - row.rs) ** 2) / (row.rc**2)) * cutoff(distance, row.rc)
    return total


def g2_value(
    indices: list[int],
    symbols: list[str],
    distances: list[float],
    angles_deg: list[float],
    positions: np.ndarray,
    row: G2,
) -> float:
    """One G2, the pyamff loop. ``angles_deg`` walks every neighbor pair."""
    total = 0.0
    pair = 0
    count = len(indices)
    pref = 2.0 ** (1.0 - row.zeta)
    for left in range(count):
        for right in range(left + 1, count):
            angle = angles_deg[pair]
            pair += 1
            if symbols[left] != row.neighbor1 or symbols[right] != row.neighbor2:
                continue
            rij = distances[left]
            rik = distances[right]
            jk = positions[indices[right]] - positions[indices[left]]
            rjk = float(np.linalg.norm(jk))
            cosine = math.cos(math.radians(angle))
            angular = (1.0 + row.lam * cosine) ** row.zeta
            radial = (
                math.exp(-row.eta * (rij**2) / (row.rc**2))
                * math.exp(-row.eta * (rik**2) / (row.rc**2))
                * math.exp(-row.eta * (rjk**2) / (row.rc**2))
            )
            total += (
                pref
                * angular
                * radial
                * cutoff(rij, row.rc)
                * cutoff(rik, row.rc)
                * cutoff(rjk, row.rc)
            )
    if pair != len(angles_deg):
        raise ValueError("angle list does not match the neighbor pairs")
    return total


def squared_error(
    pred_energy,
    ref_energy,
    pred_forces,
    ref_forces,
    n_atoms: int,
    energy_coefficient: float,
    force_coefficient: float,
) -> dict[str, float]:
    """The SE sum and the two RMSEs."""
    pred_energy = np.asarray(pred_energy, dtype=np.float64)
    ref_energy = np.asarray(ref_energy, dtype=np.float64)
    n_images = int(pred_energy.shape[0])
    energyloss = float(np.sum(((pred_energy - ref_energy) / n_atoms) ** 2))
    if force_coefficient == 0.0:
        forceloss = 0.0
        force_rmse = MISSING_FORCE_RMSE
    else:
        diff = np.asarray(pred_forces, dtype=np.float64) - np.asarray(ref_forces, dtype=np.float64)
        forceloss = float(np.sum(np.sum(diff**2, axis=(1, 2)) / (3.0 * n_atoms)))
        force_rmse = float(math.sqrt(forceloss / n_images))
    loss = energy_coefficient * energyloss + force_coefficient * forceloss
    return {
        "loss": float(loss),
        "energyloss": energyloss,
        "forceloss": forceloss,
        "energy_rmse": float(math.sqrt(energyloss / n_images)),
        "force_rmse": force_rmse,
    }


def neighbor_indices(atoms, rc: float) -> list[np.ndarray]:
    """ASE neighbors inside ``rc``, sorted by atom index."""
    from ase.neighborlist import NeighborList

    count = len(atoms)
    listing = NeighborList(
        cutoffs=[rc / 2.0] * count,
        sorted=False,
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    listing.update(atoms)
    found: list[np.ndarray] = []
    for center in range(count):
        indices, _offsets = listing.get_neighbors(center)
        indices = np.sort(np.asarray(indices, dtype=np.int64))
        if len(indices) != len(np.unique(indices)):
            raise ValueError("a neighbor index is repeated")
        found.append(indices)
    return found


def neighbor_data(atoms, rc: float) -> dict:
    """Neighbor indices plus the MIC distances and angles the pyamff loops read."""
    neighbors = neighbor_indices(atoms, rc)
    distances: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    for center, indices in enumerate(neighbors):
        dist = np.array(
            [float(atoms.get_distance(center, int(other), mic=True)) for other in indices],
            dtype=np.float64,
        )
        angs = [
            float(atoms.get_angle(int(indices[left]), center, int(indices[right]), mic=True))
            for left in range(len(indices))
            for right in range(left + 1, len(indices))
        ]
        distances.append(dist)
        angles.append(np.asarray(angs, dtype=np.float64))
    return {
        "positions": np.asarray(atoms.get_positions(), dtype=np.float64),
        "symbols": list(atoms.get_chemical_symbols()),
        "cell": np.asarray(atoms.get_cell(), dtype=np.float64),
        "neighbors": neighbors,
        "distances": distances,
        "angles": angles,
    }


def reference_fingerprints(geometry: dict, spec: Fingerprints) -> np.ndarray:
    """Fingerprints from the pyamff loops on one image."""
    symbols = geometry["symbols"]
    positions = geometry["positions"]
    width = spec.width
    values = np.zeros((len(symbols), width), dtype=np.float64)
    for center, symbol in enumerate(symbols):
        indices = [int(item) for item in geometry["neighbors"][center]]
        neigh_symbols = [symbols[item] for item in indices]
        dists = [float(item) for item in geometry["distances"][center]]
        angs = [float(item) for item in geometry["angles"][center]]
        for row in spec.g1:
            if row.center == symbol:
                values[center, row.slot] = g1_value(neigh_symbols, dists, row)
        for row in spec.g2:
            if row.center == symbol:
                values[center, row.slot] = g2_value(indices, neigh_symbols, dists, angs, positions, row)
    return values


def _integer_power(base, zeta: float):
    exponent = int(round(zeta))
    if abs(zeta - exponent) > 1e-8:
        raise ValueError(f"zeta {zeta} is not an integer")
    out = base
    for _ in range(exponent - 1):
        out = out * base
    if exponent == 0:
        return base * 0 + 1
    if exponent == 1:
        return base
    return out


def _mic(delta, cell):
    """Minimum image of Cartesian rows. ``cell`` stores lattice vectors as rows."""
    import torch

    shape = delta.shape
    flat = delta.reshape(-1, 3)
    frac = torch.linalg.solve(cell.T, flat.T).T
    frac = frac - torch.floor(frac + 0.5)
    return (frac @ cell).reshape(shape)


def fingerprints_torch(positions, symbols: list[str], neighbors: list[np.ndarray], cell, spec: Fingerprints):
    """Differentiable fingerprints for one image. ``positions`` is a tensor."""
    import torch

    count = int(positions.shape[0])
    codes = {element: index for index, element in enumerate(spec.elements)}
    sym = torch.tensor([codes[symbol] for symbol in symbols], dtype=torch.long)
    mask = torch.zeros((count, count), dtype=torch.bool)
    for center, indices in enumerate(neighbors):
        if len(indices):
            mask[center, torch.as_tensor(indices, dtype=torch.long)] = True
    delta = _mic(positions[None, :, :] - positions[:, None, :], cell)
    distance = torch.linalg.norm(delta, dim=-1)
    width = spec.width
    columns = [positions.new_zeros(count) for _ in range(width)]
    for row in spec.g1:
        center = sym == codes[row.center]
        neigh = sym == codes[row.neighbor]
        pair = mask & center[:, None] & neigh[None, :]
        radial = torch.exp(-row.eta * ((distance - row.rs) ** 2) / (row.rc**2))
        damp = 0.5 * (torch.cos(math.pi * distance / row.rc) + 1.0)
        term = torch.where(pair, radial * damp, torch.zeros_like(distance))
        columns[row.slot] = columns[row.slot] + term.sum(dim=1)
    index = torch.arange(count)
    lower = index[None, :, None] < index[None, None, :]
    pair3 = mask[:, :, None] & mask[:, None, :] & lower
    raw = positions[None, :, :] - positions[:, None, :]
    rjk = torch.linalg.norm(raw, dim=-1)
    safe = distance > 1e-12
    denom = torch.where(safe, distance, torch.ones_like(distance))
    unit = torch.where(safe[:, :, None], delta / denom[:, :, None], torch.zeros_like(delta))
    cosine = (unit[:, :, None, :] * unit[:, None, :, :]).sum(dim=-1)
    cosine = cosine.clamp(-1.0, 1.0)
    for row in spec.g2:
        center = sym == codes[row.center]
        left = sym == codes[row.neighbor1]
        right = sym == codes[row.neighbor2]
        chosen = pair3 & center[:, None, None] & left[None, :, None] & right[None, None, :]
        rij = distance[:, :, None]
        rik = distance[:, None, :]
        pref = 2.0 ** (1.0 - row.zeta)
        angular = _integer_power(1.0 + row.lam * cosine, row.zeta)
        radial = (
            torch.exp(-row.eta * (rij**2) / (row.rc**2))
            * torch.exp(-row.eta * (rik**2) / (row.rc**2))
            * torch.exp(-row.eta * (rjk[None, :, :] ** 2) / (row.rc**2))
        )
        damp = (
            (0.5 * (torch.cos(math.pi * rij / row.rc) + 1.0))
            * (0.5 * (torch.cos(math.pi * rik / row.rc) + 1.0))
            * (0.5 * (torch.cos(math.pi * rjk[None, :, :] / row.rc) + 1.0))
        )
        term = torch.where(chosen, pref * angular * radial * damp, torch.zeros_like(cosine))
        columns[row.slot] = columns[row.slot] + term.sum(dim=(1, 2))
    return torch.stack(columns, dim=1)


def fingerprints_and_jacobian(geometry: dict, spec: Fingerprints) -> tuple[np.ndarray, np.ndarray]:
    """Values and ``dG/dR`` for one image."""
    import torch
    from torch.func import jacrev

    symbols = geometry["symbols"]
    cell = torch.tensor(geometry["cell"], dtype=torch.float64)
    neighbors = geometry["neighbors"]

    def values(positions):
        return fingerprints_torch(positions, symbols, neighbors, cell, spec)

    point = torch.tensor(geometry["positions"], dtype=torch.float64)
    fps = values(point).detach().numpy()
    jacobian = jacrev(values)(point).detach().numpy()
    return np.asarray(fps, dtype=np.float64), np.asarray(jacobian, dtype=np.float64)


def read_trajectory(path: Path) -> list:
    import ase.io

    frames = ase.io.read(str(path), index=":")
    if not frames:
        raise ValueError(f"{path} has no frames")
    return frames


def trajectory_arrays(frames: list) -> dict:
    """Energies, forces, and per-image neighbor geometry. Composition is fixed."""
    symbols = list(frames[0].get_chemical_symbols())
    count = len(symbols)
    if any(list(frame.get_chemical_symbols()) != symbols for frame in frames):
        raise ValueError("the trajectory changes composition or order")
    energy = np.array([float(frame.get_potential_energy()) for frame in frames], dtype=np.float64)
    forces = np.stack([np.asarray(frame.get_forces(), dtype=np.float64) for frame in frames])
    if forces.shape != (len(frames), count, 3):
        raise ValueError("forces do not match the atoms")
    return {"symbols": symbols, "energy": energy, "forces": forces, "frames": frames}


def cache_file(problem_id: str) -> Path:
    slug = problem_id.replace("/", "-")
    return Path("/tmp/optbench-kliff-cache") / f"{slug}-fp-v{CACHE_VERSION}.npz"


def _frame_geometry(frame, rc: float) -> dict:
    return {
        "positions": np.asarray(frame.get_positions(), dtype=np.float64),
        "symbols": list(frame.get_chemical_symbols()),
        "cell": np.asarray(frame.get_cell(), dtype=np.float64),
        "neighbors": neighbor_indices(frame, rc),
    }


def _values_only(geometry: dict, spec: Fingerprints) -> np.ndarray:
    import torch

    positions = torch.tensor(geometry["positions"], dtype=torch.float64)
    cell = torch.tensor(geometry["cell"], dtype=torch.float64)
    values = fingerprints_torch(positions, geometry["symbols"], geometry["neighbors"], cell, spec)
    return values.detach().numpy()


def _jacobian_chunk(job: dict) -> tuple[int, np.ndarray]:
    """Jacobian for one slice of frames. The pool passes the slice bounds."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    spec = job["spec"]
    frames = read_trajectory(Path(job["traj"]))
    start = int(job["start"])
    stop = int(job["stop"])
    chosen = frames[start:stop]
    n_atoms = len(chosen[0])
    block = np.zeros((len(chosen), n_atoms, spec.width, n_atoms, 3), dtype=np.float64)
    for offset, frame in enumerate(chosen):
        _values, deriv = fingerprints_and_jacobian(_frame_geometry(frame, spec.rc), spec)
        block[offset] = deriv
    return start, block


def _parallel_jacobian(traj: Path, spec: Fingerprints, n_images: int, n_atoms: int, workers: int) -> np.ndarray:
    import multiprocessing

    workers = max(1, min(int(workers), n_images))
    edges = [round(index * n_images / workers) for index in range(workers + 1)]
    jobs = []
    for start, stop in zip(edges, edges[1:]):
        if start == stop:
            continue
        jobs.append({"traj": str(traj), "spec": spec, "start": start, "stop": stop})
    jacobian = np.zeros((n_images, n_atoms, spec.width, n_atoms, 3), dtype=np.float64)
    if len(jobs) == 1:
        start, block = _jacobian_chunk(jobs[0])
        jacobian[start : start + len(block)] = block
        return jacobian
    context = multiprocessing.get_context("spawn")
    with context.Pool(len(jobs)) as pool:
        for start, block in pool.imap_unordered(_jacobian_chunk, jobs):
            jacobian[start : start + len(block)] = block
            print(f"jacobian {traj.name} {start + len(block)}/{n_images}", flush=True)
    return jacobian


def ensure_features(root: Path, problem: dict, spec: Fingerprints, *, need_jacobian: bool) -> dict:
    """Load or write the fingerprint cache for this problem."""
    path = cache_file(problem["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        stored = np.load(path, allow_pickle=False)
        ready = (not need_jacobian) or ("jacobian" in stored.files)
        if ready:
            return {key: stored[key] for key in stored.files}
        fps = np.array(stored["fps"], dtype=np.float64)
        energy = np.array(stored["energy"], dtype=np.float64)
        forces = np.array(stored["forces"], dtype=np.float64)
        symbols = np.array(stored["symbols"])
    else:
        fps = energy = forces = symbols = None
    from .campaign import geometry_archive

    archive = geometry_archive(root, problem)
    frames = read_trajectory(archive)
    arrays = trajectory_arrays(frames)
    if symbols is None:
        symbols = np.asarray(arrays["symbols"])
        energy = arrays["energy"]
        forces = arrays["forces"]
        fps = np.zeros((len(frames), len(symbols), spec.width), dtype=np.float64)
        for index, frame in enumerate(frames):
            fps[index] = _values_only(_frame_geometry(frame, spec.rc), spec)
            if index % 25 == 0:
                print(f"fingerprints {problem['id']} {index + 1}/{len(frames)}", flush=True)
    jacobian = None
    if need_jacobian:
        workers = int(os.environ.get("OPTBENCH_JACOBIAN_WORKERS", "8"))
        jacobian = _parallel_jacobian(archive, spec, len(frames), len(symbols), workers)
    payload = {"fps": fps, "energy": energy, "forces": forces, "symbols": symbols}
    if jacobian is not None:
        payload["jacobian"] = jacobian
    elif path.is_file():
        previous = np.load(path, allow_pickle=False)
        if "jacobian" in previous.files:
            payload["jacobian"] = np.array(previous["jacobian"])
    np.savez(path, **payload)
    print(f"fingerprints {problem['id']} wrote {path}", flush=True)
    return payload
