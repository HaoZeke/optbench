"""Behler-Parrinello fingerprints for the PyAMFF mlopt loss.

The neighbor list is the PyAMFF list: ASE ``NeighborList`` with cutoff 6.0 Å
(3.0 Å on each atom), ``bothways``, no self pairs, and skin 0. Indices are
sorted. Distances and angles use the minimum image. The angular G2 term uses
the raw jk separation, which is what ``calculate_G2`` does with ``nl.coords``.

G1 and G2 follow ``pyamff/fingerprints/behlerParrinello.py`` at 8037eb98.
A component is then scaled from its range on the training images to [-1, 1].
A range below 1e-8 is replaced by min -1 and range 2. Derivatives of the
fingerprints, from ``jacrev`` of the same function with the neighbor list held
fixed, are scaled by that magnitude only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_VERSION = 2


def parse_fp_file(path: Path) -> tuple[list[str], dict[str, list[dict]]]:
    """Element order and the G1/G2 rows for each element, in file order."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{path} is not an fpParas file")
    if lines[1].split()[0] != "BP":
        raise ValueError(f"{path} is not a BP fingerprint file")
    header = lines[6].split()
    n_g1, n_g2 = int(header[0]), int(header[1])
    order: list[str] = []
    specs: dict[str, list[dict]] = {}
    seen_g1 = 0
    seen_g2 = 0
    for line in lines[7:]:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        kind = fields[0]
        center = fields[1]
        if center not in specs:
            order.append(center)
            specs[center] = []
        if kind == "G1":
            specs[center].append(
                {
                    "kind": "G1",
                    "neighbor": fields[2],
                    "eta": float(fields[3]),
                    "rs": float(fields[4]),
                    "rc": float(fields[5]),
                }
            )
            seen_g1 += 1
        elif kind == "G2":
            specs[center].append(
                {
                    "kind": "G2",
                    "neighbor1": fields[2],
                    "neighbor2": fields[3],
                    "eta": float(fields[4]),
                    "zeta": float(fields[5]),
                    "lam": float(fields[6]),
                    "rc": float(fields[8]),
                }
            )
            seen_g2 += 1
        else:
            raise ValueError(f"unknown fingerprint {kind} in {path}")
    if seen_g1 != n_g1 or seen_g2 != n_g2:
        raise ValueError(f"{path} declares {n_g1} G1 and {n_g2} G2, found {seen_g1} and {seen_g2}")
    return order, specs


def _neighbor_index(atoms, cutoff: float) -> np.ndarray:
    """Sorted neighbor atom indices, padded with -1. Matches PyAMFF's list."""
    from ase.neighborlist import NeighborList

    half = float(cutoff) / 2.0
    count = len(atoms)
    nl = NeighborList(
        cutoffs=[half] * count,
        sorted=False,
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    nl.update(atoms)
    rows = []
    width = 0
    for i in range(count):
        indices, _offsets = nl.get_neighbors(i)
        ordered = np.sort(np.asarray(indices, dtype=np.int32))
        rows.append(ordered)
        width = max(width, int(ordered.size))
    padded = np.full((count, max(width, 1)), -1, dtype=np.int32)
    for i, ordered in enumerate(rows):
        if ordered.size:
            padded[i, : ordered.size] = ordered
    return padded


def _minimum_image(delta, lengths, pbc):
    import jax.numpy as jnp

    frac = delta / lengths
    shifted = frac - jnp.round(frac)
    return jnp.where(pbc, shifted, frac) * lengths


def _cutoff(distance, rc):
    import jax.numpy as jnp

    return 0.5 * (jnp.cos(jnp.pi * distance / rc) + 1.0)


def _g_one_image(pos, neigh, lengths, *, center_index, symbol_ids, pbc, specs, pair_a, pair_b):
    """Fingerprints of one element on one image. ``pos`` is ``(n, 3)``."""
    import jax.numpy as jnp

    centers = pos[center_index]
    neigh_c = neigh[center_index]
    columns = []
    for spec in specs:
        if spec["kind"] == "G1":
            columns.append(
                _g1(
                    pos,
                    centers,
                    neigh_c,
                    lengths,
                    symbol_ids,
                    pbc,
                    neighbor=spec["neighbor_id"],
                    eta=spec["eta"],
                    rs=spec["rs"],
                    rc=spec["rc"],
                )
            )
        else:
            columns.append(
                _g2(
                    pos,
                    centers,
                    neigh_c,
                    lengths,
                    symbol_ids,
                    pbc,
                    pair_a,
                    pair_b,
                    neighbor1=spec["neighbor1_id"],
                    neighbor2=spec["neighbor2_id"],
                    eta=spec["eta"],
                    zeta=spec["zeta"],
                    lam=spec["lam"],
                    rc=spec["rc"],
                )
            )
    return jnp.stack(columns, axis=-1)


def _safe_norm(delta):
    """Length of ``delta``. A zero vector has derivative 0, not NaN."""
    import jax.numpy as jnp

    return jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-24)


def _g1(pos, centers, neigh_c, lengths, symbol_ids, pbc, *, neighbor, eta, rs, rc):
    import jax.numpy as jnp

    safe = jnp.where(neigh_c >= 0, neigh_c, 0)
    delta = _minimum_image(pos[safe] - centers[:, None, :], lengths, pbc)
    use = (neigh_c >= 0) & (symbol_ids[safe] == neighbor)
    # An empty slot must not differentiate a zero displacement.
    delta = jnp.where(use[..., None], delta, jnp.ones_like(delta))
    dist = _safe_norm(delta)
    val = jnp.exp(-eta * (dist - rs) ** 2 / (rc**2)) * _cutoff(dist, rc)
    return jnp.where(use, val, 0.0).sum(axis=-1)


def _g2(
    pos,
    centers,
    neigh_c,
    lengths,
    symbol_ids,
    pbc,
    pair_a,
    pair_b,
    *,
    neighbor1,
    neighbor2,
    eta,
    zeta,
    lam,
    rc,
):
    import jax.numpy as jnp

    if pair_a.size == 0:
        return jnp.zeros((centers.shape[0],))
    ja = neigh_c[:, pair_a]
    jb = neigh_c[:, pair_b]
    safe_a = jnp.where(ja >= 0, ja, 0)
    safe_b = jnp.where(jb >= 0, jb, 0)
    pos_a = pos[safe_a]
    pos_b = pos[safe_b]
    dij = _minimum_image(pos_a - centers[:, None, :], lengths, pbc)
    dik = _minimum_image(pos_b - centers[:, None, :], lengths, pbc)
    djk = pos_b - pos_a
    use = (ja >= 0) & (jb >= 0) & (symbol_ids[safe_a] == neighbor1) & (symbol_ids[safe_b] == neighbor2)
    dij = jnp.where(use[..., None], dij, jnp.asarray([1.0, 0.0, 0.0]))
    dik = jnp.where(use[..., None], dik, jnp.asarray([0.0, 1.0, 0.0]))
    djk = jnp.where(use[..., None], djk, jnp.asarray([-1.0, 1.0, 0.0]))
    rij = _safe_norm(dij)
    rik = _safe_norm(dik)
    rjk = _safe_norm(djk)
    denom = jnp.maximum(rij * rik, 1e-30)
    cos_t = jnp.clip(jnp.sum(dij * dik, axis=-1) / denom, -1.0, 1.0)
    base = jnp.maximum(1.0 + lam * cos_t, 0.0)
    angular = (2.0 ** (1.0 - zeta)) * (base**zeta)
    radial = (
        jnp.exp(-eta * rij**2 / rc**2)
        * jnp.exp(-eta * rik**2 / rc**2)
        * jnp.exp(-eta * rjk**2 / rc**2)
        * _cutoff(rij, rc)
        * _cutoff(rik, rc)
        * _cutoff(rjk, rc)
    )
    return jnp.where(use, angular * radial, 0.0).sum(axis=-1)


def _normalize(values: np.ndarray, deriv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map each component's range to [-1, 1]. ``deriv`` is scaled by the magnitude."""
    flat = values.reshape(-1, values.shape[-1])
    minv = flat.min(axis=0)
    maxv = flat.max(axis=0)
    span = maxv - minv
    small = span < 1e-8
    min_used = np.where(small, -1.0, minv)
    span_used = np.where(small, 2.0, span)
    mag = 2.0 / span_used
    intercept = -2.0 * min_used / span_used - 1.0
    scaled = values * mag + intercept
    # deriv is (images, centers, n_fp, atoms, 3). The magnitude belongs on n_fp.
    fp_axis = values.ndim - 1
    broadcast = [1] * deriv.ndim
    broadcast[fp_axis] = mag.shape[0]
    scaled_deriv = deriv * mag.reshape(broadcast)
    return scaled, scaled_deriv, mag, intercept


def _derivative_batch(natoms: int) -> int:
    """Images per derivative scan. Large cells do not fit in one vmap."""
    if natoms >= 20:
        return 4
    return 16


def _pair_slots(width: int) -> tuple[np.ndarray, np.ndarray]:
    first = []
    second = []
    for a in range(width):
        for b in range(a + 1, width):
            first.append(a)
            second.append(b)
    if not first:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    return np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32)


def fingerprints_from_arrays(
    positions: np.ndarray,
    lengths: np.ndarray,
    pbc: np.ndarray,
    symbols: list[str],
    neighbors: np.ndarray,
    order: list[str],
    specs: dict[str, list[dict]],
    *,
    normalize: bool = True,
) -> dict:
    """Normalized fingerprints and fixed-neighbor derivatives.

    ``positions`` is ``(images, atoms, 3)``. ``neighbors`` is
    ``(images, atoms, slots)`` with -1 in empty slots.
    """
    import jax
    import jax.numpy as jnp

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        symbol_index = {name: i for i, name in enumerate(order)}
        # A symbol that appears on the geometry but not in the file still needs an id.
        for name in symbols:
            if name not in symbol_index:
                symbol_index[name] = len(symbol_index)
        symbol_ids = jnp.asarray([symbol_index[name] for name in symbols], dtype=jnp.int32)
        pbc_j = jnp.asarray(np.asarray(pbc, dtype=bool))
        pos_j = jnp.asarray(positions, dtype=jnp.float64)
        len_j = jnp.asarray(lengths, dtype=jnp.float64)
        neigh_j = jnp.asarray(neighbors, dtype=jnp.int32)
        pair_a_np, pair_b_np = _pair_slots(neighbors.shape[-1])
        pair_a = jnp.asarray(pair_a_np)
        pair_b = jnp.asarray(pair_b_np)
        prepared = {
            "n_images": int(positions.shape[0]),
            "natoms": int(positions.shape[1]),
            "elements": list(order),
            "n_fp": {},
            "centers": {},
            "fps": {},
            "dgdr": {},
            "magnitude": {},
            "intercept": {},
        }
        for element in order:
            centers = np.asarray([i for i, name in enumerate(symbols) if name == element], dtype=np.int32)
            if centers.size == 0:
                raise ValueError(f"{element} is in the fingerprint file and not in the trajectory")
            element_specs = []
            for spec in specs[element]:
                row = dict(spec)
                if spec["kind"] == "G1":
                    row["neighbor_id"] = symbol_index[spec["neighbor"]]
                else:
                    row["neighbor1_id"] = symbol_index[spec["neighbor1"]]
                    row["neighbor2_id"] = symbol_index[spec["neighbor2"]]
                element_specs.append(row)
            center_j = jnp.asarray(centers)

            def g_one(pos, neigh, length, specs=element_specs, center_index=center_j):
                return _g_one_image(
                    pos,
                    neigh,
                    length,
                    center_index=center_index,
                    symbol_ids=symbol_ids,
                    pbc=pbc_j,
                    specs=specs,
                    pair_a=pair_a,
                    pair_b=pair_b,
                )

            values_fn = jax.jit(jax.vmap(g_one, in_axes=(0, 0, 0)))
            # jacrev of every image at once allocates a batch Jacobian. Pt 38
            # at 32 images asked for about 90 GB, so the derivative scan
            # evaluates a few images at a time.
            deriv_batch = min(_derivative_batch(int(positions.shape[1])), int(positions.shape[0]))

            def deriv_fn(pos, neigh, length, batch=deriv_batch):
                def one(args):
                    image_pos, image_neigh, image_length = args
                    return jax.jacrev(g_one, argnums=0)(image_pos, image_neigh, image_length)

                return jax.lax.map(one, (pos, neigh, length), batch_size=batch)

            values = np.asarray(values_fn(pos_j, neigh_j, len_j), dtype=np.float64)
            deriv = np.asarray(jax.jit(deriv_fn)(pos_j, neigh_j, len_j), dtype=np.float64)
            if normalize:
                scaled, scaled_deriv, mag, intercept = _normalize(values, deriv)
            else:
                scaled = values
                scaled_deriv = deriv
                mag = np.ones(values.shape[-1], dtype=np.float64)
                intercept = np.zeros(values.shape[-1], dtype=np.float64)
            prepared["n_fp"][element] = int(values.shape[-1])
            prepared["centers"][element] = centers
            prepared["fps"][element] = scaled
            prepared["dgdr"][element] = scaled_deriv
            prepared["magnitude"][element] = mag
            prepared["intercept"][element] = intercept
        return prepared
    finally:
        jax.config.update("jax_enable_x64", previous)


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_MLOPT_CACHE", "/tmp/optbench-mlopt"))


def _digest(traj: Path, fp_file: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"v{CACHE_VERSION}".encode())
    hasher.update(fp_file.read_bytes())
    hasher.update(traj.read_bytes())
    return hasher.hexdigest()[:16]


def load_trajectory(traj: Path, fp_file: Path, *, cutoff: float = 6.0) -> dict:
    """Fingerprints for every image, cached under ``OPTBENCH_MLOPT_CACHE``."""
    import fcntl

    from ase.io import read

    cache = _cache_dir() / _digest(traj, fp_file)
    manifest = cache / "manifest.json"
    if manifest.is_file():
        return _read_cache(cache)
    cache.mkdir(parents=True, exist_ok=True)
    lock_path = cache / ".lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if manifest.is_file():
            return _read_cache(cache)
        images = read(str(traj), index=":")
        if not images:
            raise ValueError(f"{traj} has no images")
        symbols = list(images[0].get_chemical_symbols())
        for image in images:
            if list(image.get_chemical_symbols()) != symbols:
                raise ValueError(f"{traj} changes composition between images")
        order, specs = parse_fp_file(fp_file)
        positions = np.stack([image.get_positions() for image in images]).astype(np.float64)
        forces = np.stack([image.get_forces() for image in images]).astype(np.float64)
        energies = np.asarray([image.get_potential_energy() for image in images], dtype=np.float64)
        lengths = np.stack([image.cell.lengths() for image in images]).astype(np.float64)
        pbc = np.asarray(images[0].pbc, dtype=bool)
        blocks = [_neighbor_index(image, cutoff) for image in images]
        width = max(int(block.shape[1]) for block in blocks)
        neighbors = np.full((len(images), len(symbols), width), -1, dtype=np.int32)
        for i, block in enumerate(blocks):
            neighbors[i, :, : block.shape[1]] = block
        prepared = fingerprints_from_arrays(positions, lengths, pbc, symbols, neighbors, order, specs)
        prepared["energies"] = energies
        prepared["forces"] = forces
        _write_cache(cache, prepared, symbols)
        return _read_cache(cache)


def _write_cache(cache: Path, prepared: dict, symbols: list[str]) -> None:
    manifest = {
        "version": CACHE_VERSION,
        "symbols": symbols,
        "elements": prepared["elements"],
        "n_images": prepared["n_images"],
        "natoms": prepared["natoms"],
        "n_fp": prepared["n_fp"],
    }
    np.save(cache / "energies.npy", prepared["energies"])
    np.save(cache / "forces.npy", prepared["forces"])
    for element in prepared["elements"]:
        np.save(cache / f"{element}_fps.npy", prepared["fps"][element])
        np.save(cache / f"{element}_dgdr.npy", prepared["dgdr"][element])
        np.save(cache / f"{element}_centers.npy", prepared["centers"][element])
        np.save(cache / f"{element}_magnitude.npy", prepared["magnitude"][element])
        np.save(cache / f"{element}_intercept.npy", prepared["intercept"][element])
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _read_cache(cache: Path) -> dict:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    prepared = {
        "n_images": int(manifest["n_images"]),
        "natoms": int(manifest["natoms"]),
        "elements": list(manifest["elements"]),
        "n_fp": {key: int(value) for key, value in manifest["n_fp"].items()},
        "symbols": list(manifest["symbols"]),
        "energies": np.load(cache / "energies.npy"),
        "forces": np.load(cache / "forces.npy"),
        "centers": {},
        "fps": {},
        "dgdr": {},
        "magnitude": {},
        "intercept": {},
    }
    for element in prepared["elements"]:
        prepared["centers"][element] = np.load(cache / f"{element}_centers.npy")
        prepared["fps"][element] = np.load(cache / f"{element}_fps.npy")
        prepared["dgdr"][element] = np.load(cache / f"{element}_dgdr.npy")
        prepared["magnitude"][element] = np.load(cache / f"{element}_magnitude.npy")
        prepared["intercept"][element] = np.load(cache / f"{element}_intercept.npy")
    return prepared
