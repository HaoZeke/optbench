"""Pyamff loss trained by the n2p2 standard Kalman filter.

The potential is the archived Behler-Parrinello network: fingerprints from
``fpParas.dat`` and a 10-7 tanh network. n2p2 supplies
``KalmanFilter::update``. A force call is one evaluation of the loss
gradient. Per-structure Kalman derivatives stay inside the updater.

Energy rows use the squared per-atom energy error. Force rows add the
pyamff squared force error with the archived force coefficient. The
Kalman parameters are the documented standard-filter defaults.
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

H1 = 10
H2 = 7
SEED_BASE = 20261015
N_STARTS = 50
EPOCH_CAP = 10000
GRAD_TOL = 0.02
CODE = "n2p2"
CODE_VERSION = "2.3.0"
N2P2_COMMIT = "29b9c9f10b1ac9ea45631b3d556a129ca3b701c2"
PYAMFF_COMMIT = "8037eb98e2288c132c112aa6c2032f00cd5c5277"
FORCE_RMSE = -1.0

# examples/input.nn.recommended, standard Kalman block (updater_type 1).
# The fading-memory block in that file is commented and marked not recommended.
KALMAN_EPSILON = 1.0e-2
KALMAN_Q0 = 0.01
KALMAN_QTAU = 2.302
KALMAN_QMIN = 1.0e-6
KALMAN_ETA0 = 0.01
KALMAN_ETATAU = 2.302
KALMAN_ETAMAX = 1.0

# Same recommended file, with use_short_forces set.
FORCE_WEIGHT = 1.0
FORCE_ENERGY_RATIO = 3.0
N2P2_RANDOM_SEED = 1234567

# pyamff fcoeff0_1 configs: SE, these coefficients, this cap and gradient tolerance.
ENERGY_COEFFICIENT = 1.0
FORCE_COEFFICIENT = 0.1
FORCE_GRAD_TOL = 0.1
FORCE_EPOCH_CAP = 30000

_ANGLE_CLIP = 0.999999999999

TRAJECTORY_PATHS = {
    "mlopt/emt-pt38": "website/mlopt/emt-pt38/pt_md.traj",
    "mlopt/C2H4": "website/mlopt/C2H4/C2H4_md.traj",
}
FPPARAS_PATHS = {
    "mlopt/emt-pt38": "archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0/fpParas.dat",
    "mlopt/C2H4": "archive/entries/mlopt/C2H4/pyamff-adam10-7-fcoeff0/fpParas.dat",
}
FPPARAS_FORCE_PATHS = {
    "mlopt/emt-pt38": "archive/entries/mlopt/emt-pt38/pyamff-adam10-7-fcoeff0_1/fpParas.dat",
    "mlopt/C2H4": "archive/entries/mlopt/C2H4/pyamff-adam10-7-fcoeff0_1/fpParas.dat",
}

_LIB: ctypes.CDLL | None = None


def library_path() -> Path:
    override = os.environ.get("N2P2_KALMAN_LIB")
    if override:
        return Path(override)
    return Path("/tmp/optbench-n2p2/libn2p2kalman.so")


def n2p2_source() -> Path:
    return Path(os.environ.get("N2P2_SRC", "/tmp/n2p2"))


def build_library() -> Path:
    """Compile the Kalman wrapper against a local n2p2 checkout."""
    dest = library_path()
    source = n2p2_source()
    kalman = source / "src" / "libnnptrain" / "KalmanFilter.cpp"
    wrapper = Path(__file__).resolve().parent / "native" / "n2p2_kalman.cpp"
    if dest.is_file() and dest.stat().st_mtime >= wrapper.stat().st_mtime:
        return dest
    if not kalman.is_file():
        raise FileNotFoundError(f"n2p2 KalmanFilter.cpp is not at {kalman}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get("MPICXX", "mpic++"),
        "-std=c++11",
        "-O3",
        "-fPIC",
        "-shared",
        "-march=native",
        "-DEIGEN_DONT_PARALLELIZE",
        "-DNDEBUG",
        f"-I{source / 'src' / 'libnnptrain'}",
        f"-I{source / 'src' / 'libnnp'}",
        f"-I{source / 'include'}",
        "-I/usr/include/eigen3",
        str(wrapper),
        str(kalman),
        str(source / "src" / "libnnptrain" / "Updater.cpp"),
        str(source / "src" / "libnnp" / "Stopwatch.cpp"),
        str(source / "src" / "libnnp" / "utility.cpp"),
        "-lgsl",
        "-lgslcblas",
        "-o",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    return dest


def load_library() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build_library()))
        _bind(_LIB)
    return _LIB


def _bind(lib: ctypes.CDLL) -> None:
    c_int = ctypes.c_int
    c_double = ctypes.c_double
    ptr_int = ctypes.POINTER(c_int)
    ptr_double = ctypes.POINTER(c_double)
    lib.n2p2_create.restype = ctypes.c_void_p
    lib.n2p2_create.argtypes = [
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_double,
        ptr_int,
        ptr_int,
        ptr_double,
        ptr_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
    ]
    lib.n2p2_destroy.restype = None
    lib.n2p2_destroy.argtypes = [ctypes.c_void_p]
    lib.n2p2_n_weights.restype = c_int
    lib.n2p2_n_weights.argtypes = [ctypes.c_void_p]
    lib.n2p2_set_weights.restype = c_int
    lib.n2p2_set_weights.argtypes = [ctypes.c_void_p, ptr_double, c_int]
    lib.n2p2_get_weights.restype = c_int
    lib.n2p2_get_weights.argtypes = [ctypes.c_void_p, ptr_double, c_int]
    lib.n2p2_eval.restype = c_int
    lib.n2p2_eval.argtypes = [
        ctypes.c_void_p,
        ptr_double,
        c_int,
        ptr_double,
        c_int,
        ptr_double,
        ptr_double,
        ptr_int,
        ptr_int,
    ]
    lib.n2p2_epoch.restype = c_int
    lib.n2p2_epoch.argtypes = [ctypes.c_void_p, ptr_int, c_int]
    lib.n2p2_updates.restype = ctypes.c_size_t
    lib.n2p2_updates.argtypes = [ctypes.c_void_p]
    lib.n2p2_force_rmse.restype = c_double
    lib.n2p2_force_rmse.argtypes = [ctypes.c_void_p]
    lib.n2p2_enable_forces.restype = c_int
    lib.n2p2_enable_forces.argtypes = [
        ctypes.c_void_p,
        ptr_double,
        ptr_double,
        c_double,
        c_double,
        c_double,
        c_int,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
        c_double,
    ]
    lib.n2p2_epoch_mixed.restype = c_int
    lib.n2p2_epoch_mixed.argtypes = [
        ctypes.c_void_p,
        ptr_int,
        c_int,
        ptr_int,
        c_int,
        ptr_int,
        c_int,
    ]
    lib.n2p2_rng_create.restype = ctypes.c_void_p
    lib.n2p2_rng_create.argtypes = [ctypes.c_ulong]
    lib.n2p2_rng_destroy.restype = None
    lib.n2p2_rng_destroy.argtypes = [ctypes.c_void_p]
    lib.n2p2_rng_shuffle.restype = c_int
    lib.n2p2_rng_shuffle.argtypes = [ctypes.c_void_p, ptr_int, c_int]
    lib.n2p2_rng_shuffle_schedule.restype = c_int
    lib.n2p2_rng_shuffle_schedule.argtypes = [ctypes.c_void_p, ptr_int, c_int]


def _as_int(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.int32)


def _as_float(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def _ptr_int(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _ptr_float(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@dataclass(frozen=True)
class G1:
    center: int
    neighbor: int
    eta: float
    rs: float
    rc: float
    index: int


@dataclass(frozen=True)
class G2:
    center: int
    neighbor_a: int
    neighbor_b: int
    eta: float
    zeta: float
    lam: float
    theta_s: float
    rc: float
    index: int


@dataclass(frozen=True)
class FingerprintSpec:
    elements: tuple[str, ...]
    n_fp: tuple[int, ...]
    g1: tuple[G1, ...]
    g2: tuple[G2, ...]
    rc_max: float

    @property
    def stride(self) -> int:
        return max(self.n_fp)


def element_width(n_fp: int) -> int:
    return H1 * n_fp + H1 + H2 * H1 + H2 + H2 + 1


def weight_offsets(n_fp: tuple[int, ...] | list[int]) -> list[tuple[int, int, int]]:
    """``(offset, width, n_fp)`` for each element, in fingerprint order."""
    found = []
    cursor = 0
    for count in n_fp:
        width = element_width(int(count))
        found.append((cursor, width, int(count)))
        cursor += width
    return found


def n_weights(n_fp: tuple[int, ...] | list[int]) -> int:
    return sum(element_width(int(count)) for count in n_fp)


def parameter_slices(n_fp: tuple[int, ...] | list[int]) -> list[tuple[int, int]]:
    """One slice per Linear weight or bias, element by element."""
    slices = []
    for offset, _width, count in weight_offsets(n_fp):
        sizes = (H1 * count, H1, H2 * H1, H2, H2, 1)
        cursor = offset
        for size in sizes:
            slices.append((cursor, cursor + size))
            cursor += size
    return slices


def _content_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_fpparas(text: str) -> FingerprintSpec:
    """Read a pyamff ``fpParas.dat`` the way ``read_fpParas`` does."""
    lines = _content_lines(text)
    if len(lines) < 7:
        raise ValueError("fpParas.dat is shorter than the header")
    if lines[1] != "BP":
        raise ValueError(f"fingerprint type {lines[1]} is not BP")
    elements = tuple(lines[3].split())
    n_elements = len(elements)
    if len(lines[4].split()) != n_elements:
        raise ValueError("cohesive-energy row does not match the element row")
    n_g1, n_g2 = (int(piece) for piece in lines[6].split())
    cursor = 7
    g1_rows: list[tuple[int, int, float, float, float]] = []
    if n_g1:
        cursor += 1
        for _ in range(n_g1):
            parts = lines[cursor].split()
            cursor += 1
            if parts[0] != "G1":
                raise ValueError(f"expected G1, got {parts[0]}")
            g1_rows.append(
                (
                    elements.index(parts[1]),
                    elements.index(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            )
    g2_rows: list[tuple[int, int, int, float, float, float, float, float]] = []
    if n_g2:
        cursor += 1
        for _ in range(n_g2):
            parts = lines[cursor].split()
            cursor += 1
            if parts[0] != "G2":
                raise ValueError(f"expected G2, got {parts[0]}")
            center = elements.index(parts[1])
            left = elements.index(parts[2])
            right = elements.index(parts[3])
            if left > right:
                left, right = right, left
            g2_rows.append(
                (
                    center,
                    left,
                    right,
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                    float(parts[8]),
                )
            )
    g1_counts: dict[tuple[int, int], int] = {}
    for center, neighbor, *_rest in g1_rows:
        key = (center + 1, neighbor + 1)
        g1_counts[key] = g1_counts.get(key, 0) + 1
    g1_end: dict[tuple[int, int], int] = {}
    g1_current: dict[tuple[int, int], int] = {}
    g1: list[G1] = []
    for center, neighbor, eta, rs, rc in g1_rows:
        key = (center + 1, neighbor + 1)
        g1_current[key] = g1_current.get(key, 0) + 1
        if neighbor + 1 == 1:
            g1_end[key] = g1_end.get(key, 0) + 1
            start = 1
        else:
            start = g1_end[(center + 1, neighbor)] + 1
            g1_end[key] = start + g1_counts[key] - 1
        index = start + g1_current[key] - 2
        g1.append(G1(center, neighbor, eta, rs, rc, index))
    g1_endpoint = [0] * n_elements
    for feature in g1:
        g1_endpoint[feature.center] += 1
    g2_counts: dict[tuple[int, int, int], int] = {}
    for center, left, right, *_rest in g2_rows:
        key = (center + 1, left + 1, right + 1)
        g2_counts[key] = g2_counts.get(key, 0) + 1
    g2_end: dict[tuple[int, int, int], int] = {}
    for center in range(1, n_elements + 1):
        for left in range(1, n_elements + 1):
            for right in range(left, n_elements + 1):
                count = g2_counts.get((center, left, right), 0)
                present = 1 if count else 0
                if left == 1 and right == 1:
                    start = g1_endpoint[center - 1] + present
                elif left == right:
                    start = g2_end[(center, left - 1, n_elements)] + present
                else:
                    start = g2_end[(center, left, right - 1)] + present
                g2_end[(center, left, right)] = start if present == 0 else start + count - 1
    g2_current: dict[tuple[int, int, int], int] = {}
    g2: list[G2] = []
    for center, left, right, eta, zeta, lam, theta_s, rc in g2_rows:
        key = (center + 1, left + 1, right + 1)
        g2_current[key] = g2_current.get(key, 0) + 1
        count = g2_counts[key]
        end = g2_end[key]
        start = end - count + 1
        index = start + g2_current[key] - 2
        g2.append(G2(center, left, right, eta, zeta, lam, theta_s, rc, index))
    counts = [0] * n_elements
    for feature in (*g1, *g2):
        counts[feature.center] = max(counts[feature.center], feature.index + 1)
    rc_values = [feature.rc for feature in (*g1, *g2)]
    return FingerprintSpec(
        elements=elements,
        n_fp=tuple(counts),
        g1=tuple(g1),
        g2=tuple(g2),
        rc_max=max(rc_values) if rc_values else 0.0,
    )


def _fc(distance: float, rc: float) -> float:
    return 0.5 * (math.cos(math.pi * distance / rc) + 1.0)


def _fexp(distance: float, rc: float) -> float:
    return math.exp(-(distance * distance) / (rc * rc))


def g2_value(rij: float, rik: float, rjk: float, vj: np.ndarray, vk: np.ndarray, feature: G2) -> float:
    """One angular fingerprint at the center, using the pyamff ``fg2`` product."""
    value, *_grads = g2_cartesian_grad(rij, rik, rjk, vj, vk, vk - vj, feature)
    return value


def g2_cartesian_grad(
    rij: float,
    rik: float,
    rjk: float,
    vj: np.ndarray,
    vk: np.ndarray,
    vjk: np.ndarray,
    feature: G2,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """``fg2`` value and its Cartesian derivatives at the center, ``j``, and ``k``.

    ``vj`` and ``vk`` point from the center to the two neighbors. ``vjk``
    points from ``j`` to ``k``. Outside the cutoff the masked fingerprint is
    identically zero, so the derivative is zero there too.
    """
    zero = np.zeros(3, dtype=np.float64)
    rc = feature.rc
    if rij >= rc or rik >= rc or rjk >= rc or rij <= 0.0 or rik <= 0.0 or rjk <= 0.0:
        return 0.0, zero, zero, zero
    dot = float(vj[0] * vk[0] + vj[1] * vk[1] + vj[2] * vk[2])
    cosine = dot / (rij * rik)
    if abs(cosine) > _ANGLE_CLIP:
        theta = 0.0
        dtheta_dc = 0.0
    else:
        theta = math.acos(cosine)
        dtheta_dc = -1.0 / math.sqrt(1.0 - cosine * cosine)
    delta_theta = theta - feature.theta_s
    base = 1.0 + feature.lam * math.cos(delta_theta)
    angular = base ** (feature.zeta - 1.0)
    inv_rc2 = 1.0 / (rc * rc)
    fexpi = math.exp(-(rij * rij) * inv_rc2)
    fexpk = math.exp(-(rik * rik) * inv_rc2)
    fexpjk = math.exp(-(rjk * rjk) * inv_rc2)
    radial = (fexpi * fexpk * fexpjk) ** feature.eta
    fci = _fc(rij, rc)
    fck = _fc(rik, rc)
    fcjk = _fc(rjk, rc)
    cutoff = fci * fck * fcjk
    scale = 2.0 ** (1.0 - feature.zeta)
    value = angular * base * radial * cutoff * scale
    power = angular * base
    dradial_drij = feature.eta * radial * (-2.0 * rij * inv_rc2)
    dradial_drik = feature.eta * radial * (-2.0 * rik * inv_rc2)
    dradial_drjk = feature.eta * radial * (-2.0 * rjk * inv_rc2)

    def _dfc(distance: float) -> float:
        return -0.5 * (math.pi / rc) * math.sin(math.pi * distance / rc)

    dcutoff_drij = _dfc(rij) * fck * fcjk
    dcutoff_drik = _dfc(rik) * fci * fcjk
    dcutoff_drjk = _dfc(rjk) * fci * fck
    dv_drij = scale * power * (dradial_drij * cutoff + radial * dcutoff_drij)
    dv_drik = scale * power * (dradial_drik * cutoff + radial * dcutoff_drik)
    dv_drjk = scale * power * (dradial_drjk * cutoff + radial * dcutoff_drjk)
    dv_dtheta = feature.zeta * angular * (scale * radial * cutoff) * feature.lam * (-math.sin(delta_theta))
    uij = vj / rij
    uik = vk / rik
    ujk = vjk / rjk
    dc_dRj = vk / (rij * rik) - cosine * uij / rij
    dc_dRk = vj / (rij * rik) - cosine * uik / rik
    dc_dRi = (-vj - vk) / (rij * rik) + cosine * (uij / rij + uik / rik)
    dtheta_dRi = dtheta_dc * dc_dRi
    dtheta_dRj = dtheta_dc * dc_dRj
    dtheta_dRk = dtheta_dc * dc_dRk
    grad_i = dv_drij * (-uij) + dv_drik * (-uik) + dv_dtheta * dtheta_dRi
    grad_j = dv_drij * uij + dv_drjk * (-ujk) + dv_dtheta * dtheta_dRj
    grad_k = dv_drik * uik + dv_drjk * ujk + dv_dtheta * dtheta_dRk
    return value, grad_i, grad_j, grad_k


def fingerprints_frame(
    positions: np.ndarray,
    symbols: list[str],
    lengths: np.ndarray,
    spec: FingerprintSpec,
    *,
    derivatives: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Modified Behler-Parrinello fingerprints for one orthogonal cell."""
    coords = np.asarray(positions, dtype=np.float64)
    n_atoms = coords.shape[0]
    if len(symbols) != n_atoms:
        raise ValueError("symbols do not match the positions")
    try:
        element_index = np.array([spec.elements.index(symbol) for symbol in symbols], dtype=np.int32)
    except ValueError as exc:
        raise ValueError("a symbol is outside fpParas.dat") from exc
    box = np.asarray(lengths, dtype=np.float64)
    delta = coords[:, None, :] - coords[None, :, :]
    delta -= box * np.round(delta / box)
    dist = np.linalg.norm(delta, axis=-1)
    values = np.zeros((n_atoms, spec.stride), dtype=np.float64)
    dg = np.zeros((n_atoms, n_atoms, 3, spec.stride), dtype=np.float64) if derivatives else None
    swapped = np.swapaxes(delta, 0, 1)
    unit = np.zeros_like(delta)
    positive = dist > 0.0
    unit[positive] = swapped[positive] / dist[positive, None]
    for feature in spec.g1:
        rc = feature.rc
        radial = dist - feature.rs
        exp_term = np.exp(-feature.eta * (radial * radial) / (rc * rc))
        fc = 0.5 * (np.cos(np.pi * dist / rc) + 1.0)
        raw = exp_term * fc
        mask = (dist < rc) & positive
        np.fill_diagonal(mask, False)
        neighbor = element_index == feature.neighbor
        mask &= neighbor[None, :]
        raw = np.where(mask, raw, 0.0)
        center = element_index == feature.center
        values[center, feature.index] = raw.sum(axis=1)[center]
        if dg is not None:
            slope = exp_term * (-2.0 * feature.eta * radial / (rc * rc)) * fc
            slope = slope + exp_term * (-0.5 * (math.pi / rc) * np.sin(np.pi * dist / rc))
            slope = np.where(mask, slope, 0.0)
            for atom in np.flatnonzero(center):
                dg[atom, atom, :, feature.index] += np.sum(-slope[atom, :, None] * unit[atom], axis=0)
                dg[atom, :, :, feature.index] += slope[atom, :, None] * unit[atom]
    groups: dict[tuple[int, int, int], list[G2]] = {}
    for feature in spec.g2:
        groups.setdefault((feature.center, feature.neighbor_a, feature.neighbor_b), []).append(feature)
    if groups:
        neighbors = []
        for atom in range(n_atoms):
            hits = np.flatnonzero((dist[atom] < spec.rc_max) & (np.arange(n_atoms) != atom))
            neighbors.append(hits)
        for center, js in enumerate(neighbors):
            element = int(element_index[center])
            count = len(js)
            for left in range(count):
                j = int(js[left])
                for right in range(left + 1, count):
                    k = int(js[right])
                    rjk = float(dist[j, k])
                    if rjk >= spec.rc_max:
                        continue
                    ej = int(element_index[j])
                    ek = int(element_index[k])
                    if ej > ek:
                        ej, ek = ek, ej
                    bucket = groups.get((element, ej, ek))
                    if not bucket:
                        continue
                    vj = delta[j, center]
                    vk = delta[k, center]
                    rij = float(dist[center, j])
                    rik = float(dist[center, k])
                    vjk = delta[k, j]
                    for feature in bucket:
                        if dg is None:
                            values[center, feature.index] += g2_value(rij, rik, rjk, vj, vk, feature)
                            continue
                        value, grad_i, grad_j, grad_k = g2_cartesian_grad(rij, rik, rjk, vj, vk, vjk, feature)
                        values[center, feature.index] += value
                        dg[center, center, :, feature.index] += grad_i
                        dg[center, j, :, feature.index] += grad_j
                        dg[center, k, :, feature.index] += grad_k
    if dg is not None:
        return values, dg
    return values


def init_weights(n_fp: tuple[int, ...] | list[int], rng: np.random.Generator) -> np.ndarray:
    """Uniform ``[-1, 1)`` in parameter order: weight, bias, weight, bias."""
    return rng.uniform(-1.0, 1.0, size=n_weights(n_fp))


def _atom_energy_grad(x: np.ndarray, block: np.ndarray, n_fp: int) -> tuple[float, np.ndarray]:
    energy, grad, _dedg = _atom_forward(x, block, n_fp)
    return energy, grad


def image_jacobian(
    weights: np.ndarray,
    element_index: np.ndarray,
    n_fp: tuple[int, ...],
    frame: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Total energy and ``dE/dw`` for one image."""
    jacobian = np.zeros_like(weights)
    energy = 0.0
    offsets = weight_offsets(n_fp)
    for atom, element in enumerate(element_index):
        offset, width, count = offsets[int(element)]
        y, grad = _atom_energy_grad(frame[atom, :count], weights[offset : offset + width], count)
        energy += y
        jacobian[offset : offset + width] += grad
    return energy, jacobian


def loss_and_gradient(
    weights: np.ndarray,
    element_index: np.ndarray,
    n_fp: tuple[int, ...],
    fingerprints: np.ndarray,
    energy_ref: np.ndarray,
    natoms: int,
) -> tuple[float, float, np.ndarray]:
    """Squared per-atom energy loss, its RMSE, and the parameter gradient."""
    grad = np.zeros_like(weights)
    loss = 0.0
    n_images = fingerprints.shape[0]
    for image in range(n_images):
        pred, jacobian = image_jacobian(weights, element_index, n_fp, fingerprints[image])
        per_atom = (pred - float(energy_ref[image])) / natoms
        loss += per_atom * per_atom
        grad += (2.0 * per_atom / natoms) * jacobian
    rmse = math.sqrt(loss / n_images)
    return loss, rmse, grad


def atom_input_grad(x: np.ndarray, block: np.ndarray, n_fp: int) -> np.ndarray:
    """``dE/dG`` for one atom. This is the network input adjoint."""
    _energy, _grad, dedg = _atom_forward(x, block, n_fp)
    return dedg


def _atom_forward(x: np.ndarray, block: np.ndarray, n_fp: int) -> tuple[float, np.ndarray, np.ndarray]:
    o_b1 = H1 * n_fp
    o_w2 = o_b1 + H1
    o_b2 = o_w2 + H2 * H1
    o_w3 = o_b2 + H2
    o_b3 = o_w3 + H2
    hidden = np.tanh(block[:o_b1].reshape(H1, n_fp) @ x + block[o_b1:o_w2])
    second = np.tanh(block[o_w2:o_b2].reshape(H2, H1) @ hidden + block[o_b2:o_w3])
    w3 = block[o_w3:o_b3]
    energy = float(w3 @ second + block[o_b3])
    grad = np.zeros_like(block)
    grad[o_w3:o_b3] = second
    grad[o_b3] = 1.0
    dz2 = w3 * (1.0 - second * second)
    grad[o_b2:o_w3] = dz2
    grad[o_w2:o_b2] = np.outer(dz2, hidden).ravel()
    dz1 = (block[o_w2:o_b2].reshape(H2, H1).T @ dz2) * (1.0 - hidden * hidden)
    grad[o_b1:o_w2] = dz1
    grad[:o_b1] = np.outer(dz1, x).ravel()
    dedg = block[:o_b1].reshape(H1, n_fp).T @ dz1
    return energy, grad, dedg


def atom_directional_grad(x: np.ndarray, direction: np.ndarray, block: np.ndarray, n_fp: int) -> np.ndarray:
    """Weight gradient of ``dE/dα`` at ``G = x + α direction``."""
    o_b1 = H1 * n_fp
    o_w2 = o_b1 + H1
    o_b2 = o_w2 + H2 * H1
    o_w3 = o_b2 + H2
    o_b3 = o_w3 + H2
    w1 = block[:o_b1].reshape(H1, n_fp)
    w2 = block[o_w2:o_b2].reshape(H2, H1)
    w3 = block[o_w3:o_b3]
    hidden = np.tanh(w1 @ x + block[o_b1:o_w2])
    second = np.tanh(w2 @ hidden + block[o_b2:o_w3])
    sech1 = 1.0 - hidden * hidden
    sech2 = 1.0 - second * second
    dz1 = w1 @ direction
    dh1 = sech1 * dz1
    dz2 = w2 @ dh1
    dh2 = sech2 * dz2
    g_sech2 = w3 * dz2
    g_dz2 = w3 * sech2
    g_h2 = g_sech2 * (-2.0 * second)
    g_z2 = g_h2 * sech2
    g_dh1 = w2.T @ g_dz2
    g_sech1 = g_dh1 * dz1
    g_dz1 = g_dh1 * sech1
    g_h1 = g_sech1 * (-2.0 * hidden) + w2.T @ g_z2
    g_z1 = g_h1 * sech1
    grad = np.zeros_like(block)
    grad[:o_b1] = (np.outer(g_dz1, direction) + np.outer(g_z1, x)).ravel()
    grad[o_b1:o_w2] = g_z1
    grad[o_w2:o_b2] = (np.outer(g_dz2, dh1) + np.outer(g_z2, hidden)).ravel()
    grad[o_b2:o_w3] = g_z2
    grad[o_w3:o_b3] = dh2
    return grad


def _image_force_state(
    weights: np.ndarray,
    element_index: np.ndarray,
    n_fp: tuple[int, ...],
    frame: np.ndarray,
    dg_image: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Energy, ``dE/dw``, Cartesian forces, and each atom's ``dE/dG``."""
    jacobian = np.zeros_like(weights)
    energy = 0.0
    dedg: list[np.ndarray] = []
    offsets = weight_offsets(n_fp)
    for atom, element in enumerate(element_index):
        offset, width, count = offsets[int(element)]
        y, grad, finger = _atom_forward(frame[atom, :count], weights[offset : offset + width], count)
        energy += y
        jacobian[offset : offset + width] += grad
        dedg.append(finger)
    forces = np.zeros((element_index.shape[0], 3), dtype=np.float64)
    for atom, finger in enumerate(dedg):
        count = finger.shape[0]
        forces += -np.einsum("acp,p->ac", dg_image[atom, :, :, :count], finger)
    return energy, jacobian, forces, dedg


def force_component_jacobian(
    weights: np.ndarray,
    element_index: np.ndarray,
    n_fp: tuple[int, ...],
    frame: np.ndarray,
    dg_image: np.ndarray,
    atom: int,
    component: int,
) -> tuple[float, np.ndarray]:
    """One force component and ``dF/dw`` for that component."""
    _energy, _jacobian, forces, _dedg = _image_force_state(weights, element_index, n_fp, frame, dg_image)
    jacobian = np.zeros_like(weights)
    offsets = weight_offsets(n_fp)
    for center, element in enumerate(element_index):
        offset, width, count = offsets[int(element)]
        direction = dg_image[center, atom, component, :count]
        jacobian[offset : offset + width] += atom_directional_grad(
            frame[center, :count],
            direction,
            weights[offset : offset + width],
            count,
        )
    return float(forces[atom, component]), -jacobian


def combined_loss_and_gradient(
    weights: np.ndarray,
    element_index: np.ndarray,
    n_fp: tuple[int, ...],
    fingerprints: np.ndarray,
    energy_ref: np.ndarray,
    forces_ref: np.ndarray,
    dg: np.ndarray,
    natoms: int,
    *,
    energy_coeff: float = ENERGY_COEFFICIENT,
    force_coeff: float = FORCE_COEFFICIENT,
) -> tuple[float, float, float, np.ndarray]:
    """Pyamff SE loss, energy RMSE, force RMSE, and the parameter gradient.

    ``forceloss`` is ``sum_images(sum_atoms ||F||^2 / natoms) / 3``. The RMSE
    values divide those unweighted sums by the image count. The coefficients
    enter the scalar loss and its gradient.
    """
    grad = np.zeros_like(weights)
    energyloss = 0.0
    forceloss = 0.0
    n_images = fingerprints.shape[0]
    offsets = weight_offsets(n_fp)
    for image in range(n_images):
        energy, jacobian, forces, dedg = _image_force_state(
            weights,
            element_index,
            n_fp,
            fingerprints[image],
            dg[image],
        )
        per_atom = (energy - float(energy_ref[image])) / natoms
        energyloss += per_atom * per_atom
        diff = forces - forces_ref[image]
        forceloss += float(np.sum(diff * diff) / natoms / 3.0)
        grad += (energy_coeff * 2.0 * per_atom / natoms) * jacobian
        lam = force_coeff * 2.0 * diff / natoms / 3.0
        for center, element in enumerate(element_index):
            offset, _width, count = offsets[int(element)]
            direction = -np.einsum("ac,acp->p", lam, dg[image, center, :, :, :count])
            grad[offset : offset + element_width(count)] += atom_directional_grad(
                fingerprints[image, center, :count],
                direction,
                weights[offset : offset + element_width(count)],
                count,
            )
    loss = energy_coeff * energyloss + force_coeff * forceloss
    energy_rmse = math.sqrt(energyloss / n_images)
    force_rmse = math.sqrt(forceloss / n_images)
    return loss, energy_rmse, force_rmse, grad


def projected_force_updates(n_images: int, n_atoms: int, ratio: float = FORCE_ENERGY_RATIO) -> int:
    """Force Kalman updates in one epoch at the documented force/energy ratio.

    This is ``Training::setupUpdatePlan`` with ``task_batch_size_force`` 1
    and one process: the force fraction is ``n_energy * ratio / n_force``,
    and the update count truncates toward zero.
    """
    patterns = float(n_images * n_atoms * 3)
    if patterns == 0.0:
        return 0
    epoch_fraction = (float(n_images) * float(ratio)) / patterns
    return int(patterns * epoch_fraction)


def reference_kalman_update(
    weights: np.ndarray,
    covariance: np.ndarray,
    jacobian: np.ndarray,
    residual: float,
    eta: float,
    q: float,
    *,
    etatau: float,
    etamax: float,
    qtau: float,
    qmin: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One ``KalmanFilter::update`` for a scalar energy observation.

    ``covariance`` is the full matrix. The multiply reads the lower triangle
    and mirrors it, which is what Eigen's ``selfadjointView<Lower>`` does.
    """
    lower = np.tril(covariance)
    symmetric = lower + np.tril(covariance, -1).T
    column = jacobian.reshape(-1, 1)
    gained = symmetric @ column
    scale = float((jacobian.reshape(1, -1) @ gained).item())
    if eta < etamax:
        eta *= math.exp(etatau)
    scale += 1.0 / eta
    kalman = gained / scale
    covariance = covariance - kalman @ gained.T
    diagonal = np.diag_indices(weights.size)
    covariance[diagonal] += q
    weights = weights + kalman.ravel() * residual
    if q > qmin:
        q *= math.exp(-qtau)
    return weights, covariance, eta, q


@dataclass
class MlDataset:
    elements: tuple[str, ...]
    n_fp: tuple[int, ...]
    element_index: np.ndarray
    fingerprints: np.ndarray
    energy: np.ndarray
    forces: np.ndarray | None = None
    dg: np.ndarray | None = None

    @property
    def n_images(self) -> int:
        return int(self.fingerprints.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.element_index.shape[0])


class KalmanSession:
    """One combined standard filter over a fixed fingerprint dataset."""

    def __init__(self, dataset: MlDataset, tol: float, *, loss: str = "energy"):
        self.dataset = dataset
        self.loss = loss
        self.lib = load_library()
        self.element_index = _as_int(dataset.element_index)
        self.n_fp = _as_int(np.array(dataset.n_fp, dtype=np.int32))
        self.fingerprints = _as_float(dataset.fingerprints)
        self.energy = _as_float(dataset.energy)
        self.forces = None
        self.dg = None
        self.n_force_updates = 0
        handle = self.lib.n2p2_create(
            dataset.n_images,
            dataset.n_atoms,
            len(dataset.n_fp),
            int(self.fingerprints.shape[-1]),
            dataset.n_atoms,
            float(tol),
            _ptr_int(self.element_index),
            _ptr_int(self.n_fp),
            _ptr_float(self.fingerprints),
            _ptr_float(self.energy),
            KALMAN_EPSILON,
            KALMAN_Q0,
            KALMAN_QTAU,
            KALMAN_QMIN,
            KALMAN_ETA0,
            KALMAN_ETATAU,
            KALMAN_ETAMAX,
        )
        if not handle:
            raise RuntimeError("n2p2 Kalman trainer was not created")
        self.handle = handle
        self.n_weights = int(self.lib.n2p2_n_weights(handle))
        self._grad = np.zeros(self.n_weights, dtype=np.float64)
        self._peaks = np.zeros(len(dataset.n_fp) * 6, dtype=np.float64)
        if loss == "force":
            if dataset.forces is None or dataset.dg is None:
                raise ValueError("force training needs reference forces and fingerprint derivatives")
            self.forces = _as_float(dataset.forces)
            self.dg = _as_float(dataset.dg)
            self.n_force_updates = projected_force_updates(dataset.n_images, dataset.n_atoms)
            code = self.lib.n2p2_enable_forces(
                handle,
                _ptr_float(self.forces),
                _ptr_float(self.dg),
                ENERGY_COEFFICIENT,
                FORCE_COEFFICIENT,
                FORCE_WEIGHT,
                int(self.n_force_updates),
                KALMAN_EPSILON,
                KALMAN_Q0,
                KALMAN_QTAU,
                KALMAN_QMIN,
                KALMAN_ETA0,
                KALMAN_ETATAU,
                KALMAN_ETAMAX,
            )
            if code != 0:
                raise RuntimeError("n2p2 force observations were not enabled")

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.n2p2_destroy(handle)
            self.handle = None

    def __del__(self) -> None:
        self.close()

    def set_weights(self, weights: np.ndarray) -> None:
        array = _as_float(weights)
        if self.lib.n2p2_set_weights(self.handle, _ptr_float(array), array.size) != 0:
            raise RuntimeError("weight buffer does not match the network")

    def get_weights(self) -> np.ndarray:
        array = np.zeros(self.n_weights, dtype=np.float64)
        if self.lib.n2p2_get_weights(self.handle, _ptr_float(array), array.size) != 0:
            raise RuntimeError("weight buffer does not match the network")
        return array

    def evaluate(self) -> dict:
        loss = ctypes.c_double()
        rmse = ctypes.c_double()
        finite = ctypes.c_int()
        converged = ctypes.c_int()
        code = self.lib.n2p2_eval(
            self.handle,
            _ptr_float(self._grad),
            self.n_weights,
            _ptr_float(self._peaks),
            self._peaks.size,
            ctypes.byref(loss),
            ctypes.byref(rmse),
            ctypes.byref(finite),
            ctypes.byref(converged),
        )
        if code != 0:
            raise RuntimeError("loss evaluation failed")
        return {
            "loss": float(loss.value),
            "energy_rmse": float(rmse.value),
            "force_rmse": float(self.lib.n2p2_force_rmse(self.handle)),
            "finite": bool(finite.value),
            "converged": bool(converged.value),
            "grad": self._grad.copy(),
            "tensor_maxabs": self._peaks.copy(),
        }

    def epoch(self, order: np.ndarray) -> int:
        array = _as_int(order)
        return int(self.lib.n2p2_epoch(self.handle, _ptr_int(array), array.size))

    def epoch_mixed(self, energy_order: np.ndarray, force_ids: np.ndarray, schedule: np.ndarray) -> int:
        energy = _as_int(energy_order)
        forces = _as_int(force_ids)
        steps = _as_int(schedule)
        return int(
            self.lib.n2p2_epoch_mixed(
                self.handle,
                _ptr_int(energy),
                energy.size,
                _ptr_int(forces),
                forces.size,
                _ptr_int(steps),
                steps.size,
            )
        )

    def updates(self) -> int:
        return int(self.lib.n2p2_updates(self.handle))


class N2p2Order:
    """Structure order from n2p2's ``shuffleUpdateCandidates``.

    ``Dataset::setupRandomNumberGenerator`` seeds a GSL mt19937 with
    ``random_seed``. With one process, ``Training`` then seeds
    ``std::mt19937_64 rngNew`` from the next ``gsl_rng_get`` on that
    generator, after drawing the global taus seed. Each epoch shuffles
    with that engine. The documented seed is 1234567.
    """

    def __init__(self, seed: int = N2P2_RANDOM_SEED):
        self.lib = load_library()
        self.handle = self.lib.n2p2_rng_create(ctypes.c_ulong(int(seed)))
        if not self.handle:
            raise RuntimeError("n2p2 shuffle generator was not created")

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.n2p2_rng_destroy(handle)
            self.handle = None

    def __del__(self) -> None:
        self.close()

    def permutation(self, count: int) -> np.ndarray:
        array = np.arange(count, dtype=np.int32)
        if self.lib.n2p2_rng_shuffle(self.handle, _ptr_int(array), array.size) != 0:
            raise RuntimeError("n2p2 shuffle failed")
        return array

    def shuffle_schedule(self, values: np.ndarray) -> np.ndarray:
        """Shuffle the mixed energy/force schedule with ``rngGlobalNew``."""
        array = np.array(values, dtype=np.int32, copy=True)
        if self.lib.n2p2_rng_shuffle_schedule(self.handle, _ptr_int(array), array.size) != 0:
            raise RuntimeError("n2p2 schedule shuffle failed")
        return array


def _epoch_order(dataset: MlDataset, rng: np.random.Generator | None, native: N2p2Order | None, shuffle: bool) -> np.ndarray:
    identity = np.arange(dataset.n_images, dtype=np.int32)
    if not shuffle:
        return identity
    if native is not None:
        return native.permutation(dataset.n_images)
    if rng is None:
        return identity
    return rng.permutation(dataset.n_images).astype(np.int32)


def _force_schedule(
    dataset: MlDataset,
    rng: np.random.Generator | None,
    shuffle: bool,
    native: N2p2Order | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy permutation, force-component sample, and the interleaved schedule.

    Energy visits every structure. The force sample has the documented
    update count. The schedule is the property list n2p2 builds by
    appending force updates, then energy updates, then shuffling.
    ``native`` follows Training: energy candidates, then force candidates,
    on ``rngNew``, and the property schedule on ``rngGlobalNew``.
    """
    n_images = dataset.n_images
    n_force = projected_force_updates(n_images, dataset.n_atoms)
    n_components = n_images * dataset.n_atoms * 3
    schedule = np.concatenate([np.ones(n_force, dtype=np.int32), np.zeros(n_images, dtype=np.int32)])
    if native is not None:
        energy_order = native.permutation(n_images)
        force_ids = native.permutation(n_components)[:n_force]
        schedule = native.shuffle_schedule(schedule)
    elif shuffle and rng is not None:
        energy_order = rng.permutation(n_images).astype(np.int32)
        force_ids = rng.permutation(n_components).astype(np.int32)[:n_force]
        schedule = schedule[rng.permutation(schedule.size)]
    else:
        energy_order = np.arange(n_images, dtype=np.int32)
        force_ids = np.arange(n_force, dtype=np.int32)
    return energy_order, force_ids, schedule


def train_one(
    dataset: MlDataset,
    weights: np.ndarray,
    *,
    cap: int,
    tol: float,
    rng: np.random.Generator | None,
    shuffle: bool,
    order: str = "pcg64",
    loss: str = "energy",
) -> dict:
    """Check the loss gradient, then take one Kalman epoch, up to ``cap``.

    ``rng`` supplies the harness weight draw. Dataset order is used when
    ``shuffle`` is false. ``order="n2p2"`` shuffles with n2p2's engines.
    Force loss interleaves the documented force-component updates with
    the energy updates.
    """
    session = KalmanSession(dataset, tol, loss=loss)
    native = N2p2Order() if order == "n2p2" and shuffle else None
    counter = HarnessCounter(cache=False, cap=int(cap))
    failure = None
    error = None
    success = False
    evaluation = None
    epochs = 0
    try:
        session.set_weights(weights)
        while True:
            counter.charge((float(counter.calls),))
            evaluation = session.evaluate()
            if not evaluation["finite"] or not math.isfinite(evaluation["loss"]):
                failure = "divergence"
                break
            if evaluation["converged"] and counter.calls < cap:
                success = True
                break
            if counter.calls >= cap:
                failure = "cap"
                break
            if loss == "force":
                energy_order, force_ids, schedule = _force_schedule(dataset, rng, shuffle, native)
                status = session.epoch_mixed(energy_order, force_ids, schedule)
            else:
                status = session.epoch(_epoch_order(dataset, rng, native, shuffle))
            epochs += 1
            if status != 0:
                failure = "divergence"
                break
    except ForceCallCap:
        failure = "cap"
    except Exception as exc:  # noqa: BLE001 - a raised optimizer is a crash record
        failure = "crash"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        updates = session.updates() if session.handle else 0
        session.close()
        if native is not None:
            native.close()
    if evaluation is None:
        evaluation = {
            "loss": None,
            "energy_rmse": None,
            "force_rmse": None,
            "tensor_maxabs": None,
            "converged": False,
        }
    return {
        "force_calls": int(counter.calls),
        "success": success,
        "failure": None if success else failure,
        "error": error,
        "converged": bool(success),
        "loss": evaluation["loss"],
        "energy_rmse": evaluation["energy_rmse"],
        "force_rmse": evaluation.get("force_rmse", FORCE_RMSE),
        "grad_maxabs": None if evaluation["tensor_maxabs"] is None else float(np.max(evaluation["tensor_maxabs"])),
        "kalman_epochs": epochs,
        "kalman_updates": updates,
        "verify_calls": int(counter.verify_calls),
        "order": order,
        "loss_kind": loss,
    }


def start_rng(index: int, base: int = SEED_BASE) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(base + int(index)))


def run_start(
    dataset: MlDataset,
    index: int,
    *,
    cap: int = EPOCH_CAP,
    tol: float = GRAD_TOL,
    shuffle: bool = True,
    order: str = "pcg64",
    loss: str = "energy",
) -> dict:
    """One initialization. Weights are drawn before the epoch permutations."""
    rng = start_rng(index)
    weights = init_weights(dataset.n_fp, rng)
    record = train_one(
        dataset,
        weights,
        cap=cap,
        tol=tol,
        rng=rng,
        shuffle=shuffle,
        order=order,
        loss=loss,
    )
    record["seed"] = SEED_BASE + int(index)
    record["start"] = f"{int(index):04d}"
    return record


def manifest_file(root: Path, archive_path: str) -> Path:
    import json

    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(f"{archive_path} is not in the geometry manifest")


def load_dataset(root: Path, problem_id: str) -> MlDataset:
    """Fingerprints and reference energies for one mlopt problem."""
    if problem_id not in TRAJECTORY_PATHS:
        raise KeyError(problem_id)
    spec = parse_fpparas((root / FPPARAS_PATHS[problem_id]).read_text(encoding="utf-8"))
    trajectory = manifest_file(root, TRAJECTORY_PATHS[problem_id])
    from ase.io import iread

    frames = list(iread(str(trajectory)))
    if not frames:
        raise ValueError(f"{trajectory} has no frames")
    symbols = list(frames[0].get_chemical_symbols())
    element_index = np.array([spec.elements.index(symbol) for symbol in symbols], dtype=np.int32)
    fingerprints = np.zeros((len(frames), len(symbols), spec.stride), dtype=np.float64)
    energy = np.zeros(len(frames), dtype=np.float64)
    for index, atoms in enumerate(frames):
        if list(atoms.get_chemical_symbols()) != symbols:
            raise ValueError(f"frame {index} changes the atom order")
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if np.max(np.abs(cell - np.diag(np.diag(cell)))) > 1e-8:
            raise ValueError(f"frame {index} is not orthogonal")
        lengths = np.diag(cell)
        fingerprints[index] = fingerprints_frame(atoms.get_positions(), symbols, lengths, spec)
        energy[index] = float(atoms.get_potential_energy())
    return MlDataset(
        elements=spec.elements,
        n_fp=spec.n_fp,
        element_index=element_index,
        fingerprints=fingerprints,
        energy=energy,
    )


def load_force_dataset(root: Path, problem_id: str) -> MlDataset:
    """Fingerprints, fingerprint derivatives, energies, and reference forces."""
    if problem_id not in TRAJECTORY_PATHS:
        raise KeyError(problem_id)
    spec = parse_fpparas((root / FPPARAS_FORCE_PATHS[problem_id]).read_text(encoding="utf-8"))
    trajectory = manifest_file(root, TRAJECTORY_PATHS[problem_id])
    from ase.io import iread

    frames = list(iread(str(trajectory)))
    if not frames:
        raise ValueError(f"{trajectory} has no frames")
    symbols = list(frames[0].get_chemical_symbols())
    element_index = np.array([spec.elements.index(symbol) for symbol in symbols], dtype=np.int32)
    n_images = len(frames)
    n_atoms = len(symbols)
    fingerprints = np.zeros((n_images, n_atoms, spec.stride), dtype=np.float64)
    dg = np.zeros((n_images, n_atoms, n_atoms, 3, spec.stride), dtype=np.float64)
    energy = np.zeros(n_images, dtype=np.float64)
    forces = np.zeros((n_images, n_atoms, 3), dtype=np.float64)
    for index, atoms in enumerate(frames):
        if list(atoms.get_chemical_symbols()) != symbols:
            raise ValueError(f"frame {index} changes the atom order")
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if np.max(np.abs(cell - np.diag(np.diag(cell)))) > 1e-8:
            raise ValueError(f"frame {index} is not orthogonal")
        lengths = np.diag(cell)
        fingerprints[index], dg[index] = fingerprints_frame(
            atoms.get_positions(),
            symbols,
            lengths,
            spec,
            derivatives=True,
        )
        energy[index] = float(atoms.get_potential_energy())
        forces[index] = np.asarray(atoms.get_forces(), dtype=np.float64)
    return MlDataset(
        elements=spec.elements,
        n_fp=spec.n_fp,
        element_index=element_index,
        fingerprints=fingerprints,
        energy=energy,
        forces=forces,
        dg=dg,
    )


def cache_dataset(root: Path, problem_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    dataset = load_dataset(root, problem_id)
    np.savez(
        dest,
        elements=np.array(dataset.elements),
        n_fp=np.array(dataset.n_fp, dtype=np.int32),
        element_index=dataset.element_index,
        fingerprints=dataset.fingerprints,
        energy=dataset.energy,
    )
    return dest


def cache_force_dataset(root: Path, problem_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    dataset = load_force_dataset(root, problem_id)
    np.savez(
        dest,
        elements=np.array(dataset.elements),
        n_fp=np.array(dataset.n_fp, dtype=np.int32),
        element_index=dataset.element_index,
        fingerprints=dataset.fingerprints,
        energy=dataset.energy,
        forces=dataset.forces,
        dg=dataset.dg,
    )
    return dest


def dataset_from_cache(path: Path) -> MlDataset:
    packed = np.load(path, allow_pickle=False)
    elements = tuple(str(item) for item in packed["elements"])
    forces = packed["forces"] if "forces" in packed.files else None
    dg = packed["dg"] if "dg" in packed.files else None
    return MlDataset(
        elements=elements,
        n_fp=tuple(int(item) for item in packed["n_fp"]),
        element_index=packed["element_index"],
        fingerprints=packed["fingerprints"],
        energy=packed["energy"],
        forces=forces,
        dg=dg,
    )
