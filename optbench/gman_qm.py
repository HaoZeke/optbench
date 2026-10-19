"""TSSE climbing-image NEB with QuickMin, the Gman r235 Pt-heptamer row.

The 2013 ``opt.py`` imports ``Gman`` and calls ``qm`` with ``dt=0.15`` and
``maxmove=0.2``. That package is the 2008 TSSE tree (``tsse-ugly``, version
1.0.0.01): ``neb.py``, ``opt/qm.py``, and the Fortran Morse kernel. Eight
images, spring 5, the improved tangent, climbing image, ``Fmag`` stop 0.01,
and ``fsaddle`` off. Endpoint evaluations are not counted. The published
integer is Python 2 ``%d`` truncation of the mean of ``pot.count()/8`` over
the 13 pairs.
"""

from __future__ import annotations

import copy
import multiprocessing
import tarfile
import tempfile
from math import sqrt
from pathlib import Path

import numpy as np

DE = 0.7102
BELTA = 1.6047
R0 = 2.8970
RC = 9.5
RCSQ = RC * RC
DUMMY = 2.0 * BELTA * DE
UU_C = np.exp(BELTA * (R0 - RC))
UU_SHIFT = UU_C * UU_C - 2.0 * UU_C
PAIRS = 13
IMAGES = 8
CALL_CAP = 10000
CAMPAIGN = Path("results/campaign/gman-r235-neb-pt-qm.dat")


def dnint(x: np.ndarray) -> np.ndarray:
    """Fortran DNINT: nearest integer, ties away from zero."""
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def vmag(v: np.ndarray) -> float:
    return float(sqrt(float((v * v).sum())))


def vdot(v1: np.ndarray, v2: np.ndarray) -> float:
    return float((v1 * v2).sum())


def vunit(v: np.ndarray) -> np.ndarray:
    mag = sqrt(float((v * v).sum()))
    if mag == 0:
        return v
    return v / mag


def vproj(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    mag2 = float((v2 * v2).sum())
    if mag2 == 0:
        return v1
    return v2 * (float((v1 * v2).sum()) / mag2)


class Point:
    def __init__(self) -> None:
        self.R = np.zeros((0, 3))
        self.FreeD = np.zeros((0, 3))
        self.Box = np.eye(3)
        self.U = 0.0
        self.F = np.zeros((0, 3))
        self.V = np.zeros((0, 3))
        self.N = np.zeros((0, 3))
        self.start = 0


def loadcon(path: Path) -> Point:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    point = Point()
    lengths = np.array([float(x) for x in lines[2].split()[:3]])
    point.Box = np.diag(lengths)
    n_types = int(lines[6].split()[0])
    counts = [int(x) for x in lines[7].split()[:n_types]]
    n_atoms = sum(counts)
    point.R = np.zeros((n_atoms, 3))
    point.FreeD = np.zeros((n_atoms, 3))
    row = 9
    atom = 0
    for _type in range(n_types):
        row += 2  # symbol and "Coordinates"
        for _ in range(counts[_type]):
            parts = lines[row].split()
            point.R[atom] = [float(parts[0]), float(parts[1]), float(parts[2])]
            frozen = float(parts[3]) == 1.0
            point.FreeD[atom] = 0.0 if frozen else 1.0
            atom += 1
            row += 1
    return point


def morse_force(point: Point, count: list[int], do_count: bool = True) -> None:
    """Fortran ``pot/morse/code/morse.f``, then ``F *= FreeD`` as in ``morse.py``."""
    positions = point.R
    n_atoms = positions.shape[0]
    box = np.array([point.Box[0, 0], point.Box[1, 1], point.Box[2, 2]])
    diff = positions[:, None, :] - positions[None, :, :]
    diff = diff - box * dnint(diff / box)
    distance2 = np.einsum("ijk,ijk->ij", diff, diff)
    upper = np.triu_indices(n_atoms, 1)
    pair2 = distance2[upper]
    mask = pair2 < RCSQ
    left = upper[0][mask]
    right = upper[1][mask]
    vec = diff[left, right]
    dist = np.sqrt(pair2[mask])
    uu = np.exp(BELTA * (R0 - dist))
    duu = uu * uu - uu
    force = np.zeros_like(positions)
    contrib = (duu / dist)[:, None] * vec
    np.add.at(force, left, contrib)
    np.add.at(force, right, -contrib)
    force *= DUMMY
    point.F = force * point.FreeD
    point.U = float(np.sum(duu - uu - UU_SHIFT) * DE)
    if do_count:
        count[0] += 1


class QM:
    def __init__(self, maxmove: float = 0.2, dt: float = 0.05) -> None:
        self.maxmove = maxmove
        self.dt = dt

    def step(self, path: NEB) -> None:
        for index in range(1, path.ni + 1):
            image = path.p[index]
            if not image.start:
                image.start = 1
                image.V = image.R * 0.0
            if vdot(image.F, image.V) > 0:
                image.V = vproj(image.V, image.F)
            else:
                image.V = image.V * 0.0
            image.V = image.V + self.dt * image.F
            step = self.dt * image.V
            if vmag(step) > self.maxmove:
                step = self.maxmove * vunit(step)
            image.R = image.R + step


class NEB:
    def __init__(self) -> None:
        self.ni = 0
        self.p: list[Point] = []
        self.init = False
        self.method = "off"
        self.k = 5.0
        self.umax = 0.0
        self.umaxi = 0
        self.fmaxcur = 0.0

    def new(self, start: Point, end: Point, images: int = IMAGES) -> None:
        self.ni = images
        self.p = [Point() for _ in range(self.ni + 2)]
        self.p[0] = start
        self.p[self.ni + 1] = end
        delta = (end.R - start.R) / (self.ni + 1.0)
        for index in range(1, self.ni + 1):
            self.p[index] = copy.deepcopy(start)
            self.p[index].R = self.p[index].R + delta * float(index)
        self.init = False

    def relax(self, count: list[int], opt: QM, itrmax: int = CALL_CAP, fmax: float = 0.01, k: float = 5.0) -> int:
        self.method = "ci"
        self.k = k
        if not self.init:
            morse_force(self.p[0], count, False)
            morse_force(self.p[self.ni + 1], count, False)
            for index in range(self.ni + 2):
                self.p[index].V = self.p[index].R * 0.0
            self.init = True
        for iteration in range(itrmax):
            self.umax = self.p[0].U
            self.umaxi = 0
            for index in range(1, self.ni + 1):
                morse_force(self.p[index], count, True)
                if self.p[index].U > self.umax:
                    self.umax = self.p[index].U
                    self.umaxi = index
            self.tangentset()
            self.project()
            self.fmaxcur = max(vmag(self.p[index].F) for index in range(1, self.ni + 1))
            if self.fmaxcur < fmax:
                return iteration + 1
            opt.step(self)
        return itrmax

    def tangentset(self) -> None:
        self.p[0].N = self.p[1].R - self.p[0].R
        self.p[self.ni + 1].N = self.p[self.ni + 1].R - self.p[self.ni].R
        for index in range(1, self.ni + 1):
            lower = self.p[index - 1].U > self.p[index].U
            higher = self.p[index + 1].U > self.p[index].U
            if lower != higher:
                if lower:
                    self.p[index].N = self.p[index].R - self.p[index - 1].R
                else:
                    self.p[index].N = self.p[index + 1].R - self.p[index].R
            else:
                drop = self.p[index - 1].U - self.p[index].U
                rise = self.p[index + 1].U - self.p[index].U
                small = min(abs(rise), abs(drop))
                large = max(abs(rise), abs(drop))
                if drop > rise:
                    tangent = (self.p[index + 1].R - self.p[index].R) * small
                    tangent = tangent + (self.p[index].R - self.p[index - 1].R) * large
                else:
                    tangent = (self.p[index + 1].R - self.p[index].R) * large
                    tangent = tangent + (self.p[index].R - self.p[index - 1].R) * small
                self.p[index].N = tangent
        for index in range(self.ni + 2):
            self.p[index].N = vunit(self.p[index].N)

    def project(self) -> None:
        for index in range(1, self.ni + 1):
            if self.method == "ci" and index == self.umaxi:
                self.p[index].F = self.p[index].F - 2.0 * vproj(self.p[index].F, self.p[index].N)
                continue
            perpendicular = self.p[index].F - vproj(self.p[index].F, self.p[index].N)
            backward = self.p[index - 1].R - self.p[index].R
            forward = self.p[index + 1].R - self.p[index].R
            spring = (vmag(forward) - vmag(backward)) * self.k * self.p[index].N
            self.p[index].F = (spring + perpendicular) * self.p[index].FreeD


def unpack_minima(root: Path, dest: Path) -> Path:
    from .campaign import geometry_archive, load_problem

    problem = load_problem(root / "problems" / "neb" / "Pt-heptamer.toml")
    archive = geometry_archive(root, problem)
    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(dest, filter="data")
    minima = dest / "minima"
    if not minima.is_dir():
        raise FileNotFoundError(f"{archive} has no minima directory")
    return minima


def run_pair(root: Path, index: int) -> float:
    reactant = loadcon(root / f"min{index:02d}" / "a.con")
    product = loadcon(root / f"min{index:02d}" / "b.con")
    band = NEB()
    band.new(reactant, product, images=IMAGES)
    count = [0]
    band.relax(count, QM(maxmove=0.2, dt=0.15), itrmax=CALL_CAP, fmax=0.01, k=5.0)
    return count[0] / float(IMAGES)


def collect(root: Path, workers: int | None = None) -> list[float]:
    """Per-image force calls for min01..min13, in pair order."""
    with tempfile.TemporaryDirectory(prefix="gman-qm-") as tmp:
        minima = unpack_minima(root, Path(tmp))
        jobs = [(minima, index) for index in range(1, PAIRS + 1)]
        processes = workers if workers is not None else (multiprocessing.cpu_count() or 1)
        if processes <= 1:
            return [run_pair(path, index) for path, index in jobs]
        with multiprocessing.Pool(processes) as pool:
            return pool.starmap(run_pair, jobs)


def campaign_text(calls: list[float]) -> str:
    """The 2013 ``opt.py`` integers. A pair at the cap is a failure and leaves the mean."""
    if len(calls) != PAIRS:
        raise ValueError(f"expected {PAIRS} pairs, found {len(calls)}")
    failed = [value for value in calls if value >= CALL_CAP]
    kept = [value for value in calls if value < CALL_CAP]
    if not kept:
        raise ValueError("every pair hit the force-call cap")
    mean = sum(kept) / len(kept)
    lines = [
        f"force_calls {int(mean)}",
        f"force_calls_min {int(min(calls))}",
        f"force_calls_max {int(max(calls))}",
        f"nfailed {len(failed)}",
        "code Gman",
        "code_version r235",
        "template gman-qm",
        f"unrounded_mean {mean!r}",
    ]
    return "\n".join(lines) + "\n"


def write_campaign(root: Path, calls: list[float] | None = None) -> Path:
    measured = collect(root) if calls is None else calls
    path = root / CAMPAIGN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(campaign_text(measured), encoding="utf-8")
    return path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = write_campaign(root)
    print(path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
