#!/usr/bin/env python3
"""Run Gman r235 CI-NEB FIRE on the archived Pt-heptamer pairs.

The step is the r235 fire.step and neb.relax, including the Euler update
that uses the initial dt rather than the per-image dt. The Morse kernel is
codes/gman/morse.f, the r235 pot/morse/code/morse.f.
"""
from __future__ import annotations

import ctypes
import subprocess
import tarfile
import tempfile
from math import sqrt
from pathlib import Path

import numpy as np

U0 = 0.7102
R0 = 2.8970
ALPHA = 1.6047
RC = 9.5
FORCE_COUNT = 0
LIB: ctypes.CDLL | None = None


def vmag(v: np.ndarray) -> float:
    return sqrt(float((v * v).sum()))


def vunit(v: np.ndarray) -> np.ndarray:
    return v / sqrt(float((v * v).sum()))


def vdot(v1: np.ndarray, v2: np.ndarray) -> float:
    return float((v1 * v2).sum())


def vproj(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return v2 * ((v1 * v2).sum() / ((v2 * v2).sum()))


class Point:
    pass


def loadcon(path: Path) -> Point:
    p = Point()
    lines = path.read_text().splitlines()
    idx = 0
    p.header1 = lines[idx]; idx += 1
    p.header2 = lines[idx]; idx += 1
    tmp = lines[idx].split(); idx += 1
    dim = 0
    for i, token in enumerate(tmp):
        dim = i + 1
        try:
            float(token)
        except ValueError:
            dim = i
            break
    p.dim = dim
    box_length = np.array([float(tmp[i]) for i in range(dim)], dtype=np.float64)
    angles = np.array([float(x) for x in lines[idx].split()[:dim]], dtype=np.float64)
    idx += 1
    if np.any(angles != 90):
        raise SystemExit("non-orthogonal box")
    p.box = np.diag(box_length)
    idx += 2  # headers 3 and 4
    ntypes = int(lines[idx].split()[0]); idx += 1
    num_atom = np.array([int(x) for x in lines[idx].split()[:ntypes]], dtype=np.int64)
    idx += 1
    n_atoms = int(num_atom.sum())
    mass_type = np.array([float(x) for x in lines[idx].split()[:ntypes]], dtype=np.float64)
    idx += 1
    r = np.zeros((n_atoms, dim), dtype=np.float64)
    free = np.zeros((n_atoms, dim), dtype=np.float64)
    atom = 0
    for t in range(ntypes):
        idx += 2  # name and coordinate header
        for _ in range(int(num_atom[t])):
            line = np.array([float(x) for x in lines[idx].split()], dtype=np.float64)
            idx += 1
            r[atom] = line[:dim]
            flag = 0.0 if line[dim] == 1 else 1.0
            free[atom] = flag
            atom += 1
    p.r = r
    p.free = free
    p.n = n_atoms
    p.u = 0.0
    p.f = np.zeros_like(r)
    p.v = np.zeros_like(r)
    p.nvec = np.zeros_like(r)
    return p


def bind_library(path: Path) -> None:
    global LIB
    lib = ctypes.CDLL(str(path))
    lib.force_.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.force_.restype = None
    LIB = lib


def compile_morse(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["gfortran", "-O2", "-shared", "-fPIC", "-o", str(dest), str(source)],
    )
    return dest


def morse_force(p: Point, count: bool = True) -> None:
    global FORCE_COUNT
    if LIB is None:
        raise RuntimeError("Morse library is not loaded")
    n = p.n
    ncoo = n * 3
    ra = np.ascontiguousarray(p.r.reshape(ncoo), dtype=np.float64)
    fa = np.zeros(ncoo, dtype=np.float64)
    uret = np.zeros(1, dtype=np.float64)
    ax = float(p.box[0, 0])
    ay = float(p.box[1, 1])
    az = float(p.box[2, 2])
    n_c = ctypes.c_int(n)
    scalars = [ctypes.c_double(v) for v in (ax, ay, az, U0, ALPHA, R0, RC)]
    LIB.force_(
        ctypes.byref(n_c),
        ra.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        uret.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        *(ctypes.byref(s) for s in scalars),
    )
    p.f = fa.reshape(n, 3) * p.free
    p.u = float(uret[0])
    if count:
        FORCE_COUNT += 1


def count_reset() -> None:
    global FORCE_COUNT
    FORCE_COUNT = 0


class Fire:
    def init(self, maxmove=0.2, dt=0.1, dtmax=1.0, nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99):
        self.maxmove = maxmove
        self.dt = dt
        self.dtmax = dtmax
        self.nmin = nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa

    def step(self, path: "Band") -> None:
        try:
            path.ni
        except AttributeError:
            path.ni = 0
        if not path.ni:
            path.ni = 1
        try:
            path.start
        except AttributeError:
            path.start = 0
        if not path.start:
            path.start = 1
            for i in range(1, path.ni + 1):
                path.p[i].v = path.p[i].f.copy() * 0.0
                path.p[i].a = self.astart
                path.p[i].nsteps = 0
                path.p[i].dt = self.dt
        for i in range(1, path.ni + 1):
            power = vdot(path.p[i].f, path.p[i].v)
            if power > 0.0:
                path.p[i].v = (1.0 - path.p[i].a) * path.p[i].v + path.p[i].a * vmag(path.p[i].v) * vunit(path.p[i].f)
                if path.p[i].nsteps > self.nmin:
                    path.p[i].dt = min(path.p[i].dt * self.finc, self.dtmax)
                    path.p[i].a *= self.fa
                path.p[i].nsteps += 1
            else:
                path.p[i].v *= 0.0
                path.p[i].a = self.astart
                path.p[i].dt *= self.fdec
                path.p[i].nsteps = 0
            path.p[i].v += self.dt * path.p[i].f
            dr = self.dt * path.p[i].v
            if vmag(dr) > self.maxmove:
                dr = self.maxmove * vunit(dr)
            path.p[i].r += dr


class Band:
    def new(self, p1: Point, p2: Point, images: int = 8) -> None:
        self.ni = images
        self.p = [None] * (self.ni + 2)
        self.p[0] = p1
        self.p[self.ni + 1] = p2
        dr = (p2.r - p1.r) / (self.ni + 1.0)
        for i in range(1, self.ni + 1):
            image = Point()
            image.dim = p1.dim
            image.box = p1.box.copy()
            image.n = p1.n
            image.r = p1.r + dr * float(i)
            image.free = p1.free.copy()
            image.u = 0.0
            image.f = np.zeros_like(p1.r)
            image.v = np.zeros_like(p1.r)
            image.nvec = np.zeros_like(p1.r)
            self.p[i] = image
        self.init = False

    def tangentset(self) -> None:
        self.p[0].nvec = self.p[1].r - self.p[0].r
        self.p[self.ni + 1].nvec = self.p[self.ni + 1].r - self.p[self.ni].r
        for i in range(1, self.ni + 1):
            upm1 = self.p[i - 1].u > self.p[i].u
            upp1 = self.p[i + 1].u > self.p[i].u
            if upm1 != upp1:
                if upm1:
                    self.p[i].nvec = self.p[i].r - self.p[i - 1].r
                else:
                    self.p[i].nvec = self.p[i + 1].r - self.p[i].r
            else:
                um1 = self.p[i - 1].u - self.p[i].u
                up1 = self.p[i + 1].u - self.p[i].u
                umin = min(abs(up1), abs(um1))
                umax = max(abs(up1), abs(um1))
                if um1 > up1:
                    self.p[i].nvec = (self.p[i + 1].r - self.p[i].r) * umin
                    self.p[i].nvec += (self.p[i].r - self.p[i - 1].r) * umax
                else:
                    self.p[i].nvec = (self.p[i + 1].r - self.p[i].r) * umax
                    self.p[i].nvec += (self.p[i].r - self.p[i - 1].r) * umin
        for i in range(self.ni + 2):
            self.p[i].nvec = vunit(self.p[i].nvec)

    def project(self) -> None:
        for i in range(1, self.ni + 1):
            if i == self.umaxi:
                self.p[i].f = self.p[i].f - 2.0 * vproj(self.p[i].f, self.p[i].nvec)
            else:
                fperp = self.p[i].f - vproj(self.p[i].f, self.p[i].nvec)
                rm1 = self.p[i - 1].r - self.p[i].r
                rp1 = self.p[i + 1].r - self.p[i].r
                fsn = (vmag(rp1) - vmag(rm1)) * self.k * self.p[i].nvec
                self.p[i].f = fsn + fperp
                self.p[i].f *= self.p[i].free

    def relax(self, opt: Fire, itrmax=10000, fmax=0.01, k=5.0) -> int:
        self.k = k
        if not self.init:
            morse_force(self.p[0], False)
            morse_force(self.p[self.ni + 1], False)
            for i in range(self.ni + 2):
                self.p[i].v = self.p[i].r * 0.0
            self.init = True
        converged = False
        itr = 0
        for itr in range(itrmax):
            self.umax = self.p[0].u
            self.umaxi = 0
            for i in range(1, self.ni + 1):
                morse_force(self.p[i], True)
                if self.p[i].u > self.umax:
                    self.umax = self.p[i].u
                    self.umaxi = i
            self.tangentset()
            self.project()
            fmaxcur = vmag(self.p[1].f)
            for i in range(2, self.ni + 1):
                fmaxcur = max(fmaxcur, vmag(self.p[i].f))
            if fmaxcur < fmax:
                converged = True
                break
            opt.step(self)
        if not converged:
            raise SystemExit(f"did not converge in {itrmax}")
        return itr + 1


def check_kernel(minima: Path) -> None:
    from .pes import morse_pt

    point = loadcon(minima / "min01" / "a.con")
    morse_force(point, False)
    energy, grad = morse_pt(point.r, np.diag(point.box))
    if abs(point.u - float(energy)) > 1e-8:
        raise RuntimeError(f"Morse energy {point.u} disagrees with the harness {energy}")
    free = point.free[:, 0] > 0
    if float(np.max(np.abs(point.f[free] + grad[free]))) > 1e-8:
        raise RuntimeError("Morse forces disagree with the harness on the free atoms")


def paper_integers(per_image: list[float]) -> tuple[int, int, int]:
    """Python 2 ``%d`` truncation used by the archived opt.py."""
    return int(sum(per_image) / len(per_image)), int(max(per_image)), int(min(per_image))


def _minima_root(extracted: Path) -> Path:
    found = sorted(extracted.rglob("min01/a.con"))
    if not found:
        raise FileNotFoundError(f"{extracted} has no min01/a.con")
    return found[0].parent.parent


def prepare(root: Path, problem: dict, work: Path) -> Path:
    from .campaign import geometry_archive

    source = root / "codes" / "gman" / "morse.f"
    if not source.is_file():
        raise FileNotFoundError(source)
    compile_morse(source, work / "libmorse.so")
    bind_library(work / "libmorse.so")
    extracted = work / "minima"
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(geometry_archive(root, problem)) as packed:
        packed.extractall(extracted, filter="data")
    return _minima_root(extracted)


def run_pairs(minima: Path) -> list[dict]:
    opt = Fire()
    opt.init(maxmove=0.2, dt=0.15)
    rows = []
    for index in range(1, 14):
        name = f"min{index:02d}"
        count_reset()
        band = Band()
        band.new(loadcon(minima / name / "a.con"), loadcon(minima / name / "b.con"), images=8)
        band.relax(opt)
        per = FORCE_COUNT / 8.0
        rows.append(
            {
                "call_unit": "per_image",
                "counter": "harness",
                "count_key": "force_calls_neb",
                "force_calls": int(per) if per == int(per) else per,
                "geometry": name,
                "raw": FORCE_COUNT,
                "success": True,
            }
        )
        print(f"{name} count {FORCE_COUNT} per {per}", flush=True)
    return rows


def campaign_text(template: dict, mean: int, high: int, low: int) -> str:
    lines = [
        "# Paper script: int(sum(pot.count()/8)/13) with Python %d truncation.",
        f"force_calls {mean}",
        f"force_calls_max {high}",
        f"force_calls_min {low}",
        "nfailed 0",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"problem {template['problem']}",
        "counter harness",
        f"reducer {template.get('reducer', 'harness')}",
        f"commit {template['commit']}",
        "baseline_metric force_calls",
    ]
    return "\n".join(lines) + "\n"


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    work: Path | None = None,
) -> dict:
    """Compile r235 Morse, run the 13 bands, and write the paper integers."""
    scratch = work or Path(tempfile.mkdtemp(prefix="gman-neb-"))
    scratch.mkdir(parents=True, exist_ok=True)
    minima = prepare(root, problem, scratch)
    check_kernel(minima)
    rows = run_pairs(minima)
    per_image = [float(row["force_calls"]) for row in rows]
    mean, high, low = paper_integers(per_image)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(campaign_text(template, mean, high, low), encoding="utf-8")
    from .campaign import write_starts

    write_starts(output.with_suffix(".starts.jsonl"), rows)
    print("MEAN", sum(per_image) / len(per_image), "int", mean, "min", low, "max", high)
    return {
        "force_calls": mean,
        "force_calls_max": high,
        "force_calls_min": low,
        "nfailed": 0,
        "starts": len(rows),
        "template": template["id"],
        "problem": problem["id"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    from .campaign import load_template, resolve_problem

    template = load_template(root, "gman/neb-pt-fire")
    problem = resolve_problem(root, template["problem"])
    run_template(
        root,
        problem,
        template,
        output=root / "results" / "campaign" / "gman-r235-neb-pt-fire.dat",
    )


if __name__ == "__main__":
    main()
