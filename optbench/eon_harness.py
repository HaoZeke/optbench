"""Harness runs for EON (SVN r2549), eOn develop, and the 3.5.0 eOn pin.

EON and eOn are different codes. EON is the r2549 client. eOn is
https://github.com/TheochemUI/eon at develop
7095b23ac17d49f434a5e08ccb072897ceca698e. The r2549 client is not rebuilt.

Each force evaluation is one ``HarnessCounter`` charge. eonclient is
started with ``potential = ext_pot``. A small C program forwards
``from_eon_to_extpot`` to ``PotServer``. The template still names the
2014 potential. The kernel is the problem surface:

* ``lj40`` is unshifted ``lj/cut`` with epsilon 1, sigma 1, cutoff 40.
* ``blj20`` is binary Lennard-Jones, cutoff 20, the archived
  ``lj100-binary`` pair coefficients.
* ``morse`` is LAMMPS ``morse/smooth/linear`` (D 0.7102, alpha 1.6047,
  r0 2.897, cutoff 9.5) with periodic boundaries in x and y.
* ``morse_shift`` is r2549 ``Morse::force``: the same depth, alpha, and
  r0, cutoff 9.5, pair energy shifted to zero at the cutoff, minimum
  image on the box diagonal.
* ``eam`` calls the r2549 Aluminum Fortran ``force_`` in this process.

``tip4p`` is the uncut rigid Cambridge TIP4P. A GMIN ``.aa`` start is
centre then angle-axis. EON (SVN r2549) and eOn
(develop 7095b23ac17d49f434a5e08ccb072897ceca698e) each receive the same
Cartesian ``.con``: ``optbench.tip4p.sites_from_aa`` rotated with the GMIN
map, written in the CCL order ``Tip4p::computeHH_O_`` reads (H, H pairs,
then oxygens). The kernel reports that energy to the client in eV using
the constants in ``tip4p_unit_system.hpp`` so the basin-hopping acceptance,
which divides by ``temperature * 8.6173324e-5``, sees electron volts. The
Cambridge test multiplies back to kJ/mol. eOn built with ``-Dwith_water=true``
routes ``potential = tip4p`` to ``rgpot::TIP4PPot`` (flexible CCL, cutoff
8.5 Å). That client potential is not the scored kernel. EON
``client/potentials/Water/tip4p_ccl.cpp`` is the same flexible restraint
potential. The ledger line on the campaign file records the conversion.

A tsopt success is a client stop under the cap, a free-atom Euclidean
force norm at the problem stop, and Hessian index 1. That Hessian is
computed from the kernel after the client returns and is not charged
to ``HarnessCounter``. A free cluster drops rigid modes first. A
geometry with a fixed atom uses the free-atom Hessian as it stands.

SVN r2549 does not read ``Main.parallel``. The 3.5.0 client does. The
ini written for a C++ job sets it false. The Python AKMC server rejects
the key, so those inis leave it unset. One server still receives every
``force()`` the clients open.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .asv_tracks import TrackedSearch
from .campaign import (
    benchmark_text,
    format_ini,
    geometry_archive,
    list_templates,
    load_template,
    resolve_problem,
    write_starts,
)
from .eon_jobs import _client_env, _numpy_direction, _read_dat, _seeded_mode
from .eon_parse import abs_overlap, flat_floats, xyz_to_con
from .globalopt import energy_tolerance, judge_globalopt, reference_energy
from .stationary import negative_index, vibrational_eigenvalues
from .surfaces import lennard_jones, morse_smooth_linear, read_con

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv"))
KERNEL = "problem-surface-2"
SVN_COMMIT = "r2549"
CLIENT_COMMIT = "4d8b34baf1ba8d2d980df29838bac1f969d1662c"
_CLIENT_OK: set[str] = set()
_EAM = None

FAMILIES = {
    "svn": {
        "prefix": "eon-svn/",
        "env": "EON_SVN_CLIENT",
        "default": Path.home() / "opt" / "eon-svn" / "bin" / "eonclient",
        "flag": "-v",
        "token": "r2549",
        "cache": "r2549",
    },
    "client": {
        "prefix": "eon-client/",
        "env": "EONCLIENT_350",
        "default": Path.home() / "opt" / "eon-350" / "bin" / "eonclient",
        "flag": "--version",
        "token": "4d8b34ba",
        "cache": "4d8b34ba",
    },
    "develop": {
        "prefix": "eon-develop/",
        "env": "EONCLIENT",
        "default": Path.home() / "opt" / "eon-develop" / "bin" / "eonclient",
        "flag": "--version",
        "token": "7095b23",
        "cache": "7095b23",
    },
}


class BudgetExceeded(Exception):
    """The template budget fired before the client returned."""


def full_grid() -> bool:
    return os.environ.get("OPTBENCH_ASV_FULL") == "1"


def selected(values: list) -> list:
    if full_grid():
        return list(values)
    return list(values[:1])


def _on_alarm(signum, frame):
    del signum, frame
    raise BudgetExceeded()


def bridge_binary() -> Path:
    """Compile the C ext_pot forwarder once per source hash."""
    source = Path(__file__).with_name("extpot_bridge.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest = Path("/tmp/optbench-extpot-bridge") / digest / "extpot-bridge"
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".tmp")
    subprocess.run(
        ["cc", "-O2", "-o", str(temporary), str(source)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    temporary.replace(dest)
    return dest


def _cell(box) -> np.ndarray:
    if box is None or len(box) < 9:
        return np.zeros(3)
    return np.array([float(box[0]), float(box[4]), float(box[8])], dtype=np.float64)


def lj40(positions, atomic_numbers=None, box=None):
    """Unshifted Lennard-Jones cutoff 40. Forces are the negative gradient."""
    del atomic_numbers
    coords = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    energy, gradient = lennard_jones(
        coords,
        epsilon=1.0,
        sigma=1.0,
        cutoff=40.0,
        cell=_cell(box),
        pbc=(False, False, False),
    )
    return float(energy), tuple(float(value) for value in np.ravel(-gradient))


def morse_pt(positions, atomic_numbers=None, box=None):
    """Smooth-linear Morse, periodic in x and y."""
    del atomic_numbers
    coords = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    energy, gradient = morse_smooth_linear(
        coords,
        depth=0.7102,
        alpha=1.6047,
        r0=2.897,
        cutoff=9.5,
        cell=_cell(box),
        pbc=(True, True, False),
    )
    return float(energy), tuple(float(value) for value in np.ravel(-gradient))


def morse_shift(positions, atomic_numbers=None, box=None):
    """r2549 Morse::force. Energy is zero at the cutoff."""
    from benchmarks.bench_eon_svn import _morse_fast

    flat = np.asarray(positions, dtype=np.float64).ravel()
    return _morse_fast(flat, atomic_numbers, box)


def _link_eam(objects: list[Path], dest: Path) -> subprocess.CompletedProcess[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["gfortran", "-shared", "-o", str(dest), *[str(path) for path in objects]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def ensure_eam() -> Path:
    """Shared library of the r2549 Aluminum Fortran objects.

    The exported objects are not position-independent. A failed link
    recompiles the Fortran sources with ``-fPIC``.
    """
    dest = Path.home() / "opt" / "eon-svn" / "lib" / "libeam_al.so"
    if dest.is_file():
        return dest
    folder = Path("/tmp/eon-svn-r2549/client/potentials/Aluminum")
    objects = sorted(path for path in folder.glob("*.o") if path.name != "Aluminum.o")
    if not objects:
        raise RuntimeError("r2549 Aluminum objects are not built")
    proc = _link_eam(objects, dest)
    if proc.returncode == 0:
        return dest
    sources = sorted(folder.glob("*.f"))
    if not sources:
        raise RuntimeError(proc.stderr.strip() or "gfortran could not link libeam_al.so")
    pic = Path("/tmp/optbench-eam-pic")
    pic.mkdir(parents=True, exist_ok=True)
    compiled = []
    for source in sources:
        obj = pic / f"{source.stem}.o"
        build = subprocess.run(
            ["gfortran", "-fPIC", "-O2", "-c", str(source), "-o", str(obj)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if build.returncode != 0:
            raise RuntimeError(build.stderr.strip() or f"gfortran could not compile {source.name}")
        compiled.append(obj)
    linked = _link_eam(compiled, dest)
    if linked.returncode != 0:
        raise RuntimeError(linked.stderr.strip() or "gfortran could not link libeam_al.so")
    return dest


class _EamAl:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.potinit_.argtypes = []
        self.lib.force_.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.potinit_()

    def __call__(self, positions, atomic_numbers=None, box=None):
        del atomic_numbers
        coords = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(-1, 3))
        forces = np.zeros_like(coords)
        count = ctypes.c_int64(len(coords))
        energy = ctypes.c_double()
        lengths = _cell(box)
        if np.any(lengths <= 0.0):
            lengths = np.array([100.0, 100.0, 100.0])
        axes = [ctypes.c_double(float(value)) for value in lengths]
        self.lib.force_(
            ctypes.byref(count),
            coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(energy),
            ctypes.byref(axes[0]),
            ctypes.byref(axes[1]),
            ctypes.byref(axes[2]),
        )
        return float(energy.value), tuple(float(value) for value in forces.ravel())


def eam_al(positions, atomic_numbers=None, box=None):
    global _EAM
    if _EAM is None:
        _EAM = _EamAl(ensure_eam())
    return _EAM(positions, atomic_numbers, box)


def saddle_index(kernel, text: str, *, periodic: bool) -> dict:
    """Free-atom force norm and Hessian index. Verification is not a search call.

    A free cluster drops three translations and three rotations. A periodic
    cell with no fixed atom drops translations only. A fixed substrate is
    left as the free-atom Hessian: the anchor already removes the rigid modes.
    A curvature below ``-1e-6`` counts as negative.
    """
    positions, fixed, cell = read_con(text)
    box = np.zeros(9, dtype=np.float64)
    box[0], box[4], box[8] = (float(cell[0]), float(cell[1]), float(cell[2]))
    free = ~np.asarray(fixed, dtype=bool)
    if not np.any(free):
        raise ValueError("every atom is fixed")

    def forces_of(coords: np.ndarray) -> np.ndarray:
        _energy, flat = kernel(coords, None, box)
        return np.asarray(flat, dtype=np.float64).reshape(-1, 3)[free]

    force_norm = float(np.linalg.norm(forces_of(positions)))
    free_index = np.flatnonzero(free)
    dimension = 3 * len(free_index)
    step = 1e-5
    hessian = np.zeros((dimension, dimension), dtype=np.float64)
    for local, atom in enumerate(free_index):
        for axis in range(3):
            column = 3 * local + axis
            for sign in (1.0, -1.0):
                trial = np.array(positions, copy=True)
                trial[atom, axis] += sign * step
                hessian[:, column] += -sign * forces_of(trial).ravel() / (2.0 * step)
    hessian = 0.5 * (hessian + hessian.T)
    free_positions = positions[free]
    if bool(np.all(free)):
        values = vibrational_eigenvalues(hessian, free_positions, rotations=not periodic)
    else:
        values = [float(value) for value in np.linalg.eigvalsh(hessian)]
    return {
        "force_norm": force_norm,
        "hessian_index": negative_index(values, 1e-6),
    }


def blj20(positions, atomic_numbers=None, box=None):
    """Binary Lennard-Jones, cutoff 20. Forces are the negative gradient.

    The archived ``in.lammps`` for ``globalopt/lj100-binary`` sets
    ``pair_style lj/cut 20`` and ``pair_coeff`` sigma 1.3 on type 1,
    sigma 1.0 on type 2, epsilon 1, and the unlike sigma 1.15. r2549
    ``LAMMPS.cpp`` numbers types in order of first appearance. The con
    files list H and then C, so H is type 1 (B, sigma 1.3) and C is
    type 2 (A, sigma 1.0).
    """
    from .potentials import binary_lennard_jones

    del box
    coords = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    numbers = [int(value) for value in (atomic_numbers or [])]
    if len(numbers) != len(coords):
        raise ValueError("binary Lennard-Jones needs one atomic number per atom")
    species = []
    for number in numbers:
        if number == 1:
            species.append(1)
        elif number == 6:
            species.append(0)
        else:
            raise ValueError(f"binary Lennard-Jones atomic number {number}")
    energy, gradient = binary_lennard_jones(
        coords,
        species,
        sigma_aa=1.0,
        sigma_bb=1.3,
        epsilon_aa=1.0,
        epsilon_bb=1.0,
        epsilon_ab=1.0,
        cutoff=20.0,
    )
    return float(energy), tuple(float(value) for value in np.ravel(-gradient))


def coords_to_con(text: str) -> str:
    """Bare ``x y z`` rows to a free-cluster ``.con``.

    The cube is at least 100 and at least 80 past the span, the same
    rule as ``xyz_to_con``, so a periodic image stays outside cutoff 40.
    """
    coords: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not coords:
        raise ValueError("coordinate file has no xyz rows")
    natoms = len(coords)
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    zs = [point[2] for point in coords]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    side = max(100.0, span + 80.0)
    shift = (
        side / 2.0 - sum(xs) / natoms,
        side / 2.0 - sum(ys) / natoms,
        side / 2.0 - sum(zs) / natoms,
    )
    rows = [
        "Generated by optbench",
        "0.0000 TIME",
        f"{side:.6f}  {side:.6f}  {side:.6f}",
        "90.000000   90.000000   90.000000",
        "0 0",
        "0 0 0",
        "1",
        str(natoms),
        "1.00000",
        "H",
        "Coordinates of Component 1",
    ]
    for index, (x, y, z) in enumerate(coords):
        rows.append(
            f"  {x + shift[0]:.8f}   {y + shift[1]:.8f}   {z + shift[2]:.8f}  0     {index}"
        )
    return "\n".join(rows) + "\n"


def tip4p_kernel(positions, numbers, box):
    """Rigid TIP4P for ``ext_pot``, in eV and eV/Å.

    The Cambridge value is kJ/mol. ``KJ_PER_EV`` is
    ``e_charge * N_A / 1000`` from the eOn electron-volt unit system.
    """
    del box
    from .tip4p import KJ_PER_EV, rigid_cartesian

    energy, forces = rigid_cartesian(positions, numbers)
    scale = 1.0 / KJ_PER_EV
    return float(energy) * scale, tuple(float(value) * scale for value in forces)


def aa_to_con(text: str) -> str:
    """GMIN angle-axis text to a free-cluster ``.con`` in CCL order.

    The nine-line header is the one ``Matter::con2matter`` and the develop
    readcon grammar both read: two free lines, the box, the angles, two free
    lines, the component count, one line of counts, one line of masses.
    Hydrogens are component 1 and oxygens are component 2, which is the
    order ``Tip4p::computeHH_O_`` uses. The cube is at least 100 and at
    least 80 past the span, the same rule as ``coords_to_con``.
    """
    from .tip4p import TIP4P_MASSES, ccl_from_sites, parse_aa, sites_from_aa

    positions, numbers = ccl_from_sites(sites_from_aa(parse_aa(text)))
    coords = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    molecules = coords.shape[0] // 3
    if not np.array_equal(numbers[: 2 * molecules], np.ones(2 * molecules, dtype=np.int64)):
        raise ValueError("CCL order puts the hydrogens first")
    if not np.array_equal(numbers[2 * molecules :], np.full(molecules, 8, dtype=np.int64)):
        raise ValueError("CCL order puts the oxygens last")
    span = float(np.max(np.max(coords, axis=0) - np.min(coords, axis=0))) if coords.size else 0.0
    side = max(100.0, span + 80.0)
    shift = side / 2.0 - coords.mean(axis=0)
    hydrogen_mass = float(TIP4P_MASSES[1])
    oxygen_mass = float(TIP4P_MASSES[0])
    rows = [
        "Generated by optbench",
        "0.0000 TIME",
        f"{side:.6f}  {side:.6f}  {side:.6f}",
        "90.000000   90.000000   90.000000",
        "0 0",
        "0 0 0",
        "2",
        f"{2 * molecules} {molecules}",
        f"{hydrogen_mass:.6f} {oxygen_mass:.6f}",
        "H",
        "Coordinates of Component 1",
    ]
    for index in range(2 * molecules):
        point = coords[index] + shift
        rows.append(f"  {point[0]:.8f}   {point[1]:.8f}   {point[2]:.8f}  0     {index}")
    rows.append("O")
    rows.append("Coordinates of Component 2")
    for index in range(2 * molecules, 3 * molecules):
        point = coords[index] + shift
        rows.append(f"  {point[0]:.8f}   {point[1]:.8f}   {point[2]:.8f}  0     {index}")
    return "\n".join(rows) + "\n"


def kernel_for(name: str):
    if name == "lj40":
        return lj40
    if name == "blj20":
        return blj20
    if name == "morse":
        return morse_pt
    if name == "morse_shift":
        return morse_shift
    if name == "eam":
        return eam_al
    if name == "tip4p":
        return tip4p_kernel
    raise ValueError(f"no harness kernel {name}")


def classify_client(*, calls, reason, cap, overlap_ok=None, crashed=False) -> dict:
    if calls is None:
        return {"force_calls": 0.0, "success": False, "failure": "crash"}
    count = int(calls)
    if overlap_ok is True and count < cap:
        return {"force_calls": float(count), "success": True, "failure": None}
    if overlap_ok is False and count < cap:
        return {"force_calls": float(count), "success": False, "failure": "wrong_stationary"}
    if count >= cap:
        return {"force_calls": float(count), "success": False, "failure": "cap"}
    if crashed or reason in (None, ""):
        return {"force_calls": float(count), "success": False, "failure": "crash"}
    if str(reason) != "0":
        return {"force_calls": float(count), "success": False, "failure": "divergence"}
    return {"force_calls": float(count), "success": True, "failure": None}


def classify_flag(*, calls, cap, ok, crashed=False) -> dict:
    """A finished job with no termination reason. ``ok`` is the success bit."""
    if calls is None:
        return {"force_calls": 0.0, "success": False, "failure": "crash"}
    count = int(calls)
    if count >= cap or (not ok and not crashed and count >= cap):
        return {"force_calls": float(count), "success": False, "failure": "cap"}
    if count >= cap:
        return {"force_calls": float(count), "success": False, "failure": "cap"}
    if crashed or count == 0:
        return {"force_calls": float(count), "success": False, "failure": "crash"}
    if ok:
        return {"force_calls": float(count), "success": True, "failure": None}
    return {"force_calls": float(count), "success": False, "failure": "wrong_stationary"}


def classify_cambridge(
    *,
    calls,
    cap: int,
    energy: float | None,
    returncode: int,
    reference: float,
    tolerance: float,
) -> dict:
    """Cambridge energy rule for one basin-hopping start.

    A call count at the cap is ``cap`` whether or not a minimum was
    written. Under the cap, a missing energy or a nonzero client status
    is a crash. A clean exit is ``judge_globalopt``: at or below the
    published minimum plus the tolerance, and still under the cap.
    """
    finite = energy is not None and math.isfinite(float(energy))
    stored = float(energy) if finite else None
    if calls is None:
        return {
            "force_calls": 0.0,
            "success": False,
            "failure": "crash",
            "minimum_energy": stored,
        }
    count = int(calls)
    if count >= cap:
        return {
            "force_calls": float(count),
            "success": False,
            "failure": "cap",
            "minimum_energy": stored,
        }
    if returncode != 0 or not finite:
        return {
            "force_calls": float(count),
            "success": False,
            "failure": "crash",
            "minimum_energy": stored,
        }
    judged = judge_globalopt(
        float(energy),
        count,
        reference=reference,
        tolerance=tolerance,
        cap=cap,
    )
    return {
        "force_calls": float(judged["force_calls"]),
        "success": bool(judged["success"]),
        "failure": judged["failure"],
        "minimum_energy": stored,
    }


def classify_akmc(*, calls, cap, reached, crashed) -> dict:
    if calls is None:
        return {"force_calls": 0.0, "success": False, "failure": "crash"}
    count = int(calls)
    if count >= cap:
        return {"force_calls": float(count), "success": False, "failure": "cap"}
    if crashed:
        return {"force_calls": float(count), "success": False, "failure": "crash"}
    if reached:
        return {"force_calls": float(count), "success": True, "failure": None}
    return {"force_calls": float(count), "success": False, "failure": "cap"}


def _client(family: str) -> str:
    spec = FAMILIES[family]
    env = os.environ.get(spec["env"])
    path = env or (str(spec["default"]) if spec["default"].is_file() else "")
    if not path or not Path(path).is_file():
        raise RuntimeError(f"set {spec['env']} to the {family} eonclient")
    if family not in _CLIENT_OK:
        proc = subprocess.run(
            [path, spec["flag"]],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        text = proc.stdout or ""
        if spec["token"] not in text:
            raise RuntimeError(f"{path} printed {text!r}, expected {spec['token']}")
        _CLIENT_OK.add(family)
    return path


def _family_of(template: dict) -> str:
    ident = str(template["id"])
    for name, spec in FAMILIES.items():
        if ident.startswith(spec["prefix"]) and template.get("kind"):
            return name
    raise ValueError(f"{ident} is not an eOn harness template")


def _ini_text(template: dict, seed: int, wrapper: Path, *, python_job: bool) -> str:
    sections = json.loads(json.dumps(template["ini"]))
    main = sections.setdefault("Main", {})
    main["random_seed"] = int(seed)
    if not python_job:
        main["parallel"] = False
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    return format_ini(sections)


def _archive(path: str) -> Path:
    return geometry_archive(ROOT, {"id": path, "input": {"archive_path": path}})


def _extract_basename(archive: Path, basename: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as packed:
        match = None
        for member in packed.getmembers():
            if member.isfile() and Path(member.name).name == basename:
                match = member
                break
        if match is None:
            raise FileNotFoundError(f"{basename} is not in {archive}")
        extracted = packed.extractfile(match)
        if extracted is None:
            raise FileNotFoundError(basename)
        dest.write_bytes(extracted.read())
    return dest


@lru_cache(maxsize=32)
def _archive_start_ids(archive_path: str, select: str) -> tuple[str, ...]:
    """Numbered geometry ids in one archived start set.

    ``coords`` keeps ``coords.<digits>`` and drops an unnumbered
    ``coords`` member. ``con`` keeps ``<digits>.con``.
    """
    archive = _archive(archive_path)
    found: set[int] = set()
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            base = Path(member.name).name
            if select == "coords":
                prefix = "coords."
                suffix = base[len(prefix) :] if base.startswith(prefix) else ""
                if suffix.isdigit():
                    found.add(int(suffix))
            elif select == "con":
                if base.endswith(".con") and base[: -len(".con")].isdigit():
                    found.add(int(base[: -len(".con")]))
            elif select == "aa":
                if base.endswith(".aa") and base[: -len(".aa")].isdigit():
                    found.add(int(base[: -len(".aa")]))
            else:
                raise ValueError(f"unknown geometry select {select}")
    return tuple(str(value) for value in sorted(found))


def template_starts(template: dict) -> list[str]:
    """Starts for one harness template.

    An explicit ``starts`` list wins. Otherwise a template archive is
    listed. The LJ38 basin and minima templates leave both unset and
    keep ``default_starts``.
    """
    explicit = template.get("starts")
    if explicit:
        return [str(start) for start in explicit]
    raw = (template.get("input") or {}).get("archive_path")
    if raw:
        select = str((template.get("input") or {}).get("select") or "con")
        return list(_archive_start_ids(str(raw), select))
    return default_starts(str(template["kind"]))


def _pos_con(template: dict | None, start: str, work: Path, kind: str):
    """Reactant ``pos.con`` for a basin or minima start."""
    work.mkdir(parents=True, exist_ok=True)
    raw = (template or {}).get("input", {}).get("archive_path") if template else None
    if not raw:
        archive = _archive("website/globalopt/lj38/100_lj38.tar.gz")
        con = _extract_suffix(archive, f"/{int(start)}.con", work / "src.con")
        return [(str(con), "pos.con")], []
    select = str((template.get("input") or {}).get("select") or "con")
    archive = _archive(str(raw))
    if select == "coords":
        text_path = _extract_basename(archive, f"coords.{start}", work / "coords.txt")
        dest = work / "src.con"
        dest.write_text(
            coords_to_con(text_path.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
        return [(str(dest), "pos.con")], []
    if select == "con":
        con = _extract_basename(archive, f"{start}.con", work / "src.con")
        return [(str(con), "pos.con")], []
    if select == "aa":
        text_path = _extract_basename(archive, f"{start}.aa", work / "start.aa")
        dest = work / "src.con"
        dest.write_text(
            aa_to_con(text_path.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
        return [(str(dest), "pos.con")], []
    raise ValueError(f"{kind} select {select} is not coords, con, or aa")


def _extract_suffix(archive: Path, suffix: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as packed:
        match = None
        for member in packed.getmembers():
            if member.isfile() and member.name.endswith(suffix):
                match = member
                break
        if match is None:
            raise FileNotFoundError(f"{suffix} is not in {archive}")
        extracted = packed.extractfile(match)
        if extracted is None:
            raise FileNotFoundError(suffix)
        dest.write_bytes(extracted.read())
    return dest


def _replace_coords(text: str, positions: np.ndarray) -> str:
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("coordinate"))
    for row, point in enumerate(positions):
        old = lines[header + 1 + row].split()
        tail = " ".join(old[3:]) if len(old) > 3 else "0"
        lines[header + 1 + row] = f"  {point[0]:.8f}   {point[1]:.8f}   {point[2]:.8f}  {tail}"
    return "\n".join(lines) + "\n"


def _pbc_delta(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    lengths = np.asarray(cell, dtype=np.float64)
    if np.any(lengths <= 0.0):
        return np.asarray(delta, dtype=np.float64)
    box = np.diag(lengths)
    folded = np.asarray(delta, dtype=np.float64) @ np.linalg.inv(box)
    folded = (folded % 1.0 + 1.5) % 1.0 - 0.5
    return folded @ box


def _mode_text(rows: np.ndarray) -> str:
    return "".join(f"{x:.16e} {y:.16e} {z:.16e}\n" for x, y, z in rows)


def _interpolated(reactant: Path, product: Path, dest: Path) -> tuple[str, str]:
    """2014 ``interpolate.py`` plus ``make_mode.py`` for an orthorhombic cell."""
    left_text = reactant.read_text(encoding="utf-8", errors="replace")
    left, _fixed, cell = read_con(left_text)
    right, _fixed_right, _cell_right = read_con(product.read_text(encoding="utf-8", errors="replace"))
    if len(left) != len(right):
        raise ValueError(f"{product} has {len(right)} atoms and {reactant} has {len(left)}")
    delta = _pbc_delta(right - left, cell)
    norm = float(np.linalg.norm(delta))
    if norm == 0.0:
        raise ValueError(f"{product} matches {reactant}")
    displaced = _replace_coords(left_text, left + 0.5 * delta)
    (dest / "displacement.con").write_text(displaced, encoding="utf-8")
    return str(dest / "displacement.con"), _mode_text(delta / norm)


def _files(kind: str, start: str, work: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    work.mkdir(parents=True, exist_ok=True)
    if kind == "ts-lj38":
        archive = _archive("website/tsopt/LJ38/coords-con.tar.gz")
        con = _extract_suffix(archive, f"/{start}.con", work / "src.con")
        mode = _seeded_mode(38, int(start))
        return [(str(con), "pos.con"), (str(con), "displacement.con")], [("direction.dat", mode)]
    if kind == "ts-pt":
        archive = _archive("website/tsopt/pt-island/pt-island-con.tgz")
        index = int(start.split("_")[-1])
        reactant = _extract_suffix(archive, "/reactant.con", work / "reactant.con")
        initial = _extract_suffix(archive, f"/initial_{index}.con", work / "initial.con")
        from .eon_jobs import _difference_mode

        return (
            [(str(reactant), "pos.con"), (str(initial), "displacement.con")],
            [("direction.dat", _difference_mode(reactant, initial))],
        )
    if kind == "neb-pt":
        archive = _archive("website/neb/Pt-heptamer/eon-lbfgs/minima.tgz")
        reactant = _extract_suffix(archive, f"/{start}/a.con", work / "a.con")
        product = _extract_suffix(archive, f"/{start}/b.con", work / "b.con")
        return [(str(reactant), "reactant.con"), (str(product), "product.con")], []
    if kind in {"de-lj38", "de-lj38-mode"}:
        archive = _archive("website/double-ended-saddle-search/lj38/coords-con.tgz")
        index = int(start.split("_")[-1])
        reactant = _extract_suffix(archive, f"/start_{index:02d}.con", work / "start.con")
        product = _extract_suffix(archive, f"/end_{index:02d}.con", work / "end.con")
        if kind.endswith("-mode"):
            _displacement, mode = _interpolated(reactant, product, work)
            return (
                [(str(reactant), "pos.con"), (str(work / "displacement.con"), "displacement.con")],
                [("direction.dat", mode)],
            )
        return [(str(reactant), "reactant.con"), (str(product), "product.con")], []
    if kind in {"de-pt", "de-pt-mode"}:
        archive = _archive("website/double-ended-saddle-search/pt-island/pt-island-con.tgz")
        index = int(start.split("_")[-1])
        reactant = _extract_suffix(archive, "/reactant.con", work / "reactant.con")
        product = _extract_suffix(archive, f"/product_{index}.con", work / "product.con")
        if kind.endswith("-mode"):
            _displacement, mode = _interpolated(reactant, product, work)
            return (
                [(str(reactant), "pos.con"), (str(work / "displacement.con"), "displacement.con")],
                [("direction.dat", mode)],
            )
        return [(str(reactant), "reactant.con"), (str(product), "product.con")], []
    if kind in {"de-2ts", "de-2ts-mode"}:
        archive = _archive("website/double-ended-saddle-search/lj38_2ts_separation/coords.tgz")
        index = int(start.split("_")[-1])
        start_xyz = _extract_suffix(archive, f"/start_{index}.xyz", work / "start.xyz")
        end_xyz = _extract_suffix(archive, f"/end_{index}.xyz", work / "end.xyz")
        reactant_text = xyz_to_con(start_xyz.read_text(encoding="utf-8"))
        product_text = xyz_to_con(end_xyz.read_text(encoding="utf-8"))
        reactant = work / "reactant-src.con"
        product = work / "product-src.con"
        reactant.write_text(reactant_text, encoding="utf-8")
        product.write_text(product_text, encoding="utf-8")
        if kind.endswith("-mode"):
            _displacement, mode = _interpolated(reactant, product, work)
            return (
                [(str(reactant), "pos.con"), (str(work / "displacement.con"), "displacement.con")],
                [("direction.dat", mode)],
            )
        return [], [("reactant.con", reactant_text), ("product.con", product_text)]
    if kind in {"basin", "minima"}:
        return _pos_con(None, start, work, kind)
    if kind in {"prd", "hyper"}:
        path = _archive("website/accelerated-md/reactant.con")
        return [(str(path), "pos.con")], []
    raise ValueError(f"unknown harness kind {kind}")


def neb_starts() -> list[str]:
    archive = _archive("website/neb/Pt-heptamer/eon-lbfgs/minima.tgz")
    found = []
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if member.isfile() and member.name.endswith("/a.con"):
                found.append(Path(member.name).parent.name)
    return sorted(set(found))


def default_starts(kind: str) -> list[str]:
    if kind == "ts-lj38":
        return [f"{i:04d}" for i in range(200)]
    if kind == "ts-pt":
        return [f"initial_{i}" for i in range(49)]
    if kind == "neb-pt":
        return neb_starts()
    if kind in {"de-lj38", "de-lj38-mode"}:
        return [f"pair_{i:02d}" for i in range(50)]
    if kind in {"de-pt", "de-pt-mode"}:
        return [f"product_{i}" for i in range(59)]
    if kind in {"de-2ts", "de-2ts-mode"}:
        return [f"pair_{i:02d}" for i in range(51)]
    if kind == "lowest":
        return [f"{i:04d}" for i in range(200)]
    if kind in {"basin", "minima"}:
        return [str(i) for i in range(100)]
    if kind in {"akmc", "htst", "prd", "hyper"}:
        return ["0"]
    raise ValueError(f"unknown harness kind {kind}")


def _run_client(client: str, ini: str, dest: Path, copies, texts, timeout: int | None) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.ini").write_text(ini, encoding="utf-8")
    for src, name in copies:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, target)
    for name, text in texts:
        (dest / name).write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [client],
        cwd=dest,
        env=_client_env(client),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        timeout=timeout,
    )
    fields = _read_dat(dest / "results.dat")
    return {"returncode": proc.returncode, "fields": fields, "stderr": proc.stderr or ""}


def _guard(record: dict, client_calls) -> dict:
    if record["force_calls"] == 0 and client_calls not in (None, 0):
        record["success"] = False
        record["failure"] = "crash"
    return record


def _cache_path(family: str, template_name: str, seed: int, start: str, passes: int) -> Path:
    safe = template_name.replace("/", "-")
    template = load_template(ROOT, template_name)
    kernel = str(template.get("kernel") or "unset")
    token = str(template.get("cache_token") or "")
    parts = [
        CACHE,
        "harness-1",
        FAMILIES[family]["cache"],
        KERNEL,
        kernel,
    ]
    if token:
        parts.append(token)
    parts.extend([safe, f"passes-{passes}", str(seed), f"{start}.json"])
    return Path(*parts)


def _read_cache(family: str, template_name: str, seed: int, start: str, passes: int) -> dict | None:
    path = _cache_path(family, template_name, seed, start, passes)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(family: str, template_name: str, seed: int, start: str, passes: int, row: dict) -> None:
    path = _cache_path(family, template_name, seed, start, passes)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _logged_basin_minimum(run: Path) -> float | None:
    """Best global minimum printed in ``client.log``.

    The client writes ``minimum_energy`` when the step loop returns. A
    force cap stops the client inside a quench, so that line is absent
    and the log line is the measured minimum.
    """
    path = run / "client.log"
    if not path.is_file():
        return None
    best = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 6 or parts[0] != "[Basin" or parts[1] != "Hopping]":
            continue
        if not parts[2][:1].isdigit():
            continue
        try:
            value = float(parts[5])
        except ValueError:
            continue
        best = value if best is None else min(best, value)
    return best


def _earr_min(path: Path) -> float | None:
    if not path.is_file():
        return None
    best = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        best = value if best is None else min(best, value)
    return best


def _run_lowest(template, seed, start, work, server, wrapper, cap, client) -> dict:
    from .eon_jobs import _lowest_once
    from .potserver import remote_count, remote_reset

    archive = _archive("website/lowest-eigenvector/lj38/coords-con.tar.gz")
    con = _extract_suffix(archive, f"/{start}.con", work / "src.con")
    mode = _extract_suffix(archive, f"/{start}_mode", work / "ref.mode")
    reference = flat_floats(mode.read_text(encoding="utf-8", errors="replace"))
    direction = _numpy_direction()
    sections = json.loads(json.dumps(template["ini"]))
    sections.setdefault("Main", {})["random_seed"] = int(seed)
    sections["Main"]["parallel"] = False
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    method = sections["Saddle Search"]["min_mode_method"]
    limit = int(template.get("iteration_limit", 1000))
    threshold = float(template.get("overlap", 0.99))
    records = []
    for attempt in range(1, limit + 1):
        sections_k = json.loads(json.dumps(sections))
        if method == "dimer":
            sections_k.setdefault("Dimer", {})["rotations_max"] = attempt
        elif method == "lanczos":
            sections_k.setdefault("Lanczos", {})["max_iterations"] = attempt
        else:
            raise ValueError(f"unknown lowest mode {method}")
        remote_reset(server.host, server.port)
        found = _lowest_once(client, format_ini(sections_k), str(work / "run"), str(con), direction)
        calls, _verify = remote_count(server.host, server.port)
        overlap = None
        if found["mode_text"]:
            try:
                overlap = abs_overlap(flat_floats(found["mode_text"]), reference)
            except ValueError:
                overlap = None
        records.append({"calls": calls, "overlap": overlap, "found": found, "attempt": attempt})
        if overlap is not None and overlap >= threshold:
            break
    chosen = None
    for row in records:
        if row["overlap"] is not None and row["overlap"] >= threshold:
            chosen = row
            break
    if chosen is None:
        chosen = records[-1] if records else None
    if chosen is None:
        return classify_client(calls=None, reason=None, cap=cap, crashed=True)
    reached = chosen["overlap"] is not None and chosen["overlap"] >= threshold
    record = classify_client(
        calls=chosen["calls"],
        reason=chosen["found"].get("reason"),
        cap=cap,
        overlap_ok=reached,
    )
    record["client_calls"] = chosen["found"].get("raw")
    record["client_reason"] = chosen["found"].get("reason")
    record["k"] = chosen["attempt"] if reached else None
    record["overlap"] = chosen["overlap"]
    return _guard(record, record.get("client_calls"))


def _geometry(template: dict) -> Path | None:
    raw = template.get("geometry")
    if not raw:
        return None
    return _archive(str(raw))


def run_case(
    template_name: str,
    seed: int,
    start: str,
    *,
    passes: int | None = None,
    budget: int | None = None,
) -> dict:
    """One seed and one start. The dict is what the tracks report."""
    from .eon_akmc import _one_repetition
    from .potserver import PotServer, remote_count

    template = load_template(ROOT, template_name)
    family = _family_of(template)
    kind = str(template["kind"])
    if passes is None:
        passes = int(template.get("pass_limit") or 0)
    cached = _read_cache(family, template_name, int(seed), str(start), int(passes))
    if cached is not None:
        return cached
    cap = int(template["failure_cap"])
    python_job = kind in {"akmc", "htst"}
    if budget is None:
        budget = int(template.get("budget_seconds") or 0)
    else:
        budget = int(budget)
    if budget > 0:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(budget)
    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="optbench-eon-"))
    server = None
    try:
        client = _client(family)
        evaluate = kernel_for(str(template["kernel"]))
        server = PotServer(evaluate, cap=cap).start()
        os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
        try:
            wrapper = bridge_binary()
            if kind == "lowest":
                record = _run_lowest(template, seed, start, work, server, wrapper, cap, client)
            elif kind in {"akmc", "htst"}:
                patched = dict(template)
                patched["ini"] = json.loads(json.dumps(template["ini"]))
                patched["ini"]["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
                row = _one_repetition(
                    ROOT,
                    patched,
                    Path(client),
                    work / "akmc",
                    int(seed),
                    int(template.get("workers") or 1),
                    passes=int(passes),
                    geometry=_geometry(template),
                )
                calls, _verify = remote_count(server.host, server.port)
                crashed = bool(row.get("error")) and not row.get("total_searches")
                if kind == "htst":
                    reached = int(row.get("qualifying") or 0) >= 1
                    record = classify_flag(calls=calls, cap=cap, ok=reached and not crashed, crashed=crashed)
                else:
                    record = classify_akmc(
                        calls=calls,
                        cap=cap,
                        reached=bool(row.get("reached")),
                        crashed=crashed,
                    )
                record["client_calls"] = row.get("force_calls")
                record["fraction_connected"] = row.get("fraction_connected")
                record["passes"] = row.get("passes")
                record["qualifying"] = row.get("qualifying")
                record["total_searches"] = row.get("total_searches")
                record["unique_saddles"] = row.get("unique_saddles")
                record["error"] = row.get("error") or ""
                if record["error"] and int(calls or 0) == 0:
                    record["uncached"] = True
                record = _guard(record, record.get("client_calls"))
            else:
                if kind in {"basin", "minima"} and (template.get("input") or {}).get("archive_path"):
                    copies, texts = _pos_con(template, str(start), work / "src", kind)
                else:
                    copies, texts = _files(kind, str(start), work / "src")
                found = _run_client(
                    client,
                    _ini_text(template, int(seed), wrapper, python_job=python_job),
                    work / "run",
                    copies,
                    texts,
                    None,
                )
                calls, _verify = remote_count(server.host, server.port)
                fields = found["fields"]
                if kind == "basin" and template.get("cambridge"):
                    energy_text = fields.get("minimum_energy")
                    if energy_text is not None:
                        energy = float(energy_text)
                    else:
                        energy = _logged_basin_minimum(work / "run")
                    problem = resolve_problem(ROOT, template["problem"])
                    scale = template.get("energy_kj_per_ev")
                    if scale not in (None, "") and energy is not None:
                        energy = float(energy) * float(scale)
                    record = classify_cambridge(
                        calls=calls,
                        cap=cap,
                        energy=energy,
                        returncode=int(found["returncode"]),
                        reference=reference_energy(problem),
                        tolerance=energy_tolerance(problem),
                    )
                elif kind == "basin":
                    energy_text = fields.get("minimum_energy")
                    stop = float(template["ini"]["Basin Hopping"]["stop_energy"])
                    energy = float(energy_text) if energy_text is not None else None
                    ok = energy is not None and energy < stop and found["returncode"] == 0
                    crashed = energy is None or found["returncode"] != 0
                    record = classify_flag(calls=calls, cap=cap, ok=ok, crashed=crashed)
                    record["minimum_energy"] = energy
                elif kind == "minima":
                    energy = _earr_min(work / "run" / "earr.dat")
                    stop = float(template["ini"]["Global Optimization"]["target_energy"])
                    ok = energy is not None and energy < stop and found["returncode"] == 0
                    record = classify_flag(
                        calls=calls,
                        cap=cap,
                        ok=ok,
                        crashed=energy is None or found["returncode"] != 0,
                    )
                    record["minimum_energy"] = energy
                elif kind in {"prd", "hyper"}:
                    transition = fields.get("transition_found") == "1"
                    crashed = found["returncode"] != 0 or "total_force_calls" not in fields and calls == 0
                    record = classify_flag(calls=calls, cap=cap, ok=transition, crashed=crashed)
                    record["transition_found"] = 1 if transition else 0
                else:
                    reason = fields.get("termination_reason")
                    record = classify_client(
                        calls=calls,
                        reason=reason,
                        cap=cap,
                        crashed=found["returncode"] != 0 and reason is None,
                    )
                    if kind in {"ts-lj38", "ts-pt"} and record["success"]:
                        saddle = work / "run" / "saddle.con"
                        if not saddle.is_file():
                            record["success"] = False
                            record["failure"] = "crash"
                        else:
                            checked = saddle_index(
                                evaluate,
                                saddle.read_text(encoding="utf-8", errors="replace"),
                                periodic=str(template["kernel"]) == "morse",
                            )
                            record["force_norm"] = checked["force_norm"]
                            record["hessian_index"] = checked["hessian_index"]
                            stop = float(template["ini"]["Optimizer"]["converged_force"])
                            if checked["force_norm"] > stop:
                                record["success"] = False
                                record["failure"] = "divergence"
                            elif checked["hessian_index"] != 1:
                                record["success"] = False
                                record["failure"] = "wrong_stationary"
                raw_text = fields.get("total_force_calls") or fields.get("force_calls_neb")
                record["client_calls"] = int(float(raw_text)) if raw_text is not None else None
                record["client_reason"] = fields.get("termination_reason")
                record = _guard(record, record.get("client_calls"))
        finally:
            server.stop()
            os.environ.pop("OPTBENCH_POTSERVER", None)
        record["seconds"] = time.monotonic() - started
        record["counter"] = "harness"
        record["kernel"] = template["kernel"]
        if not record.get("uncached"):
            _write_cache(family, template_name, int(seed), str(start), int(passes), record)
        return record
    except BudgetExceeded:
        calls = 0.0
        if server is not None:
            calls = float(server.counter.calls)
        return {
            "timed_out": True,
            "force_calls": calls,
            "success": False,
            "failure": "crash",
            "seconds": float(budget),
            "counter": "harness",
            "uncached": True,
        }
    finally:
        if budget > 0:
            signal.alarm(0)
        shutil.rmtree(work, ignore_errors=True)


def _jobs(family: str) -> list[dict]:
    prefix = FAMILIES[family]["prefix"]
    found = []
    for template in list_templates(ROOT):
        if str(template.get("id", "")).startswith(prefix) and template.get("kind"):
            found.append(load_template(ROOT, template["id"]))
    return found


def grid_points(template: dict, *, full: bool, limit: int = 0) -> list[tuple[int, str]]:
    seeds = [int(seed) for seed in template.get("seeds", [1])]
    starts = template_starts(template)
    if not full:
        seeds, starts = seeds[:1], starts[:1]
    points = [(seed, start) for seed in seeds for start in starts]
    if limit > 0:
        return points[:limit]
    return points


def _start_row(seed: int, start: str, record: dict) -> dict:
    row = {
        "call_unit": "total",
        "count_key": "harness",
        "counter": "harness",
        "force_calls": int(round(float(record["force_calls"]))),
        "geometry": start,
        "raw": int(round(float(record["force_calls"]))),
        "seed": int(seed),
        "success": bool(record["success"]),
    }
    if record.get("failure"):
        row["failure"] = record["failure"]
    for key in (
        "seconds",
        "client_calls",
        "client_reason",
        "overlap",
        "k",
        "minimum_energy",
        "qualifying",
        "total_searches",
        "unique_saddles",
        "fraction_connected",
        "passes",
        "transition_found",
        "error",
        "force_norm",
        "hessian_index",
    ):
        if record.get(key) not in (None, ""):
            row[key] = record[key]
    return row


def write_campaign(template: dict, rows: list[tuple[int, str, dict]], output: Path) -> dict | None:
    """Write benchmark.dat and the per-start file. Timeouts stay out."""
    from .board import stated_full_n

    kept = [
        (seed, start, row)
        for seed, start, row in rows
        if not row.get("timed_out") and not row.get("uncached")
    ]
    if not kept:
        return None
    problem = resolve_problem(ROOT, template["problem"])
    full_n = stated_full_n(ROOT, problem)
    partial = bool(template.get("declare_partial")) or full_n is None or len(kept) < int(full_n or 0)
    from .score import score_v2

    scored = [
        {
            "force_calls": float(row["force_calls"]),
            "success": bool(row.get("success")),
            "failure": row.get("failure"),
        }
        for _seed, _start, row in kept
    ]
    cap = int(template["failure_cap"]) if template.get("failure_cap") not in (None, "") else None
    ranked = score_v2(scored, cap)
    calls = [float(row["force_calls"]) for _seed, _start, row in kept]
    energies = [
        float(row["minimum_energy"])
        for _seed, _start, row in kept
        if row.get("minimum_energy") is not None and math.isfinite(float(row["minimum_energy"]))
    ]
    summary = {
        "nfailed": int(ranked["nfailed"]),
        "call_unit": "total",
    }
    text = benchmark_text(summary, template)

    def _fmt(value: float) -> str:
        return f"{float(value):.10g}"

    extra = [
        "counter harness",
        "count_keys HarnessCounter",
        f"n {ranked['n']}",
        f"par2_mean {_fmt(ranked['par2_mean'])}",
        f"median {_fmt(ranked['median'])}",
        f"median_low {_fmt(ranked['median_low'])}",
        f"median_high {_fmt(ranked['median_high'])}",
        f"wilson_low {_fmt(ranked['wilson_low'])}",
        f"wilson_high {_fmt(ranked['wilson_high'])}",
        f"success_rate {_fmt(ranked['success_rate'])}",
        f"n_cap {ranked['n_cap']}",
        f"n_wrong_stationary {ranked['n_wrong_stationary']}",
        f"n_divergence {ranked['n_divergence']}",
        f"n_crash {ranked['n_crash']}",
        f"force_calls_min {int(min(calls))}",
        f"force_calls_max {int(max(calls))}",
        f"force_calls_median {_fmt(float(np.median(calls)))}",
    ]
    if cap is not None:
        extra.append(f"cap {cap}")
    if energies:
        extra.append(f"energy_min {_fmt(min(energies))}")
        extra.append(f"energy_max {_fmt(max(energies))}")
    if template.get("tuning"):
        extra.append(f"tuning {template['tuning']}")
    from .board import visible_label

    label = visible_label(
        {"label": template.get("label", ""), "tuning": template.get("tuning", "")},
        template,
    )
    if label:
        extra.append(f"label {label}")
    if template.get("ledger"):
        extra.append(f"ledger {template['ledger']}")
    if partial:
        extra.append("partial true")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip("\n") + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    write_starts(output.with_suffix(".starts"), [_start_row(seed, start, row) for seed, start, row in kept])
    summary["starts"] = len(kept)
    summary["partial"] = partial
    summary["template"] = template["id"]
    summary["problem"] = template["problem"]
    return summary


def refresh_campaign_file(template_name: str) -> dict | None:
    """Rewrite one campaign file from its starts, including capped rows.

    The score is PAR-2 over every kept start. A complete set is not
    marked partial.
    """
    template = load_template(ROOT, template_name)
    stem = str(template["id"]).replace("/", "-")
    starts_path = ROOT / "results" / "campaign" / f"{stem}.starts"
    if not starts_path.is_file():
        return None
    rows = []
    for line in starts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        record = {
            "force_calls": item["force_calls"],
            "success": item["success"],
            "failure": item.get("failure"),
        }
        for key in ("minimum_energy", "seconds", "timed_out", "uncached"):
            if key in item:
                record[key] = item[key]
        rows.append((int(item["seed"]), str(item.get("geometry") or item.get("start")), record))
    return write_campaign(template, rows, starts_path.with_suffix(".dat"))


def _start_order(start: str) -> tuple:
    text = str(start)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _publish(template: dict, rows: list) -> dict | None:
    ordered = sorted(rows, key=lambda item: (item[0], _start_order(item[1])))
    stem = str(template["id"]).replace("/", "-")
    output = ROOT / "results" / "campaign" / f"{stem}.dat"
    summary = write_campaign(template, ordered, output)
    if summary:
        print(
            f"{summary['problem']} {summary['template']} "
            f"force_calls={summary.get('force_calls')} nfailed={summary['nfailed']} "
            f"starts={summary['starts']} partial={summary['partial']}",
            flush=True,
        )
    return summary


def fill(family: str, *, kinds: set[str] | None = None, workers: int = 1, limit: int = 0, full: bool = True) -> list[dict]:
    """Run harness starts and write one campaign file per template.

    The campaign file is rewritten after each finished start, sorted by
    seed and start index, so a long grid keeps the starts that returned.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = []
    templates = {}
    for template in _jobs(family):
        kind = str(template["kind"])
        if kinds and kind not in kinds and template["id"] not in kinds and template["id"].split("/", 1)[-1] not in kinds:
            continue
        templates[template["id"]] = template
        passes = int(template.get("pass_limit") or 0)
        for seed, start in grid_points(template, full=full, limit=limit):
            jobs.append((template["id"], seed, start, passes))
    grouped: dict[str, list] = {ident: [] for ident in templates}

    def _take(row: dict) -> None:
        grouped.setdefault(row["template"], []).append((row["seed"], row["start"], row))
        template = templates.get(row["template"])
        if template is not None:
            _publish(template, grouped[row["template"]])

    if workers <= 1:
        for job in jobs:
            _take(_fill_job(job))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            pending = [pool.submit(_fill_job, job) for job in jobs]
            for future in as_completed(pending):
                _take(future.result())
    written = []
    for ident, template in templates.items():
        rows = grouped.get(ident, [])
        summary = _publish(template, rows)
        if summary:
            written.append(summary)
        else:
            errors = [str(row.get("error") or row.get("failure") or "") for _seed, _start, row in rows]
            print(
                f"{template['problem']} {ident} wrote nothing "
                f"rows={len(rows)} errors={errors[:4]}",
                flush=True,
            )
    return written


def _fill_job(payload: tuple) -> dict:
    template_name, seed, start, passes = payload
    try:
        row = run_case(template_name, int(seed), str(start), passes=int(passes))
    except Exception as exc:
        row = {
            "force_calls": 0.0,
            "success": False,
            "failure": "crash",
            "error": f"{type(exc).__name__}: {exc}",
            "counter": "harness",
            "uncached": True,
        }
    row["template"] = template_name
    row["seed"] = int(seed)
    row["start"] = str(start)
    return row


def _optional(row: dict, key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    return float(value)


def install_benchmarks(family: str, namespace: dict) -> None:
    """Define one asv class per harness template in ``namespace``."""
    for template in _jobs(family):
        kind = str(template["kind"])
        asv_full = bool(template.get("asv_full", True))
        seeds = [int(seed) for seed in template.get("seeds", [1])]
        starts = template_starts(template)
        if full_grid() and asv_full:
            used_seeds, used_starts = seeds, starts
            passes = int(template.get("pass_limit") or 0)
        else:
            used_seeds, used_starts = seeds[:1], starts[:1]
            passes = 1 if template.get("pass_limit") else 0
        spec = {
            "cls": str(template["asv_class"]),
            "template": template["id"],
            "cap": int(template["failure_cap"]),
            "seeds": used_seeds,
            "starts": used_starts,
            "passes": passes,
            "timeout": int(template.get("asv_timeout") or 900),
            "kind": kind,
        }
        namespace[spec["cls"]] = _bind(spec)


def _bind(spec: dict):
    class _Benchmark(TrackedSearch):
        param_names = ["seed", "start"]
        params = [spec["seeds"], spec["starts"]]
        timeout = spec["timeout"]
        template_name = spec["template"]
        cap = spec["cap"]
        passes = spec["passes"]
        kind = spec["kind"]

        def setup(self, seed, start):
            return None

        def measure(self, seed, start):
            self.record = run_case(self.template_name, int(seed), str(start), passes=int(self.passes))
            return self.record

        def time_search(self, seed, start):
            self.measure(seed, start)

        def track_force_calls(self, seed, start) -> float:
            return float(self.measure(seed, start)["force_calls"])

        def track_success(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["success"] else 0.0

        def track_failure_cap(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "cap" else 0.0

        def track_failure_wrong_stationary(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "wrong_stationary" else 0.0

        def track_failure_divergence(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "divergence" else 0.0

        def track_failure_crash(self, seed, start) -> float:
            return 1.0 if self.measure(seed, start)["failure"] == "crash" else 0.0

    _Benchmark.__name__ = spec["cls"]
    _Benchmark.__qualname__ = spec["cls"]
    for name in (
        "track_force_calls",
        "track_success",
        "track_failure_cap",
        "track_failure_wrong_stationary",
        "track_failure_divergence",
        "track_failure_crash",
    ):
        getattr(_Benchmark, name).unit = {
            "track_force_calls": "force calls",
            "track_success": "fraction",
        }.get(name, "count")
    if spec["kind"] in {"akmc", "htst"}:

        def track_unique_saddles(self, seed, start) -> float:
            return _optional(self.measure(seed, start), "unique_saddles")

        def track_total_searches(self, seed, start) -> float:
            return _optional(self.measure(seed, start), "total_searches")

        track_unique_saddles.unit = "saddles"
        track_total_searches.unit = "searches"
        _Benchmark.track_unique_saddles = track_unique_saddles
        _Benchmark.track_total_searches = track_total_searches
    return _Benchmark


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run eOn harness campaigns")
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("--kind", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    fill(
        args.family,
        kinds=set(args.kind) if args.kind else None,
        workers=args.workers,
        limit=1 if args.smoke and args.limit == 0 else args.limit,
        full=not args.smoke,
    )


if __name__ == "__main__":
    main()
