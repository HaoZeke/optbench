"""Native pair kernels and the eonclient ``ext_pot`` bridge.

The C loops are the same arithmetic as the Python kernels, in the same
order. ``OPTBENCH_FASTPOT=0`` keeps the Python loops. A machine without
``gcc`` keeps them too. The bridge only forwards ``from_eon_to_extpot``
to the potential server; the server still owns the counter.
"""

from __future__ import annotations

import array
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

NATIVE = Path(__file__).resolve().parent / "native"
_KERNEL_SRC = NATIVE / "kernels.c"
_AVX_SRC = NATIVE / "lj_avx.c"
_QUENCH_SRC = NATIVE / "lj_quench.c"
_BRIDGE_SRC = NATIVE / "extpot_bridge.c"
_CFLAGS = [
    "-std=c11",
    "-O3",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fno-builtin",
    "-Wall",
    "-Wextra",
]


def enabled() -> bool:
    return os.environ.get("OPTBENCH_FASTPOT") != "0"


def _stamp(source: Path) -> str:
    data = source.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _build_dir() -> Path:
    root = Path(os.environ.get("OPTBENCH_FASTPOT_DIR", Path(tempfile.gettempdir()) / "optbench-fastpot"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _compile_many(dest: Path, units: list[tuple[Path, list[str]]], link: list[str]) -> bool:
    """Compile each unit with its own flags and link one shared library."""
    gcc = shutil.which("gcc")
    if gcc is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    objects: list[str] = []
    for source, flags in units:
        if not source.is_file():
            return False
        obj = dest.with_name(dest.name + f".{source.stem}.o")
        compiled = subprocess.run(
            [gcc, *_CFLAGS, "-fPIC", *flags, "-c", "-o", str(obj), str(source)],
            check=False,
            capture_output=True,
            text=True,
        )
        log = dest.with_suffix(dest.suffix + ".log")
        if compiled.returncode != 0:
            log.write_text(compiled.stderr, encoding="utf-8")
            return False
        objects.append(str(obj))
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    result = subprocess.run(
        [gcc, "-shared", "-o", str(temporary), *objects, *link, "-lm"],
        check=False,
        capture_output=True,
        text=True,
    )
    log = dest.with_suffix(dest.suffix + ".log")
    if result.returncode != 0:
        log.write_text(result.stderr, encoding="utf-8")
        temporary.unlink(missing_ok=True)
        return False
    log.unlink(missing_ok=True)
    temporary.replace(dest)
    dest.chmod(dest.stat().st_mode | 0o755)
    return True


def _compile(source: Path, dest: Path, extra: list[str]) -> bool:
    if not source.is_file():
        return False
    return _compile_many(dest, [(source, extra)], [])


def _lj_units(include: Path | None = None) -> list[tuple[Path, list[str]]]:
    quench_flags = [f"-I{include}"] if include is not None else []
    return [
        (_KERNEL_SRC, []),
        (_AVX_SRC, ["-mavx512f", "-mfma"]),
        (_QUENCH_SRC, quench_flags),
    ]


def _source_stamp(*sources: Path) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


_KERNELS: ctypes.CDLL | None | bool = None


def kernels() -> ctypes.CDLL | None:
    """The shared pair library, or ``None`` when it cannot be built."""
    global _KERNELS
    if not enabled():
        return None
    if _KERNELS is False:
        return None
    if isinstance(_KERNELS, ctypes.CDLL):
        return _KERNELS
    dest = _build_dir() / f"kernels-{_source_stamp(_KERNEL_SRC, _AVX_SRC)}.so"
    if not dest.is_file() and not _compile_many(
        dest,
        [(_KERNEL_SRC, []), (_AVX_SRC, ["-mavx512f", "-mfma"])],
        [],
    ):
        _KERNELS = False
        return None
    lib = ctypes.CDLL(str(dest))
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.optbench_morse.argtypes = [
        pointer,
        ctypes.c_int,
        pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        pointer,
        pointer,
    ]
    lib.optbench_morse.restype = ctypes.c_int
    lib.optbench_lj_cluster.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        pointer,
        pointer,
    ]
    lib.optbench_lj_cluster.restype = ctypes.c_int
    lib.optbench_lj_cut.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        pointer,
        pointer,
    ]
    lib.optbench_lj_cut.restype = ctypes.c_int
    lib.optbench_lj_gradient.argtypes = [
        pointer,
        ctypes.c_int,
        ctypes.c_double,
        pointer,
        pointer,
        pointer,
    ]
    lib.optbench_lj_gradient.restype = ctypes.c_int
    _KERNELS = lib
    return lib


def bridge_executable() -> Path | None:
    """Path to the native ``ext_pot`` bridge, or ``None``."""
    if not enabled():
        return None
    dest = _build_dir() / f"extpot-bridge-{_stamp(_BRIDGE_SRC)}"
    if dest.is_file():
        return dest
    if not _compile(_BRIDGE_SRC, dest, []):
        return None
    return dest


def _pointer(buf: array.array):
    address, _length = buf.buffer_info()
    return ctypes.cast(address, ctypes.POINTER(ctypes.c_double))


def _positions(values) -> tuple[array.array, int]:
    count = len(values)
    if count % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    return array.array("d", (float(value) for value in values)), count // 3


def _forces(count: int) -> array.array:
    if count == 0:
        return array.array("d")
    return array.array("d", [0.0]) * count


def morse_forces(positions, lengths, well: float, width: float, equilibrium: float, cutoff: float, shift: float):
    """Energy and forces, or ``None`` when the native kernel is unavailable."""
    lib = kernels()
    if lib is None:
        return None
    pos, atoms = _positions(positions)
    out = _forces(atoms * 3)
    energy = ctypes.c_double()
    length_buf = None
    length_ptr = None
    if lengths is not None:
        length_buf = array.array("d", (float(value) for value in lengths))
        if len(length_buf) != 3:
            raise ValueError("morse lengths must be three floats")
        length_ptr = _pointer(length_buf)
    code = lib.optbench_morse(
        _pointer(pos) if atoms else None,
        atoms,
        length_ptr,
        float(well),
        float(width),
        float(equilibrium),
        float(cutoff),
        float(shift),
        ctypes.byref(energy),
        _pointer(out) if atoms else None,
    )
    if code != 0:
        return None
    return float(energy.value), tuple(out)


def lj_cluster_forces(positions, u0: float, psi: float, cutoff: float, cutoff_u: float):
    lib = kernels()
    if lib is None:
        return None
    pos, atoms = _positions(positions)
    out = _forces(atoms * 3)
    energy = ctypes.c_double()
    code = lib.optbench_lj_cluster(
        _pointer(pos) if atoms else None,
        atoms,
        float(u0),
        float(psi),
        float(cutoff),
        float(cutoff_u),
        ctypes.byref(energy),
        _pointer(out) if atoms else None,
    )
    if code != 0:
        return None
    return float(energy.value), tuple(out)


def lj_cut_forces(positions, epsilon: float, sigma: float, cutoff: float):
    lib = kernels()
    if lib is None:
        return None
    pos, atoms = _positions(positions)
    out = _forces(atoms * 3)
    energy = ctypes.c_double()
    code = lib.optbench_lj_cut(
        _pointer(pos) if atoms else None,
        atoms,
        float(epsilon),
        float(sigma),
        float(cutoff),
        ctypes.byref(energy),
        _pointer(out) if atoms else None,
    )
    if code != 0:
        return None
    return float(energy.value), tuple(out)


def lj_gradient(coords, cutoff: float = 40.0):
    """Energy, gradient, and minimum pair distance for cutoff 12-6.

    Returns ``None`` when the native kernel is unavailable. The gradient
    is ``dE/dx``. A coincident pair has energy +infinity and a zero
    gradient. An AVX-512 build uses the same potential; its energy can
    differ from the scalar sum by about 1e-13.
    """
    import numpy as np

    lib = kernels()
    if lib is None:
        return None
    pos = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1)
    if pos.size % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    atoms = pos.size // 3
    grad = np.zeros(pos.size, dtype=np.float64)
    energy = ctypes.c_double()
    minimum = ctypes.c_double()
    pointer = ctypes.POINTER(ctypes.c_double)
    code = lib.optbench_lj_gradient(
        pos.ctypes.data_as(pointer) if atoms else None,
        atoms,
        float(cutoff),
        ctypes.byref(energy),
        grad.ctypes.data_as(pointer) if atoms else None,
        ctypes.byref(minimum),
    )
    if code != 0:
        return None
    return float(energy.value), grad, float(minimum.value)


class _QuenchOut(ctypes.Structure):
    _fields_ = [
        ("energy", ctypes.c_double),
        ("min_distance", ctypes.c_double),
        ("calls", ctypes.c_int),
        ("code", ctypes.c_int),
        ("capped", ctypes.c_int),
    ]


_QUENCH: ctypes.CDLL | None | bool = None


def _nlopt_dirs() -> tuple[Path, Path] | None:
    """Include and lib directories for the installed NLopt C library."""
    import importlib.util

    spec = importlib.util.find_spec("_nlopt")
    if spec is None or not spec.origin:
        return None
    library = Path(spec.origin).resolve().parents[2]
    include = library.parent / "include"
    if not (include / "nlopt.h").is_file():
        return None
    if not any(library.glob("libnlopt.so*")):
        return None
    return include, library


def quench_library() -> ctypes.CDLL | None:
    """NLopt LD_LBFGS quench linked against the cutoff kernel, or ``None``."""
    global _QUENCH
    if not enabled():
        return None
    if _QUENCH is False:
        return None
    if isinstance(_QUENCH, ctypes.CDLL):
        return _QUENCH
    dirs = _nlopt_dirs()
    if dirs is None or not _QUENCH_SRC.is_file() or not _KERNEL_SRC.is_file() or not _AVX_SRC.is_file():
        _QUENCH = False
        return None
    include, library = dirs
    dest = _build_dir() / f"lj-quench-{_source_stamp(_KERNEL_SRC, _AVX_SRC, _QUENCH_SRC)}.so"
    if not dest.is_file() and not _compile_many(
        dest,
        _lj_units(include),
        [f"-L{library}", "-lnlopt", f"-Wl,-rpath,{library}"],
    ):
        _QUENCH = False
        return None
    lib = ctypes.CDLL(str(dest))
    lib.optbench_lj_quench.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(_QuenchOut),
    ]
    lib.optbench_lj_quench.restype = ctypes.c_int
    _QUENCH = lib
    return lib


def lj_lbfgs_quench(
    origin,
    *,
    cap: int,
    local_max: int,
    tolg: float,
    cutoff: float = 40.0,
):
    """One stock LD_LBFGS quench, with no coordinate bounds.

    Returns ``(point, energy, min_distance, calls, code, capped)``, or
    ``None`` when the native quench is unavailable. ``calls`` counts fresh
    geometries only. ``capped`` means the next geometry was refused.
    """
    import numpy as np

    lib = quench_library()
    if lib is None:
        return None
    point = np.array(origin, dtype=np.float64, copy=True).reshape(-1)
    if point.size % 3 != 0:
        raise ValueError("positions must be x, y, z triples")
    out = _QuenchOut()
    pointer = ctypes.POINTER(ctypes.c_double)
    code = lib.optbench_lj_quench(
        point.ctypes.data_as(pointer),
        int(point.size),
        int(cap),
        int(local_max),
        float(tolg),
        float(cutoff),
        ctypes.byref(out),
    )
    if code != 0:
        return None
    return (
        point,
        float(out.energy),
        float(out.min_distance),
        int(out.calls),
        int(out.code),
        bool(out.capped),
    )
