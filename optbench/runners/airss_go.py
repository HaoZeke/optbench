"""AIRSS random structure search on the harness Lennard-Jones potential.

``buildcell`` from AIRSS 0.9.4 draws each cell. The relaxation is the
pp3 two-point step (Barzilai and Borwein), two passes, on the spec
Lennard-Jones potential. pp3's built-in pair potential subtracts a
cutoff shift, so the scored evaluations stay on the spec potential.
Each evaluation is one harness call. The compiled kernel in
``lj_flat.c`` is that potential; the NumPy function is the reference.

One archived start name is one independent search. The search stops
when the harness energy is at or below the Cambridge Cluster Database
minimum plus the problem tolerance. A problem force-call cap, or the
template ``failure_cap`` when the problem leaves the number out, also
stops the search.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.globalopt import judge_globalopt
from optbench.potentials import binary_lennard_jones, lennard_jones
from optbench.score import half_even, score_v2
from optbench.v2 import collect_provenance

WALES_URL = "https://www-wales.ch.cam.ac.uk/examples/GMIN/LJ38/lowest"
TARBALL_SHA256 = "5593cc4ee360d998764fceb188284fdf0ef781d206754176dcb360c65103ddaa"
# Runaway guard. A hit is a crash, not a scored cap: the problem states none.
SAFETY_STRUCTURES = 4000
_PASSES = 2
_LIB = None
_LIB_FAILED = False


class BuildFailure(RuntimeError):
    """buildcell returned no usable cell."""


class _StepPotential:
    """One harness call per ``energy_forces`` entry.

    The two-point step asks for the energy and the gradient together, so
    there is no second read of the same geometry to collapse.
    """

    def __init__(self, evaluate):
        self.calls = 0
        self._evaluate = evaluate

    def energy_forces(self, coords):
        self.calls += 1
        energy, gradient = self._evaluate(coords)
        return float(energy), gradient


def search_cap(problem: dict, template: dict) -> int | None:
    """Force-call cap from the problem, or the template when the problem omits it.

    Every globalopt problem states ``max_force_calls`` 2000000. A template
    ``failure_cap`` is used only when a problem omits the key.
    """
    raw = (problem.get("convergence") or {}).get("max_force_calls")
    if raw not in (None, ""):
        return int(raw)
    raw = template.get("failure_cap")
    if raw not in (None, ""):
        return int(raw)
    return None


def _kernel_library():
    """Compile ``lj_flat.c`` once and return the shared library, or None."""
    global _LIB, _LIB_FAILED
    if _LIB is not None:
        return _LIB
    if _LIB_FAILED:
        return None
    source = Path(__file__).with_name("lj_flat.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path(os.environ.get("OPTBENCH_LJ_CACHE", Path.home() / ".cache" / "optbench"))
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"lj_flat-{digest}.so"
    if not so.is_file():
        temporary = cache / f"lj_flat-{digest}-{os.getpid()}.so"
        try:
            subprocess.run(
                ["gcc", "-O3", "-march=native", "-fPIC", "-shared", "-o", str(temporary), str(source), "-lm"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            temporary.unlink(missing_ok=True)
            _LIB_FAILED = True
            return None
        os.replace(temporary, so)
    lib = ctypes.CDLL(str(so))
    lib.lj_cluster.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.lj_cluster.restype = None
    _LIB = lib
    return lib


def cluster_lennard_jones(coords) -> tuple[float, np.ndarray]:
    """Spec Lennard-Jones energy and flat gradient.

    Uses the compiled kernel when gcc can build it. Otherwise the NumPy
    potential, which is the same formula.
    """
    pos = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1)
    if pos.size % 3 != 0:
        raise ValueError("coordinates must be a multiple of 3")
    lib = _kernel_library()
    if lib is None:
        return lennard_jones(pos)
    grad = np.empty(pos.shape, dtype=np.float64)
    energy = ctypes.c_double()
    lib.lj_cluster(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pos.size // 3,
        ctypes.byref(energy),
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return float(energy.value), grad


def wales_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "lj38_wales.txt"


def load_wales_coordinates(path: Path | None = None) -> np.ndarray:
    """Cartesian rows from the vendored Wales LJ38 minimum."""
    rows = []
    for line in (path or wales_path()).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
    coords = np.asarray(rows, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"{path or wales_path()} has no coordinates")
    return coords


def reference_energy(coords: np.ndarray | None = None) -> float:
    """Harness energy of the published minimum. The header value is rounded."""
    point = load_wales_coordinates() if coords is None else np.asarray(coords, dtype=np.float64)
    energy, _gradient = lennard_jones(point)
    return float(energy)


def reached_minimum(energy: float, reference: float, tolerance: float) -> bool:
    """True when ``energy`` is at or below the known minimum plus ``tolerance``."""
    return math.isfinite(energy) and energy <= reference + tolerance


def seed_words(start_index: int, structure_index: int) -> tuple[int, ...]:
    """Eight gfortran ``random_seed`` integers for one cell.

    buildcell reads ``#SEED=`` into an array of length 8. A different
    length makes every cell share one seed.
    """
    words = []
    for slot in range(8):
        value = (
            1664525 * (structure_index + 1)
            + 1013904223 * (start_index + 1)
            + 22695477 * (slot + 1)
        ) % 2147483647
        words.append(int(value) or 1)
    return tuple(words)


def find_buildcell() -> Path:
    """The AIRSS ``buildcell`` binary, from the environment or ``PATH``."""
    env = os.environ.get("AIRSS_BUILDCELL")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"AIRSS_BUILDCELL={env} is not an executable")
    found = shutil.which("buildcell")
    if found:
        return Path(found)
    home = Path.home() / "airss" / "bin" / "buildcell"
    if home.is_file() and os.access(home, os.X_OK):
        return home
    raise FileNotFoundError("buildcell is not on PATH and AIRSS_BUILDCELL is unset")


def cell_text(
    *,
    natoms: int,
    box: float,
    posamp: float,
    minsep: float,
    symmops: str,
    nform: int,
    seed: tuple[int, ...],
    element: str = "Al",
    composition: tuple[tuple[str, int], ...] | None = None,
    minsep_pairs: tuple[tuple[str, str, float], ...] | None = None,
) -> str:
    """AIRSS example 1.4, with the lengths passed in.

    Example 1.4 uses sigma 2. The spec potential uses sigma 1, so the
    caller passes lengths scaled by one half and a free cubic box.
    ``composition`` replaces the single ``NUM`` line when the cluster has
    more than one species.
    """
    if len(seed) != 8:
        raise ValueError("buildcell expects eight seed integers")
    counts = composition if composition else ((element, int(natoms)),)
    if sum(count for _label, count in counts) != int(natoms):
        raise ValueError("composition does not add up to natoms")
    side = int(box) if float(box).is_integer() else box
    seed_text = " ".join(str(int(word)) for word in seed)
    position_lines = [
        f"{label} 0 0 0 # {label}1 % NUM={int(count)}" for label, count in counts
    ]
    if minsep_pairs:
        pair_text = " ".join(f"{left}-{right}={gap}" for left, right, gap in minsep_pairs)
        minsep_line = f"#MINSEP={minsep} {pair_text}"
    else:
        minsep_line = f"#MINSEP={minsep}"
    return (
        "%BLOCK LATTICE_CART\n"
        f"{side} 0 0\n"
        f"0 {side} 0\n"
        f"0 0 {side}\n"
        "#FIX\n"
        "%ENDBLOCK LATTICE_CART\n"
        "%BLOCK POSITIONS_FRAC\n"
        + "\n".join(position_lines)
        + "\n%ENDBLOCK POSITIONS_FRAC\n"
        "FIX_ALL_CELL : true\n"
        f"{minsep_line}\n"
        "#CLUSTER\n"
        f"#POSAMP={posamp}\n"
        f"#SEED={seed_text}\n"
        f"#SYMMOPS={symmops}\n"
        f"#NFORM={int(nform)}\n"
    )


def _parsed_cell(text: str) -> tuple[np.ndarray, list[str]]:
    """Cartesian positions and species labels from a buildcell cell."""
    length = None
    in_abc = False
    in_pos = False
    coords: list[list[float]] = []
    labels: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("%BLOCK LATTICE_ABC"):
            in_abc = True
            continue
        if upper.startswith("%ENDBLOCK LATTICE_ABC"):
            in_abc = False
            continue
        if in_abc and length is None and stripped and not stripped.startswith("#"):
            length = float(stripped.split()[0])
            continue
        if upper.startswith("%BLOCK POSITIONS_FRAC"):
            in_pos = True
            continue
        if upper.startswith("%ENDBLOCK POSITIONS_FRAC"):
            in_pos = False
            continue
        if in_pos and stripped and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) < 4:
                raise BuildFailure("a position line has fewer than four fields")
            labels.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if length is None:
        raise BuildFailure("buildcell wrote no LATTICE_ABC length")
    if not coords:
        raise BuildFailure("buildcell wrote no positions")
    return np.asarray(coords, dtype=np.float64) * length, labels


def positions_from_cell(text: str, natoms: int) -> np.ndarray:
    """Cartesian positions from a buildcell ``LATTICE_ABC`` cell."""
    coords, _labels = _parsed_cell(text)
    if len(coords) != natoms:
        raise BuildFailure(f"buildcell wrote {len(coords)} atoms, expected {natoms}")
    return coords


def build_structure(
    binary: Path,
    text: str,
    natoms: int,
    *,
    timeout: float = 60.0,
) -> tuple[np.ndarray, list[str]]:
    """Run ``buildcell`` and return Cartesian coordinates and species labels."""
    try:
        proc = subprocess.run(
            [str(binary)],
            input=text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildFailure(exc.__class__.__name__) from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {proc.returncode}"
        raise BuildFailure(tail)
    coords, labels = _parsed_cell(proc.stdout)
    if len(coords) != natoms:
        raise BuildFailure(f"buildcell wrote {len(coords)} atoms, expected {natoms}")
    return coords, labels


def build_positions(binary: Path, text: str, natoms: int, *, timeout: float = 60.0) -> np.ndarray:
    """Run ``buildcell`` on ``text`` and return Cartesian coordinates."""
    coords, _labels = build_structure(binary, text, natoms, timeout=timeout)
    return coords


def _bb_pass(
    pos: np.ndarray,
    box: float,
    potential: _StepPotential,
    *,
    etol: float,
    maxsteps: int,
) -> tuple[np.ndarray, float, int, bool]:
    """One ``opt_tpsd`` pass. The step is ``abs(xg/gg)`` and the shift is the new cell image.

    ``g`` is the force, the negative of the harness gradient, with the
    mean removed. The returned energy is the last evaluated energy.
    The returned coordinates include the last step.
    """
    pos = np.array(pos, dtype=np.float64, copy=True)
    pos0 = pos.copy()
    g = np.ones_like(pos)
    h = 1.0
    dh = 1e300
    steps = 0
    converged = True
    while abs(dh) > etol:
        steps += 1
        if steps > maxsteps:
            converged = False
            break
        h0 = h
        g0 = g
        energy, grad = potential.energy_forces(pos.reshape(-1))
        g = -np.asarray(grad, dtype=np.float64).reshape(pos.shape)
        g = g - g.mean(axis=0)
        h = float(energy)
        delta_g = g - g0
        xg = float(np.sum((pos - pos0) * delta_g))
        gg = float(np.sum(delta_g * delta_g))
        if abs(xg) > 0.0:
            step = abs(xg / gg)
        else:
            glength = math.sqrt(float(np.sum(g * g)))
            if glength > 1e8:
                step = 1e-2 / math.sqrt(glength)
            else:
                step = 1e-8
        pos0 = pos.copy()
        pos = pos + step * g
        shift = np.floor(pos / box)
        pos0 = (pos0 / box - shift) * box
        pos = (pos / box - shift) * box
        dh = h - h0
    return pos, h, steps, converged


def relax_positions(
    pos: np.ndarray,
    box: float,
    *,
    etol: float = 1e-10,
    maxsteps: int = 9999,
    evaluate=None,
) -> tuple[np.ndarray, float, int, bool]:
    """Two pp3 passes on a counted harness potential.

    ``evaluate`` defaults to the unary Lennard-Jones potential. A binary
    cluster passes ``binary_lennard_jones`` closed over its species labels.
    """
    if evaluate is None:
        evaluate = cluster_lennard_jones
    potential = _StepPotential(evaluate)
    point = np.asarray(pos, dtype=np.float64)
    converged = True
    energy = 0.0
    for _pass in range(_PASSES):
        point, energy, _steps, ok = _bb_pass(point, box, potential, etol=etol, maxsteps=maxsteps)
        converged = converged and ok
    return point, float(energy), int(potential.calls), converged


def _config(template: dict) -> dict:
    section = template.get("airss") or {}
    required = ("natoms", "box", "posamp", "minsep", "symmops", "nform")
    missing = [key for key in required if section.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{template['id']} needs airss.{', airss.'.join(missing)}")
    config = {
        "natoms": int(section["natoms"]),
        "box": float(section["box"]),
        "posamp": float(section["posamp"]),
        "minsep": float(section["minsep"]),
        "symmops": str(section["symmops"]),
        "nform": int(section["nform"]),
        "etol": float(section.get("etol", 1e-10)),
        "maxsteps": int(section.get("maxsteps", 9999)),
    }
    if section.get("budget_structures") not in (None, ""):
        config["budget_structures"] = int(section["budget_structures"])
    if section.get("starts") not in (None, ""):
        config["starts"] = int(section["starts"])
        config["start_label"] = str(section.get("start_label") or "start")
    composition = section.get("composition")
    if composition:
        parsed = []
        for item in composition:
            if isinstance(item, str):
                label, count = item.split(":")
                parsed.append((label, int(count)))
            else:
                label, count = item
                parsed.append((str(label), int(count)))
        config["composition"] = tuple(parsed)
    pairs = section.get("minsep_pairs")
    if pairs:
        parsed_pairs = []
        for item in pairs:
            if isinstance(item, str):
                names, gap = item.rsplit(":", 1)
                left, right = names.split("-")
                parsed_pairs.append((left, right, float(gap)))
            else:
                left, right, gap = item
                parsed_pairs.append((str(left), str(right), float(gap)))
        config["minsep_pairs"] = tuple(parsed_pairs)
    if section.get("potential"):
        config["potential"] = str(section["potential"])
    return config


def _job(
    template: dict,
    index: int,
    start: str,
    binary: Path,
    reference: float | None,
    tolerance: float | None,
    cap: int | None,
) -> dict:
    config = _config(template)
    threshold = None if reference is None or tolerance is None else float(reference) + float(tolerance)
    job = {
        "index": index,
        "start": start,
        "buildcell": str(binary),
        "reference_energy": reference,
        "threshold": threshold,
        "cap": cap,
        "safety": SAFETY_STRUCTURES,
        "code": template.get("code", "AIRSS"),
        "code_version": template.get("code_version", ""),
        "commit": template.get("commit", ""),
        "tuning": template.get("tuning", ""),
        "natoms": config["natoms"],
        "box": config["box"],
        "posamp": config["posamp"],
        "minsep": config["minsep"],
        "symmops": config["symmops"],
        "nform": config["nform"],
        "etol": config["etol"],
        "maxsteps": config["maxsteps"],
        "potential": config.get("potential", "lennard-jones"),
    }
    if config.get("composition"):
        job["composition"] = config["composition"]
    if config.get("minsep_pairs"):
        job["minsep_pairs"] = config["minsep_pairs"]
    if config.get("budget_structures"):
        job["budget"] = int(config["budget_structures"])
    return job


def _crash(job: dict, calls: int, energy: float | None, structures: int, error: str) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "error": error,
        "energy": energy,
        "structures": structures,
        "reference_energy": job["reference_energy"],
        "converged": False,
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }


def _capped(job: dict, calls: int, energy: float, structures: int) -> dict:
    """A start that stopped at the structure budget. The problem states no success rule."""
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "cap",
        "reasons": ["cap"],
        "energy": energy,
        "structures": structures,
        "reference_energy": job.get("reference_energy"),
        "converged": False,
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }


def _evaluate(job: dict, labels: list[str]):
    """Harness energy and gradient for one built cell."""
    if job.get("potential") == "binary-lennard-jones":
        species = list(labels)

        def binary(coords):
            return binary_lennard_jones(np.asarray(coords, dtype=np.float64), species)

        return binary
    return cluster_lennard_jones


def _judged(job: dict, calls: int, energy: float | None, structures: int) -> dict:
    """Classify a finished start with the problem cap and the CCD window."""
    finite = energy is not None and math.isfinite(float(energy))
    judged = judge_globalopt(
        float(energy) if finite else None,
        calls,
        reference=float(job["reference_energy"]),
        tolerance=float(job["threshold"]) - float(job["reference_energy"]),
        cap=int(job["cap"]),
    )
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": judged["success"],
        "failure": judged["failure"],
        "reasons": judged["reasons"],
        "energy": judged["energy"],
        "structures": structures,
        "reference_energy": job["reference_energy"],
        "converged": bool(judged["success"]),
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }


def _checkpoint_file(job: dict) -> Path | None:
    raw = job.get("checkpoint")
    if not raw:
        return None
    return Path(raw)


def _read_checkpoint(path: Path) -> dict | None:
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def search_start(job: dict) -> dict:
    """Search one start until the energy window or the force-call cap.

    With no threshold and no cap, stop after ``budget`` structures and keep
    the lowest energy. A cap uses ``judge_globalopt``: success is an energy
    at or below the reference plus the tolerance, with fewer calls than the cap.
    A checkpoint file resumes at the next structure. The seed is a function
    of the structure index, so the continued search matches a run that was
    not interrupted.
    """
    binary = Path(job["buildcell"])
    calls = 0
    structures = 0
    build_failures = 0
    nonfinite = 0
    best = math.inf
    cap = job.get("cap")
    threshold = job.get("threshold")
    budget = job.get("budget")
    structure_index = 0
    checkpoint = _checkpoint_file(job)
    restored = _read_checkpoint(checkpoint) if checkpoint is not None else None
    if restored:
        structure_index = int(restored["structure_index"])
        calls = int(restored["calls"])
        structures = int(restored["structures"])
        nonfinite = int(restored["nonfinite"])
        best = math.inf if restored.get("best") is None else float(restored["best"])
        print(
            f"{job['start']} resume structures={structures} calls={calls}",
            flush=True,
        )

    def _persist() -> None:
        if checkpoint is None:
            return
        _write_checkpoint(
            checkpoint,
            {
                "structure_index": structure_index,
                "calls": calls,
                "structures": structures,
                "best": None if best is math.inf else best,
                "nonfinite": nonfinite,
            },
        )

    def _finish(record: dict) -> dict:
        if checkpoint is not None:
            checkpoint.unlink(missing_ok=True)
        return record

    while True:
        if cap is not None and calls >= int(cap):
            return _finish(_judged(job, calls, None if best is math.inf else best, structures))
        if budget is not None and structures >= int(budget):
            if best is math.inf:
                return _finish(_crash(job, calls, None, structures, "budget"))
            return _finish(_capped(job, calls, best, structures))
        if cap is None and budget is None and structures >= int(job["safety"]):
            return _finish(_crash(job, calls, None if best is math.inf else best, structures, "safety"))
        text = cell_text(
            natoms=job["natoms"],
            box=job["box"],
            posamp=job["posamp"],
            minsep=job["minsep"],
            symmops=job["symmops"],
            nform=job["nform"],
            seed=seed_words(job["index"], structure_index),
            composition=job.get("composition"),
            minsep_pairs=job.get("minsep_pairs"),
        )
        structure_index += 1
        try:
            pos, labels = build_structure(binary, text, job["natoms"])
        except BuildFailure:
            build_failures += 1
            if build_failures >= 30:
                return _finish(_crash(job, calls, None if best is math.inf else best, structures, "buildcell"))
            continue
        build_failures = 0
        structures += 1
        _point, energy, used, _ok = relax_positions(
            pos,
            job["box"],
            etol=job["etol"],
            maxsteps=job["maxsteps"],
            evaluate=_evaluate(job, labels),
        )
        calls += used
        if used <= 0:
            return _finish(_crash(job, calls, None if best is math.inf else best, structures, "relax"))
        if not math.isfinite(energy):
            nonfinite += 1
            if nonfinite >= 20:
                return _finish(_crash(job, calls, None if best is math.inf else best, structures, "nonfinite"))
            _persist()
            continue
        if energy < best:
            best = energy
        _persist()
        if structures % 100 == 0:
            print(
                f"{job['start']} structures={structures} calls={calls} best={best:.6f}",
                flush=True,
            )
        if threshold is not None and reached_minimum(
            energy,
            float(job["reference_energy"]),
            float(job["threshold"]) - float(job["reference_energy"]),
        ):
            if cap is None or calls < int(cap):
                return _finish(_success(job, calls, energy, structures))
        if cap is not None and calls >= int(cap):
            return _finish(_judged(job, calls, best, structures))


def _success(job: dict, calls: int, energy: float, structures: int) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": True,
        "failure": None,
        "reasons": [],
        "energy": energy,
        "structures": structures,
        "reference_energy": job["reference_energy"],
        "converged": True,
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }


def _load_start_names(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path) -> list[str]:
    import tarfile

    from optbench.campaign import geometry_archive, iter_geometries

    if geometries is None:
        archive = geometry_archive(root, problem)
        dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".extracted"
        recorded = archive.name
        if marker.is_file():
            previous = marker.read_text(encoding="utf-8").strip()
            if previous != recorded:
                raise RuntimeError(f"{dest} was extracted from {previous}, not {recorded}")
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text(recorded, encoding="utf-8")
        cons = sorted(path for path in dest.rglob("*.con") if path.is_file())
        numbered = []
        for path in dest.rglob("coords.*"):
            if not path.is_file():
                continue
            suffix = path.name.split(".", 1)[1]
            if suffix.isdigit():
                numbered.append((int(suffix), path))
        numbered.sort()
    else:
        cons = list(iter_geometries(geometries))
        numbered = []
    expected = (problem.get("sample") or {}).get("starts")
    if cons and (expected is None or len(cons) == int(expected)):
        # ``0.con`` is start ``0``. ``Path.stem`` drops that suffix only.
        return [path.stem for path in cons]
    if numbered and (expected is None or len(numbered) == int(expected)):
        # ``coords.12`` has stem ``coords``. The start name is the filename.
        return [path.name for _number, path in numbered]
    found = len(cons) or len(numbered)
    raise ValueError(f"{problem['id']} expected {expected} starts, found {found}")


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "structures_min": summary.get("structures_min"),
        "structures_max": summary.get("structures_max"),
        "partial": "true" if summary.get("partial") else None,
        "cap": summary.get("cap"),
    }
    if summary.get("partial"):
        lines = [
            "# counter harness. One force call is one potential evaluation inside the two-point step.",
            "# The problem states no energy window. Each start stops after the structure budget.",
            "# failure is cap. cap is the largest harness count in this file, so PAR-2 is twice that count.",
            "# force_calls is the half-even mean of those counts. energy_min and energy_max are the best energies.",
        ]
    elif summary.get("cap") is not None:
        lines = [
            "# counter harness. One force call is one potential evaluation inside the two-point step.",
            "# buildcell draws the cell. The archived files name the starts.",
            "# A start succeeds when the harness energy is at or below the Cambridge minimum plus energy_above_minimum and the call count is under the cap.",
            "# A miss costs twice the cap.",
        ]
    else:
        lines = [
            "# counter harness. One force call is one potential evaluation inside the two-point step.",
            "# buildcell draws the cell. The archived .con files name the starts.",
            "# A start succeeds when the harness energy is at or below the Cambridge minimum plus energy_above_minimum.",
            "# The problem states no force-call cap, so par2_mean is unset and the median is the call median.",
        ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def assemble_campaign(
    root: Path,
    problem: dict,
    template: dict,
    shards: list[Path],
    output: Path,
    *,
    work: Path | None = None,
) -> dict:
    """Score one campaign from shard ``.starts.jsonl`` files.

    ``shards`` are the ``.dat`` paths or the jsonl paths. The archived
    start list supplies the order. A partial set raises.
    """
    import json

    names = _load_start_names(root, problem, None, work, output)
    records = []
    for shard in shards:
        path = shard if shard.name.endswith(".jsonl") else shard.with_suffix(".starts.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    ordered = order_starts(names, records)
    config = _config(template)
    cap = search_cap(problem, template)
    raw_tolerance = (problem.get("convergence") or {}).get("energy_above_minimum")
    if raw_tolerance is None:
        reference = None
        tolerance = None
    else:
        tolerance = float(raw_tolerance)
        stated = (problem.get("reference") or {}).get("energy")
        reference = None if stated in (None, "") else float(stated)
    return _publish(
        problem,
        template,
        config,
        ordered,
        output,
        reference=reference,
        tolerance=tolerance,
        cap=cap,
        budget_only=reference is None and cap is None,
    )


def order_starts(names: list[str], records: list[dict]) -> list[dict]:
    """Put shard records into archived start order.

    A record may still carry ``index`` from a checkpoint. The written row
    does not. Every archived name is required once.
    """
    position = {name: index for index, name in enumerate(names)}
    found: dict[int, dict] = {}
    for record in records:
        start = str(record["start"])
        if start not in position:
            raise ValueError(f"unknown start {start}")
        index = position[start]
        if index in found:
            raise ValueError(f"duplicate start {start}")
        row = dict(record)
        row.pop("index", None)
        found[index] = row
    missing = [name for index, name in enumerate(names) if index not in found]
    if missing:
        raise ValueError(f"missing {len(missing)} starts, first {missing[0]}")
    return [found[index] for index in range(len(names))]


def _publish(
    problem: dict,
    template: dict,
    config: dict,
    records: list[dict],
    output: Path,
    *,
    reference: float | None,
    tolerance: float | None,
    cap: int | None,
    budget_only: bool,
) -> dict:
    from optbench.campaign import write_starts

    missed = [row for row in records if not row["success"]]
    calls = [int(row["force_calls"]) for row in records]
    energies = [float(row["energy"]) for row in records if row.get("energy") is not None]
    structures = [int(row["structures"]) for row in records]
    scored_cap = cap if cap is not None else (max(calls) if budget_only else None)
    summary = {
        "force_calls": half_even(sum(calls) / len(calls)),
        "force_calls_min": min(calls),
        "force_calls_max": max(calls),
        "nfailed": len(missed),
        "energy_min": min(energies) if energies else None,
        "energy_max": max(energies) if energies else None,
        "structures_min": min(structures),
        "structures_max": max(structures),
        "partial": budget_only,
        "cap": scored_cap,
    }
    scored = [
        {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
        for row in records
    ]
    metrics = score_v2(scored, scored_cap)

    def _jsonable(value):
        if isinstance(value, tuple):
            return [_jsonable(item) for item in value]
        return value

    passed = {key: _jsonable(value) for key, value in config.items()}
    if reference is not None:
        passed["reference_energy"] = reference
        passed["energy_above_minimum"] = tolerance
        if (problem.get("reference") or {}).get("page"):
            passed["reference_page"] = problem["reference"]["page"]
        else:
            passed["wales"] = WALES_URL
    if scored_cap is not None:
        passed["max_force_calls"] = scored_cap
    provenance = collect_provenance(
        code=str(template.get("code", "AIRSS")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "measured",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter on the spec potential. "
            "buildcell generates each cell. "
            + (
                "The problem states no energy window, so each start stops at the structure budget."
                if budget_only
                else (
                    "Success is the Cambridge minimum plus energy_above_minimum, under the force-call cap."
                    if scored_cap is not None
                    else "The problem states no force-call cap."
                )
            )
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code", "AIRSS"),
                "algorithm": template.get("algorithm", "random structure search"),
                "code_version": template.get("code_version", ""),
                "commit": template.get("commit", ""),
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": passed,
                "provenance": provenance,
                "metrics": metrics,
                "starts": records,
            }
        ],
    }
    import json

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["median"] = metrics["median"]
    summary["v2_nfailed"] = metrics["nfailed"]
    return summary


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    index_start: int = 0,
    index_stop: int | None = None,
) -> dict:
    from optbench.campaign import write_starts

    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    if template.get("commit") != TARBALL_SHA256:
        raise RuntimeError(f"{template['id']} commit is not the AIRSS 0.9.4 tarball sha256")
    convergence = problem.get("convergence") or {}
    config = _config(template)
    raw_tolerance = convergence.get("energy_above_minimum")
    cap = search_cap(problem, template)
    if raw_tolerance is None and cap is None and not config.get("budget_structures"):
        raise ValueError(
            f"{problem['id']} has no energy_above_minimum and {template['id']} has no airss.budget_structures"
        )
    binary = find_buildcell()
    if config.get("starts"):
        label = str(config.get("start_label") or "start")
        names = [f"{label}.{index}" for index in range(1, int(config["starts"]) + 1)]
    else:
        names = _load_start_names(root, problem, geometries, work, output)
    indexed = list(enumerate(names))
    stop = len(indexed) if index_stop is None else int(index_stop)
    if index_start < 0 or stop > len(indexed) or index_start >= stop:
        raise ValueError(f"{template['id']} index range {index_start}:{stop} is outside 0:{len(indexed)}")
    indexed = indexed[int(index_start) : stop]
    if raw_tolerance is None:
        reference = None
        tolerance = None
    else:
        tolerance = float(raw_tolerance)
        stated = (problem.get("reference") or {}).get("energy")
        if stated not in (None, ""):
            reference = float(stated)
        else:
            coords = load_wales_coordinates()
            if coords.shape[0] != config["natoms"]:
                raise ValueError(f"Wales file has {coords.shape[0]} atoms, template has {config['natoms']}")
            reference = reference_energy(coords)
    checkpoint_dir = output.parent / f"{output.name}.checkpoints"
    jobs = []
    for index, name in indexed:
        job = _job(template, index, name, binary, reference, tolerance, cap)
        job["checkpoint"] = str(checkpoint_dir / f"{index}.json")
        jobs.append(job)
    if _kernel_library() is None:
        print(f"{template['id']} Lennard-Jones kernel: numpy", flush=True)
    else:
        print(f"{template['id']} Lennard-Jones kernel: compiled", flush=True)
    import json

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    progress_path = output.with_suffix(".progress.jsonl")
    saved: dict[int, dict] = {}
    if progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            saved[int(row["index"])] = row
    pending = [job for job in jobs if job["index"] not in saved]
    progress_file = progress_path.open("a", encoding="utf-8")

    def _store(record: dict) -> None:
        saved[int(record["index"])] = record
        progress_file.write(json.dumps(record, sort_keys=True) + "\n")
        progress_file.flush()

    total = len(jobs)
    try:
        if not pending:
            pass
        elif workers <= 1:
            done = total - len(pending)
            for job in pending:
                record = search_start(job)
                _store(record)
                done += 1
                print(
                    f"{template['id']} {done}/{total} start={record['start']} "
                    f"calls={record['force_calls']} energy={record['energy']} success={record['success']}",
                    flush=True,
                )
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(search_start, job) for job in pending]
                done = total - len(pending)
                for future in as_completed(futures):
                    record = future.result()
                    _store(record)
                    done += 1
                    print(
                        f"{template['id']} {done}/{total} start={record['start']} "
                        f"calls={record['force_calls']} energy={record['energy']} success={record['success']}",
                        flush=True,
                    )
    finally:
        progress_file.close()
    records = [saved[job["index"]] for job in jobs]
    budget_only = reference is None and cap is None
    crashes = [row for row in records if row.get("failure") == "crash"]
    if crashes:
        partial_path = output.with_suffix(".partial.jsonl")
        for row in records:
            row.pop("index", None)
        write_starts(partial_path, records)
        raise RuntimeError(f"{len(crashes)} of {len(records)} starts crashed; wrote {partial_path}")
    progress_path.unlink(missing_ok=True)
    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
    for row in records:
        row.pop("index", None)
    return _publish(
        problem,
        template,
        config,
        records,
        output,
        reference=reference,
        tolerance=tolerance,
        cap=cap,
        budget_only=budget_only,
    )
