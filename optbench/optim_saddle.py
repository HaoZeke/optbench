"""OPTIM hybrid eigenvector-following on the Pt island.

One search displaces a single listed atom by 0.1 Å, climbs with the
stock ``BFGSTS`` block, and quenches both endpoints. The socket
potential is ``morse/smooth/linear`` with the problem's periodic axes.
``EV`` and ``ANG`` scales are 1. OPTIM's RMS stop is the Euclidean
force tolerance divided by ``sqrt(3N)``, because frozen atoms stay in
the coordinate vector and their gradient components are zero. The
harness then checks the free-atom Euclidean norm itself.

The climb budget is the template iteration limit. Each endpoint quench
has its own budget. The initial quench is outside the search count.
A search that spends the climb budget is a cap. Hessian finite
differences are verification and are not part of that count.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .engines.optim_search import COMMIT, _RELAY, _fortran_float, _read_points, _write_xyz, binary_path
from .failures import classify_failure
from .rgsaddle_minmode import HarnessSurface, _bind, _hessian_spectrum, _norm, _stored, _unit
from .score import score_v2
from .stationary import negative_index
from .surfaces import potential_from_problem, read_con
from .v2 import collect_provenance, write_starts

KIND = "morse_smooth"
ENDPOINT_BUDGET = 2000
CLIMB_OVERSHOOT = 64
INITIAL_BUDGET = 5000
HANG_S = 900.0
SEARCH_SEED = 20261015


def rms_stop(force_tol: float, natoms: int) -> float:
    """OPTIM RMS that matches a Euclidean free-atom norm of ``force_tol``.

    Frozen gradient components are cleared and the coordinate count stays
    ``3N``, so the root-mean-square is the free-atom norm over ``sqrt(3N)``.
    """
    count = 3 * int(natoms)
    if count < 1:
        raise ValueError("a geometry needs one coordinate")
    return float(force_tol) / math.sqrt(count)


def draw_island_kick(
    rng: np.random.Generator,
    listed: list[int],
    free: np.ndarray,
    magnitude: float,
    minimum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """One listed atom, one unit Gaussian, shared with the other full rows.

    ``minimum`` and the returned point are free-atom coordinates. The
    kick is the unit direction in that same layout.
    """
    atom = int(listed[int(rng.integers(0, len(listed)))])
    slots = np.flatnonzero(np.asarray(free) == atom)
    if len(slots) != 1:
        raise RuntimeError(f"displace atom {atom} is not one free atom")
    slot = int(slots[0])
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    point = np.array(minimum, dtype=np.float64, copy=True)
    point[3 * slot : 3 * slot + 3] += float(magnitude) * direction
    kick = np.zeros_like(point)
    kick[3 * slot : 3 * slot + 3] = direction
    return point, kick, atom


def climb_reasons(
    *,
    hang: bool,
    error: str | None,
    finite: bool,
    capped: bool,
    calls: int,
    climb_cap: int,
    force_norm: float | None,
    force_tol: float,
    index: int | None,
) -> list[str] | None:
    """Reasons that stop a search before the endpoint quenches.

    ``None`` means the climb is finite, inside the climb budget, at the
    Euclidean force stop, and Hessian index 1.
    """
    if hang:
        return ["crash"]
    if capped or int(calls) >= int(climb_cap):
        return ["cap", "force_norm"]
    if error:
        return ["crash"]
    if not finite:
        return ["divergence"]
    if force_norm is None or not math.isfinite(float(force_norm)) or float(force_norm) > float(force_tol) + 1e-10:
        return ["force_norm"]
    if index != 1:
        return ["hessian_index"]
    return None


def link_reasons(*, linked: bool, novel: bool, barrier: float, barrier_max: float) -> list[str]:
    """Empty when exactly one endpoint links, the saddle is new, and the barrier is inside the goal."""
    reasons: list[str] = []
    if not linked:
        reasons.append("not_connected")
    if not novel:
        reasons.append("duplicate")
    if not math.isfinite(barrier) or float(barrier) >= float(barrier_max):
        reasons.append("barrier")
    return reasons


def _ase_line(python: str, hook: Path) -> str:
    return f"ASE geom.xyz {python} {hook} dummy dummy cpu FIXCELL PBC EV ANG"


def _freeze_lines(fixed: np.ndarray) -> list[str]:
    """One 1-based ``FREEZE`` line per frozen atom. A bare ``FREEZE`` aborts."""
    return [f"FREEZE {index}" for index in (int(atom) + 1 for atom, flag in enumerate(np.asarray(fixed, dtype=bool)) if flag)]


def climb_odata(
    *,
    python: str,
    hook: Path,
    section: dict,
    gmax: float,
    steps: int,
    seed: int,
    freeze: list[str],
) -> str:
    """Stock hybrid-EF block. Scales stay in eV and Å."""
    lines = [
        _ase_line(python, hook),
        "BFGSTS {nevs} {nbfgsmax1} {nbfgsmax2} {ceig}".format(
            nevs=int(section["nevs"]),
            nbfgsmax1=int(section["nbfgsmax1"]),
            nbfgsmax2=int(section["nbfgsmax2"]),
            ceig=_fortran_float(float(section["ceig"])),
        ),
        "NOHESS",
        f"USEDIAG {int(section['usediag'])}",
        f"STEPS {int(steps)}",
        f"BFGSSTEPS {int(steps)}",
        f"CONVERGE {_fortran_float(float(section['convu']))} {_fortran_float(gmax)}",
        f"BFGSCONV {_fortran_float(gmax)}",
        f"MAXSTEP {_fortran_float(float(section['maxstep']))}",
        f"MAXMAX {_fortran_float(float(section['maxmax']))}",
        f"MAXBFGS {_fortran_float(float(section['maxbfgs']))}",
        f"UPDATES {int(section['updates'])} {int(section['updates'])}",
        "MAXERISE {rise} {fraction}".format(
            rise=_fortran_float(float(section["maxerise"])),
            fraction=_fortran_float(float(section["xmaxerise"])),
        ),
        f"BFGSTSPC {_fortran_float(float(section['bfgstspc']))}",
        f"RANSEED {int(seed) % 10000}",
        "READVEC",
        "DUMPVECTOR",
        *freeze,
    ]
    return "\n".join(lines) + "\n"


def minimize_odata(*, python: str, hook: Path, gmax: float, steps: int, seed: int, freeze: list[str]) -> str:
    """Reactant and endpoint quench. The RMS token is the problem stop."""
    lines = [
        _ase_line(python, hook),
        f"BFGSMIN {_fortran_float(gmax)}",
        f"STEPS {int(steps)}",
        f"BFGSSTEPS {int(steps)}",
        "UPDATES 4",
        f"MAXSTEP {_fortran_float(0.2)}",
        f"MAXMAX {_fortran_float(0.2)}",
        f"MAXBFGS {_fortran_float(0.2)}",
        f"RANSEED {int(seed) % 10000}",
        *freeze,
    ]
    return "\n".join(lines) + "\n"


def write_vector(path: Path, values: np.ndarray, *, eigenvalue: float = 0.0) -> None:
    """Eigenvalue, then the full Cartesian vector in groups of three."""
    body = [f"{float(eigenvalue):.16e}"]
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    for offset in range(0, len(flat), 3):
        chunk = flat[offset : offset + 3]
        body.append(" ".join(f"{float(item):.16e}" for item in chunk))
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def read_vector_dump(path: Path, nopt: int) -> np.ndarray | None:
    """Last complete ``vector.dump`` record. A trailing fragment is ignored."""
    if not path.is_file():
        return None
    try:
        numbers = [float(item.replace("D", "E").replace("d", "e")) for item in path.read_text(encoding="utf-8", errors="replace").split()]
    except ValueError:
        return None
    width = 1 + int(nopt)
    complete = len(numbers) // width
    if complete < 1:
        return None
    start = (complete - 1) * width
    return np.asarray(numbers[start + 1 : start + width], dtype=np.float64)


def _cartesian_rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
    delta = np.asarray(moved, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _full(reference: np.ndarray, free: np.ndarray, flat: np.ndarray) -> np.ndarray:
    moved = np.array(reference, dtype=np.float64, copy=True)
    moved[free] = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
    return moved


def _mode_vector(natoms: int, free: np.ndarray, kick: np.ndarray) -> np.ndarray:
    mode = np.zeros((natoms, 3), dtype=np.float64)
    mode[free] = np.asarray(kick, dtype=np.float64).reshape(-1, 3)
    return mode.reshape(-1)


def _rng_state(rng: np.random.Generator) -> dict:
    return json.loads(json.dumps(rng.bit_generator.state))


def _rng_restore(payload: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = payload
    return rng


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")
        handle.flush()


def _count_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _install_hook(work: Path) -> Path:
    hook = work / "hook"
    relay = hook / "pysocket" / "fmlip_relay.py"
    relay.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_RELAY, relay)
    return hook


def _geometry(folder: Path, natoms: int) -> np.ndarray | None:
    point = _read_points(folder / "last.txt", natoms)
    if point is not None:
        return point
    path = folder / "points.final"
    if not path.is_file():
        return None
    numbers = []
    for token in path.read_text(encoding="utf-8", errors="replace").replace("D", "E").replace("d", "e").split():
        try:
            numbers.append(float(token))
        except ValueError:
            return None
    width = 3 * natoms
    if len(numbers) < width:
        return None
    return np.asarray(numbers[:width], dtype=np.float64).reshape(natoms, 3)


def _launch(spec: dict) -> dict:
    """One OPTIM process. ``calls`` is the socket counter.

    OPTIM starts the relay with its own directory as the working
    directory and joins a relative script path onto that directory.
    The hook path has to be absolute.
    """
    work = Path(spec["work"]).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(spec["positions"], dtype=np.float64)
    reference = np.asarray(spec["reference"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    natoms = int(len(positions))
    cap = int(spec["cap"])
    method = str(spec["method"])
    hook = _install_hook(work)
    _write_xyz(work / "geom.xyz", positions, cell)
    if method != "minimize" and spec.get("vector") is not None:
        write_vector(work / "vector.dump", np.asarray(spec["vector"], dtype=np.float64))
    if method == "minimize":
        text = minimize_odata(
            python=str(spec["python"]),
            hook=hook,
            gmax=float(spec["gmax"]),
            steps=cap,
            seed=int(spec["seed"]),
            freeze=list(spec["freeze"]),
        )
    else:
        text = climb_odata(
            python=str(spec["python"]),
            hook=hook,
            section=dict(spec["section"]),
            gmax=float(spec["gmax"]),
            steps=cap,
            seed=int(spec["seed"]),
            freeze=list(spec["freeze"]),
        )
    (work / "odata").write_text(text, encoding="utf-8")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OPTBENCH_KIND"] = KIND
    env["OPTBENCH_CAP"] = str(cap)
    env["OPTBENCH_COUNT_PATH"] = str(work / "calls.txt")
    env["OPTBENCH_LAST_PATH"] = str(work / "last.txt")
    env["OPTBENCH_CAPPED_PATH"] = str(work / "capped.txt")
    env["OPTBENCH_E_SCALE"] = "1"
    env["OPTBENCH_L_SCALE"] = "1"
    env["OPTBENCH_BOX"] = " ".join(f"{float(value):.16g}" for value in np.asarray(cell, dtype=np.float64).reshape(-1)[:3])
    env["OPTBENCH_PBC"] = " ".join("1" if flag else "0" for flag in spec["pbc"])
    env["OPTBENCH_MORSE"] = " ".join(str(float(value)) for value in spec["morse"])
    root = str(spec["repo"])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    log_path = work / "stdout.txt"
    hang = False
    returncode = None
    log = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [str(spec["binary"])],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        last_calls = -1
        last_change = time.monotonic()
        while True:
            returncode = proc.poll()
            calls_now = _count_file(work / "calls.txt")
            if calls_now is not None and calls_now > last_calls:
                last_calls = calls_now
                last_change = time.monotonic()
            if returncode is not None:
                break
            if time.monotonic() - last_change >= HANG_S:
                hang = True
                _kill_group(proc)
                returncode = proc.poll()
                break
            time.sleep(0.05)
    finally:
        log.close()
    calls = _count_file(work / "calls.txt") or 0
    capped = (work / "capped.txt").is_file() or calls >= cap
    coords = None
    vector = None
    log_text = ""
    try:
        if not hang:
            point = _geometry(work, natoms)
            if point is not None and np.all(np.isfinite(point)):
                coords = np.array(point, dtype=np.float64, copy=True)
                coords[fixed] = reference[fixed]
            if method != "minimize":
                vector = read_vector_dump(work / "vector.dump", 3 * natoms)
        if log_path.is_file():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
    finally:
        if os.environ.get("OPTBENCH_OPTIM_KEEP") != "1":
            shutil.rmtree(work, ignore_errors=True)
    error = None
    if hang:
        error = "no new force calls"
    elif coords is None and not capped:
        detail = "\n".join(log_text.splitlines()[-30:])
        error = f"exit {returncode}: {detail}"
    return {
        "calls": int(calls),
        "capped": bool(capped),
        "hang": bool(hang),
        "returncode": None if returncode is None else int(returncode),
        "coords": None if coords is None else coords.tolist(),
        "vector": None if vector is None else vector.tolist(),
        "error": error,
    }


def _surface(problem: dict, reference: np.ndarray, free: np.ndarray, cell: np.ndarray) -> HarnessSurface:
    return HarnessSurface(_bind(potential_from_problem(problem), reference, free, cell))


def _harness_force(problem: dict, reference: np.ndarray, free: np.ndarray, cell: np.ndarray, coords: np.ndarray) -> tuple[float, float]:
    surface = _surface(problem, reference, free, cell)
    energy, gradient = surface.energy_forces(np.asarray(coords, dtype=np.float64).reshape(-1, 3)[free].reshape(-1))
    return float(energy), _norm(gradient)


def _hessian(problem: dict, reference: np.ndarray, free: np.ndarray, cell: np.ndarray, coords: np.ndarray) -> tuple[int, float, int]:
    surface = _surface(problem, reference, free, cell)
    point = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[free].reshape(-1)
    eigenvalues, _vectors = _hessian_spectrum(surface, point)
    index = negative_index(list(eigenvalues), 1e-6)
    lowest = float(eigenvalues[0]) if eigenvalues else math.nan
    return int(index), lowest, int(surface.verify_calls)


def _axis(vector: list | None, fixed: np.ndarray, kick: np.ndarray) -> np.ndarray:
    if vector is not None:
        values = np.asarray(vector, dtype=np.float64)
        if values.size == 3 * len(fixed) and np.all(np.isfinite(values)):
            free_vec = values.reshape(len(fixed), 3)[~np.asarray(fixed, dtype=bool)].reshape(-1)
            if float(np.linalg.norm(free_vec)) > 1e-12:
                return _unit(free_vec)
    return _unit(np.asarray(kick, dtype=np.float64))


def _outcome(reasons: list[str]) -> tuple[bool, str | None]:
    failure = classify_failure(reasons)
    return failure is None, failure


def _morse_tuple(problem: dict) -> tuple[float, float, float, float]:
    spec = problem.get("potential") or {}
    if str(spec.get("pair_style", "")) != "morse/smooth/linear":
        raise RuntimeError(f"{problem['id']} potential is not morse/smooth/linear")
    return (float(spec["D"]), float(spec["alpha"]), float(spec["r0"]), float(spec.get("cutoff", 9.5)))


def _pbc_flags(problem: dict) -> tuple[bool, bool, bool]:
    raw = (problem.get("potential") or {}).get("pbc")
    if not isinstance(raw, list) or len(raw) != 3:
        raise RuntimeError(f"{problem['id']} needs three periodic flags")
    return (bool(raw[0]), bool(raw[1]), bool(raw[2]))


def _base_spec(common: dict, positions: np.ndarray, cap: int, method: str, vector: np.ndarray | None) -> dict:
    spec = dict(common)
    spec["positions"] = np.asarray(positions, dtype=np.float64)
    spec["cap"] = int(cap)
    spec["method"] = method
    spec["vector"] = None if vector is None else np.asarray(vector, dtype=np.float64)
    return spec


def _public(record: dict) -> dict:
    hidden = {"saddle_x", "rng_state", "kick"}
    return {key: value for key, value in record.items() if key not in hidden}


def run_saddle(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
    work: Path | None = None,
) -> dict:
    """Run the stock hybrid-EF search and write the campaign files."""
    del resume
    root = Path(root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if template.get("id", "").endswith("-tuned") or str(template.get("tuning", "")) == "tuned":
        raise RuntimeError(f"{template['id']} is not a ranked search")
    binary = binary_path()
    if not binary.is_file():
        raise RuntimeError(f"OPTIM binary is not present at {binary}. The pin is SVNREV {COMMIT}.")
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = read_con(_stored(root, con_path).read_text(encoding="utf-8"))
    free = np.flatnonzero(~np.asarray(fixed, dtype=bool))
    search = (template.get("ini") or {}).get("Search") or {}
    section = (template.get("ini") or {}).get("Optim") or {}
    if search.get("displace_radius") not in (None, "", 0, 0.0):
        raise RuntimeError(f"{template['id']} sets a displacement radius; this problem moves one listed atom")
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if str(part).strip()]
    magnitude = float(search.get("displace_magnitude", 0.1))
    connect_rmsd = float(search.get("rmsd_connect", 0.5))
    max_searches = int(search.get("max_searches", 20000))
    if int(limit) > 0:
        max_searches = min(max_searches, int(limit))
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    force_tol = float((problem.get("convergence") or {})["force_norm"])
    climb_cap = int(((template.get("ini") or {}).get("Optimizer") or {}).get("max_iterations", 1000))
    search_cap = climb_cap + CLIMB_OVERSHOOT + 2 * ENDPOINT_BUDGET
    gmax = rms_stop(force_tol, len(positions))
    morse = _morse_tuple(problem)
    pbc = _pbc_flags(problem)
    seed = int(section.get("seed", 2014))
    freeze = _freeze_lines(fixed)
    slug = str(template["id"]).replace("/", "-")
    folder = Path(work).resolve() if work is not None else (output.parent / "work" / slug).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    header_path = folder / "header.json"
    checkpoint_path = folder / "checkpoint.jsonl"
    batch_path = folder / "batch.json"
    header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
    if header and header.get("template") == template["id"] and header.get("potential") == KIND:
        reference = np.asarray(header["reference"], dtype=np.float64)
        minimum = np.asarray(header["minimum"], dtype=np.float64)
        min_energy = float(header["min_energy"])
        minimize_calls = int(header["minimize_calls"])
        min_ok = bool(header["min_ok"])
        rng = _rng_restore(header["rng_state"])
    else:
        reference, min_energy, minimize_calls, min_ok = _minimize_reactant(
            root,
            problem,
            positions,
            fixed,
            free,
            cell,
            gmax,
            seed,
            freeze,
            morse,
            pbc,
            force_tol,
            binary,
            folder / "reactant",
        )
        minimum = reference[free].reshape(-1).copy()
        rng = np.random.default_rng(SEARCH_SEED + 1)
        header = {
            "template": template["id"],
            "potential": KIND,
            "reference": reference.tolist(),
            "minimum": minimum.tolist(),
            "min_energy": min_energy,
            "minimize_calls": minimize_calls,
            "min_ok": bool(min_ok),
            "rng_state": _rng_state(rng),
        }
        _write_json(header_path, header)
        checkpoint_path.unlink(missing_ok=True)
        batch_path.unlink(missing_ok=True)
        print(
            f"{template['id']} reactant calls {minimize_calls} force checked converged {int(min_ok)}",
            flush=True,
        )
    saved = _load_jsonl(checkpoint_path)
    found = [np.asarray(row["saddle_x"], dtype=np.float64) for row in saved if row.get("saddle_x") is not None]
    connected = sum(1 for row in saved if row.get("success"))
    common = {
        "reference": reference,
        "fixed": np.asarray(fixed, dtype=bool),
        "cell": np.asarray(cell, dtype=np.float64),
        "gmax": gmax,
        "seed": seed,
        "freeze": freeze,
        "section": section,
        "pbc": pbc,
        "morse": morse,
        "python": sys.executable,
        "binary": str(Path(binary).resolve()),
        "repo": str(root),
    }
    while connected < goal and len(saved) < max_searches:
        batch = json.loads(batch_path.read_text(encoding="utf-8")) if batch_path.is_file() else None
        done = {row["start"] for row in saved}
        if batch and batch.get("template") == template["id"]:
            jobs = [job for job in batch["jobs"] if job["name"] not in done]
            rng_after = batch["rng_after"]
        else:
            jobs = []
            room = min(max(1, int(workers)), max_searches - len(saved))
            for _ in range(room):
                point, kick, atom = draw_island_kick(rng, listed, free, magnitude, minimum)
                name = f"rep1-search{len(saved) + len(jobs) + 1}"
                jobs.append(
                    {
                        "name": name,
                        "atom": int(atom),
                        "positions": _full(reference, free, point).tolist(),
                        "kick": kick.tolist(),
                    }
                )
            rng_after = _rng_state(rng)
            batch = {"template": template["id"], "jobs": jobs, "rng_after": rng_after}
            _write_json(batch_path, batch)
            jobs = [job for job in jobs if job["name"] not in done]
        if not jobs:
            header["rng_state"] = rng_after
            _write_json(header_path, header)
            batch_path.unlink(missing_ok=True)
            rng = _rng_restore(rng_after)
            continue
        _run_window(
            jobs,
            common,
            folder,
            problem,
            reference,
            free,
            cell,
            fixed,
            force_tol,
            climb_cap,
            connect_rmsd,
            barrier_max,
            min_energy,
            found,
            checkpoint_path,
            workers,
        )
        saved = _load_jsonl(checkpoint_path)
        found = [np.asarray(row["saddle_x"], dtype=np.float64) for row in saved if row.get("saddle_x") is not None]
        connected = sum(1 for row in saved if row.get("success"))
        header["rng_state"] = rng_after
        _write_json(header_path, header)
        batch_path.unlink(missing_ok=True)
        rng = _rng_restore(rng_after)
        print(
            f"{template['id']} searches {len(saved)} connected {connected} cap {search_cap}",
            flush=True,
        )
    saved = _until_goal(saved, goal)
    found = [np.asarray(row["saddle_x"], dtype=np.float64) for row in saved if row.get("saddle_x") is not None]
    connected = sum(1 for row in saved if row.get("success"))
    return _finish(
        output,
        saved,
        template,
        problem,
        search_cap,
        goal,
        minimize_calls,
        min_ok,
        connected,
        found,
        header_path,
        checkpoint_path,
    )


def _until_goal(saved: list[dict], goal: int) -> list[dict]:
    """Keep the search that reaches the goal and drop later searches in that window."""
    kept = []
    linked = 0
    for row in saved:
        if linked >= goal:
            break
        kept.append(row)
        if row.get("success"):
            linked += 1
    return kept


def _minimize_reactant(
    root,
    problem,
    positions,
    fixed,
    free,
    cell,
    gmax,
    seed,
    freeze,
    morse,
    pbc,
    force_tol,
    binary,
    folder,
) -> tuple[np.ndarray, float, int, bool]:
    spec = {
        "work": str(folder),
        "positions": positions,
        "reference": positions,
        "fixed": np.asarray(fixed, dtype=bool),
        "cell": np.asarray(cell, dtype=np.float64),
        "gmax": gmax,
        "seed": seed,
        "freeze": freeze,
        "section": {},
        "pbc": pbc,
        "morse": morse,
        "python": sys.executable,
        "binary": str(Path(binary).resolve()),
        "repo": str(root),
        "cap": INITIAL_BUDGET,
        "method": "minimize",
        "vector": None,
    }
    result = _launch(spec)
    if result["coords"] is None:
        raise RuntimeError(f"reactant quench failed: {result['error']}")
    coords = np.asarray(result["coords"], dtype=np.float64)
    energy, force = _harness_force(problem, positions, free, cell, coords)
    return coords, energy, int(result["calls"]), bool(force <= force_tol + 1e-10)


def _run_window(
    jobs,
    common,
    folder,
    problem,
    reference,
    free,
    cell,
    fixed,
    force_tol,
    climb_cap,
    connect_rmsd,
    barrier_max,
    min_energy,
    found,
    checkpoint_path,
    workers,
) -> None:
    climb_specs = []
    for job in jobs:
        spec = _base_spec(common, np.asarray(job["positions"], dtype=np.float64), climb_cap, "climb", _mode_vector(len(fixed), free, np.asarray(job["kick"])))
        spec["work"] = str(folder / job["name"] / "climb")
        climb_specs.append(spec)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        climbs = list(pool.map(_launch, climb_specs))
    pending = []
    endpoint_specs = []
    judged = []
    for job, climb in zip(jobs, climbs):
        coords = None if climb["coords"] is None else np.asarray(climb["coords"], dtype=np.float64)
        finite = coords is not None and np.all(np.isfinite(coords))
        energy = None
        force = None
        if finite and not climb["hang"] and not (climb["capped"] or int(climb["calls"]) >= climb_cap):
            energy, force = _harness_force(problem, reference, free, cell, coords)
        early = climb_reasons(
            hang=bool(climb["hang"]),
            error=climb["error"] if not finite else None,
            finite=finite,
            capped=bool(climb["capped"]),
            calls=int(climb["calls"]),
            climb_cap=climb_cap,
            force_norm=force,
            force_tol=force_tol,
            index=1,
        )
        # Index is filled in only after the force stop. A missing index is not a pass.
        if early is None:
            index, lowest, verify_calls = _hessian(problem, reference, free, cell, coords)
            early = climb_reasons(
                hang=False,
                error=None,
                finite=True,
                capped=False,
                calls=int(climb["calls"]),
                climb_cap=climb_cap,
                force_norm=force,
                force_tol=force_tol,
                index=index,
            )
        else:
            index, lowest, verify_calls = None, None, 0
        item = {
            "job": job,
            "climb": climb,
            "coords": coords,
            "energy": energy,
            "force": force,
            "reasons": early,
            "index": index,
            "lowest": lowest,
            "verify_calls": verify_calls,
            "plus": None,
            "minus": None,
        }
        judged.append(item)
        if early is None and coords is not None:
            axis = _axis(climb["vector"], np.asarray(fixed, dtype=bool), np.asarray(job["kick"], dtype=np.float64))
            saddle = coords[free].reshape(-1)
            for label, sign in (("plus", 1.0), ("minus", -1.0)):
                moved = _full(reference, free, saddle + sign * 0.1 * axis)
                spec = _base_spec(common, moved, ENDPOINT_BUDGET, "minimize", None)
                spec["work"] = str(folder / job["name"] / label)
                endpoint_specs.append(spec)
                pending.append((item, label))
    if endpoint_specs:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            quenched = list(pool.map(_launch, endpoint_specs))
        for (item, label), result in zip(pending, quenched):
            item[label] = result
    for item in judged:
        record = _record(item, reference, free, found, connect_rmsd, barrier_max, min_energy, force_tol)
        _append_jsonl(checkpoint_path, record)
        if record.get("saddle_x") is not None:
            found.append(np.asarray(record["saddle_x"], dtype=np.float64))


def _record(item, reference, free, found, connect_rmsd, barrier_max, min_energy, force_tol) -> dict:
    job = item["job"]
    climb = item["climb"]
    reasons = list(item["reasons"] or [])
    endpoint_calls = 0
    barrier = None
    plus_rmsd = None
    minus_rmsd = None
    saved_point = None
    saddle = None if item["coords"] is None else np.asarray(item["coords"], dtype=np.float64)[free].reshape(-1)
    if item["reasons"] is None and saddle is not None and item["plus"] is not None and item["minus"] is not None:
        plus_calls = int(item["plus"]["calls"])
        minus_calls = int(item["minus"]["calls"])
        endpoint_calls = plus_calls + minus_calls
        plus_coords = None if item["plus"]["coords"] is None else np.asarray(item["plus"]["coords"], dtype=np.float64)
        minus_coords = None if item["minus"]["coords"] is None else np.asarray(item["minus"]["coords"], dtype=np.float64)
        if plus_coords is None or minus_coords is None:
            reasons = ["crash"]
        else:
            plus_free = plus_coords[free].reshape(-1, 3)
            minus_free = minus_coords[free].reshape(-1, 3)
            minimum = np.asarray(reference, dtype=np.float64)[free].reshape(-1, 3)
            plus_rmsd = _cartesian_rmsd(plus_free, minimum)
            minus_rmsd = _cartesian_rmsd(minus_free, minimum)
            linked = (plus_rmsd <= connect_rmsd) ^ (minus_rmsd <= connect_rmsd)
            novel = all(_cartesian_rmsd(saddle.reshape(-1, 3), old.reshape(-1, 3)) > connect_rmsd for old in found)
            barrier = float(item["energy"] - float(min_energy))
            reasons = link_reasons(linked=linked, novel=novel, barrier=barrier, barrier_max=barrier_max)
            if novel:
                saved_point = saddle
    success, failure = _outcome(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": job["name"],
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "force_calls": int(climb["calls"]) + int(endpoint_calls),
        "climb_calls": int(climb["calls"]),
        "endpoint_calls": int(endpoint_calls),
        "converged": success or (item["force"] is not None and item["force"] <= force_tol + 1e-10 and item["index"] == 1),
        "hessian_index": item["index"],
        "lowest_eigenvalue": item["lowest"],
        "verify_calls": int(item["verify_calls"]),
        "force_norm": item["force"],
        "energy": item["energy"],
        "barrier": barrier,
        "rmsd_plus": plus_rmsd,
        "rmsd_minus": minus_rmsd,
        "displace_atom": int(job["atom"]),
    }
    if saved_point is not None:
        record["saddle_x"] = saved_point.tolist()
    return record


def _finish(output, saved, template, problem, search_cap, goal, minimize_calls, min_ok, connected, found, header_path, checkpoint_path) -> dict:
    records = [_public(row) for row in saved]
    over = [row["start"] for row in records if row.get("success") and int(row["force_calls"]) >= int(search_cap)]
    if over:
        raise RuntimeError(f"{template['id']} success {over[0]} spent {search_cap} or more force calls")
    metrics = score_v2(
        [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records],
        search_cap,
    ) if records else None
    fraction = connected / len(records) if records else 0.0
    unique = len(found)
    lines = [
        f"# rep 1 force_calls {sum(int(row['force_calls']) for row in records)} unique_saddles {unique} "
        f"fraction_connected {fraction} total_searches {len(records)} connected {connected} "
        f"minimize_calls {minimize_calls} minimum_converged {int(min_ok)} seed {SEARCH_SEED + 1}",
    ]
    if records:
        total = sum(int(row["force_calls"]) for row in records)
        lines.append(f"force_calls {float(total)}")
        lines.append(f"force_calls_min {total}")
        lines.append(f"force_calls_max {total}")
    lines.extend(
        [
            f"code {template.get('code', 'optim')}",
            f"code_version {template.get('code_version', '')}",
            f"commit {template.get('commit', '')}",
            f"algorithm {template.get('algorithm', '')}",
            f"tuning {template.get('tuning', '')}",
            f"template {template['id']}",
            f"problem {problem['id']}",
            "counter harness",
            "repetitions 1",
            f"nfailed {int(connected < goal)}",
            f"goal {goal}",
            f"cap {search_cap}",
            f"n {len(records)}",
        ]
    )
    if metrics is not None:
        lines.append(f"par2_mean {metrics['par2_mean']}")
        lines.append(f"v2_nfailed {metrics['nfailed']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    if metrics is not None:
        provenance = collect_provenance(
            code=str(template.get("code", "optim")),
            code_version=str(template.get("code_version", "")),
            commit=str(template.get("commit", "")),
            parameters={"template": template["id"], "problem": problem["id"], "cap": search_cap, "potential": KIND},
        )
        document = {
            "schema": "optbench.result/v2",
            "status": "archive",
            "problem": problem["id"],
            "scoring": "par2-wilson-bootstrap",
            "par_factor": 2,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 20261015,
            "note": (
                "One row is one displacement search. The cap is the climb budget, "
                "64 calls of overshoot, and the two endpoint quenches. "
                "The initial quench is outside that count."
            ),
            "runs": [
                {
                    "entry": output.stem,
                    "code": template.get("code", "optim"),
                    "algorithm": template.get("algorithm", ""),
                    "code_version": template.get("code_version", ""),
                    "commit": template.get("commit", ""),
                    "tuning": template.get("tuning", ""),
                    "present_source": True,
                    "cap": search_cap,
                    "provenance": provenance,
                    "metrics": metrics,
                    "starts": records,
                }
            ],
        }
        output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    header_path.unlink(missing_ok=True)
    totals = [int(row["force_calls"]) for row in records]
    return {
        "force_calls": statistics.fmean(totals) if totals else None,
        "nfailed": int(connected < goal),
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": None if metrics is None else metrics["par2_mean"],
        "connected": connected,
        "records": records,
    }
