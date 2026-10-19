"""OPTIM Pt-island saddle search on the 2014 preamble.

One search is a ``BFGSTS`` climb through the ASE socket, then two
``BFGSMIN`` quenches. The socket potential is ``pes.evaluate("morse_eon")``,
eOn ``Morse::force``. The count is the socket. OPTIM's own counter is
not the score.

The start is ``draw_search_start``: ``search_seed(rep, attempt)``, one
epicentre from the free listed atoms, and one Gaussian of standard
deviation ``displace_magnitude`` on every free atom inside
``displace_radius``. That Gaussian is the geometry change and the
min-mode direction.

The link is ``connected_endpoints``. Novelty is ``ProcessCatalog`` on
the saddle geometry, which is ``AKMCState.find_repeat``. A search that
never minimizes both endpoints is not ``not_connected``.

The climb socket stops at ``max_iterations`` calls. Each quench stops
at 2000. PAR-2 adds 64 calls of overshoot, so the stored cap is
``1000 + 64 + 2 * 2000``. The initial minimization is outside that
count. Nothing here turns a running search into a failure because a
wall clock expired. A process whose force-call counter does not
advance is killed and recorded as a crash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from optbench.connection import ProcessCatalog, connected_endpoints, link_tolerance, per_atom_max
from optbench.endpoints import EndpointArchive, align_resume, stamp_endpoint, stored_cell
from optbench.eon_displace import draw_search_start, problem_pbc, search_seed
from optbench.failures import classify_failure
from optbench.score import score_v2
from optbench.v2 import collect_provenance, write_starts

from .engines.optim_search import (
    _ase_line,
    _fortran_float,
    _freeze_lines,
    _geometry_file,
    _install_hook,
    _optim_env,
    _params_line,
    _write_xyz,
    binary_path,
)

# The relay refuses the call that would pass these budgets and does not
# count it. PAR-2 still adds one step of overshoot on the climb.
CLIMB_OVERSHOOT = 64
ENDPOINT_BUDGET = 2000
MINIMIZE_BUDGET = 5000
PUSH = 0.1
# A stall this long, with no new socket call, is a dead process.
HANG_S = 900.0
KIND = "morse_eon"
PAPER_BFGSCONV = "4.3895D-5"

search_displacement = draw_search_start


def optim_seed(seed: int) -> int:
    """Residue ``SDPRND`` can tell from another seed."""
    return int(seed) % 10000


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
    """Reasons that stop the search before the endpoint quenches.

    ``None`` means the climb is finite, under the climb budget, at the
    Euclidean force stop, and Hessian index 1.
    """
    if hang or error:
        return ["crash"]
    if not finite:
        return ["divergence"]
    if capped or int(calls) >= int(climb_cap):
        return ["cap", "force_norm"]
    if force_norm is None or not math.isfinite(float(force_norm)) or float(force_norm) > float(force_tol) + 1e-10:
        return ["force_norm"]
    if index != 1:
        return ["hessian_index"]
    return None


def paper_keyword_lines(preamble: str) -> list[str]:
    """Non-comment lines of the 2014 preamble, unchanged."""
    lines = []
    for line in preamble.splitlines():
        if line.strip() and not line.startswith("comment"):
            lines.append(line)
    return lines


def climb_odata(
    *,
    python: str,
    hook: Path,
    periodic: bool,
    preamble: str,
    params: str,
    seed: int,
    freeze: list[str],
) -> str:
    """Paper preamble, then PARAMS, RANSEED, and one FREEZE line per frozen atom."""
    lines = [
        _ase_line(python, hook, periodic),
        preamble.rstrip("\n"),
        params,
        f"RANSEED {optim_seed(seed)}",
        *freeze,
    ]
    text = "\n".join(lines) + "\n"
    for forbidden in ("MAXERISE", "CONVERGE"):
        for line in text.splitlines():
            if line.split()[:1] == [forbidden]:
                raise RuntimeError(f"paper odata must not add {forbidden}")
    return text


def minimize_odata(
    *,
    python: str,
    hook: Path,
    periodic: bool,
    params: str,
    seed: int,
    freeze: list[str],
) -> str:
    """Endpoint and reactant quench. The RMS token is the paper ``bfgsconv``."""
    lines = [
        _ase_line(python, hook, periodic),
        "BFGSMIN",
        f"BFGSCONV {PAPER_BFGSCONV}",
        "STEPS 100000",
        "MAXBFGS 0.2",
        "UPDATES 4",
        params,
        f"RANSEED {optim_seed(seed)}",
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
    """Last complete ``vector.dump`` record.

    OPTIM rewinds the file and scans until EOF, so a short rewrite can
    leave an older tail and a trailing partial record is ignored.
    """
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


def _launch(spec: dict) -> dict:
    """One OPTIM process. ``calls`` is the socket counter."""
    work = Path(spec["work"]).resolve()
    if work.exists():
        for child in work.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                import shutil

                shutil.rmtree(child)
    work.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(spec["positions"], dtype=np.float64)
    reference = np.asarray(spec["reference"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    natoms = int(len(positions))
    scale_l = float(spec["scale_l"])
    scale_e = float(spec["scale_e"])
    cap = int(spec["cap"])
    kind = str(spec["kind"])
    method = str(spec["method"])
    hook = _install_hook(work)
    _write_xyz(work / "geom.xyz", positions * scale_l, cell * scale_l)
    if method != "minimize" and spec.get("vector") is not None:
        write_vector(work / "vector.dump", np.asarray(spec["vector"], dtype=np.float64) * scale_l)
    if method == "minimize":
        odata = minimize_odata(
            python=str(spec["python"]),
            hook=hook,
            periodic=bool(spec["periodic"]),
            params=str(spec["params"]),
            seed=int(spec["seed"]),
            freeze=list(spec["freeze"]),
        )
    else:
        odata = climb_odata(
            python=str(spec["python"]),
            hook=hook,
            periodic=bool(spec["periodic"]),
            preamble=str(spec["preamble"]),
            params=str(spec["params"]),
            seed=int(spec["seed"]),
            freeze=list(spec["freeze"]),
        )
    (work / "odata").write_text(odata, encoding="utf-8")
    env = _optim_env(work, kind, cap, cell, scale_l, scale_e, None, work / "traj.xyz")
    root = str(spec["repo"])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
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
            time.sleep(1.0)
    finally:
        log.close()
    calls = _count_file(work / "calls.txt") or 0
    capped = (work / "capped.txt").is_file() or calls >= cap
    coords = None
    vector = None
    text = ""
    try:
        if not hang and not capped:
            point = _geometry_file(work, natoms, scale_l, kind, cell, scale_e, float(spec["force_tol"]))
            if point is not None and np.all(np.isfinite(point)):
                coords = np.array(point, dtype=np.float64, copy=True)
                coords[fixed] = reference[fixed]
            if method != "minimize":
                vector = read_vector_dump(work / "vector.dump", 3 * natoms)
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    finally:
        if os.environ.get("OPTBENCH_OPTIM_KEEP") != "1":
            import shutil

            shutil.rmtree(work, ignore_errors=True)
    converged = ("**** CONVERGED ****" in text) or ("Converged to TS" in text)
    error = None
    if hang:
        error = "no new force calls"
    elif coords is None and not capped and calls < cap:
        detail = "\n".join(text.splitlines()[-30:])
        error = f"exit {returncode}: {detail}"
    return {
        "calls": int(calls),
        "capped": bool(capped),
        "hang": bool(hang),
        "returncode": None if returncode is None else int(returncode),
        "coords": coords,
        "vector": vector,
        "converged": bool(converged),
        "error": error,
    }


def _endpoint_axis(vector: np.ndarray | None, fixed: np.ndarray, kick: np.ndarray) -> np.ndarray:
    from optbench.rgsaddle_minmode import _unit

    if vector is not None and vector.size == 3 * len(fixed) and np.all(np.isfinite(vector)):
        free = np.asarray(vector, dtype=np.float64).reshape(len(fixed), 3)[~np.asarray(fixed, dtype=bool)].reshape(-1)
        if float(np.linalg.norm(free)) > 1e-12:
            return _unit(free)
    return _unit(np.asarray(kick, dtype=np.float64))


def _harness_force(spec: dict, coords: np.ndarray) -> tuple[float, float, int]:
    """Energy, Euclidean force norm, and verification calls. The socket is not charged."""
    from optbench.rgsaddle_minmode import HarnessSurface, _bind, _norm
    from optbench.surfaces import potential_from_problem

    fixed = np.asarray(spec["fixed"], dtype=bool)
    free = np.flatnonzero(~fixed)
    reference = np.asarray(spec["reference"], dtype=np.float64)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    point = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[free].reshape(-1)
    raw = potential_from_problem({"id": "saddle-search/pt-island", "potential": spec["potential"]})
    surface = HarnessSurface(_bind(raw, reference, free, cell))
    with surface.verification():
        energy, gradient = surface.energy_forces(point)
    return float(energy), _norm(gradient), int(surface.verify_calls)


def _hessian_index(spec: dict, coords: np.ndarray) -> tuple[int, float, int]:
    from optbench.rgsaddle_minmode import HarnessSurface, _bind, _hessian_spectrum
    from optbench.stationary import negative_index
    from optbench.surfaces import potential_from_problem

    fixed = np.asarray(spec["fixed"], dtype=bool)
    free = np.flatnonzero(~fixed)
    reference = np.asarray(spec["reference"], dtype=np.float64)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    point = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[free].reshape(-1)
    raw = potential_from_problem({"id": "saddle-search/pt-island", "potential": spec["potential"]})
    surface = HarnessSurface(_bind(raw, reference, free, cell))
    eigenvalues, _vectors = _hessian_spectrum(surface, point)
    # The substrate pins the island. Do not drop rigid modes.
    index = negative_index(list(eigenvalues), 1e-6)
    lowest = float(eigenvalues[0]) if eigenvalues else math.nan
    return int(index), lowest, int(surface.verify_calls)


def _saddle_attempt(spec: dict) -> dict:
    """Climb, and quench both ends when the climb is an index-1 saddle."""
    result = {
        "climb": 0,
        "capped": False,
        "hang": False,
        "error": None,
        "finite": False,
        "coords": None,
        "vector": None,
        "force_norm": None,
        "energy": None,
        "hessian_index": None,
        "lowest_eigenvalue": None,
        "verify_calls": 0,
        "plus": None,
        "minus": None,
        "endpoint_calls": 0,
        "endpoint_error": None,
        "converged": False,
    }
    climb_work = Path(spec["work"])
    try:
        outcome = _launch(spec)
        result["climb"] = int(outcome["calls"])
        result["capped"] = bool(outcome["capped"])
        result["hang"] = bool(outcome["hang"])
        result["error"] = outcome.get("error")
        result["converged"] = bool(outcome.get("converged"))
        coords = outcome.get("coords")
        finite = coords is not None and bool(np.all(np.isfinite(coords)))
        result["finite"] = finite
        if outcome.get("vector") is not None:
            result["vector"] = np.asarray(outcome["vector"], dtype=np.float64)
        if not finite or result["hang"] or result["error"] or result["capped"] or result["climb"] >= int(spec["cap"]):
            return result
        result["coords"] = np.asarray(coords, dtype=np.float64)
        energy, force_norm, verify_calls = _harness_force(spec, result["coords"])
        result["energy"] = energy
        result["force_norm"] = force_norm
        result["verify_calls"] = verify_calls
        if not (math.isfinite(force_norm) and force_norm <= float(spec["force_tol"]) + 1e-10):
            return result
        index, lowest, hess_calls = _hessian_index(spec, result["coords"])
        result["hessian_index"] = int(index)
        result["lowest_eigenvalue"] = float(lowest)
        result["verify_calls"] = int(verify_calls + hess_calls)
        if index != 1:
            return result
        fixed = np.asarray(spec["fixed"], dtype=bool)
        free = np.flatnonzero(~fixed)
        axis = _endpoint_axis(result["vector"], fixed, np.asarray(spec["kick"], dtype=np.float64))
        free_point = result["coords"].reshape(-1, 3)[free].reshape(-1)
        plus_full = np.array(result["coords"], copy=True)
        minus_full = np.array(result["coords"], copy=True)
        plus_full.reshape(-1, 3)[free] = (free_point + PUSH * axis).reshape(-1, 3)
        minus_full.reshape(-1, 3)[free] = (free_point - PUSH * axis).reshape(-1, 3)
        endpoint_calls = 0
        ends = {}
        for name, moved in (("plus", plus_full), ("minus", minus_full)):
            quench = dict(spec)
            quench["method"] = "minimize"
            quench["cap"] = ENDPOINT_BUDGET
            quench["positions"] = moved
            quench["work"] = str(climb_work) + f"-{name}"
            quench["vector"] = None
            done = _launch(quench)
            endpoint_calls += int(done["calls"])
            point = done.get("coords")
            if done.get("hang") or done.get("error") or point is None or not np.all(np.isfinite(point)):
                result["endpoint_error"] = done.get("error") or ("no new force calls" if done.get("hang") else "endpoint has no geometry")
                ends[name] = None
            else:
                ends[name] = np.asarray(point, dtype=np.float64)
        result["endpoint_calls"] = int(endpoint_calls)
        result["plus"] = ends.get("plus")
        result["minus"] = ends.get("minus")
        return result
    except Exception:
        result["error"] = traceback.format_exc()[-800:]
        return result


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
    return rows


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, default=_json_default, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _public(record: dict) -> dict:
    hidden = {"saddle_x"}
    return {key: value for key, value in record.items() if key not in hidden}


def _as_rows(values: np.ndarray, free: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1, 3)[free]


def _protocol(header: dict) -> dict:
    keys = (
        "template",
        "preamble_sha256",
        "listed",
        "radius",
        "magnitude",
        "max_searches",
        "climb_cap",
        "kind",
        "rep",
    )
    return {key: header[key] for key in keys}


def _tarball_comment() -> str:
    path = Path("/tmp/src/svn.tar.bz2")
    pin = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
    else:
        digest = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
        size = 122737951
    return (
        f"# built from {path} sha256 {digest} size {size}. "
        f"The registry pin is {pin} (122743078 bytes) and is not this tarball."
    )


def _outcome(reasons: list[str]) -> tuple[bool, str | None]:
    failure = classify_failure(reasons)
    return failure is None, failure


def run_saddle(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    resume: bool = False,
    limit: int = 0,
) -> dict:
    """Run until 27 qualifying saddles or ``max_searches``, whichever is first."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from optbench.pes import evaluate
    from optbench.rgsaddle_minmode import _stored
    from optbench.surfaces import load_con

    root = Path(root).resolve()
    output = Path(output).resolve()
    section = dict(template.get("optim") or {})
    if str(section.get("method") or "") != "hef-ffd":
        raise RuntimeError(f"{template['id']} is not the paper BFGSTS saddle search")
    if str(section.get("scale") or "") != "morse-2014":
        raise RuntimeError(f"{template['id']} must scale coordinates by r0 and energy by D")
    preamble_path = root / str(section.get("preamble") or "")
    if not preamble_path.is_file():
        raise RuntimeError(f"{template['id']} preamble is not at {preamble_path}")
    preamble = preamble_path.read_text(encoding="utf-8")
    if PAPER_BFGSCONV not in preamble:
        raise RuntimeError(f"{preamble_path} does not contain {PAPER_BFGSCONV}")
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = load_con(_stored(root, con_path))
    fixed = np.asarray(fixed, dtype=bool)
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    search = ((template.get("ini") or {}).get("Search") or {})
    optimizer = ((template.get("ini") or {}).get("Optimizer") or {})
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    repetitions = int(search.get("repetitions", 1))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    if search.get("displace_radius") is None:
        raise RuntimeError(f"{template['id']} has no displace_radius")
    radius = float(search["displace_radius"])
    magnitude = float(search.get("displace_magnitude", 0.1))
    stated_max = int(search.get("max_searches", 20000))
    max_searches = stated_max if not limit or int(limit) <= 0 else min(stated_max, int(limit))
    climb_cap = int(optimizer.get("max_iterations", 1000))
    force_tol = float(optimizer.get("converged_force", 1.0e-3))
    search_cap = climb_cap + CLIMB_OVERSHOOT + 2 * ENDPOINT_BUDGET
    tolerance = link_tolerance(problem)
    pbc = problem_pbc(problem)
    periodic = any(pbc)
    scale_l, scale_e = (1.0 / 2.897, 1.0 / 0.7102)
    params = _params_line(cell, scale_l)
    freeze = _freeze_lines(~fixed)
    free = np.flatnonzero(~fixed)
    potential = problem["potential"]
    # Compile Morse::force before the workers start, so they share the library.
    evaluate(KIND, positions[:2], cell)
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    header_path = output.with_name(output.stem + ".header.json")
    partial_path = output.with_name(output.stem + ".partial.jsonl")
    archive = EndpointArchive(output.with_name(output.stem + ".endpoints.npz"), int(free.size))
    preamble_hash = hashlib.sha256(preamble.encode("utf-8")).hexdigest()
    if output.is_file() and not header_path.is_file():
        raise RuntimeError(f"{output} is already a finished campaign")
    all_records: list[dict] = []
    comments: list[str] = [_tarball_comment()]
    failed_reps = 0
    width = max(1, int(workers))

    def _base_spec(seed: int, work: Path) -> dict:
        return {
            "binary": str(binary_path()),
            "python": sys.executable,
            "repo": str(root),
            "kind": KIND,
            "potential": potential,
            "fixed": fixed,
            "reference": positions,
            "cell": cell,
            "periodic": periodic,
            "scale_l": scale_l,
            "scale_e": scale_e,
            "params": params,
            "preamble": preamble,
            "freeze": freeze,
            "force_tol": force_tol,
            "seed": int(seed),
            "work": str(work),
        }

    def _minimize(seed: int, coords: np.ndarray, cap: int, work: Path) -> dict:
        spec = _base_spec(seed, work)
        spec.update({"method": "minimize", "cap": int(cap), "positions": coords, "vector": None, "kick": np.zeros(3 * len(free))})
        return _launch(spec)

    executor = ProcessPoolExecutor(max_workers=width) if width > 1 else None
    try:
        for rep in range(1, repetitions + 1):
            fresh = {
                "template": template["id"],
                "preamble_sha256": preamble_hash,
                "listed": listed,
                "radius": radius,
                "magnitude": magnitude,
                "max_searches": max_searches,
                "climb_cap": climb_cap,
                "kind": KIND,
                "rep": rep,
            }
            header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
            if header is not None:
                if not resume:
                    raise RuntimeError(f"{header_path.name} exists; pass --resume")
                if _protocol(header) != fresh:
                    raise RuntimeError(f"{header_path.name} does not match this template")
                saved = _load_jsonl(partial_path)
                align_resume(archive, len(all_records) + len(saved))
                minimum = np.asarray(header["minimum"], dtype=np.float64)
                min_energy = float(header["min_energy"])
                minimize_calls = int(header["minimize_calls"])
                min_ok = bool(header["min_ok"])
                catalog = ProcessCatalog(cell, tolerance=tolerance, barrier_max=barrier_max)
                for row in saved:
                    linked = bool(row.get("judged")) and "not_connected" not in (row.get("reasons") or [])
                    saddle = None if row.get("saddle_x") is None else np.asarray(row["saddle_x"], dtype=np.float64)
                    catalog.observe(
                        judged=bool(row.get("judged")),
                        linked=linked,
                        structure=saddle,
                        barrier=row.get("barrier"),
                    )
                    all_records.append(_public(row))
                print(
                    f"{template['id']} rep {rep} resume searches {catalog.searches} connected {catalog.qualifying}",
                    flush=True,
                )
            else:
                if rep == 1:
                    archive.reset(stored_cell(cell), free)
                quenched = _minimize(search_seed(rep, 0), positions, MINIMIZE_BUDGET, scratch / f"rep{rep}-min")
                point = quenched.get("coords")
                if point is None or quenched.get("hang") or quenched.get("error"):
                    raise RuntimeError(f"initial minimization failed: {quenched.get('error')}")
                energy, force_norm, _verify = _harness_force(_base_spec(search_seed(rep, 0), scratch / "unused"), point)
                min_ok = bool(math.isfinite(force_norm) and force_norm <= force_tol + 1e-10 and not quenched.get("capped"))
                if not min_ok:
                    raise RuntimeError(
                        f"initial minimization force {force_norm} calls {quenched['calls']} capped {quenched['capped']}"
                    )
                minimum = np.asarray(point, dtype=np.float64).reshape(-1, 3)[free].reshape(-1)
                min_energy = float(energy)
                minimize_calls = int(quenched["calls"])
                catalog = ProcessCatalog(cell, tolerance=tolerance, barrier_max=barrier_max)
                header = dict(fresh)
                header.update(
                    {
                        "minimum": minimum.tolist(),
                        "min_energy": min_energy,
                        "minimize_calls": minimize_calls,
                        "min_ok": True,
                    }
                )
                _write_json(header_path, header)
                partial_path.unlink(missing_ok=True)
                print(
                    f"{template['id']} rep {rep} minimum calls {minimize_calls} force {force_norm:.6g} energy {min_energy:.8g}",
                    flush=True,
                )
            reactant = np.array(positions, copy=True)
            reactant[free] = minimum.reshape(-1, 3)
            while catalog.qualifying < goal and catalog.searches < max_searches:
                room = min(width, max_searches - catalog.searches)
                jobs = []
                drawn_rows = []
                for offset in range(room):
                    attempt = catalog.searches + offset + 1
                    drawn = search_displacement(
                        reactant,
                        fixed,
                        cell,
                        pbc,
                        listed,
                        rep=rep,
                        attempt=attempt,
                        radius=radius,
                        magnitude=magnitude,
                    )
                    spec = _base_spec(int(drawn["seed"]), scratch / f"rep{rep}-search{attempt}")
                    spec.update(
                        {
                            "method": "hef-ffd",
                            "cap": climb_cap,
                            "positions": drawn["start"],
                            "vector": drawn["vector"],
                            "kick": drawn["kick"],
                        }
                    )
                    jobs.append(spec)
                    drawn_rows.append(drawn)
                if executor is None:
                    outcomes = [_saddle_attempt(job) for job in jobs]
                else:
                    futures = [executor.submit(_saddle_attempt, job) for job in jobs]
                    outcomes = []
                    for job, future in zip(jobs, futures):
                        try:
                            outcomes.append(future.result())
                        except Exception as exc:
                            outcomes.append(
                                {
                                    "climb": 0,
                                    "capped": False,
                                    "hang": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "finite": False,
                                    "coords": None,
                                    "vector": None,
                                    "force_norm": None,
                                    "energy": None,
                                    "hessian_index": None,
                                    "lowest_eigenvalue": None,
                                    "verify_calls": 0,
                                    "plus": None,
                                    "minus": None,
                                    "endpoint_calls": 0,
                                    "endpoint_error": None,
                                    "converged": False,
                                }
                            )
                stop = False
                for drawn, outcome in zip(drawn_rows, outcomes):
                    if catalog.qualifying >= goal or catalog.searches >= max_searches:
                        stop = True
                        break
                    attempt = catalog.searches + 1
                    record, stored = _score_one(
                        catalog,
                        outcome,
                        drawn,
                        minimum=minimum,
                        min_energy=min_energy,
                        free=free,
                        fixed=fixed,
                        cell=cell,
                        tolerance=tolerance,
                        climb_cap=climb_cap,
                        force_tol=force_tol,
                        name=f"rep{rep}-search{attempt}",
                        archive=archive,
                    )
                    calls = int(record["force_calls"])
                    if record["success"] and calls >= search_cap:
                        raise RuntimeError(f"{record['start']} spent {calls} force calls, cap {search_cap}")
                    all_records.append(record)
                    _append_jsonl(partial_path, stored)
                    failure = record["failure"]
                    if catalog.searches == 1 or catalog.searches % 10 == 0 or record["success"] or catalog.qualifying >= goal:
                        print(
                            f"{template['id']} rep {rep} searches {catalog.searches} "
                            f"connected {catalog.qualifying} unique {catalog.unique} "
                            f"calls {calls} failure {failure}",
                            flush=True,
                        )
                if stop or catalog.qualifying >= goal or catalog.searches >= max_searches:
                    break
            if catalog.qualifying < goal:
                failed_reps += 1
            repetition_calls = minimize_calls + sum(int(row["force_calls"]) for row in all_records)
            comments.append(
                f"# rep {rep} force_calls {repetition_calls} unique_saddles {catalog.unique} "
                f"fraction_connected {catalog.fraction_connected} total_searches {catalog.searches} "
                f"connected {catalog.qualifying} minimize_calls {minimize_calls} "
                f"minimum_converged {int(min_ok)} search_seed (20261015+{rep})*1000003+attempt"
            )
            _finish(
                output,
                comments,
                all_records,
                template,
                problem,
                search_cap,
                goal,
                repetitions,
                failed_reps,
                catalog,
                archive,
                max_searches,
                tolerance,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if not output.is_file():
        raise RuntimeError(f"{template['id']} stopped before writing {output.name}")
    partial_path.unlink(missing_ok=True)
    header_path.unlink(missing_ok=True)
    archive.drop_log()
    return {
        "force_calls": statistics.fmean([int(row["force_calls"]) for row in all_records]) if all_records else None,
        "nfailed": failed_reps,
        "starts": len(all_records),
        "template": template["id"],
        "problem": problem["id"],
    }


def _score_one(
    catalog: ProcessCatalog,
    outcome: dict,
    drawn: dict,
    *,
    minimum: np.ndarray,
    min_energy: float,
    free: np.ndarray,
    fixed: np.ndarray,
    cell,
    tolerance: float,
    climb_cap: int,
    force_tol: float,
    name: str,
    archive: EndpointArchive,
) -> tuple[dict, dict]:
    del fixed
    reasons = climb_reasons(
        hang=bool(outcome.get("hang")),
        error=outcome.get("error"),
        finite=bool(outcome.get("finite")),
        capped=bool(outcome.get("capped")),
        calls=int(outcome.get("climb") or 0),
        climb_cap=climb_cap,
        force_norm=None if outcome.get("force_norm") is None else float(outcome["force_norm"]),
        force_tol=force_tol,
        index=None if outcome.get("hessian_index") is None else int(outcome["hessian_index"]),
    )
    plus = outcome.get("plus")
    minus = outcome.get("minus")
    judged = False
    links = False
    saddle_rows = None
    barrier = None
    plus_rows = None
    minus_rows = None
    per_plus = None
    per_minus = None
    if reasons is None and (plus is None or minus is None or outcome.get("endpoint_error")):
        reasons = ["crash"] if outcome.get("endpoint_error") or outcome.get("error") else ["divergence"]
    if reasons is None:
        minimum_rows = np.asarray(minimum, dtype=np.float64).reshape(-1, 3)
        plus_rows = _as_rows(plus, free)
        minus_rows = _as_rows(minus, free)
        saddle_rows = _as_rows(outcome["coords"], free)
        links = connected_endpoints(minimum_rows, plus_rows, minus_rows, cell, tolerance)
        barrier = float(outcome["energy"]) - float(min_energy)
        per_plus = per_atom_max(plus_rows, minimum_rows, cell)
        per_minus = per_atom_max(minus_rows, minimum_rows, cell)
        judged = True
        decision = catalog.observe(judged=True, linked=links, structure=saddle_rows, barrier=barrier)
        reasons = list(decision["reasons"])
        success = bool(decision["success"])
        failure = None if success else classify_failure(reasons)
    else:
        catalog.observe(judged=False)
        success, failure = _outcome(reasons)
    reactant_rows = np.asarray(minimum, dtype=np.float64).reshape(-1, 3)
    status = archive.add(reactant_rows, plus_rows if judged else None, minus_rows if judged else None)
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons or [],
        "judged": judged,
        "force_calls": int(outcome.get("climb") or 0) + int(outcome.get("endpoint_calls") or 0),
        "climb_calls": int(outcome.get("climb") or 0),
        "endpoint_calls": int(outcome.get("endpoint_calls") or 0),
        "verify_calls": int(outcome.get("verify_calls") or 0),
        "converged": bool(outcome.get("converged")),
        "energy": outcome.get("energy"),
        "force_norm": outcome.get("force_norm"),
        "barrier": barrier,
        "hessian_index": outcome.get("hessian_index"),
        "cartesian_index": outcome.get("hessian_index"),
        "rigid_modes_removed": 0,
        "expected_index": 1,
        "lowest_eigenvalue": outcome.get("lowest_eigenvalue"),
        "per_atom_max_plus": per_plus,
        "per_atom_max_minus": per_minus,
        "seed": int(drawn["seed"]),
        "optim_seed": optim_seed(int(drawn["seed"])),
        "epicenter": int(drawn["epicenter"]),
    }
    if outcome.get("error"):
        record["error"] = str(outcome["error"])[-500:]
    stamp_endpoint(record, archive.name, archive.count - 1, status)
    stored = dict(record)
    if saddle_rows is not None:
        stored["saddle_x"] = np.asarray(saddle_rows, dtype=np.float64).reshape(-1).tolist()
    return record, stored


def _finish(
    output: Path,
    comments: list[str],
    records: list[dict],
    template: dict,
    problem: dict,
    search_cap: int,
    goal: int,
    repetitions: int,
    failed_reps: int,
    catalog: ProcessCatalog,
    archive: EndpointArchive,
    max_searches: int,
    tolerance: float,
) -> None:
    totals = []
    for line in comments:
        parts = line.split()
        if "force_calls" in parts and line.startswith("# rep"):
            totals.append(int(parts[parts.index("force_calls") + 1]))
    metrics = (
        score_v2(
            [
                {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
                for row in records
            ],
            search_cap,
        )
        if records
        else None
    )
    # A run that reached the goal or the search cap is the whole campaign.
    # Do not mark it partial.
    reached = catalog.qualifying >= goal or catalog.searches >= max_searches
    if not reached:
        return
    text = "\n".join(comments) + "\n"
    if totals:
        text += f"force_calls {statistics.fmean(totals)}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
    text += (
        f"code {template.get('code', 'optim')}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"potential {KIND}\n"
        f"counter harness\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(records)}\n"
        f"unique_saddles {catalog.unique}\n"
        f"fraction_connected {catalog.fraction_connected}\n"
        f"total_searches {catalog.searches}\n"
        f"connected {catalog.qualifying}\n"
    )
    if metrics is not None:
        text += (
            f"par2_mean {metrics['par2_mean']}\n"
            f"v2_nfailed {metrics['nfailed']}\n"
            f"n_cap {metrics['n_cap']}\n"
            f"n_crash {metrics['n_crash']}\n"
            f"n_divergence {metrics['n_divergence']}\n"
            f"n_wrong_stationary {metrics['n_wrong_stationary']}\n"
            f"successes {metrics['successes']}\n"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.save()
    output.write_text(text, encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    if metrics is not None:
        document = {
            "schema": "optbench.score/v2",
            "problem": problem["id"],
            "template": template["id"],
            "counter": "harness",
            "provenance": collect_provenance(
                code=str(template.get("code", "optim")),
                code_version=str(template.get("code_version", "")),
                commit=str(template.get("commit", "")),
                parameters={
                    "cap": search_cap,
                    "counter": "harness",
                    "goal": goal,
                    "max_searches": max_searches,
                    "distance_difference": tolerance,
                    "potential": KIND,
                    "preamble": "templates/optim/preambles/saddle-pt.txt",
                },
            ),
            "score": metrics,
        }
        output.with_suffix(".v2.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
