"""OPTIM campaigns through the ASE socket and the harness potential.

OPTIM stops on an RMS gradient. The problem stop is the Euclidean norm
of the free atoms, so ``BFGSCONV`` is that tolerance divided by
``sqrt(3 N_free)``. Frozen components are zero in the gradient and are
left out of the RMS count, so the RMS stop is the Euclidean stop.

Hybrid eigenvector-following is ``BFGSTS`` with ``NOHESS`` and forward
Rayleigh-Ritz differences (``USEDIAG 3``). BFGS-TS is the same search
with central differences (``USEDIAG 2``). The lowest-eigenvector search
is ``EIGENONLY``. That path leaves the base gradient unset, so the
forward curvature is not the Hessian quotient; the central difference
is the one that tracks the reference mode. The overlap stop is 0.99.
A saddle search climbs with hybrid eigenvector-following and minimises
both ends with ``BFGSMIN``.

The harness tally is the socket counter. OPTIM's own ``FCALL`` is not
the score. Hessian calls in the parent are verification.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .rgsaddle_minmode import (
    HarnessSurface,
    _bind,
    _cartesian_rmsd,
    _geometries,
    _hessian_spectrum,
    _norm,
    _outcome,
    _unit,
    _vibrational_eigenvalues,
)
from .score import half_even, score_v2
from .stationary import judge_stationary, negative_index
from .surfaces import load_con, potential_from_problem, read_mode
from .v2 import collect_provenance, write_starts

_RELAY_ROOT = Path("/tmp/obroot")
_ENDPOINT_BUDGET = 2000
_CLIMB_OVERSHOOT = 64
_MINIMIZE_BUDGET = 5000
_PUSH = 0.1


def optim_binary() -> Path:
    path = Path(os.environ.get("OPTIM_BIN", "/tmp/optimbld/OPTIM"))
    if not path.is_file():
        raise RuntimeError(f"OPTIM_BIN is not a file: {path}")
    return path


def fortran_float(value: float) -> str:
    """Decimal Fortran accepts. ``E`` and ``D`` are both legal."""
    return f"{float(value):.16E}".replace("E", "D")


def rms_stop(force_tol: float, n_coordinates: int) -> float:
    """RMS threshold whose Euclidean image is ``force_tol``.

    OPTIM reports ``sqrt(sum(g^2) / n)`` with frozen components removed
    from ``n``. A stop at this RMS is a Euclidean norm of ``force_tol``.
    """
    count = int(n_coordinates)
    if count < 1:
        raise ValueError("rms stop needs at least one coordinate")
    return float(force_tol) / math.sqrt(float(count))


def install_relay() -> Path:
    dest = _RELAY_ROOT / "pysocket"
    dest.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("optim_fmlip_relay.py")
    target = dest / "fmlip_relay.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return _RELAY_ROOT


def _section(template: dict, name: str) -> dict:
    return ((template.get("ini") or {}).get(name) or {})


def odata_text(
    *,
    python: str,
    relay_root: Path,
    method: str,
    n_coordinates: int,
    force_tol: float,
    periodic: bool,
    frozen: bool,
    read_vector: bool,
    optim: dict,
) -> str:
    """Keyword file for one OPTIM process. Coordinates live in ``geom.xyz``."""
    if " " in python or " " in str(relay_root):
        raise RuntimeError("OPTIM ASE paths cannot contain spaces")
    pbc = "pbc" if periodic else "nopbc"
    lines = [
        "NOTE optbench harness socket",
        f"ASE geom.xyz {python} {relay_root} harness job cpu fixcell {pbc} ev ang",
    ]
    rms = fortran_float(rms_stop(force_tol, n_coordinates))
    # The unscaled eigenvector step is not the problem stop. A step
    # allowance of one length unit leaves the RMS force in charge.
    kind = method
    if kind == "minimize":
        lines.extend(
            [
                "BFGSMIN",
                f"BFGSCONV {rms}",
                "STEPS 100000",
                "MAXBFGS 0.2",
                "UPDATES 4",
                "RANSEED 2014",
            ]
        )
    elif kind == "rr-ffd":
        ceig = fortran_float(float(optim.get("ceig", 1.0e-4)))
        cover = fortran_float(float(optim.get("cover", 0.99)))
        lines.extend(
            [
                f"EIGENONLY {ceig} {cover}",
                "NOHESS",
                f"USEDIAG {int(optim.get('usediag', 2))}",
                "READVEC",
                "DUMPVECTOR",
                "STEPS 100000",
                "RANSEED 2014",
            ]
        )
    elif kind in {"hef-ffd", "bfgsts"}:
        usediag = 3 if kind == "hef-ffd" else 2
        if "usediag" in optim and kind == "bfgsts":
            usediag = int(optim["usediag"])
        if kind == "hef-ffd":
            usediag = int(optim.get("usediag", 3))
        nevs = int(optim.get("nevs", 500))
        nbfgs1 = int(optim.get("nbfgsmax1", 10))
        nbfgs2 = int(optim.get("nbfgsmax2", 100))
        ceig = fortran_float(float(optim.get("ceig", 1.0e-4)))
        lines.extend(
            [
                f"BFGSTS {nevs} {nbfgs1} {nbfgs2} {ceig}",
                "NOHESS",
                f"USEDIAG {usediag}",
                "STEPS 100000",
                f"CONVERGE {fortran_float(float(optim.get('convu', 1.0)))} {rms}",
                f"BFGSCONV {rms}",
                f"MAXSTEP {fortran_float(float(optim.get('maxstep', 0.2)))}",
                f"MAXBFGS {fortran_float(float(optim.get('maxbfgs', 0.2)))}",
                f"UPDATES {int(optim.get('updates', 4))} {int(optim.get('xmupdate', optim.get('updates', 4)))}",
                f"RANSEED {int(optim.get('seed', 2014))}",
            ]
        )
        if read_vector:
            lines.extend(["READVEC", "DUMPVECTOR"])
    else:
        raise ValueError(f"unknown OPTIM method {method}")
    if frozen:
        lines.append("FREEZE")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_xyz(path: Path, positions: np.ndarray, cell: np.ndarray, periodic: bool, symbol: str) -> None:
    rows = [str(len(positions))]
    if periodic:
        lattice = (
            f"{cell[0]:.16g} 0.0 0.0 0.0 {cell[1]:.16g} 0.0 0.0 0.0 {cell[2]:.16g}"
        )
        rows.append(f'Lattice="{lattice}"')
    else:
        rows.append("optbench")
    for point in np.asarray(positions, dtype=np.float64):
        rows.append(f"{symbol} {point[0]:.16g} {point[1]:.16g} {point[2]:.16g}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_vector(path: Path, values: np.ndarray, *, eigenvalue: float = 0.0, skip_line: str | None = None) -> None:
    body = [f"{float(eigenvalue):.16e}" if skip_line is None else skip_line]
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    for offset in range(0, len(flat), 3):
        chunk = flat[offset : offset + 3]
        body.append(" ".join(f"{float(item):.16e}" for item in chunk))
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def read_coordinates(work: Path, natoms: int) -> np.ndarray | None:
    final = work / "points.final"
    if final.is_file():
        values = []
        for line in final.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    values.extend(float(part) for part in parts[:3])
                except ValueError:
                    continue
        if len(values) >= 3 * natoms:
            return np.asarray(values[: 3 * natoms], dtype=np.float64).reshape(natoms, 3)
    for name in ("unconverged.xyz", "converged.xyz"):
        path = work / name
        if not path.is_file():
            continue
        coords = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                coords.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
            except ValueError:
                continue
        if len(coords) >= natoms:
            return np.asarray(coords[:natoms], dtype=np.float64)
    return None


def read_vector_dump(path: Path, nopt: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        numbers = [float(item) for item in path.read_text(encoding="utf-8", errors="replace").split()]
    except ValueError:
        return None
    if len(numbers) < 1 + nopt:
        return None
    # The final dump is rewritten from the start of the file. A short
    # rewrite can leave an older tail, so the vector is the first block.
    vector = np.asarray(numbers[1 : 1 + nopt], dtype=np.float64)
    return vector


def read_counter(path: Path) -> tuple[int, bool, np.ndarray | None]:
    if not path.is_file():
        return 0, False, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0, False, None
    raw = payload.get("positions")
    positions = None
    if isinstance(raw, list) and raw:
        positions = np.asarray(raw, dtype=np.float64)
    return int(payload.get("calls") or 0), bool(payload.get("capped")), positions


def _full_vector(free_values: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    count = len(fixed)
    full = np.zeros(3 * count, dtype=np.float64)
    full.reshape(count, 3)[~fixed] = np.asarray(free_values, dtype=np.float64).reshape(-1, 3)
    return full


def _launch(spec: dict) -> dict:
    """Run one OPTIM process. ``calls`` is the socket counter."""
    work = Path(spec["work"])
    if work.exists():
        for child in work.iterdir():
            if child.is_file():
                child.unlink()
    else:
        work.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    natoms = int(len(positions))
    n_free = int(np.count_nonzero(~fixed))
    periodic = bool(spec["periodic"])
    method = str(spec["method"])
    read_vector = spec.get("vector") is not None and method != "minimize"
    write_xyz(work / "geom.xyz", positions, cell, periodic, str(spec.get("symbol") or "C"))
    if np.any(fixed):
        indices = [str(int(index) + 1) for index in np.flatnonzero(fixed)]
        (work / "frozen").write_text(str(len(indices)) + "\n" + "\n".join(indices) + "\n", encoding="utf-8")
    if spec.get("vector") is not None:
        write_vector(work / "vector.dump", np.asarray(spec["vector"], dtype=np.float64))
    if spec.get("target") is not None:
        write_vector(work / "target.vector", np.asarray(spec["target"], dtype=np.float64), skip_line="reference")
    (work / "odata").write_text(
        odata_text(
            python=str(spec["python"]),
            relay_root=Path(spec["relay_root"]),
            method=method,
            n_coordinates=3 * n_free,
            force_tol=float(spec["force_tol"]),
            periodic=periodic,
            frozen=bool(np.any(fixed)),
            read_vector=read_vector,
            optim=spec.get("optim") or {},
        ),
        encoding="utf-8",
    )
    counter = work / "counter.json"
    job = {
        "potential": spec["potential"],
        "cell": [float(item) for item in cell],
        "pbc": [bool(item) for item in spec["pbc"]],
        "cap": int(spec["cap"]),
        "counter": str(counter),
        "natoms": natoms,
    }
    (work / "job.json").write_text(json.dumps(job), encoding="utf-8")
    env = os.environ.copy()
    env["OPTBENCH_OPTIM_JOB"] = str(work / "job.json")
    env["OPTBENCH_ROOT"] = str(spec["repo"])
    env["PYTHONPATH"] = str(spec["repo"]) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    log_path = work / "optim.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(spec["binary"])],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=float(spec.get("timeout") or 1200))
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -9
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    calls, capped, last_positions = read_counter(counter)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    coords = read_coordinates(work, natoms)
    if coords is None and last_positions is not None and last_positions.size >= 3 * natoms:
        coords = np.asarray(last_positions, dtype=np.float64).reshape(natoms, 3)
    vector = read_vector_dump(work / "vector.dump", 3 * natoms) if method != "minimize" else None
    if coords is not None and np.any(fixed):
        coords = np.array(coords, copy=True)
        coords[fixed] = positions[fixed]
    converged = ("**** CONVERGED ****" in text) or ("Converged to TS" in text) or ("overlap converged" in text)
    error = None
    if timed_out:
        error = "timeout"
    elif coords is None and calls < int(spec["cap"]):
        tail = "\n".join(text.splitlines()[-30:])
        error = f"exit {returncode}\n{tail}"
    return {
        "calls": calls,
        "capped": capped or calls >= int(spec["cap"]),
        "returncode": int(returncode),
        "coords": coords,
        "vector": vector,
        "converged": converged,
        "error": error,
        "timed_out": timed_out,
    }


def _surface(spec: dict, positions: np.ndarray, fixed: np.ndarray):
    free = np.flatnonzero(~fixed)
    raw = potential_from_problem({"potential": spec["potential"]})
    return HarnessSurface(_bind(raw, positions, free, spec["cell"])), free


def _judge_ts(spec: dict, outcome: dict) -> dict:
    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    cap = int(spec["cap"])
    force_tol = float(spec["force_tol"])
    calls = int(outcome["calls"])
    name = str(spec["name"])
    coords = outcome.get("coords")
    if coords is None or not np.all(np.isfinite(coords)):
        reasons = ["crash"] if calls < cap and not outcome.get("capped") else ["cap"]
        success, failure = _outcome(reasons)
        return _base_record(name, calls, success, failure, reasons, error=outcome.get("error"))
    surface, free = _surface(spec, positions, fixed)
    point = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[free].reshape(-1)
    with surface.verification():
        energy, gradient = surface.energy_forces(point)
    raw_eigenvalues, vectors = _hessian_spectrum(surface, point)
    project = not bool(np.any(fixed))
    if project:
        eigenvalues, removed = _vibrational_eigenvalues(point, raw_eigenvalues, vectors)
    else:
        eigenvalues, removed = list(raw_eigenvalues), 0
    index = negative_index(eigenvalues, 1e-6)
    judged = judge_stationary(
        kind="saddle",
        eigenvalues=eigenvalues,
        energy=float(energy),
        coords=np.asarray(point, dtype=np.float64).reshape(-1, 3).tolist(),
        reference_energy=float(energy),
        reference_coords=np.asarray(point, dtype=np.float64).reshape(-1, 3).tolist(),
        energy_tol=0.0,
        rmsd_tol=0.0,
    )
    reasons = [reason for reason in judged["reasons"] if reason not in {"rmsd", "energy"}]
    force_norm = _norm(gradient)
    if calls >= cap or outcome.get("capped"):
        reasons.append("cap")
    if not math.isfinite(force_norm) or force_norm > force_tol + 1e-10:
        reasons.append("force_norm")
    reasons = list(dict.fromkeys(reasons))
    success, failure = _outcome(reasons)
    record = _base_record(name, calls, success, failure, reasons)
    record.update(
        {
            "verify_calls": int(surface.verify_calls),
            "energy": float(energy),
            "force_norm": float(force_norm),
            "hessian_index": int(index),
            "cartesian_index": int(negative_index(raw_eigenvalues, 1e-6)),
            "rigid_modes_removed": int(removed),
            "expected_index": 1,
            "lowest_eigenvalue": float(eigenvalues[0]) if eigenvalues else None,
            "lowest_eigenvalues": [float(value) for value in eigenvalues[:8]],
            "converged": bool(outcome.get("converged")) and success,
            "optim_returncode": outcome.get("returncode"),
        }
    )
    return record


def _judge_mode(spec: dict, outcome: dict) -> dict:
    cap = int(spec["cap"])
    calls = int(outcome["calls"])
    name = str(spec["name"])
    fixed = np.asarray(spec["fixed"], dtype=bool)
    reference = _unit(np.asarray(spec["reference"], dtype=np.float64))
    vector = outcome.get("vector")
    if vector is None or not np.all(np.isfinite(vector)):
        reasons = ["crash"] if calls < cap and not outcome.get("capped") else ["cap", "mode_overlap"]
        success, failure = _outcome(reasons)
        return _base_record(name, calls, success, failure, reasons, error=outcome.get("error"))
    free = vector.reshape(len(fixed), 3)[~fixed].reshape(-1)
    overlap = float(abs(np.dot(_unit(free), reference)))
    reasons = []
    if calls >= cap or outcome.get("capped"):
        reasons.append("cap")
    if not (overlap >= float(spec["overlap"]) and calls < cap):
        reasons.append("mode_overlap")
    reasons = list(dict.fromkeys(reasons))
    success, failure = _outcome(reasons)
    record = _base_record(name, calls, success, failure, reasons)
    record.update(
        {
            "overlap": overlap,
            "converged": bool(success),
            "hessian_index": None,
            "expected_index": None,
            "optim_returncode": outcome.get("returncode"),
        }
    )
    return record


def _base_record(name: str, calls: int, success: bool, failure: str | None, reasons: list[str], error: str | None = None) -> dict:
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": bool(success),
        "failure": failure,
        "reasons": reasons,
        "converged": False,
    }
    if error:
        record["error"] = error[-2000:]
    return record


def _prepare(spec: dict) -> dict:
    try:
        outcome = _launch(spec)
        if spec["task"] == "mode":
            return _judge_mode(spec, outcome)
        if spec["task"] == "raw":
            coords = outcome.get("coords")
            vector = outcome.get("vector")
            return {
                "calls": int(outcome["calls"]),
                "capped": bool(outcome.get("capped")),
                "coords": None if coords is None else coords.tolist(),
                "vector": None if vector is None else vector.reshape(-1).tolist(),
                "converged": bool(outcome.get("converged")),
                "error": outcome.get("error"),
                "returncode": outcome.get("returncode"),
            }
        return _judge_ts(spec, outcome)
    except Exception as exc:
        reasons = ["crash"]
        success, failure = _outcome(reasons)
        record = _base_record(str(spec.get("name") or "start"), 0, success, failure, reasons, error=traceback.format_exc())
        record["exception"] = str(exc)
        return record


def _common(spec_base: dict, *, method: str, cap: int, vector=None, target=None, task: str, name: str, positions, timeout: float) -> dict:
    prepared = dict(spec_base)
    prepared.update(
        {
            "method": method,
            "cap": int(cap),
            "vector": vector,
            "target": target,
            "task": task,
            "name": name,
            "positions": positions,
            "timeout": timeout,
        }
    )
    return prepared


def _geometry_specs(root: Path, problem: dict, template: dict, work: Path, binary: Path, relay_root: Path) -> list[dict]:
    rows = _geometries(root, problem, work)
    expected = int((problem.get("sample") or {}).get("starts") or len(rows))
    if len(rows) != expected:
        raise RuntimeError(f"{problem['id']} has {len(rows)} geometries, the problem states {expected}")
    optim = _section(template, "Optim")
    method = str(_section(template, "Main").get("method") or optim.get("method") or "hef-ffd")
    task = "mode" if problem["id"] == "lowest-eigenvector/lj38" else "saddle"
    convergence = problem.get("convergence") or {}
    force_tol = float(_section(template, "Optimizer").get("converged_force", convergence.get("force_norm", 1e-3)))
    cap = int(convergence.get("max_force_calls") or _section(template, "Optimizer").get("max_iterations") or 10000)
    potential = problem.get("potential") or {}
    symbol = "Pt" if potential.get("kind") == "morse" else "C"
    pbc_raw = potential.get("pbc") or [False, False, False]
    specs = []
    for row in rows:
        positions, fixed, cell = load_con(Path(row["path"]))
        vector = None
        target = None
        if task == "mode":
            reference = read_mode(Path(row["mode"]).read_text(encoding="utf-8", errors="replace"))
            initial = read_mode(Path(row["initial_mode"]).read_text(encoding="utf-8", errors="replace"))
            target = _full_vector(reference, fixed)
            vector = _full_vector(initial, fixed)
            reference_free = reference.reshape(-1, 3)[~fixed].reshape(-1)
        else:
            reference_free = None
            if row.get("reactant") is not None:
                other, other_fixed, _cell = load_con(Path(row["reactant"]))
                direction = positions.copy()
                direction[~fixed] = positions[~fixed] - other[~other_fixed]
                direction[fixed] = 0.0
                scale = float(np.linalg.norm(direction))
                if scale > 0.0:
                    vector = (direction / scale).reshape(-1)
        periodic = any(bool(item) for item in pbc_raw)
        specs.append(
            {
                "name": row["name"],
                "positions": positions,
                "fixed": fixed,
                "cell": cell,
                "potential": potential,
                "pbc": [bool(item) for item in pbc_raw],
                "periodic": periodic,
                "symbol": symbol,
                "method": method,
                "task": task,
                "cap": cap,
                "force_tol": force_tol,
                "optim": optim,
                "vector": vector,
                "target": target,
                "reference": None if reference_free is None else reference_free,
                "overlap": float((problem.get("sample") or {}).get("overlap", 0.99)),
                "python": sys.executable,
                "relay_root": str(relay_root),
                "binary": str(binary),
                "repo": str(root),
                "work": str(work / "runs" / row["name"]),
                "timeout": 1800.0 if task == "saddle" else 900.0,
            }
        )
    return specs


def _summarize(records: list[dict], template: dict, cap: int, problem_id: str) -> tuple[str, dict]:
    ok = [row for row in records if row["success"]]
    counts = [int(row["force_calls"]) for row in ok]
    lines = []
    if counts:
        lines.append(f"force_calls {half_even(sum(counts) / len(counts))}")
        lines.append(f"force_calls_min {min(counts)}")
        lines.append(f"force_calls_median {float(np.median(counts))}")
        lines.append(f"force_calls_max {max(counts)}")
    lines.append(f"nfailed {len(records) - len(ok)}")
    metrics = score_v2(records, cap)
    lines.append(f"par2_mean {metrics['par2_mean']:.10f}")
    lines.append(f"successes {metrics['successes']}")
    lines.append(f"wilson_low {metrics['wilson_low']:.10f}")
    lines.append(f"wilson_high {metrics['wilson_high']:.10f}")
    lines.append(f"par2_median {metrics['median']:.10f}")
    lines.append(f"n_cap {metrics['n_cap']}")
    lines.append(f"n_wrong_stationary {metrics['n_wrong_stationary']}")
    lines.append(f"n_divergence {metrics['n_divergence']}")
    lines.append(f"n_crash {metrics['n_crash']}")
    lines.append("counter harness")
    lines.append(f"code {template.get('code', 'optim')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem_id}")
    lines.append(f"cap {cap}")
    lines.append(f"n {len(records)}")
    return "\n".join(lines) + "\n", metrics


def _public(records: list[dict], template: dict, provenance: dict) -> list[dict]:
    rows = []
    for record in records:
        copied = {key: value for key, value in record.items() if key not in {"coords", "vector", "x"}}
        copied["code"] = provenance["code"]
        copied["code_version"] = provenance["code_version"]
        copied["commit"] = provenance["commit"]
        copied["tuning"] = template.get("tuning", "")
        copied["template"] = template["id"]
        rows.append(copied)
    return rows


def _run_pool(specs: list[dict], workers: int, partial: Path | None = None) -> list[dict]:
    """Run starts. A partial file receives each record as it finishes."""
    if workers <= 1:
        records = []
        for spec in specs:
            record = _prepare(spec)
            records.append(record)
            if partial is not None:
                _append_jsonl(partial, record)
        return records
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_prepare, spec) for spec in specs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            if partial is not None:
                _append_jsonl(partial, record)
    return records


def _load_done(path: Path, template_id: str) -> dict[str, dict]:
    if not path.is_file():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("template") not in {None, template_id}:
            continue
        if row.get("start"):
            done[str(row["start"])] = row
    return done


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Run one OPTIM template on the problem's starts."""
    from .campaign import _check_stop

    if problem["id"] == "saddle-search/pt-island":
        return _run_saddle(root, problem, template, output=output, work=work, resume=resume)
    binary = optim_binary()
    relay_root = install_relay()
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    specs = _geometry_specs(root, problem, template, scratch, binary, relay_root)
    if limit and limit > 0:
        specs = specs[: int(limit)]
    cap = int(specs[0]["cap"]) if specs else _check_stop(problem, template)
    partial_path = output.with_suffix(".starts.jsonl")
    if not resume and partial_path.is_file():
        partial_path.unlink()
    done = _load_done(partial_path, template["id"]) if resume else {}
    pending = [spec for spec in specs if spec["name"] not in done]
    fresh = _run_pool(pending, workers, partial_path) if pending else []
    by_name = dict(done)
    for record in fresh:
        by_name[str(record["start"])] = record
    records = [by_name[spec["name"]] for spec in specs if spec["name"] in by_name]
    partial = len(records) < int((problem.get("sample") or {}).get("starts") or len(specs))
    text, metrics = _summarize(records, template, cap, problem["id"])
    if partial:
        text += "partial true\nranked false\n"
    provenance = collect_provenance(
        code=str(template.get("code", "optim")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={
            "template": template["id"],
            "problem": problem["id"],
            "cap": cap,
            "method": specs[0]["method"] if specs else "",
            "socket": "ase-fmlip",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    public = _public(records, template, provenance)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    successes = sum(1 for row in records if row["success"])
    print(
        f"{template['id']} {problem['id']} starts {len(records)} successes {successes} "
        f"par2 {metrics['par2_mean']:.6g}",
        flush=True,
    )
    return {
        "force_calls": half_even(float(np.mean([row["force_calls"] for row in records if row["success"]]))) if successes else None,
        "nfailed": metrics["nfailed"],
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
        "records": records,
    }


def _endpoint_axis(vector: np.ndarray | None, fixed: np.ndarray, kick: np.ndarray) -> np.ndarray:
    if vector is not None and np.all(np.isfinite(vector)):
        free = np.asarray(vector, dtype=np.float64).reshape(len(fixed), 3)[~fixed].reshape(-1)
        if float(np.linalg.norm(free)) > 1e-12:
            return _unit(free)
    return _unit(np.asarray(kick, dtype=np.float64))


def _minimize_calls(base: dict, positions: np.ndarray, cap: int, work: Path) -> tuple[np.ndarray | None, int, bool]:
    spec = _common(
        base,
        method="minimize",
        cap=cap,
        task="raw",
        name="minimize",
        positions=positions,
        timeout=900.0,
    )
    spec["work"] = str(work)
    outcome = _prepare(spec)
    coords = outcome.get("coords")
    point = None if coords is None else np.asarray(coords, dtype=np.float64)
    return point, int(outcome["calls"]), bool(outcome.get("converged"))


def _run_saddle(root: Path, problem: dict, template: dict, *, output: Path, work: Path | None, resume: bool) -> dict:
    """One displacement campaign. The cap is the climb plus both ends."""
    binary = optim_binary()
    relay_root = install_relay()
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    from .rgsaddle_minmode import _stored

    positions, fixed, cell = load_con(_stored(root, con_path))
    search = _section(template, "Search")
    optim = _section(template, "Optim")
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    repetitions = int(search.get("repetitions", 1))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    magnitude = float(search.get("displace_magnitude", 0.1))
    connect_rmsd = float(search.get("rmsd_connect", 0.5))
    max_searches = int(search.get("max_searches", 20000))
    climb_cap = int(_section(template, "Optimizer").get("max_iterations", 1000))
    force_tol = float(_section(template, "Optimizer").get("converged_force", 1.0e-3))
    search_cap = climb_cap + _CLIMB_OVERSHOOT + 2 * _ENDPOINT_BUDGET
    potential = problem["potential"]
    pbc_raw = potential.get("pbc") or [False, False, False]
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    base = {
        "fixed": fixed,
        "cell": cell,
        "potential": potential,
        "pbc": [bool(item) for item in pbc_raw],
        "periodic": any(bool(item) for item in pbc_raw),
        "symbol": "Pt",
        "force_tol": force_tol,
        "optim": optim,
        "python": sys.executable,
        "relay_root": str(relay_root),
        "binary": str(binary),
        "repo": str(root),
    }
    free = np.flatnonzero(~fixed)
    header_path = output.with_suffix(".min.json")
    partial_path = output.with_suffix(".starts.jsonl")
    all_records: list[dict] = []
    comments = []
    failed_reps = 0
    for rep in range(1, repetitions + 1):
        saved = []
        if resume and partial_path.is_file() and header_path.is_file():
            header = json.loads(header_path.read_text(encoding="utf-8"))
            if header.get("template") == template["id"] and int(header.get("rep", 0)) == rep:
                saved = [
                    json.loads(line)
                    for line in partial_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and json.loads(line).get("rep") == rep
                ]
        if saved and resume:
            minimum = np.asarray(header["minimum"], dtype=np.float64)
            min_energy = float(header["min_energy"])
            minimize_calls = int(header["minimize_calls"])
            min_ok = bool(header["min_ok"])
            rng = np.random.default_rng(20261015 + rep)
            if saved[-1].get("rng_state"):
                state = saved[-1]["rng_state"]
                rng.bit_generator.state = state["bit_generator"]
            found = [np.asarray(row["saddle_free"], dtype=np.float64) for row in saved if row.get("saddle_free") is not None]
            connected = sum(1 for row in saved if row.get("success"))
            searches = len(saved)
            for row in saved:
                all_records.append({key: value for key, value in row.items() if key not in {"rng_state", "saddle_free"}})
            print(f"{template['id']} rep {rep} resume searches {searches} connected {connected}", flush=True)
        else:
            minimum_full, minimize_calls, min_ok = _minimize_calls(
                base, positions, _MINIMIZE_BUDGET, scratch / f"rep{rep}-min"
            )
            if minimum_full is None:
                raise RuntimeError("initial OPTIM minimization returned no coordinates")
            minimum = minimum_full.reshape(-1, 3)[free].reshape(-1)
            raw = potential_from_problem(problem)
            surface = HarnessSurface(_bind(raw, positions, free, cell))
            with surface.verification():
                min_energy, _gradient = surface.energy_forces(minimum)
            rng = np.random.default_rng(20261015 + rep)
            found = []
            connected = 0
            searches = 0
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(
                json.dumps(
                    {
                        "template": template["id"],
                        "rep": rep,
                        "minimum": minimum.tolist(),
                        "min_energy": float(min_energy),
                        "minimize_calls": minimize_calls,
                        "min_ok": bool(min_ok),
                    }
                ),
                encoding="utf-8",
            )
            if partial_path.is_file() and not resume:
                partial_path.unlink()
        while connected < goal and searches < max_searches:
            searches += 1
            atom = int(listed[int(rng.integers(0, len(listed)))])
            slots = np.flatnonzero(free == atom)
            if len(slots) != 1:
                raise RuntimeError(f"displace atom {atom} is not one free atom")
            slot = int(slots[0])
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            kick = np.zeros(3 * len(free), dtype=np.float64)
            kick[3 * slot : 3 * slot + 3] = direction
            start = positions.copy()
            start[free] = minimum.reshape(-1, 3)
            start.reshape(-1, 3)[free[slot]] = start.reshape(-1, 3)[free[slot]] + magnitude * direction
            vector = _full_vector(kick, fixed)
            spec = _common(
                base,
                method=str(_section(template, "Main").get("method") or "hef-ffd"),
                cap=climb_cap,
                vector=vector,
                task="raw",
                name=f"rep{rep}-search{searches}",
                positions=start,
                timeout=900.0,
            )
            spec["work"] = str(scratch / spec["name"])
            outcome = _prepare(spec)
            climb = int(outcome["calls"])
            record = {
                "schema": "optbench.start/v2",
                "start": spec["name"],
                "rep": rep,
                "counter": "harness",
                "comparable": True,
                "success": False,
                "reasons": ["force_norm"],
                "converged": bool(outcome.get("converged")),
                "hessian_index": None,
            }
            saved_free = None
            point = None if outcome.get("coords") is None else np.asarray(outcome["coords"], dtype=np.float64)
            finite = point is not None and np.all(np.isfinite(point))
            if not finite:
                record["reasons"] = ["crash"] if climb < climb_cap and not outcome.get("capped") else ["cap", "force_norm"]
                if outcome.get("error"):
                    record["error"] = str(outcome["error"])[-500:]
            else:
                free_point = point.reshape(-1, 3)[free].reshape(-1)
                raw = potential_from_problem(problem)
                surface = HarnessSurface(_bind(raw, positions, free, cell))
                with surface.verification():
                    energy, gradient = surface.energy_forces(free_point)
                force_norm = _norm(gradient)
                record["force_norm"] = float(force_norm)
                record["energy"] = float(energy)
                if force_norm <= force_tol + 1e-10 and climb < climb_cap:
                    raw_eigenvalues, vectors = _hessian_spectrum(surface, free_point)
                    index = negative_index(list(raw_eigenvalues), 1e-6)
                    record["hessian_index"] = int(index)
                    record["lowest_eigenvalue"] = float(raw_eigenvalues[0])
                    if index == 1:
                        axis = _endpoint_axis(
                            None if outcome.get("vector") is None else np.asarray(outcome["vector"], dtype=np.float64),
                            fixed,
                            kick,
                        )
                        plus_full = point.copy()
                        minus_full = point.copy()
                        plus_full.reshape(-1, 3)[free] = (free_point + _PUSH * axis).reshape(-1, 3)
                        minus_full.reshape(-1, 3)[free] = (free_point - _PUSH * axis).reshape(-1, 3)
                        plus, plus_calls, _plus_ok = _minimize_calls(base, plus_full, _ENDPOINT_BUDGET, scratch / f"{spec['name']}-plus")
                        minus, minus_calls, _minus_ok = _minimize_calls(base, minus_full, _ENDPOINT_BUDGET, scratch / f"{spec['name']}-minus")
                        record["endpoint_calls"] = int(plus_calls + minus_calls)
                        if plus is None or minus is None:
                            record["reasons"] = ["divergence"]
                        else:
                            plus_free = plus.reshape(-1, 3)[free]
                            minus_free = minus.reshape(-1, 3)[free]
                            minimum_xyz = np.asarray(minimum, dtype=np.float64).reshape(-1, 3)
                            plus_rmsd = _cartesian_rmsd(plus_free, minimum_xyz)
                            minus_rmsd = _cartesian_rmsd(minus_free, minimum_xyz)
                            links = (plus_rmsd <= connect_rmsd) ^ (minus_rmsd <= connect_rmsd)
                            novel = all(
                                _cartesian_rmsd(free_point.reshape(-1, 3), old.reshape(-1, 3)) > connect_rmsd
                                for old in found
                            )
                            barrier = float(energy - min_energy)
                            record["barrier"] = barrier
                            record["rmsd_plus"] = plus_rmsd
                            record["rmsd_minus"] = minus_rmsd
                            record["reasons"] = []
                            if novel:
                                found.append(free_point)
                                saved_free = free_point
                            if links and novel and barrier < barrier_max:
                                connected += 1
                                record["success"] = True
                            else:
                                if not links:
                                    record["reasons"].append("not_connected")
                                if not novel:
                                    record["reasons"].append("duplicate")
                                if barrier >= barrier_max:
                                    record["reasons"].append("barrier")
                    else:
                        record["reasons"] = ["hessian_index"]
                elif climb >= climb_cap or outcome.get("capped"):
                    record["reasons"] = ["cap", "force_norm"]
                else:
                    record["reasons"] = ["force_norm"]
            endpoint_calls = int(record.get("endpoint_calls") or 0)
            record["force_calls"] = climb + endpoint_calls
            record["climb_calls"] = climb
            success, failure = _outcome(record["reasons"])
            record["success"] = success
            record["failure"] = failure
            public = {key: value for key, value in record.items() if key != "saddle_free"}
            all_records.append(public)
            stored = dict(record)
            stored["rng_state"] = {"bit_generator": rng.bit_generator.state}
            if saved_free is not None:
                stored["saddle_free"] = saved_free.tolist()
            _append_jsonl(partial_path, stored)
            if searches == 1 or searches % 10 == 0 or connected >= goal:
                print(
                    f"{template['id']} rep {rep} searches {searches} connected {connected} "
                    f"calls {record['force_calls']} failure {failure}",
                    flush=True,
                )
                _flush_saddle(
                    output,
                    template,
                    problem["id"],
                    all_records,
                    search_cap=search_cap,
                    goal=goal,
                    repetitions=repetitions,
                    failed_reps=failed_reps + (1 if connected < goal else 0),
                    comments=[
                        f"# rep {rep} force_calls {sum(int(row['force_calls']) for row in all_records if row.get('rep', rep) == rep)} "
                        f"unique_saddles {len(found)} fraction_connected {connected / searches if searches else 0.0} "
                        f"total_searches {searches} connected {connected} minimize_calls {minimize_calls} "
                        f"minimum_converged {int(bool(min_ok))} seed {20261015 + rep}"
                    ],
                    reached=connected >= goal,
                )
        fraction = connected / searches if searches else 0.0
        if connected < goal:
            failed_reps += 1
        comments.append(
            f"# rep {rep} force_calls {sum(int(row['force_calls']) for row in all_records if row.get('rep', rep) == rep)} "
            f"unique_saddles {len(found)} fraction_connected {fraction} total_searches {searches} "
            f"connected {connected} minimize_calls {minimize_calls} minimum_converged {int(bool(min_ok))} seed {20261015 + rep}"
        )
    text = "\n".join(comments) + "\n"
    totals = []
    for line in comments:
        parts = line.split()
        totals.append(int(parts[parts.index("force_calls") + 1]))
    if totals:
        text += f"force_calls {float(np.mean(totals))}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
    metrics = score_v2(
        [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in all_records],
        search_cap,
    ) if all_records else None
    reached = all(int(line.split()[line.split().index("connected") + 1]) >= goal for line in comments) if comments else False
    text += (
        f"code {template.get('code', 'optim')}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(all_records)}\n"
    )
    if metrics is not None:
        text += (
            f"par2_mean {metrics['par2_mean']}\n"
            f"successes {metrics['successes']}\n"
            f"wilson_low {metrics['wilson_low']}\n"
            f"wilson_high {metrics['wilson_high']}\n"
            f"par2_median {metrics['median']}\n"
            f"n_cap {metrics['n_cap']}\n"
            f"n_wrong_stationary {metrics['n_wrong_stationary']}\n"
            f"n_divergence {metrics['n_divergence']}\n"
            f"n_crash {metrics['n_crash']}\n"
            f"v2_nfailed {metrics['nfailed']}\n"
        )
    if not reached:
        text += "partial true\nranked false\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    provenance = collect_provenance(
        code=str(template.get("code", "optim")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={"template": template["id"], "problem": problem["id"], "cap": search_cap},
    )
    public_rows = []
    for record in all_records:
        copied = {key: value for key, value in record.items() if key not in {"rng_state", "saddle_free"}}
        copied["code"] = provenance["code"]
        copied["code_version"] = provenance["code_version"]
        copied["commit"] = provenance["commit"]
        copied["tuning"] = template.get("tuning", "")
        copied["template"] = template["id"]
        public_rows.append(copied)
    write_starts(partial_path, public_rows)
    print(
        f"{template['id']} searches {len(all_records)} connected "
        f"{sum(1 for row in all_records if row['success'])}",
        flush=True,
    )
    return {
        "force_calls": metrics["par2_mean"] if metrics else None,
        "nfailed": failed_reps,
        "starts": len(all_records),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": None if metrics is None else metrics["par2_mean"],
        "median": None if metrics is None else metrics["median"],
        "records": all_records,
    }


def _flush_saddle(
    output: Path,
    template: dict,
    problem_id: str,
    records: list[dict],
    *,
    search_cap: int,
    goal: int,
    repetitions: int,
    failed_reps: int,
    comments: list[str],
    reached: bool,
) -> None:
    """Rewrite the campaign file from the searches finished so far."""
    text = "\n".join(comments) + "\n"
    totals = []
    for line in comments:
        parts = line.split()
        totals.append(int(parts[parts.index("force_calls") + 1]))
    if totals:
        text += f"force_calls {float(np.mean(totals))}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
    metrics = (
        score_v2(
            [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records],
            search_cap,
        )
        if records
        else None
    )
    text += (
        f"code {template.get('code', 'optim')}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem_id}\n"
        f"counter harness\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(records)}\n"
    )
    if metrics is not None:
        text += (
            f"par2_mean {metrics['par2_mean']}\n"
            f"successes {metrics['successes']}\n"
            f"wilson_low {metrics['wilson_low']}\n"
            f"wilson_high {metrics['wilson_high']}\n"
            f"par2_median {metrics['median']}\n"
            f"median_low {metrics['median_low']}\n"
            f"median_high {metrics['median_high']}\n"
            f"n_cap {metrics['n_cap']}\n"
            f"n_wrong_stationary {metrics['n_wrong_stationary']}\n"
            f"n_divergence {metrics['n_divergence']}\n"
            f"n_crash {metrics['n_crash']}\n"
            f"v2_nfailed {metrics['nfailed']}\n"
        )
    if not reached:
        text += "partial true\nranked false\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(type(value).__name__)

