"""SaddleSearch.jl campaigns on the harness potential.

The Julia process loads the pinned source and calls ``run!``. Every
energy or gradient request is one ``HarnessCounter`` entry, with the
in-process cache. A lowest-mode job keeps the translation step at zero
and stops on the problem overlap. A saddle job stops on the Euclidean
force norm. The Hessian that decides the index is a verification cost.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import statistics
import subprocess
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from .counter import ForceCallCap
from .rgsaddle_minmode import (
    _SADDLE_CLIMB_OVERSHOOT,
    _SADDLE_ENDPOINT_BUDGET,
    HarnessSurface,
    _append_partial,
    _bind,
    _cartesian_rmsd,
    _finite_point,
    _geometries,
    _hessian_spectrum,
    _lbfgs_minimize,
    _load_partial,
    _norm,
    _outcome,
    _rng_restore,
    _rng_state,
    _saddle_record,
    _unit,
    _vibrational_eigenvalues,
)
from .score import half_even, score_v2
from .stationary import negative_index
from .surfaces import load_con, potential_from_problem, read_mode
from .v2 import collect_provenance, write_starts

_SESSION = None
_DRIVER = Path(__file__).with_name("saddlesearch_driver.jl")


def julia_bin() -> str:
    env = os.environ.get("JULIA")
    if env:
        return env
    found = shutil.which("julia")
    if found:
        return found
    home = Path.home() / "julia" / "bin" / "julia"
    if home.is_file():
        return str(home)
    raise RuntimeError("set JULIA to the Julia 1.10 binary")


def source_dir() -> Path:
    env = os.environ.get("SADDLESEARCHJL_SRC")
    if not env:
        home = Path.home() / "src" / "SaddleSearch.jl"
        if (home / "src" / "SaddleSearch.jl").is_file():
            return home
        raise RuntimeError("set SADDLESEARCHJL_SRC to the pinned SaddleSearch.jl checkout")
    path = Path(env)
    if not (path / "src" / "SaddleSearch.jl").is_file():
        raise RuntimeError(f"SADDLESEARCHJL_SRC is not a SaddleSearch.jl checkout: {path}")
    return path


def _assert_source_commit(expected: str) -> None:
    if not expected:
        return
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=source_dir(),
        text=True,
    ).strip()
    if head != expected:
        raise RuntimeError(f"SaddleSearch.jl HEAD {head} does not match the template commit {expected}")


def _hex(value: float) -> str:
    import struct

    bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    return f"{bits:016x}"


def _unhex(text: str) -> float:
    import struct

    return struct.unpack("<d", struct.pack("<Q", int(text, 16)))[0]


def _hex_line(values) -> str:
    """IEEE hex of a vector, high nibble first, matching ``_hex``."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    if arr.size == 0:
        return ""
    bits = np.ascontiguousarray(arr.view(np.uint64).byteswap())
    blob = bits.tobytes().hex()
    return " ".join(blob[index : index + 16] for index in range(0, len(blob), 16))


def _floats_from_hex_line(text: str, width: int) -> np.ndarray:
    raw = bytes.fromhex(text)
    if len(raw) != width * 8:
        raise ValueError(f"expected {width} hex floats and read {len(raw) // 8}")
    native = np.frombuffer(raw, dtype=">u8").astype(np.uint64)
    return native.view(np.float64)


class JuliaSession:
    """One long-lived Julia process speaking the driver protocol."""

    def __init__(self, proc: subprocess.Popen, stderr: deque[str]):
        self.proc = proc
        self.stderr = stderr
        self._write_lock = threading.Lock()

    @classmethod
    def start(cls) -> JuliaSession:
        src = source_dir()
        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["JULIA_NUM_THREADS"] = "1"
        proc = subprocess.Popen(
            [julia_bin(), f"--project={src}", str(_DRIVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        stderr: deque[str] = deque(maxlen=40)
        threading.Thread(target=_drain, args=(proc, stderr), daemon=True).start()
        session = cls(proc, stderr)
        while True:
            line = session._readline()
            if line == "ready":
                return session
            if line == "":
                raise RuntimeError(f"SaddleSearch driver exited {proc.poll()} {list(stderr)}")

    def close(self) -> None:
        proc = self.proc
        if proc.poll() is not None:
            return
        try:
            self._writeline("bye")
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()

    def _readline(self) -> str:
        if self.proc.stdout is None:
            return ""
        line = self.proc.stdout.readline()
        if line == "":
            return ""
        return line.rstrip("\n")

    def _writeline(self, text: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("SaddleSearch driver has no stdin")
        with self._write_lock:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()

    def _write_vec(self, name: str, values) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self._writeline(name)
        self._writeline(str(int(array.size)))
        if array.size:
            self._writeline(_hex_line(array))

    def search(self, potential: HarnessSurface, settings: dict, x0, v0, reference) -> dict:
        """Run one solver. ``potential`` receives every harness evaluation."""
        self._writeline("job")
        for key, value in settings.items():
            self._writeline(f"{key} {value}")
        self._write_vec("x", x0)
        self._write_vec("v", v0)
        self._write_vec("ref", np.zeros(0) if reference is None else reference)
        self._writeline("end")
        while True:
            line = self._readline()
            if line == "eval":
                self._serve(potential)
                continue
            if line == "done":
                return self._read_done()
            if line == "":
                raise RuntimeError(
                    f"SaddleSearch driver exited during a search {self.proc.poll()} {list(self.stderr)}"
                )
            raise RuntimeError(f"unexpected driver line {line}")

    def _serve(self, potential: HarnessSurface) -> None:
        width = int(self._readline())
        line = self._readline()
        try:
            coords = _floats_from_hex_line(line, width)
        except ValueError as exc:
            self._writeline("err")
            self._writeline(str(exc)[:500])
            return
        try:
            energy, gradient = potential.energy_forces(coords)
        except ForceCallCap:
            self._writeline("cap")
            return
        except Exception as exc:
            self._writeline("err")
            self._writeline(str(exc).replace("\n", " ")[:500])
            return
        flat = np.asarray(gradient, dtype=np.float64).reshape(-1)
        self._writeline("ok")
        self._writeline(str(int(potential.calls)))
        self._writeline(_hex(float(energy)))
        self._writeline(str(int(flat.size)))
        self._writeline(_hex_line(flat))

    def _read_done(self) -> dict:
        fields: dict[str, str] = {}
        vectors: dict[str, np.ndarray] = {}
        while True:
            line = self._readline()
            if line == "end":
                break
            if line == "":
                raise RuntimeError(f"driver closed before end {list(self.stderr)}")
            if line in {"x", "v", "gradient"}:
                count = int(self._readline())
                if count == 0:
                    vectors[line] = np.zeros(0, dtype=np.float64)
                else:
                    parts = self._readline()
                    vectors[line] = _floats_from_hex_line(parts, count)
                continue
            key, value = line.split(" ", 1)
            fields[key] = value
        def optional(name: str) -> float:
            raw = fields.get(name, "")
            if not raw:
                return float("nan")
            return _unhex(raw)

        return {
            "status": fields.get("status", "error"),
            "message": fields.get("message", ""),
            "library_numdE": int(fields.get("library_numdE", "0")),
            "steps": int(fields.get("steps", "0")),
            "energy": optional("energy"),
            "force_norm": optional("force_norm"),
            "overlap": optional("overlap"),
            "curvature": optional("curvature"),
            "x": vectors.get("x", np.zeros(0, dtype=np.float64)),
            "v": vectors.get("v", np.zeros(0, dtype=np.float64)),
            "gradient": vectors.get("gradient", np.zeros(0, dtype=np.float64)),
        }


def _drain(proc: subprocess.Popen, sink: deque[str]) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        sink.append(line.rstrip())


def _section(template: dict, name: str) -> dict:
    return ((template.get("ini") or {}).get(name) or {})


def _settings(template: dict, cap: int, force_tol: float, task: str, overlap: float) -> dict:
    search = template.get("search") or {}
    extrapolate = search.get("extrapolate", True)
    return {
        "task": task,
        "method": str(search["method"]),
        "force_tol": repr(float(force_tol)),
        "cap": int(cap),
        "len": repr(float(search.get("len", 1e-3))),
        "tol_rot": repr(float(search.get("tol_rot", 1e-2))),
        "overlap": repr(float(overlap)),
        "a_trans": repr(float(search.get("a_trans", 0.002))),
        "a_rot": repr(float(search.get("a_rot", 0.002))),
        "a0_trans": repr(float(search.get("a0_trans", 0.001))),
        "a0_rot": repr(float(search.get("a0_rot", 0.001))),
        "maximum_translation": repr(float(search.get("maximum_translation", 0.001))),
        "max_num_rot": int(search.get("max_num_rot", 1)),
        "trial_angle": repr(float(search.get("trial_angle", math.pi / 4))),
        "trial_trans_step": repr(float(search.get("trial_trans_step", 1e-4))),
        "extrapolate": 0 if extrapolate is False else 1,
        "translation_method": str(search.get("translation_method", "CG")),
    }


def _surface(evaluate, cap: int) -> HarnessSurface:
    surface = HarnessSurface(evaluate)
    surface.rgpot.counter.cap = int(cap)
    return surface


def _evaluate_for(spec: dict):
    if spec.get("evaluate") is not None:
        return spec["evaluate"]
    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    free = np.flatnonzero(~fixed)
    cell = np.asarray(spec["cell"], dtype=np.float64)
    raw = potential_from_problem({"potential": spec["potential"]})
    return _bind(raw, positions, free, cell)


def _free_x(spec: dict) -> np.ndarray:
    if spec.get("x") is not None:
        return np.asarray(spec["x"], dtype=np.float64).reshape(-1).copy()
    positions = np.asarray(spec["positions"], dtype=np.float64)
    fixed = np.asarray(spec["fixed"], dtype=bool)
    return positions[~fixed].reshape(-1).copy()


def _seed_mode(spec: dict, surface: HarnessSurface, x: np.ndarray) -> np.ndarray:
    how = str(spec.get("mode_seed", "gradient"))
    if how == "reactant" and spec.get("reactant") is not None:
        return _unit(x - np.asarray(spec["reactant"], dtype=np.float64))
    if how == "file" and spec.get("file_mode") is not None:
        return _unit(np.asarray(spec["file_mode"], dtype=np.float64))
    _energy, gradient = surface.energy_forces(x)
    return _unit(np.asarray(gradient, dtype=np.float64))


def _crash_record(spec: dict, message: str, calls: int) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": spec["name"],
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "message": message[:500],
        "hessian_index": None,
        "expected_index": 1,
        "converged": False,
    }


def _mode_record(spec: dict, surface: HarnessSurface, outcome: dict, cap: int, reference: np.ndarray) -> dict:
    point = np.asarray(outcome["x"], dtype=np.float64)
    mode = np.asarray(outcome["v"], dtype=np.float64)
    finite = _finite_point(point) and _finite_point(mode) and mode.size == reference.size
    overlap = 0.0
    if finite:
        overlap = float(abs(np.dot(_unit(mode), _unit(reference))))
    reasons: list[str] = []
    if outcome["status"] == "error" and not finite:
        reasons.append("crash")
    else:
        if not (finite and overlap >= float(spec["overlap"]) and surface.calls < cap):
            reasons.append("mode_overlap")
        if surface.calls >= cap or outcome["status"] == "cap":
            reasons.append("cap")
        if outcome["status"] == "error":
            reasons.append("solver")
    success, failure = _outcome(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": spec["name"],
        "force_calls": int(surface.calls),
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "overlap": overlap,
        "converged": bool(success),
        "library_evaluations": int(outcome["library_numdE"]),
        "steps": int(outcome["steps"]),
        "status": outcome["status"],
        "message": outcome["message"],
        "hessian_index": None,
        "expected_index": 1,
    }
    if finite:
        energy = outcome["energy"]
        gradient = outcome["gradient"]
        if gradient.size == point.size and math.isfinite(float(energy)):
            record["energy"] = float(energy)
            record["force_norm"] = _norm(gradient)
        fixed = np.asarray(spec.get("fixed", []), dtype=bool)
        raw, vectors = _hessian_spectrum(surface, point)
        if fixed.size and np.any(fixed):
            eigenvalues, removed = list(raw), 0
        else:
            eigenvalues, removed = _vibrational_eigenvalues(point, raw, vectors)
        record["hessian_index"] = negative_index(eigenvalues, 1e-6)
        record["rigid_modes_removed"] = int(removed)
        record["lowest_eigenvalue"] = float(eigenvalues[0]) if eigenvalues else None
        record["lowest_eigenvalues"] = [float(value) for value in eigenvalues[:8]]
        record["verify_calls"] = int(surface.verify_calls)
        record["cartesian_index"] = negative_index(raw, 1e-6)
    return record


def _open_record(spec: dict, surface: HarnessSurface, outcome: dict, cap: int, force_tol: float) -> dict:
    """A search that is not at the force stop. The index is not required."""
    point = np.asarray(outcome["x"], dtype=np.float64)
    reasons: list[str] = []
    if outcome["status"] == "error" and not _finite_point(point):
        reasons.append("crash")
    force = outcome.get("force_norm")
    finite_force = force is not None and math.isfinite(float(force))
    if surface.calls >= cap or outcome["status"] == "cap":
        reasons.append("cap")
    if not finite_force or float(force) > force_tol + 1e-10:
        if "cap" not in reasons and "crash" not in reasons:
            reasons.append("force_norm")
    if outcome["status"] == "error" and "crash" not in reasons:
        reasons.append("solver")
    success, failure = _outcome(reasons or ["force_norm"])
    record = {
        "schema": "optbench.start/v2",
        "start": spec["name"],
        "force_calls": int(surface.calls),
        "verify_calls": int(surface.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "hessian_index": None,
        "expected_index": 1,
        "converged": False,
        "library_evaluations": int(outcome["library_numdE"]),
        "steps": int(outcome["steps"]),
        "status": outcome["status"],
        "message": outcome["message"],
    }
    if finite_force:
        record["force_norm"] = float(force)
    energy = outcome.get("energy")
    if energy is not None and math.isfinite(float(energy)):
        record["energy"] = float(energy)
    if _finite_point(point):
        record["x"] = [float(value) for value in point]
    mode = np.asarray(outcome["v"], dtype=np.float64)
    if _finite_point(mode):
        record["mode"] = [float(value) for value in mode]
    return record


def _at_force_stop(outcome: dict, surface: HarnessSurface, cap: int, force_tol: float) -> bool:
    if surface.calls >= cap or outcome["status"] == "cap":
        return False
    force = outcome.get("force_norm")
    if force is None or not math.isfinite(float(force)):
        return True
    return float(force) <= force_tol + 1e-10


def _saddle_from_outcome(spec: dict, surface: HarnessSurface, outcome: dict, cap: int, force_tol: float) -> dict:
    point = np.asarray(outcome["x"], dtype=np.float64)
    if not _at_force_stop(outcome, surface, cap, force_tol):
        return _open_record(spec, surface, outcome, cap, force_tol)
    if outcome["status"] == "error" and not _finite_point(point):
        record = _crash_record(spec, outcome["message"] or "solver", surface.calls)
        record["library_evaluations"] = int(outcome["library_numdE"])
        record["steps"] = int(outcome["steps"])
        record["status"] = outcome["status"]
        record["message"] = outcome["message"]
        return record
    fixed = np.asarray(spec.get("fixed", []), dtype=bool)
    try:
        record = _saddle_record(
            spec["name"],
            surface,
            point,
            cap=cap,
            force_tol=force_tol,
            library_converged=outcome["status"] == "converged",
            rotations=0,
            library_evals=int(outcome["library_numdE"]),
            steps=int(outcome["steps"]),
            project_rigid=not (fixed.size and bool(np.any(fixed))),
        )
    except ForceCallCap:
        record = _open_record(spec, surface, outcome, cap, force_tol)
        record["failure"] = "cap"
        record["reasons"] = ["cap"]
        record["success"] = False
        record["message"] = outcome["message"] or "force-call cap during the index"
    record["status"] = outcome["status"]
    record["message"] = outcome["message"]
    record["mode"] = [float(value) for value in np.asarray(outcome["v"], dtype=np.float64)]
    return record


def execute(spec: dict, session: JuliaSession | None = None) -> dict:
    """One start. The session is the process-local driver when omitted."""
    active = session if session is not None else _SESSION
    if active is None:
        raise RuntimeError("SaddleSearch Julia session is not running")
    cap = int(spec["cap"])
    force_tol = float(spec["force_tol"])
    task = str(spec["task"])
    try:
        surface = _surface(_evaluate_for(spec), cap)
        x = _free_x(spec)
        v = _seed_mode(spec, surface, x)
        reference = None
        if task == "mode":
            reference = np.asarray(spec["reference_mode"], dtype=np.float64)
        outcome = active.search(
            surface,
            _settings(spec["template"], cap, force_tol, task, float(spec.get("overlap", 0.99))),
            x,
            v,
            reference,
        )
        if task == "mode":
            record = _mode_record(spec, surface, outcome, cap, reference)
        else:
            record = _saddle_from_outcome(spec, surface, outcome, cap, force_tol)
    except ForceCallCap:
        record = _crash_record(spec, "force-call cap", cap)
        record["failure"] = "cap"
        record["reasons"] = ["cap"]
        record["force_calls"] = cap
        record["success"] = False
    except Exception as exc:
        record = _crash_record(spec, f"{type(exc).__name__}: {exc}", 0)
    record["seed"] = str(spec.get("seed", "0"))
    return record


def _init_worker() -> None:
    global _SESSION
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["JULIA_NUM_THREADS"] = "1"
    _SESSION = JuliaSession.start()


def _worker_execute(spec: dict) -> dict:
    try:
        return execute(spec)
    except Exception as exc:
        global _SESSION
        if _SESSION is not None:
            _SESSION.close()
        _SESSION = JuliaSession.start()
        try:
            return execute(spec)
        except Exception as second:
            return _crash_record(spec, f"{type(second).__name__}: {second}", 0)


def _jsonable(record: dict) -> dict:
    copied = {}
    for key, value in record.items():
        if key in {"x", "mode"}:
            continue
        if isinstance(value, np.ndarray):
            copied[key] = [float(item) for item in value.reshape(-1)]
        elif isinstance(value, np.generic):
            copied[key] = value.item()
        else:
            copied[key] = value
    return copied


def _spec_from_geometry(row: dict, problem: dict, template: dict, cap: int, force_tol: float) -> dict:
    positions, fixed, cell = load_con(Path(row["path"]))
    search = template.get("search") or {}
    reactant = None
    if row.get("reactant"):
        other, other_fixed, _cell = load_con(Path(row["reactant"]))
        reactant = other[~other_fixed].reshape(-1)
    reference = None
    if row.get("mode"):
        full = read_mode(Path(row["mode"]).read_text(encoding="utf-8", errors="replace"))
        reference = full.reshape(-1, 3)[~fixed].reshape(-1)
    file_mode = None
    if str(search.get("mode_seed", "gradient")) == "file" and row.get("initial_mode"):
        full = read_mode(Path(row["initial_mode"]).read_text(encoding="utf-8", errors="replace"))
        file_mode = full.reshape(-1, 3)[~fixed].reshape(-1)
    task = "mode" if problem["id"] == "lowest-eigenvector/lj38" else str(search.get("task", "saddle"))
    mode_seed = str(search.get("mode_seed", "gradient"))
    if reactant is not None and mode_seed == "gradient":
        mode_seed = "reactant"
    return {
        "name": row["name"],
        "positions": positions,
        "fixed": fixed,
        "cell": cell,
        "potential": problem.get("potential") or {},
        "template": template,
        "task": task,
        "mode_seed": mode_seed,
        "cap": cap,
        "force_tol": force_tol,
        "reactant": reactant,
        "file_mode": file_mode,
        "reference_mode": reference,
        "overlap": float((problem.get("sample") or {}).get("overlap", search.get("overlap", 0.99))),
        "seed": "0",
    }


def _summarize(records: list[dict], template: dict, cap: int, problem_id: str, partial: bool) -> str:
    ok = [row for row in records if row.get("success")]
    counts = [int(row["force_calls"]) for row in ok]
    lines = []
    if counts:
        lines.append(f"force_calls {half_even(sum(counts) / len(counts))}")
        lines.append(f"force_calls_min {min(counts)}")
        lines.append(f"force_calls_median {statistics.median(counts)}")
        lines.append(f"force_calls_max {max(counts)}")
    failed = len(records) - len(ok)
    lines.append(f"nfailed {failed}")
    metrics = score_v2(records, cap) if records else None
    if metrics is not None:
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
    lines.append(f"code {template.get('code', 'SaddleSearch.jl')}")
    lines.append(f"code_version {template.get('code_version', '')}")
    lines.append(f"commit {template.get('commit', '')}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', '')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem_id}")
    lines.append(f"n {len(records)}")
    if partial:
        lines.append("partial true")
    if metrics is not None:
        lines.append(f"cap {cap}")
    return "\n".join(lines) + "\n", metrics


def _stamp(records: list[dict], template: dict) -> list[dict]:
    stamped = []
    for record in records:
        copied = _jsonable(record)
        copied["code"] = template.get("code", "SaddleSearch.jl")
        copied["code_version"] = template.get("code_version", "")
        copied["commit"] = template.get("commit", "")
        copied["tuning"] = template.get("tuning", "")
        copied["template"] = template["id"]
        copied["problem"] = template.get("problem", copied.get("problem", ""))
        stamped.append(copied)
    return stamped


def _run_geometries(root, problem, template, specs, output: Path, workers: int, cap: int, partial: bool) -> dict:
    partial_path = output.with_name(output.stem + ".partial.jsonl")
    done = {row["start"]: row for row in _load_partial(partial_path)}
    published = output.with_suffix(".starts.jsonl")
    if published.is_file():
        for row in _load_partial(published):
            done.setdefault(str(row["start"]), row)
    pending = [spec for spec in specs if spec["name"] not in done]
    fresh: list[dict] = []
    if pending:
        if workers <= 1:
            session = JuliaSession.start()
            try:
                for spec in pending:
                    if session.proc.poll() is not None:
                        session = JuliaSession.start()
                    record = execute(spec, session)
                    if record.get("failure") == "crash" and session.proc.poll() is not None:
                        session = JuliaSession.start()
                        record = execute(spec, session)
                    fresh.append(record)
                    _append_partial(partial_path, _jsonable(record))
                    print(
                        f"{template['id']} {spec['name']} calls {record['force_calls']} "
                        f"failure {record.get('failure')}",
                        flush=True,
                    )
            finally:
                session.close()
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                for record in pool.map(_worker_execute, pending):
                    fresh.append(record)
                    _append_partial(partial_path, _jsonable(record))
                    print(
                        f"{template['id']} {record['start']} calls {record['force_calls']} "
                        f"failure {record.get('failure')}",
                        flush=True,
                    )
    ordered = []
    by_name = dict(done)
    for record in fresh:
        by_name[str(record["start"])] = _jsonable(record)
    for spec in specs:
        if spec["name"] in by_name:
            ordered.append(by_name[spec["name"]])
    stamped = _stamp(ordered, template)
    for record in stamped:
        record["problem"] = problem["id"]
    text, metrics = _summarize(stamped, template, cap, problem["id"], partial or len(stamped) < len(specs))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    write_starts(published, stamped)
    if len(stamped) == len(specs) and not partial:
        partial_path.unlink(missing_ok=True)
    successes = [row for row in stamped if row.get("success")]
    return {
        "force_calls": half_even(statistics.fmean([row["force_calls"] for row in successes])) if successes else None,
        "nfailed": len(stamped) - len(successes),
        "starts": len(stamped),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": None if metrics is None else metrics["par2_mean"],
        "median": None if metrics is None else metrics["median"],
        "records": stamped,
    }


def _endpoint_axis(point: np.ndarray, mode) -> np.ndarray:
    if mode is not None:
        vector = np.asarray(mode, dtype=np.float64).reshape(-1)
        if vector.size == np.asarray(point).size and _finite_point(vector):
            return _unit(vector)
    return _unit(np.asarray(point, dtype=np.float64))


def _as_point(record: dict) -> np.ndarray | None:
    raw = record.get("x")
    if raw is None:
        return None
    point = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not _finite_point(point):
        return None
    return point


def _catalog_comment(rep, minimize_calls, records, found, searches, connected, goal_seed) -> str:
    spent = minimize_calls + sum(int(row["force_calls"]) for row in records if str(row["start"]).startswith(f"rep{rep}-"))
    fraction = connected / searches if searches else 0.0
    return (
        f"# rep {rep} force_calls {spent} unique_saddles {len(found)} "
        f"fraction_connected {fraction} total_searches {searches} "
        f"connected {connected} minimize_calls {minimize_calls} seed {goal_seed}"
    )


def _write_catalog(output, template, problem, comments, records, search_cap, repetitions, failed_reps, goal, complete):
    text = "\n".join(comments) + "\n"
    totals = []
    for line in comments:
        parts = line.split()
        totals.append(int(parts[parts.index("force_calls") + 1]))
    if totals:
        text += f"force_calls {statistics.fmean(totals)}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
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
    text += (
        f"code {template.get('code', 'SaddleSearch.jl')}\n"
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
        f"n {len(records)}\n"
    )
    if not complete:
        text += "partial true\n"
    if metrics is not None:
        text += f"par2_mean {metrics['par2_mean']}\nv2_nfailed {metrics['nfailed']}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".dat.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    stamped = _stamp(records, template)
    for record in stamped:
        record["problem"] = problem["id"]
    starts = output.with_suffix(".starts.jsonl")
    temporary_starts = output.with_name(output.stem + ".starts.jsonl.tmp")
    body = "\n".join(json.dumps(row, sort_keys=True) for row in stamped)
    temporary_starts.write_text(body + ("\n" if body else ""), encoding="utf-8")
    temporary_starts.replace(starts)
    return totals, metrics, stamped


def _load_catalog_partial(path: Path) -> list[dict]:
    """Read a checkpoint. A torn last line is the kill point and is dropped."""
    if not path.is_file():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    last = len(lines) - 1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            if index == last:
                return rows
            raise
    return rows


def _catalog_displacement(
    rng,
    listed,
    free,
    minimum,
    positions,
    magnitude,
    rep,
    searches,
    problem,
    template,
    cap,
    force_tol,
    fixed,
    cell,
) -> dict:
    atom = int(listed[int(rng.integers(0, len(listed)))])
    slots = np.flatnonzero(free == atom)
    if len(slots) != 1:
        raise RuntimeError(f"displace atom {atom} is not one free atom")
    slot = int(slots[0])
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    x0 = minimum.copy()
    x0[3 * slot : 3 * slot + 3] += magnitude * direction
    kick = np.zeros_like(x0)
    kick[3 * slot : 3 * slot + 3] = direction
    moved = positions.copy()
    moved[free] = x0.reshape(-1, 3)
    return {
        "name": f"rep{rep}-search{searches}",
        "positions": moved,
        "fixed": fixed,
        "cell": cell,
        "potential": problem["potential"],
        "template": template,
        "task": "saddle",
        "mode_seed": "file",
        "file_mode": kick,
        "cap": cap,
        "force_tol": force_tol,
        "overlap": 0.99,
        "seed": str(20261015 + rep),
    }


def _open_pool(workers: int) -> ProcessPoolExecutor:
    # Fork after the parent has loaded NumPy. The child initializer starts
    # Julia, so the parent never holds a Julia runtime across the fork.
    ctx = multiprocessing.get_context("fork")
    return ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, mp_context=ctx)


def _judge_climb(
    climbed: dict,
    spec: dict,
    outer,
    minimum: np.ndarray,
    min_energy: float,
    found: list,
    connect_rmsd: float,
    barrier_max: float,
    force_tol: float,
    cap: int,
    search_cap: int,
    template_id: str,
) -> tuple[dict, np.ndarray | None, int]:
    climb = int(climbed.get("force_calls") or 0)
    point = _as_point(climbed)
    before = int(outer.calls)
    record = {
        "schema": "optbench.start/v2",
        "start": spec["name"],
        "counter": "harness",
        "comparable": True,
        "success": False,
        "reasons": list(climbed.get("reasons") or ["force_norm"]),
        "converged": bool(climbed.get("converged")),
        "library_evaluations": int(climbed.get("library_evaluations") or 0),
        "steps": int(climbed.get("steps") or 0),
        "curvature": climbed.get("curvature"),
        "hessian_index": climbed.get("hessian_index"),
        "force_norm": climbed.get("force_norm"),
        "energy": climbed.get("energy"),
        "seed": spec["seed"],
        "failure": climbed.get("failure") or "divergence",
    }
    saved_point = None
    delta = 0
    force_ok = (
        point is not None
        and climbed.get("force_norm") is not None
        and math.isfinite(float(climbed["force_norm"]))
        and float(climbed["force_norm"]) <= force_tol + 1e-10
        and climb < cap
        and climbed.get("hessian_index") == 1
    )
    if force_ok and point is not None:
        axis = _endpoint_axis(point, climbed.get("mode"))
        plus, _plus_ok = _lbfgs_minimize(outer, point + 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET)
        minus, _minus_ok = _lbfgs_minimize(outer, point - 0.1 * axis, force_tol, _SADDLE_ENDPOINT_BUDGET)
        plus_rmsd = _cartesian_rmsd(plus.reshape(-1, 3), minimum.reshape(-1, 3))
        minus_rmsd = _cartesian_rmsd(minus.reshape(-1, 3), minimum.reshape(-1, 3))
        links = (plus_rmsd <= connect_rmsd) ^ (minus_rmsd <= connect_rmsd)
        novel = all(
            _cartesian_rmsd(point.reshape(-1, 3), old.reshape(-1, 3)) > connect_rmsd for old in found
        )
        barrier = float(climbed["energy"]) - float(min_energy)
        record["barrier"] = barrier
        record["rmsd_plus"] = plus_rmsd
        record["rmsd_minus"] = minus_rmsd
        record["reasons"] = []
        if novel:
            found.append(point)
            saved_point = point
        if links and novel and barrier < barrier_max:
            delta = 1
            record["success"] = True
            record["failure"] = None
        else:
            if not links:
                record["reasons"].append("not_connected")
            if not novel:
                record["reasons"].append("duplicate")
            if barrier >= barrier_max:
                record["reasons"].append("barrier")
            record["success"], record["failure"] = _outcome(record["reasons"])
    endpoint_calls = int(outer.calls - before)
    record["force_calls"] = climb + endpoint_calls
    record["endpoint_calls"] = endpoint_calls
    record["climb_calls"] = climb
    if record["force_calls"] >= search_cap and record["success"]:
        raise RuntimeError(f"{template_id} {record['start']} spent {search_cap} or more force calls")
    return record, saved_point, delta


def _run_saddle_catalog(
    root, problem, template, output: Path, cap: int, force_tol: float, workers: int = 1
) -> dict:
    """Displace the Pt island until the connected-saddle goal, or the search budget.

    One search is the climb plus the two endpoint minimizations. The
    initial minimization is outside that count. Connection is Cartesian
    RMSD of the free atoms, because the substrate fixes the frame.
    Climbs in a batch share no state. Endpoints and novelty stay in
    search order, and a climb past the goal is dropped.
    """
    from .rgsaddle_minmode import _stored

    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = load_con(_stored(root, con_path))
    search = template.get("search") or {}
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    repetitions = int(search.get("repetitions", 1))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    magnitude = float(search.get("displace_magnitude", 0.1))
    connect_rmsd = float(search.get("rmsd_connect", 0.5))
    max_searches = int(search.get("max_searches", 20000))
    free = np.flatnonzero(~fixed)
    raw = potential_from_problem(problem)
    search_cap = int(cap) + _SADDLE_CLIMB_OVERSHOOT + 2 * _SADDLE_ENDPOINT_BUDGET
    comments = []
    all_records: list[dict] = []
    failed_reps = 0
    width = max(1, int(workers))
    session = JuliaSession.start() if width == 1 else None
    pool = _open_pool(width) if width > 1 else None
    try:
        for rep in range(1, repetitions + 1):
            partial_path = output.with_name(output.stem + f".rep{rep}.partial.jsonl")
            header_path = output.with_name(output.stem + f".rep{rep}.header.json")
            saved = _load_catalog_partial(partial_path) if header_path.is_file() else []
            outer = _surface(_bind(raw, positions, free, cell), cap=10**9)
            if header_path.is_file() and saved:
                header = json.loads(header_path.read_text(encoding="utf-8"))
                minimum = np.asarray(header["minimum"], dtype=np.float64)
                min_energy = float(header["min_energy"])
                minimize_calls = int(header["minimize_calls"])
                connected = sum(1 for row in saved if row.get("success"))
                searches = len(saved)
                found = [
                    np.asarray(row["saddle_x"], dtype=np.float64)
                    for row in saved
                    if row.get("saddle_x") is not None
                ]
                rng = _rng_restore(saved[-1]["rng_state"])
                for row in saved:
                    kept = {
                        key: value
                        for key, value in row.items()
                        if key not in {"rng_state", "saddle_x", "climb_calls"}
                    }
                    all_records.append(kept)
                print(
                    f"{template['id']} rep {rep} resume searches {searches} connected {connected}",
                    flush=True,
                )
            else:
                minimum, min_ok = _lbfgs_minimize(
                    outer, positions[free].reshape(-1).copy(), force_tol, 5000
                )
                min_energy, _grad = outer.energy_forces(minimum)
                minimize_calls = int(outer.calls)
                rng = np.random.default_rng(20261015 + rep)
                found = []
                connected = 0
                searches = 0
                header_path.parent.mkdir(parents=True, exist_ok=True)
                header_path.write_text(
                    json.dumps(
                        {
                            "template": template["id"],
                            "minimum": minimum.tolist(),
                            "min_energy": min_energy,
                            "minimize_calls": minimize_calls,
                            "min_ok": bool(min_ok),
                        }
                    ),
                    encoding="utf-8",
                )
                partial_path.unlink(missing_ok=True)
                saved = []
            drawn = searches
            pending: deque = deque()

            def _note(number: int, spec: dict, rng_state: dict, climbed: dict) -> None:
                nonlocal connected, searches
                record, saved_point, delta = _judge_climb(
                    climbed,
                    spec,
                    outer,
                    minimum,
                    min_energy,
                    found,
                    connect_rmsd,
                    barrier_max,
                    force_tol,
                    cap,
                    search_cap,
                    template["id"],
                )
                connected += delta
                searches = number
                all_records.append({key: value for key, value in record.items() if key != "climb_calls"})
                stored = _jsonable(record)
                stored["rng_state"] = rng_state
                if saved_point is not None:
                    stored["saddle_x"] = saved_point.tolist()
                _append_partial(partial_path, stored)
                if searches == 1 or searches % 25 == 0 or connected >= goal:
                    print(
                        f"{template['id']} rep {rep} searches {searches} connected {connected} "
                        f"calls {record['force_calls']} failure {record['failure']}",
                        flush=True,
                    )
                if searches % 100 == 0:
                    provisional = comments + [
                        _catalog_comment(
                            rep, minimize_calls, all_records, found, searches, connected, 20261015 + rep
                        )
                    ]
                    _write_catalog(
                        output,
                        template,
                        problem,
                        provisional,
                        all_records,
                        search_cap,
                        repetitions,
                        failed_reps + (0 if connected >= goal else 1),
                        goal,
                        False,
                    )

            def _draw() -> tuple[int, dict, dict]:
                nonlocal drawn
                drawn += 1
                spec = _catalog_displacement(
                    rng,
                    listed,
                    free,
                    minimum,
                    positions,
                    magnitude,
                    rep,
                    drawn,
                    problem,
                    template,
                    cap,
                    force_tol,
                    fixed,
                    cell,
                )
                return drawn, spec, _rng_state(rng)

            while connected < goal and searches < max_searches:
                if width == 1:
                    assert session is not None
                    number, spec, rng_state = _draw()
                    if session.proc.poll() is not None:
                        session = JuliaSession.start()
                    climbed = execute(spec, session)
                    if climbed.get("failure") == "crash" and session.proc.poll() is not None:
                        session = JuliaSession.start()
                        climbed = execute(spec, session)
                    _note(number, spec, rng_state, climbed)
                    continue
                assert pool is not None
                while len(pending) < width and drawn < max_searches and connected < goal:
                    number, spec, rng_state = _draw()
                    pending.append((number, spec, rng_state, pool.submit(_worker_execute, spec)))
                if not pending:
                    break
                number, spec, rng_state, future = pending.popleft()
                try:
                    climbed = future.result()
                except BrokenProcessPool:
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = _open_pool(width)
                    redo = [(number, spec, rng_state)] + [
                        (item_number, item_spec, item_state) for item_number, item_spec, item_state, _item in pending
                    ]
                    pending.clear()
                    for item_number, item_spec, item_state in redo:
                        pending.append((item_number, item_spec, item_state, pool.submit(_worker_execute, item_spec)))
                    number, spec, rng_state, future = pending.popleft()
                    climbed = future.result()
                _note(number, spec, rng_state, climbed)
                if connected >= goal:
                    pending.clear()
                    break
            if connected < goal:
                failed_reps += 1
            comments.append(
                _catalog_comment(rep, minimize_calls, all_records, found, searches, connected, 20261015 + rep)
            )
            if connected >= goal or searches >= max_searches:
                partial_path.unlink(missing_ok=True)
                header_path.unlink(missing_ok=True)
    finally:
        if session is not None:
            session.close()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    # A repetition that reached the goal or the search budget is complete.
    complete = bool(comments) and all(
        f" connected {goal} " in f" {line} "
        or int(line.split("total_searches")[1].split()[0]) >= max_searches
        for line in comments
    )
    totals, metrics, stamped = _write_catalog(
        output,
        template,
        problem,
        comments,
        all_records,
        search_cap,
        repetitions,
        failed_reps,
        goal,
        complete,
    )
    return {
        "force_calls": statistics.fmean(totals) if totals else None,
        "nfailed": failed_reps,
        "starts": len(stamped),
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": None if metrics is None else metrics["par2_mean"],
        "records": stamped,
    }


def run_template(root: Path, problem: dict, template: dict, *, output: Path, workers: int = 1, work: Path | None = None, limit: int = 0) -> dict:
    """Run one SaddleSearch.jl template on the problem's starts."""
    from .campaign import _check_stop

    _assert_source_commit(str(template.get("commit", "")))
    cap = _check_stop(problem, template)
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    convergence = problem.get("convergence") or {}
    optimizer = _section(template, "Optimizer")
    force_tol = float(optimizer.get("converged_force", convergence.get("force_norm", 1e-3)))
    if problem["id"] == "saddle-search/pt-island":
        return _run_saddle_catalog(root, problem, template, output, cap, force_tol, workers)
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    scratch.mkdir(parents=True, exist_ok=True)
    rows = _geometries(root, problem, scratch)
    expected = int((problem.get("sample") or {}).get("starts") or len(rows))
    if len(rows) != expected:
        raise RuntimeError(f"{problem['id']} has {len(rows)} geometries, the problem states {expected}")
    if limit:
        rows = rows[:limit]
    specs = [_spec_from_geometry(row, problem, template, cap, force_tol) for row in rows]
    return _run_geometries(
        root,
        problem,
        template,
        specs,
        output,
        workers,
        cap,
        partial=bool(limit) and limit < expected,
    )
