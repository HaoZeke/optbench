"""OPTIM through the ASE socket.

The binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. SVNREV in the current tree is
``df8da599``. Each ``COMPUTE`` on the relay is one ``pes.evaluate``.
The ranked score is that harness count. OPTIM's own counter is recorded
and is not the score.

Minimization L-BFGS and the LJ38 Rayleigh-Ritz lowest eigenvector are
scored. ``optim/unhooked`` and every other template raise before a file
is written. Noise, saddle search, NEB, and PATHSAMPLE are not this runner.

The Rayleigh-Ritz row is unmodified stock OPTIM. ``USEDIAG 3`` is the
first-order forward quotient. ``EIGENONLY``'s second value is the archive
overlap stop. Memory, step, and the percent-change gate stay at the
source defaults.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import judge_mode, judge_saddle, run_jobs, write_campaign
from .geometries import load_starts, point_settings

ENGINE = "optim"
CODE = "optim"
SNAPSHOT = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
SHA256 = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
SVNREV = "df8da599267534fbe43a250e578f7396a7946311"
CODE_VERSION = "df8da599"
_RELAY = Path(__file__).with_name("fmlip_relay.py")


class _Calls:
    """Socket tally. Verification does not spend it."""

    def __init__(self, calls: int) -> None:
        self.calls = int(calls)
        self.verify_calls = 0

    def verification(self):
        return _Quiet()


class _Quiet:
    def __enter__(self):
        return None

    def __exit__(self, *_exc) -> bool:
        return False


def binary_path() -> Path:
    env = os.environ.get("OPTIM_BIN")
    if env:
        return Path(env)
    return Path("/tmp/src/wales-full/build-optim/OPTIM")


def available() -> str | None:
    binary = binary_path()
    if binary.is_file() and _RELAY.is_file():
        return None
    missing = binary if not binary.is_file() else _RELAY
    return (
        "OPTIM has no harness potential in this tree. "
        f"The pin is the sha256 {SHA256} of {SNAPSHOT} "
        "(122743078 bytes). The archive is not a git commit. "
        f"{missing} is not present. "
        "OPTIM's own call count is not a score."
    )


def _section(template: dict) -> dict:
    section = dict(template.get("optim") or {})
    section["method"] = str(section.get("method", ""))
    section["updates"] = int(section.get("updates", 4))
    section["maxstep"] = float(section.get("maxstep", 0.2))
    return section


def _rr_section(template: dict) -> dict:
    """Stock Rayleigh-Ritz. Only the method and the overlap stop are set."""
    section = dict(template.get("optim") or {})
    section["method"] = str(section.get("method", ""))
    section["usediag"] = int(section.get("usediag", 3))
    section["cover"] = float(section.get("cover", 0.99))
    return section


def _unhooked(template_name: str) -> RuntimeError:
    return RuntimeError(
        f"{template_name} writes nothing. The pin is the sha256 {SHA256}. "
        "Only a minimization L-BFGS template calls the ASE socket. "
        "Noise, saddle search, NEB, tsopt, hTST, and PATHSAMPLE are not scored."
    )


def _write_xyz(path: Path, positions: np.ndarray, box) -> None:
    rows = np.asarray(positions, dtype=np.float64)
    lines = [str(len(rows))]
    if box is None:
        lines.append("optbench")
    else:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        lattice = (
            f"{lengths[0]:.16g} 0 0 0 {lengths[1]:.16g} 0 0 0 {lengths[2]:.16g}"
        )
        lines.append(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3')
    for row in rows:
        lines.append(f"Ar {row[0]:.16g} {row[1]:.16g} {row[2]:.16g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_points(path: Path, natoms: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        tokens = text.replace("D", "E").replace("d", "e").split()
        if len(tokens) < 3:
            return None
        try:
            rows.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        except ValueError:
            return None
    if len(rows) != natoms:
        return None
    return np.asarray(rows, dtype=np.float64)


def _tail(path: Path) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


def _fortran_float(value: float) -> str:
    """List-directed OPTIM input rejects a lowercase ``e`` exponent."""
    return f"{float(value):.16E}".replace("E", "D")


def _odata(folder: Path, python: str, hook: Path, gmax: float, updates: int, steps: int, maxstep: float, periodic: bool) -> None:
    pbc = "PBC" if periodic else "NOPBC"
    lines = [
        f"ASE geom.xyz {python} {hook} dummy dummy cpu FIXCELL {pbc} EV ANG",
        f"BFGSMIN {_fortran_float(gmax)}",
        f"UPDATES {int(updates)}",
        f"STEPS {int(steps)}",
        f"MAXSTEP {_fortran_float(maxstep)}",
        f"MAXMAX {_fortran_float(maxstep)}",
    ]
    (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _odata_rr(folder: Path, python: str, hook: Path, section: dict) -> None:
    """Stock odata. ``1.0D-10`` is the source ``CEIG``. The second value sets ``OVERCONV``."""
    lines = [
        "NOTE optbench",
        f"ASE geom.xyz {python} {hook} dummy dummy cpu FIXCELL NOPBC EV ANG",
        f"EIGENONLY 1.0D-10 {_fortran_float(section['cover'])}",
        f"USEDIAG {int(section['usediag'])}",
        "READVEC",
        "DUMPVECTOR",
    ]
    (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")


def paper_calls(log: str) -> tuple[int | None, bool]:
    """Field 6 of the last energy+gradient line, and the CONVERGED banner."""
    calls = None
    for line in log.splitlines():
        if "energy+gradient calls" not in line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            calls = int(float(parts[5]))
        except ValueError:
            continue
    return calls, "**** CONVERGED ****" in log


def _write_vector_dump(path: Path, raw_mode: str) -> None:
    """Archive ``run.sh`` writes ``100`` then the shared ``initial_mode``."""
    body = raw_mode if raw_mode.endswith("\n") else raw_mode + "\n"
    path.write_text("100\n" + body, encoding="utf-8")


def _dumped_vector(path: Path, nopt: int) -> np.ndarray | None:
    """``DUMPVECTOR`` writes the eigenvalue, then ``nopt`` components."""
    if not path.is_file():
        return None
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        for token in text.replace("D", "E").replace("d", "e").split():
            try:
                values.append(float(token))
            except ValueError:
                return None
    if len(values) < nopt + 1:
        return None
    return np.asarray(values[1 : nopt + 1], dtype=np.float64)


def _overlap(vector: np.ndarray | None, reference: np.ndarray) -> float:
    if vector is None or vector.size != reference.size:
        return 0.0
    scale = float(np.linalg.norm(vector))
    target = float(np.linalg.norm(reference))
    if scale < 1e-14 or target < 1e-14:
        return 0.0
    return abs(float(np.dot(vector / scale, reference / target)))


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from optbench.pes import evaluate

    positions = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = str(job["kind"])
    cap = int(job["cap"])
    force_norm = float(job["force_norm"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    natoms = int(positions.shape[0])
    components = 3 * natoms
    gmax = force_norm / math.sqrt(components)
    binary = binary_path()
    point = None
    energy = None
    gradient = None
    calls = 0
    capped = False
    error = ""
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="optim-") as raw:
            folder = Path(raw)
            hook = folder / "hook"
            relay = hook / "pysocket" / "fmlip_relay.py"
            relay.parent.mkdir(parents=True)
            shutil.copyfile(_RELAY, relay)
            _write_xyz(folder / "geom.xyz", positions, box)
            _odata(
                folder,
                sys.executable,
                hook,
                gmax,
                int(job["updates"]),
                cap,
                float(job["maxstep"]),
                box is not None,
            )
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["OPTBENCH_KIND"] = kind
            env["OPTBENCH_CAP"] = str(cap)
            env["OPTBENCH_COUNT_PATH"] = str(folder / "calls.txt")
            env["OPTBENCH_LAST_PATH"] = str(folder / "last.txt")
            env["OPTBENCH_CAPPED_PATH"] = str(folder / "capped.txt")
            env["OPTBENCH_E_SCALE"] = "1"
            env["OPTBENCH_L_SCALE"] = "1"
            if box is None:
                env.pop("OPTBENCH_BOX", None)
            else:
                lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
                env["OPTBENCH_BOX"] = " ".join(f"{float(value):.16g}" for value in lengths)
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.run(
                        [str(binary)],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=3600,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                proc = None
                error = "OPTIM exceeded 3600s"
            count_path = folder / "calls.txt"
            if count_path.is_file():
                calls = int(count_path.read_text(encoding="utf-8").strip() or "0")
            capped = (folder / "capped.txt").is_file() or calls >= cap
            if proc is not None and proc.returncode == 0 and not capped:
                point = _read_points(folder / "points.final", natoms)
            if point is None:
                point = _read_points(folder / "last.txt", natoms)
            if proc is not None and proc.returncode != 0 and not capped:
                detail = _tail(stderr_path) or _tail(stdout_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
            elif proc is not None and point is None and calls == 0 and not capped:
                detail = _tail(stdout_path) or _tail(stderr_path) or "no geometry"
                error = f"OPTIM produced no geometry: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if point is not None and not error:
        energy, gradient = evaluate(kind, point, box)
        gradient = np.asarray(gradient, dtype=np.float64)
    prefix: list[str] = []
    if error:
        prefix.append("crash")
    elif capped:
        prefix.append("cap")
    rotations = 0 if box is not None and bool(np.all(free)) else None
    record = judge_saddle(
        job["name"],
        point,
        None if energy is None else float(energy),
        gradient,
        counter=_Calls(calls),
        cap=cap,
        force_norm=force_norm,
        kind=kind,
        free=free,
        box=box,
        prefix=prefix,
        expected_index=0,
        rotations=rotations,
    )
    if error:
        record["error"] = error
    record["optim_method"] = "lbfgs"
    return record


def _job(start: dict, settings: dict, template: dict) -> dict:
    section = _section(template)
    return {
        "name": start["name"],
        "positions": start["positions"],
        "free": [not bool(flag) for flag in start["fixed"]],
        "box": start["box"],
        "kind": settings["potential"],
        "cap": int(settings["cap"]),
        "force_norm": float(settings["force_norm"]),
        "updates": section["updates"],
        "maxstep": section["maxstep"],
    }


def _relay_env(folder: Path, job: dict) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    root = str(job.get("pythonpath") or "")
    if root:
        previous = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root if not previous else root + os.pathsep + previous
    env["OPTBENCH_KIND"] = str(job["kind"])
    env["OPTBENCH_CAP"] = str(int(job["cap"]))
    env["OPTBENCH_COUNT_PATH"] = str(folder / "calls.txt")
    env["OPTBENCH_LAST_PATH"] = str(folder / "last.txt")
    env["OPTBENCH_CAPPED_PATH"] = str(folder / "capped.txt")
    env["OPTBENCH_E_SCALE"] = "1"
    env["OPTBENCH_L_SCALE"] = "1"
    box = job["box"]
    if box is None:
        env.pop("OPTBENCH_BOX", None)
    else:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        env["OPTBENCH_BOX"] = " ".join(f"{float(value):.16g}" for value in lengths)
    return env


def _execute_rr(job: dict) -> dict:
    """One Rayleigh-Ritz lowest-eigenvector search."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    positions = np.asarray(job["positions"], dtype=np.float64)
    cap = int(job["cap"])
    natoms = int(positions.shape[0])
    nopt = 3 * natoms
    reference = np.asarray(job["reference"], dtype=np.float64).reshape(-1)
    binary = binary_path()
    calls = 0
    capped = False
    error = ""
    overlap = 0.0
    native: int | None = None
    converged = False
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="optim-rr-") as raw:
            folder = Path(raw)
            hook = folder / "hook"
            relay = hook / "pysocket" / "fmlip_relay.py"
            relay.parent.mkdir(parents=True)
            shutil.copyfile(_RELAY, relay)
            _write_xyz(folder / "geom.xyz", positions, None)
            _write_vector_dump(folder / "vector.dump", str(job["initial_mode"]))
            (folder / "target.vector").write_text(str(job["target"]), encoding="utf-8")
            _odata_rr(folder, sys.executable, hook, job["section"])
            env = _relay_env(folder, job)
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            proc = None
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.run(
                        [str(binary)],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=3600,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                error = "OPTIM exceeded 3600s"
            log = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
            native, converged = paper_calls(log)
            count_path = folder / "calls.txt"
            if count_path.is_file():
                calls = int(count_path.read_text(encoding="utf-8").strip() or "0")
            capped = (folder / "capped.txt").is_file() or calls >= cap
            overlap = _overlap(_dumped_vector(folder / "vector.dump", nopt), reference)
            if proc is not None and proc.returncode != 0 and not capped and not error and not converged:
                detail = _tail(stderr_path) or _tail(stdout_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
            elif proc is not None and calls == 0 and not capped and not error:
                detail = _tail(stdout_path) or _tail(stderr_path) or "no evaluations"
                error = f"OPTIM produced no evaluations: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if capped and "force-call cap" not in error:
        error = "force-call cap"
    record = judge_mode(job["name"], calls, cap, overlap, float(job["overlap"]), error)
    record["optim_method"] = "rr-ffd"
    record["usediag"] = int(job["section"]["usediag"])
    record["optim_energy_gradient_calls"] = native
    record["optim_converged"] = bool(converged)
    if record["success"] and not converged:
        record["success"] = False
        record["failure"] = "divergence"
        record["converged"] = False
        record["reasons"] = list(record.get("reasons") or []) + ["divergence"]
    return record


def _mode_text(folder: Path, name: str) -> str:
    path = folder / "coords-con" / f"{name}_mode"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _shared_initial_mode(folder: Path, nopt: int) -> str:
    path = folder / "coords-con" / "initial_mode"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    count = 0
    for line in text.splitlines():
        piece = line.strip()
        if piece and not piece.startswith("#"):
            count += len(piece.split())
    if count != nopt:
        raise ValueError(f"{path} has {count} components, the geometry has {nopt}")
    return text


def _rr_job(start: dict, settings: dict, section: dict, initial_mode: str, target: str, root: Path) -> dict:
    if start.get("reference") is None:
        raise ValueError(f"{start['name']} has no reference mode")
    return {
        "name": start["name"],
        "positions": start["positions"],
        "reference": start["reference"],
        "kind": settings["potential"],
        "box": start["box"],
        "cap": int(settings["cap"]),
        "overlap": float(settings["overlap"]),
        "section": section,
        "initial_mode": initial_mode,
        "target": target,
        "pythonpath": str(root),
    }


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    if template.get("noise"):
        raise _unhooked(template_name)
    section = _section(template)
    settings = point_settings(problem)
    if settings["cap"] is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    if problem.get("suite") == "lowest-eigenvector" and section["method"] == "rr-ffd":
        return _run_rr(root, problem, template, settings, scratch, output, workers)
    if problem.get("suite") != "minimization" or section["method"] != "lbfgs":
        raise _unhooked(template_name)
    jobs = [_job(start, settings, template) for start in load_starts(root, problem, scratch)]
    records = run_jobs(jobs, _execute, workers)
    comment = (
        "# counter harness. Each ASE COMPUTE is one pes.evaluate. "
        "OPTIM's own counter is not the score.\n"
        "# BFGSMIN RMS stop is the Euclidean force norm over sqrt(3N). "
        "STEPS is the cap. A success is under the cap, at the force stop, "
        "and Hessian index 0."
    )
    parameters = {
        "method": "lbfgs",
        "keyword": "BFGSMIN",
        "updates": section["updates"],
        "maxstep": section["maxstep"],
        "counter": "harness",
        "socket": "ASE",
    }
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )


def _run_rr(root, problem, template, settings, scratch, output, workers: int) -> dict:
    section = _rr_section(template)
    if int(section["usediag"]) != 3:
        raise ValueError(f"{template['id']} USEDIAG {section['usediag']} is not the first-order quotient")
    starts = load_starts(root, problem, scratch)
    folder = scratch / "geometries"
    nopt = 3 * len(starts[0]["positions"])
    initial_mode = _shared_initial_mode(folder, nopt)
    jobs = [
        _rr_job(start, settings, section, initial_mode, _mode_text(folder, start["name"]), root)
        for start in starts
    ]
    records = run_jobs(jobs, _execute_rr, workers)
    comment = (
        f"# OPTIM SVNREV {CODE_VERSION} ({SVNREV}), unmodified, stock keywords.\n"
        "# counter harness. Each ASE COMPUTE is one pes.evaluate. "
        "OPTIM energy+gradient calls are recorded and are not the score.\n"
        "# Rayleigh-Ritz EIGENONLY 1.0D-10 0.99, USEDIAG 3 (forward differences). "
        "Memory, step, and BFGSTSPC stay at the source defaults.\n"
        "# vector.dump is 100 plus the shared initial_mode. "
        "target.vector is the per-start mode file.\n"
        "# A success is CONVERGED, under the cap, and absolute overlap of at least 0.99."
    )
    parameters = {
        "method": "rr-ffd",
        "keyword": "EIGENONLY",
        "usediag": section["usediag"],
        "ceig": 1.0e-10,
        "cover": section["cover"],
        "version": CODE_VERSION,
        "svnrev": SVNREV,
        "counter": "harness",
        "native_counter": "optim energy+gradient calls",
        "socket": "ASE",
    }
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )
