"""OPTIM L-BFGS.

The binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit.

``optim/morse-lbfgs-tuned`` calls OPTIM's bulk Morse. The score is the
``energy+gradient calls`` line. Every other minimization L-BFGS template
uses the ASE socket, and each ``COMPUTE`` is one ``pes.evaluate``.

``optim/unhooked`` and every template that is not minimization L-BFGS
raise before a file is written. Noise, saddle search, NEB, and PATHSAMPLE
are not this runner.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import judge_saddle, run_jobs, write_campaign
from .geometries import load_starts, point_settings

ENGINE = "optim"
CODE = "optim"
SNAPSHOT = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
SHA256 = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
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


_CALL_RE = re.compile(r"# of energy\+gradient calls=\s+(-?\d+)")
_ENERGY_RE = re.compile(r"Final energy is\s+([+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)")


def _section(template: dict) -> dict:
    section = dict(template.get("optim") or {})
    section["method"] = str(section.get("method", ""))
    section["updates"] = int(section.get("updates", 4))
    section["maxstep"] = float(section.get("maxstep", 0.2))
    section["potential"] = str(section.get("potential", ""))
    if section["potential"] == "morse-bulk":
        section["maxbfgs"] = float(section.get("maxbfgs", 2.0))
        section["rho"] = float(section.get("rho", 1.6047))
        section["r0"] = float(section.get("r0", 2.897))
        section["cutoff"] = float(section.get("cutoff", 9.5))
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


def morse_bulk_lines(
    coords: list[list[float]],
    box: list[float],
    *,
    gmax: float,
    updates: int,
    maxbfgs: float,
    steps: int,
    rho: float,
    r0: float,
    cutoff: float,
) -> list[str]:
    """odata for OPTIM's bulk Morse in the units ``run.sh`` builds.

    Coordinates, the orthorhombic box, and the cutoff are divided by
    ``r0``. ``PARAMS`` rho is ``alpha * r0``. The potential itself fixes
    the well at distance 1 and energy 1. ``BFGSMIN`` is the Euclidean
    force norm divided by ``sqrt(3N)``.
    """
    if r0 <= 0.0 or cutoff <= 0.0 or rho <= 0.0:
        raise ValueError("Morse r0, rho, and cutoff must be positive")
    if len(box) != 3:
        raise ValueError("Morse bulk needs three box lengths")
    scale = 1.0 / float(r0)
    lengths = [float(value) * scale for value in box]
    lines = [
        "BULK",
        "PARAMS "
        + " ".join(
            _fortran_float(value)
            for value in (float(rho) * float(r0), lengths[0], lengths[1], lengths[2], float(cutoff) * scale)
        ),
        f"BFGSMIN {_fortran_float(gmax)}",
        f"UPDATES {int(updates)}",
        f"MAXBFGS {_fortran_float(maxbfgs)}",
        f"BFGSSTEPS {int(steps)}",
        "POINTS",
    ]
    for row in coords:
        if len(row) != 3:
            raise ValueError("a Morse coordinate row needs three values")
        lines.append("M " + " ".join(_fortran_float(float(value) * scale) for value in row))
    return lines


def _parse_optim_log(text: str) -> tuple[int | None, float | None, bool]:
    calls = None
    for line in text.splitlines():
        if "Hessian" in line:
            continue
        match = _CALL_RE.search(line)
        if match:
            calls = int(match.group(1))
    energy = None
    found = _ENERGY_RE.search(text.replace("D", "E").replace("d", "e"))
    if found:
        energy = float(found.group(1))
    return calls, energy, "**** CONVERGED ****" in text


def _execute_morse(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tempfile

    from .common import classify_failure

    name = str(job["name"])
    cap = int(job["cap"])
    calls = 0
    energy = None
    reasons: list[str] = []
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-morse-") as raw:
            folder = Path(raw)
            lines = morse_bulk_lines(
                job["positions"],
                job["box"],
                gmax=float(job["gmax"]),
                updates=int(job["updates"]),
                maxbfgs=float(job["maxbfgs"]),
                steps=cap,
                rho=float(job["rho"]),
                r0=float(job["r0"]),
                cutoff=float(job["cutoff"]),
            )
            (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            try:
                proc = subprocess.run(
                    [str(job["binary"])],
                    cwd=folder,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                proc = None
                error = "OPTIM exceeded 3600s"
                reasons.append("crash")
            if proc is not None:
                text = (proc.stdout or "") + "\n" + (proc.stderr or "")
                parsed, energy, converged = _parse_optim_log(text)
                if parsed is not None:
                    calls = parsed
                if proc.returncode != 0:
                    reasons.append("crash")
                    error = error or f"OPTIM exit {proc.returncode}"
                elif not converged:
                    reasons.append("cap" if calls >= cap else "divergence")
                elif calls >= cap:
                    reasons.append("cap")
    except Exception as exc:
        reasons.append("crash")
        error = f"{type(exc).__name__}: {exc}"
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": str(job.get("counter") or "optim"),
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "converged": failure is None,
        "energy": energy,
        "hessian_index": None,
    }
    if error:
        record["error"] = error
    return record


def _run_morse_bulk(root: Path, problem: dict, template: dict, section: dict, settings: dict, output: Path, workers: int) -> dict:
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    if not starts:
        raise RuntimeError(f"{problem['id']} has no starts")
    binary = binary_path()
    cap = int(settings["cap"])
    jobs = []
    for start in starts:
        positions = start["positions"]
        atoms = len(positions)
        if start["box"] is None:
            raise ValueError(f"{start['name']} has no periodic box")
        jobs.append(
            {
                "name": start["name"],
                "positions": positions,
                "box": start["box"],
                "cap": cap,
                "gmax": float(settings["force_norm"]) / math.sqrt(3 * atoms),
                "updates": int(section["updates"]),
                "maxbfgs": float(section["maxbfgs"]),
                "rho": float(section["rho"]),
                "r0": float(section["r0"]),
                "cutoff": float(section["cutoff"]),
                "binary": str(binary),
                "counter": str(template.get("counter") or "optim"),
            }
        )
    records = run_jobs(jobs, _execute_morse, workers)
    label = str(template.get("counter") or "optim")
    comment = (
        "# force_calls is OPTIM's energy+gradient count. Each one is a potential evaluation.\n"
        "# Coordinates, the box, and the cutoff are divided by r0. morse_bulk keeps the well at distance 1 and energy 1.\n"
        "# PARAMS rho is alpha times r0.\n"
        f"# UPDATES {int(section['updates'])} and MAXBFGS {float(section['maxbfgs'])} are the values in the template.\n"
        "# BFGSMIN is the Euclidean force norm divided by sqrt(3N). BFGSSTEPS is the cap.\n"
        "# A start fails when it does not converge under the force-call cap."
    )
    parameters = {
        "method": "lbfgs",
        "keyword": "BFGSMIN",
        "potential": "morse-bulk",
        "updates": int(section["updates"]),
        "maxbfgs": float(section["maxbfgs"]),
        "rho": float(section["rho"]),
        "r0": float(section["r0"]),
        "cutoff": float(section["cutoff"]),
        "counter": "optim energy+gradient calls" if label == "harness" else "optim",
        "version": str(template.get("code_version") or ""),
    }
    stamped = dict(template)
    stamped["counter"] = label
    stamped["code"] = "optim"
    return write_campaign(
        output,
        records,
        stamped,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )


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


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    code_id = str(template.get("code") or CODE)
    require_pin(template, registry_pin(root, code_id))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    section = _section(template)
    if problem.get("suite") != "minimization" or section["method"] != "lbfgs":
        raise _unhooked(template_name)
    if template.get("noise"):
        raise _unhooked(template_name)
    settings = point_settings(problem)
    if settings["cap"] is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    if section["potential"] == "morse-bulk":
        return _run_morse_bulk(root, problem, template, section, settings, output, workers)
    scratch = output.parent / "work" / template["id"].replace("/", "-")
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
