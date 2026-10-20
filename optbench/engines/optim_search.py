"""OPTIM L-BFGS through the ASE socket, and the current climbing string.

The minimization binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not that score.

``optim/unhooked`` and every template that is not minimization L-BFGS
raise before a file is written. Noise, saddle search, NEB, and PATHSAMPLE
are not the socket runner.

``optim/de-pt-cl-string-latest`` is the current ``svn.tar.bz2`` release.
The tally is OPTIM's own ``energy+gradient calls`` line on the native
periodic Morse. ``STEPS`` and the ``GROWSTRING`` iteration density are
the problem force-call cap. Image count is the problem's interior images.
"""

from __future__ import annotations

import math
import os
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
LATEST_CODE = "optim-df8da599"
LATEST_URL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
LATEST_SHA256 = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
LATEST_VERSION = "df8da599"
LATEST_SVNREV = "df8da599267534fbe43a250e578f7396a7946311"
# Unmodified morse_bulk hardcodes R0 = 1 and A = 1. These are the Pt Morse
# constants the problem states, expressed in that unit system.
LATEST_R0 = 2.8970
LATEST_RHO = 1.6047
LATEST_RCUT = 9.5
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


def clstring_calls(text: str) -> int | None:
    """Last OPTIM energy-and-gradient tally.

    The 2014 ``run_optim.sh`` keeps awk field 6 of the last line that
    contains ``energy+gradient calls``. That field is the integer after
    ``calls=``.
    """
    found = None
    for line in text.splitlines():
        if "energy+gradient calls" not in line:
            continue
        parts = line.split()
        if len(parts) < 6 or not parts[5].lstrip("-").isdigit():
            return None
        found = int(parts[5])
    return found


def clstring_converged(text: str) -> bool:
    """True when OPTIM printed its transition-state stop."""
    return "Converged to TS" in text


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


def latest_binary() -> Path:
    env = os.environ.get("OPTIM_LATEST_BIN")
    if env:
        return Path(env)
    return Path("/tmp/src/wales-latest/wales/OPTIM/builds/gfortran/OPTIM")


def stock_converged(text: str) -> bool:
    """OPTIM's own climbing-string stop, not a harness force threshold."""
    return (
        "Converged to TS" in text
        or "SUCCESS! - the last image converged!" in text
        or "SUCCESS! - whole string converged!" in text
    )


def stock_odata(
    reactant: np.ndarray,
    free: np.ndarray,
    box: np.ndarray,
    product: np.ndarray,
    *,
    images: int,
    steps: int,
) -> tuple[str, str]:
    """Stock ``odata`` and ``finish`` for one pair.

    ``GROWSTRING`` takes the problem image count and the force-call cap
    as its iteration density. ``STEPS`` is that same cap, which is also
    what OPTIM copies onto ``BFGSSTEPS`` when that keyword is still 1.
    Reparameterisation, growth, convergence, and the maximum string step
    stay at the values the source initialises. ``DESMDEBUG`` is the
    documented print switch for a double-ended search: the climbing-string
    convergence line is written only when that switch is on. Frozen atoms
    and the two endpoints are the pair.
    """
    scale = 1.0 / LATEST_R0
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3] * scale
    lines = [
        "BULK",
        "CLSTRING",
        f"GROWSTRING {int(images)} {int(steps)}",
        f"STEPS {int(steps)}",
        "DESMDEBUG",
    ]
    lines.append(
        "PARAMS %.16g %.16g %.16g %.16g %.16g"
        % (
            LATEST_RHO * LATEST_R0,
            float(lengths[0]),
            float(lengths[1]),
            float(lengths[2]),
            LATEST_RCUT * scale,
        )
    )
    frozen = np.flatnonzero(~np.asarray(free, dtype=bool)) + 1
    for start in range(0, len(frozen), 8):
        chunk = " ".join(str(int(index)) for index in frozen[start : start + 8])
        lines.append(f"FREEZE {chunk}")
    lines.append("points")
    scaled = np.asarray(reactant, dtype=np.float64) * scale
    for row in scaled:
        lines.append("M %.16g %.16g %.16g" % (float(row[0]), float(row[1]), float(row[2])))
    finish_rows = np.asarray(product, dtype=np.float64) * scale
    finish = "".join(
        "%.16g %.16g %.16g\n" % (float(row[0]), float(row[1]), float(row[2])) for row in finish_rows
    )
    return "\n".join(lines) + "\n", finish


def _execute_stock(job: dict) -> dict:
    import tempfile

    name = str(job["name"])
    cap = int(job["cap"])
    error = ""
    calls = None
    converged = False
    try:
        with tempfile.TemporaryDirectory(prefix="optim-stock-") as raw:
            folder = Path(raw)
            odata, finish = stock_odata(
                job["reactant"],
                job["free"],
                job["box"],
                job["product"],
                images=int(job["images"]),
                steps=int(job["cap"]),
            )
            (folder / "odata").write_text(odata, encoding="utf-8")
            (folder / "finish").write_text(finish, encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            try:
                proc = subprocess.run(
                    [str(job["binary"])],
                    cwd=folder,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=int(job["timeout"]),
                    check=False,
                )
                log = f"{proc.stdout}\n{proc.stderr}"
                if proc.returncode != 0:
                    error = f"OPTIM exit {proc.returncode}"
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                log = f"{stdout}\n{stderr}"
                error = f"OPTIM exceeded {int(job['timeout'])}s"
            calls = clstring_calls(log)
            converged = stock_converged(log)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    counted = calls is not None
    success = error == "" and converged and counted and int(calls) < cap
    failure = None
    reasons: list[str] = []
    if not success:
        if error or not counted:
            reasons = ["crash"]
            failure = "crash"
        elif int(calls) >= cap:
            reasons = ["cap"]
            failure = "cap"
        else:
            reasons = ["divergence"]
            failure = "divergence"
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": 0 if calls is None else int(calls),
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": 1,
        "converged": success,
    }
    if error:
        record["error"] = error
    return record


def _run_latest(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    from .common import band_settings, cached_pairs, registry_pin, require_pin

    pin = registry_pin(root, LATEST_CODE)
    require_pin(template, pin)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template['id']} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    if problem.get("suite") != "double-ended-saddle-search":
        raise ValueError(f"{template['id']} is a double-ended climbing string")
    binary = latest_binary()
    if not binary.is_file():
        raise RuntimeError(
            "The latest OPTIM release is not built. "
            f"The pin is sha256 {LATEST_SHA256} of {LATEST_URL}. "
            f"SVNREV is {LATEST_SVNREV}. {binary} is not present. "
            "Set OPTIM_LATEST_BIN to the cmake OPTIM target."
        )
    settings = band_settings(problem)
    timeout = int((template.get("optim") or {}).get("timeout", 120))
    pairs = cached_pairs(root, problem, settings)
    jobs = []
    for index in range(int(settings["pairs"])):
        pair = pairs[f"product_{index}"]
        if pair.box is None:
            raise ValueError(f"{pair.name} has no box")
        jobs.append(
            {
                "name": pair.name,
                "reactant": np.asarray(pair.reactant, dtype=np.float64),
                "product": np.asarray(pair.product, dtype=np.float64),
                "free": np.asarray(pair.free, dtype=bool),
                "box": np.asarray(pair.box, dtype=np.float64),
                "binary": str(binary),
                "cap": int(settings["cap"]),
                "images": int(settings["n_interior"]),
                "timeout": timeout,
            }
        )
    records = run_jobs(jobs, _execute_stock, workers)
    comment = (
        f"# OPTIM SVNREV {LATEST_SVNREV}, {LATEST_URL}, sha256 {LATEST_SHA256}.\n"
        "# BULK, CLSTRING, GROWSTRING with the problem image count and the force-call cap "
        "as the iteration density, STEPS at that cap, and DESMDEBUG so OPTIM prints "
        "the climbing-string convergence line.\n"
        "# Reparameterisation, growth, convergence, and the maximum string step stay at the source defaults.\n"
        "# Native periodic Morse. The unmodified source hardcodes R0 = 1 and A = 1, "
        "so the Pt coordinates, rho, and cutoff are in units of r0.\n"
        "# The count is OPTIM's last energy+gradient calls line. "
        "A start counts only when OPTIM prints its convergence line and the tally is under the cap."
    )
    published = dict(template)
    published["code"] = "optim"
    return write_campaign(
        output,
        records,
        published,
        problem,
        {"cap": int(settings["cap"])},
        {
            "method": "clstring",
            "keywords": "BULK CLSTRING GROWSTRING STEPS DESMDEBUG",
            "images": int(settings["n_interior"]),
            "steps": int(settings["cap"]),
            "counter": "optim energy+gradient calls",
            "potential": "morse_bulk",
            "version": LATEST_VERSION,
            "svnrev": LATEST_SVNREV,
            "tarball_sha256": LATEST_SHA256,
        },
        images=None,
        comment=comment,
    )


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if str((template.get("optim") or {}).get("method", "")) == "clstring-stock":
        return _run_latest(root, problem, template, output, workers)
    reason = available()
    if reason:
        raise RuntimeError(reason)
    from .common import registry_pin, require_pin

    require_pin(template, registry_pin(root, CODE))
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
