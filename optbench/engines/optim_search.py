"""OPTIM minimization through the ASE socket, and one 2014 double-ended row.

Minimization uses the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not the minimization score.

``optim/de-lj38`` builds OPTIM from ``wales-27058.tar.bz2``. That tree
writes the NEWNEB RMS tolerance into ``RMSTOL``. The score is the last
``energy+gradient calls`` field, and a start succeeds when the log
contains ``Converged to TS``.

``optim/unhooked`` and every other non-L-BFGS template raise before a
file is written.
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
WALES_27058_URL = "https://optbench.org/codes/wales-27058.tar.bz2"
WALES_27058_SHA256 = "144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"
_RELAY = Path(__file__).with_name("fmlip_relay.py")
_FCALL = re.compile(r"# of energy\+gradient calls=\s+(-?\d+)")


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


def wales_27058_binary() -> Path:
    """OPTIM built from the 2014 ``wales-27058.tar.bz2`` tarball."""
    env = os.environ.get("OPTIM_27058_BIN")
    if env:
        return Path(env)
    return Path("/tmp/wales27058/build/OPTIM")


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


def _symbol_lines(positions: np.ndarray, symbol: str) -> list[str]:
    lines = []
    for row in np.asarray(positions, dtype=np.float64):
        coords = " ".join(_fortran_float(float(value)) for value in row)
        lines.append(f"{symbol}  {coords}")
    return lines


def _bare_lines(positions: np.ndarray) -> str:
    lines = []
    for row in np.asarray(positions, dtype=np.float64):
        lines.append(" ".join(_fortran_float(float(value)) for value in row))
    return "\n".join(lines) + "\n"


def _paper_calls(text: str) -> int | None:
    """Last energy+gradient field. ``run_optim.sh`` takes awk field 6."""
    found = _FCALL.findall(text)
    if not found:
        return None
    return int(found[-1])


def _dneb_record(name: str, calls: int | None, converged: bool, cap: int, error: str) -> dict:
    """One start under the 2014 ``run_optim.sh`` rule."""
    from optbench.failures import classify_failure

    reasons: list[str] = []
    if error:
        reasons.append("crash")
    counted = 0 if calls is None else int(calls)
    if calls is None and not reasons:
        reasons.append("crash")
    if counted >= cap:
        reasons.append("cap")
    success = bool(converged) and not reasons and counted < cap
    if success:
        reasons = []
    elif not reasons:
        reasons.append("divergence")
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": counted,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": classify_failure(reasons),
        "reasons": reasons,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": 1,
        "converged": success,
        "error": error,
    }


def _dneb_one(job: dict) -> dict:
    """One wales-27058 DNEB plus hybrid eigenvector-following search."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = np.asarray(job["product"], dtype=np.float64)
    cap = int(job["cap"])
    error = ""
    calls = None
    converged = False
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="optim-27058-") as raw:
            folder = Path(raw)
            lines = [str(job["preamble"]).rstrip(), "points", *_symbol_lines(reactant, str(job["symbol"]))]
            (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (folder / "finish").write_text(_bare_lines(product), encoding="utf-8")
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
            except subprocess.TimeoutExpired:
                proc = None
                error = "OPTIM exceeded the wall limit"
            if proc is not None:
                log = proc.stdout
                calls = _paper_calls(log)
                converged = "Converged to TS" in log
                if proc.returncode != 0 and calls is None and error == "":
                    detail = proc.stderr.strip().splitlines()
                    tail = detail[-1] if detail else f"exit {proc.returncode}"
                    error = f"OPTIM exit {proc.returncode}: {tail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _dneb_record(str(job["name"]), calls, converged, cap, error)


def _dneb_job(pair, settings: dict, template: dict, preamble: str, binary: Path) -> dict:
    section = template.get("optim") or {}
    return {
        "name": pair.name,
        "reactant": pair.reactant,
        "product": pair.product,
        "preamble": preamble,
        "symbol": str(section.get("symbol", "AX")),
        "cap": int(settings["cap"]),
        "timeout": int(section.get("timeout", 600)),
        "binary": str(binary),
    }


def _run_dneb(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    from .common import band_settings, cached_pairs

    section = template.get("optim") or {}
    binary = wales_27058_binary()
    if not binary.is_file():
        raise RuntimeError(
            f"{template['id']} needs the OPTIM binary from {WALES_27058_URL}. "
            f"The pin is sha256 {WALES_27058_SHA256}. "
            f"{binary} is not present. Build it with optbench/wales27058_build.sh "
            "or set OPTIM_27058_BIN."
        )
    if template.get("commit") != WALES_27058_SHA256:
        raise ValueError(f"{template['id']} commit does not match wales-27058")
    preamble_path = root / str(section.get("preamble", ""))
    if not preamble_path.is_file():
        raise RuntimeError(f"{template['id']} preamble is not at {preamble_path}")
    settings = band_settings(problem)
    if settings["potential"] != "lj":
        raise ValueError(f"{template['id']} uses builtin LJDIFF and {settings['potential']} is not lj")
    preamble = preamble_path.read_text(encoding="utf-8")
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_dneb_job(pair, settings, template, preamble, binary) for pair in pairs]
    records = run_jobs(jobs, _dneb_one, workers)
    comment = (
        "# counter harness. The score is OPTIM's last energy+gradient call count "
        "from wales-27058. A success is a log that contains Converged to TS.\n"
        f"# odata preamble {preamble_path.name} is the 2014 archive file. "
        f"Symbol {section.get('symbol', 'AX')} is LJDIFF. "
        f"Archive force_calls {section.get('archive_force_calls', '')}."
    )
    parameters = {
        "method": "dneb",
        "preamble": str(section.get("preamble", "")),
        "symbol": str(section.get("symbol", "AX")),
        "counter": "energy+gradient",
        "binary": WALES_27058_SHA256,
        "archive_force_calls": section.get("archive_force_calls"),
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


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    section = _section(template)
    if problem.get("suite") == "double-ended-saddle-search" and section["method"] == "dneb":
        if template.get("code") != CODE:
            raise ValueError(f"{template_name} code is not {CODE}")
        return _run_dneb(root, problem, template, output=output, workers=workers)
    reason = available()
    if reason:
        raise RuntimeError(reason)
    require_pin(template, registry_pin(root, CODE))
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
