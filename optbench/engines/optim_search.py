"""OPTIM L-BFGS through the ASE socket.

The binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not the score.

``optim/unhooked`` and every template that is not minimization L-BFGS
or the current-image DNEB row raise before a file is written. The
current-image row runs the latest ``svn.tar.bz2`` and writes OPTIM's
energy+gradient count as a harness score. Noise and PATHSAMPLE are not
this runner.
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
        "Minimization L-BFGS calls the ASE socket. "
        "The current-image Pt-island DNEB row is optim/de-pt-stock. "
        "Noise and PATHSAMPLE are not scored."
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


# bc -l values from archive run.sh: 1/r0, rho*r0, rcut/r0.
_SCALE = ".34518467380048325854"
_RHO_SCALED = "4.64881590"
_RCUT_SCALED = "3.27925440110459095616"

def _optim_tally(text: str) -> tuple[int | None, bool]:
    """Last energy+gradient count and whether the log converged to a TS.

    ``run_optim.sh`` takes awk field 6 of the energy+gradient line and
    treats a missing ``Converged to TS`` line as a failure.
    """
    calls = None
    for line in text.splitlines():
        if "energy+gradient calls" not in line or "Hessian" in line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            calls = int(parts[5])
        except ValueError:
            calls = None
    return calls, "Converged to TS" in text


def _freeze_lines(free: np.ndarray) -> list[str]:
    frozen = np.flatnonzero(~np.asarray(free, dtype=bool)) + 1
    lines = []
    for start in range(0, len(frozen), 8):
        chunk = " ".join(str(int(index)) for index in frozen[start : start + 8])
        lines.append(f"FREEZE {chunk}")
    return lines


def _symbol_block(path: Path, symbol: str) -> list[str]:
    lines = []
    for row in path.read_text(encoding="utf-8").splitlines():
        parts = row.split()
        if len(parts) < 3:
            continue
        lines.append(f"{symbol} " + " ".join(parts[:3]))
    return lines


def _native_inputs(folder: Path, job: dict) -> None:
    """odata and finish in the form archive run.sh writes."""
    scale = float(_SCALE)
    reactant = np.asarray(job["reactant"], dtype=np.float64) * scale
    product = np.asarray(job["product"], dtype=np.float64) * scale
    box = np.asarray(job["box"], dtype=np.float64).reshape(-1)[:3] * scale
    np.savetxt(folder / "coords", reactant, fmt="%.16f")
    np.savetxt(folder / "finish", product, fmt="%.16f")
    box_text = "%f %f %f" % tuple(float(value) for value in box)
    lines = [
        str(job["preamble"]).rstrip("\n"),
        f"PARAMS {_RHO_SCALED} {box_text} {_RCUT_SCALED}",
        *_freeze_lines(np.asarray(job["free"], dtype=bool)),
        "points",
        *_symbol_block(folder / "coords", str(job["symbol"])),
    ]
    (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _native_job(pair, template: dict, preamble: str, cap: int) -> dict:
    section = template.get("optim") or {}
    return {
        "name": pair.name,
        "reactant": pair.reactant,
        "product": pair.product,
        "free": pair.free,
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64),
        "preamble": preamble,
        "symbol": str(section.get("symbol", "M")),
        "timeout": int(section.get("timeout", 120)),
        "cap": int(cap),
        "problem": template.get("problem", ""),
        "template": template["id"],
        "code": template["code"],
        "code_version": template["code_version"],
        "commit": template["commit"],
        "algorithm": template["algorithm"],
        "tuning": template["tuning"],
    }


_CURRENT_TARBALL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
_CURRENT_SHA256 = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
_CURRENT_VERSION = "df8da599"
_CURRENT_CODE = "optim-df8da599"
_CURRENT_SOURCE = "df8da599267534fbe43a250e578f7396a7946311"
_CURRENT_BINARY = Path("/tmp/wales-latest/wales/OPTIM/builds/gfortran/OPTIM")
_STOCK_KEYWORDS = (
    "NEWCONNECT\n"
    "NEWNEB 6 3000\n"
    "BFGSTS 11 7 30 0.08 100\n"
    "BFGSMIN 0.001\n"
    "BFGSSTEPS 3000\n"
    "STEPS 3000\n"
    "NOHESS\n"
    "BULK 6.630583 6.562582 10.355540\n"
    "NONEBMIND\n"
    "USEDIAG 3\n"
    "OPTIMIZETS\n"
)


def current_binary_path() -> Path:
    env = os.environ.get("OPTIM_CURRENT_BIN")
    if env:
        return Path(env)
    return _CURRENT_BINARY


def _stock_search(job: dict) -> dict:
    """One current OPTIM DNEB search.

    The stored count is the last energy+gradient line. A start is a
    success when the log contains ``Converged to TS`` and the count is
    under the problem cap.
    """
    import tempfile
    import time

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    calls = None
    converged = False
    error = ""
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="optim-stock-") as raw:
            folder = Path(raw)
            _native_inputs(folder, job)
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            stdout_path = folder / "OPTIM.log"
            stderr_path = folder / "OPTIM.err"
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr:
                    proc = subprocess.run(
                        [str(job["binary"])],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=int(job.get("timeout", 120)),
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                error = "OPTIM exceeded the wall limit"
                proc = None
            log = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
            calls, converged = _optim_tally(log)
            if proc is not None and proc.returncode not in (0, None) and error == "":
                detail = _tail(stderr_path) or _tail(stdout_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
            elif calls is None and error == "":
                detail = _tail(stdout_path) or _tail(stderr_path) or "no energy+gradient line"
                error = f"OPTIM produced no energy+gradient count: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    cap = int(job["cap"])
    counted = None if calls is None else int(calls)
    success = bool(converged) and counted is not None and counted < cap and not error
    if success:
        failure = None
    elif counted is None or error:
        failure = "crash"
    elif counted >= cap:
        failure = "cap"
    else:
        failure = "divergence"
    record = {
        "schema": "optbench.start/v2",
        "start": job["name"],
        "force_calls": cap if counted is None else counted,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": [],
        "error": error,
        "converged": bool(converged),
        "optim_method": "dneb-stock",
        "optim_energy_gradient_calls": None if calls is None else int(calls),
        "wall_s": float(time.perf_counter() - started),
    }
    return record


def _run_stock_dneb(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    from .common import band_settings, cached_pairs, write_campaign

    binary = current_binary_path()
    if not binary.is_file():
        raise RuntimeError(
            f"OPTIM {_CURRENT_VERSION} is not available. "
            f"The pin is the sha256 {_CURRENT_SHA256} of {_CURRENT_TARBALL}. "
            f"{binary} is not present."
        )
    section = template.get("optim") or {}
    settings = band_settings(problem)
    if settings["potential"] != "morse_pt":
        raise ValueError(f"{template['id']} stock DNEB is the Pt Morse row")
    preamble_path = root / str(section.get("preamble", ""))
    if not preamble_path.is_file():
        raise RuntimeError(f"{template['id']} preamble is not at {preamble_path}")
    preamble = preamble_path.read_text(encoding="utf-8")
    if preamble != _STOCK_KEYWORDS:
        raise RuntimeError(f"{template['id']} preamble is not the stock keyword block")
    pairs = list(cached_pairs(root, problem, settings).values())
    pairs.sort(key=lambda pair: int(pair.name.rsplit("_", 1)[-1]))
    if len(pairs) != int(settings["pairs"]):
        raise RuntimeError(f"{template['id']} saw {len(pairs)} pairs, the problem states {settings['pairs']}")
    jobs = []
    for pair in pairs:
        job = _native_job(pair, template, preamble, int(settings["cap"]))
        job["binary"] = str(binary)
        jobs.append(job)
    records = run_jobs(jobs, _stock_search, workers)
    published = dict(template)
    published["code"] = "optim"
    comment = (
        f"# OPTIM source {_CURRENT_SOURCE}, {_CURRENT_TARBALL}. "
        f"sha256 {_CURRENT_SHA256}. INCLUDE revision 26635.\n"
        "# NEWCONNECT with NEWNEB 6 3000, BFGSTS 11 7 30 0.08 100, "
        "BFGSMIN 0.001, BFGSSTEPS 3000, STEPS 3000, USEDIAG 3, OPTIMIZETS.\n"
        "# Six images is the optim-6 entry: comments.txt says six images are used in DNEB. "
        "NEWNEB's first field is that interior count. "
        "The problem file band.interior_images is 3, the divisor the other rows use.\n"
        "# STEPS is the outer transition-state step count and equals the problem cap 3000. "
        "BFGSSTEPS and the NEWNEB iteration field are that same cap. "
        "The BFGSTS fields are the archive eigenvector search: 11 smallest-eigenvector steps, "
        "7 and 30 tangent-space steps, eigenvalue RMS 0.08, and 100 largest-eigenvector steps. "
        "The archive deck has no CONVERGE line and no MAXERISE line, so those stay at OPTIM defaults. "
        "BFGSMIN sets the RMS gradient to the problem value 0.001. "
        "USEDIAG 3 is the archive first-order Rayleigh-Ritz difference. "
        "OPTIMIZETS is the archive eigenvector-following keyword; NEWCONNECT sets the same flag. "
        "The band RMS stays at the OPTIM default 0.01.\n"
        "# BULK repeats the PARAMS cell so the connection is not run in a zero box. "
        "Spring constant, push-off, and image density stay at OPTIM defaults.\n"
        "# NOHESS is set because morse bulk has no analytic second derivatives. "
        "NONEBMIND keeps the supplied endpoints.\n"
        "# The count is the last energy+gradient line. That is the band total the archive script stored, "
        "and the unit the other rows rank. The cap 3000 is that total. "
        "A start is kept when the log contains Converged to TS and the tally is under the cap."
    )
    return write_campaign(
        output,
        records,
        published,
        problem,
        settings,
        {
            "method": "dneb-stock",
            "keyword": "NEWCONNECT NEWNEB BFGSTS",
            "counter": "optim energy+gradient calls",
            "potential": "morse_bulk",
            "version": _CURRENT_VERSION,
            "source": _CURRENT_SOURCE,
        },
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
    if section["method"] == "dneb-stock":
        require_pin(template, registry_pin(root, _CURRENT_CODE))
        return _run_stock_dneb(root, problem, template, output=output, workers=workers)
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
