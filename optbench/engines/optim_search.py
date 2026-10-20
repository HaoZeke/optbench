"""OPTIM L-BFGS through the ASE socket, and one 2014 saddle search.

The minimization binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not that score.

``optim/hef-ffd-ts-pt-latest`` is the current snapshot on the Pt-island
hybrid eigenvector-following search. The archive names that method with
a forward finite-difference second derivative. OPTIM's keyword for the
search is ``BFGSTS``; ``USEDIAG 3`` and ``NOHESS`` select the derivative.
``FREEZE`` is the substrate fixed in the con file. ``BULK`` is the
orthorhombic cell in the same reduced units as ``PARAMS``. ``BFGSCONV``
is the problem L2 stop as an RMS over the free coordinates. ``STEPS``,
``BFGSSTEPS``, ``NEVS``, and ``NEVL`` are the problem cap.
``NBFGSMAX1`` and ``NBFGSMAX2`` stay at the source defaults. The harness
stops the search at the force-call cap. The score is the energy-plus-gradient
count.

``optim/unhooked`` and every other template that is not minimization
L-BFGS raise before a file is written. Noise, NEB, and PATHSAMPLE are
not this runner.
"""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .common import judge_saddle, run_jobs, write_campaign
from .geometries import load_starts, point_settings

ENGINE = "optim"
CODE = "optim"
SNAPSHOT = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
SHA256 = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
_RELAY = Path(__file__).with_name("fmlip_relay.py")
_CALLCAP_SRC = Path(__file__).with_name("optim_callcap.c")


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


# ``bc -l`` output of the archived run.sh expressions. r0 = 2.8970,
# rho = 1.6047, rcut = 9.5. Scale is 1/r0, rho is multiplied by r0,
# and rcut is divided by r0.
_BC_SCALE = ".34518467380048325854"
_BC_RHO = "4.64881590"
_BC_RCUT = "3.27925440110459095616"
# keywords.F90 initialises CEIG to this value. It is not a step limit.
_CEIG = "1.0D-10"
# Tangent-space steps per outer iteration. keywords.F90 initialises these
# to 10 and 100. The force-call cap is a different limit: one tangent-space
# loop exits on the unprojected RMS, so a cap-sized allowance spends the
# whole budget before the uphill step.
_NBFGSMAX1 = 10
_NBFGSMAX2 = 100


def snapshot_binary() -> Path:
    """Current Wales OPTIM. ``OPTIM_BIN`` wins, then the local build."""
    env = os.environ.get("OPTIM_BIN")
    if env:
        return Path(env)
    for candidate in (Path("/tmp/optimbld/OPTIM"), Path("/tmp/src/wales-full/build-optim/OPTIM")):
        if candidate.is_file():
            return candidate
    return Path("/tmp/optimbld/OPTIM")


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


def _published_con(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 2014 ``read_con_file`` layout: box on line 3, count on line 8, coords from line 12."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    box = np.array([float(part) for part in lines[2].split()[:3]], dtype=np.float64)
    count = int(float(lines[7].split()[0]))
    coords = np.zeros((count, 3), dtype=np.float64)
    frozen = np.zeros(count, dtype=bool)
    row = 0
    for line in lines[11:]:
        if row >= count:
            break
        parts = line.split()
        coords[row] = [float(parts[0]), float(parts[1]), float(parts[2])]
        frozen[row] = int(float(parts[3])) != 0
        row += 1
    if row != count:
        raise ValueError(f"{path} has {row} coordinates, the header states {count}")
    return coords, frozen, box


def _displacement(start: Path, reactant: Path, dest: Path) -> None:
    """genmode.x: eigenvalue 100, then the unscaled start minus the reactant."""

    def points(path: Path) -> list[tuple[float, float, float]]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        found = []
        for line in lines[11:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                found.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                break
        return found

    left = points(start)
    right = points(reactant)
    if len(left) != len(right):
        raise ValueError(f"{start.name} and {reactant.name} have different atom counts")
    lines = ["  100.0"]
    for (x1, y1, z1), (x2, y2, z2) in zip(left, right, strict=True):
        lines.append(f" {x1 - x2: .16e} {y1 - y2: .16e} {z1 - z2: .16e}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _callcap_binary() -> Path:
    """Compile the gradient-call stop. A newer source replaces the binary."""
    binary = Path(tempfile.gettempdir()) / "optbench-optim-callcap"
    if binary.is_file() and binary.stat().st_mtime >= _CALLCAP_SRC.stat().st_mtime:
        return binary
    subprocess.run(
        ["gcc", "-O2", "-o", str(binary), str(_CALLCAP_SRC)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def _text_symbol(binary: Path, name: str) -> int:
    text = subprocess.check_output(["nm", str(binary)], text=True, errors="replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "T" and parts[2] == name:
            return int(parts[0], 16)
    raise RuntimeError(f"{binary} has no text symbol {name}")


def _reduced_box(box: np.ndarray) -> str:
    """Orthorhombic cell in the reduced units ``PARAMS`` already uses."""
    lengths = np.asarray(box, dtype=np.float64) * float(_BC_SCALE)
    return "%f %f %f" % (float(lengths[0]), float(lengths[1]), float(lengths[2]))


def _cap_status(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    found: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2:
            found[parts[0]] = int(parts[1])
    if "gradient" not in found:
        return None
    return found


def _log_stop(log_text: str) -> str:
    """Last RMS line, so a failure names the stop OPTIM actually reached."""
    last = ""
    last_rms = ""
    fractional = "fractional coordinates" in log_text
    for line in log_text.splitlines():
        text = " ".join(line.split())
        if not text:
            continue
        last = text
        if "RMS" in text:
            last_rms = text
    chosen = last_rms or last or "no OPTIM log"
    if fractional:
        chosen = "OPTIM used fractional coordinates; " + chosen
    return chosen[:300]


def _stock_gmax(n_free: int, force_norm: float) -> float:
    """Problem L2 stop as OPTIM's RMS over the free coordinates.

    OPTIM sets the RMS to ``sqrt(sum(g^2) / (3 N_free))`` after zeroing
    frozen atoms. The native Morse energy is in units of the well depth
    ``A = 0.7102``, so the threshold is that RMS divided by ``A``.
    """
    count = 3 * int(n_free)
    if count < 1:
        raise ValueError("RMS stop needs a free coordinate")
    return float(force_norm) / math.sqrt(count) / 0.7102


def _hef_odata(
    folder: Path,
    coords: np.ndarray,
    frozen: np.ndarray,
    box: np.ndarray,
    cap: int,
    force_norm: float,
) -> None:
    """Hybrid eigenvector-following with a forward finite-difference curvature.

    ``BFGSTS`` is OPTIM's keyword for that search. ``USEDIAG 3`` is the
    archive's first-order forward difference. ``NOHESS`` uses that
    derivative instead of the analytic Hessian. ``BULK`` carries the
    orthorhombic cell in the same reduced units as ``PARAMS``.     ``STEPS``, ``BFGSSTEPS``, ``NEVS``, and ``NEVL`` are the problem cap.
    ``NBFGSMAX1`` and ``NBFGSMAX2`` stay at the source defaults.
    ``CEIG`` stays at the source default.
    """
    scale = float(_BC_SCALE)
    scaled = coords * scale
    box_text = _reduced_box(box)
    n_free = int(coords.shape[0] - np.count_nonzero(frozen))
    gmax = _stock_gmax(n_free, force_norm)
    limit = int(cap)
    lines = [
        "COMMENT hybrid eigenvector-following, forward finite-difference second derivatives",
        f"BFGSTS {limit} {_NBFGSMAX1} {_NBFGSMAX2} {_CEIG} {limit}",
        f"BFGSSTEPS {limit}",
        f"STEPS {limit}",
        f"BFGSCONV {_fortran_float(gmax)}",
        f"BULK {box_text}",
        "NOHESS",
        "USEDIAG 3",
        "READVEC",
        f"PARAMS {_BC_RHO} {box_text} {_BC_RCUT}",
    ]
    indices = [str(index) for index in (np.flatnonzero(frozen) + 1).tolist()]
    for start in range(0, len(indices), 8):
        lines.append("FREEZE " + " ".join(indices[start : start + 8]))
    lines.append("points")
    for row in scaled:
        lines.append("M " + " ".join(f"{float(value):.16f}" for value in row))
    (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paper_calls(log_text: str) -> tuple[int | None, bool]:
    calls = None
    for line in log_text.splitlines():
        if "energy+gradient calls" in line:
            calls = int(line.split()[5])
    return calls, "**** CONVERGED ****" in log_text


def _execute_hef(job: dict) -> dict:
    """One native OPTIM search. The harness stops it at the force-call cap."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    name = str(job["name"])
    cap = int(job["cap"])
    calls = cap
    success = False
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-hef-") as raw:
            folder = Path(raw)
            coords, frozen, box = _published_con(Path(job["con"]))
            _hef_odata(folder, coords, frozen, box, cap, float(job["force_norm"]))
            _displacement(Path(job["con"]), Path(job["reactant"]), folder / "vector.dump")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["GFORTRAN_UNBUFFERED_ALL"] = "1"
            stdout_path = folder / "OPTIM.log"
            status_path = folder / "callcap.txt"
            command = [
                str(job["tracer"]),
                hex(int(job["symbol"])),
                str(cap),
                str(status_path),
                str(job["binary"]),
            ]
            proc = None
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout:
                    proc = subprocess.Popen(
                        command,
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                if proc is not None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                error = "OPTIM exceeded 1800s"
            log_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
            detail = _log_stop(log_text)
            counted = _cap_status(status_path)
            parsed, converged = _paper_calls(log_text)
            gradient = None if counted is None else int(counted["gradient"])
            capped = bool(counted and int(counted.get("capped", 0)))
            if capped:
                calls = cap
                error = f"harness stopped the search at {cap} energy+gradient calls; {detail}"
            elif gradient is None:
                if parsed is not None:
                    calls = parsed
                error = error or "the harness call counter wrote no count"
                if detail and detail not in error:
                    error = f"{error}; {detail}"
            else:
                calls = int(gradient if parsed is None else parsed)
                if parsed is not None and parsed != gradient:
                    error = (
                        f"harness counted {gradient} energy+gradient calls; "
                        f"OPTIM printed {parsed}; {detail}"
                    )
            returncode = None if proc is None else proc.returncode
            if gradient is None or error.startswith("harness counted"):
                success, failure, reasons = False, "crash", ["crash"]
            else:
                success, failure, reasons, error = _hef_outcome(
                    calls,
                    cap,
                    converged=converged,
                    parsed=parsed if parsed is not None else gradient,
                    error=error,
                    returncode=returncode,
                )
                if not success and detail and detail not in error:
                    error = f"{error}; {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        success = False
        failure = "crash"
        reasons = ["crash"]
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "hessian_index": None,
        "expected_index": 1,
        "converged": success,
    }
    if error and not success:
        record["error"] = error
    return record


def _hef_outcome(
    calls: int,
    cap: int,
    *,
    converged: bool,
    parsed: int | None,
    error: str,
    returncode: int | None,
) -> tuple[bool, str | None, list[str], str]:
    """Success is convergence under the force-call cap. A finished overrun is a cap failure."""
    from optbench.failures import classify_failure

    success = converged and parsed is not None and calls < cap
    if success:
        return True, None, [], ""
    timed_out = error.startswith("OPTIM exceeded")
    bad_exit = returncode not in (None, 0)
    if timed_out or (bad_exit and parsed is None):
        reason = "crash"
        if bad_exit and not error:
            error = f"OPTIM exit {returncode}"
    elif calls >= cap:
        reason = "cap"
        if not error:
            error = f"OPTIM used {calls} energy+gradient calls, at or above the cap {cap}"
    else:
        reason = "divergence"
        if not error:
            error = "OPTIM did not converge"
    return False, classify_failure([reason]), [reason], error


def _run_hef(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    if str(template.get("code") or "") != CODE:
        raise ValueError(f"{template['id']} is not the current OPTIM forward-difference search")
    binary = snapshot_binary()
    if not binary.is_file():
        raise RuntimeError(
            "The current OPTIM snapshot is not built. "
            f"The pin is the sha256 {SHA256} of {SNAPSHOT} "
            "(122743078 bytes). Set OPTIM_BIN to that OPTIM binary."
        )
    from .geometries import _extract, _stored, load_starts

    scratch = output.parent / "work" / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    folder = _extract(_stored(root, problem["input"]["archive_path"]), scratch / "geometries")
    island = folder / "pt-island-con"
    reactant = island / "reactant.con"
    if not reactant.is_file():
        raise FileNotFoundError(reactant)
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    force_norm = float((problem.get("convergence") or {})["force_norm"])
    sample, sample_frozen, sample_box = _published_con(island / "initial_0.con")
    n_free = int(sample.shape[0] - np.count_nonzero(sample_frozen))
    box_text = _reduced_box(sample_box)
    tracer = _callcap_binary()
    symbol = _text_symbol(binary, "morse_bulk_wrapper_")
    jobs = []
    for start in starts:
        con = island / f"{start['name']}.con"
        if not con.is_file():
            raise FileNotFoundError(con)
        jobs.append(
            {
                "name": start["name"],
                "con": str(con),
                "reactant": str(reactant),
                "binary": str(binary),
                "tracer": str(tracer),
                "symbol": symbol,
                "cap": cap,
                "force_norm": force_norm,
            }
        )
    records = run_jobs(jobs, _execute_hef, workers)
    settings = point_settings(problem)
    settings["cap"] = cap
    comment = (
        "# counter harness. OPTIM df8da599 hybrid eigenvector-following.\n"
        "# USEDIAG 3 is the forward finite-difference second derivative. NOHESS selects it.\n"
        "# FREEZE is the substrate fixed in the con file. BULK is the orthorhombic cell\n"
        "# in the same reduced units as PARAMS. BFGSCONV is the problem L2 stop as an RMS\n"
        "# over the free coordinates, divided by the Morse well depth.\n"
        "# STEPS, BFGSSTEPS, NEVS, and NEVL are the problem cap. "
        "NBFGSMAX1 and NBFGSMAX2 stay at 10 and 100. CEIG stays 1.0D-10.\n"
        "# The harness stops a search at the force-call cap. "
        "A success is **** CONVERGED **** under that cap."
    )
    parameters = {
        "method": "hef-ffd",
        "keyword": "BFGSTS",
        "usediag": 3,
        "steps": cap,
        "bfgssteps": cap,
        "nevs": cap,
        "nbfgsmax1": _NBFGSMAX1,
        "nbfgsmax2": _NBFGSMAX2,
        "nevl": cap,
        "ceig": _CEIG,
        "scale": _BC_SCALE,
        "rho": _BC_RHO,
        "rcut": _BC_RCUT,
        "bulk": box_text,
        "n_free": n_free,
        "counter": "harness",
        "binary": "df8da599",
        "bfgsconv": _stock_gmax(n_free, force_norm),
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
        counter="harness",
    )


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, str(template.get("code") or CODE)))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    section = _section(template)
    if section["method"] == "hef-ffd":
        if problem_name != "tsopt/pt-island":
            raise ValueError(f"{template_name} is the Pt-island forward-difference search")
        return _run_hef(root, problem, template, output, workers)
    reason = available()
    if reason:
        raise RuntimeError(reason)
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
