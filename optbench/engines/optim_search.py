"""OPTIM minimization through the ASE socket, and Pt-island DNEB.

Minimization uses the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not that score.

Pt-island DNEB runs wales-27058. The potential is the built-in
``morse_bulk`` in reduced units, and the score is OPTIM's own
``energy+gradient calls`` line. A success is ``Converged to TS``.

``optim/unhooked`` and every other template raise before a file is
written. Noise, single-ended saddle search, and PATHSAMPLE are not this
runner.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from optbench.failures import classify_failure

from .common import band_settings, cached_pairs, judge_saddle, run_jobs, write_campaign
from .geometries import load_starts, point_settings

ENGINE = "optim"
CODE = "optim"
SNAPSHOT = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
SHA256 = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
# sha256 of https://optbench.org/codes/wales-27058.tar.bz2. This is the
# paper binary. It is not the minimization snapshot pin.
WALES_27058 = "144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"
_R0 = 2.8970
_RHO = 1.6047
_RCUT = 9.5
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


def dneb_binary() -> Path:
    """wales-27058. Separate from the minimization socket binary."""
    env = os.environ.get("OPTIM_DNEB_BIN")
    if env:
        return Path(env)
    return Path("/tmp/wales/OPTIM/bin/gfortran/OPTIM")


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


def _require_27058(template: dict) -> None:
    if template.get("code") != CODE:
        raise ValueError(f"{template['id']} code is not {CODE}")
    if template.get("commit") != WALES_27058:
        raise ValueError(f"{template['id']} commit is not the wales-27058 tarball")
    if str(template.get("code_version")) != "27058":
        raise ValueError(f"{template['id']} code_version is not 27058")


def _paper_score(text: str) -> tuple[int, bool]:
    """Last ``energy+gradient calls`` field and the TS line.

    ``run_optim.sh`` takes awk field 6 of the last matching line. The
    Hessian line does not contain that phrase.
    """
    matched = [line for line in text.splitlines() if "energy+gradient calls" in line]
    calls = -1
    if matched:
        parts = matched[-1].split()
        if len(parts) >= 6:
            try:
                calls = int(parts[5])
            except ValueError:
                calls = -1
    return calls, "Converged to TS" in text


def _dneb_record(name: str, calls: int, success: bool, error: str) -> dict:
    reasons: list[str] = []
    if error or calls < 0:
        reasons.append("crash")
        success = False
    elif not success:
        reasons.append("divergence")
    if success:
        reasons = []
    failure = None if success else classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": calls if calls >= 0 else 0,
        "verify_calls": 0,
        "counter": "optim",
        "comparable": False,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": 1,
        "converged": failure is None,
        "error": error,
    }


def _odata_dneb(folder: Path, preamble: str, reactant: np.ndarray, box: np.ndarray, frozen: np.ndarray, symbol: str) -> None:
    rho = f"{_RHO * _R0:.16g}"
    rcut = f"{_RCUT / _R0:.16g}"
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
    box_text = " ".join(f"{float(value):.6f}" for value in lengths)
    lines = preamble.rstrip().splitlines()
    lines.append(f"PARAMS {rho} {box_text} {rcut}")
    indices = [str(int(index)) for index in np.asarray(frozen, dtype=np.int64).reshape(-1)]
    for start in range(0, len(indices), 8):
        lines.append("FREEZE " + " ".join(indices[start : start + 8]))
    lines.append("points")
    for row in np.asarray(reactant, dtype=np.float64):
        lines.append(f"{symbol} {row[0]:.16f} {row[1]:.16f} {row[2]:.16f}")
    (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute_dneb(job: dict) -> dict:
    import tempfile

    name = str(job["name"])
    try:
        with tempfile.TemporaryDirectory(prefix="optim-dneb-") as raw:
            folder = Path(raw)
            reactant = np.asarray(job["reactant"], dtype=np.float64)
            product = np.asarray(job["product"], dtype=np.float64)
            _odata_dneb(
                folder,
                str(job["preamble"]),
                reactant,
                np.asarray(job["box"], dtype=np.float64),
                np.asarray(job["frozen"], dtype=np.int64),
                str(job["symbol"]),
            )
            np.savetxt(folder / "finish", product, fmt="%.16f")
            stdout_path = folder / "OPTIM.log"
            stderr_path = folder / "stderr.txt"
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            text = ""
            error = ""
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.run(
                        [str(job["binary"])],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=int(job["timeout"]),
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                proc = None
                error = f"OPTIM exceeded {int(job['timeout'])}s"
            if stdout_path.is_file():
                text = stdout_path.read_text(encoding="utf-8", errors="replace")
            calls, converged = _paper_score(text)
            success = converged and calls >= 0 and not error
            if proc is not None and proc.returncode != 0 and not success and not error:
                detail = _tail(stderr_path) or _tail(stdout_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
            if not success and calls < 0 and not error:
                detail = _tail(stdout_path) or _tail(stderr_path) or "no energy+gradient line"
                error = f"OPTIM produced no call count: {detail}"
            return _dneb_record(name, calls, success, error)
    except Exception as exc:
        return _dneb_record(name, -1, False, f"{type(exc).__name__}: {exc}")


def _run_dneb(root: Path, problem: dict, template: dict, section: dict, output: Path, workers: int) -> dict:
    binary = dneb_binary()
    if not binary.is_file():
        raise RuntimeError(
            "Pt-island DNEB needs the wales-27058 OPTIM binary. "
            f"The tarball sha256 is {WALES_27058}. {binary} is not present."
        )
    _require_27058(template)
    preamble_path = Path(str(section.get("preamble") or ""))
    if not preamble_path.is_file():
        preamble_path = root / str(section.get("preamble") or "")
    if not preamble_path.is_file():
        raise ValueError(f"{template['id']} has no OPTIM preamble")
    preamble = preamble_path.read_text(encoding="utf-8")
    symbol = str(section.get("symbol") or "M")
    timeout = int(section.get("timeout", 7200))
    settings = band_settings(problem)
    scale = 1.0 / _R0
    jobs = []
    for pair in cached_pairs(root, problem, settings).values():
        if pair.box is None:
            raise ValueError(f"{pair.name} has no box")
        frozen = np.flatnonzero(~np.asarray(pair.free, dtype=bool)) + 1
        jobs.append(
            {
                "name": pair.name,
                "reactant": np.asarray(pair.reactant, dtype=np.float64) * scale,
                "product": np.asarray(pair.product, dtype=np.float64) * scale,
                "box": np.asarray(pair.box, dtype=np.float64) * scale,
                "frozen": frozen,
                "preamble": preamble,
                "symbol": symbol,
                "binary": str(binary),
                "timeout": timeout,
            }
        )
    order = {f"product_{index}": index for index in range(len(jobs))}
    jobs.sort(key=lambda job: order.get(job["name"], 10**9))
    records = run_jobs(jobs, _execute_dneb, workers)
    comment = (
        "# wales-27058 morse_bulk. Coordinates, the box, and rcut are divided by r0 2.897. "
        "PARAMS rho is 1.6047*r0. The potential is OPTIM's reduced Morse, A=1 and R0=1.\n"
        "# force_calls is the last OPTIM energy+gradient calls count. "
        "A success is the log line Converged to TS. Failed starts stay out of the mean."
    )
    parameters = {
        "method": "dneb",
        "binary": "wales-27058",
        "rho": _RHO * _R0,
        "r0": _R0,
        "rcut": _RCUT / _R0,
        "counter": "optim",
        "success": "Converged to TS",
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
        counter="optim",
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
    if section["method"] == "dneb":
        return _run_dneb(root, problem, template, section, output, workers)
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
