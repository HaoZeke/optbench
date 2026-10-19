"""OPTIM L-BFGS through the ASE socket.

The binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not the score.

``optim/unhooked`` raises before a file is written. Minimization L-BFGS,
the archived double-ended preambles, and the Pt-island saddle search
call the ASE socket. Noise, NEB, and PATHSAMPLE are not this runner.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
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
        "Minimization L-BFGS and the archived double-ended preambles call the ASE socket. "
        "This template does not."
    )


# Pt Morse constants from the 2014 pt-island OPTIM run.sh.
_R0 = 2.897
_WELL = 0.7102
_RHO = 1.6047
_RCUT = 9.5


def _pair_scale(kind: str) -> tuple[float, float]:
    """Length and energy factors from engine units into the harness.

    ``l_scale`` is ``OPTBENCH_L_SCALE``: harness coordinates are the
    OPTIM coordinates divided by this factor. Morse uses the 2014
    scaling, coordinates over ``r0`` and energy over ``D``.
    """
    if kind == "morse_pt":
        return 1.0 / _R0, 1.0 / _WELL
    return 1.0, 1.0


def _ase_line(python: str, hook: Path, periodic: bool) -> str:
    pbc = "PBC" if periodic else "NOPBC"
    return f"ASE geom.xyz {python} {hook} dummy dummy cpu FIXCELL {pbc} EV ANG"


def _install_hook(folder: Path) -> Path:
    hook = folder / "hook"
    relay = hook / "pysocket" / "fmlip_relay.py"
    relay.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_RELAY, relay)
    return hook


def _freeze_lines(free: np.ndarray) -> list[str]:
    frozen = [str(index + 1) for index, flag in enumerate(np.asarray(free, dtype=bool)) if not flag]
    return [f"FREEZE {index}" for index in frozen]


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


def _params_line(box: np.ndarray, scale_l: float) -> str:
    reduced = np.asarray(box, dtype=np.float64).reshape(-1)[:3] * scale_l
    values = [_RHO * _R0, float(reduced[0]), float(reduced[1]), float(reduced[2]), _RCUT * scale_l]
    body = " ".join(_fortran_float(value) for value in values)
    return f"PARAMS {body}"


def _optim_env(
    folder: Path,
    kind: str,
    cap: int,
    box,
    scale_l: float,
    scale_e: float,
    call_log: Path | None,
    traj: Path | None = None,
) -> dict:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPTBENCH_KIND"] = kind
    env["OPTBENCH_CAP"] = str(cap)
    env["OPTBENCH_COUNT_PATH"] = str(folder / "calls.txt")
    env["OPTBENCH_LAST_PATH"] = str(folder / "last.txt")
    env["OPTBENCH_CAPPED_PATH"] = str(folder / "capped.txt")
    env["OPTBENCH_E_SCALE"] = _fortran_float(scale_e).replace("D", "E")
    env["OPTBENCH_L_SCALE"] = _fortran_float(scale_l).replace("D", "E")
    if call_log is None:
        env.pop("OPTBENCH_CALL_LOG", None)
    else:
        env["OPTBENCH_CALL_LOG"] = str(call_log)
    if traj is None:
        env.pop("OPTBENCH_TRAJ_PATH", None)
    else:
        env["OPTBENCH_TRAJ_PATH"] = str(traj)
    if box is None:
        env.pop("OPTBENCH_BOX", None)
    else:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        env["OPTBENCH_BOX"] = " ".join(f"{float(value):.16g}" for value in lengths)
    return env


def _xyz_frames(path: Path, natoms: int) -> list[np.ndarray]:
    """XYZ frames. The species token and the comment line are ignored."""
    if not path.is_file():
        return []
    frames: list[np.ndarray] = []
    rows: list[list[float]] = []
    expecting = 0
    skip_comment = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if skip_comment:
            skip_comment = False
            continue
        if expecting:
            parts = text.replace("D", "E").replace("d", "e").split()
            if len(parts) < 3:
                return frames
            try:
                rows.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
            except ValueError:
                return frames
            expecting -= 1
            if expecting == 0:
                if len(rows) == natoms:
                    frames.append(np.asarray(rows, dtype=np.float64))
                rows = []
            continue
        if text.isdigit():
            expecting = int(text)
            rows = []
            skip_comment = True
    return frames


def _xyz_last(path: Path, natoms: int) -> np.ndarray | None:
    """Last XYZ frame. The species token is ignored."""
    frames = _xyz_frames(path, natoms)
    if not frames:
        return None
    return frames[-1]


def _energy_token(line: str) -> float | None:
    token = line.replace("D", "E").split()[-1]
    try:
        return float(token)
    except ValueError:
        return None


def _mylbfgs_energies(text: str) -> list[float]:
    """Each ``mylbfgs> Final energy is`` value, in OPTIM's energy unit."""
    found = []
    for line in text.splitlines():
        if "mylbfgs> Final energy is" not in line:
            continue
        value = _energy_token(line)
        if value is not None:
            found.append(value)
    return found


def _ts_energies(text: str) -> list[float]:
    """Final energies that the log then calls a converged transition state.

    NEWCONNECT reoptimises the endpoints after the path. Those later
    minima are also ``mylbfgs`` final energies. The saddle is the energy
    printed just before ``Converged to TS``.
    """
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        if "mylbfgs> Final energy is" not in line:
            continue
        window = "\n".join(lines[index:index + 6])
        if "Converged to TS" not in window:
            continue
        value = _energy_token(line)
        if value is not None:
            found.append(value)
    return found


def _geometry_file(
    folder: Path,
    natoms: int,
    scale_l: float,
    kind: str,
    box,
    scale_e: float,
    force_norm: float,
) -> np.ndarray | None:
    """Transition-state geometry in harness units.

    A converging ``mylbfgs`` call returns when the RMS test passes and
    does not dump that point into ``lbfgs.xyz``. The file still holds
    the geometry from the start of the call. ``last.txt`` is the
    curvature probe after that return. The socket trajectory holds the
    point whose energy is the printed final energy.
    """
    from optbench.pes import evaluate

    candidates: list[np.ndarray] = []
    for path in sorted(folder.glob("lbfgs.xyz*")):
        frame = _xyz_last(path, natoms)
        if frame is not None:
            candidates.append(frame / scale_l)
    bare = _read_points(folder / "points.final", natoms)
    if bare is not None:
        candidates.append(bare / scale_l)
    # traj.xyz is already in harness units. lbfgs.xyz is in OPTIM units.
    candidates.extend(_xyz_frames(folder / "traj.xyz", natoms))
    log = _joined_logs(folder)
    targets = _ts_energies(log) or _mylbfgs_energies(log)
    if targets and candidates:
        scored: list[tuple[np.ndarray, float, float]] = []
        for coords in candidates:
            energy, gradient = evaluate(kind, coords, box)
            force = float(np.linalg.norm(np.asarray(gradient, dtype=np.float64)))
            scored.append((coords, float(energy), force))
        for target in reversed(targets):
            harness_energy = float(target) / float(scale_e)
            matches = [row for row in scored if abs(row[1] - harness_energy) < 1e-8]
            if not matches:
                continue
            under = [row for row in matches if row[2] <= float(force_norm)]
            if under:
                return under[-1][0]
            return min(matches, key=lambda row: row[2])[0]
        harness_energy = float(targets[-1]) / float(scale_e)
        best = min(scored, key=lambda row: abs(row[1] - harness_energy))
        if abs(best[1] - harness_energy) < 1e-4:
            return best[0]
    if candidates:
        return candidates[-1]
    return _read_points(folder / "last.txt", natoms)


def _joined_logs(folder: Path) -> str:
    chunks = []
    for path in sorted(folder.glob("OPTIM.*")):
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    stdout = folder / "stdout.txt"
    if stdout.is_file():
        chunks.append(stdout.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _count_file(path: Path) -> int:
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8").strip()
    return int(text or "0")


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


def _double_ended(job: dict) -> dict:
    """One archived OPTIM double-ended search. The score is the socket."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from optbench.pes import evaluate

    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = np.asarray(job["product"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    kind = str(job["kind"])
    cap = int(job["cap"])
    force_norm = float(job["force_norm"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    scale_l, scale_e = _pair_scale(kind)
    natoms = int(reactant.shape[0])
    start = reactant * scale_l
    finish = product * scale_l
    reduced_box = None if box is None else np.asarray(box, dtype=np.float64).reshape(-1)[:3] * scale_l
    point = None
    energy = None
    gradient = None
    calls = 0
    capped = False
    error = ""
    wall = 0.0
    import tempfile

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="optim-de-") as raw:
            folder = Path(raw)
            hook = _install_hook(folder)
            _write_xyz(folder / "geom.xyz", start, reduced_box)
            periodic = box is not None
            lines = [_ase_line(sys.executable, hook, periodic), job["preamble"].rstrip()]
            if kind == "morse_pt":
                if box is None:
                    raise RuntimeError("morse_pt double-ended search has no cell")
                lines.append(_params_line(box, scale_l))
                lines.extend(_freeze_lines(free))
            lines.append("points")
            lines.extend(_symbol_lines(start, str(job["symbol"])))
            (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (folder / "finish").write_text(_bare_lines(finish), encoding="utf-8")
            env = _optim_env(folder, kind, cap, box, scale_l, scale_e, None, folder / "traj.xyz")
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            proc = None
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.run(
                        [str(binary_path())],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=int(job.get("timeout", 3600)),
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                error = "OPTIM exceeded the wall limit"
            calls = _count_file(folder / "calls.txt")
            capped = (folder / "capped.txt").is_file() or calls >= cap
            if error == "":
                point = _geometry_file(folder, natoms, scale_l, kind, box, scale_e, force_norm)
            if proc is not None and proc.returncode not in (0, None) and not capped and error == "":
                detail = _tail(stderr_path) or _tail(stdout_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
            elif proc is not None and point is None and calls == 0 and not capped and error == "":
                detail = _tail(stdout_path) or _tail(stderr_path) or "no geometry"
                error = f"OPTIM produced no geometry: {detail}"
            elif point is None and error == "":
                detail = _tail(stdout_path) or _tail(stderr_path) or "no geometry"
                error = f"OPTIM produced no geometry: {detail}"
            if job.get("keep"):
                destination = Path(job["keep"])
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(folder, destination)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
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
        expected_index=1,
        rotations=rotations,
    )
    if error:
        record["error"] = error
    record["wall_s"] = float(wall)
    record["optim_method"] = str(job["method"])
    return record


def _de_job(pair, settings: dict, template: dict, preamble: str) -> dict:
    section = template.get("optim") or {}
    return {
        "name": pair.name,
        "reactant": pair.reactant,
        "product": pair.product,
        "free": pair.free,
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64),
        "kind": settings["potential"],
        "cap": int(settings["cap"]),
        "force_norm": float(settings["force_norm"]),
        "preamble": preamble,
        "symbol": str(section.get("symbol", "AX")),
        "method": str(section.get("method", "")),
        "timeout": int(section.get("timeout", 3600)),
    }


def _run_double_ended(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    from .common import band_settings, cached_pairs

    section = _section(template)
    settings = band_settings(problem)
    preamble_path = root / str(section.get("preamble", ""))
    if not preamble_path.is_file():
        raise RuntimeError(f"{template['id']} preamble is not at {preamble_path}")
    preamble = preamble_path.read_text(encoding="utf-8")
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_de_job(pair, settings, template, preamble) for pair in pairs]
    records = run_jobs(jobs, _double_ended, workers)
    walls = [float(row["wall_s"]) for row in records]
    comment = (
        "# counter harness. Each ASE COMPUTE is one pes.evaluate. "
        "OPTIM's energy+gradient calls are not the score.\n"
        f"# odata preamble {preamble_path.name} is the 2014 archive file. "
        f"Archive force_calls {section.get('archive_force_calls', '')}. "
        "The binary is the svn snapshot, not wales-27058.\n"
        f"# wall_s_median {statistics_median(walls)}"
    )
    parameters = {
        "method": section["method"],
        "preamble": str(section.get("preamble", "")),
        "symbol": str(section.get("symbol", "AX")),
        "scale": "angstrom/r0 and eV/D" if settings["potential"] == "morse_pt" else "lj",
        "counter": "harness",
        "socket": "ASE",
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


def statistics_median(values: list[float]) -> str:
    import statistics

    if not values:
        return "none"
    return f"{statistics.median(values):.6g}"


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    resume: bool = False,
    limit: int = 0,
) -> dict:
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
    section = _section(template)
    if problem.get("suite") == "saddle-search":
        from optbench.optim_saddle import run_saddle

        return run_saddle(
            root,
            problem,
            template,
            output=output,
            workers=workers,
            resume=resume,
            limit=limit,
        )
    if problem.get("suite") == "double-ended-saddle-search" and section["method"] in {"dneb", "newconnect"}:
        return _run_double_ended(root, problem, template, output=output, workers=workers)
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
