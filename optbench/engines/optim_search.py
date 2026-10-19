"""OPTIM L-BFGS through the ASE socket, and one 2014 saddle search.

The minimization binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not that score.

``optim/hef-ffd-ts-pt`` is the archived Pt-island hybrid eigenvector
following row. It runs the ``wales-27058`` binary on the native Morse
bulk potential. The score is that binary's energy-plus-gradient count.

``optim/unhooked`` and every other template that is not minimization
L-BFGS raise before a file is written. Noise, NEB, and PATHSAMPLE are
not this runner.
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


# ``bc -l`` output of the archived run.sh expressions. r0 = 2.8970,
# rho = 1.6047, rcut = 9.5. Scale is 1/r0, rho is multiplied by r0,
# and rcut is divided by r0.
_BC_SCALE = ".34518467380048325854"
_BC_RHO = "4.64881590"
_BC_RCUT = "3.27925440110459095616"
_PREAMBLE = Path(__file__).with_name("optim_ffd_preamble.txt")
_WALES_27058 = "https://optbench.org/codes/wales-27058.tar.bz2"
_WALES_27058_SHA = "144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"


def hef_binary() -> Path:
    env = os.environ.get("OPTIM_27058_BIN")
    if env:
        return Path(env)
    return Path("/tmp/optim27058/build/OPTIM")


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


def _hef_odata(folder: Path, coords: np.ndarray, frozen: np.ndarray, box: np.ndarray) -> None:
    scale = float(_BC_SCALE)
    scaled = coords * scale
    lengths = box * scale
    box_text = "%f %f %f" % (float(lengths[0]), float(lengths[1]), float(lengths[2]))
    preamble = _PREAMBLE.read_text(encoding="utf-8")
    lines = [preamble.rstrip("\n"), f"PARAMS {_BC_RHO} {box_text} {_BC_RCUT}"]
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
    """One native OPTIM search. The score is the paper's energy-plus-gradient count."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tempfile

    name = str(job["name"])
    calls = 10000
    success = False
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-hef-") as raw:
            folder = Path(raw)
            coords, frozen, box = _published_con(Path(job["con"]))
            _hef_odata(folder, coords, frozen, box)
            _displacement(Path(job["con"]), Path(job["reactant"]), folder / "vector.dump")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            stdout_path = folder / "OPTIM.log"
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout:
                    proc = subprocess.run(
                        [str(job["binary"])],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=600,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                proc = None
                error = "OPTIM exceeded 600s"
            log_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
            parsed, converged = _paper_calls(log_text)
            if parsed is not None:
                calls = parsed
            success = converged and parsed is not None and calls < int(job["cap"])
            if proc is not None and proc.returncode != 0 and not success:
                error = error or f"OPTIM exit {proc.returncode}"
            elif not success and not error:
                error = "OPTIM did not converge"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        success = False
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": "optim",
        "comparable": False,
        "success": success,
        "failure": None if success else "divergence",
        "reasons": [] if success else ["divergence"],
        "hessian_index": None,
        "expected_index": 1,
        "converged": success,
    }
    if error and not success:
        record["error"] = error
    return record


def _run_hef(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    binary = hef_binary()
    if not binary.is_file():
        raise RuntimeError(
            "OPTIM 27058 is not built. "
            f"The pin is the sha256 {_WALES_27058_SHA} of {_WALES_27058} "
            "(20942854 bytes). Set OPTIM_27058_BIN to that OPTIM binary."
        )
    if not _PREAMBLE.is_file():
        raise RuntimeError(f"{_PREAMBLE} is missing")
    from .geometries import _extract, _stored, load_starts

    scratch = output.parent / "work" / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    folder = _extract(_stored(root, problem["input"]["archive_path"]), scratch / "geometries")
    island = folder / "pt-island-con"
    reactant = island / "reactant.con"
    if not reactant.is_file():
        raise FileNotFoundError(reactant)
    cap = int((problem.get("convergence") or {})["max_force_calls"])
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
                "cap": cap,
            }
        )
    records = run_jobs(jobs, _execute_hef, workers)
    settings = point_settings(problem)
    settings["cap"] = cap
    comment = (
        "# counter optim. Wales OPTIM 27058 energy+gradient calls. "
        "The mean is the 2014 round of converged starts.\n"
        "# odata is the archived preamble. Coordinates, rho, and rcut use the run.sh bc -l scaling. "
        "The mode is the genmode displacement from reactant.con.\n"
        "# A success is the line **** CONVERGED **** under the cap. Failures stay out of the mean."
    )
    parameters = {
        "method": "hef-ffd",
        "keyword": "BFGSTS",
        "usediag": 3,
        "scale": _BC_SCALE,
        "rho": _BC_RHO,
        "rcut": _BC_RCUT,
        "counter": "optim",
        "binary": "wales-27058",
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
