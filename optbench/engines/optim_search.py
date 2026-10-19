"""OPTIM L-BFGS through the ASE socket, and the r27058 climbing string.

The minimization binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. That path does not score OPTIM's own counter.

The LJ38 climbing-string row is the other path. It runs OPTIM 27058 with
the 2014 preamble and scores the last ``energy+gradient calls`` field.
Success is the line ``Converged to TS``. Failures stay out of the mean.

``optim/unhooked`` and every other template raise before a file is written.
Noise, NEB, and PATHSAMPLE are not this runner.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tarfile
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


def _xyz_coordinate_lines(text: str) -> list[str]:
    """Columns 2-4 of an XYZ body, with Fortran exponents.

    This is ``awk '(NR>2){print $2,$3,$4}' | sed 's/[eE]/D/g'`` on a file
    whose first line is the atom count.
    """
    lines = text.splitlines()
    count = int(lines[0].split()[0])
    body = lines[2 : 2 + count]
    coords = []
    for line in body:
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError("xyz coordinate is missing x y z")
        coords.append(" ".join(token.replace("e", "D").replace("E", "D") for token in tokens[1:4]))
    if len(coords) != count:
        raise ValueError(f"xyz announced {count} atoms and held {len(coords)}")
    return coords


def _pair_texts(archive: Path, count: int) -> list[tuple[str, str, str]]:
    texts: dict[str, str] = {}
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            texts[Path(member.name).name] = extracted.read().decode("utf-8", errors="replace")
    pairs = []
    for index in range(count):
        start_name = f"start_{index:02d}.xyz"
        end_name = f"end_{index:02d}.xyz"
        if start_name not in texts or end_name not in texts:
            raise FileNotFoundError(f"{archive} has no {start_name}/{end_name}")
        pairs.append((start_name, texts[start_name], texts[end_name]))
    return pairs


def _last_call_count(log: str) -> int | None:
    """Last ``energy+gradient calls`` integer. awk field 6 of that line."""
    found = None
    for line in log.splitlines():
        if "energy+gradient calls" not in line:
            continue
        parts = line.split()
        if len(parts) >= 6:
            found = parts[5]
    if found is None or found == "":
        return None
    return int(float(found))


def _paper_integers(rows: list[tuple[int, bool]]) -> dict[str, int]:
    """getdata.py: half-even mean, median, min, and max of the successes."""
    successes = [calls for calls, ok in rows if ok]
    if not successes:
        raise RuntimeError("climbing string produced no converged transition states")
    values = np.asarray(successes, dtype=np.float64)
    return {
        "force_calls": int(round(float(np.mean(values)))),
        "force_calls_min": int(round(float(np.min(values)))),
        "force_calls_max": int(round(float(np.max(values)))),
        "force_calls_median": int(round(float(np.median(values)))),
        "nfailed": sum(1 for _calls, ok in rows if not ok),
        "n": len(rows),
    }


def _require_27058(binary: Path) -> None:
    try:
        proc = subprocess.run(
            [str(binary)],
            input="",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{binary} did not print a version within 60s") from exc
    banner = f"{proc.stdout}\n{proc.stderr}"
    if "version 27058" not in banner:
        raise RuntimeError(f"{binary} is not OPTIM version 27058")


def _execute_cl_string(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tempfile

    name = str(job["name"])
    calls = -1
    success = False
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-cl-") as raw:
            folder = Path(raw)
            (folder / "odata").write_text(str(job["odata"]), encoding="utf-8")
            (folder / "finish").write_text(str(job["finish"]), encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            log_path = folder / "OPTIM.log"
            stderr_path = folder / "stderr.txt"
            try:
                with log_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.run(
                        [str(job["binary"])],
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
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            parsed = _last_call_count(log)
            if parsed is not None:
                calls = parsed
            success = "Converged to TS" in log
            if proc is not None and proc.returncode != 0 and not success and parsed is None:
                detail = _tail(stderr_path) or _tail(log_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if success:
        failure = None
        reasons: list[str] = []
    elif error and calls < 0:
        failure = "crash"
        reasons = ["crash"]
    else:
        failure = "divergence"
        reasons = ["divergence"]
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": "optim",
        "comparable": False,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "error": error,
        "converged": success,
    }


def _cl_string_text(summary: dict[str, int], template: dict) -> str:
    lines = [
        "# counter optim. wales-27058 energy+gradient calls.",
        "# Success is the line Converged to TS. Failures stay out of the mean.",
        "# The harness cap is not this failure rule.",
        f"force_calls {summary['force_calls']}",
        f"force_calls_min {summary['force_calls_min']}",
        f"force_calls_max {summary['force_calls_max']}",
        f"force_calls_median {summary['force_calls_median']}",
        f"nfailed {summary['nfailed']}",
        f"starts {summary['n']}",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"problem {template['problem']}",
        f"tuning {template['tuning']}",
        "counter optim",
        "ranked false",
    ]
    return "\n".join(lines) + "\n"


def _run_cl_string(root: Path, problem: dict, template: dict, section: dict, *, output: Path, workers: int) -> dict:
    from optbench.campaign import geometry_archive, write_starts
    from optbench.engines.common import band_settings

    binary = binary_path()
    if not binary.is_file():
        raise RuntimeError(f"{binary} is not present")
    _require_27058(binary)
    settings = band_settings(problem)
    if settings["geometry"] != "xyz-pairs":
        raise ValueError(f"{problem['id']} geometry is {settings['geometry']}")
    preamble_name = str(section.get("preamble") or "")
    if not preamble_name:
        raise ValueError(f"{template['id']} has no optim.preamble")
    preamble_path = Path(preamble_name)
    if not preamble_path.is_file():
        preamble_path = root / preamble_name
    preamble = preamble_path.read_text(encoding="utf-8")
    if not preamble.endswith("\n"):
        preamble += "\n"
    symbol = str(section.get("symbol") or "AX")
    jobs = []
    for name, start_text, end_text in _pair_texts(geometry_archive(root, problem), settings["pairs"]):
        start_lines = _xyz_coordinate_lines(start_text)
        end_lines = _xyz_coordinate_lines(end_text)
        odata = preamble + "points\n" + "\n".join(f"{symbol} {row}" for row in start_lines) + "\n"
        finish = "\n".join(end_lines) + "\n"
        jobs.append({"name": name, "odata": odata, "finish": finish, "binary": str(binary)})
    records = run_jobs(jobs, _execute_cl_string, workers)
    stamped = []
    for record in records:
        row = dict(record)
        row["problem"] = problem["id"]
        row["template"] = template["id"]
        row["code"] = template["code"]
        row["code_version"] = template["code_version"]
        row["commit"] = template["commit"]
        row["algorithm"] = template["algorithm"]
        row["tuning"] = template["tuning"]
        stamped.append(row)
    summary = _paper_integers([(int(row["force_calls"]), bool(row["success"])) for row in stamped])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_cl_string_text(summary, template), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), stamped)
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": summary["force_calls"],
        "force_calls_median": summary["force_calls_median"],
        "force_calls_min": summary["force_calls_min"],
        "force_calls_max": summary["force_calls_max"],
        "nfailed": summary["nfailed"],
        "starts": summary["n"],
        "successes": summary["n"] - summary["nfailed"],
        "output": str(output),
    }


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
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    section = _section(template)
    if section["method"] == "cl-string":
        return _run_cl_string(root, problem, template, section, output=output, workers=workers)
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
