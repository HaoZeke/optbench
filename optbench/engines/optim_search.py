"""OPTIM L-BFGS through the ASE socket, plus the 2014 LJ38 central-difference search.

The minimization binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not that score.

``optim/lj38-hef-cd`` is the 2014 tsopt row. It runs wales-27058 on the
archived odata and coords-xyz, and the score is OPTIM's energy+gradient
counter. That tarball is not the snapshot pin.

``optim/unhooked`` and every other template that is not minimization
L-BFGS raise before a file is written. Noise, NEB, and PATHSAMPLE are
not this runner.
"""

from __future__ import annotations

import hashlib
import json
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
WALES_27058 = "https://optbench.org/codes/wales-27058.tar.bz2"
WALES_27058_SHA256 = "144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"
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


def binary_27058() -> Path:
    """OPTIM from wales-27058.tar.bz2, not the later svn snapshot."""
    env = os.environ.get("OPTIM_27058")
    if env:
        return Path(env)
    return Path("/tmp/src/wales/OPTIM/bin/gfortran/OPTIM")


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


def odata_text(preamble: str, points: list[tuple[str, str, str]]) -> str:
    """Archived ``run_optim.sh``: preamble, ``points``, then ``AX`` plus three spaces."""
    body = preamble if preamble.endswith("\n") else preamble + "\n"
    lines = [body.rstrip("\n"), "points"]
    for x_coord, y_coord, z_coord in points:
        lines.append(f"AX   {x_coord} {y_coord} {z_coord}")
    return "\n".join(lines) + "\n"


def energy_gradient_calls(log: str) -> int | None:
    """Field 6 of the last ``energy+gradient calls`` line. That is OPTIM's FCALL."""
    last = None
    for line in log.splitlines():
        if "energy+gradient calls" not in line:
            continue
        parts = line.split()
        if len(parts) >= 6:
            last = parts[5]
    if last is None:
        return None
    try:
        return int(float(last))
    except ValueError:
        return None


def paper_counts(calls: list[int], successes: list[bool]) -> dict:
    """``getdata.py``: unsuccessful starts stay out of the mean and count as nfailed.

    ``round`` is the same call the 2014 script makes on the numpy mean.
    """
    if len(calls) != len(successes):
        raise ValueError("calls and successes differ in length")
    kept = [int(call) for call, ok in zip(calls, successes, strict=True) if ok]
    failed = sum(1 for ok in successes if not ok)
    if not kept:
        raise ValueError("no successful starts")
    values = np.asarray(kept, dtype=float)
    mean = np.mean(values)
    median = np.median(values)
    return {
        "force_calls": int(round(mean)),
        "force_calls_median": int(round(median)),
        "force_calls_min": int(round(np.min(values))),
        "force_calls_max": int(round(np.max(values))),
        "nfailed": int(failed),
        "unrounded_mean": float(mean),
        "unrounded_median": float(median),
        "n": len(successes),
        "successes": len(kept),
    }


def _stored_archive(root: Path, archive_path: str) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") != archive_path:
            continue
        found = root / row["stored"]
        digest = hashlib.sha256(found.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"{archive_path} sha256 is {digest}, the manifest says {row['sha256']}")
        return found
    raise FileNotFoundError(archive_path)


def _xyz_starts(archive: Path) -> list[tuple[str, list[tuple[str, str, str]]]]:
    rows: list[tuple[str, list[tuple[str, str, str]]]] = []
    with tarfile.open(archive) as packed:
        members = [item for item in packed.getmembers() if item.name.endswith(".xyz")]
        members.sort(key=lambda item: item.name)
        for member in members:
            extracted = packed.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)
            lines = extracted.read().decode("utf-8").splitlines()
            points: list[tuple[str, str, str]] = []
            for line in lines[2:]:
                parts = line.split()
                if len(parts) < 4:
                    continue
                points.append((parts[1], parts[2], parts[3]))
            rows.append((Path(member.name).stem, points))
    return rows


def _hef_one(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tempfile

    name = str(job["name"])
    calls = None
    success = False
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix=f"optim-cd-{name}-") as raw:
            folder = Path(raw)
            (folder / "odata").write_text(odata_text(job["preamble"], job["points"]), encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            try:
                proc = subprocess.run(
                    [str(job["binary"])],
                    cwd=folder,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                log = proc.stdout + "\n" + proc.stderr
            except subprocess.TimeoutExpired:
                log = ""
                error = "OPTIM exceeded 600s"
            calls = energy_gradient_calls(log)
            success = "**** CONVERGED ****" in log
            if calls is None and not error:
                error = "OPTIM log has no energy+gradient calls"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": -1 if calls is None else int(calls),
        "success": bool(success),
        "failure": None if success else "divergence",
        "counter": "optim",
        "comparable": False,
        "template": job["template"],
        "problem": job["problem"],
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
    }
    if error:
        record["error"] = error
    return record


def _hef_dat(counts: dict, template: dict) -> str:
    lines = [
        "# counter optim. OPTIM energy+gradient calls. The ASE socket is not this score.",
        "# wales-27058. USEDIAG is unset, so NSECDIAG stays 1 (central differences on energies).",
        f"# unrounded_mean {counts['unrounded_mean']:.10g}",
        f"# unrounded_median {counts['unrounded_median']:.10g}",
        f"force_calls {counts['force_calls']}",
        f"force_calls_median {counts['force_calls_median']}",
        f"force_calls_min {counts['force_calls_min']}",
        f"force_calls_max {counts['force_calls_max']}",
        f"nfailed {counts['nfailed']}",
        f"n {counts['n']}",
        f"successes {counts['successes']}",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"problem {template.get('problem', '')}",
        f"tuning {template['tuning']}",
        "counter optim",
    ]
    return "\n".join(lines) + "\n"


def _run_hef_cd(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int) -> dict:
    from optbench.campaign import load_template, resolve_problem

    root = Path(root)
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    if template.get("code") != CODE:
        raise ValueError(f"{template_name} code is not {CODE}")
    if template.get("commit") != WALES_27058_SHA256:
        raise ValueError(f"{template_name} commit is not the wales-27058 sha256 {WALES_27058_SHA256}")
    section = dict(template.get("optim") or {})
    if str(section.get("method", "")) != "hef-cd":
        raise ValueError(f"{template_name} method is not hef-cd")
    binary = binary_27058()
    if not binary.is_file():
        raise RuntimeError(
            "OPTIM 27058 is not built. "
            f"The pin is the sha256 {WALES_27058_SHA256} of {WALES_27058}. "
            f"{binary} is not present. Set OPTIM_27058 to the gfortran binary."
        )
    preamble_path = root / str(section["preamble"])
    preamble = preamble_path.read_text(encoding="utf-8")
    archive = _stored_archive(root, str(section["geometry"]))
    starts = _xyz_starts(archive)
    expected = int(section.get("starts", (problem.get("sample") or {}).get("starts", 0)))
    if len(starts) != expected:
        raise RuntimeError(f"{archive} has {len(starts)} xyz starts, the template states {expected}")
    jobs = [
        {
            "name": name,
            "points": points,
            "preamble": preamble,
            "binary": str(binary),
            "template": template["id"],
            "problem": problem["id"],
            "code": template["code"],
            "code_version": str(template["code_version"]),
            "commit": template["commit"],
            "tuning": template["tuning"],
        }
        for name, points in starts
    ]
    records = run_jobs(jobs, _hef_one, workers)
    counts = paper_counts(
        [int(row["force_calls"]) for row in records],
        [bool(row["success"]) for row in records],
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_hef_dat(counts, template), encoding="utf-8")
    starts_path = output.with_suffix(".starts.jsonl")
    starts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    return {
        "force_calls": counts["force_calls"],
        "nfailed": counts["nfailed"],
        "starts": counts["n"],
        "template": template["id"],
        "problem": problem["id"],
        "unrounded_mean": counts["unrounded_mean"],
        "unrounded_median": counts["unrounded_median"],
    }


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    from optbench.campaign import load_template

    preview = load_template(Path(root), template_name)
    if str((preview.get("optim") or {}).get("method", "")) == "hef-cd":
        return _run_hef_cd(root, problem_name, template_name, output=output, workers=workers)
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
