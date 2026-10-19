"""OPTIM L-BFGS through the ASE socket.

The binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not the score.

``optim/unhooked`` and every template that is not minimization L-BFGS
or the archived six-image DNEB row raise before a file is written.
The six-image row runs wales-27058 and scores OPTIM's own
energy+gradient count. Noise and PATHSAMPLE are not this runner.
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
        "The six-image Pt-island DNEB row is optim/de-pt-6. "
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
_WALES_CODE = "optim-27058"


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


def _native_search(job: dict) -> dict:
    """One wales-27058 DNEB search. The score is OPTIM's own counter."""
    import tempfile
    import time

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    calls = None
    success = False
    error = ""
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="optim-de6-") as raw:
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
                        [str(binary_path())],
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
            calls, success = _optim_tally(log)
            if proc is not None and proc.returncode not in (0, None) and error == "":
                detail = _tail(stderr_path) or _tail(stdout_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
                success = False
            elif calls is None and error == "":
                detail = _tail(stdout_path) or _tail(stderr_path) or "no energy+gradient line"
                error = f"OPTIM produced no energy+gradient count: {detail}"
                success = False
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        success = False
    wall = time.perf_counter() - started
    if not success:
        recorded = int(job["cap"]) if calls is None else int(calls)
        failure = "crash" if error else "divergence"
    else:
        recorded = int(calls)
        failure = None
    return {
        "schema": "optbench.start/v2",
        "start": job["name"],
        "force_calls": recorded,
        "verify_calls": 0,
        "counter": "optim",
        "comparable": False,
        "success": bool(success and not error),
        "failure": failure,
        "reasons": [],
        "error": error,
        "converged": bool(success and not error),
        "optim_method": "dneb-native",
        "wall_s": float(wall),
        "problem": job.get("problem", ""),
        "template": job.get("template", ""),
        "code": job.get("code", ""),
        "code_version": job.get("code_version", ""),
        "commit": job.get("commit", ""),
        "algorithm": job.get("algorithm", ""),
        "tuning": job.get("tuning", ""),
    }


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


def _write_native_campaign(output: Path, records: list[dict], template: dict, problem: dict, cap: int) -> dict:
    import json
    import statistics

    from optbench.score import score_v2
    from optbench.v2 import collect_provenance, write_starts

    output.parent.mkdir(parents=True, exist_ok=True)
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    lines = [
        "# counter optim. The score is the last 'energy+gradient calls' field.",
        "# A start is a failure when the log has no 'Converged to TS' line.",
        "# Coordinates, PARAMS, and FREEZE follow archive run.sh. The potential is morse_bulk.",
        f"# binary wales-27058 sha256 {template['commit']}",
    ]
    if successes:
        mean = statistics.fmean(successes)
        lines.append(f"# unrounded_mean {mean:.10g}")
        lines.append(f"force_calls {int(round(mean))}")
        lines.append(f"force_calls_min {min(successes)}")
        lines.append(f"force_calls_max {max(successes)}")
        median = statistics.median(successes)
        lines.append(
            f"force_calls_median {int(median) if float(median).is_integer() else median}"
        )
    lines.extend(
        [
            f"successes {len(successes)}",
            f"nfailed {len(records) - len(successes)}",
            f"n {len(records)}",
            f"code {template['code']}",
            f"code_version {template['code_version']}",
            f"commit {template['commit']}",
            f"algorithm {template['algorithm']}",
            f"template {template['id']}",
            f"problem {problem['id']}",
            f"tuning {template['tuning']}",
            "counter optim",
            "images 6",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    scored = [
        {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
        for row in records
    ]
    summary = score_v2(scored, cap)
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "optim",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters={
                "method": "dneb-native",
                "preamble": str((template.get("optim") or {}).get("preamble", "")),
                "images": 6,
                "potential": "morse_bulk",
                "scale": "angstrom/r0",
                "counter": "energy+gradient",
            },
        ),
        "score": summary,
        "legacy": {
            "force_calls": int(round(statistics.fmean(successes))) if successes else None,
            "force_calls_min": min(successes) if successes else None,
            "force_calls_max": max(successes) if successes else None,
            "force_calls_median": statistics.median(successes) if successes else None,
            "nfailed": len(records) - len(successes),
            "n": len(records),
            "successes": len(successes),
        },
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": document["legacy"]["force_calls"],
        "nfailed": document["legacy"]["nfailed"],
        "starts": len(records),
        "successes": len(successes),
        "output": str(output),
    }


def _run_native_dneb(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    from .common import band_settings, cached_pairs

    section = template.get("optim") or {}
    settings = band_settings(problem)
    if settings["potential"] != "morse_pt":
        raise ValueError(f"{template['id']} native DNEB is the Pt Morse row")
    preamble_path = root / str(section.get("preamble", ""))
    if not preamble_path.is_file():
        raise RuntimeError(f"{template['id']} preamble is not at {preamble_path}")
    preamble = preamble_path.read_text(encoding="utf-8")
    if "NEWNEB 6 10 0.06" not in preamble:
        raise RuntimeError(f"{template['id']} preamble is not the six-image archive file")
    pairs = list(cached_pairs(root, problem, settings).values())
    pairs.sort(key=lambda pair: int(pair.name.rsplit("_", 1)[-1]))
    jobs = [_native_job(pair, template, preamble, int(settings["cap"])) for pair in pairs]
    records = run_jobs(jobs, _native_search, workers)
    return _write_native_campaign(output, records, template, problem, int(settings["cap"]))


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
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
    if section["method"] == "dneb-native":
        pin = registry_pin(root, _WALES_CODE)
        if template.get("code") != CODE:
            raise ValueError(f"{template_name} code is not {CODE}")
        if template.get("commit") != pin["commit"]:
            raise ValueError(f"{template_name} commit does not match the {_WALES_CODE} registry")
        return _run_native_dneb(root, problem, template, output=output, workers=workers)
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
