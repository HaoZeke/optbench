"""PATHSAMPLE driving OPTIM through the ASE socket.

PATHSAMPLE does not call the potential. Each child OPTIM does. The score
for one pair is the sum of those ``pes.evaluate`` calls. The connect
file is the 2014 OPTIM preamble. Two endpoint quenches use the archived
minimizer keywords, ``DUMPDATA``, and ``NOFRQS``, so the frequency
product is a dummy and the Hessian is not charged.

``CONNECTIONS 1 1`` is one connect attempt. The library default is 10.
The 2014 row is one OPTIM process, so the extra calls are the two
endpoint quenches.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .common import band_settings, cached_pairs, judge_saddle, registry_pin, require_pin, run_jobs, write_campaign
from .optim_search import (
    SHA256,
    _Calls,
    _ase_line,
    _bare_lines,
    _count_file,
    _freeze_lines,
    _geometry_file,
    _install_hook,
    _optim_env,
    _pair_scale,
    _params_line,
    _symbol_lines,
    _tail,
    _write_xyz,
    available as optim_available,
    binary_path,
    statistics_median,
)

ENGINE = "pathsample"
CODE = "optim"


def pathsample_path() -> Path:
    env = os.environ.get("PATHSAMPLE_BIN")
    if env:
        return Path(env)
    return Path("/tmp/src/wales-full/build-pathsample/PATHSAMPLE")


def available() -> str | None:
    optim_reason = optim_available()
    if optim_reason:
        return optim_reason
    binary = pathsample_path()
    if binary.is_file():
        return None
    return (
        "PATHSAMPLE is not present. It drives OPTIM, and the score is the "
        f"sum of those OPTIM socket calls. The pin is the sha256 {SHA256}. "
        f"{binary} is not present."
    )


def _section(template: dict) -> dict:
    section = dict(template.get("pathsample") or {})
    section["symbol"] = str(section.get("symbol", "AX"))
    section["cycles"] = int(section.get("cycles", 1))
    section["connections"] = int(section.get("connections", 1))
    section["attempts"] = int(section.get("attempts", 1))
    section["timeout"] = int(section.get("timeout", 7200))
    return section


def _pathdata(folder: Path, natoms: int, section: dict) -> None:
    symbol = section["symbol"]
    lines = [
        f"NATOMS {int(natoms)}",
        f"SYSTEM {symbol}",
        "PERMDIST",
        "MAKESTART",
        "DIJINITSTART",
        f"CONNECTIONS {int(section['connections'])} {int(section['attempts'])}",
        f"CYCLES {int(section['cycles'])}",
        "CPUS 1",
        "ETOL 1.0D-6",
        "GEOMDIFFTOL 0.05D0",
        "SEED 1",
        "DIRECTION AB",
        "TEMPERATURE 0.1D0",
        f"EXEC {binary_path()}",
    ]
    (folder / "pathdata").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _endpoint(folder: Path, name: str, positions: np.ndarray, job: dict, hook: Path, preamble: str) -> None:
    kind = str(job["kind"])
    box = job["box"]
    scale_l, _scale_e = _pair_scale(kind)
    periodic = box is not None
    lines = [_ase_line(sys.executable, hook, periodic), preamble.rstrip()]
    if kind == "morse_pt":
        lines.append(_params_line(box, scale_l))
        lines.append("bulk")
        lines.extend(_freeze_lines(job["free"]))
    lines.append("points")
    lines.extend(_symbol_lines(positions, str(job["symbol"])))
    (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute(job: dict) -> dict:
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
    import tempfile

    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="pathsample-") as raw:
            folder = Path(raw)
            hook = _install_hook(folder)
            _write_xyz(folder / "geom.xyz", start, reduced_box)
            # ASE reads the initial coordinates from start.<extension>, not from
            # the POINTS block. Job 1 is the reactant quench and job 2 is the product.
            (folder / "start.1").write_text(_bare_lines(start), encoding="utf-8")
            (folder / "start.2").write_text(_bare_lines(finish), encoding="utf-8")
            _endpoint(folder, "odata.start", start, {**job, "box": box, "free": free}, hook, job["endpoint"])
            _endpoint(folder, "odata.finish", finish, {**job, "box": box, "free": free}, hook, job["endpoint"])
            connect = [
                _ase_line(sys.executable, hook, box is not None),
                str(job["preamble"]).rstrip(),
            ]
            if kind == "morse_pt":
                if box is None:
                    raise RuntimeError("morse_pt PATHSAMPLE search has no cell")
                connect.append(_params_line(box, scale_l))
                connect.extend(_freeze_lines(free))
            connect.append("points")
            (folder / "odata.connect").write_text("\n".join(connect) + "\n", encoding="utf-8")
            _pathdata(folder, natoms, job)
            call_log = folder / "calls.log"
            call_log.write_text("0\n", encoding="utf-8")
            env = _optim_env(folder, kind, cap, box, scale_l, scale_e, call_log, folder / "traj.xyz")
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            proc = None
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.run(
                        [str(pathsample_path())],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=int(job["timeout"]),
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                error = "PATHSAMPLE exceeded the wall limit"
            calls = _count_file(call_log)
            capped = (folder / "capped.txt").is_file() or calls >= cap
            if error == "":
                point = _geometry_file(folder, natoms, scale_l, kind, box, scale_e, force_norm)
            log_tail = _tail(stderr_path) or _tail(stdout_path)
            if proc is not None and proc.returncode not in (0, None) and not capped and error == "":
                error = f"PATHSAMPLE exit {proc.returncode}: {log_tail or 'no message'}"
            elif point is None and calls == 0 and not capped and error == "":
                error = f"PATHSAMPLE produced no geometry: {log_tail or 'no message'}"
            elif point is None and error == "":
                error = f"PATHSAMPLE produced no geometry: {log_tail or 'no message'}"
            if "ERROR" in (stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else "") and point is None and error == "":
                error = f"PATHSAMPLE stopped: {log_tail or 'ERROR'}"
            if job.get("keep"):
                destination = Path(job["keep"])
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(folder, destination)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    if point is not None and error == "":
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
    return record


def _job(pair, settings: dict, template: dict, preamble: str, endpoint: str) -> dict:
    section = _section(template)
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64),
        "product": np.asarray(pair.product, dtype=np.float64),
        "free": np.asarray(pair.free, dtype=bool),
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64),
        "kind": settings["potential"],
        "cap": int(settings["cap"]),
        "force_norm": float(settings["force_norm"]),
        "preamble": preamble,
        "endpoint": endpoint,
        "symbol": section["symbol"],
        "cycles": section["cycles"],
        "connections": section["connections"],
        "attempts": section["attempts"],
        "timeout": section["timeout"],
    }


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    if problem.get("suite") != "double-ended-saddle-search":
        raise RuntimeError(f"{template_name} is a double-ended PATHSAMPLE template")
    section = _section(template)
    settings = band_settings(problem)
    preamble_path = root / str(section["preamble"])
    endpoint_path = root / str(section["endpoint"])
    if not preamble_path.is_file() or not endpoint_path.is_file():
        raise RuntimeError(f"{template['id']} is missing {preamble_path.name} or {endpoint_path.name}")
    preamble = preamble_path.read_text(encoding="utf-8")
    endpoint = endpoint_path.read_text(encoding="utf-8")
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_job(pair, settings, template, preamble, endpoint) for pair in pairs]
    records = run_jobs(jobs, _execute, workers)
    walls = [float(row["wall_s"]) for row in records]
    comment = (
        "# counter harness. PATHSAMPLE launches OPTIM. Each ASE COMPUTE on every child is one pes.evaluate.\n"
        f"# odata.connect is {preamble_path.name}. Endpoint quenches are {endpoint_path.name} plus DUMPDATA and NOFRQS.\n"
        f"# CONNECTIONS {section['connections']} {section['attempts']}. "
        f"Archive OPTIM force_calls {section.get('archive_force_calls', '')} is one OPTIM process on wales-27058.\n"
        f"# wall_s_median {statistics_median(walls)}"
    )
    parameters = {
        "driver": "PATHSAMPLE",
        "preamble": str(section["preamble"]),
        "endpoint": str(section["endpoint"]),
        "connections": section["connections"],
        "attempts": section["attempts"],
        "cycles": section["cycles"],
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
