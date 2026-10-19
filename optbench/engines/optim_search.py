"""OPTIM minimization through the ASE socket, and OPTIM 27058 hybrid eigenvector-following.

The minimization binary is the OPTIM target from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. Each ``COMPUTE`` on the relay
is one ``pes.evaluate``. OPTIM's own counter is not the score for that socket.

``optim/unhooked`` and every template that is not minimization L-BFGS
raise before a file is written. Noise, NEB, and PATHSAMPLE are not this runner.
The LJ38 hybrid eigenvector-following row is the current Wales svn tarball,
unmodified, with stock ``BFGSTS`` defaults. ``BFGSCONV`` is the archive
Euclidean stop and ``STEPS`` is the problem cap. Starts are ``coords-xyz``.
The score is the last ``energy+gradient calls`` line. The 2014 ``wales-27058``
preamble path remains callable and is not the ranked row.
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


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    template = load_template(root, template_name)
    method = str((template.get("optim") or {}).get("method") or "")
    if method == "hef-ffd":
        return _run_hef_ffd(
            root,
            resolve_problem(root, problem_name),
            template,
            output=output,
            workers=workers,
        )
    if method == "hef-stock":
        return _run_hef_stock(
            root,
            resolve_problem(root, problem_name),
            template,
            output=output,
            workers=workers,
        )
    reason = available()
    if reason:
        raise RuntimeError(reason)
    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
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


_HEF_TARBALL = "https://optbench.org/codes/wales-27058.tar.bz2"
_HEF_SHA256 = "144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"


def hef_binary() -> Path | None:
    """OPTIM 27058. ``OPTIM_BIN`` wins, then the gfortran build of that tarball."""
    env = os.environ.get("OPTIM_BIN")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/tmp/src/wales/OPTIM/bin/gfortran/OPTIM"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _stored_archive(root: Path, archive_path: str) -> Path:
    import json

    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(archive_path)


def odata_for_xyz(preamble: str, xyz_text: str) -> str:
    """The 2014 ``run_optim.sh`` file: preamble, then ``points``, then ``AX`` rows.

    ``awk '{print "AX  ", $2,$3,$4}'`` inserts the output separator, so the
    symbol is followed by three spaces.
    """
    body = preamble if preamble.endswith("\n") else preamble + "\n"
    lines = [body + "points"]
    for raw in xyz_text.splitlines()[2:]:
        parts = raw.split()
        if len(parts) < 4:
            continue
        lines.append(f"AX   {parts[1]} {parts[2]} {parts[3]}")
    return "\n".join(lines) + "\n"


def parse_optim_log(text: str) -> dict:
    """Last ``energy+gradient calls`` line and the CONVERGED marker.

    ``getdata.py`` keeps a start out of the mean when that marker is absent.
    """
    calls = None
    for line in text.splitlines():
        if "energy+gradient calls=" not in line:
            continue
        token = line.split("energy+gradient calls=", 1)[1].split()[0]
        calls = int(token)
    return {"force_calls": calls, "converged": "**** CONVERGED ****" in text}


def _execute_hef(job: dict) -> dict:
    import tempfile

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name = str(job["name"])
    cap = int(job["cap"])
    calls = None
    converged = False
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-hef-") as raw:
            folder = Path(raw)
            (folder / "odata").write_text(job["odata"], encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            stdout_path = folder / "OPTIM.log"
            stderr_path = folder / "OPTIM.err"
            try:
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.run(
                        [str(job["binary"])],
                        cwd=folder,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=600,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                proc = None
                error = "OPTIM exceeded 600s"
            parsed = parse_optim_log(stdout_path.read_text(encoding="utf-8", errors="replace"))
            calls = parsed["force_calls"]
            converged = bool(parsed["converged"])
            if proc is not None and proc.returncode != 0 and not converged:
                detail = _tail(stderr_path) or _tail(stdout_path) or f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    success = bool(converged) and calls is not None and calls < cap and not error
    failure = None
    reasons: list[str] = []
    if not success:
        if error or calls is None:
            failure = "crash"
            reasons = ["crash"]
        elif calls >= cap:
            failure = "cap"
            reasons = ["cap"]
        else:
            failure = "divergence"
            reasons = ["divergence"]
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls or 0),
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "converged": bool(converged),
        "hessian_index": None,
        "expected_index": 1,
        "error": error,
    }


def _run_hef_ffd(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    """Native OPTIM 27058 on the archived preamble and ``coords-xyz`` starts."""
    import tarfile

    from .common import registry_pin, require_pin, run_jobs, write_campaign

    if template["engine"] != ENGINE:
        raise ValueError(f"{template['id']} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    require_pin(template, registry_pin(root, str(template["code"])))
    section = dict(template.get("optim") or {})
    binary = hef_binary()
    if binary is None:
        raise RuntimeError(
            f"{template['id']} needs OPTIM 27058. The pin is the sha256 {_HEF_SHA256} "
            f"of {_HEF_TARBALL} (20942854 bytes). Set OPTIM_BIN or build "
            "source/build.csh gfortran optim. The score is energy+gradient calls."
        )
    import tempfile

    with tempfile.TemporaryDirectory(prefix="optim-version-") as raw:
        banner = subprocess.run(
            [str(binary)],
            cwd=raw,
            input="",
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    version_text = f"{banner.stdout}\n{banner.stderr}"
    if "27058" not in version_text:
        raise RuntimeError(f"{binary} is not OPTIM 27058")
    preamble_path = root / str(section["preamble"])
    preamble = preamble_path.read_text(encoding="utf-8")
    archive = _stored_archive(root, str(section["coords"]))
    settings = point_settings(problem)
    if settings["cap"] is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    jobs = []
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            name = Path(member.name).name
            if not name.endswith(".xyz"):
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            xyz = extracted.read().decode("utf-8")
            jobs.append(
                {
                    "name": Path(name).stem,
                    "binary": str(binary),
                    "cap": int(settings["cap"]),
                    "odata": odata_for_xyz(preamble, xyz),
                }
            )
    jobs.sort(key=lambda job: job["name"])
    if len(jobs) != int((problem.get("sample") or {}).get("starts") or len(jobs)):
        raise RuntimeError(f"{template['id']} read {len(jobs)} xyz starts")
    records = run_jobs(jobs, _execute_hef, workers)
    comment = (
        "# OPTIM 27058 native Lennard-Jones, symbol AX, no cutoff. "
        "The odata preamble is the 2014 file: BFGSTS 40 14 16 0.33, "
        "UPDATES 82 10, BFGSSTEPS 100, BFGSCONV 9.36585D-05, NOHESS, "
        "MAXMAX 0.2, MAXBFGS 0.2 0.2, USEDIAG 3.\n"
        "# Starts are coords-xyz, 0000.xyz through 0199.xyz. "
        "force_calls is the last energy+gradient calls line. "
        "A start is a success only when the log contains CONVERGED and the count is under the cap."
    )
    parameters = {
        "method": "hef-ffd",
        "keyword": "BFGSTS",
        "bfgsts": "40 14 16 0.33",
        "updates": "82 10",
        "usediag": 3,
        "bfgsconv": "9.36585D-05",
        "counter": "energy+gradient",
        "potential": "LJDIFF",
        "starts": "coords-xyz",
        "code_file": "wales-27058.tar.bz2",
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


# Today's Wales export. The registry pin is an older file of the same URL.
_LATEST_URL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
_LATEST_SHA256 = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
_LATEST_SVNREV = "df8da599267534fbe43a250e578f7396a7946311"
_LATEST_VERSION = "20261019-df8da599"
# Archive comment: Euclidean norm 1e-3 is this RMS stop. Not a stock default.
_ARCHIVE_BFGSCONV = "9.36585D-05"


def latest_binary() -> Path | None:
    """Unmodified gfortran build of the current svn tarball."""
    env = os.environ.get("OPTIM_LATEST_BIN")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/tmp/wales-svn/wales/OPTIM/builds/gfortran/OPTIM"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_latest(binary: Path) -> None:
    svnrev = binary.parent.parent.parent / "source" / "SVNREV"
    if not svnrev.is_file():
        raise RuntimeError(f"{binary} has no source SVNREV beside the build")
    found = svnrev.read_text(encoding="utf-8").strip()
    if found != _LATEST_SVNREV:
        raise RuntimeError(f"{binary} source revision is {found}, not {_LATEST_SVNREV}")
    tarball = Path(os.environ.get("OPTIM_LATEST_TARBALL", "/tmp/svn.tar.bz2"))
    if not tarball.is_file():
        raise RuntimeError(
            f"latest OPTIM tarball is missing at {tarball}. "
            f"The release is {_LATEST_URL} sha256 {_LATEST_SHA256}."
        )
    digest = _sha256_file(tarball)
    if digest != _LATEST_SHA256:
        raise RuntimeError(f"{tarball} sha256 is {digest}, not {_LATEST_SHA256}")


def stock_odata(xyz_text: str, *, steps: int) -> str:
    """Stock ``BFGSTS`` plus the archive RMS stop and the problem step cap.

    ``bfgsts`` with no numbers keeps NEVS 500, NBFGSMAX1 10, NBFGSMAX2 100,
    and CEIG 1e-10. Default NSTEPS is 1, so ``steps`` is the problem cap.
    """
    preamble = f"bfgsts\nbfgsconv {_ARCHIVE_BFGSCONV}\nsteps {int(steps)}\n"
    return odata_for_xyz(preamble, xyz_text)


def _run_hef_stock(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    """Current OPTIM, stock hybrid eigenvector-following, native call count."""
    import tarfile

    from .common import run_jobs, write_campaign

    if template["engine"] != ENGINE:
        raise ValueError(f"{template['id']} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    if str(template.get("code_version") or "") != _LATEST_VERSION:
        raise RuntimeError(f"{template['id']} version is not {_LATEST_VERSION}")
    if str(template.get("commit") or "") != _LATEST_SHA256:
        raise RuntimeError(f"{template['id']} commit is not the current tarball")
    binary = latest_binary()
    if binary is None:
        raise RuntimeError(
            f"{template['id']} needs the current OPTIM binary. "
            f"The release is {_LATEST_URL} sha256 {_LATEST_SHA256}, "
            f"source {_LATEST_SVNREV}. Set OPTIM_LATEST_BIN."
        )
    _verify_latest(binary)
    section = dict(template.get("optim") or {})
    archive = _stored_archive(root, str(section["coords"]))
    settings = point_settings(problem)
    if settings["cap"] is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    jobs = []
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            name = Path(member.name).name
            if not name.endswith(".xyz"):
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            xyz = extracted.read().decode("utf-8")
            jobs.append(
                {
                    "name": Path(name).stem,
                    "binary": str(binary),
                    "cap": int(settings["cap"]),
                    "odata": stock_odata(xyz, steps=int(settings["cap"])),
                }
            )
    jobs.sort(key=lambda job: job["name"])
    expected = int((problem.get("sample") or {}).get("starts") or len(jobs))
    if len(jobs) != expected:
        raise RuntimeError(f"{template['id']} read {len(jobs)} xyz starts")
    records = run_jobs(jobs, _execute_hef, workers)
    comment = (
        f"# OPTIM source {_LATEST_SVNREV}, tarball {_LATEST_URL} "
        f"sha256 {_LATEST_SHA256}. Unmodified stock BFGSTS "
        "(NEVS 500, NBFGSMAX1 10, NBFGSMAX2 100, CEIG 1e-10). "
        f"BFGSCONV {_ARCHIVE_BFGSCONV} is the archive Euclidean 0.001 stop. "
        f"STEPS {int(settings['cap'])} is the problem cap. Symbol AX selects LJDIFF.\n"
        "# Starts are coords-xyz, 0000.xyz through 0199.xyz. "
        "force_calls is the last energy+gradient calls line. "
        "A start is a success only when the log contains CONVERGED and the count is under the cap."
    )
    parameters = {
        "method": "hef-stock",
        "keyword": "BFGSTS",
        "bfgsts": "defaults",
        "nevs": 500,
        "nbfgsmax1": 10,
        "nbfgsmax2": 100,
        "ceig": 1e-10,
        "bfgsconv": _ARCHIVE_BFGSCONV,
        "steps": int(settings["cap"]),
        "counter": "optim energy+gradient calls",
        "potential": "LJDIFF",
        "starts": "coords-xyz",
        "source_revision": _LATEST_SVNREV,
        "tarball_sha256": _LATEST_SHA256,
    }
    published = dict(template)
    published["code"] = "optim"
    return write_campaign(
        output,
        records,
        published,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )
