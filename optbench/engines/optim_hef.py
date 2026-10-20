"""Hybrid eigenvector following, central-difference curvature, through the ASE socket.

The binary is OPTIM from the current public Wales tarball
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. ``OPTIM/source/SVNREV``
is ``df8da599267534fbe43a250e578f7396a7946311``. The archive row is
hybrid eigenvector following whose Rayleigh-Ritz step uses a
second-order central-difference expansion. That is ``BFGSTS`` with
``USEDIAG 2`` (gradients, the source default). ``BFGSSTEPS`` and
``STEPS`` default to 1, and both are the problem force-call cap.
``NEVS`` and ``NEVL``, the eigenvector iteration limits, are that cap
too. ``NBFGSMAX1`` and ``NBFGSMAX2`` stay at the source defaults 10 and
100, and ``CEIG`` stays ``1.0D-10``. ``BFGSCONV`` is the problem
Euclidean force norm over ``sqrt(3 N_free)``. With ``FREEZE``, OPTIM's
RMS divides by ``NOPT - 3*NFREEZE``, so ``N`` is the free atoms.
``NOHESS`` is set because the
ASE potential returns energy and gradient only. The initial mode is
the displacement from the reactant, and fixed atoms are the con file's
frozen list. Each ``COMPUTE`` is one harness evaluation of
``morse/smooth/linear``, the same entry OPTIM counts as an
energy+gradient call.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from optbench.engines.common import judge_saddle, run_jobs, write_campaign
from optbench.engines.geometries import load_starts, point_settings
from optbench.engines.optim_cd import read_con, vector_dump
from optbench.engines.optim_search import _fortran_float, _read_points, _tail, _write_xyz

TARBALL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
TARBALL_SHA256 = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
SVNREV = "df8da599267534fbe43a250e578f7396a7946311"
_BUILT = Path("/tmp/optim-latest/build/OPTIM")
_RELAY = Path(__file__).with_name("fmlip_relay.py")


def binary_path() -> Path:
    env = os.environ.get("OPTIM_HEF_BIN")
    if env:
        return Path(env)
    return _BUILT


def rms_stop(force_norm: float, n_free: int) -> float:
    """RMS that makes OPTIM stop at the free-atom Euclidean norm.

    ``potential.F90`` zeros frozen gradients and sets
    ``RMSDOF = NOPT - 3*NFREEZE``. ``RMS`` is ``||g_free|| / sqrt(RMSDOF)``.
    ``BFGSCONV`` has to use that denominator. Dividing by every atom
    asks for ``||g_free|| < force_norm * sqrt(N_free / N)``.
    """
    count = int(n_free)
    if count < 1:
        raise ValueError("BFGSCONV needs at least one free atom")
    return float(force_norm) / math.sqrt(3 * count)


def odata_text(python: str, hook: Path, periodic: bool, gmax: float, steps: int, frozen: np.ndarray) -> str:
    """Paper curvature, problem stop, and the iteration caps.

    ``BFGSTS`` arguments are ``NEVS``, ``NBFGSMAX1``, ``NBFGSMAX2``,
    ``CEIG``, and ``NEVL``. ``NEVS`` and ``NEVL`` are the problem cap.
    ``10`` and ``100`` are the source defaults of the tangent-space step
    limits, and ``1.0D-10`` is the source default of ``CEIG``. They are
    written so the eigenvector limits can be set. The 2014 values
    ``30 11 30 0.063`` are not copied.
    """
    pbc = "PBC" if periodic else "NOPBC"
    cap = int(steps)
    lines = [
        f"ASE geom.xyz {python} {hook} dummy dummy cpu FIXCELL {pbc} EV ANG",
        f"BFGSTS {cap} 10 100 1.0D-10 {cap}",
        "NOHESS",
        f"BFGSCONV {_fortran_float(gmax)}",
        f"BFGSSTEPS {cap}",
        f"STEPS {cap}",
        "USEDIAG 2",
        "READVEC",
    ]
    indices = [str(int(index) + 1) for index in np.flatnonzero(np.asarray(frozen, dtype=bool))]
    for start in range(0, len(indices), 8):
        lines.append("FREEZE " + " ".join(indices[start : start + 8]))
    return "\n".join(lines) + "\n"


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from optbench.pes import evaluate

    positions = np.asarray(job["positions"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    frozen = ~free
    kind = str(job["kind"])
    cap = int(job["cap"])
    force_norm = float(job["force_norm"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    natoms = int(positions.shape[0])
    gmax = rms_stop(force_norm, int(np.count_nonzero(free)))
    binary = Path(job["binary"])
    point = None
    energy = None
    gradient = None
    calls = 0
    capped = False
    error = ""
    log = ""
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="optim-hef-") as raw:
            folder = Path(raw)
            hook = folder / "hook"
            relay = hook / "pysocket" / "fmlip_relay.py"
            relay.parent.mkdir(parents=True)
            shutil.copyfile(_RELAY, relay)
            _write_xyz(folder / "geom.xyz", positions, box)
            (folder / "odata").write_text(
                odata_text(sys.executable, hook, box is not None, gmax, cap, frozen),
                encoding="utf-8",
            )
            (folder / "vector.dump").write_text(
                vector_dump(positions, np.asarray(job["reactant"], dtype=np.float64)),
                encoding="utf-8",
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
                        timeout=7200,
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                proc = None
                error = "OPTIM exceeded 7200s"
            count_path = folder / "calls.txt"
            if count_path.is_file():
                calls = int(count_path.read_text(encoding="utf-8").strip() or "0")
            if stdout_path.is_file():
                log = stdout_path.read_text(encoding="utf-8", errors="replace")
            keep = os.environ.get("OPTBENCH_KEEP_LOG")
            if keep and log:
                Path(keep).write_text(log, encoding="utf-8")
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
        expected_index=1,
        rotations=rotations,
    )
    if error:
        record["error"] = error
    record["optim_method"] = "hef"
    record["curvature"] = "usediag-2"
    gradient_calls, rr_steps = log_counters(log)
    record["optim_energy_gradient_calls"] = gradient_calls
    record["rayleigh_ritz_steps"] = rr_steps
    if os.environ.get("OPTBENCH_PROGRESS"):
        print(
            f"{record['start']} calls={record['force_calls']} success={record['success']} failure={record.get('failure')}",
            flush=True,
        )
    return record


_GRADIENT_CALLS = re.compile(r"energy\+gradient calls=\s*(\d+)")
_RR_STEP = re.compile(r"xmylbfgs> Eigenvalue, RMS, % changes?=.* after\s+(\d+)\s+steps")


def log_counters(log: str) -> tuple[int | None, int]:
    """OPTIM's energy+gradient tally, and Rayleigh-Ritz steps.

    Each ``xmylbfgs`` progress line is one completed Rayleigh-Ritz
    iteration. Those iterations are not the score. ``USEDIAG 2`` spends
    two gradient evaluations inside one iteration, and both are
    energy+gradient calls. The harness counts the same evaluations:
    one socket ``COMPUTE`` is one ``pes.evaluate``.
    """
    calls = None
    for line in log.splitlines():
        match = _GRADIENT_CALLS.search(line)
        if match:
            calls = int(match.group(1))
    return calls, len(_RR_STEP.findall(log))


class _Calls:
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


def _reactant(work: Path) -> np.ndarray:
    path = work / "geometries" / "pt-island-con" / "reactant.con"
    coords, _frozen, _box = read_con(path)
    return coords


def run_hef(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    binary = binary_path()
    if not binary.is_file() or not _RELAY.is_file():
        missing = binary if not binary.is_file() else _RELAY
        raise RuntimeError(
            "OPTIM hybrid eigenvector following has no built binary. "
            f"The current public tarball is {TARBALL}, sha256 {TARBALL_SHA256}. "
            f"OPTIM/source/SVNREV is {SVNREV}. {missing} is not present."
        )
    settings = point_settings(problem)
    if settings["cap"] is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    reactant = _reactant(scratch)
    jobs = []
    for start in starts:
        jobs.append(
            {
                "name": start["name"],
                "positions": start["positions"],
                "free": [not bool(flag) for flag in start["fixed"]],
                "box": start["box"],
                "kind": "morse_smooth_pt",
                "cap": int(settings["cap"]),
                "force_norm": float(settings["force_norm"]),
                "binary": str(binary),
                "reactant": reactant.tolist(),
            }
        )
    if int(workers) <= 1:
        records = [_execute(job) for job in jobs]
    else:
        records = run_jobs(jobs, _execute, workers)
    published = dict(template)
    published["code"] = "OPTIM"
    comment = (
        "# counter harness. Each ASE COMPUTE is one pes.evaluate of morse/smooth/linear, "
        "periodic in x and y. That is OPTIM's energy+gradient call, not a Rayleigh-Ritz step.\n"
        f"# OPTIM {SVNREV} from {TARBALL} sha256 {TARBALL_SHA256}. "
        "BFGSTS with USEDIAG 2. NEVS, NEVL, BFGSSTEPS, and STEPS are the problem cap. "
        "NBFGSMAX1 10, NBFGSMAX2 100, and CEIG 1.0D-10 are the source defaults. "
        "BFGSCONV is the Euclidean force norm over sqrt(3 N_free). "
        "NOHESS because ASE returns no Hessian. READVEC is initial minus reactant. "
        "A success is under the cap, at the force stop, and Hessian index 1."
    )
    parameters = {
        "method": "hef",
        "keyword": "BFGSTS",
        "usediag": 2,
        "ceig": "1.0D-10",
        "nohess": True,
        "counter": "harness",
        "socket": "ASE",
        "potential": "morse/smooth/linear",
        "svnrev": SVNREV,
        "tarball_sha256": TARBALL_SHA256,
    }
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
