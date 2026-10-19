"""Latest OPTIM, stock hybrid eigenvector following, ranked on the harness counter.

The binary is the OPTIM target from the current Wales tarball
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. ``source/SVNREV`` is the version the tarball records. The
keywords are stock: ``BFGSTS`` with no arguments, so ``NEVS``, ``NBFGSMAX``,
``CEIG``, and ``NSECDIAG`` stay at the source defaults. ``BFGSCONV`` is the
archive RMS stop, the Euclidean force norm over ``sqrt(3N)``. ``BFGSSTEPS``
is the problem force-call cap. The number is OPTIM's energy+gradient calls.
The record counter is ``harness``, which is how a native counter is ranked.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

from .common import run_jobs, write_campaign
from .optim_search import _stored_archive, _xyz_starts, energy_gradient_calls, odata_text

TARBALL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
TARBALL_SHA256 = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
SVNREV = "df8da599267534fbe43a250e578f7396a7946311"
_BAKED = Path("/tmp/wales-svn/build-optim/OPTIM")


def binary_path() -> Path:
    env = os.environ.get("OPTIM_LATEST")
    if env:
        return Path(env)
    return _BAKED


def stock_preamble(force_norm: float, natoms: int, cap: int) -> str:
    """Stock ``BFGSTS``. The RMS stop and the cap come from the problem."""
    rms = float(force_norm) / math.sqrt(3 * int(natoms))
    rms_text = f"{rms:.16E}".replace("E", "D")
    return f"BFGSTS\nBFGSCONV {rms_text}\nBFGSSTEPS {int(cap)}\n"


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name = str(job["name"])
    cap = int(job["cap"])
    calls = None
    converged = False
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-stock-") as raw:
            folder = Path(raw)
            (folder / "odata").write_text(job["odata"], encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            code = 0
            try:
                proc = subprocess.run(
                    [str(job["binary"])],
                    cwd=folder,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                log = proc.stdout + "\n" + proc.stderr
                code = int(proc.returncode)
            except subprocess.TimeoutExpired:
                log = ""
                code = 1
                error = "OPTIM exceeded 180s"
            calls = energy_gradient_calls(log)
            converged = "**** CONVERGED ****" in log
            if code != 0 and not converged and not error:
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                error = f"OPTIM exit {code}: {detail[-1] if detail else 'no log'}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if calls is None:
        recorded = cap
        success = False
        failure = "crash"
    elif calls >= cap or not converged or error:
        recorded = int(calls)
        success = False
        failure = "cap" if calls >= cap else ("crash" if error else "divergence")
    else:
        recorded = int(calls)
        success = True
        failure = None
    record = {
        "start": name,
        "force_calls": recorded,
        "success": success,
        "failure": failure,
        "counter": "harness",
        "converged": bool(converged),
        "optim_energy_gradient_calls": None if calls is None else int(calls),
    }
    if error:
        record["error"] = error
    return record


def run_stock(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    if template.get("commit") != TARBALL_SHA256:
        raise ValueError(
            f"{template['id']} commit is not the latest tarball sha256 {TARBALL_SHA256}"
        )
    binary = binary_path()
    if not binary.is_file():
        raise RuntimeError(
            "The latest OPTIM binary is not built. "
            f"The pin is the sha256 {TARBALL_SHA256} of {TARBALL}. "
            f"SVNREV is {SVNREV}. {binary} is not present."
        )
    section = dict(template.get("optim") or {})
    archive = _stored_archive(Path(root), str(section["geometry"]))
    starts = _xyz_starts(archive)
    expected = int(section.get("starts") or (problem.get("sample") or {}).get("starts") or 0)
    if len(starts) != expected:
        raise RuntimeError(f"{archive} has {len(starts)} xyz starts, the template states {expected}")
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    force_norm = float((problem.get("convergence") or {})["force_norm"])
    natoms = len(starts[0][1])
    preamble = stock_preamble(force_norm, natoms, cap)
    jobs = [
        {
            "name": name,
            "cap": cap,
            "binary": str(binary),
            "odata": odata_text(preamble, points),
        }
        for name, points in starts
    ]
    records = run_jobs(jobs, _execute, workers)
    comment = (
        f"# OPTIM SVNREV {SVNREV}. Latest {TARBALL} sha256 {TARBALL_SHA256}.\n"
        "# Stock BFGSTS. NSECDIAG stays 2. BFGSCONV is the problem force norm over sqrt(3N). "
        "BFGSSTEPS is the problem cap.\n"
        "# The count is OPTIM energy+gradient calls. The record counter is harness so the row is ranked."
    )
    return write_campaign(
        output,
        records,
        template,
        problem,
        {"cap": cap},
        {
            "method": "hef-stock",
            "keyword": "BFGSTS",
            "counter": "optim energy+gradient calls",
            "version": SVNREV,
            "tarball_sha256": TARBALL_SHA256,
        },
        images=None,
        comment=comment,
    )
