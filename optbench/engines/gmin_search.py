"""GMIN core-orbit symmetrisation on LJ150.

The binary is GMIN from the current Wales ``svn.tar.bz2``. ``GMIN/source/SVNREV``
is ``df8da599``. The potential is GMIN's native Lennard-Jones. The score is
the function-call field of ``gsaveit> Target hit after``, the same field
``run_gmin.sh`` keeps. Quenches are the field before that. CPU is the
Fortran ``CPU_TIME`` on the ``mc> target hit`` line.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

from .common import run_jobs, write_campaign

TARBALL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
TARBALL_SHA256 = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
VERSION = "df8da599267534fbe43a250e578f7396a7946311"
CLUSTERS = (
    "inputs/sha256/dcecb7ffe7565a786bc1837cc9edc6e8ebce8fa47f331b07d406523984de3d42/"
    "lj150_clusters.tgz"
)
_DATA = Path(__file__).resolve().parents[2] / "templates" / "gmin" / "lj150-symmetrise.data"
_BAKED = Path("/tmp/gmin-build/GMIN.matrix")
_BLAS = Path("/tmp/blas-ref")

ENGINE = "gmin"
CODE = "gmin"


def binary_path() -> Path:
    env = os.environ.get("GMIN_BIN")
    if env:
        return Path(env)
    return _BAKED


def available() -> str | None:
    binary = binary_path()
    if binary.is_file() and _DATA.is_file():
        return None
    missing = binary if not binary.is_file() else _DATA
    return (
        "GMIN is not available. "
        f"The pin is the sha256 {TARBALL_SHA256} of {TARBALL}. "
        f"SVNREV is {VERSION}. {missing} is not present. "
        "The LJ150 score is GMIN's function calls at Target hit."
    )


def paper_calls(log: str) -> tuple[int | None, int | None, float | None]:
    """``run_gmin.sh`` fields: quenches, function calls, and Fortran CPU time."""
    quenches = None
    calls = None
    cpu = None
    for line in log.splitlines():
        if "Target hit after" in line:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    quenches = int(parts[4])
                    calls = int(parts[6])
                except ValueError:
                    continue
        if "target hit, quenches, function calls, time:" in line:
            parts = line.split()
            if parts:
                try:
                    cpu = float(parts[-1])
                except ValueError:
                    continue
    return quenches, calls, cpu


def _execute(job: dict) -> dict:
    import subprocess
    import tempfile

    name = str(job["name"])
    binary = Path(job["binary"])
    calls = None
    quenches = None
    cpu = None
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="gmin-lj150-") as raw:
            folder = Path(raw)
            (folder / "data").write_text(job["data"], encoding="utf-8")
            (folder / "coords").write_text(job["coords"], encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            if _BLAS.is_dir():
                env["LD_LIBRARY_PATH"] = str(_BLAS) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.run(
                    [str(binary), "output"],
                    cwd=folder,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=7200,
                    check=False,
                )
            log_path = folder / "output"
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            quenches, calls, cpu = paper_calls(log)
            if proc.returncode != 0 and calls is None:
                tail = stderr_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                detail = tail[-1] if tail else f"exit {proc.returncode}"
                error = f"GMIN exit {proc.returncode}: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    success = calls is not None and not error
    record = {
        "start": name,
        "force_calls": int(calls) if calls is not None else 0,
        "success": success,
        "failure": None if success else "crash",
        "counter": "harness",
        "comparable": True,
        "converged": success,
        "basin_hopping_steps": None if quenches is None else int(quenches),
        "cpu_time": cpu,
        "energy_calls": None if calls is None else int(calls),
    }
    if error:
        record["error"] = error
    return record


def _coords(archive: Path, index: int) -> str:
    with tarfile.open(archive) as packed:
        member = packed.getmember(f"lj150-clusters/coords.{index}")
        extracted = packed.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{archive} has no coords.{index}")
        return extracted.read().decode("utf-8")


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    """One serial GMIN search per archived start. The binary is not patched."""
    from optbench.campaign import load_template, resolve_problem

    root = Path(root)
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    binary = binary_path()
    reason = available()
    if reason:
        raise RuntimeError(reason)
    data = _DATA.read_text(encoding="utf-8")
    archive = root / CLUSTERS
    if not archive.is_file():
        raise FileNotFoundError(f"{archive} is not present")
    jobs = []
    for index in range(1, 101):
        jobs.append(
            {
                "name": f"coords.{index}",
                "binary": str(binary),
                "data": data,
                "coords": _coords(archive, index),
            }
        )
    records = run_jobs(jobs, _execute, workers)
    comment = (
        f"# GMIN SVNREV {VERSION}, {TARBALL} sha256 {TARBALL_SHA256}.\n"
        "# Unmodified Release build. Archived data file, coords.1-100, seed 0.\n"
        "# The harness score is GMIN function calls at Target hit, run_gmin.sh field 7.\n"
        "# Quenches are field 5. CPU is Fortran CPU_TIME."
    )
    published = dict(template)
    published["code"] = CODE
    return write_campaign(
        output,
        records,
        published,
        problem,
        {"cap": None},
        {
            "method": "symmetrise",
            "counter": "gmin function calls",
            "potential": "lennard-jones",
            "version": VERSION,
            "tarball_sha256": TARBALL_SHA256,
        },
        images=None,
        comment=comment,
    )
