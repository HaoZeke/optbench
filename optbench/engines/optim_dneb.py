"""OPTIM double-ended search for the LJ38 two-transition pairs.

The binary is the current public Wales ``svn.tar.bz2``.
``OPTIM/source/SVNREV`` in that tree is ``df8da599``, the same identifier
the other OPTIM and GMIN rows use. The keyword block is the archived
``odata`` deck, ``templates/optim/preambles/de-2ts.txt``, copied as
written. ``NEWCONNECT`` stays at 20 cycles. ``NEWNEB`` stays commented.
``BFGSSTEPS`` is not set; ``STEPS 200`` is the step limit inside one
optimiser call. ``REOPTIMISEENDPOINTS`` quenches an endpoint whose RMS
is above the ``BFGSMIN`` tolerance, and those calls stay in the same
energy+gradient count. The problem force-call cap is only the outer
budget: a run whose count reaches that cap is a cap failure.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .common import run_jobs, write_campaign

VERSION = "df8da599"
SOURCE_COMMIT = "df8da599267534fbe43a250e578f7396a7946311"
_PREAMBLE = Path(__file__).resolve().parents[2] / "templates" / "optim" / "preambles" / "de-2ts.txt"
_BAKED = Path("/tmp/wales-latest/build/OPTIM")


def binary_path() -> Path:
    env = os.environ.get("OPTIM_DNEB_BIN") or os.environ.get("OPTIM_BIN")
    if env:
        return Path(env)
    return _BAKED


def read_xyz(path: Path) -> np.ndarray:
    """Archive xyz. Line 1 is the atom count. Coordinates start on line 3."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    count = int(lines[0].split()[0])
    coords = np.zeros((count, 3), dtype=np.float64)
    for index, line in enumerate(lines[2 : 2 + count]):
        parts = line.split()
        coords[index] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return coords


def odata_text(coords: np.ndarray, cap: int | None = None) -> str:
    """Archived odata plus POINTS. ``cap`` is the outer budget, not a keyword."""
    del cap
    preamble = _PREAMBLE.read_text(encoding="utf-8")
    if not preamble.endswith("\n"):
        preamble += "\n"
    lines = [preamble.rstrip("\n"), "POINTS"]
    for row in np.asarray(coords, dtype=np.float64):
        lines.append(f"AX {row[0]:.16f} {row[1]:.16f} {row[2]:.16f}")
    return "\n".join(lines) + "\n"


def finish_text(coords: np.ndarray) -> str:
    lines = [f"{row[0]:.16f} {row[1]:.16f} {row[2]:.16f}" for row in np.asarray(coords, dtype=np.float64)]
    return "\n".join(lines) + "\n"


def _stop_reason(log: str, connected: bool, error: str) -> str:
    """The OPTIM line that says why the pair stopped."""
    if error:
        return error
    if connected:
        return "Connected path found"
    markers = (
        "Number of allowed connection steps reached",
        "Bad endpoints",
        "no connections left",
        "checkendpoints>",
    )
    chosen = ""
    for line in log.splitlines():
        text = line.strip()
        if any(marker in text for marker in markers):
            chosen = text
    return chosen or "OPTIM finished without a connected path"


def paper_calls(log: str) -> tuple[int | None, bool]:
    """Field 6 of the last energy+gradient line, and Connected path found."""
    calls = None
    for line in log.splitlines():
        if "energy+gradient calls" not in line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            calls = int(float(parts[5]))
        except ValueError:
            continue
    return calls, "Connected path found" in log


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name = str(job["name"])
    cap = int(job["cap"])
    binary = Path(job["binary"])
    calls = None
    connected = False
    error = ""
    log = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-dneb-") as raw:
            folder = Path(raw)
            (folder / "odata").write_text(job["odata"], encoding="utf-8")
            (folder / "finish").write_text(job["finish"], encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            stdout_path = folder / "OPTIM.log"
            stderr_path = folder / "stderr.txt"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.run(
                    [str(binary)],
                    cwd=folder,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=28800,
                    check=False,
                )
            log = stdout_path.read_text(encoding="utf-8", errors="replace")
            calls, connected = paper_calls(log)
            if proc.returncode != 0 and not connected:
                tail = stderr_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                detail = tail[-1] if tail else f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if calls is None or error:
        failure = "crash"
    elif calls >= cap:
        failure = "cap"
    elif not connected:
        failure = "divergence"
    else:
        failure = None
    return {
        "start": name,
        "force_calls": int(calls) if calls is not None else cap,
        "success": failure is None,
        "failure": failure,
        "counter": "harness",
        "converged": bool(connected),
        "optim_energy_gradient_calls": None if calls is None else int(calls),
        "optim_stop": _stop_reason(log, connected, error),
        **({"error": error} if error else {}),
    }


def run_dneb_stock(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    binary = binary_path()
    if not binary.is_file():
        raise RuntimeError(f"OPTIM {VERSION} is not available. {binary} is not present.")
    if not _PREAMBLE.is_file():
        raise RuntimeError(f"{_PREAMBLE} is missing")
    from optbench.engines.geometries import _extract, _stored

    archive = _stored(Path(root), str((problem.get("input") or {})["archive_path"]))
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    folder = _extract(archive, scratch / "geometries")
    coords = folder / "coords"
    if not coords.is_dir():
        found = list(folder.rglob("start_0.xyz"))
        if not found:
            raise FileNotFoundError(f"{archive} has no start_0.xyz")
        coords = found[0].parent
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    sample = int((problem.get("sample") or {}).get("pairs") or 51)
    jobs = []
    for index in range(sample):
        jobs.append(
            {
                "name": f"pair_{index:02d}",
                "cap": cap,
                "binary": str(binary),
                "odata": odata_text(read_xyz(coords / f"start_{index}.xyz"), cap),
                "finish": finish_text(read_xyz(coords / f"end_{index}.xyz")),
            }
        )
    records = run_jobs(jobs, _execute, workers)
    comment = (
        f"# OPTIM {VERSION}, OPTIM/source/SVNREV in the current public svn.tar.bz2. "
        "odata is the archived LJ38 two-transition deck. "
        "NEWCONNECT cycles stay 20. NEWNEB stays commented. STEPS stays 200. "
        f"The outer force-call budget is {cap}.\n"
        "# Archive pairs 0-50. The count is OPTIM's energy+gradient calls. "
        "A success is Connected path found under that budget. "
        "REOPTIMISEENDPOINTS quenches an endpoint above BFGSMIN and those calls count."
    )
    published = dict(template)
    published["code"] = "optim"
    images = (problem.get("band") or {}).get("interior_images")
    return write_campaign(
        output,
        records,
        published,
        problem,
        {"cap": cap},
        {
            "method": "dneb-stock",
            "counter": "optim energy+gradient calls",
            "potential": "LJDIFF",
            "version": VERSION,
            "keywords": "archived odata",
            "newconnect_cycles": 20,
            "steps": 200,
            "reoptimiseendpoints": True,
            "pairs": sample,
        },
        images=int(images) if images else None,
        comment=comment,
    )
