"""OPTIM hybrid eigenvector following for the 2014 Pt-island CD row.

The binary is OPTIM from ``wales-27058.tar.bz2``. That tree's SVNREV and
the version line are 27059. The potential is OPTIM's native periodic
Morse (atom type M, BULK, PARAMS), not the harness smooth Morse and not
the ASE socket. USEDIAG 2 is Rayleigh-Ritz with second-order central
differences. The score is the last ``energy+gradient calls`` line.
``getdata.py`` keeps a start only when the log contains ``CONVERGED``.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .common import run_jobs, write_campaign

TARBALL = "https://optbench.org/codes/wales-27058.tar.bz2"
TARBALL_SHA256 = "144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"
VERSION = "27059"
R0 = 2.8970
RHO = 1.6047
RCUT = 9.5
_PREAMBLE = Path(__file__).resolve().parents[2] / "templates" / "optim" / "preambles" / "pt-island-cd.txt"
_BAKED = Path("/tmp/wales27058/wales/OPTIM/bin/gfortran/OPTIM")


def binary_path() -> Path:
    env = os.environ.get("OPTIM_CD_BIN")
    if env:
        return Path(env)
    if _BAKED.is_file():
        return _BAKED
    fallback = os.environ.get("OPTIM_BIN")
    if fallback:
        return Path(fallback)
    return _BAKED


def _missing(binary: Path) -> str:
    return (
        "OPTIM 27059 is not available. "
        f"The pin is the sha256 {TARBALL_SHA256} of {TARBALL}. "
        f"{binary} is not present. "
        "The Pt-island CD score is OPTIM's energy+gradient calls."
    )


def read_con(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tsase and eOn con files, the line layout ``con2coords.py`` uses.

    Line 3 is the box, line 8 is the atom count, and coordinates start
    on line 12. The frozen flag is the fourth field. Indices are 1-based
    in those descriptions and 0-based here.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    box = np.array([float(part) for part in lines[2].split()[:3]], dtype=np.float64)
    count = int(lines[7].split()[0])
    coords = np.zeros((count, 3), dtype=np.float64)
    frozen = np.zeros(count, dtype=bool)
    for index, line in enumerate(lines[11 : 11 + count]):
        parts = line.split()
        coords[index] = [float(parts[0]), float(parts[1]), float(parts[2])]
        frozen[index] = bool(int(parts[3]))
    return coords, frozen, box


def _scaled_block(coords: np.ndarray, frozen: np.ndarray, box: np.ndarray) -> str:
    """PARAMS, FREEZE, and points in the units ``run.sh`` wrote."""
    scale = 1.0 / R0
    scaled = np.asarray(coords, dtype=np.float64) * scale
    lengths = np.asarray(box, dtype=np.float64) * scale
    rho = RHO * R0
    cutoff = RCUT / R0
    lines = [
        "PARAMS %.20f %f %f %f %.20f"
        % (rho, float(lengths[0]), float(lengths[1]), float(lengths[2]), cutoff)
    ]
    indices = [str(int(index) + 1) for index in np.flatnonzero(frozen)]
    for start in range(0, len(indices), 8):
        lines.append("FREEZE " + " ".join(indices[start : start + 8]))
    lines.append("points")
    buffer = io.StringIO()
    np.savetxt(buffer, scaled, fmt="%.16f")
    text = buffer.getvalue()
    for row in text.splitlines():
        if row.strip():
            lines.append("M " + row.strip())
    return "\n".join(lines) + "\n"


def odata_text(coords: np.ndarray, frozen: np.ndarray, box: np.ndarray) -> str:
    preamble = _PREAMBLE.read_text(encoding="utf-8")
    if not preamble.endswith("\n"):
        preamble += "\n"
    return preamble + _scaled_block(coords, frozen, box)


def vector_dump(initial: np.ndarray, reactant: np.ndarray) -> str:
    """Displacement OPTIM reads, then normalises.

    ``genmode.x`` writes eigenvalue 100 and the unscaled
    initial-minus-reactant vector. READVEC is followed by VECNORM, so
    the direction is the input.
    """
    delta = np.asarray(initial, dtype=np.float64) - np.asarray(reactant, dtype=np.float64)
    lines = ["100.0"]
    for row in delta:
        lines.append(f"{row[0]:.16e} {row[1]:.16e} {row[2]:.16e}")
    return "\n".join(lines) + "\n"


def paper_calls(log: str) -> tuple[int | None, bool]:
    """``run_optim.sh``: field 6 of the last energy+gradient line, and CONVERGED."""
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
    return calls, "**** CONVERGED ****" in log


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name = str(job["name"])
    cap = int(job["cap"])
    binary = Path(job["binary"])
    calls = None
    converged = False
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="optim-cd-") as raw:
            folder = Path(raw)
            (folder / "odata").write_text(job["odata"], encoding="utf-8")
            (folder / "vector.dump").write_text(job["vector"], encoding="utf-8")
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
                    timeout=3600,
                    check=False,
                )
            log = stdout_path.read_text(encoding="utf-8", errors="replace")
            calls, converged = paper_calls(log)
            if proc.returncode != 0 and not converged:
                tail = stderr_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                detail = tail[-1] if tail else f"exit {proc.returncode}"
                error = f"OPTIM exit {proc.returncode}: {detail}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    success = bool(converged) and calls is not None and calls < cap and not error
    record = {
        "start": name,
        "force_calls": int(calls) if calls is not None else cap,
        "success": success,
        "failure": None if success else ("crash" if calls is None or error else "divergence"),
        "counter": "harness",
        "converged": bool(converged),
        "optim_energy_gradient_calls": None if calls is None else int(calls),
    }
    if error:
        record["error"] = error
    return record


def _assert_version(binary: Path) -> None:
    proc = subprocess.run(
        [str(binary)],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    if VERSION not in text:
        line = next((item for item in text.splitlines() if item.strip()), "")
        raise RuntimeError(
            f"{binary} prints {line!r}. The Pt-island CD row needs OPTIM version {VERSION}. "
            f"The pin is the sha256 {TARBALL_SHA256} of {TARBALL}."
        )


def run_hef_cd(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    binary = binary_path()
    if not binary.is_file():
        raise RuntimeError(_missing(binary))
    if not _PREAMBLE.is_file():
        raise RuntimeError(f"{_PREAMBLE} is missing")
    _assert_version(binary)
    from optbench.engines.geometries import _extract, _stored

    archive = _stored(Path(root), str((problem.get("input") or {})["archive_path"]))
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    folder = _extract(archive, scratch / "geometries") / "pt-island-con"
    reactant, _frozen_reactant, _box_reactant = read_con(folder / "reactant.con")
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    jobs = []
    sample = int((problem.get("sample") or {}).get("starts") or 49)
    for index in range(sample):
        path = folder / f"initial_{index}.con"
        coords, frozen, box = read_con(path)
        jobs.append(
            {
                "name": f"initial_{index}.con",
                "cap": cap,
                "binary": str(binary),
                "odata": odata_text(coords, frozen, box),
                "vector": vector_dump(coords, reactant),
            }
        )
    records = run_jobs(jobs, _execute, workers)
    comment = (
        "# OPTIM version 27059, wales-27058.tar.bz2. "
        "Native periodic Morse, USEDIAG 2, BFGSTS 30 11 30 0.063 100.\n"
        "# The count is OPTIM's energy+gradient calls, the 2014 paper counter. "
        "A start is kept when the log contains CONVERGED."
    )
    published = dict(template)
    published["code"] = "optim"
    return write_campaign(
        output,
        records,
        published,
        problem,
        {"cap": cap},
        {
            "method": "hef-cd",
            "keyword": "BFGSTS",
            "usediag": 2,
            "counter": "optim energy+gradient calls",
            "potential": "morse_bulk",
            "version": VERSION,
        },
        images=None,
        comment=comment,
    )
