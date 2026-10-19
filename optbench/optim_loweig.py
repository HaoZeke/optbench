"""Second-order Rayleigh-Ritz lowest eigenvector for current OPTIM.

The binary is the current Wales OPTIM snapshot, SVNREV ``df8da599``,
unmodified. The odata is the archived website preamble and nothing else.
The geometries are ``website/lowest-eigenvector/lj38/coords.tar.gz``.
The stop is the archived cover, overlap 0.99.

The score is OPTIM's printed energy-and-gradient count. That native
counter is written as the harness score so the row ranks. The archive
mean stays the paper number. A gap is left as measured.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from optbench.campaign import load_template, resolve_problem
from optbench.engines.common import write_campaign

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_KEY = "website/lowest-eigenvector/lj38/coords.tar.gz"
CAP = 10000
OVERLAP = 0.99
COMMIT = "df8da599267534fbe43a250e578f7396a7946311"
VERSION = "df8da599"
TEMPLATE = "optim/rr-second-order"
OUTPUT = ROOT / "results" / "campaign" / "optim-df8da599-rr-second-order.dat"

# Exact bytes of the 2014 website odata.preamble, including the trailing
# space on the updates line. No keyword is added.
PREAMBLE_LINES = (
    "debug",
    "bfgssteps 1000",
    "comment convergence tolerance is norm(gradient)=1e-3, or that RMS(gradient)=1e-3 / sqrt(3*natoms) = 9.36585D-05",
    "comment bfgsconv 9.36585D-05",
    "nohess",
    "eigenonly 1D-13 0.99",
    "maxbfgs 0.2 1.0",
    "updates 4 100 ",
    "usediag 2",
    "READVEC",
)
PREAMBLE = "\n".join(PREAMBLE_LINES) + "\n"


def binary_path() -> Path:
    """Current OPTIM. Not an older tarball."""
    env = os.environ.get("OPTIM_BIN", "")
    path = Path(env) if env else Path("/tmp/src/wales/OPTIM/build/gfortran/OPTIM")
    if not path.is_file():
        raise FileNotFoundError(f"current OPTIM binary is not at {path}")
    return path


def energy_gradient_calls(log: str) -> int | None:
    """``run_optim.sh`` field: awk ``$6`` of the energy+gradient line.

    The Hessian line is a different counter and is not this score.
    """
    found = None
    for line in log.splitlines():
        if "energy+gradient calls" not in line or "Hessian" in line:
            continue
        parts = line.split()
        if len(parts) >= 6:
            found = int(float(parts[5]))
    return found


def last_overlap(log: str) -> float | None:
    found = None
    for line in log.splitlines():
        if "overlap between current and target vector" not in line:
            continue
        found = float(line.split()[-1])
    return found


def _odata(xyz: Path) -> str:
    lines = [PREAMBLE.rstrip("\n"), "points"]
    body = xyz.read_text(encoding="utf-8").splitlines()
    for line in body[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        lines.append(f"AX  {parts[1]} {parts[2]} {parts[3]}")
    return "\n".join(lines) + "\n"


def _run_one(binary: str, coords: str, index: int) -> dict:
    root = Path(coords)
    prefix = f"{index:04d}"
    work = root / f"run-{prefix}"
    work.mkdir(parents=True, exist_ok=True)
    mode = (root / f"{prefix}_mode").read_text(encoding="utf-8")
    initial = (root / "initial_mode").read_text(encoding="utf-8")
    (work / "odata").write_text(_odata(root / f"{prefix}.xyz"), encoding="utf-8")
    (work / "target.vector").write_text(mode, encoding="utf-8")
    (work / "vector.dump").write_text("100\n" + initial, encoding="utf-8")
    result = subprocess.run(
        [binary],
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log = result.stdout + result.stderr
    (work / "OPTIM.log").write_text(log, encoding="utf-8")
    calls = energy_gradient_calls(log)
    overlap = last_overlap(log)
    converged = "**** CONVERGED ****" in log
    if calls is None:
        raise RuntimeError(f"start {prefix} printed no energy+gradient count")
    success = converged and calls < CAP and overlap is not None and overlap > OVERLAP
    return {
        "start": prefix,
        "force_calls": calls,
        "optim_energy_gradient_calls": calls,
        "success": success,
        "failure": None if success else "divergence",
        "converged": converged,
        "overlap": overlap,
        "counter": "harness",
    }


def _job(spec: tuple[str, str, int]) -> dict:
    binary, coords, index = spec
    return _run_one(binary, coords, index)


def extract_coords(root: Path, dest: Path) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    archive = None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == ARCHIVE_KEY:
            archive = root / row["stored"]
            break
    if archive is None or not archive.is_file():
        raise FileNotFoundError(f"{ARCHIVE_KEY} is not in inputs/manifest.jsonl")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(dest, filter="data")
    coords = dest / "coords"
    if not (coords / "0000.xyz").is_file() or not (coords / "initial_mode").is_file():
        raise FileNotFoundError(f"{archive} has no coords/0000.xyz")
    return coords


def run_all(coords: Path, binary: Path, workers: int = 1) -> list[dict]:
    if "BFGSTSPC" in PREAMBLE.upper():
        raise RuntimeError("the archived preamble must not add a keyword")
    specs = [(str(binary), str(coords), index) for index in range(200)]
    if workers <= 1:
        rows = [_run_one(str(binary), str(coords), index) for index in range(200)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, specs))
    rows.sort(key=lambda row: row["start"])
    return rows


def write_ranked(rows: list[dict]) -> dict:
    failed = [row for row in rows if not row["success"]]
    if len(rows) != 200 or failed:
        raise RuntimeError(f"{len(failed)} of {len(rows)} starts missed the archived cover")
    problem = resolve_problem(ROOT, "lowest-eigenvector/lj38")
    template = load_template(ROOT, TEMPLATE)
    published = dict(template)
    published["code"] = "optim"
    published["tuning"] = "archive"
    comment = (
        "# Current OPTIM, SVNREV df8da599, unmodified. "
        "Archived website preamble, no extra keyword.\n"
        "# The count is OPTIM's energy+gradient calls, written on the harness counter. "
        "The archive mean is the paper number."
    )
    return write_campaign(
        OUTPUT,
        rows,
        published,
        problem,
        {"cap": CAP},
        {
            "method": "xmylbfgs",
            "usediag": 2,
            "updates": "4 100",
            "maxbfgs": "0.2 1.0",
            "cover": 0.99,
            "counter": "optim energy+gradient calls",
            "potential": "lennard-jones",
            "version": VERSION,
        },
        images=None,
        comment=comment,
    )


def main() -> None:
    workers = int(os.environ.get("WORKERS", "8"))
    coords = extract_coords(ROOT, Path("/tmp/optbench-optim-df8da599-rank"))
    rows = run_all(coords, binary_path(), workers=workers)
    summary = write_ranked(rows)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
