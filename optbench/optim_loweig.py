"""Second-order Rayleigh-Ritz lowest eigenvector for OPTIM.

The 2014 website preamble is the archived setup: ``usediag 2`` (central
differences of the gradient), Rayleigh-Ritz memory 100, and a maximum
Rayleigh-Ritz step of 1. The geometries are the archived xyz set in
``website/lowest-eigenvector/lj38/coords.tar.gz``.

``df8da599`` still requires the eigenvalue percent change to stay under
``EVPC`` (default 1) after the overlap crosses the cover. The keyword
documentation says a cover value takes ``ceig`` out of function.
``BFGSTSPC 1.0D10`` restores that cover stop. The score is OPTIM's
printed energy-and-gradient count, the same field ``run_optim.sh`` reads.
A start is a failure only when that count is at least 10000.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_KEY = "website/lowest-eigenvector/lj38/coords.tar.gz"
CAP = 10000
OVERLAP = 0.99
COMMIT = "df8da599267534fbe43a250e578f7396a7946311"
TEMPLATE = "optim/rr-second-order"

# The website odata.preamble, plus BFGSTSPC. The extra keyword is not in
# the 2014 preamble. It disables the later eigenvalue-change veto.
PREAMBLE = """\
debug
bfgssteps 1000
comment convergence tolerance is norm(gradient)=1e-3, or that RMS(gradient)=1e-3 / sqrt(3*natoms) = 9.36585D-05
comment bfgsconv 9.36585D-05
nohess
eigenonly 1D-13 0.99
maxbfgs 0.2 1.0
updates 4 100
usediag 2
READVEC
BFGSTSPC 1.0D10
"""


def binary_path() -> Path:
    env = os.environ.get("OPTIM_BIN", "")
    path = Path(env) if env else Path("/tmp/src/wales/OPTIM/build/gfortran/OPTIM")
    if not path.is_file():
        raise FileNotFoundError(f"OPTIM binary is not at {path}")
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
    work.mkdir(parents=True)
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
        "success": success,
        "converged": converged,
        "overlap": overlap,
        "returncode": result.returncode,
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


def run_all(coords: Path, workers: int = 1) -> list[dict]:
    binary = str(binary_path())
    specs = [(binary, str(coords), index) for index in range(200)]
    if workers <= 1:
        rows = [_run_one(binary, str(coords), index) for index in range(200)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, specs))
    rows.sort(key=lambda row: row["start"])
    return rows


def write_dat(rows: list[dict], output: Path) -> dict:
    from optbench.score import half_even

    if len(rows) != 200:
        raise RuntimeError(f"expected 200 starts, got {len(rows)}")
    calls = [int(row["force_calls"]) for row in rows]
    failed = [row for row in rows if int(row["force_calls"]) >= CAP or not row["success"]]
    if failed:
        raise RuntimeError(f"{len(failed)} starts did not finish under the cover stop")
    raw = sum(calls) / len(calls)
    rounded = half_even(raw)
    median = statistics.median(calls)
    note = (
        "Website odata.preamble: usediag 2, updates 4 100, maxbfgs 0.2 1.0, "
        "eigenonly 1D-13 0.99, READVEC, native Lennard-Jones. "
        "BFGSTSPC 1.0D10 turns off the df8da599 eigenvalue-change veto so the "
        "cover stop is the overlap. The score is OPTIM energy+gradient calls. "
        f"Unrounded mean {raw:.10g}. Half-even mean {rounded}."
    )
    text = [
        f"force_calls {rounded}",
        f"force_calls_min {min(calls)}",
        f"force_calls_max {max(calls)}",
        f"force_calls_median {median:g}",
        f"unrounded_mean {raw:.10g}",
        "nfailed 0",
        "starts 200",
        "code OPTIM",
        "code_version df8da599",
        f"commit {COMMIT}",
        "algorithm hybrid eigenvector following",
        f"template {TEMPLATE}",
        "problem lowest-eigenvector/lj38",
        "counter optim",
        "reducer all",
        "parameter_source archive",
        f"note {note}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    starts = output.with_name(output.stem + ".starts.jsonl")
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "schema": "optbench.start/v2",
                    "start": row["start"],
                    "seed": 0,
                    "force_calls": int(row["force_calls"]),
                    "success": True,
                    "overlap": row["overlap"],
                    "counter": "optim",
                    "comparable": False,
                },
                sort_keys=True,
            )
        )
    starts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "force_calls": rounded,
        "unrounded_mean": raw,
        "min": min(calls),
        "max": max(calls),
        "median": median,
        "nfailed": 0,
        "n": 200,
    }


def main() -> None:
    dest = Path("/tmp/optbench-optim-loweig")
    coords = extract_coords(ROOT, dest)
    workers = int(os.environ.get("WORKERS", "8"))
    rows = run_all(coords, workers=workers)
    output = ROOT / "results" / "campaign" / "optim-df8da599-rr-second-order.dat"
    summary = write_dat(rows, output)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
