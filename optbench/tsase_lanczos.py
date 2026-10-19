"""TSASE r410 Lanczos lowest-eigenvector campaign.

Archive ``lowest-eigenvector/lj38/tsase-lanczos/dimer_rotate.py`` builds
``lanczos.lanczos_atoms`` with ``phi_tol`` 0, ``ss`` false, and ``dR``
0.001, then raises ``rotationMax`` until the absolute overlap with the
archived mode exceeds 0.99. ``report.sh`` keeps ``forceCalls`` from that
last object only. A start is outside the mean only when that total is
at least 10000, and the archived script writes ``nfailed`` as 0.

The class is commit ``f77cda4`` (git revision 410, 19 Jun 2014).
``minmodesearch`` finite-differences the force along the Lanczos vector
and returns the Ritz vector of the tridiagonal block. With ``ss`` false
the stress block is zero, so the cell rows stay zero and the atomic
displacement is ``dR`` times the unit mode. The potential is the archive
``lj/cut`` (epsilon 1, sigma 1, cutoff 40, no shift). ``random_mode.py``
draws one shared direction from ``numpy.random.seed(314159)`` and ignores
the start index.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

from optbench.pes import lennard_jones

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f77cda4a261872a633da45d05d77225d0bba0f96"
TEMPLATE_ID = "tsase-410/lj38-lanczos"
DR = 0.001
OVERLAP = 0.99
FAILURE_CALLS = 10000


def _vmag(vector: np.ndarray) -> float:
    return float(np.sqrt(np.vdot(vector, vector).real))


def _vunit(vector: np.ndarray) -> np.ndarray:
    magnitude = _vmag(vector)
    if magnitude == 0.0:
        return vector
    return vector / magnitude


def _forces(positions: np.ndarray) -> np.ndarray:
    _energy, gradient = lennard_jones(positions, None)
    return -np.asarray(gradient, dtype=np.float64)


def read_positions(path: Path) -> np.ndarray:
    """Cartesian coordinates from one eOn ``.con`` frame."""
    lines = path.read_text(encoding="utf-8").splitlines()
    num_types = int(lines[6].split()[0])
    counts = [int(token) for token in lines[7].split()[:num_types]]
    index = 9
    positions: list[list[float]] = []
    for count in counts:
        index += 2
        for _ in range(count):
            parts = lines[index].split()
            positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
            index += 1
    return np.asarray(positions, dtype=np.float64)


def read_mode(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    return np.reshape(np.asarray(values, dtype=np.float64), (-1, 3))


def shared_mode(n_atoms: int = 38) -> np.ndarray:
    """The direction written by ``random_mode.py`` for every start."""
    generator = np.random.RandomState(314159)
    mode = generator.random((n_atoms, 3))
    mode = mode / np.linalg.norm(mode)
    padded = np.vstack((mode, np.zeros((3, 3))))
    return _vunit(padded)


def _ritz(block: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(block)
    order = np.argsort(values.real)
    coeff = np.real(vectors[:, order][:, 0])
    return _vunit(basis @ coeff)


def force_calls(positions: np.ndarray, lowest: np.ndarray, mode: np.ndarray) -> int:
    """``lanczos_atoms.forceCalls`` of the first ``rotationMax`` past overlap 0.99.

    One pass matches the archived restart. ``rotationMax`` does not change
    the Krylov vectors, and the recorded total is only the successful run:
    the centre force, the first finite difference, and one force per
    iteration through the stopping index.
    """
    n_atoms = positions.shape[0]
    size = (n_atoms + 3) * 3
    direction = np.asarray(mode, dtype=np.float64).copy()
    centre = np.vstack((_forces(positions), np.zeros((3, 3))))
    calls = 1

    def displaced(vector: np.ndarray) -> np.ndarray:
        return positions + DR * vector[:-3]

    end = np.vstack((_forces(displaced(direction)), np.zeros((3, 3))))
    calls += 1
    trial = direction.flatten()
    beta = _vmag(trial)
    tridiagonal = np.zeros((size, size))
    basis = np.zeros((size, size))
    for index in range(size):
        basis[:, index] = trial / beta
        product = -(end - centre) / DR
        trial = product.flatten()
        if index > 0:
            trial = trial - beta * basis[:, index - 1]
        alpha = float(np.vdot(basis[:, index], trial).real)
        trial = trial - alpha * basis[:, index]
        trial = trial - basis @ (basis.T @ trial)
        tridiagonal[index, index] = alpha
        if index > 0:
            tridiagonal[index - 1, index] = beta
            tridiagonal[index, index - 1] = beta
        beta = _vmag(trial)
        if beta < 1e-18:
            raise RuntimeError(f"Lanczos broke down at iteration {index}")
        direction = np.reshape(trial / beta, (-1, 3))
        end = np.vstack((_forces(displaced(direction)), np.zeros((3, 3))))
        calls += 1
        if index == 0:
            estimate = direction
        else:
            estimate = np.reshape(_ritz(tridiagonal[: index + 1, : index + 1], basis[:, : index + 1]), (-1, 3))
        # phi_tol 0 never stops the inner loop. The script stops on overlap.
        if index >= 1:
            overlap = abs(float(np.vdot(estimate[:-3], lowest).real))
            if overlap > OVERLAP:
                return calls
    return calls


def run_all(coords: Path) -> list[int]:
    mode = shared_mode()
    calls = []
    for index in range(200):
        prefix = f"{index:04d}"
        positions = read_positions(coords / f"{prefix}.con")
        lowest = read_mode(coords / f"{prefix}_mode")
        calls.append(force_calls(positions, lowest, mode))
    return calls


def paper_mean(calls: list[int]) -> int:
    """``report.sh`` awk ``printf "%.0f"`` of the totals under 10000."""
    kept = [value for value in calls if value < FAILURE_CALLS]
    if not kept:
        raise ValueError("every start is at the 10000-call failure line")
    return int(round(sum(kept) / len(kept)))


def write_dat(calls: list[int], output: Path) -> dict:
    kept = [value for value in calls if value < FAILURE_CALLS]
    raw = sum(kept) / len(kept)
    rounded = paper_mean(calls)
    text = [
        f"force_calls {rounded}",
        f"force_calls_min {min(kept)}",
        f"force_calls_max {max(kept)}",
        "nfailed 0",
        f"starts {len(calls)}",
        "code tsase",
        "code_version 410",
        f"commit {COMMIT}",
        "algorithm Lanczos",
        f"template {TEMPLATE_ID}",
        "problem lowest-eigenvector/lj38",
        "counter harness",
        "reducer all",
        "parameter_source archive",
        f"unrounded_mean {raw:.6g}",
        (
            "note Archive dimer_rotate.py: lanczos_atoms phi_tol 0, ss false, "
            "dR 0.001, shared RandomState(314159) mode. report.sh keeps forceCalls "
            "of the rotationMax that first exceeds overlap 0.99 and writes nfailed 0. "
            "One harness force call is one update_general_forces on that successful object. "
            "The potential is lj/cut epsilon 1 sigma 1 cutoff 40."
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    starts = output.with_name(output.stem + ".starts.jsonl")
    lines = []
    for index, value in enumerate(calls):
        lines.append(
            json.dumps(
                {
                    "schema": "optbench.start/v2",
                    "start": f"{index:04d}",
                    "seed": 0,
                    "force_calls": value,
                    "success": value < FAILURE_CALLS,
                    "counter": "harness",
                    "comparable": True,
                },
                sort_keys=True,
            )
        )
    starts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "force_calls": rounded,
        "unrounded_mean": raw,
        "min": min(kept),
        "max": max(kept),
        "n": len(calls),
        "nfailed": 0,
    }


def _coords_dir() -> Path:
    manifest = ROOT / "inputs" / "manifest.jsonl"
    archive = None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == "website/lowest-eigenvector/lj38/coords-con.tar.gz":
            archive = ROOT / row["stored"]
            break
    if archive is None or not archive.is_file():
        raise FileNotFoundError("coords-con.tar.gz is not in inputs/manifest.jsonl")
    dest = Path("/tmp/optbench-tsase-lanczos")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(dest, filter="data")
    return dest / "coords-con"


def main() -> None:
    calls = run_all(_coords_dir())
    output = ROOT / "results" / "campaign" / "tsase-410-lj38-lanczos.dat"
    summary = write_dat(calls, output)
    print(summary)


if __name__ == "__main__":
    main()
