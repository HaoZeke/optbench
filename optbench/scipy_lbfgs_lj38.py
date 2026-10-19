#!/usr/bin/env python3
"""LJ38 minimization with pele's 28 Aug 2013 scipy L-BFGS driver.

SVN r267 ``bin/js850/minimization_lbfgs.py --lbfgs-scipy`` calls
``pele.optimize.lbfgs_scipy`` (pele commit 2d5bc8e, 27 Aug 2013). That
wrapper is ``scipy.optimize.fmin_l_bfgs_b`` with ``factr=10``, memory 10,
``maxfun=10000``, and ``pgtol = 0.01 / sqrt(3 * 38)``. The structures are
the ``.xyz`` files in ``lj38-clusters.tgz``. A call is one entry to
``getEnergyGradient`` during the quench. The initial energy read and the
post-quench check are not counted. A start fails when the population
standard deviation of the gradient is above ``pgtol``.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.optimize

NATOMS = 38
CAP = 10000
FACTR = 10.0
TOL = 0.01 / np.sqrt(3.0 * NATOMS)


def read_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].split()[0])
    xyz = np.zeros((count, 3), dtype=float)
    for index, line in enumerate(lines[2 : 2 + count]):
        parts = line.split()
        xyz[index] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return xyz.ravel()


def lennard_jones(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Unshifted Lennard-Jones, epsilon = sigma = 1, no cutoff."""
    points = np.asarray(coords, dtype=float).reshape(-1, 3)
    delta = points[:, None, :] - points[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", delta, delta)
    pairs = np.triu_indices(points.shape[0], 1)
    pair_distance2 = distance2[pairs]
    inverse2 = 1.0 / pair_distance2
    inverse6 = inverse2 ** 3
    inverse12 = inverse6 * inverse6
    energy = 4.0 * float(np.sum(inverse12 - inverse6))
    slope = 4.0 * (-6.0 * inverse12 + 3.0 * inverse6) * inverse2
    left, right = pairs
    pair_gradient = (slope * 2.0)[:, None] * delta[left, right]
    gradient = np.zeros_like(points)
    np.add.at(gradient, left, pair_gradient)
    np.add.at(gradient, right, -pair_gradient)
    return energy, gradient.ravel().copy()


def quench(coords: np.ndarray) -> dict:
    calls = 0

    def objective(point: np.ndarray):
        nonlocal calls
        calls += 1
        return lennard_jones(point)

    final, _energy, info = scipy.optimize.fmin_l_bfgs_b(
        objective,
        coords,
        iprint=-1,
        pgtol=TOL,
        maxfun=CAP,
        factr=FACTR,
    )
    counted = calls
    _energy, gradient = lennard_jones(final)
    rms = float(np.std(gradient))
    force_norm = float(np.linalg.norm(gradient))
    return {
        "force_calls": counted,
        "success": int(rms <= TOL),
        "rms": rms,
        "force_norm": force_norm,
        "warnflag": int(info["warnflag"]),
    }


def _job(path: str) -> dict:
    result = quench(read_xyz(Path(path)))
    result["start"] = Path(path).name
    return result


def write_dat(rows: list[dict], output: Path) -> None:
    calls = np.asarray([row["force_calls"] for row in rows], dtype=float)
    failed = int(sum(1 - row["success"] for row in rows))
    mean = float(calls.mean())
    text = (
        f"force_calls {round(mean)}\n"
        f"force_calls_min {int(calls.min())}\n"
        f"force_calls_max {int(calls.max())}\n"
        f"nfailed {failed}\n"
        "code pele\n"
        "code_version 2d5bc8e3\n"
        "algorithm lbfgs_scipy\n"
        "template pele/lj38-scipy-lbfgs\n"
        "problem minimization/lj38\n"
        "counter harness\n"
        "scipy 1.14.1\n"
        f"factr {FACTR:g}\n"
        f"pgtol {TOL:.16g}\n"
        "memory 10\n"
        f"maxfun {CAP}\n"
        f"unrounded_mean {mean:.6f}\n"
        "starts 1000\n"
        "note r267 minimization_lbfgs.py --lbfgs-scipy on the xyz starts. "
        "pgtol is 0.01/sqrt(114). A start fails when the population std of "
        "the gradient exceeds pgtol.\n"
    )
    output.write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    paths = [args.geometries / f"cluster_{index:04d}.xyz" for index in range(1000)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{missing[0]} is missing; the 2013 driver reads 1000 xyz files")
    jobs = [str(path) for path in paths]
    if args.workers == 1:
        rows = [_job(path) for path in jobs]
    else:
        with Pool(args.workers) as pool:
            rows = list(pool.map(_job, jobs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_dat(rows, args.output)
    starts = args.output.with_suffix(".starts.jsonl")
    lines = []
    for row in rows:
        # cluster_0000.xyz and cluster_0000.con are the same index.
        name = row["start"].replace(".xyz", ".con")
        failed = row["success"] == 0 or row["force_calls"] >= CAP or row["force_norm"] > 0.01
        lines.append(
            json.dumps(
                {
                    "failed": failed,
                    "force_calls": row["force_calls"],
                    "force_norm": row["force_norm"],
                    "rms": row["rms"],
                    "start": name,
                    "counter": "harness",
                },
                sort_keys=True,
            )
        )
    starts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {starts}")


if __name__ == "__main__":
    main()
