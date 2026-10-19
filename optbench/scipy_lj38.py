#!/usr/bin/env python3
"""LJ38 minimization with the current SciPy L-BFGS-B, stock settings.

The archived row is pele's 2013 ``lbfgs_scipy`` score. This run is
SciPy's own minimizer on the same problem: the harness Lennard-Jones
(epsilon = sigma = 1, cutoff 40, no shift), the 1000 ``cluster_*.con``
starts, and the Euclidean force stop 0.01 with a 10000-call cap.

``scipy.optimize.minimize(..., method="L-BFGS-B")`` keeps the 1.18.1
defaults: ``maxcor`` 10, ``ftol`` 2.2204460492503131e-09, ``gtol`` 1e-5,
``eps`` 1e-8, ``maxls`` 20, ``maxiter`` 15000. ``maxfun`` is the problem
cap. A callback stops when the harness force norm is at or below 0.01.
Nothing is tuned toward the archived call counts.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy
import scipy.optimize

from optbench.potentials import lennard_jones, read_con
from optbench.score import half_even

SCIPY_VERSION = "1.18.1"
NATOMS = 38
CAP = 10000
FORCE_NORM = 0.01
GEOMETRIES = 1000


def quench(coords: np.ndarray) -> dict:
    state = {"calls": 0, "gradient": None}

    def objective(point: np.ndarray):
        state["calls"] += 1
        energy, gradient = lennard_jones(point)
        state["gradient"] = gradient
        return energy, gradient

    def callback(intermediate_result):
        gradient = state["gradient"]
        if state["calls"] >= CAP or float(np.linalg.norm(gradient)) <= FORCE_NORM:
            raise StopIteration

    scipy.optimize.minimize(
        objective,
        np.asarray(coords, dtype=float).ravel(),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxfun": CAP},
    )
    gradient = state["gradient"]
    force_norm = float("inf") if gradient is None else float(np.linalg.norm(gradient))
    calls = int(state["calls"])
    failed = force_norm > FORCE_NORM or calls >= CAP
    return {"force_calls": calls, "force_norm": force_norm, "failed": failed}


def _job(path: str) -> dict:
    positions, _box = read_con(Path(path).read_text(encoding="utf-8"))
    if len(positions) != NATOMS:
        raise RuntimeError(f"{path} has {len(positions)} atoms")
    result = quench(positions)
    result["start"] = Path(path).name
    return result


def write_dat(rows: list[dict], output: Path) -> None:
    calls = [int(row["force_calls"]) for row in rows]
    failed = [row["failed"] or row["force_calls"] >= CAP for row in rows]
    kept = [count for count, bad in zip(calls, failed) if not bad]
    if not kept:
        raise RuntimeError("every start failed")
    text = (
        f"force_calls {half_even(sum(kept) / len(kept))}\n"
        f"force_calls_min {min(kept)}\n"
        f"force_calls_max {max(kept)}\n"
        f"nfailed {sum(failed)}\n"
        "code SciPy\n"
        f"code_version {SCIPY_VERSION}\n"
        "algorithm L-BFGS-B\n"
        "template scipy/lj38-lbfgs\n"
        "problem minimization/lj38\n"
        "counter harness\n"
        "tuning stock\n"
        "maxcor 10\n"
        "ftol 2.2204460492503131e-09\n"
        "gtol 1e-5\n"
        "maxls 20\n"
        "maxiter 15000\n"
        f"maxfun {CAP}\n"
        f"force_norm {FORCE_NORM}\n"
        f"starts {len(rows)}\n"
        "note SciPy 1.18.1 minimize method L-BFGS-B at the library defaults, "
        "including ftol. maxfun is the problem cap. A callback also stops at "
        "Euclidean force norm 0.01. ftol can return first, and that start is "
        "a harness failure. The potential is unshifted LJ cutoff 40.\n"
    )
    output.write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    if scipy.__version__ != SCIPY_VERSION:
        raise SystemExit(f"expected SciPy {SCIPY_VERSION}, found {scipy.__version__}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    paths = [args.geometries / f"cluster_{index:04d}.con" for index in range(GEOMETRIES)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{missing[0]} is missing")
    jobs = [str(path) for path in paths]
    if args.workers == 1:
        rows = [_job(path) for path in jobs]
    else:
        with Pool(args.workers) as pool:
            rows = list(pool.map(_job, jobs, chunksize=4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_dat(rows, args.output)
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "counter": "harness",
                    "failed": bool(row["failed"]),
                    "force_calls": int(row["force_calls"]),
                    "force_norm": row["force_norm"],
                    "start": row["start"],
                },
                sort_keys=True,
            )
        )
    starts = args.output.with_suffix(".starts.jsonl")
    starts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {starts}")


if __name__ == "__main__":
    main()
