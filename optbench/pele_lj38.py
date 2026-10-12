"""pele L-BFGS on the Lennard-Jones 38 starts.

The stop is the Euclidean norm of the force at 0.01, or 10000 gradient
calls. The potential is unshifted Lennard-Jones. Every pair in these
clusters is inside the problem cutoff of 40.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pele.optimize._lbfgs_py import LBFGS
from pele.potentials import LJ

FORCE = 0.01
CAP = 10000
VARIANTS = (
    ("M100-maxstep0.2", 100, 0.2),
    ("M20-maxstep0.1", 20, 0.1),
    ("M10-maxstep0.1", 10, 0.1),
    ("M4-maxstep0.2", 4, 0.2),
    ("M4-maxstep0.1", 4, 0.1),
    ("M4-maxstep0.05", 4, 0.05),
    ("M1-maxstep0.1", 1, 0.1),
)


def read_con(text: str) -> np.ndarray:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Coordinates of Component")) + 1
    xyz = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 3:
            break
        try:
            xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            break
    return np.asarray(xyz, dtype=float).ravel()


def _stop(energy, gradient, tol, coords) -> bool:
    return float(np.linalg.norm(gradient)) < FORCE


def run_one(job: tuple[str, int, float]) -> int:
    path, memory, maxstep = job
    pot = LJ(eps=1.0, sig=1.0)
    coords = read_con(Path(path).read_text(encoding="utf-8"))
    quench = LBFGS(
        coords,
        pot,
        M=memory,
        maxstep=maxstep,
        nsteps=CAP,
        alternate_stop_criterion=_stop,
        iprint=-1,
    )
    result = quench.run()
    calls = int(result.nfev)
    if (not result.success) or calls >= CAP:
        return CAP
    return calls


def write_dat(calls: list[int], memory: int, maxstep: float, template: str, output: Path) -> None:
    kept = [value for value in calls if value < CAP]
    failed = len(calls) - len(kept)
    if not kept:
        text = f"nfailed {failed}\ncode pele\ncode_version 0.1.2\ntemplate {template}\n"
    else:
        values = np.asarray(kept, dtype=float)
        text = (
            f"force_calls {values.mean():.0f}\n"
            f"force_calls_min {int(values.min())}\n"
            f"force_calls_max {int(values.max())}\n"
            f"nfailed {failed}\n"
            f"code pele\n"
            f"code_version 0.1.2\n"
            f"algorithm L-BFGS M={memory} maxstep={maxstep}\n"
            f"template {template}\n"
        )
    output.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    paths = sorted(str(path) for path in args.geometries.glob("*.con"))
    if len(paths) != 1000:
        raise SystemExit(f"{args.geometries} has {len(paths)} con files, the problem has 1000")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, memory, maxstep in VARIANTS:
        jobs = [(path, memory, maxstep) for path in paths]
        print(f"BEGIN pele {name}", flush=True)
        with Pool(args.workers) as pool:
            calls = list(pool.map(run_one, jobs))
        write_dat(calls, memory, maxstep, f"pele-lbfgs-{name}", args.output_dir / f"pele-{name}.dat")
        print(f"DONE pele {name}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
