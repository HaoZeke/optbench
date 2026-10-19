"""LJ38 transition-state campaign for Sella 0.1.1.

Archive ``tsopt/LJ38/sella/tsopt_lj38.py`` builds ``Sella`` with the
archived trust region and the 0.1.1 constructor defaults (method
``rsprfo``, order 1, eigenvalue following on). The loop is ASE 3.18
``irun(0, 1000)``. The stop is the Euclidean force norm at 1e-3. A
start fails when ``nsteps`` reaches 1000. The score is
``PESWrapper.calls``. The potential is ASE Lennard-Jones, epsilon 1,
sigma 1, cutoff 40.

Sella 0.1.1 is the PyPI sdist ``Sella-0.1.1.tar.gz``
(sha256 ``a28b4efba00f0d158b9045bac5804636e3467b16d3bf2531ed1faaf00d993b00``).
Point ``SELLA_011`` at a built tree of that release. The shipped Cython
0.29.7 sources do not compile on Python 3.12; regenerate them with
Cython 0.29.37 and build the extensions. NumPy's ``np.infty`` and
``np.int`` aliases and SciPy's ``LinearOperator`` namespace check are
shimmed. ASE 3.29 ``Optimizer.irun`` is not the 3.18 loop the script
calls, so that loop is installed before Sella is imported. None of
these shims change the trust-region step.
"""

from __future__ import annotations

import json
import os
import sys
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAXITER = 1000
GRADTOL = 1e-3
HYPER = dict(
    eta=1e-6,
    gamma=0.4,
    rho_inc=1.035,
    rho_dec=5.0,
    sigma_inc=1.15,
    sigma_dec=0.65,
    delta0=1.3e-3,
)


def _sella_root() -> Path:
    env = os.environ.get("SELLA_011", "")
    path = Path(env) if env else Path("/tmp/sella011/Sella-0.1.1")
    if not (path / "sella" / "peswrapper.py").is_file():
        raise FileNotFoundError(f"Sella 0.1.1 tree is not at {path}")
    return path


def _shim_numerics() -> None:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator

    np.infty = np.inf
    if not hasattr(np, "int"):
        np.int = int
    original = LinearOperator.dot

    def dot(self, x):
        if not hasattr(self, "_xp"):
            self._xp = np
        return original(self, x)

    LinearOperator.dot = dot
    LinearOperator._check_matching_namespace = lambda self, x: None


def _install_ase_318_optimizer() -> None:
    """ASE 3.18.1 ``Optimizer``, which is what Sella 0.1.1 calls.

    ``irun`` yields once before any step, then once after each step,
    and ``fmax`` 0 never satisfies the max-component test. The archived
    script breaks on the Euclidean norm during those yields.
    """
    import ase.optimize.optimize as optmod

    class Optimizer:
        def __init__(
            self,
            atoms,
            restart=None,
            logfile="-",
            trajectory=None,
            master=None,
            force_consistent=False,
        ):
            self.atoms = atoms
            self.restart = restart
            self.trajectory = trajectory
            self.force_consistent = False if force_consistent is None else bool(force_consistent)
            self.nsteps = 0
            self.max_steps = 100000000
            self.fmax = None
            self.observers = []
            self.logfile = None

        def irun(self, fmax=0.05, steps=None):
            self.fmax = fmax
            if steps:
                self.max_steps = steps
            return self._irun()

        def _irun(self):
            self.atoms.get_forces()
            yield False
            if self.nsteps == 0:
                self.log()
                self.call_observers()
            while (not self.converged()) and self.nsteps < self.max_steps:
                self.step()
                self.nsteps += 1
                yield False
                self.log()
                self.call_observers()
            yield self.converged()

        def converged(self, forces=None):
            if forces is None:
                forces = self.atoms.get_forces()
            return (forces**2).sum(axis=1).max() < self.fmax**2

        def log(self, forces=None):
            if self.logfile is None:
                return None
            if forces is None:
                forces = self.atoms.get_forces()
            fmax = sqrt((forces**2).sum(axis=1).max())
            energy = self.atoms.get_potential_energy()
            self.logfile.write(f"{self.nsteps:4d} {energy:15.6f} {fmax:12.4f}\n")
            return None

        def call_observers(self):
            return None

    optmod.Optimizer = Optimizer


def _prepare_import() -> None:
    root = str(_sella_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    _shim_numerics()
    _install_ase_318_optimizer()
    # Import after the 3.18 optimizer is installed. A leftover Sella 2.6.0
    # module would bind the wrong class.
    for name in list(sys.modules):
        if name == "sella" or name.startswith("sella."):
            del sys.modules[name]


def run_index(coords: Path, index: int) -> dict:
    """One archived structure, using the 2019 script's counting rules."""
    import numpy as np
    from ase.calculators.lj import LennardJones
    from ase.io import read

    _prepare_import()
    from sella import Sella

    saved = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        atoms = read(coords / f"{index:04d}.con")
        atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=40.0)
        dyn = Sella(atoms, logfile=None, **HYPER)
        for _ in dyn.irun(0.0, MAXITER):
            if np.linalg.norm(atoms.get_forces()) < GRADTOL:
                break
        failed = int(dyn.nsteps == MAXITER)
        calls = int(dyn.pes.calls)
    finally:
        sys.stdout.close()
        sys.stdout = saved
    return {"start": f"{index:04d}", "force_calls": calls, "failed": failed}


def _job(spec: tuple[str, int]) -> dict:
    coords, index = spec
    row = run_index(Path(coords), index)
    return row


def run_all(coords: Path, workers: int = 1) -> list[dict]:
    specs = [(str(coords), index) for index in range(200)]
    if workers <= 1:
        return [run_index(coords, index) for index in range(200)]
    import multiprocessing as mp

    context = mp.get_context("spawn")
    with context.Pool(workers) as pool:
        rows = pool.map(_job, specs)
    rows.sort(key=lambda row: row["start"])
    return rows


def write_dat(rows: list[dict], output: Path) -> dict:
    import numpy as np

    calls = np.asarray([int(row["force_calls"]) for row in rows], dtype=float)
    failed = int(sum(int(row["failed"]) for row in rows))
    raw = float(calls.mean()) if len(calls) else 0.0
    text = [
        f"force_calls {raw:.6g}",
        f"force_calls_min {int(calls.min())}",
        f"force_calls_median {float(np.median(calls))}",
        f"force_calls_max {int(calls.max())}",
        f"nfailed {failed}",
        f"starts {len(rows)}",
        "code Sella",
        "code_version 0.1.1",
        "commit 0.1.1",
        "algorithm Sella",
        "template sella-0.1.1/lj38-ts",
        "problem tsopt/LJ38",
        "counter harness",
        "reducer all",
        "parameter_source archive",
        "note Archive tsopt_lj38.py: Sella 0.1.1 method rsprfo, order 1, eta 1e-6, gamma 0.4, delta0 1.3e-3, rho_inc 1.035, rho_dec 5, sigma_inc 1.15, sigma_dec 0.65. ASE 3.18 irun with fmax 0 and 1000 steps. Stop is the Euclidean force norm at 1e-3. A start fails when nsteps reaches 1000. ASE Lennard-Jones cutoff 40. One harness force call is one PESWrapper.calls evaluation.",
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
                    "success": not bool(row["failed"]),
                    "counter": "harness",
                    "comparable": True,
                },
                sort_keys=True,
            )
        )
    starts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "force_calls": raw,
        "min": int(calls.min()),
        "median": float(np.median(calls)),
        "max": int(calls.max()),
        "nfailed": failed,
        "n": len(rows),
    }


def _coords_from_manifest() -> Path:
    import tarfile

    archive = None
    manifest = ROOT / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == "website/tsopt/LJ38/sella/coords-con.tar.gz":
            archive = ROOT / row["stored"]
            break
    if archive is None or not archive.is_file():
        raise FileNotFoundError("coords-con.tar.gz is not in inputs/manifest.jsonl")
    dest = Path("/tmp/optbench-sella-lj38")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(dest)
    return dest / "coords-con"


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    workers = int(os.environ.get("WORKERS", "8"))
    rows = run_all(_coords_from_manifest(), workers=workers)
    output = ROOT / "results" / "campaign" / "sella-0.1.1-lj38-ts.dat"
    print(write_dat(rows, output))


if __name__ == "__main__":
    main()
