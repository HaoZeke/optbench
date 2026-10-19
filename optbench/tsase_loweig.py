"""Lowest-eigenvector campaign for TSASE revision 408.

Archive ``lowest-eigenvector/lj38/tsase-dimer-cg/dimer_rotate.py`` builds a
new ``SSDimer_atoms`` for ``rotationMax`` = 1, 2, ... and stops when the
absolute overlap with the reference mode exceeds 0.99. The score is that
attempt's ``forceCalls``. ``report.sh`` prints the mean with ``%.0f`` and
leaves out a start whose total is at least 10000.

Revision 408 is git commit ``5584461`` (``git rev-list --count``), dated
18 Jun 2014, the day on the archived ``benchmark.dat``. Point ``TSASE_408``
at a checkout that contains that commit. The 408 dimer is Python 2; the
loader rewrites ``print`` statements and stubs the package imports. The
rotation arithmetic is the 408 source.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "55844612b607a0d4b9c0205917021b0c34975cc6"
TEMPLATE_ID = "tsase-408/lj38-dimer-cg"
PROBLEM_ID = "lowest-eigenvector/lj38"
OVERLAP = 0.99
LIMIT = 1000
FAILURE_CAP = 10000

_DIMER = None
_MODE = None


def _checkout() -> Path:
    env = os.environ.get("TSASE_408", "")
    path = Path(env) if env else Path("/tmp/src/tsase")
    if not (path / ".git").exists() and not (path / "tsase" / "dimer" / "ssdimer.py").is_file():
        raise FileNotFoundError(f"TSASE checkout is not at {path}")
    return path


def _source_text() -> str:
    root = _checkout()
    if (root / ".git").exists():
        text = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{COMMIT}:tsase/dimer/ssdimer.py"],
            text=True,
        )
    else:
        text = (root / "tsase" / "dimer" / "ssdimer.py").read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("print ") and not stripped.startswith("print("):
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}print({stripped[6:]})")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _util() -> types.ModuleType:
    module = types.ModuleType("tsase_util_408")

    def vmag(vector):
        return np.sqrt(np.vdot(vector, vector))

    def vunit(vector):
        magnitude = vmag(vector)
        if magnitude == 0:
            return vector
        return vector / magnitude

    def vrand(vector):
        return np.random.randn(vector.size).reshape(vector.shape)

    module.vmag = vmag
    module.vunit = vunit
    module.vrand = vrand
    module.sPBC = lambda vector: vector
    return module


def load_dimer():
    """``SSDimer_atoms`` from revision 408."""
    sys.modules["tsase_util_408"] = _util()
    source = _source_text()
    source = source.replace(
        "from tsase.neb.util import vunit, vmag, vrand, sPBC",
        "from tsase_util_408 import vunit, vmag, vrand, sPBC",
    )
    source = source.replace("from tsase.io import read_con", "read_con = None")
    source = source.replace("from tsase import neb", "neb = None")
    module = types.ModuleType("ssdimer_408")
    exec(compile(source, "ssdimer_408.py", "exec"), module.__dict__)
    return module.SSDimer_atoms


def numpy_direction() -> np.ndarray:
    """The archived ``random_mode.py`` vector, including the ``savetxt`` round trip."""
    np.random.seed(314159)
    mode = np.random.random((38, 3))
    mode /= np.linalg.norm(mode)
    buffer = io.StringIO()
    np.savetxt(buffer, mode)
    loaded = np.loadtxt(io.StringIO(buffer.getvalue()))
    magnitude = np.sqrt(np.vdot(loaded, loaded))
    return loaded / magnitude


def read_con(path: Path):
    """One orthogonal con file, using the 408 ``length_angle_to_box`` cell."""
    from ase import Atoms

    lines = path.read_text(encoding="utf-8").splitlines()
    lengths = np.array([float(value) for value in lines[2].split()[:3]])
    angles = np.array([float(value) for value in lines[3].split()[:3]]) * np.pi / 180.0
    box = np.zeros((3, 3))
    box[0, 0] = 1.0
    box[1, 0] = np.cos(angles[0])
    box[1, 1] = np.sin(angles[0])
    box[2, 0] = np.cos(angles[1])
    box[2, 1] = (np.cos(angles[2]) - box[1, 0] * box[2, 0]) / box[1, 1]
    box[2, 2] = np.sqrt(1.0 - box[2, 0] ** 2 - box[2, 1] ** 2)
    box[0, :] *= lengths[0]
    box[1, :] *= lengths[1]
    box[2, :] *= lengths[2]
    ntypes = int(lines[6].split()[0])
    counts = [int(value) for value in lines[7].split()[:ntypes]]
    index = 9
    positions = []
    for count in counts:
        index += 2
        for _ in range(count):
            parts = lines[index].split()
            positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
            index += 1
    return Atoms(symbols=f"H{sum(counts)}", positions=positions, cell=box, pbc=True)


class _LJ:
    """Unshifted Lennard-Jones, built lazily so import does not require ASE."""

    def __new__(cls):
        from ase.calculators.calculator import Calculator

        from optbench.pes import lennard_jones

        class LJCut(Calculator):
            implemented_properties = ("energy", "forces")
            nolabel = True

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
                Calculator.calculate(self, atoms, properties, system_changes)
                energy, gradient = lennard_jones(np.asarray(atoms.positions, dtype=np.float64), None)
                self.results["energy"] = float(energy)
                self.results["forces"] = -np.asarray(gradient, dtype=np.float64)

        return LJCut()


def _init_worker() -> None:
    global _DIMER, _MODE
    warnings.filterwarnings("ignore", category=FutureWarning)
    _DIMER = load_dimer()
    _MODE = numpy_direction()


def run_index(index: int, coords: Path | None = None) -> dict:
    """Force calls for one archived structure. ``coords`` is the unpacked directory."""
    folder = coords or Path(os.environ["TSASE_LOWEIG_COORDS"])
    atoms = read_con(folder / f"{index:04d}.con")
    atoms.calc = _LJ()
    reference = np.loadtxt(folder / f"{index:04d}_mode").reshape(-1, 3)
    dimer_cls = _DIMER or load_dimer()
    mode = _MODE if _MODE is not None else numpy_direction()
    overlap = 0.0
    calls = None
    rotations = None
    for attempt in range(1, LIMIT + 1):
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            dimer = dimer_cls(
                R0=atoms,
                mode=mode.copy(),
                rotationMax=attempt,
                phi_tol=0.0,
                ss=False,
                dR=0.001,
                rotationOpt="cg",
            )
            dimer.get_forces()
            found = np.asarray(dimer.get_mode(), dtype=np.float64)
            calls = int(dimer.forceCalls)
        finally:
            sys.stdout = stdout
        overlap = abs(float(np.vdot(found, reference)))
        rotations = attempt
        if overlap > OVERLAP:
            break
    success = calls is not None and calls < FAILURE_CAP and overlap > OVERLAP
    return {
        "index": index,
        "force_calls": calls,
        "overlap": overlap,
        "rotations": rotations,
        "success": success,
    }


def _job(spec: tuple[str, int]) -> dict:
    folder, index = spec
    return run_index(index, Path(folder))


def run_all(coords: Path, workers: int = 1) -> list[dict]:
    specs = [(str(coords), index) for index in range(200)]
    if workers <= 1:
        _init_worker()
        return [run_index(index, coords) for index in range(200)]
    import multiprocessing as mp

    context = mp.get_context("fork")
    with context.Pool(workers, initializer=_init_worker) as pool:
        rows = pool.map(_job, specs)
    rows.sort(key=lambda row: row["index"])
    return rows


def write_dat(rows: list[dict], output: Path) -> dict:
    """``report.sh``: mean of totals under 10000, printed with ``%.0f``."""
    from optbench.score import half_even

    kept = [int(row["force_calls"]) for row in rows if row["success"] and int(row["force_calls"]) < FAILURE_CAP]
    failed = len(rows) - len(kept)
    if not kept:
        raise RuntimeError("no start finished under the cap")
    raw = sum(kept) / len(kept)
    printed = half_even(raw)
    note = (
        "Archive dimer_rotate.py: SSDimer CG, phi_tol 0, dR 0.001, ss false. "
        "rotationMax rises from 1 until absolute overlap exceeds 0.99. "
        "The score is that attempt's forceCalls. "
        f"The unrounded mean is {raw:.6g}; report.sh prints {printed}."
    )
    text = [
        f"force_calls {printed}",
        f"force_calls_min {min(kept)}",
        f"force_calls_max {max(kept)}",
        f"nfailed {failed}",
        f"starts {len(rows)}",
        "code tsase",
        "code_version 408",
        f"commit {COMMIT}",
        "algorithm dimer CG",
        f"template {TEMPLATE_ID}",
        f"problem {PROBLEM_ID}",
        "counter harness",
        "reducer all",
        "parameter_source archive",
        f"unrounded_mean {raw:.6g}",
        f"note {note}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text) + "\n", encoding="utf-8")
    import json

    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "schema": "optbench.start/v2",
                    "start": f"{row['index']:04d}",
                    "seed": 0,
                    "force_calls": row["force_calls"],
                    "success": bool(row["success"]),
                    "counter": "harness",
                    "comparable": True,
                    "overlap": row["overlap"],
                },
                sort_keys=True,
            )
        )
    output.with_name(output.stem + ".starts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "force_calls": printed,
        "unrounded_mean": raw,
        "min": min(kept),
        "max": max(kept),
        "nfailed": failed,
        "n": len(rows),
    }


def main() -> None:
    import json
    import tarfile

    from optbench.campaign import geometry_archive
    from optbench.problems import load_problem

    problem = load_problem(ROOT / "problems" / "lowest-eigenvector" / "lj38.toml")
    archive = geometry_archive(ROOT, problem)
    dest = Path("/tmp/optbench-tsase-loweig")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(dest)
    coords = dest / "coords-con"
    workers = int(os.environ.get("WORKERS", "8"))
    rows = run_all(coords, workers=workers)
    output = ROOT / "results" / "campaign" / "tsase-408-lj38-dimer-cg.dat"
    summary = write_dat(rows, output)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
