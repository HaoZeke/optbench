"""TSASE r408 dimer BFGS for the LJ38 lowest eigenvector.

Archive ``lowest-eigenvector/lj38/tsase-dimer-bfgs/dimer_rotate.py`` builds
``SSDimer_atoms`` with ``rotationOpt='bfgs'``, ``phi_tol=0``, ``ss=False``
and ``dR=0.001``, then increases ``rotationMax`` until the absolute overlap
with the reference mode exceeds 0.99. The recorded number is that object's
``forceCalls``. ``report.sh`` averages the counts below 10000 with
``printf %.0f`` and writes ``nfailed 0``.

Revision 408 is commit ``55844612`` (the 408th commit on the Henkelman
history). The dimer source at that commit is the 18 Jun 2014 min-mode
reader. A fresh dimer at ``rotationMax=k`` matches iteration ``k`` of one
longer rotation, so the runner stops that one trajectory at the first
overlap above 0.99. The initial direction is NumPy's legacy stream at seed
314159, the same vector ``random_mode.py`` writes for every start. The
potential is unshifted ``lj/cut`` with epsilon 1, sigma 1, and cutoff 40.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ENGINE = "tsase-408"
CODE = "tsase-408"
COMMIT = "55844612b607a0d4b9c0205917021b0c34975cc6"
OVERLAP = 0.99
ROTATION_LIMIT = 1000

_CLASS = None


def source_root() -> Path:
    env = os.environ.get("TSASE_408", "")
    return Path(env) if env else Path("/tmp/src/tsase-408")


def available() -> str | None:
    root = source_root()
    dimer = root / "tsase" / "dimer" / "ssdimer.py"
    if not dimer.is_file():
        return (
            f"TSASE r408 ssdimer.py is not under {root}. "
            f"Set TSASE_408 to the checkout of {COMMIT}."
        )
    if (root / ".git").exists():
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if head != COMMIT:
            return f"TSASE checkout {root} is {head}, not r408 {COMMIT}."
    return None


def _python3(text: str) -> str:
    """Print statements are the only Python 2 syntax this module executes."""
    lines = []
    for line in text.splitlines():
        body = line.lstrip(" ")
        indent = line[: len(line) - len(body)]
        if body.startswith("print ") and not body.startswith("print("):
            lines.append(f"{indent}print({body[len('print '):]})")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _load_class():
    """Execute r408 ``ssdimer.py`` and return ``SSDimer_atoms``."""
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    reason = available()
    if reason:
        raise RuntimeError(reason)
    source = (source_root() / "tsase" / "dimer" / "ssdimer.py").read_text(encoding="utf-8")
    kept = []
    for line in _python3(source).splitlines():
        body = line.strip()
        if body.startswith(("from tsase", "from ase", "import ase")):
            continue
        kept.append(line)
    text = "\n".join(kept) + "\n"
    needle = "            iteration += 1\n"
    stop = (
        "            iteration += 1\n"
        "            _target = getattr(self, '_lowest', None)\n"
        "            if _target is not None and abs(float(np.vdot(self.N[:-3], _target))) > float(getattr(self, '_overlap_tol', 0.99)):\n"
        "                break\n"
    )
    if text.count(needle) != 1:
        raise RuntimeError("TSASE r408 ssdimer.py has no single rotation loop")
    text = text.replace(needle, stop)
    helpers = (
        "import numpy as np\n"
        "def vmag(v):\n"
        "    return np.sqrt(np.vdot(v, v))\n"
        "def vunit(v):\n"
        "    mag = vmag(v)\n"
        "    if mag == 0:\n"
        "        return v\n"
        "    return v / mag\n"
        "def vrand(v):\n"
        "    return np.random.randn(v.size).reshape(v.shape)\n"
        "def sPBC(vdir):\n"
        "    return (vdir % 1.0 + 1.5) % 1.0 - 0.5\n"
    )
    namespace: dict = {}
    exec(helpers + text, namespace)
    _CLASS = namespace["SSDimer_atoms"]
    return _CLASS


def initial_mode(n_atoms: int = 38) -> np.ndarray:
    """``random_mode.py``: one shared direction, seed 314159."""
    rng = np.random.RandomState(314159)
    mode = rng.random_sample((n_atoms, 3))
    mode /= np.linalg.norm(mode)
    return mode


def length_angle_to_box(lengths: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """``tsase/io/con.py`` cell. Angles are degrees."""
    angles = np.asarray(angles_deg, dtype=float) * np.pi / 180.0
    box = np.zeros((3, 3))
    box[0, 0] = 1.0
    box[1, 0] = np.cos(angles[0])
    box[1, 1] = np.sin(angles[0])
    box[2, 0] = np.cos(angles[1])
    box[2, 1] = (np.cos(angles[2]) - box[1, 0] * box[2, 0]) / box[1, 1]
    box[2, 2] = np.sqrt(1.0 - box[2, 0] ** 2 - box[2, 1] ** 2)
    box[0, :] *= float(lengths[0])
    box[1, :] *= float(lengths[1])
    box[2, :] *= float(lengths[2])
    return box


def read_con(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    lengths = np.array([float(value) for value in lines[2].split()[:3]])
    angles = np.array([float(value) for value in lines[3].split()[:3]])
    n_types = int(lines[6].split()[0])
    counts = [int(value) for value in lines[7].split()[:n_types]]
    positions = []
    cursor = 9
    for count in counts:
        cursor += 2
        for _ in range(count):
            positions.append([float(value) for value in lines[cursor].split()[:3]])
            cursor += 1
    return np.asarray(positions, dtype=float), length_angle_to_box(lengths, angles)


def reference_mode(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    return np.asarray(values, dtype=float).reshape(-1, 3)


def _calculator():
    from ase.calculators.calculator import Calculator

    from optbench.pes import lennard_jones

    class _LennardJones(Calculator):
        implemented_properties = ("energy", "forces")
        nolabel = True

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
            if system_changes is None:
                system_changes = ["positions"]
            Calculator.calculate(self, atoms, properties, system_changes)
            energy, gradient = lennard_jones(np.asarray(atoms.get_positions(), dtype=float))
            self.results["energy"] = float(energy)
            self.results["forces"] = -np.asarray(gradient, dtype=float)

    return _LennardJones()


def _atoms(positions: np.ndarray, cell: np.ndarray):
    from ase import Atoms

    atoms = Atoms(
        symbols=["H"] * len(positions),
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(cell, dtype=float),
        pbc=True,
    )
    atoms.calc = _calculator()
    return atoms


def rotate(positions: np.ndarray, cell: np.ndarray, mode: np.ndarray, lowest: np.ndarray, rotation_max: int, *, stop: bool) -> tuple[int, float]:
    """One BFGS rotation. ``stop`` ends at the first overlap above 0.99."""
    import warnings
    from contextlib import redirect_stdout

    warnings.filterwarnings("ignore", category=FutureWarning)
    dimer_cls = _load_class()
    atoms = _atoms(positions, cell)
    dimer = dimer_cls(
        atoms,
        mode=np.array(mode, dtype=float, copy=True),
        rotationMax=int(rotation_max),
        phi_tol=0.0,
        ss=False,
        dR=0.001,
        rotationOpt="bfgs",
    )
    if stop:
        dimer._lowest = lowest
        dimer._overlap_tol = OVERLAP
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        dimer.get_forces()
    overlap = abs(float(np.vdot(dimer.get_mode(), lowest)))
    return int(dimer.forceCalls), overlap


def run_geometry(coords: Path, index: int, *, cap: int = 10000) -> dict:
    from .common import judge_mode

    positions, cell = read_con(coords / f"{index:04d}.con")
    lowest = reference_mode(coords / f"{index:04d}_mode")
    calls, overlap = rotate(positions, cell, initial_mode(len(positions)), lowest, ROTATION_LIMIT, stop=True)
    return judge_mode(f"{index:04d}", calls, cap, overlap, OVERLAP, "")


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return run_geometry(Path(job["coords"]), int(job["index"]), cap=int(job["cap"]))


def _collect(jobs: list[dict], workers: int) -> list[dict]:
    """Fork the pure-Python dimer. There is no native library to reset."""
    if workers <= 1 or len(jobs) <= 1:
        records = [_execute(job) for job in jobs]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            records = list(pool.map(_execute, jobs))
    order = {job["name"]: index for index, job in enumerate(jobs)}
    records.sort(key=lambda row: order[row["start"]])
    return records


def _extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as packed:
        packed.extractall(dest, filter="data")
    folder = dest / "coords-con"
    if not (folder / "0000.con").is_file():
        raise FileNotFoundError(f"{archive} has no coords-con/0000.con")
    return folder


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    from optbench.campaign import geometry_archive, load_template, resolve_problem

    from .common import registry_pin, write_campaign

    root = Path(root)
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    pin = registry_pin(root, CODE)
    if template.get("commit") != pin["commit"]:
        raise ValueError(f"{template_name} commit does not match the {CODE} registry")
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    cap = int((problem.get("convergence") or {}).get("max_force_calls") or 10000)
    archive = geometry_archive(root, problem)
    with tempfile.TemporaryDirectory(prefix="tsase-408-") as scratch:
        coords = _extract(archive, Path(scratch))
        jobs = [{"coords": str(coords), "index": index, "cap": cap, "name": f"{index:04d}"} for index in range(200)]
        records = _collect(jobs, workers)
    comment = (
        "# counter harness. One force call is one SSDimer.forceCalls increment at TSASE r408.\n"
        "# dimer_rotate.py: rotationOpt bfgs, phi_tol 0, dR 0.001, ss false, overlap above 0.99.\n"
        "# The initial mode is the shared seed 314159. The potential is unshifted lj/cut, cutoff 40.\n"
        "# report.sh prints the mean with printf %.0f and writes nfailed 0."
    )
    parameters = {
        "rotation_opt": "bfgs",
        "phi_tol": 0.0,
        "dr": 0.001,
        "ss": False,
        "overlap": OVERLAP,
        "rotation_limit": ROTATION_LIMIT,
        "seed": 314159,
        "potential": "lennard-jones",
        "cutoff": 40.0,
        "counter": "harness",
    }
    return write_campaign(
        output,
        records,
        template,
        problem,
        {"cap": cap},
        parameters,
        images=None,
        comment=comment,
    )
