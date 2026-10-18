"""Global optimization with BPGA and PyBCGA on the harness Lennard-Jones potential.

The potential is ``optbench.potentials.lennard_jones``: epsilon 1, sigma 1,
cutoff 40, no shift. That is the archived Lennard-Jones 38 basin-hopping
job (``pair_style lj/cut 40``). A force call is one ``HarnessCounter`` entry
during a local quench. The genetic operators are the pinned sources.

PyBCGA exposes a ``Minimiser``. The quench is L-BFGS-B on that potential.
Pele's own Lennard-Jones is not used. BPGA submits each structure through
``subString``. The harness answers that command from the POSCAR the code
just wrote and returns an OUTCAR the parser already reads.

A start is one archived geometry. It is the first member of the population.
The search stops when the best quenched energy is within the problem's
``energy_above_minimum`` of the Lennard-Jones 38 truncated octahedron, or
when the template force budget or the code's own iteration limit is spent.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from itertools import permutations, product
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

BPGA_COMMIT = "b32cd227e8071429825f990449048eb21c7f8730"
PYBCGA_COMMIT = "fb646c19b1829656bf16cc72ca318bef2d37bfeb"
BPGA_REPOSITORY = "https://github.com/jbadavis/bpga"
PYBCGA_REPOSITORY = "https://github.com/marktoakley/PyBCGA"
HARNESS_CMD = "optbench-harness-lj"
QUENCH_FORCE = 1e-3
QUENCH_MAXFUN = 1000
# Archived eon basin-hopping job for this problem: converged_force 0.001,
# max_iterations 1000. The local quench uses those two numbers.


def scheduled_pybcga_cap(pop_size: int, generations: int, offspring: int, maxfun: int) -> int:
    """Force calls if every scheduled cluster uses the whole quench budget.

    Each generation adds ``offspring`` children and, in the worst single
    pass, one mutant for every member of that expanded population. The
    initial population is quenched as well.
    """
    if pop_size <= 0 or generations < 0 or offspring < 0 or maxfun <= 0:
        raise ValueError("PyBCGA budget needs positive sizes")
    per_generation = offspring + pop_size + offspring
    return (pop_size + generations * per_generation) * maxfun


def scheduled_pool_cap(n_pool: int, iterations: int, maxfun: int) -> int:
    """Force calls for the pool fill plus the hardcoded offspring loop.

    ``NewPoolGA.poolGA.run`` fills ``n_pool`` structures and then takes
    1000 mutate-or-crossover steps. ``iterations`` is that loop count.
    """
    if n_pool <= 0 or iterations < 0 or maxfun <= 0:
        raise ValueError("pool budget needs positive sizes")
    return (n_pool + iterations) * maxfun


def lj38_truncated_octahedron() -> np.ndarray:
    """Ideal 38-atom fcc truncated octahedron, nearest neighbour 2**(1/6).

    The sites are the sign-and-axis copies of (2, 1, 0), (1, 1, 1), and
    (1, 0, 0), in units of 2**(-5/6). That is 24 + 8 + 6 atoms.
    """
    unit = 2.0 ** (-5.0 / 6.0)
    points: list[tuple[int, int, int]] = []
    for template in ((2, 1, 0), (1, 1, 1), (1, 0, 0)):
        for perm in set(permutations(template)):
            free = [axis for axis, value in enumerate(perm) if value]
            for signs in product((-1, 1), repeat=len(free)):
                point = list(perm)
                for axis, sign in zip(free, signs):
                    point[axis] *= sign
                points.append((point[0], point[1], point[2]))
    unique = list(dict.fromkeys(points))
    if len(unique) != 38:
        raise RuntimeError(f"truncated octahedron has {len(unique)} sites")
    return unit * np.asarray(unique, dtype=np.float64)


def lj38_reference_energy() -> tuple[float, np.ndarray]:
    """Relax the truncated octahedron on the spec potential.

    The returned energy is the reference the 0.001 rule is measured from.
    The relaxation is not a search call.
    """
    from scipy.optimize import minimize

    ideal = lj38_truncated_octahedron().reshape(-1)

    def fun(x: np.ndarray):
        energy, grad = lennard_jones(x)
        return float(energy), np.asarray(grad, dtype=np.float64)

    result = minimize(
        fun,
        ideal,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": 1e-12, "ftol": 1e-15, "maxfun": 8000, "maxiter": 4000},
    )
    energy, grad = fun(result.x)
    force = float(np.linalg.norm(grad))
    if not math.isfinite(energy) or force > 1e-6:
        raise RuntimeError(f"LJ38 reference did not relax: energy {energy} force {force}")
    return float(energy), np.asarray(result.x, dtype=np.float64).reshape(-1, 3)


def _threads_one() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


class _QuenchStop(Exception):
    """The Euclidean gradient is at the local-quench stop. The call is already counted."""


def _quench(coords: np.ndarray, counter: HarnessCounter, force_stop: float, maxfun: int) -> dict:
    """L-BFGS-B on the spec potential. One fresh coordinate is one harness call.

    The stop is the Euclidean norm of the gradient, the same quantity as
    the archived eon ``converged_force``. SciPy's ``gtol`` is a projected
    infinity norm, so it is held tight and the Euclidean check ends the step.
    """
    from scipy.optimize import minimize

    point = np.asarray(coords, dtype=np.float64).reshape(-1).copy()
    best = {"x": point.copy(), "energy": None, "force": None}
    stopped = {"hit": False}

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in flat)
        fresh, cached = counter.charge(key)
        if not fresh and cached is not None:
            energy, grad = cached
            grad = np.asarray(grad, dtype=np.float64)
        else:
            energy, grad = lennard_jones(flat)
            grad = np.asarray(grad, dtype=np.float64)
            counter.remember(key, (float(energy), tuple(float(value) for value in grad)))
        force = float(np.linalg.norm(grad))
        if best["energy"] is None or float(energy) < float(best["energy"]):
            best["x"] = flat.copy()
            best["energy"] = float(energy)
            best["force"] = force
        if force <= force_stop:
            best["x"] = flat.copy()
            best["energy"] = float(energy)
            best["force"] = force
            stopped["hit"] = True
            raise _QuenchStop()
        return float(energy), grad

    try:
        result = minimize(
            evaluate,
            point,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": int(maxfun),
                "maxiter": int(maxfun),
                "gtol": 1e-12,
                "ftol": 1e-15,
            },
        )
        final = np.asarray(result.x, dtype=np.float64)
    except _QuenchStop:
        final = best["x"]
    except ForceCallCap:
        final = best["x"]
        if best["energy"] is None:
            raise
    can_eval = counter.cap is None or counter.calls < counter.cap
    if stopped["hit"] or not can_eval:
        energy = float(best["energy"])
        force = float(best["force"]) if best["force"] is not None else math.inf
        final = best["x"]
    else:
        try:
            energy, grad = evaluate(final)
        except _QuenchStop:
            energy = float(best["energy"])
            force = float(best["force"]) if best["force"] is not None else math.inf
            final = best["x"]
        except ForceCallCap:
            if best["energy"] is None:
                raise
            energy = float(best["energy"])
            force = float(best["force"]) if best["force"] is not None else math.inf
            final = best["x"]
        else:
            force = float(np.linalg.norm(grad))
            energy = float(energy)
    return {
        "x": np.asarray(final, dtype=np.float64).reshape(-1, 3),
        "energy": energy,
        "force": force,
        "converged": math.isfinite(energy) and math.isfinite(force) and force <= force_stop,
    }


def _final_force(coords: np.ndarray, counter: HarnessCounter) -> tuple[float, float]:
    """Energy and force of a finished structure. Verification does not spend the cap."""
    flat = np.asarray(coords, dtype=np.float64).reshape(-1)
    with counter.verification():
        key = tuple(float(value) for value in flat)
        fresh, cached = counter.charge(key)
        if not fresh and cached is not None:
            energy, grad = cached
            grad = np.asarray(grad, dtype=np.float64)
        else:
            energy, grad = lennard_jones(flat)
            grad = np.asarray(grad, dtype=np.float64)
            counter.remember(key, (float(energy), tuple(float(value) for value in grad)))
    return float(energy), float(np.linalg.norm(grad))


def _judge(energy: float, force: float, calls: int, cap: int, target: float | None, force_stop: float) -> dict:
    """Classify one finished search.

    A missing energy target is not a success. The row stays partial.
    Spending the force budget is a cap hit when the energy is finite.
    A converged quench that misses the target is the wrong stationary point.
    """
    reasons: list[str] = []
    energy_ok = math.isfinite(energy)
    force_ok = math.isfinite(force)
    if not energy_ok:
        reasons.append("divergence")
    if calls >= cap:
        reasons.append("cap")
    elif not force_ok or force > force_stop:
        reasons.append("force_norm")
    elif target is None or energy > target:
        reasons.append("energy")
    failure = classify_failure(reasons)
    success = (
        failure is None
        and target is not None
        and calls < cap
        and energy_ok
        and force_ok
        and energy <= target
        and force <= force_stop
    )
    return {"success": success, "failure": failure, "reasons": reasons}


def _note_trace(trace: list[list[float]], calls: int, energy: float) -> None:
    if not math.isfinite(energy):
        return
    if not trace or energy < float(trace[-1][1]) - 1e-12:
        trace.append([int(calls), float(energy)])


def _ensure_checkout(repository: str, commit: str, dest: Path) -> Path:
    dest = dest.resolve()
    if (dest / ".git").is_dir():
        head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if head != commit:
            raise RuntimeError(f"{dest} is {head}, not {commit}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "--quiet", repository, str(dest)], check=True)
    head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)
        head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise RuntimeError(f"checkout {dest} is {head}, not {commit}")
    return dest


def pybcga_source(root: Path | None = None) -> Path:
    """Pinned PyBCGA tree. ``OPTBENCH_PYBCGA`` overrides the cache."""
    env = os.environ.get("OPTBENCH_PYBCGA")
    if env:
        return _ensure_checkout(PYBCGA_REPOSITORY, PYBCGA_COMMIT, Path(env))
    cache = Path(os.environ.get("OPTBENCH_SOURCE_CACHE", "/tmp/optbench-src"))
    return _ensure_checkout(PYBCGA_REPOSITORY, PYBCGA_COMMIT, cache / "PyBCGA")


def bpga_source() -> Path:
    env = os.environ.get("OPTBENCH_BPGA")
    cache = Path(os.environ.get("OPTBENCH_SOURCE_CACHE", "/tmp/optbench-src"))
    dest = Path(env) if env else cache / "bpga"
    return _ensure_checkout(BPGA_REPOSITORY, BPGA_COMMIT, dest)


def build_bpga(source: Path | None = None) -> Path:
    """Python 3 tree of the GA package. The pin stays the upstream commit.

    ``MinimiseRan`` calls ``Database.updatePool`` without the calculation
    number that ``MinimiseOff`` and ``MinimiseMut`` pass. The build inserts
    ``self.calcNum`` so a full-pool replacement does not raise TypeError.
    ``Database.addEle`` indexes with ``i/3``. On Python 3 that is a float,
    so the build uses integer division. Four lines that mix a space into a
    tab indent are rewritten as tabs at the same column, and three ``print``
    statements gain parentheses. No other source line is changed.
    """
    upstream = source or bpga_source()
    dest = Path(os.environ.get("OPTBENCH_BPGA_BUILD", f"/tmp/optbench-build/bpga-{BPGA_COMMIT[:12]}"))
    ga = dest / "GA"
    marker = dest / "COMMIT"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == BPGA_COMMIT and (ga / "NewPoolGA.py").is_file():
        return ga
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(upstream / "GA", ga)
    # The pin is the 2016 source. A few lines mix a space into a tab indent.
    # The column is the same as the surrounding tab indent (tab width 8), and
    # Python 3 rejects the mixture. Those lines are rewritten as tabs.
    # Three print statements are the remaining Python 2 syntax.
    indent_fixes = {
        "CoM.py": {" \tfor i in range(len(clus)):": "\tfor i in range(len(clus)):"},
        "Crossover.py": {
            " \t\t\tself.pair[i] = self.rotate(self.pair[i])": "\t\t\tself.pair[i] = self.rotate(self.pair[i])"
        },
        "DFT_output.py": {
            '   \t\tprint "Found the final energy"': '\t\tprint("Found the final energy")',
            "   \t\treturn energy[6]": "\t\treturn energy[6]",
        },
        "checkPool.py": {
            " \tdef checkEnergy(self,newEnergy):": "\tdef checkEnergy(self,newEnergy):",
            '\t\t\tprint "Calculation Converged"': '\t\t\tprint("Calculation Converged")',
        },
        "DFT_submit.py": {'    print "*Submitting Job*"': '    print("*Submitting Job*")'},
    }
    for name, mapping in indent_fixes.items():
        path = ga / name
        text = path.read_text(encoding="utf-8")
        for old, new in mapping.items():
            count = text.count(old)
            if count != 1:
                raise RuntimeError(f"{name} fix {old!r} matched {count} times")
            text = text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8")
    database = ga / "Database.py"
    text = database.read_text(encoding="utf-8")
    needle = "eleList[i/3]"
    if text.count(needle) != 1:
        raise RuntimeError("Database.addEle index was not the expected shape")
    # Python 3 would use a float index. The 2016 source meant integer division.
    database.write_text(text.replace(needle, "eleList[i//3]", 1), encoding="utf-8")
    ran = ga / "MinimiseRan.py"
    text = ran.read_text(encoding="utf-8")
    needle = ",Index,self.eleNums"
    if text.count(needle) != 1:
        raise RuntimeError("MinimiseRan updatePool call was not the expected shape")
    ran.write_text(text.replace(needle, ",self.calcNum,Index,self.eleNums", 1), encoding="utf-8")
    compiled = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(ga)],
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        raise RuntimeError(compiled.stderr or compiled.stdout or "BPGA failed to compile")
    marker.write_text(BPGA_COMMIT + "\n", encoding="utf-8")
    return ga


def _read_poscar(path: Path) -> tuple[np.ndarray, float]:
    """Cartesian coordinates and the cubic box length BPGA wrote."""
    lines = path.read_text(encoding="utf-8").splitlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(value) for value in lines[i].split()] for i in range(2, 5)], dtype=np.float64)
    direct = 0
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("direct"):
            direct = index
            break
    rows = []
    for line in lines[direct + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            break
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    frac = np.asarray(rows, dtype=np.float64)
    cart = frac @ (scale * lattice)
    cart -= scale / 2.0
    return cart, scale


def _write_outcar(path: Path, coords: np.ndarray, box: float, energy: float) -> None:
    """OUTCAR fragment ``vasp_output`` reads. Coordinates are cell-centred."""
    shifted = np.asarray(coords, dtype=np.float64) + box / 2.0
    lines = [
        f"  energy  without entropy=      {energy:.10f}  energy(sigma->0) =      {energy:.10f}\n",
        " POSITION                                       TOTAL-FORCE\n",
        " -----------------------------------------------------------------------------------\n",
    ]
    for x, y, z in shifted:
        lines.append(f"  {x:.10f}  {y:.10f}  {z:.10f}  0.0  0.0  0.0\n")
    lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def _search_pybcga(job: dict, counter: HarnessCounter, trace: list[list[float]]) -> tuple[float, np.ndarray]:
    source = str(job["pybcga_src"])
    if source not in sys.path:
        sys.path.insert(0, source)
    from bcga.cluster_factory import ClusterFactory
    from bcga.genetic_algorithm import GeneticAlgorithm
    from bcga.selector import TournamentSelector

    start = np.asarray(job["coords"], dtype=np.float64).reshape(-1, 3)
    natoms = int(start.shape[0])
    holder: dict = {}

    class _HarnessMinimiser:
        def minimise(self, cluster):
            quenched = _quench(cluster._coords, counter, float(job["quench_force"]), int(job["quench_maxfun"]))
            cluster._coords = quenched["x"]
            cluster.energy = float(quenched["energy"])
            cluster.quenched = True
            _note_trace(trace, counter.calls, cluster.energy)
            return cluster

    minimiser = _HarnessMinimiser()
    original = ClusterFactory.get_random_cluster
    placed = {"done": False}

    def _first_is_start(factory):
        cluster = original(factory)
        if not placed["done"]:
            placed["done"] = True
            cluster._coords = start.copy()
            cluster.energy = None
            cluster.quenched = False
        return cluster

    ClusterFactory.get_random_cluster = _first_is_start
    try:
        np.random.seed(int(job["seed"]))
        ga = GeneticAlgorithm(
            natoms,
            minimiser,
            pop_size=int(job["pop_size"]),
            max_generation=int(job["max_generation"]),
            offspring=int(job["offspring"]),
            mutant_rate=float(job["mutant_rate"]),
            selector=TournamentSelector(int(job["tournament"])),
            remove_duplicates=bool(job["remove_duplicates"]),
            mass_extinction=bool(job["mass_extinction"]),
        )
        holder["ga"] = ga
        target = job.get("target")
        energy = float(ga.population.get_lowest_energy())
        _note_trace(trace, counter.calls, energy)
        if target is not None and energy <= float(target):
            return energy, np.asarray(ga.population[0]._coords, dtype=np.float64)
        for _generation in range(1, int(job["max_generation"]) + 1):
            if counter.cap is not None and counter.calls >= counter.cap:
                break
            ga.make_offspring()
            ga.make_mutants()
            if ga.remove_duplicates:
                ga.population.remove_duplicates()
            ga.population.truncate()
            energy = float(ga.population.get_lowest_energy())
            ga.mean_energy_series.append(ga.population.get_mean_energy())
            ga.min_energy_series.append(energy)
            _note_trace(trace, counter.calls, energy)
            if ga.mass_extinction:
                diff = ga.mean_energy_series[-2] - ga.mean_energy_series[-1]
                if 0 < diff < ga.epoch_thresh:
                    ga.population.mass_extinction()
                    while len(ga.population) < ga.population.max_size:
                        ga.population.append(ga.factory.get_random_cluster())
                    energy = float(ga.population.get_lowest_energy())
                    _note_trace(trace, counter.calls, energy)
            ga.population.sort_energy()
            if target is not None and energy <= float(target):
                break
            if counter.cap is not None and counter.calls >= counter.cap:
                break
        return float(ga.population.get_lowest_energy()), np.asarray(ga.population[0]._coords, dtype=np.float64)
    finally:
        ClusterFactory.get_random_cluster = original


def _pool_best(path: Path) -> float | None:
    if not path.is_file():
        return None
    best = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Energy" not in line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            value = float(parts[2])
        except ValueError:
            continue
        if best is None or value < best:
            best = value
    return best


def _write_pool_member(path: Path, coords: np.ndarray, energy: float, symbol: str, directory: int) -> None:
    rows = [f"{len(coords)}\n", f"Energy = {energy:.12f} Dir = {directory}\n"]
    for x, y, z in np.asarray(coords, dtype=np.float64):
        rows.append(f"{symbol} {x:.12f} {y:.12f} {z:.12f}\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(rows)


def _search_pool(job: dict, counter: HarnessCounter, trace: list[list[float]], work: Path) -> tuple[float, np.ndarray]:
    ga_dir = str(job["bpga_ga"])
    if ga_dir not in sys.path:
        sys.path.insert(0, ga_dir)
    # These names are the upstream modules. They are imported from the build tree.
    import Database as database
    from Input import masses
    from MinimiseMut import minMut
    from MinimiseOff import minOff
    from MinimiseRan import minRan

    start = np.asarray(job["coords"], dtype=np.float64).reshape(-1, 3)
    natoms = int(start.shape[0])
    symbol = "Ni"
    names = [symbol]
    counts = [natoms]
    mass = masses(names)
    previous = os.getcwd()
    previous_pwd = os.environ.get("PWD")
    real_system = os.system
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    os.environ["PWD"] = str(work)

    def _system(command: str) -> int:
        text = str(command).strip()
        if text != HARNESS_CMD:
            # vasp_input copies POTCAR, INCAR, and KPOINTS with a bash brace.
            # The harness writes OUTCAR itself, so that copy is a no-op.
            if text.startswith("cp "):
                return 0
            return int(real_system(text))
        poscar = Path("POSCAR")
        coords, box = _read_poscar(poscar)
        quenched = _quench(coords, counter, float(job["quench_force"]), int(job["quench_maxfun"]))
        _write_outcar(Path("OUTCAR"), quenched["x"], box, float(quenched["energy"]))
        _note_trace(trace, counter.calls, float(quenched["energy"]))
        return 0

    os.system = _system
    best_coords = start.copy()
    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    try:
        seeded = _quench(start, counter, float(job["quench_force"]), int(job["quench_maxfun"]))
        best_coords = seeded["x"]
        _note_trace(trace, counter.calls, float(seeded["energy"]))
        pool_path = work / "pool.dat"
        if pool_path.exists():
            pool_path.unlink()
        _write_pool_member(pool_path, seeded["x"], float(seeded["energy"]), symbol, 0)
        target = job.get("target")
        if target is not None and float(seeded["energy"]) <= float(target):
            return float(seeded["energy"]), best_coords

        def _cleanup_dirs() -> None:
            for child in work.iterdir():
                if child.is_dir() and child.name.isdigit():
                    shutil.rmtree(child, ignore_errors=True)
            lock = work / "lock.db"
            if lock.exists():
                lock.unlink()

        kwargs = dict(
            natoms=natoms,
            r_ij=float(job["r_ij"]),
            eleNums=counts,
            eleNames=names,
            eleMasses=mass,
            nPool=int(job["n_pool"]),
            stride=natoms + 2,
            subString=HARNESS_CMD,
            boxAdd=float(job["box_add"]),
            surface=None,
            surfGA=False,
        )
        while _pool_size(pool_path, natoms) < int(job["n_pool"]):
            if counter.cap is not None and counter.calls >= counter.cap:
                break
            minRan(**kwargs)
            _cleanup_dirs()
            energy = _pool_best(pool_path)
            if energy is not None:
                _note_trace(trace, counter.calls, float(energy))
                if target is not None and energy <= float(target):
                    return float(energy), _coords_of_best(pool_path, natoms)
        mutate_rate = float(job["mutate"]) * int(job["n_pool"])
        import random

        for _step in range(int(job["iterations"])):
            if counter.cap is not None and counter.calls >= counter.cap:
                break
            choice = random.uniform(0, int(job["n_pool"]))
            if choice < mutate_rate:
                minMut(
                    natoms,
                    float(job["r_ij"]),
                    str(job["mut_type"]),
                    counts,
                    names,
                    mass,
                    int(job["n_pool"]),
                    natoms + 2,
                    HARNESS_CMD,
                    float(job["box_add"]),
                    None,
                    False,
                )
            else:
                minOff(
                    natoms,
                    counts,
                    names,
                    mass,
                    int(job["n_pool"]),
                    str(job["cross"]),
                    natoms + 2,
                    HARNESS_CMD,
                    float(job["box_add"]),
                    None,
                    False,
                )
            _cleanup_dirs()
            energy = _pool_best(pool_path)
            if energy is None:
                continue
            _note_trace(trace, counter.calls, float(energy))
            if target is not None and energy <= float(target):
                break
        energy = _pool_best(pool_path)
        if energy is None:
            raise RuntimeError("pool.dat has no energy")
        return float(energy), _coords_of_best(pool_path, natoms)
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        os.system = real_system
        os.chdir(previous)
        held = getattr(sys.modules.get("Database"), "fd", None)
        if isinstance(held, int):
            try:
                os.close(held)
            except OSError:
                pass
            sys.modules["Database"].fd = None
        stale_lock = work / "lock.db"
        if stale_lock.exists():
            stale_lock.unlink()
        if previous_pwd is None:
            os.environ.pop("PWD", None)
        else:
            os.environ["PWD"] = previous_pwd
        # Drop the upstream modules so a later method sees a clean import.
        for name in (
            "Database",
            "Input",
            "MinimiseMut",
            "MinimiseOff",
            "MinimiseRan",
            "DFT_input",
            "DFT_output",
            "checkPool",
            "CoM",
            "Explode",
            "fixOverlap",
            "Crossover",
            "Select",
            "SurfOpt",
            "surfacePOSCAR",
        ):
            sys.modules.pop(name, None)
        del database


def _pool_size(path: Path, natoms: int) -> int:
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stride = natoms + 2
    if stride <= 0:
        return 0
    return len(lines) // stride


def _coords_of_best(path: Path, natoms: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stride = natoms + 2
    best_energy = None
    best_block: list[str] | None = None
    for start in range(0, len(lines) - stride + 1, stride):
        block = lines[start : start + stride]
        parts = block[1].split()
        try:
            energy = float(parts[2])
        except (IndexError, ValueError):
            continue
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_block = block
    if best_block is None:
        raise RuntimeError("pool.dat has no coordinates")
    rows = []
    for line in best_block[2:]:
        parts = line.split()
        rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(rows, dtype=np.float64)


def search_job(job: dict) -> dict:
    """One start. Failures stay in the record."""
    _threads_one()
    started = time.perf_counter()
    cap = int(job["cap"])
    counter = HarnessCounter(cache=True, cap=cap)
    trace: list[list[float]] = []
    error = ""
    coords = np.asarray(job["coords"], dtype=np.float64).reshape(-1, 3)
    energy = math.nan
    force = math.nan
    try:
        work = Path(job["work"]) / f"start-{job['index']}"
        if work.exists():
            shutil.rmtree(work)
        if job["method"] == "pybcga":
            energy, coords = _search_pybcga(job, counter, trace)
        elif job["method"] == "pool":
            energy, coords = _search_pool(job, counter, trace, work)
        else:
            raise ValueError(f"unknown method {job['method']}")
        energy, force = _final_force(coords, counter)
        _note_trace(trace, counter.calls, energy)
    except ForceCallCap:
        error = ""
        if trace:
            energy = float(trace[-1][1])
    except Exception as exc:
        error = exc.__class__.__name__
        traceback.print_exc()
        if trace:
            energy = float(trace[-1][1])
    calls = int(counter.calls)
    if calls > cap:
        calls = cap
    target = job.get("target")
    target_value = None if target is None else float(target)
    if error:
        judged = {"success": False, "failure": "crash", "reasons": ["crash"]}
    else:
        if not math.isfinite(force):
            force = math.inf
        if not math.isfinite(energy):
            energy = math.nan
        judged = _judge(
            float(energy) if math.isfinite(energy) else math.nan,
            float(force),
            calls,
            cap,
            target_value,
            float(job["quench_force"]),
        )
    calls_to_best = int(trace[-1][0]) if trace else calls
    return {
        "schema": "optbench.start/v2",
        "index": int(job["index"]),
        "start": str(job["start"]),
        "problem": str(job["problem"]),
        "template": str(job["template"]),
        "force_calls": calls,
        "verify_calls": int(counter.verify_calls),
        "calls_to_best": calls_to_best,
        "counter": "harness",
        "comparable": True,
        "success": bool(judged["success"]),
        "failure": judged["failure"],
        "reasons": judged["reasons"],
        "energy": None if not math.isfinite(energy) else float(energy),
        "force_norm": None if not math.isfinite(force) else float(force),
        "trace": trace,
        "seconds": time.perf_counter() - started,
        "error": error,
        "code": str(job["code"]),
        "code_version": str(job["code_version"]),
        "commit": str(job["commit"]),
        "tuning": str(job["tuning"]),
        "cap": cap,
        "target": target_value,
        "reference_energy": job.get("reference_energy"),
    }


def _section(template: dict) -> dict:
    section = template.get("bpga") or {}
    if "method" not in section:
        raise ValueError(f"{template['id']} needs bpga.method")
    return section


def _target(problem: dict, reference: float | None) -> float | None:
    convergence = problem.get("convergence") or {}
    if "energy_above_minimum" not in convergence:
        return None
    if reference is None:
        raise ValueError(f"{problem['id']} states an energy window and has no reference energy")
    return float(reference) + float(convergence["energy_above_minimum"])


def _reference_for(problem: dict) -> float | None:
    convergence = problem.get("convergence") or {}
    for key in ("reference_energy", "minimum_energy", "energy_minimum", "known_minimum"):
        if key in convergence:
            return float(convergence[key])
    if "energy_above_minimum" not in convergence:
        return None
    if problem["id"] == "globalopt/lj38":
        energy, _coords = lj38_reference_energy()
        return energy
    return None


def _load_starts(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path) -> list[tuple[str, np.ndarray]]:
    if geometries is not None:
        paths = sorted(Path(geometries).rglob("*.con")) + sorted(Path(geometries).rglob("coords.*"))
    else:
        from optbench.campaign import geometry_archive

        archive = None
        if (problem.get("input") or {}).get("archive_path"):
            archive = geometry_archive(root, problem)
        elif problem["id"] == "globalopt/lj75":
            archive = _manifest_archive(root, "website/globalopt/lj75/lj75_clusters.tgz")
        else:
            raise ValueError(f"{problem['id']} has no geometry archive")
        dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".extracted"
        recorded = archive.name
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != recorded:
            if any(dest.iterdir()):
                for child in dest.iterdir():
                    if child.name != ".extracted":
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
            import tarfile

            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text(recorded + "\n", encoding="utf-8")
        paths = [path for path in dest.rglob("*") if path.is_file() and path.name != ".extracted"]
    loaded = []
    for path in paths:
        if path.suffix == ".con" or path.name.endswith(".con"):
            pos, _box = read_con(path.read_text(encoding="utf-8", errors="replace"))
            loaded.append((path.stem, pos))
        elif path.name.startswith("coords."):
            rows = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if not rows:
                continue
            loaded.append((path.name, np.asarray(rows, dtype=np.float64)))
    def sort_key(item: tuple[str, np.ndarray]) -> tuple:
        name = item[0]
        token = name.split(".")[-1]
        try:
            return (0, int(token))
        except ValueError:
            return (1, name)

    loaded.sort(key=sort_key)
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected not in (None, "", 0, "0") and len(loaded) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(loaded)}")
    if not loaded:
        raise FileNotFoundError(f"{problem['id']} has no starts")
    return loaded


def _manifest_archive(root: Path, archive_path: str) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(f"{archive_path} is not in the geometry manifest")


def _job_from_template(template: dict, problem: dict, item: tuple[int, str, np.ndarray], reference: float | None, paths: dict) -> dict:
    section = _section(template)
    method = str(section["method"])
    cap = int(template["failure_cap"])
    quench_force = float(section.get("quench_force", QUENCH_FORCE))
    quench_maxfun = int(section.get("quench_maxfun", QUENCH_MAXFUN))
    seed_base = int(section.get("seed", 20261015))
    index, name, coords = item
    job = {
        "index": index,
        "start": name,
        "coords": coords.reshape(-1).tolist(),
        "method": method,
        "cap": cap,
        "target": _target(problem, reference),
        "reference_energy": reference,
        "quench_force": quench_force,
        "quench_maxfun": quench_maxfun,
        "seed": seed_base + index,
        "code": str(template["code"]),
        "code_version": str(template["code_version"]),
        "commit": str(template["commit"]),
        "tuning": str(template.get("tuning") or ""),
        "template": str(template["id"]),
        "problem": str(problem["id"]),
        "algorithm": str(template.get("algorithm") or method),
        "work": str(paths["work"]),
        "pybcga_src": str(paths.get("pybcga") or ""),
        "bpga_ga": str(paths.get("bpga_ga") or ""),
        "pop_size": int(section.get("pop_size", 10)),
        "max_generation": int(section.get("max_generation", 10)),
        "offspring": int(section.get("offspring", 8)),
        "mutant_rate": float(section.get("mutant_rate", 0.1)),
        "tournament": int(section.get("tournament", 3)),
        "remove_duplicates": bool(section.get("remove_duplicates", False)),
        "mass_extinction": bool(section.get("mass_extinction", False)),
        "n_pool": int(section.get("n_pool", 10)),
        "iterations": int(section.get("iterations", 1000)),
        "mutate": float(section.get("mutate", 0.1)),
        "mut_type": str(section.get("mut_type", "random")),
        "cross": str(section.get("cross", "weighted")),
        "r_ij": float(section.get("r_ij", 3.0)),
        "box_add": float(section.get("box_add", 10.0)),
    }
    if method == "pybcga":
        scheduled = scheduled_pybcga_cap(job["pop_size"], job["max_generation"], job["offspring"], job["quench_maxfun"])
        if cap > scheduled:
            raise ValueError(f"{template['id']} failure_cap {cap} exceeds the generation budget {scheduled}")
    elif method == "pool":
        scheduled = scheduled_pool_cap(job["n_pool"], job["iterations"], job["quench_maxfun"])
        if cap != scheduled:
            raise ValueError(f"{template['id']} failure_cap {cap} is not the pool budget {scheduled}")
    else:
        raise ValueError(f"{template['id']} method {method} is not pybcga or pool")
    return job


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, partial: bool) -> str:
    lines = [
        "# counter harness. One force call is one L-BFGS-B evaluation of optbench.potentials.lennard_jones.",
        "# The potential is epsilon 1, sigma 1, cutoff 40, no shift.",
        "# A success is a quenched energy within energy_above_minimum of the reference, under the force cap.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "force_calls_median": summary.get("median"),
        "nfailed": summary.get("nfailed"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "cap": template.get("failure_cap"),
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "reference_energy": summary.get("reference_energy"),
    }
    if partial:
        rows["partial"] = "true"
    body = list(lines)
    for key, value in rows.items():
        if value is None:
            continue
        body.append(f"{key} {value}")
    return "\n".join(body) + "\n"


def _partial_path(output: Path) -> Path:
    return output.with_name(output.stem + ".partial.jsonl")


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None = None,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Run every start and write ``benchmark.dat``, the jsonl, and the v2 document."""
    _threads_one()
    reference = _reference_for(problem)
    partial = _target(problem, reference) is None or limit > 0
    loaded = _load_starts(root, problem, geometries, work, output)
    if limit > 0:
        loaded = loaded[:limit]
    paths = {"work": str((work or output.parent / "work" / template["id"].replace("/", "-")).resolve())}
    Path(paths["work"]).mkdir(parents=True, exist_ok=True)
    method = str(_section(template)["method"])
    if method == "pybcga":
        paths["pybcga"] = str(pybcga_source())
    elif method == "pool":
        paths["bpga_ga"] = str(build_bpga())
    jobs = [
        _job_from_template(template, problem, (index, name, coords), reference, paths)
        for index, (name, coords) in enumerate(loaded)
    ]
    done: dict[str, dict] = {}
    partial_file = _partial_path(output)
    if resume and partial_file.is_file():
        for line in partial_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["start"])] = row
    pending = [job for job in jobs if job["start"] not in done]
    records = [done[job["start"]] for job in jobs if job["start"] in done]

    def _store(row: dict) -> None:
        partial_file.parent.mkdir(parents=True, exist_ok=True)
        with partial_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        records.append(row)

    if pending:
        if workers <= 1:
            for index, job in enumerate(pending, 1):
                row = search_job(job)
                _store(row)
                print(
                    f"{template['id']} {row['start']} calls={row['force_calls']} "
                    f"energy={row['energy']} success={row['success']} ({index}/{len(pending)})",
                    flush=True,
                )
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(search_job, job) for job in pending]
                total = len(futures)
                for index, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    _store(row)
                    print(
                        f"{template['id']} {row['start']} calls={row['force_calls']} "
                        f"energy={row['energy']} success={row['success']} ({index}/{total})",
                        flush=True,
                    )
    by_start = {str(row["start"]): row for row in records}
    ordered = [by_start[job["start"]] for job in jobs]
    for row in ordered:
        row.pop("index", None)
    return _write_result(output, template, problem, ordered, reference, partial)


def _write_result(output: Path, template: dict, problem: dict, records: list[dict], reference: float | None, partial: bool) -> dict:
    cap = int(template["failure_cap"])
    legacy_counts = [cap if not row["success"] else int(row["force_calls"]) for row in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary["median"] = float(np.median(successes)) if successes else None
    energies = [float(row["energy"]) for row in records if row.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    summary["reference_energy"] = reference
    metrics = score_v2(
        [
            {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
            for row in records
        ],
        cap,
    )
    version = str(template["code_version"])
    commit = str(template["commit"])
    provenance = collect_provenance(
        code=str(template["code"]),
        code_version=version,
        commit=commit,
        parameters={key: value for key, value in (template.get("bpga") or {}).items()},
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "sota",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "reference_energy": reference,
        "note": str(template.get("note") or ""),
        "runs": [
            {
                "entry": output.stem,
                "code": template["code"],
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": template.get("bpga") or {},
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "force_calls": summary.get("force_calls"),
                    "force_calls_min": summary.get("force_calls_min"),
                    "force_calls_max": summary.get("force_calls_max"),
                    "nfailed": summary.get("nfailed"),
                    "median": summary.get("median"),
                },
                "starts": records,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, partial), encoding="utf-8")
    public = []
    for row in records:
        copied = dict(row)
        copied.pop("trace", None)
        public.append(copied)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["successes"] = metrics["successes"]
    return summary
