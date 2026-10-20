"""Global optimization with AGOX on the harness Lennard-Jones potential.

One force call is one ASE ``calculate`` on the harness calculator. ASE
skips ``calculate`` when the geometry is unchanged, so an energy read and
a force read of one geometry are one call. The cap raises before the
potential runs, and that refused call is not counted.

The local step is SciPy L-BFGS-B stopped on the ASE maximum atomic force.
AGOX ships ASE BFGS for this step. Both use the same force threshold.
L-BFGS-B is the optimizer whose calls are counted here.

A start succeeds when the problem states ``energy_above_minimum`` and the
lowest quenched energy is within that window, with the harness count
strictly under the cap. The count is the total at the end of the iteration
that first entered the window. With no window the start is not a success.
The record keeps the lowest energy and the call count at that energy.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import statistics
import tarfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ase\.")

from optbench.counter import HarnessCounter, attach_ase_counter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

_LJ75_ARCHIVE = "website/globalopt/lj75/lj75_clusters.tgz"
_METHODS = ("basin_hopping", "rss", "ga", "gofee")
_NUMBER = re.compile(r"\d+")


class _OptStop(Exception):
    """The local optimizer hit its force or step limit."""


def _is_cap(exc: BaseException | None) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if "force-call cap" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def reference_energy(root: Path, problem: dict, n_atoms: int) -> tuple[float | None, str]:
    """Problem-stated minimum, or the Cambridge coordinates on this potential."""
    convergence = problem.get("convergence") or {}
    for key in ("minimum_energy", "known_minimum", "reference_energy"):
        if key in convergence and convergence[key] not in (None, ""):
            return float(convergence[key]), "problem"
    stated = (problem.get("reference") or {}).get("energy")
    if stated not in (None, ""):
        return float(stated), "problem"
    path = root / "optbench" / "data" / f"lj{n_atoms}-minimum.txt"
    if not path.is_file():
        return None, "missing"
    rows = _xyz_rows(path.read_text(encoding="utf-8"))
    if len(rows) != n_atoms:
        raise ValueError(f"{path} has {len(rows)} rows, expected {n_atoms}")
    energy, _gradient = lennard_jones(rows)
    return float(energy), "cambridge"


def target_energy(problem: dict, reference: float | None) -> float | None:
    window = (problem.get("convergence") or {}).get("energy_above_minimum")
    if window in (None, "") or reference is None:
        return None
    return float(reference) + float(window)


def _xyz_rows(text: str):
    import numpy as np

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    if not rows:
        raise ValueError("no coordinates")
    return np.asarray(rows, dtype=np.float64)


def _positions(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    if "Coordinates of Component" in text:
        pos, _box = read_con(text)
        return pos
    return _xyz_rows(text)


def _file_key(path: Path) -> tuple:
    return tuple(int(piece) for piece in _NUMBER.findall(path.name))


def _select_geometry_files(files: list[Path]) -> list[Path]:
    """One file per start. A paired ``.con`` wins over ``.xyz``."""
    grouped: dict[tuple, list[Path]] = {}
    for path in files:
        grouped.setdefault(_file_key(path), []).append(path)
    chosen = []
    for key in sorted(grouped):
        group = grouped[key]
        cons = [path for path in group if path.suffix == ".con"]
        chosen.append(sorted(cons or group, key=lambda path: path.name)[0])
    return chosen


def _archive(root: Path, problem: dict) -> Path:
    from optbench.campaign import geometry_archive

    try:
        return geometry_archive(root, problem)
    except ValueError:
        if problem["id"] != "globalopt/lj75":
            raise
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == _LJ75_ARCHIVE:
            return root / row["stored"]
    raise FileNotFoundError(f"{_LJ75_ARCHIVE} is not in the geometry manifest")


def _load_geometries(root: Path, problem: dict, geometries: Path | None, work: Path) -> list[tuple[str, object]]:
    if geometries is None:
        archive = _archive(root, problem)
        dest = work / "geometries" / problem["id"].replace("/", "-")
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".extracted"
        recorded = archive.name
        if marker.is_file():
            previous = marker.read_text(encoding="utf-8").strip()
            if previous != recorded:
                raise RuntimeError(f"{dest} was extracted from {previous}, not {recorded}")
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text(recorded, encoding="utf-8")
        files = _select_geometry_files(
            [path for path in dest.rglob("*") if path.is_file() and path.name != ".extracted"]
        )
        if not files:
            raise FileNotFoundError(f"{archive} has no geometries")
    else:
        from optbench.campaign import iter_geometries

        files = list(iter_geometries(geometries))
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected is not None and len(files) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(files)}")
    loaded = []
    for path in files:
        pos = _positions(path)
        loaded.append((path.name, pos))
    return loaded


def _section(template: dict) -> dict:
    section = dict(template.get("agox") or {})
    method = str(section.get("method") or "")
    if method not in _METHODS:
        raise ValueError(f"{template['id']} agox.method must be one of {', '.join(_METHODS)}")
    return section


def _registry_row(root: Path) -> dict:
    import tomllib

    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data.get("code") or []:
        if row.get("id") == "agox":
            return row
    raise RuntimeError("codes/registry.toml has no agox entry")


class LennardJones:
    """ASE calculator for the harness cutoff-40 Lennard-Jones potential.

    The class is built without subclassing ``Calculator`` at import, so a
    default test run does not need ASE. ``bind`` returns the instance.
    """

    def __new__(cls):
        from ase.calculators.calculator import Calculator, all_changes

        class _LennardJones(Calculator):
            implemented_properties = ["energy", "forces", "free_energy"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                import numpy as np

                energy, gradient = lennard_jones(self.atoms.positions)
                self.results["energy"] = float(energy)
                self.results["free_energy"] = float(energy)
                self.results["forces"] = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
                self.last_energy = float(energy)
                self.last_positions = np.array(self.atoms.positions, copy=True)

        return _LennardJones()


class FastLBFGS:
    """SciPy L-BFGS-B with the ASE optimizer observer protocol.

    ``run`` calls observers at step 0 so AGOX can record ``E_initial``.
    The step stops when the harness maximum atomic force is at ``fmax`` or
    ``steps`` evaluations have run. Bounds come from an AGOX box constraint
    on the atoms. While the peak atomic force is above 1000, and for at
    most 40 steps, each atom moves at most 0.2 along that force. Without
    that cap, L-BFGS-B puts an overlapped cluster on the confinement wall.
    The stop uses the harness force. The box constraint removes the outward
    component, and a dissociated cluster on the wall then looks converged.
    If L-BFGS-B returns before that stop, at most 25 steps of size
    min(0.05, 5/peak) follow while the peak is above 30, and L-BFGS-B
    starts again. That repeat happens at most four times.
    """

    def __new__(cls, atoms, logfile=None, trajectory=None, **kwargs):
        from ase.optimize.optimize import Optimizer

        class _FastLBFGS(Optimizer):
            def step(self):
                raise RuntimeError("FastLBFGS evaluates inside run")

            def run(self, fmax=0.05, steps=500):
                import numpy as np
                from scipy.optimize import minimize

                self.fmax = float(fmax)
                opt = self.optimizable
                self.nsteps = 0
                self.call_observers()
                x0 = np.array(opt.get_x(), dtype=np.float64, copy=True)
                bounds = _box_bounds(self.atoms)
                lo = hi = None
                if bounds is not None:
                    lo = np.array([pair[0] for pair in bounds], dtype=np.float64)
                    hi = np.array([pair[1] for pair in bounds], dtype=np.float64)
                    x0 = np.minimum(np.maximum(x0, lo), hi)
                    opt.set_x(x0)

                def harness_forces():
                    forces = np.asarray(self.atoms.calc.results["forces"], dtype=np.float64).reshape(-1, 3)
                    return forces

                self._peak = 0.0

                def evaluate(x):
                    opt.set_x(x)
                    energy = float(opt.get_value())
                    forces = harness_forces()
                    # The projected gradient is what L-BFGS-B line search
                    # expects. The stop below uses the harness force: on the
                    # wall the projected force is almost zero.
                    gradient = np.asarray(opt.get_gradient(), dtype=np.float64)
                    self.nsteps += 1
                    self.call_observers()
                    self._peak = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
                    if self.nsteps >= int(steps) or self._peak < self.fmax:
                        raise _OptStop()
                    return energy, gradient

                # L-BFGS-B takes the box bound on the first step when the
                # force is huge. The atoms then sit on the confinement wall
                # and stop interacting. Walk at most 0.2 along the force
                # until the peak atomic force is 1000, and at most 40 such
                # steps. A near-minimum structure leaves this loop at once.
                x = x0
                walked = 0
                try:
                    while self.nsteps < int(steps):
                        _energy, gradient = evaluate(x)
                        atomic = np.linalg.norm(gradient.reshape(-1, 3), axis=1)
                        walked += 1
                        if self._peak <= 1000.0 or walked >= 40:
                            break
                        direction = gradient.reshape(-1, 3) / np.maximum(atomic, 1e-30)[:, None]
                        x = x - (0.2 * direction).reshape(-1)
                        if lo is not None:
                            x = np.minimum(np.maximum(x, lo), hi)
                    else:
                        raise _OptStop()

                    def fun(x):
                        return evaluate(x)

                    options = {
                        "maxiter": int(steps),
                        "maxfun": int(steps),
                        "ftol": 1e-18,
                        "gtol": 1e-14,
                    }
                    for _attempt in range(4):
                        if self.nsteps >= int(steps):
                            break
                        minimize(
                            fun,
                            np.array(opt.get_x(), dtype=np.float64, copy=True),
                            method="L-BFGS-B",
                            jac=True,
                            bounds=bounds,
                            options=options,
                        )
                        if self._peak < self.fmax:
                            break
                        # A failed line search reports ftol convergence and
                        # leaves the overlapped geometry where it was. A
                        # short steepest step keeps the next guess off the wall.
                        x = np.array(opt.get_x(), dtype=np.float64, copy=True)
                        for _fine in range(25):
                            if self.nsteps >= int(steps) or self._peak <= 30.0:
                                break
                            _energy, gradient = evaluate(x)
                            atomic = np.linalg.norm(gradient.reshape(-1, 3), axis=1)
                            step = min(0.05, 5.0 / max(self._peak, 1e-30))
                            direction = gradient.reshape(-1, 3) / np.maximum(atomic, 1e-30)[:, None]
                            x = x - (step * direction).reshape(-1)
                            if lo is not None:
                                x = np.minimum(np.maximum(x, lo), hi)
                except _OptStop:
                    pass
                return self._peak < self.fmax

        return _FastLBFGS(atoms, logfile=logfile, trajectory=trajectory, **kwargs)


def _box_bounds(atoms):
    import numpy as np

    for constraint in list(getattr(atoms, "constraints", []) or []):
        cell = getattr(constraint, "cell", None)
        corner = getattr(constraint, "corner", None)
        if cell is None or corner is None:
            continue
        cell = np.asarray(cell, dtype=np.float64)
        corner = np.asarray(corner, dtype=np.float64).reshape(3)
        lengths = np.diag(cell) if cell.shape == (3, 3) else cell.reshape(3)
        lo = np.tile(corner + 1e-8, len(atoms))
        hi = np.tile(corner + lengths - 1e-8, len(atoms))
        return list(zip(lo.tolist(), hi.tolist()))
    return None


def _place(coords, cell_side: float, confinement: float):
    import numpy as np

    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3).copy()
    pos -= pos.mean(axis=0)
    span = float((pos.max(axis=0) - pos.min(axis=0)).max()) if len(pos) else 0.0
    side = max(float(confinement), span + 6.0)
    cell = max(float(cell_side), side + 8.0)
    pos += cell / 2.0
    corner = (cell - side) / 2.0
    pos = np.clip(pos, corner + 1e-6, corner + side - 1e-6)
    return pos, cell, side, corner


def _seed_candidate(environment, atoms):
    from uuid import uuid4

    from agox.candidates import StandardCandidate

    candidate = StandardCandidate.from_atoms(environment.get_template(), atoms)
    uid = str(uuid4())
    candidate.add_meta_information("uuid", uid)
    candidate.add_meta_information("spawn_uuids", [uid])
    candidate.add_meta_information("generator", "start")
    return candidate


def _evaluator(calc, environment, fmax: float, steps: int, *, gets: str, number: int, order: int):
    from agox.evaluators import LocalOptimizationEvaluator

    return LocalOptimizationEvaluator(
        calc,
        optimizer=FastLBFGS,
        optimizer_kwargs={"logfile": None},
        optimizer_run_kwargs={"fmax": float(fmax), "steps": int(steps)},
        store_trajectory=False,
        order=order,
        constraints=environment.get_constraints(),
        gets={"get_key": gets},
        number_to_evaluate=int(number),
    )


class _SilentRayPool:
    """Stand-in for AGOX's Ray pool.

    ``GPR`` constructs ``RayPoolUser`` before it reads ``use_ray``. That
    constructor starts a local Ray cluster. The harness search stays in
    this process, so the pool's attach and update methods do nothing.
    """

    def __init__(self) -> None:
        self.attached_handlers: list = []

    def attach(self, handler) -> None:
        return None

    def print_modules(self) -> None:
        return None

    def update_module_interconnections(self) -> None:
        return None

    def update_modules(self) -> None:
        return None

    def add_module(self, module) -> int:
        return 0


def _keep_gofee_in_process() -> None:
    """Stop GOFEE from starting Ray, and accept a one-element likelihood."""
    import agox.utils.ray.pool_user as pool_user

    def get_ray_pool(*args, **kwargs):
        return _SilentRayPool(), {"CPU": 1}

    pool_user.get_ray_pool = get_ray_pool
    from agox.models.GPR.GPR import GPR

    original = GPR._log_marginal_likelihood_gradient

    def likelihood(self, theta):
        import numpy as np

        log_p, gradient = original(self, theta)
        log_p = np.asarray(log_p, dtype=np.float64).reshape(-1)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
        if log_p.size == 1:
            log_p = float(log_p[0])
        return log_p, gradient

    GPR._log_marginal_likelihood_gradient = likelihood


def _build_search(method: str, environment, calc, database, seed_candidate, settings: dict, seed: int):
    import numpy as np

    from agox import AGOX

    np.random.seed(int(seed))
    fmax = float(settings["fmax"])
    steps = int(settings["max_steps"])
    amplitude = float(settings["rattle_amplitude"])
    fraction = float(settings["rattle_fraction"])
    n_atoms = len(seed_candidate)
    n_rattle = settings.get("n_rattle")
    if n_rattle in (None, ""):
        n_rattle = max(1, int(n_atoms * fraction))
    else:
        n_rattle = int(n_rattle)
    confinement = environment.get_confinement()

    if method == "basin_hopping":
        from agox.databases import Database
        from agox.generators import RattleGenerator
        from agox.samplers import MetropolisSampler

        assert isinstance(database, Database)
        sampler = MetropolisSampler(temperature=float(settings["temperature"]), order=3)
        sampler.sample = [seed_candidate]
        generator = RattleGenerator(
            **confinement,
            environment=environment,
            sampler=sampler,
            n_rattle=n_rattle,
            rattle_amplitude=amplitude,
            order=1,
        )
        evaluator = _evaluator(calc, environment, fmax, steps, gets="candidates", number=1, order=2)
        return AGOX(generator, database, sampler, evaluator, seed=int(seed), report=False)

    if method == "rss":
        from agox.generators import RandomGenerator

        generator = RandomGenerator(**confinement, environment=environment, order=1)
        evaluator = _evaluator(calc, environment, fmax, steps, gets="candidates", number=1, order=2)
        return AGOX(generator, database, evaluator, seed=int(seed), report=False)

    if method == "ga":
        from agox.collectors import StandardCollector
        from agox.generators import RandomGenerator, RattleGenerator
        from agox.models.descriptors.simple_fingerprint import SimpleFingerprint
        from agox.samplers import DistanceComparator, GeneticSampler

        population = int(settings["population"])
        descriptor = SimpleFingerprint(environment=environment)
        comparator = DistanceComparator(descriptor, threshold=float(settings["comparator_threshold"]))
        sampler = GeneticSampler(
            population_size=population,
            comparator=comparator,
            order=4,
            database=database,
        )
        sampler.population.append(seed_candidate)
        generators = [
            RandomGenerator(**confinement),
            RattleGenerator(**confinement, n_rattle=n_rattle, rattle_amplitude=amplitude),
        ]
        # The published schedule draws ``population`` random structures at
        # iteration 0. One archived member is already in the population, so
        # the first draw is one shorter and the population stays at that size.
        num_candidates = {
            0: [population - 1, 0],
            5: [min(2, population), max(population - 2, 0)],
            10: [0, population],
        }
        collector = StandardCollector(
            generators=generators,
            sampler=sampler,
            environment=environment,
            num_candidates=num_candidates,
            order=1,
        )
        evaluator = _evaluator(
            calc, environment, fmax, steps, gets="candidates", number=population, order=2
        )
        return AGOX(collector, database, evaluator, seed=int(seed), report=False)

    if method == "gofee":
        _keep_gofee_in_process()
        from agox.acquisitors import LowerConfidenceBoundAcquisitor
        from agox.collectors import StandardCollector
        from agox.generators import RandomGenerator, RattleGenerator
        from agox.models.GPR import GPR
        from agox.models.GPR.kernels import RBF, Noise
        from agox.models.GPR.kernels import Constant as C
        from agox.models.GPR.priors import Repulsive
        from agox.models.descriptors.fingerprint import Fingerprint
        from agox.postprocessors import RelaxPostprocess
        from agox.samplers import KMeansSampler

        descriptor = Fingerprint(environment=environment)
        beta = 0.01
        k0 = C(beta, (beta, beta)) * RBF()
        k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
        kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))
        model = GPR(
            descriptor=descriptor,
            kernel=kernel,
            database=database,
            prior=Repulsive(),
            use_ray=False,
        )
        sample_size = int(settings.get("sample_size") or 10)
        sampler = KMeansSampler(descriptor=descriptor, database=database, sample_size=sample_size)
        generators = [
            RandomGenerator(**confinement),
            RattleGenerator(**confinement, n_rattle=n_rattle, rattle_amplitude=amplitude),
        ]
        num_candidates = {0: [sample_size, 0], 5: [3, 7]}
        collector = StandardCollector(
            generators=generators,
            sampler=sampler,
            environment=environment,
            num_candidates=num_candidates,
            order=1,
        )
        acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=2, order=3)
        relaxer = RelaxPostprocess(
            model=acquisitor.get_acquisition_calculator(),
            constraints=environment.get_constraints(),
            optimizer=FastLBFGS,
            optimizer_kwargs={"logfile": None},
            optimizer_run_kwargs={"fmax": fmax, "steps": int(settings["surrogate_steps"])},
            start_relax=int(settings["surrogate_start"]),
            order=2,
        )
        evaluator = _evaluator(calc, environment, fmax, steps, gets="prioritized_candidates", number=1, order=4)
        return AGOX(collector, relaxer, acquisitor, evaluator, database, seed=int(seed), report=False)

    raise ValueError(f"unknown method {method}")


def _restore_last(atoms) -> None:
    positions = getattr(atoms.calc, "last_positions", None)
    if positions is not None:
        atoms.set_positions(positions)


def _quench(atoms, fmax: float, steps: int) -> float | None:
    opt = FastLBFGS(atoms, logfile=None)
    try:
        opt.run(fmax=fmax, steps=steps)
    except _OptStop:
        pass
    except Exception as exc:
        _restore_last(atoms)
        if not _is_cap(exc):
            raise
    else:
        _restore_last(atoms)
    energy = getattr(atoms.calc, "last_energy", None)
    return None if energy is None else float(energy)


def search_job(job: dict) -> dict:
    """Run one archived start. The return value is one v2 start record."""
    import numpy as np
    from ase import Atoms

    index = int(job["index"])
    cap = int(job["cap"])
    seed = int(job["seed"])
    method = str(job["method"])
    settings = job["settings"]
    target = None if job.get("target") is None else float(job["target"])
    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / "status.txt"
    log_path = work / "search.log"
    previous = Path.cwd()
    error = ""
    idle_stop = False
    iterations = 0
    best = math.inf
    calls_at = 0
    counter: HarnessCounter | None = None
    started = time.perf_counter()

    def consider(energy, calls: int) -> None:
        nonlocal best, calls_at
        if not _finite(energy):
            return
        if float(energy) < best:
            best = float(energy)
            calls_at = int(calls)

    def status(line: str) -> None:
        status_path.write_text(line + "\n", encoding="utf-8")

    try:
        os.chdir(work)
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            import matplotlib

            matplotlib.use("Agg", force=True)
            from agox.databases import Database
            from agox.environments import Environment
            from agox.main.state import State

            coords = np.asarray(job["coords"], dtype=np.float64)
            pos, cell_side, side, corner = _place(
                coords, float(settings["cell"]), float(settings["confinement"])
            )
            n_atoms = int(pos.shape[0])
            template = Atoms(cell=np.eye(3) * cell_side, pbc=False)
            environment = Environment(
                template=template,
                symbols=f"N{n_atoms}",
                confinement_cell=np.eye(3) * side,
                confinement_corner=np.array([corner, corner, corner]),
                print_report=False,
            )
            atoms = Atoms(f"N{n_atoms}", positions=pos, cell=template.cell, pbc=False)
            atoms.set_constraint(environment.get_box_constraint())
            counter = HarnessCounter(cache=False, cap=cap)
            atoms.calc = attach_ase_counter(LennardJones(), counter=counter)
            quenched = _quench(atoms, float(settings["fmax"]), int(settings["max_steps"]))
            consider(quenched, counter.calls)
            status(f"quench calls {counter.calls} energy {best}")
            hit = target is not None and best <= target and counter.calls < cap
            if not hit and counter.calls < cap and _finite(best):
                seed_candidate = _seed_candidate(environment, atoms)
                database = Database(filename=str(work / "candidates.db"), order=5, initialize=True)
                database.store_candidate(seed_candidate)
                agox = _build_search(method, environment, atoms.calc, database, seed_candidate, settings, seed)
                state = State()
                idle = 0
                while not state.get_convergence_status() and iterations < 1_000_000:
                    if counter.calls >= cap:
                        break
                    before = counter.calls
                    status(
                        f"iter {iterations + 1} calls {before} best {best:.6f} "
                        f"elapsed {time.perf_counter() - started:.1f}"
                    )
                    try:
                        agox.dispatch_to_observers(state=state)
                    except Exception as exc:
                        if not _is_cap(exc):
                            raise
                    iterations += 1
                    for energy in list(getattr(database, "candidate_energies", []) or []):
                        consider(energy, counter.calls)
                    status(
                        f"iter {iterations} calls {counter.calls} best {best:.6f} "
                        f"elapsed {time.perf_counter() - started:.1f}"
                    )
                    if target is not None and best <= target and counter.calls < cap:
                        break
                    if counter.calls >= cap:
                        break
                    if counter.calls == before:
                        idle += 1
                        if idle >= 5:
                            idle_stop = True
                            break
                    else:
                        idle = 0
                    state.clear()
                    state.advance_iteration_counter()
    except Exception as exc:
        if not _is_cap(exc):
            error = f"{type(exc).__name__}: {exc}"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
    finally:
        os.chdir(previous)

    calls = 0 if counter is None else int(counter.calls)
    finite = _finite(best)
    if idle_stop and not error:
        error = "search made no force calls"
    success = (
        target is not None
        and finite
        and best <= target
        and calls < cap
        and not error
        and not idle_stop
    )
    reasons: list[str] = []
    if not success:
        if error or idle_stop:
            reasons.append("crash")
        elif calls >= cap:
            reasons.append("cap")
        elif not finite:
            reasons.append("divergence")
        else:
            reasons.append("energy")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": index,
        "force_calls": calls,
        "counter": "harness",
        "comparable": True,
        "success": bool(success),
        "failure": failure,
        "reasons": reasons,
        "lowest_energy": None if not finite else float(best),
        "calls_at_lowest": int(calls_at),
        "iterations": int(iterations),
        "seed": seed,
        "optimizer": "scipy-lbfgsb",
    }
    if target is not None:
        record["target_energy"] = float(target)
    if job.get("reference") is not None:
        record["reference_energy"] = float(job["reference"])
    if error:
        record["error"] = error
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, *, partial: bool, note: str) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "force_calls_median": summary.get("force_calls_median"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "cap": summary.get("cap"),
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "n_cap": metrics.get("n_cap"),
        "n_wrong_stationary": metrics.get("n_wrong_stationary"),
        "n_divergence": metrics.get("n_divergence"),
        "n_crash": metrics.get("n_crash"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "reference_energy": summary.get("reference_energy"),
        "target_energy": summary.get("target_energy"),
    }
    lines = [
        "# counter harness. One force call is one ASE calculate on the harness Lennard-Jones calculator.",
        "# Local steps are SciPy L-BFGS-B stopped on the ASE maximum atomic force.",
        "# " + note.replace("\n", " "),
    ]
    if partial:
        lines.append("# partial true. The row is not a full ranked score.")
        rows["partial"] = "true"
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _publish(
    output: Path,
    records: list[dict],
    *,
    problem: dict,
    template: dict,
    cap: int,
    version: str,
    commit: str,
    parameters: dict,
    partial: bool,
    reference: float | None,
    target: float | None,
    reference_source: str,
) -> dict:
    published = []
    for record in sorted(records, key=lambda row: int(row.get("index", 0))):
        row = dict(record)
        row.pop("index", None)
        published.append(row)
    legacy_counts = [cap if not record["success"] else int(record["force_calls"]) for record in published]
    summary = summarize_capped(legacy_counts, cap) if published else {
        "force_calls": None,
        "force_calls_min": None,
        "force_calls_max": None,
        "nfailed": 0,
    }
    successes = [int(record["force_calls"]) for record in published if record["success"]]
    summary["median"] = statistics.median(successes) if successes else None
    if "force_calls_median" in ((problem.get("metrics") or {}).get("required") or []) and successes:
        summary["force_calls_median"] = statistics.median(successes)
    energies = [float(record["lowest_energy"]) for record in published if record.get("lowest_energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    summary["cap"] = cap
    if reference is not None:
        summary["reference_energy"] = reference
    if target is not None:
        summary["target_energy"] = target
    metrics = (
        score_v2(
            [
                {
                    "force_calls": record["force_calls"],
                    "success": record["success"],
                    "failure": record["failure"],
                }
                for record in published
            ],
            cap,
        )
        if published
        else {}
    )
    note = str(template.get("note") or "").strip()
    if reference_source == "cambridge" and target is not None:
        note += " The reference energy is the Cambridge cluster coordinates evaluated with this potential."
    if target is None:
        note += " The problem file states no energy window, so a start is not marked successful."
    provenance = collect_provenance(
        code="agox",
        code_version=version,
        commit=commit,
        parameters=parameters,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": note,
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": parameters,
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "force_calls": summary.get("force_calls"),
                    "force_calls_min": summary.get("force_calls_min"),
                    "force_calls_max": summary.get("force_calls_max"),
                    "nfailed": summary.get("nfailed"),
                    "median": summary.get("median"),
                },
                "starts": published,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if published:
        output.write_text(_dat(summary, template, metrics, partial=partial, note=note), encoding="utf-8")
        write_starts(output.with_suffix(".starts.jsonl"), published)
        output.with_suffix(".v2.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary["starts"] = len(published)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics.get("par2_mean")
    summary["v2_nfailed"] = metrics.get("nfailed")
    summary["nfailed"] = summary.get("nfailed")
    return summary


def _load_resume(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _rows_for_cap(rows: list[dict], cap: int) -> list[dict]:
    """Keep starts that already finished under this cap.

    A cap stop recorded with fewer calls than ``cap`` is an unfinished
    start. The search runs again from the archived geometry.
    """
    kept = []
    for row in rows:
        calls = int(row.get("force_calls") or 0)
        if row.get("failure") == "cap" and calls < int(cap):
            continue
        kept.append(row)
    return kept


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    import importlib.metadata as metadata

    version = metadata.version("agox")
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"agox {version} does not match template {template['code_version']}")
    pinned = _registry_row(root)
    commit = str(template.get("commit") or "")
    if commit != str(pinned["commit"]):
        raise RuntimeError(f"{template['id']} commit is not the registry pin {pinned['commit']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    from optbench.campaign import _check_stop

    cap = _check_stop(problem, template)
    if cap is None:
        raise ValueError(f"{template['id']} has no force-call cap")
    settings = _section(template)
    for key in ("fmax", "max_steps", "rattle_amplitude", "rattle_fraction", "cell", "confinement"):
        if key not in settings:
            raise ValueError(f"{template['id']} needs agox.{key}")
    settings.setdefault("temperature", 0.25)
    settings.setdefault("population", 10)
    settings.setdefault("surrogate_steps", 5)
    settings.setdefault("surrogate_start", 8)
    settings.setdefault("comparator_threshold", 0.5)
    seed_base = int(settings.get("seed_base") or 314159)
    work_root = work or Path("/tmp/optbench-agox") / template["id"].replace("/", "-")
    work_root.mkdir(parents=True, exist_ok=True)
    loaded = _load_geometries(root, problem, geometries, work_root)
    if limit and limit > 0:
        loaded = loaded[: int(limit)]
    n_atoms = int(len(loaded[0][1])) if loaded else 0
    reference, reference_source = reference_energy(root, problem, n_atoms) if n_atoms else (None, "missing")
    target = target_energy(problem, reference)
    done = _load_resume(output.with_suffix(".starts.jsonl")) if resume else []
    done = _rows_for_cap(done, int(cap))
    done_names = {str(row["start"]) for row in done}
    jobs = []
    for index, (name, coords) in enumerate(loaded):
        if name in done_names:
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "cap": int(cap),
                "seed": seed_base + index,
                "method": str(settings["method"]),
                "settings": {key: value for key, value in settings.items() if key != "method"},
                "target": target,
                "reference": reference,
                "work": str(work_root / f"{index:04d}-{name}"),
            }
        )
    records = list(done)
    # Resume rows may lack the archive index. Put them back in file order.
    order = {name: index for index, (name, _coords) in enumerate(loaded)}
    for record in records:
        record["index"] = order.get(str(record["start"]), int(record.get("index", 0)))
    parameters = {
        "method": str(settings["method"]),
        "cap": int(cap),
        "fmax": float(settings["fmax"]),
        "max_steps": int(settings["max_steps"]),
        "seed_base": seed_base,
        "optimizer": "scipy-lbfgsb",
        "rattle_amplitude": float(settings["rattle_amplitude"]),
        "rattle_fraction": float(settings["rattle_fraction"]),
        "cell": float(settings["cell"]),
        "confinement": float(settings["confinement"]),
        "reference_source": reference_source,
    }
    if settings["method"] == "basin_hopping":
        parameters["temperature"] = float(settings["temperature"])
    if settings["method"] == "ga":
        parameters["population"] = int(settings["population"])
    if settings["method"] == "gofee":
        parameters["surrogate_steps"] = int(settings["surrogate_steps"])
        parameters["use_ray"] = False

    def publish(finished: bool) -> dict:
        stated_n = (problem.get("sample") or {}).get("starts")
        short = stated_n is not None and len(records) < int(stated_n)
        return _publish(
            output,
            records,
            problem=problem,
            template=template,
            cap=int(cap),
            version=version,
            commit=commit,
            parameters=parameters,
            partial=bool(short or target is None or not finished),
            reference=reference,
            target=target,
            reference_source=reference_source,
        )

    if jobs:
        if workers <= 1:
            for job in jobs:
                record = search_job(job)
                records.append(record)
                print(
                    f"{template['id']} {record['start']} calls={record['force_calls']} "
                    f"{record['failure'] or 'success'} E={record.get('lowest_energy')}",
                    flush=True,
                )
                publish(False)
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(search_job, job) for job in jobs]
                total = len(futures) + len(done)
                finished_n = len(done)
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    finished_n += 1
                    print(
                        f"{template['id']} {finished_n}/{total} {record['start']} "
                        f"calls={record['force_calls']} {record['failure'] or 'success'} "
                        f"E={record.get('lowest_energy')}",
                        flush=True,
                    )
                    publish(False)
    return publish(True)
