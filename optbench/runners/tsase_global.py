"""TSASE basin hopping and minima hopping on the harness Lennard-Jones potential.

The search calls ``optbench.potentials.lennard_jones``: epsilon 1, sigma 1,
cutoff 40, no shift, no periodic images. That is the potential in
``problems/minimization/lj38.toml`` and in the archived global-optimization
LAMMPS input. TSASE's Fortran Lennard-Jones is shifted and applies the
minimum image, so the runner does not use it. A basin-hopping quench and a
minima-hopping dimer or energy comparison both go through one
``HarnessCounter``.

Lennard-Jones 38 succeeds when the lowest harness energy is at or below the
published minimum plus ``energy_above_minimum``. The coordinates are minimum
1 of the Wales group GMIN example ``expected_output/lowest``. A problem
without that window keeps the lowest energy and the call count and is
written partial: there is no success rule to score.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One thread per start. OpenBLAS otherwise multiplies across the pool.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np

from optbench.campaign import benchmark_text, geometry_archive
from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2
from optbench.v2 import collect_provenance

# Published coordinates, minimum 1 of the Wales group GMIN LJ38 example.
LJ38_MINIMUM_XYZ = """
0.6299851998 0.3044874184 -1.6035162520
1.1252384555 -0.8376752902 1.0454755576
-0.0258273467 -1.6955955872 0.4303470738
1.1783957802 -0.9591345465 -0.8673414572
-1.6048179648 -0.6967336611 -0.0003749866
-0.0987361306 1.1034007222 -1.3541286680
1.4280194730 0.2282084762 0.9846521618
-1.4796993828 -0.1016457922 0.9279107049
-0.3286648600 0.7562811073 1.5429992836
0.8796320967 1.4917617806 0.2485168796
0.0257685132 1.6955885585 -0.4303467947
-0.8796352186 -1.4917579073 -0.2485807413
0.7028350409 1.0232580431 1.2327986075
-0.7027918382 -1.0232259477 -1.2328501015
-1.0007304721 1.4298791363 -0.1216593597
-1.1252315384 0.8376897138 -1.0454425466
-1.1783855809 0.9590780744 0.8674130279
1.0006909095 -1.4299156331 0.1217334261
-1.4280068641 -0.2282068999 -0.9846538177
1.6048131701 0.6967000362 0.0003595574
-0.6299807560 -0.3044435411 1.6034997522
0.0987300364 -1.1033675736 1.3541123556
0.3287279777 -0.7562466173 -1.5429773471
1.4796730002 0.1015969411 -0.9279077496
-0.1493186593 1.2210290032 0.5543720709
0.1493209182 -1.2210305856 -0.5543688693
0.7504855901 0.8952977342 -0.6751064029
1.2975472507 -0.3652210476 0.0592511328
-0.3977353806 0.0394908142 -1.2887118611
0.3977475045 -0.0394599505 1.2886930693
-1.2975550606 0.3651936416 -0.0592293874
-0.7504908377 -0.8952893529 0.6750807656
-0.4481267163 0.1622264302 0.6123069451
0.2724467715 -0.6277636046 0.3657193526
0.5718516939 0.4262235761 0.3055893368
0.4481315943 -0.1622255764 -0.6123116069
-0.5718512290 -0.4262224546 -0.3056029890
-0.2724551395 0.6277703619 -0.3657201223
"""

# Header energy on that Wales file. The harness energy of the coordinates
# is the reference the window is added to; this constant is only the check.
PUBLISHED_LJ38_ENERGY = -173.9284262
DEFAULT_SEED = 314159
MAX_HOPS = 1_000_000
FAILED_QUENCHES = 50
# A healthy line search reads one geometry a handful of times. ASE's
# BFGSLineSearch, which is what ``QuasiNewton`` names in ASE 3.29, can
# also propose one clamped step forever on an overlapped cluster. Accept
# that step before the repeated-geometry guard stops the search.
STUCK_LINE_STEPS = 15


class Exploded(Exception):
    """The potential energy left the range a local step can recover from."""


class SearchStop(BaseException):
    """Leave a TSASE loop. Basin hopping uses a bare ``except`` around the quench."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def lj38_minimum_positions() -> np.ndarray:
    """Coordinates of the published Lennard-Jones 38 global minimum."""
    rows = []
    for line in LJ38_MINIMUM_XYZ.splitlines():
        if not line.strip():
            continue
        rows.append([float(part) for part in line.split()])
    positions = np.asarray(rows, dtype=np.float64)
    if positions.shape != (38, 3):
        raise RuntimeError("LJ38 minimum table does not have 38 atoms")
    return positions


def lj38_minimum_energy() -> float:
    """Harness energy of the published Lennard-Jones 38 minimum."""
    energy, _gradient = lennard_jones(lj38_minimum_positions().reshape(-1))
    return float(energy)


def energy_threshold(problem: dict) -> float | None:
    """Energy at or below which a start has found the minimum.

    ``None`` when the problem does not state ``energy_above_minimum``.
    A stated reference energy is the Cambridge value on the problem.
    Lennard-Jones 38 falls back to the harness energy of the published
    structure when that reference is absent.
    """
    window = (problem.get("convergence") or {}).get("energy_above_minimum")
    if window in (None, ""):
        return None
    reference = (problem.get("reference") or {}).get("energy")
    if reference not in (None, ""):
        return float(reference) + float(window)
    if problem.get("id") == "globalopt/lj38":
        return lj38_minimum_energy() + float(window)
    raise ValueError(f"{problem['id']} states an energy window and has no reference energy")


def judge_global(
    *,
    calls: int,
    cap: int,
    energy: float | None,
    threshold: float | None,
    crashed: bool,
    exploded: bool,
) -> tuple[bool, str | None, list[str]]:
    """Success bit, failure class, and reasons for one start.

    Without a threshold every finite run is a failure. The row is partial
    because the problem has no success rule. A crash or a non-finite energy
    keeps its own class.
    """
    finite = energy is not None and math.isfinite(float(energy))
    reasons: list[str] = []
    if crashed:
        reasons.append("crash")
    elif not finite or exploded:
        reasons.append("divergence")
    elif calls >= cap or threshold is None:
        reasons.append("cap")
    elif float(energy) > float(threshold):
        reasons.append("energy")
    failure = classify_failure(reasons)
    return failure is None, failure, reasons


class HarnessLennardJones:
    """ASE-shaped calculator. One new geometry is one harness call.

    Forces are minus the harness gradient. The cap raises ``SearchStop``
    before a search call past the budget. An energy inside the problem
    window raises ``SearchStop`` after that call is counted.
    """

    implemented_properties = ("energy", "forces")

    def __init__(self, counter: HarnessCounter, threshold: float | None):
        self.counter = counter
        self.threshold = None if threshold is None else float(threshold)
        self.results: dict = {}
        self.atoms = None

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=("positions",)):
        del properties, system_changes
        if atoms is None:
            atoms = self.atoms
        coords = np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in coords)
        try:
            fresh, cached = self.counter.charge(key)
        except ForceCallCap as exc:
            raise SearchStop("cap") from exc
        if not fresh and cached is not None:
            energy, gradient = cached
            # A stuck line search repeats one geometry. A real step moves.
            self._repeat = getattr(self, "_repeat", 0) + 1
            if self._repeat > 100:
                raise Exploded("repeated geometry")
        else:
            self._repeat = 0
            energy, gradient = lennard_jones(coords)
            gradient = tuple(float(value) for value in np.asarray(gradient, dtype=np.float64).reshape(-1))
            energy = float(energy)
            self.counter.remember(key, (energy, gradient))
        lowest = getattr(self.counter, "lowest_energy", None)
        if lowest is None or energy < lowest:
            self.counter.lowest_energy = float(energy)
            lowest = self.counter.lowest_energy
        calls = self.counter.calls
        if calls and calls % 100000 == 0 and calls != getattr(self, "_reported", None):
            self._reported = calls
            print(
                f"{Path.cwd().name} calls={calls} lowest={lowest}",
                flush=True,
            )
        # The Lennard-Jones 75 archive starts overlapped, and a quench can
        # still climb out. Only a non-finite energy is an immediate stop.
        if not math.isfinite(energy):
            raise Exploded(f"energy {energy}")
        forces = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
        self.results = {"energy": float(energy), "forces": forces}
        if self.threshold is not None and energy <= self.threshold:
            raise SearchStop("energy")

    def get_potential_energy(self, atoms=None, force_consistent=False):
        del force_consistent
        self.calculate(atoms)
        return float(self.results["energy"])

    def get_forces(self, atoms=None):
        self.calculate(atoms)
        return np.array(self.results["forces"], dtype=np.float64, copy=True)

    def get_stress(self, atoms=None):
        del atoms
        return None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def calculation_required(self, atoms, quantities):
        del atoms, quantities
        return True


def _install_harness(counter: HarnessCounter, threshold: float | None) -> HarnessLennardJones:
    """Route ``tsase.calculators.lj`` through one harness calculator.

    Minima hopping constructs that class for the dimer and for the energy
    comparison. The replacement keeps those calls on the spec potential.
    """
    import tsase.calculators as calculators

    shared = HarnessLennardJones(counter, threshold)

    class _Factory:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get_potential_energy(self, atoms=None, force_consistent=False):
            return shared.get_potential_energy(atoms, force_consistent)

        def get_forces(self, atoms=None):
            return shared.get_forces(atoms)

        def get_stress(self, atoms=None):
            return shared.get_stress(atoms)

        def set_atoms(self, atoms):
            shared.set_atoms(atoms)

        def calculation_required(self, atoms, quantities):
            return shared.calculation_required(atoms, quantities)

    calculators.lj = _Factory
    return shared


def _import_tsase():
    import numpy

    # NumPy 2 removed the aliases TSASE still calls.
    if not hasattr(numpy, "sometrue"):
        numpy.sometrue = numpy.any
    if not hasattr(numpy, "alltrue"):
        numpy.alltrue = numpy.all
    numpy.seterr(all="ignore")
    import tsase

    # Importing the package sets numpy errors to raise. SDLBFGS restarts on a
    # negative curvature with a division that can warn. Leave warnings as
    # warnings so a quench is not reported as a failed step.
    numpy.seterr(all="warn")
    return tsase


def _make_atoms(positions: np.ndarray):
    from ase import Atoms

    count = int(positions.shape[0])
    atoms = Atoms(
        symbols=["X"] * count,
        positions=np.asarray(positions, dtype=np.float64),
        cell=[100.0, 100.0, 100.0],
        pbc=False,
    )
    atoms.set_masses(np.ones(count, dtype=np.float64))
    return atoms


def _attach(atoms, counter: HarnessCounter, threshold: float | None) -> None:
    calc = _install_harness(counter, threshold)
    atoms.calc = calc


def _section(template: dict) -> dict:
    section = template.get("tsase") or {}
    if "method" not in section:
        raise ValueError(f"{template['id']} needs tsase.method")
    method = str(section["method"])
    if method not in {"basin_hopping", "minima_hopping"}:
        raise ValueError(f"{template['id']} method {method} is not basin_hopping or minima_hopping")
    return section


def _cap(template: dict) -> int:
    stop = template.get("stop") or {}
    raw = stop.get("max_force_calls", template.get("failure_cap"))
    if raw in (None, ""):
        raise ValueError(f"{template['id']} needs stop.max_force_calls")
    cap = int(raw)
    if cap <= 0:
        raise ValueError(f"{template['id']} cap must be positive")
    return cap


def basin_settings(section: dict, threshold: float | None) -> dict:
    """Constructor arguments. Unset knobs stay at the library defaults."""
    from ase.units import kB

    settings = {
        "logfile": None,
        "trajectory": None,
        "optimizer_logfile": None,
        "local_minima_trajectory": None,
    }
    if threshold is not None:
        # The loop stops on a strict ``<``. One ulp above the window still
        # accepts an energy that lands on the boundary.
        settings["minenergy"] = float(threshold) + 1e-9
    if "temperature_K" in section:
        settings["temperature"] = float(section["temperature_K"]) * float(kB)
    elif "temperature" in section:
        settings["temperature"] = float(section["temperature"])
    if "fmax" in section:
        settings["fmax"] = float(section["fmax"])
    if "dr" in section:
        settings["dr"] = float(section["dr"])
    return settings


def minima_settings(section: dict) -> dict:
    settings = {}
    for key in ("T0", "fmax", "timestep", "mdmin", "Ediff0", "beta1", "beta2", "beta3"):
        if key in section:
            value = section[key]
            settings[key] = int(value) if key == "mdmin" else float(value)
    return settings


def describe_parameters(section: dict, threshold: float | None) -> dict:
    """JSON parameters for the result. Library defaults fill the unset knobs."""
    from ase.units import kB

    method = str(section["method"])
    if method == "basin_hopping":
        described = {
            "method": method,
            "temperature": float(100 * kB),
            "fmax": 0.1,
            "dr": 0.1,
            "maxstep": 0.05,
            "optimizer": "SDLBFGS",
            "distribution": "uniform",
            "jumpmax": 5000,
            "pushapart": 0.4,
            "seed": int(section.get("seed", DEFAULT_SEED)),
        }
        described.update(
            {
                key: value
                for key, value in basin_settings(section, threshold).items()
                if key not in {"logfile", "trajectory", "optimizer_logfile", "local_minima_trajectory"}
            }
        )
        return described
    described = {
        "method": method,
        "T0": 1000.0,
        "fmax": 0.05,
        "timestep": 0.1,
        "mdmin": 2,
        "Ediff0": 0.5,
        "optimizer": "QuasiNewton",
        "seed": int(section.get("seed", DEFAULT_SEED)),
    }
    described.update(minima_settings(section))
    if threshold is not None:
        described["minEnergy"] = float(threshold)
    return described


def _run_basin(job: dict, positions: np.ndarray, counter: HarnessCounter) -> str:
    _import_tsase()
    from tsase.optimize.basin import BasinHopping

    class _Hop(BasinHopping):
        def get_energy(self, pos, symbols):
            if counter.calls >= int(job["cap"]):
                raise SearchStop("cap")
            if np.any(self.positions != pos) or self.swap_atoms:
                self.positions = pos
                self.atoms.set_positions(pos)
                self.atoms.set_chemical_symbols(symbols)
                try:
                    if self.optimizer.__name__ == "FIRE":
                        opt = self.optimizer(
                            self.atoms,
                            maxmove=1.0,
                            dt=0.2,
                            dtmax=1.0,
                            logfile=self.optimizer_logfile,
                        )
                    else:
                        opt = self.optimizer(
                            self.atoms,
                            logfile=self.optimizer_logfile,
                            maxstep=self.mss,
                        )
                    opt.run(fmax=self.fmax)
                    self.energy = self.atoms.get_potential_energy()
                    self.local_min_pos = self.atoms.get_positions()
                except SearchStop:
                    raise
                except Exception:
                    self._failed = getattr(self, "_failed", 0) + 1
                    if self._failed >= FAILED_QUENCHES:
                        raise SearchStop("divergence") from None
                    return None
                self._failed = 0
            if counter.calls >= int(job["cap"]):
                raise SearchStop("cap")
            return self.energy

    atoms = _make_atoms(positions)
    _attach(atoms, counter, job["threshold"])
    hopper = _Hop(atoms, **job["bh_settings"])
    if job["threshold"] is not None and float(hopper.E_global_min) <= float(job["threshold"]):
        return ""
    hopper.run(MAX_HOPS)
    return ""


class _DiscardTrajectory:
    """Stand-in for the per-step hop files. A fresh start never reads them.

    The file is created so minima hopping can remove it before the next
    hop. Frames are not stored.
    """

    def __init__(self, filename, *args, **kwargs):
        del args, kwargs
        self.filename = str(filename)
        Path(self.filename).touch()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, *args, **kwargs):
        del args, kwargs

    def set_description(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        return None

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())


def _patch_repeated_line_search() -> None:
    """Accept a QuasiNewton step that the max-step clamp keeps repeating.

    ``ase.optimize.QuasiNewton`` is BFGSLineSearch. On an overlapped
    Lennard-Jones start the search direction is long, so every trial is
    clamped onto the point just evaluated. FIRE, the old QuasiNewton, and
    TSASE's own SDLBFGS all take a real step on that geometry. Leaving the
    line search there lets this QuasiNewton build a new direction instead
    of reading one geometry until the harness calls it divergence.
    """
    from ase.utils.linesearch import LineSearch

    if getattr(LineSearch.step, "_optbench_repeat", False):
        return
    original = LineSearch.step

    def step(self, stp, f, g, c1, c2, xtol, isave, dsave):
        out = original(self, stp, f, g, c1, c2, xtol, isave, dsave)
        same = False
        if str(self.task)[:2] == "FG" and self.old_stp:
            scale = max(1.0, abs(float(self.old_stp)))
            same = abs(float(out) - float(self.old_stp)) <= float(xtol) * scale
        count = getattr(self, "_optbench_same", 0)
        self._optbench_same = count + 1 if same else 0
        if self._optbench_same >= STUCK_LINE_STEPS:
            self.task = "CONVERGENCE"
            self._optbench_same = 0
        return out

    step._optbench_repeat = True
    LineSearch.step = step


def _patch_zero_line_search() -> None:
    """Replace a zero warning step with one max-step along the search direction.

    After the clamped step above is accepted, BFGSLineSearch can return
    ``WARNING: ROUNDING ERRORS PREVENT PROGRESS`` with step length 0. ASE
    only drops a warning when ``task[1:4] == 'WARN'``, which does not match
    that string, so QuasiNewton takes the zero step and stays on the
    overlapped geometry. One max-step along the current direction is enough
    for the same optimizer to finish the quench. A healthy line search does
    not emit the warning.
    """
    from ase.utils.linesearch import LineSearch

    if getattr(LineSearch._line_search, "_optbench_zero", False):
        return
    original = LineSearch._line_search

    def _line_search(self, *args, **kwargs):
        stp, fval, old, flag = original(self, *args, **kwargs)
        warned = str(self.task).startswith("WARN")
        zero = stp is None or float(stp) == 0.0
        if warned and zero:
            length = float(self.get_gradient_norm(np.asarray(self.pk, dtype=float)))
            if length > 0.0 and float(self.maxstep) > 0.0:
                stp = float(self.maxstep) / length
                self.task = "CONVERGENCE"
        return stp, fval, old, flag

    _line_search._optbench_zero = True
    LineSearch._line_search = _line_search


def _patch_hop_trajectories() -> None:
    """Drop qn.traj and md.traj. The minima list stays on disk."""
    import ase.io
    import ase.io.trajectory as trajectory

    if getattr(trajectory.Trajectory, "_optbench_discard", False):
        return
    original = trajectory.Trajectory

    def open_trajectory(filename, *args, **kwargs):
        if os.path.basename(str(filename)) in {"qn.traj", "md.traj"}:
            return _DiscardTrajectory(filename)
        return original(filename, *args, **kwargs)

    open_trajectory._optbench_discard = True
    trajectory.Trajectory = open_trajectory
    ase.io.Trajectory = open_trajectory


def _patch_velocity_verlet() -> None:
    """ASE 3.29 names the step ``timestep``. TSASE still passes ``dt``."""
    import tsase.optimize.minimahopping as hopping
    from ase.md.verlet import VelocityVerlet

    if getattr(hopping.VelocityVerlet, "_optbench_dt", False):
        return

    class _VelocityVerlet(VelocityVerlet):
        _optbench_dt = True

        def __init__(self, atoms, timestep=None, trajectory=None, logfile=None, loginterval=1, dt=None, **kwargs):
            if timestep is None:
                timestep = dt
            if timestep is None:
                raise TypeError("VelocityVerlet needs a timestep")
            super().__init__(
                atoms,
                float(timestep),
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                **kwargs,
            )

    hopping.VelocityVerlet = _VelocityVerlet


def _run_minima(job: dict, positions: np.ndarray, counter: HarnessCounter) -> str:
    _import_tsase()
    _patch_velocity_verlet()
    _patch_repeated_line_search()
    _patch_zero_line_search()
    _patch_hop_trajectories()
    from tsase.optimize.minimahopping import MinimaHopping

    atoms = _make_atoms(positions)
    _attach(atoms, counter, job["threshold"])
    hopper = MinimaHopping(atoms, **job["mh_settings"])
    # A missing window must not stop the search, and ``None`` is not ordered.
    stop_energy = job["threshold"] if job["threshold"] is not None else -1.0e300
    hopper(totalsteps=MAX_HOPS, minEnergy=stop_energy)
    log = Path("hop.log")
    if log.is_file() and "exploded" in log.read_text(encoding="utf-8", errors="replace").lower():
        return "divergence"
    return ""


def run_start(job: dict) -> dict:
    """One start in its own working directory."""
    work = Path(job["workdir"])
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    previous = Path.cwd()
    os.chdir(work)
    seed = int(job["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    counter = HarnessCounter(cache=True, cap=int(job["cap"]))
    positions = np.asarray(job["positions"], dtype=np.float64)
    crashed = False
    exploded = False
    error = ""
    try:
        if job["method"] == "basin_hopping":
            reason = _run_basin(job, positions, counter)
        else:
            reason = _run_minima(job, positions, counter)
        exploded = reason == "divergence"
    except SearchStop as exc:
        exploded = exc.reason == "divergence"
    except Exploded:
        exploded = True
    except Exception as exc:
        crashed = True
        error = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
    lowest = getattr(counter, "lowest_energy", None)
    if lowest is not None:
        lowest = float(lowest)
    calls = int(counter.calls)
    success, failure, reasons = judge_global(
        calls=calls,
        cap=int(job["cap"]),
        energy=lowest,
        threshold=job["threshold"],
        crashed=crashed,
        exploded=exploded,
    )
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": int(job["index"]),
        "force_calls": calls,
        "verify_calls": int(counter.verify_calls),
        "counter": "harness",
        "comparable": job["threshold"] is not None and not crashed,
        "success": bool(success),
        "failure": failure,
        "reasons": reasons,
        "energy": lowest,
        "lowest_energy": lowest,
        "threshold": job["threshold"],
        "converged": bool(success),
    }
    if error:
        record["error"] = error
    if job["threshold"] is None:
        record["unruled"] = True
        record["comparable"] = False
    return record


def _member_index(name: str) -> int:
    base = Path(name).name
    if base.endswith(".con"):
        return int(base[: -len(".con")])
    if base.startswith("coords."):
        return int(base.split(".")[-1])
    raise ValueError(f"unrecognised geometry name {base}")


def _positions_from_text(name: str, text: str) -> np.ndarray:
    base = Path(name).name
    if base.endswith(".con"):
        positions, _box = read_con(text)
        return np.asarray(positions, dtype=np.float64)
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
        raise ValueError(f"{base} has no coordinates")
    return np.asarray(rows, dtype=np.float64)


def _manifest_archive(root: Path, problem: dict) -> Path:
    """Geometry archive for a problem file that does not name one yet.

    ``globalopt/lj75`` ships ``website/globalopt/lj75/lj75_clusters.tgz`` in
    the manifest. The problem file on this main does not point at it.
    """
    manifest = root / "inputs" / "manifest.jsonl"
    prefix = f"website/{problem['id']}/"
    found = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = str(row.get("path") or "")
        if path.startswith(prefix) and "/tmp/" not in path and path.endswith(".tgz"):
            found.append(root / row["stored"])
    if len(found) != 1:
        raise FileNotFoundError(f"{problem['id']} has no single geometry archive")
    return found[0]


def load_geometries(root: Path, problem: dict) -> list[tuple[str, np.ndarray]]:
    """Starts in archive order, numbered rather than lexicographic."""
    try:
        archive = geometry_archive(root, problem)
    except (ValueError, FileNotFoundError):
        archive = _manifest_archive(root, problem)
    found = []
    with tarfile.open(archive) as packed:
        members = []
        for member in packed.getmembers():
            if not member.isfile():
                continue
            base = Path(member.name).name
            if base.endswith(".con") or base.startswith("coords."):
                members.append(member)
        members.sort(key=lambda member: _member_index(member.name))
        for member in members:
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode()
            base = Path(member.name).name
            label = base[: -len(".con")] if base.endswith(".con") else base
            found.append((label, _positions_from_text(base, text)))
    if not found:
        raise FileNotFoundError(f"{problem['id']} archive has no starts")
    return found


def _resume_rows(output: Path, loaded: list[tuple[str, np.ndarray]]) -> list[dict]:
    """Starts already written. The checkpoint keeps the geometry index.

    ``starts.jsonl`` is the published copy and omits that index. A resume
    can rebuild it from the archive order.
    """
    by_index: dict[int, dict] = {}
    for row in _checkpoint(output.with_suffix(".partial.jsonl")):
        by_index[int(row["index"])] = row
    labels = {name: index for index, (name, _positions) in enumerate(loaded)}
    for row in _checkpoint(output.with_suffix(".starts.jsonl")):
        if "index" in row:
            index = int(row["index"])
        else:
            index = labels.get(str(row.get("start")))
        if index is None or index in by_index:
            continue
        copied = dict(row)
        copied["index"] = index
        by_index[index] = copied
    return list(by_index.values())


def _checkpoint(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _publish(
    output: Path,
    template: dict,
    records: list[dict],
    *,
    problem_id: str,
    cap: int,
    parameters: dict,
    threshold: float | None,
    expected: int,
    limit: int,
) -> dict:
    """Write the dat, the starts, and the v2 document from the starts so far."""
    ordered = sorted(records, key=lambda row: int(row["index"]))
    public = []
    for record in ordered:
        copied = dict(record)
        copied.pop("index", None)
        public.append(copied)
    ruled = threshold is not None and len(public) == expected and limit == 0
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in public
        ],
        cap,
    )
    calls = [int(record["force_calls"]) for record in public]
    successes = [call for record, call in zip(public, calls) if record["success"]]
    failed = len(public) - len(successes)
    summary = {
        "force_calls": round(sum(successes) / len(successes)) if successes else None,
        "force_calls_min": min(calls) if calls else None,
        "force_calls_max": max(calls) if calls else None,
        "force_calls_median": float(np.median(calls)) if calls else None,
        "nfailed": failed,
    }
    provenance = collect_provenance(
        code=str(template.get("code") or "tsase"),
        code_version=str(template.get("code_version") or ""),
        commit=str(template.get("commit") or ""),
        parameters=parameters,
    )
    if problem_id == "globalopt/lj75":
        minimum = "The Lennard-Jones 75 minimum is the Cambridge Cluster Database energy on the problem file."
    else:
        minimum = "The Lennard-Jones 38 minimum is the harness energy of the published truncated octahedron."
    note = f"Harness Lennard-Jones, cutoff 40, no shift. {minimum}"
    if threshold is None:
        note = (
            "Harness Lennard-Jones, cutoff 40, no shift. "
            "This problem file has no energy_above_minimum, so the starts are partial. "
            "lowest_energy is the best harness energy on that start."
        )
    document = {
        "schema": "optbench.result/v2",
        "status": "campaign",
        "problem": problem_id,
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": note,
        "partial": not ruled,
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code", "tsase"),
                "algorithm": template.get("algorithm", parameters.get("method", "")),
                "code_version": template.get("code_version", ""),
                "commit": template.get("commit", ""),
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": parameters,
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "force_calls": summary["force_calls"],
                    "force_calls_min": summary["force_calls_min"],
                    "force_calls_max": summary["force_calls_max"],
                    "force_calls_median": summary["force_calls_median"],
                    "nfailed": failed,
                },
                "starts": public,
            }
        ],
    }
    _atomic_text(output, _dat(summary, template, metrics, not ruled))
    body = "\n".join(json.dumps(row, sort_keys=True) for row in public)
    _atomic_text(output.with_suffix(".starts.jsonl"), body + ("\n" if body else ""))
    _atomic_text(
        output.with_suffix(".v2.json"),
        json.dumps(document, indent=2, sort_keys=True) + "\n",
    )
    summary["starts"] = len(public)
    summary["template"] = template["id"]
    summary["problem"] = problem_id
    summary["par2_mean"] = metrics["par2_mean"]
    summary["nfailed"] = metrics["nfailed"]
    return summary


def _dat(summary: dict, template: dict, metrics: dict, partial: bool) -> str:
    text = benchmark_text(summary, template)
    lines = [text.rstrip("\n")]
    lines.append("counter harness")
    lines.append("reducer harness")
    lines.append(f"cap {int(template['stop']['max_force_calls'])}")
    lines.append(f"tuning {template.get('tuning', '')}")
    if partial:
        lines.append("partial true")
    if metrics.get("par2_mean") is not None:
        lines.append(f"par2_mean {metrics['par2_mean']}")
    return "\n".join(lines) + "\n"


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
    """Run one TSASE template and write ``benchmark.dat``, the starts, and the v2 document."""
    del geometries
    section = _section(template)
    method = str(section["method"])
    cap = _cap(template)
    threshold = energy_threshold(problem)
    partial = threshold is None or limit > 0
    parameters = describe_parameters(section, threshold)
    seed = int(section.get("seed", DEFAULT_SEED))
    loaded = load_geometries(root, problem)
    if limit > 0:
        loaded = loaded[:limit]
    scratch = work or output.parent / f"{output.stem}-work"
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".partial.jsonl")
    saved = {int(row["index"]): row for row in _resume_rows(output, loaded)} if resume else {}
    if not resume and checkpoint.exists():
        checkpoint.unlink()
        saved = {}
    jobs = []
    for index, (name, positions) in enumerate(loaded):
        if index in saved:
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "positions": positions.tolist(),
                "method": method,
                "cap": cap,
                "threshold": threshold,
                "seed": seed,
                "workdir": str(scratch / f"start-{index}"),
                "bh_settings": basin_settings(section, threshold) if method == "basin_hopping" else {},
                "mh_settings": minima_settings(section) if method == "minima_hopping" else {},
            }
        )
    records = list(saved.values())
    expected = len(loaded)
    del partial

    def publish() -> dict:
        return _publish(
            output,
            template,
            records,
            problem_id=problem["id"],
            cap=cap,
            parameters=parameters,
            threshold=threshold,
            expected=expected,
            limit=limit,
        )

    if workers <= 1:
        for job in jobs:
            record = run_start(job)
            records.append(record)
            _append_checkpoint(checkpoint, record)
            publish()
            print(
                f"{template['id']} {record['start']} calls={record['force_calls']} "
                f"energy={record['lowest_energy']} success={record['success']} "
                f"failure={record['failure']}",
                flush=True,
            )
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(run_start, job): job for job in jobs}
            total = len(loaded)
            done = len(saved)
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                _append_checkpoint(checkpoint, record)
                publish()
                done += 1
                print(
                    f"{template['id']} {done}/{total} {record['start']} "
                    f"calls={record['force_calls']} energy={record['lowest_energy']} "
                    f"success={record['success']} failure={record['failure']}",
                    flush=True,
                )
    return publish()
