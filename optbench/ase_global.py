"""ASE basin hopping and minima hopping on the global-optimization problems.

A force call is one calculator evaluation, counted by ``HarnessCounter``.
The potential is the free-cluster Lennard-Jones potential used by
``ase.calculators.lj.LennardJones`` with epsilon 1, sigma 1, and cutoff 40.
That is the archived LAMMPS ``lj/cut`` potential at distances inside the cutoff.

Lennard-Jones 38 stops at the archived eOn energy. The config in
``archive/entries/globalopt/lj38/eon-basinhopping/config.ini`` sets
``stop_energy=-173.927427``. The problem file sets
``energy_above_minimum = 0.001``, so the reference minimum is -173.928427.
A start at or below that stop is a success when the call count is under the
template cap. The problem file does not state a force-call cap. The template
cap is the budget that makes a miss a PAR-2 failure.

``BasinHopping.run`` is the ASE 3.29.0 loop. The harness adds two exits:
the stop energy, and the force-call cap. The displacement, the Metropolis
test, and the local minimizer are the library's. ASE displaces from the
coordinates it stored before the quench. The local iteration bound is the
template value. Without it the ASE optimizer default is 1e8 steps, and one
overlapping trial does not return.

``MinimaHopping`` keeps the library molecular dynamics, local minimizer,
acceptance test, and temperature update. Step trajectories are not written.
ASE reads those files only to resume, and each frame is fsynced. The minima
list is kept in memory. The identity test is ASE's greedy same-element
pairing, evaluated at the same distances, so a revisit decision matches
``ComparePositions``. The scan stops at the first match. That can change the
logged closest distance and leaves the acceptance decision unchanged.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .ase_lj import LennardJonesASE
from .campaign import geometry_archive, load_template, resolve_problem
from .counter import attach_ase_counter
from .failures import classify_failure
from .score import half_even, score_v2

CONTRIBUTOR = "Rohit Goswami"
CODE_VERSION = "3.29.0"
COMMIT = "f27c0005ae6a67ea419f996e728668865bfc1f86"
# Archived stop in archive/entries/globalopt/lj38/eon-basinhopping/config.ini.
LJ38_STOP = -173.927427
LJ38_MINIMUM = -173.928427

ARCHIVES = {
    "globalopt/lj38": "website/globalopt/lj38/100_lj38.tar.gz",
    "globalopt/lj75": "website/globalopt/lj75/lj75_clusters.tgz",
    "globalopt/lj98": "website/globalopt/lj98/lj98_clusters.tgz",
    "globalopt/lj150": "website/globalopt/lj150/lj150_clusters.tgz",
}

_OPTIMIZERS = ("FIRE", "LBFGS", "BFGS", "QuasiNewton")


def target_energy(problem: dict) -> float | None:
    """Stop energy from the problem file, or None when the file states no rule.

    Lennard-Jones 38 keeps the archived eOn stop, which is the Cambridge
    minimum plus ``energy_above_minimum``. A later problem states that
    minimum under ``reference.energy``. The stop is that minimum plus the gap.
    """
    convergence = problem.get("convergence") or {}
    if "energy_above_minimum" not in convergence:
        return None
    gap = float(convergence["energy_above_minimum"])
    if problem.get("id") == "globalopt/lj38" and abs(gap - 0.001) < 1e-15:
        return LJ38_STOP
    if "minimum_energy" in convergence:
        return float(convergence["minimum_energy"]) + gap
    reference = (problem.get("reference") or {}).get("energy")
    if reference not in (None, ""):
        return float(reference) + gap
    if problem.get("id") == "globalopt/lj38":
        return LJ38_MINIMUM + gap
    return None


def _force_cap(problem: dict, template: dict) -> int:
    """Problem cap when the file states one, otherwise the template cap."""
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    if stated not in (None, "", 0, "0"):
        return int(stated)
    return int((template.get("stop") or {})["max_force_calls"])


def _archive_path(problem: dict, template: dict) -> str:
    stated = (problem.get("input") or {}).get("archive_path")
    if stated:
        return str(stated)
    stated = (template.get("input") or {}).get("archive_path")
    if stated:
        return str(stated)
    found = ARCHIVES.get(problem["id"])
    if found:
        return found
    raise ValueError(f"{problem['id']} has no geometry archive")


def _archive_file(root: Path, problem: dict, template: dict) -> Path:
    path = _archive_path(problem, template)
    return geometry_archive(root, {"id": problem["id"], "input": {"archive_path": path}})


def _start_index(name: str) -> tuple:
    base = Path(name).name
    stem = Path(base).stem
    if stem.isdigit():
        return (0, int(stem), base)
    suffix = base.rsplit(".", 1)[-1]
    if suffix.isdigit() and suffix != base:
        return (0, int(suffix), base)
    return (1, 0, base)


def list_start_names(archive: Path) -> list[str]:
    """Numbered geometries in the archive. A bare ``coords`` file is not a start."""
    with tarfile.open(archive) as packed:
        names = [member.name for member in packed.getmembers() if member.isfile()]
    numbered = [name for name in names if _start_index(name)[0] == 0]
    chosen = numbered or names
    con = [name for name in chosen if name.endswith(".con")]
    if con:
        chosen = con
    chosen.sort(key=_start_index)
    if not chosen:
        raise FileNotFoundError(f"{archive} has no starts")
    return chosen


def materialize(archive: Path, dest: Path) -> Path:
    """Extract the archive once. A marker skips a second extract."""
    marker = dest / ".ready"
    if marker.is_file():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as packed:
        packed.extractall(dest)
    marker.write_text("ok\n", encoding="utf-8")
    return dest


def _read_atoms(path: Path, kind: str):
    from ase import Atoms
    from ase.io import read

    if kind == "eon":
        atoms = read(path, format="eon")
    else:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if not rows:
            raise ValueError(f"{path} has no coordinates")
        atoms = Atoms(f"C{len(rows)}", positions=rows)
    atoms.set_pbc(False)
    return atoms


def _kind(name: str) -> str:
    return "eon" if name.endswith(".con") else "coords"


def _cap_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "force-call cap" in str(exc)


def _limited(cls, steps: int):
    class Limited(cls):
        def run(self, fmax=0.05, steps=steps):
            return super().run(fmax=fmax, steps=steps)

    Limited.__name__ = cls.__name__
    return Limited


def _optimizer(name: str):
    from ase.optimize import BFGS, FIRE, LBFGS, QuasiNewton

    classes = {
        "BFGS": BFGS,
        "FIRE": FIRE,
        "LBFGS": LBFGS,
        "QuasiNewton": QuasiNewton,
    }
    if name not in classes:
        raise ValueError(f"optimizer {name} is not one of {', '.join(_OPTIMIZERS)}")
    return classes[name]


def _basin_optimizer(parameters: dict):
    from ase.optimize.basin import BasinHopping
    import inspect

    signature = inspect.signature(BasinHopping.__init__)
    name = parameters.get("optimizer")
    cls = _optimizer(name) if name else signature.parameters["optimizer"].default
    return _limited(cls, int(parameters["local_steps"]))


def _basin_kwargs(parameters: dict) -> dict:
    from ase import units

    kwargs = {
        "optimizer": _basin_optimizer(parameters),
        "logfile": None,
        "trajectory": None,
        "optimizer_logfile": None,
        "local_minima_trajectory": None,
    }
    if "temperature_kelvin" in parameters:
        kwargs["temperature"] = float(parameters["temperature_kelvin"]) * units.kB
    elif "temperature" in parameters:
        kwargs["temperature"] = float(parameters["temperature"])
    if "dr" in parameters:
        kwargs["dr"] = float(parameters["dr"])
    if "fmax" in parameters:
        kwargs["fmax"] = float(parameters["fmax"])
    if "adjust_cm" in parameters:
        kwargs["adjust_cm"] = bool(parameters["adjust_cm"])
    return kwargs


class _Basin:
    """ASE's ``BasinHopping.run`` with an energy stop and a cap exit.

    The accept test and the stored pre-quench displacement match
    ``ase.optimize.basin.BasinHopping.run`` at ASE 3.29.0.
    """

    def __init__(self, atoms, target: float | None, **kwargs):
        import numpy as np
        from ase.optimize.basin import BasinHopping

        self._np = np
        self.target = float("-inf") if target is None else float(target)
        self.hit_cap = False
        self.hops = 0
        self._inner_cls = BasinHopping
        self.opt = BasinHopping(atoms, **kwargs)
        self.Emin = float(self.opt.Emin)

    def run(self, steps: int) -> None:
        import numpy as np

        opt = self.opt
        if opt.Emin <= self.target:
            self.Emin = float(opt.Emin)
            return
        ro = opt.positions
        Eo = opt.get_energy(ro)
        for step in range(int(steps)):
            if opt.Emin <= self.target:
                break
            En = None
            while En is None:
                rn = opt.move(ro)
                try:
                    En = opt.get_energy(rn)
                except RuntimeError as exc:
                    if _cap_error(exc):
                        self.hit_cap = True
                        self.hops = step
                        self.Emin = float(opt.Emin)
                        return
                    raise
            self.hops = step + 1
            if En < opt.Emin:
                opt.Emin = En
                opt.rmin = opt.optimizable.get_x().reshape(-1, 3)
                opt.call_observers()
            if self.hops == 1 or self.hops % 100 == 0:
                counter = getattr(getattr(opt._atoms().calc, "optbench_harness", None), "calls", None)
                print(f"basin hop {self.hops} emin {float(opt.Emin):.6f} calls {counter}", flush=True)
            accept = np.exp((Eo - En) / opt.kT) > np.random.uniform()
            if accept:
                ro = rn.copy()
                Eo = En
        self.Emin = float(opt.Emin)


def _record(
    payload: dict,
    *,
    calls: int,
    energy: float | None,
    hops: int,
    reasons: list[str],
    error: str | None = None,
) -> dict:
    cap = int(payload["cap"])
    failure = classify_failure(reasons)
    success = failure is None and calls < cap
    if calls >= cap and failure is None:
        reasons = ["cap", *reasons]
        failure = "cap"
        success = False
    row = {
        "schema": "optbench.start/v2",
        "start": payload["start"],
        "problem": payload["problem"],
        "template": payload["template"],
        "code": "asebh",
        "code_version": CODE_VERSION,
        "commit": COMMIT,
        "tuning": payload["tuning"],
        "algorithm": payload["algorithm"],
        "seed": int(payload["seed"]),
        "force_calls": int(calls),
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "lowest_energy": energy,
        "target_energy": payload.get("target"),
        "hops": int(hops),
        "cap": cap,
    }
    if error:
        row["error"] = error
    return row


def _outcome(payload: dict, calls: int, energy: float | None, hops: int, hit_cap: bool) -> dict:
    reasons: list[str] = []
    target = payload.get("target")
    finite = energy is not None and math.isfinite(float(energy))
    if not finite:
        reasons.append("divergence")
    if hit_cap or calls >= int(payload["cap"]):
        reasons.append("cap")
    reached = finite and target is not None and float(energy) <= float(target)
    if finite and not reached and "cap" not in reasons and "divergence" not in reasons:
        reasons.append("energy")
    if reached and calls < int(payload["cap"]) and "divergence" not in reasons:
        reasons = []
    return _record(
        payload,
        calls=calls,
        energy=None if energy is None or not math.isfinite(float(energy)) else float(energy),
        hops=hops,
        reasons=reasons,
    )


def _run_basin(atoms, calc, payload: dict) -> dict:
    import numpy as np

    np.random.seed(int(payload["seed"]))
    parameters = dict(payload["parameters"])
    try:
        search = _Basin(atoms, payload.get("target"), **_basin_kwargs(parameters))
    except RuntimeError as exc:
        if _cap_error(exc):
            return _outcome(payload, int(calc.optbench_harness.calls), None, 0, True)
        raise
    try:
        search.run(int(payload["max_hops"]))
    except Exception as exc:
        return _record(
            payload,
            calls=int(calc.optbench_harness.calls),
            energy=None,
            hops=search.hops,
            reasons=["crash"],
            error=exc.__class__.__name__,
        )
    return _outcome(
        payload,
        int(calc.optbench_harness.calls),
        search.Emin,
        search.hops,
        search.hit_cap,
    )


def _mh_settings(parameters: dict, work: Path, seed: int) -> dict:
    import numpy as np
    from ase.optimize.minimahopping import MinimaHopping

    allowed = set(MinimaHopping._default_settings)
    kwargs = {}
    for key, value in parameters.items():
        if key in {"local_steps", "optimizer", "seed"}:
            continue
        if key not in allowed:
            raise ValueError(f"MinimaHopping has no parameter {key}")
        kwargs[key] = value
    name = parameters.get("optimizer")
    cls = _optimizer(name) if name else MinimaHopping._default_settings["optimizer"]
    kwargs["optimizer"] = _limited(cls, int(parameters["local_steps"]))
    kwargs["logfile"] = str(work / "hop.log")
    kwargs["minima_traj"] = str(work / "minima.traj")
    kwargs["rng"] = np.random.RandomState(int(seed))
    return kwargs


class _StepTrajectory:
    """In-memory stand-in for an ``md`` or ``qn`` trajectory.

    The optimizer and the dynamics writer call ``write``, ``set_description``,
    and ``len``. None of those results are read during a fresh search.
    """

    def __init__(self, filename, mode="w", atoms=None, **kwargs):
        del filename, mode, kwargs
        self.atoms = atoms
        self._count = 0

    def write(self, atoms=None, **kwargs):
        del atoms, kwargs
        self._count += 1

    def close(self) -> None:
        return None

    def __len__(self) -> int:
        return self._count

    def set_description(self, description) -> None:
        del description

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _radius_gap(pos1, pos2) -> float:
    """Lower bound on the translated match distance.

    Distances from each centre of mass are translation invariant. Any pairing
    leaves some atom at least this far from its partner, so the greedy match
    is at least this large.
    """
    import numpy as np

    def radii(pos):
        centred = pos - pos.mean(axis=0)
        return np.sort(np.sqrt(np.einsum("ij,ij->i", centred, centred)))

    return float(np.max(np.abs(radii(pos1) - radii(pos2))))


def _one_element(atoms) -> bool:
    symbols = atoms.get_chemical_symbols()
    return bool(symbols) and all(symbol == symbols[0] for symbol in symbols)


def _greedy_dmax(pos1, pos2) -> float:
    """Maximum greedy nearest-neighbor distance, in ``pos1`` order.

    ``np.argmin`` keeps the first tied index, which is the order of the
    remaining atoms in ``ComparePositions``.
    """
    import numpy as np

    remaining = np.array(pos2, dtype=float, copy=True)
    worst = 0.0
    for point in pos1:
        delta = remaining - point
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        index = int(np.argmin(dist))
        if dist[index] > worst:
            worst = float(dist[index])
        remaining = np.delete(remaining, index, axis=0)
    return worst


def _translated_dmax(pos1, pos2) -> float:
    """Smallest ``_greedy_dmax`` after translating each atom of ``pos2`` onto ``pos1[0]``."""
    import numpy as np

    standard = pos1 - pos1[0]
    shifted = pos2[None, :, :] - pos2[:, None, :]
    count = pos1.shape[0]
    active = np.ones((count, count), dtype=bool)
    worst = np.zeros(count, dtype=float)
    rows = np.arange(count)
    for point in standard:
        delta = shifted - point
        dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
        dist = np.where(active, dist, np.inf)
        chosen = np.argmin(dist, axis=1)
        np.maximum(worst, dist[rows, chosen], out=worst)
        active[rows, chosen] = False
    return float(worst.min())


class _MinimaRuntime:
    """Drop step trajectories and evaluate the identity test in numpy.

    The patch is process-local. A mixed-element or periodic pair falls back
    to ``ComparePositions``.
    """

    def __enter__(self):
        import ase.io as io
        import ase.io.trajectory as trajectory
        from ase.optimize.minimahopping import ComparePositions, MinimaHopping

        self._io = io
        self._trajectory = trajectory
        self._compare = ComparePositions
        self._hopping = MinimaHopping
        self._io_trajectory = io.Trajectory
        self._module_trajectory = trajectory.Trajectory
        self._compare_call = ComparePositions.__call__
        self._unique = MinimaHopping._unique_minimum_position
        self._record = MinimaHopping._record_minimum
        self._read = MinimaHopping._read_minima
        real = trajectory.Trajectory

        def open_trajectory(filename, mode="r", atoms=None, **kwargs):
            name = Path(str(filename)).name
            if name.startswith(("md", "qn")):
                return _StepTrajectory(filename, mode, atoms)
            return real(filename, mode, atoms, **kwargs)

        def compare_call(compare, atoms1, atoms2):
            import numpy as np

            if (
                np.any(atoms1.pbc)
                or np.any(atoms2.pbc)
                or not _one_element(atoms1)
                or atoms1.get_chemical_symbols() != atoms2.get_chemical_symbols()
            ):
                return self._compare_call(compare, atoms1, atoms2)
            pos1 = np.asarray(atoms1.positions, dtype=float)
            pos2 = np.asarray(atoms2.positions, dtype=float)
            if pos1.shape != pos2.shape or pos1.ndim != 2 or pos1.shape[1] != 3:
                return self._compare_call(compare, atoms1, atoms2)
            if compare._translate:
                return _translated_dmax(pos1, pos2)
            return _greedy_dmax(pos1, pos2)

        def remember(self):
            from ase import Atoms

            symbols = self._atoms.get_chemical_symbols()
            stored = Atoms(
                symbols,
                positions=self._atoms.get_positions().copy(),
                pbc=self._atoms.pbc,
                cell=self._atoms.get_cell(),
            )
            found = getattr(self, "_optbench_minima", None)
            if found is None:
                found = []
                self._optbench_minima = found
                self._optbench_loaded = True
            found.append(stored)
            self._minima = found
            self._log("msg", "Recorded minima #%i." % (len(found) - 1))

        def recall(self):
            if getattr(self, "_optbench_loaded", False):
                self._minima = self._optbench_minima
                return True
            exists = self_read(self)
            self._optbench_minima = list(self._minima)
            self._optbench_loaded = True
            return exists

        def unique(self):
            import numpy as np

            closest = 99999.0
            compare = ComparePositions(translate=True)
            self._read_minima()
            threshold = float(self._minima_threshold)
            current = np.asarray(self._atoms.positions, dtype=float)
            for minimum in reversed(self._minima):
                other = np.asarray(minimum.positions, dtype=float)
                if _radius_gap(current, other) >= threshold:
                    continue
                distance = float(compare(minimum, self._atoms))
                if distance < closest:
                    closest = distance
                if distance < threshold:
                    return False, closest
            return True, closest

        self_read = self._read
        io.Trajectory = open_trajectory
        trajectory.Trajectory = open_trajectory
        ComparePositions.__call__ = compare_call
        MinimaHopping._record_minimum = remember
        MinimaHopping._read_minima = recall
        MinimaHopping._unique_minimum_position = unique
        return self

    def __exit__(self, *exc) -> None:
        self._io.Trajectory = self._io_trajectory
        self._trajectory.Trajectory = self._module_trajectory
        self._compare.__call__ = self._compare_call
        self._hopping._record_minimum = self._record
        self._hopping._read_minima = self._read
        self._hopping._unique_minimum_position = self._unique


def _run_minima(atoms, calc, payload: dict) -> dict:
    from ase.optimize.minimahopping import MinimaHopping

    work = Path(tempfile.mkdtemp(prefix="asebh-mh-"))
    previous = Path.cwd()
    hops = 0
    best = None
    hit_cap = False
    try:
        os.chdir(work)
        with _MinimaRuntime():
            return _minima_search(atoms, calc, payload, work, MinimaHopping)
    except Exception as exc:
        if _cap_error(exc):
            return _outcome(payload, int(calc.optbench_harness.calls), best, hops, True)
        return _record(
            payload,
            calls=int(getattr(getattr(calc, "optbench_harness", None), "calls", 0)),
            energy=best,
            hops=hops,
            reasons=["crash"],
            error=exc.__class__.__name__,
        )
    finally:
        os.chdir(previous)
        shutil.rmtree(work, ignore_errors=True)


def _minima_search(atoms, calc, payload: dict, work: Path, minima_hopping) -> dict:
    hops = 0
    best = None
    hit_cap = False
    try:
        kwargs = _mh_settings(payload["parameters"], work, int(payload["seed"]))
        hopper = minima_hopping(atoms, **kwargs)
        try:
            hopper._startup()
        except RuntimeError as exc:
            if _cap_error(exc):
                hit_cap = True
                return _outcome(payload, int(calc.optbench_harness.calls), best, hops, True)
            raise
        best = _finite_energy(atoms)
        target = payload.get("target")
        while not hit_cap:
            if target is not None and best is not None and best <= float(target):
                break
            if hops >= int(payload["max_hops"]):
                break
            try:
                hopper._previous_optimum = hopper._atoms.copy()
                hopper._previous_energy = hopper._atoms.get_potential_energy()
                hopper._molecular_dynamics()
                hopper._optimize()
                hopper._counter += 1
                hopper._check_results()
            except RuntimeError as exc:
                if _cap_error(exc):
                    hit_cap = True
                    break
                raise
            hops = int(hopper._counter)
            current = _finite_energy(atoms)
            if hops == 1 or hops % 25 == 0:
                counter = getattr(getattr(calc, "optbench_harness", None), "calls", None)
                print(f"minima hop {hops} energy {current} calls {counter}", flush=True)
            if current is not None and (best is None or current < best):
                best = current
        return _outcome(payload, int(calc.optbench_harness.calls), best, hops, hit_cap)
    except Exception as exc:
        if _cap_error(exc):
            return _outcome(payload, int(calc.optbench_harness.calls), best, hops, True)
        return _record(
            payload,
            calls=int(getattr(getattr(calc, "optbench_harness", None), "calls", 0)),
            energy=best,
            hops=hops,
            reasons=["crash"],
            error=exc.__class__.__name__,
        )


def _finite_energy(atoms) -> float | None:
    try:
        value = float(atoms.get_potential_energy())
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def run_start(payload: dict) -> dict:
    """One start. The seed is applied before ASE draws displacements or momenta."""
    started = time.perf_counter()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    potential = payload["potential"]
    if potential.get("smooth"):
        raise ValueError("asebh uses the unsmoothed ASE Lennard-Jones potential")
    kernel = LennardJonesASE(
        float(potential.get("epsilon", 1.0)),
        float(potential.get("sigma", 1.0)),
        float(potential.get("rc", 40.0)),
    )
    calc = kernel.build()
    from .counter import HarnessCounter

    counter = HarnessCounter(cache=False, cap=int(payload["cap"]))
    attach_ase_counter(calc, cap=int(payload["cap"]), counter=counter)
    atoms = _read_atoms(Path(payload["path"]), payload["kind"])
    atoms.calc = calc
    method = payload["method"]
    if method == "BasinHopping":
        row = _run_basin(atoms, calc, payload)
    elif method == "MinimaHopping":
        row = _run_minima(atoms, calc, payload)
    else:
        raise ValueError(method)
    row["seconds"] = time.perf_counter() - started
    return row


def _format(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(template: dict, records: list[dict], *, partial: bool, cap: int, target: float | None) -> str:
    calls = [int(row["force_calls"]) for row in records]
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    energies = [float(row["lowest_energy"]) for row in records if row.get("lowest_energy") is not None]
    scored = [
        {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
        for row in records
    ]
    metrics = score_v2(scored, cap) if records else {}
    legacy_mean = half_even(sum(successes) / len(successes)) if successes else None
    lines = [
        "# counter harness. One force call is one calculator evaluation.",
        "# The potential is free-cluster Lennard-Jones, epsilon 1, sigma 1, cutoff 40.",
    ]
    if template.get("algorithm") == "MinimaHopping":
        lines.append("# ASE 3.29.0 MinimaHopping keeps the library dynamics, quench, and acceptance test.")
    else:
        lines.append("# ASE 3.29.0 BasinHopping keeps the library displacement and Metropolis test.")
    lines.append("# A success is a lowest energy at or below the stop, with calls under the cap.")
    if target is None:
        lines.append("# This problem file states no stop energy. The row is partial.")
    elif template.get("problem") == "globalopt/lj38":
        lines.append(f"# stop energy {target}. Lennard-Jones 38 uses the archived eOn stop.")
    else:
        lines.append(f"# stop energy {target}. The stop is the stated reference plus the energy gap.")
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    fields = {
        "force_calls": legacy_mean,
        "force_calls_min": min(successes) if successes else (min(calls) if calls else None),
        "force_calls_max": max(successes) if successes else (max(calls) if calls else None),
        "nfailed": len(records) - len(successes),
        "starts": len(records),
        "code": template.get("code", "asebh"),
        "code_version": CODE_VERSION,
        "commit": COMMIT,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template["problem"],
        "tuning": template.get("tuning") or template.get("profile") or "",
        "counter": "harness",
        "reducer": "harness",
        "cap": cap,
        "seed": template.get("seed"),
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "n_cap": metrics.get("n_cap"),
        "n_wrong_stationary": metrics.get("n_wrong_stationary"),
        "n_divergence": metrics.get("n_divergence"),
        "n_crash": metrics.get("n_crash"),
        "energy_min": min(energies) if energies else None,
        "energy_max": max(energies) if energies else None,
        "date": date.today().isoformat(),
        "contributor": CONTRIBUTOR,
    }
    if target is not None:
        fields["target_energy"] = target
    for key, value in fields.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format(value)}")
    if partial:
        lines.append("partial true")
    return "\n".join(lines) + "\n"


def _load_done(path: Path) -> dict[str, dict]:
    done = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done[str(row["start"])] = row
    return done


def _write_outputs(
    template: dict,
    names: list[str],
    done: dict[str, dict],
    *,
    output: Path,
    starts_path: Path,
    cap: int,
    target: float | None,
    full: int,
) -> None:
    ordered = [done[name] for name in names if name in done]
    starts_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
    starts_path.write_text(body, encoding="utf-8")
    partial = len(ordered) < full or target is None
    output.write_text(
        _dat(template, ordered, partial=partial, cap=cap, target=target),
        encoding="utf-8",
    )


def run_point(root: Path, template_name: str, start: str, seed: int) -> dict:
    """One start, for a test or an asv parameter."""
    root = Path(root)
    loaded = load_template(root, template_name)
    spec = resolve_problem(root, loaded["problem"])
    archive = _archive_file(root, spec, loaded)
    cache = materialize(archive, Path(tempfile.gettempdir()) / "optbench-asebh" / spec["id"].replace("/", "-"))
    path = cache / start
    if not path.is_file():
        raise FileNotFoundError(path)
    parameters = dict((loaded.get("method") or {}).get("parameters") or {})
    parameters.setdefault("local_steps", 1000)
    payload = {
        "start": start,
        "path": str(path),
        "kind": _kind(start),
        "method": str((loaded.get("method") or {})["name"]),
        "parameters": parameters,
        "potential": dict(loaded.get("potential") or {}),
        "seed": int(seed),
        "cap": _force_cap(spec, loaded),
        "max_hops": int(loaded["stop"].get("max_hops", 10_000_000)),
        "target": target_energy(spec),
        "problem": spec["id"],
        "template": loaded["id"],
        "tuning": loaded.get("tuning") or loaded.get("profile") or "",
        "algorithm": loaded.get("algorithm") or "",
    }
    return run_start(payload)


def run(
    root: Path,
    problem: str,
    template: str,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
    geometries: Path | None = None,
    work: Path | None = None,
    client: Path | None = None,
) -> dict:
    """Run one asebh template and write benchmark.dat plus the per-start file."""
    del geometries, work, client
    root = Path(root)
    loaded = load_template(root, template)
    if loaded.get("engine") != "asebh":
        raise ValueError(f"{template} is not an asebh template")
    spec = resolve_problem(root, loaded["problem"])
    if problem not in {spec["id"], loaded["problem"]}:
        raise ValueError(f"{problem} does not match {loaded['id']}")
    archive = _archive_file(root, spec, loaded)
    names = list_start_names(archive)
    stated = (spec.get("sample") or {}).get("starts")
    if stated not in (None, "", 0, "0") and int(stated) != len(names):
        raise ValueError(f"{spec['id']} states {stated} starts and the archive has {len(names)}")
    if limit > 0:
        names = names[: int(limit)]
    cache = materialize(archive, Path(tempfile.gettempdir()) / "optbench-asebh" / spec["id"].replace("/", "-"))
    target = target_energy(spec)
    cap = _force_cap(spec, loaded)
    max_hops = int((loaded.get("stop") or {}).get("max_hops", 10_000_000))
    parameters = dict((loaded.get("method") or {}).get("parameters") or {})
    parameters.setdefault("local_steps", 1000)
    method = str((loaded.get("method") or {})["name"])
    potential = dict(loaded.get("potential") or {})
    output = Path(output)
    starts_path = output.with_name(output.stem + ".starts.jsonl")
    done = _load_done(starts_path) if resume else {}
    if not resume and starts_path.exists():
        starts_path.unlink()
    payloads = []
    for name in names:
        if name in done:
            continue
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append(
            {
                "start": name,
                "path": str(path),
                "kind": _kind(name),
                "method": method,
                "parameters": parameters,
                "potential": potential,
                "seed": int(loaded.get("seed", 0)),
                "cap": cap,
                "max_hops": max_hops,
                "target": target,
                "problem": spec["id"],
                "template": loaded["id"],
                "tuning": loaded.get("tuning") or loaded.get("profile") or "",
                "algorithm": loaded.get("algorithm") or method,
            }
        )
    if payloads:
        if workers <= 1:
            for index, payload in enumerate(payloads, 1):
                row = run_start(payload)
                done[row["start"]] = row
                _write_outputs(
                    loaded, names, done, output=output, starts_path=starts_path, cap=cap, target=target, full=len(names)
                )
                print(
                    f"{loaded['id']} {row['start']} calls={row['force_calls']} "
                    f"energy={row['lowest_energy']} success={row['success']} {index}/{len(payloads)}",
                    flush=True,
                )
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(run_start, payload) for payload in payloads]
                total = len(futures)
                for index, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    done[row["start"]] = row
                    _write_outputs(
                        loaded,
                        names,
                        done,
                        output=output,
                        starts_path=starts_path,
                        cap=cap,
                        target=target,
                        full=len(names),
                    )
                    print(
                        f"{loaded['id']} {row['start']} calls={row['force_calls']} "
                        f"energy={row['lowest_energy']} success={row['success']} {index}/{total}",
                        flush=True,
                    )
    ordered = [done[name] for name in names]
    _write_outputs(
        loaded, names, done, output=output, starts_path=starts_path, cap=cap, target=target, full=len(names)
    )
    summary = {
        "problem": spec["id"],
        "template": loaded["id"],
        "force_calls": None,
        "nfailed": sum(1 for row in ordered if not row["success"]),
        "starts": len(ordered),
        "output": str(output),
    }
    successes = [int(row["force_calls"]) for row in ordered if row["success"]]
    if successes:
        summary["force_calls"] = half_even(sum(successes) / len(successes))
    return summary
