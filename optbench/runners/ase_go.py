"""ASE basin hopping and minima hopping on the Cartesian globalopt starts.

The basin-hopping parameters are the documentation example in
``doc/ase/optimize.rst`` at ASE commit f27c0005ae6a67ea419f996e728668865bfc1f86:
``temperature=100 * kB``, ``dr=0.5``, ``optimizer=LBFGS``, ``fmax=0.1``.
``ase/test/test_basin.py`` at that commit uses the same temperature and ``dr``.
The constructor defaults in ``ase/optimize/basin.py`` are
``temperature=100 * units.kB``, ``dr=0.1``, ``optimizer=FIRE``, and ``fmax=0.1``.

Minima hopping is ``MinimaHopping(atoms)``. The same file lists the defaults,
and ``ase/optimize/minimahopping.py`` stores them on ``_default_settings``.

``BasinHopping.run`` keeps the displaced coordinates returned by ``move``.
The rigid TIP4P kernel is six coordinates per molecule. These two optimizers
displace Cartesian atoms, so the runner leaves the water problems untouched.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import tarfile
import tempfile
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from optbench.campaign import geometry_archive, load_template
from optbench.globalopt import energy_tolerance, judge_globalopt, reference_energy
from optbench.potentials import binary_lennard_jones, lennard_jones
from optbench.problems import load_problem
from optbench.score import score_v2

COMMIT = "f27c0005ae6a67ea419f996e728668865bfc1f86"
CODE_VERSION = "3.29.0"
SEED = 314159
LONG_CAP = 10_000_000
CARTESIAN = (
    "globalopt/lj38",
    "globalopt/lj75",
    "globalopt/lj98",
    "globalopt/lj150",
    "globalopt/lj100-binary",
)
RIGID = (
    "globalopt/tip4p10",
    "globalopt/tip4p15",
    "globalopt/tip4p20",
)
METHODS = ("basin", "minima")

BASIN_NOTE = (
    "ASE 3.29.0 BasinHopping at commit f27c0005ae6a67ea419f996e728668865bfc1f86. "
    "doc/ase/optimize.rst at that commit constructs BasinHopping with temperature = 100 * kB, "
    "dr = 0.5, optimizer = LBFGS, and fmax = 0.1. ase/test/test_basin.py at that commit calls "
    "BasinHopping with temperature = 100 * kB and dr = 0.5. The constructor in "
    "ase/optimize/basin.py defaults to temperature = 100 * units.kB, dr = 0.1, optimizer = FIRE, "
    "and fmax = 0.1. This row uses the documentation example. The harness seeds numpy.random with "
    "314159 plus the start index, the seed integer in templates/eon-svn/bh-lj38.toml. "
    "BasinHopping.initialize stores the quenched coordinates, and an accepted step stores the displaced coordinates from move. The energy window is the problem file."
)
MINIMA_NOTE = (
    "ASE 3.29.0 MinimaHopping at commit f27c0005ae6a67ea419f996e728668865bfc1f86. "
    "doc/ase/optimize.rst at that commit calls MinimaHopping(atoms) and lists T0 = 1000 K, "
    "beta1 = 1.1, beta2 = 1.1, beta3 = 1/1.1, Ediff0 = 0.5 eV, alpha1 = 0.98, alpha2 = 1/0.98, "
    "mdmin = 2, minima_threshold = 0.5 A, timestep = 1.0 fs, optimizer = QuasiNewton, and "
    "fmax = 0.05 eV/A. Those values are _default_settings in ase/optimize/minimahopping.py. "
    "QuasiNewton is BFGSLineSearch. The call omits totalsteps, so the search runs until the "
    "force-call cap. rng stays None, and the harness seeds numpy.random with 314159 plus the "
    "start index before thermalize_momenta reads it. The energy window is the problem file."
)
GEOMETRY_NOTE = (
    " Coords files carry positions only. Those atoms use symbol C, the symbol on the lj38 con "
    "geometries, and ASE's carbon mass. Binary con files keep the written symbol and mass. "
    "Component 1 is species B and component 2 is species A, matching pair_coeff sigma 1.3 and "
    "sigma 1.0 in archive/entries/globalopt/lj100-binary/eon-basinhopping/in.lammps."
)


class TargetReached(Exception):
    """A calculator call landed at or below the published energy stop."""

    def __init__(self, energy: float):
        super().__init__("target energy")
        self.energy = float(energy)


def template_id(problem_id: str, method: str) -> str:
    slug = problem_id.split("/", 1)[1]
    return f"ase-globalopt/{slug}-{method}"


def campaign_stem(problem_id: str, method: str) -> str:
    slug = problem_id.split("/", 1)[1]
    return f"ase-{CODE_VERSION}-{slug}-{method}"


def basin_settings() -> dict:
    """The documentation example in ``doc/ase/optimize.rst``."""
    from ase import units
    from ase.optimize import LBFGS

    return {
        "temperature": 100 * units.kB,
        "dr": 0.5,
        "fmax": 0.1,
        "optimizer": LBFGS,
        "seed": SEED,
    }


def minima_settings() -> dict:
    """``MinimaHopping._default_settings`` without the file-path entries."""
    from ase.optimize.minimahopping import MinimaHopping

    defaults = MinimaHopping._default_settings
    keys = (
        "T0",
        "beta1",
        "beta2",
        "beta3",
        "Ediff0",
        "alpha1",
        "alpha2",
        "mdmin",
        "minima_threshold",
        "timestep",
        "fmax",
    )
    settings = {key: defaults[key] for key in keys}
    settings["optimizer"] = defaults["optimizer"]
    settings["rng"] = defaults["rng"]
    settings["seed"] = SEED
    return settings


def require_cartesian(problem_id: str) -> None:
    if problem_id in RIGID or problem_id.split("/", 1)[1].startswith("tip4p"):
        raise ValueError(
            f"{problem_id} uses the rigid TIP4P kernel, six coordinates per molecule. "
            "BasinHopping reshapes positions to (-1, 3), displaces those Cartesian rows, "
            "and shifts the centre of mass. MinimaHopping integrates Cartesian positions "
            "with VelocityVerlet. Neither optimizer takes the rigid coordinate."
        )
    if problem_id not in CARTESIAN:
        raise ValueError(f"{problem_id} is outside this runner")


def _problem_path(root: Path, problem_id: str) -> Path:
    return root / "problems" / f"{problem_id}.toml"


def _cap(root: Path, problem: dict, template: dict) -> int:
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    if stated not in (None, ""):
        cap = int(stated)
        if int(template["stop"]["max_force_calls"]) != cap:
            raise ValueError(f"{template['id']} cap differs from {problem['id']}")
        return cap
    if problem["id"] != "globalopt/lj38":
        raise ValueError(f"{problem['id']} has no max_force_calls")
    cap = int(template["failure_cap"])
    eon = load_template(root, "eon-svn/bh-lj38")
    if cap != int(eon["failure_cap"]):
        raise ValueError("lj38 failure_cap differs from eon-svn/bh-lj38")
    return cap


def _spec(root: Path, problem_id: str, method: str) -> dict:
    require_cartesian(problem_id)
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    problem = load_problem(_problem_path(root, problem_id))
    template = load_template(root, template_id(problem_id, method))
    if template["problem"] != problem_id:
        raise ValueError(f"{template['id']} points at {template['problem']}")
    tolerance = energy_tolerance(problem)
    if abs(float(template["stop"]["energy_above_minimum"]) - tolerance) > 0.0:
        raise ValueError(f"{template['id']} energy window differs from the problem")
    reference = reference_energy(problem)
    cap = _cap(root, problem, template)
    potential = problem.get("potential") or {}
    if problem_id == "globalopt/lj100-binary":
        kind = "binary-lennard-jones"
    elif problem_id.startswith("globalopt/lj"):
        kind = str(potential.get("kind") or "lennard-jones")
    else:
        kind = str(potential["kind"])
    if method == "basin":
        documented = basin_settings()
        parameters = template["optimizer"]["parameters"]
        if abs(float(parameters["temperature"]) - float(documented["temperature"])) > 0.0:
            raise ValueError(f"{template['id']} temperature differs from 100 * kB")
        if float(parameters["dr"]) != documented["dr"]:
            raise ValueError(f"{template['id']} dr differs from the documentation example")
        if float(parameters["fmax"]) != documented["fmax"]:
            raise ValueError(f"{template['id']} fmax differs from the documentation example")
        if str(template["optimizer"].get("local")) != "LBFGS":
            raise ValueError(f"{template['id']} local optimizer differs from LBFGS")
        if int(parameters["seed"]) != SEED:
            raise ValueError(f"{template['id']} seed differs from {SEED}")
        settings = documented
    else:
        documented = minima_settings()
        parameters = template["optimizer"]["parameters"]
        for key, value in documented.items():
            if key in {"optimizer", "rng", "seed"}:
                continue
            if abs(float(parameters[key]) - float(value)) > 0.0:
                raise ValueError(f"{template['id']} {key} differs from MinimaHopping._default_settings")
        if str(template["optimizer"].get("local")) != "QuasiNewton":
            raise ValueError(f"{template['id']} local optimizer differs from QuasiNewton")
        if int(parameters["seed"]) != SEED:
            raise ValueError(f"{template['id']} seed differs from {SEED}")
        settings = documented
    binary = None
    if kind == "binary-lennard-jones":
        potential = problem["potential"]
        binary = {
            "sigma_aa": float(potential["sigma_AA"]),
            "sigma_bb": float(potential["sigma_BB"]),
            "epsilon_aa": float(potential["epsilon_AA"]),
            "epsilon_bb": float(potential["epsilon_BB"]),
            "epsilon_ab": float(potential["epsilon_AB"]),
        }
    elif kind != "lennard-jones":
        raise ValueError(f"{problem_id} potential {kind} is not a Cartesian harness kernel")
    return {
        "problem_id": problem_id,
        "method": method,
        "template_id": template["id"],
        "reference": reference,
        "tolerance": tolerance,
        "target": reference + tolerance,
        "cap": cap,
        "starts": int(problem["sample"]["starts"]),
        "kind": kind,
        "binary": binary,
        "settings": settings,
        "atoms": _expected_atoms(problem),
    }


def _expected_atoms(problem: dict) -> int:
    slug = problem["id"].split("/", 1)[1]
    if slug == "lj100-binary":
        potential = problem["potential"]
        return int(potential["n_A"]) + int(potential["n_B"])
    if slug.startswith("lj") and slug[2:].isdigit():
        return int(slug[2:])
    raise ValueError(f"{problem['id']} has no atom count")


def _step_artifact(name: str) -> bool:
    base = os.path.basename(str(name))
    return base.startswith(("md", "qn")) and base.endswith((".log", ".traj"))


def _install_filters():
    """Drop per-step MD and quench files. The minima list and hop log stay.

    A start at the lj98 cap writes one frame per force call. Those frames are
    a restart log. The in-memory positions are the search.
    """
    original_open = builtins.open

    def open_discard(file, *args, **kwargs):
        try:
            text = os.fspath(file)
        except TypeError:
            text = ""
        if str(text).endswith(".log") and _step_artifact(str(text)):
            return original_open(os.devnull, *args, **kwargs)
        return original_open(file, *args, **kwargs)

    builtins.open = open_discard
    import ase.io.trajectory as trajectory

    original_write = trajectory.TrajectoryWriter.write

    def write_discard(self, atoms=None, **kwargs):
        if _step_artifact(str(self.filename)):
            return None
        return original_write(self, atoms, **kwargs)

    trajectory.TrajectoryWriter.write = write_discard

    def restore() -> None:
        builtins.open = original_open
        trajectory.TrajectoryWriter.write = original_write

    return restore


def _parse_con(text: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Positions, symbols, masses, and species from an eOn con file.

    Component order follows ``archive/entries/globalopt/lj100-binary/eon-basinhopping/in.lammps``:
    type 1, the first component, has sigma 1.3 and is species B. Type 2 has sigma 1.0 and is species A.
    A one-component file is unary and the species array is unused.
    """
    lines = text.splitlines()
    components: list[tuple[str, list[list[float]]]] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("Coordinates of Component"):
            symbol = lines[index - 1].strip()
            rows: list[list[float]] = []
            index += 1
            while index < len(lines):
                parts = lines[index].split()
                if len(parts) < 3:
                    break
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    break
                index += 1
            if not rows:
                raise ValueError("con component has no coordinates")
            components.append((symbol, rows))
            continue
        index += 1
    if not components:
        raise ValueError("con file has no coordinates")
    first = next(i for i, line in enumerate(lines) if line.startswith("Coordinates of Component"))
    masses: list[float] | None = None
    for line in lines[:first]:
        parts = line.split()
        if len(parts) != len(components):
            continue
        try:
            masses = [float(part) for part in parts]
        except ValueError:
            continue
    if masses is None:
        raise ValueError("con file has no mass line")
    positions = []
    symbols = []
    mass_rows = []
    species = []
    for component_index, (symbol, rows) in enumerate(components):
        positions.extend(rows)
        symbols.extend([symbol] * len(rows))
        mass_rows.extend([masses[component_index]] * len(rows))
        if len(components) == 1:
            species.extend([0] * len(rows))
        elif component_index == 0:
            species.extend([1] * len(rows))
        elif component_index == 1:
            species.extend([0] * len(rows))
        else:
            raise ValueError("binary con file has more than two components")
    return (
        np.asarray(positions, dtype=np.float64),
        symbols,
        np.asarray(mass_rows, dtype=np.float64),
        np.asarray(species, dtype=np.int64),
    )


def _parse_coords(text: str, atoms: int) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    if len(rows) != atoms:
        raise ValueError(f"expected {atoms} atoms, read {len(rows)}")
    return np.asarray(rows, dtype=np.float64)


def _member_index(name: str, kind: str) -> int | None:
    base = name.rstrip("/").rsplit("/", 1)[-1]
    if kind == "con":
        if not base.endswith(".con"):
            return None
        stem = base[: -len(".con")]
        if not stem.isdigit():
            return None
        return int(stem)
    if kind == "coords":
        prefix = "coords."
        if not base.startswith(prefix):
            return None
        suffix = base[len(prefix) :]
        if not suffix.isdigit():
            return None
        return int(suffix)
    raise ValueError(kind)


def load_starts(archive: Path, problem_id: str, atoms: int) -> list[dict]:
    kind = "con" if problem_id in {"globalopt/lj38", "globalopt/lj100-binary"} else "coords"
    found: list[dict] = []
    with tarfile.open(archive, "r:*") as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            index = _member_index(member.name, kind)
            if index is None:
                continue
            raw = packed.extractfile(member)
            if raw is None:
                continue
            text = raw.read().decode("utf-8", errors="replace")
            if kind == "con":
                positions, symbols, masses, species = _parse_con(text)
                if positions.shape != (atoms, 3):
                    raise ValueError(f"{member.name} has {len(positions)} atoms")
                found.append(
                    {
                        "index": index,
                        "label": f"{index}.con",
                        "positions": positions,
                        "symbols": symbols,
                        "masses": masses,
                        "species": species,
                    }
                )
            else:
                positions = _parse_coords(text, atoms)
                found.append(
                    {
                        "index": index,
                        "label": f"coords.{index}",
                        "positions": positions,
                        "symbols": ["C"] * atoms,
                        "masses": None,
                        "species": None,
                    }
                )
    found.sort(key=lambda item: item["index"])
    if len({item["index"] for item in found}) != len(found):
        raise ValueError(f"{archive} repeats a start index")
    return found


def _reported_energy(hopper, calculator) -> float | None:
    if hopper is not None:
        emin = getattr(hopper, "Emin", None)
        if emin is None:
            emin = getattr(hopper, "best_energy", None)
        if emin is not None:
            value = float(emin)
            if np.isfinite(value) and value < 1.0e30:
                return value
    last = getattr(calculator, "last_energy", None)
    if last is None:
        return None
    value = float(last)
    return value if np.isfinite(value) else None


def _is_cap(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if "force-call cap" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _masses(start: dict) -> np.ndarray:
    if start["masses"] is not None:
        return np.asarray(start["masses"], dtype=np.float64)
    from ase.data import atomic_masses, atomic_numbers

    count = len(start["positions"])
    return np.full(count, float(atomic_masses[atomic_numbers["C"]]), dtype=np.float64)


def _recorded_minima_hopping():
    from ase.optimize.minimahopping import MinimaHopping

    class _Recorded(MinimaHopping):
        def __init__(self, atoms, **kwargs):
            self.best_energy = None
            super().__init__(atoms, **kwargs)

        def _optimize(self):
            super()._optimize()
            results = getattr(self._atoms.calc, "results", None) or {}
            if "energy" in results:
                energy = float(results["energy"])
            else:
                energy = float(self._atoms.get_potential_energy())
            if np.isfinite(energy) and (self.best_energy is None or energy < self.best_energy):
                self.best_energy = energy
            calls = int(self._atoms.calc.optbench_counter["calls"])
            print(
                f"minima optimize calls={calls} energy={energy:.8g} best={self.best_energy}",
                flush=True,
            )

    return _Recorded


def search_one(
    start: dict,
    *,
    problem_id: str,
    method: str,
    root: Path = Path("."),
    cap: int | None = None,
    hops: int | None = None,
) -> dict:
    """Run one start. The seed is 314159 plus the coordinate index."""
    spec = _spec(root, problem_id, method)
    if cap is not None:
        spec["cap"] = int(cap)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    np.random.seed(SEED + int(start["index"]))
    restore = _install_filters()
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix=f"ase-go-{method}-") as tmp:
            os.chdir(tmp)
            return _search_in_directory(start, spec, hops=hops)
    finally:
        os.chdir(previous)
        restore()


def _search_in_directory(start: dict, spec: dict, *, hops: int | None) -> dict:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize.basin import BasinHopping

    from optbench.counter import attach_ase_counter

    positions = np.asarray(start["positions"], dtype=np.float64)
    species = None if start["species"] is None else np.asarray(start["species"], dtype=np.int64)
    label = str(start["label"])
    index = int(start["index"])

    class HarnessCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, target: float, binary: dict | None, kind: str):
            super().__init__()
            self.target = float(target)
            self.binary = binary
            self.kind = kind
            self.last_energy: float | None = None
            self.species = species

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            coords = np.asarray(self.atoms.positions, dtype=np.float64)
            if self.kind == "lennard-jones":
                energy, gradient = lennard_jones(coords)
            else:
                energy, gradient = binary_lennard_jones(coords, self.species, **self.binary)
            self.last_energy = float(energy)
            self.results["energy"] = self.last_energy
            self.results["forces"] = -np.asarray(gradient, dtype=np.float64).reshape((-1, 3))
            calls = int(self.optbench_counter["calls"])
            if calls % 20000 == 0:
                print(
                    f"{spec['problem_id']} {spec['method']} {label} calls={calls} energy={self.last_energy:.8g}",
                    flush=True,
                )
            if np.isfinite(self.last_energy) and self.last_energy <= self.target:
                raise TargetReached(self.last_energy)

    atoms = Atoms(
        symbols=list(start["symbols"]),
        positions=positions,
        masses=_masses(start),
        pbc=False,
    )
    calculator = attach_ase_counter(HarnessCalculator(spec["target"], spec["binary"], spec["kind"]), cap=spec["cap"])
    atoms.calc = calculator
    energy = None
    crashed = False
    error = ""
    hopper = None
    started = time.perf_counter()
    try:
        if spec["method"] == "basin":
            settings = spec["settings"]
            hopper = BasinHopping(
                atoms,
                temperature=settings["temperature"],
                optimizer=settings["optimizer"],
                fmax=settings["fmax"],
                dr=settings["dr"],
                logfile=None,
                trajectory=None,
                optimizer_logfile=None,
                local_minima_trajectory=None,
            )
            hopper.run(spec["cap"] if hops is None else int(hops))
            energy = _reported_energy(hopper, calculator)
        else:
            recorded = _recorded_minima_hopping()
            settings = spec["settings"]
            hopper = recorded(
                atoms,
                T0=settings["T0"],
                beta1=settings["beta1"],
                beta2=settings["beta2"],
                beta3=settings["beta3"],
                Ediff0=settings["Ediff0"],
                alpha1=settings["alpha1"],
                alpha2=settings["alpha2"],
                mdmin=int(settings["mdmin"]),
                minima_threshold=settings["minima_threshold"],
                timestep=settings["timestep"],
                fmax=settings["fmax"],
                optimizer=settings["optimizer"],
            )
            if settings["rng"] is not None:
                raise ValueError("MinimaHopping rng default is None")
            hopper()
            energy = _reported_energy(hopper, calculator)
    except TargetReached as exc:
        energy = exc.energy
    except Exception as exc:
        crashed = not _is_cap(exc)
        if crashed:
            error = f"{exc.__class__.__name__}: {exc}"[:500]
            traceback.print_exc()
        energy = _reported_energy(hopper, calculator)
    calls = int(calculator.optbench_counter["calls"])
    if crashed:
        judged = {
            "success": False,
            "failure": "crash",
            "reasons": ["crash"],
            "energy": energy if energy is not None and np.isfinite(energy) else None,
            "force_calls": float(calls),
        }
    else:
        judged = judge_globalopt(
            energy,
            calls,
            reference=spec["reference"],
            tolerance=spec["tolerance"],
            cap=spec["cap"],
        )
    print(
        f"{label} calls={calls} energy={judged['energy']} success={judged['success']} failure={judged['failure']}",
        flush=True,
    )
    return {
        "schema": "optbench.start/v2",
        "start": label,
        "problem": spec["problem_id"],
        "template": spec["template_id"],
        "code": "ASE",
        "code_version": CODE_VERSION,
        "commit": COMMIT,
        "tuning": "default",
        "counter": "harness",
        "comparable": True,
        "seed": SEED + index,
        "method": spec["method"],
        "success": judged["success"],
        "failure": judged["failure"],
        "reasons": judged["reasons"],
        "force_calls": calls,
        "energy": judged["energy"],
        "error": error,
        "seconds": time.perf_counter() - started,
    }


def start_index(label: str) -> int:
    base = str(label).rsplit("/", 1)[-1]
    if base.startswith("coords."):
        return int(base.split(".")[1])
    if base.endswith(".con"):
        return int(base[: -len(".con")])
    raise ValueError(f"unrecognised start label {label}")


def load_campaign_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_campaign(root: Path, records: list[dict], *, spec: dict, partial: bool) -> tuple[Path, Path]:
    cap = int(spec["cap"])
    summary = score_v2(records, cap)
    directory = root / "results" / "campaign"
    directory.mkdir(parents=True, exist_ok=True)
    stem = campaign_stem(spec["problem_id"], spec["method"])
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    calls = [int(record["force_calls"]) for record in records]
    if spec["method"] == "basin":
        settings = spec["settings"]
        detail = (
            f"# temperature {settings['temperature']:.16g} (100 * kB), dr {settings['dr']}, "
            f"LBFGS, fmax {settings['fmax']}. Seed {SEED} plus the start index."
        )
        title = "BasinHopping documentation example"
    else:
        settings = spec["settings"]
        detail = (
            f"# T0 {settings['T0']}, Ediff0 {settings['Ediff0']}, fmax {settings['fmax']}, "
            f"timestep {settings['timestep']}, mdmin {settings['mdmin']}, QuasiNewton. "
            f"Seed {SEED} plus the start index."
        )
        title = "MinimaHopping defaults"
    lines = [
        f"# ASE {CODE_VERSION} {title} on {spec['problem_id']}.",
        detail,
        "# The energy stop and the force-call cap are the problem file.",
        f"problem {spec['problem_id']}",
        f"template {spec['template_id']}",
        "code ASE",
        f"code_version {CODE_VERSION}",
        f"commit {COMMIT}",
        f"algorithm {'BasinHopping' if spec['method'] == 'basin' else 'MinimaHopping'}",
        "tuning default",
        "counter harness",
        f"n {summary['n']}",
        f"nfailed {summary['nfailed']}",
        f"par2_mean {_format(summary['par2_mean'])}",
        f"median {_format(summary['median'])}",
        f"median_low {_format(summary['median_low'])}",
        f"median_high {_format(summary['median_high'])}",
        f"wilson_low {_format(summary['wilson_low'])}",
        f"wilson_high {_format(summary['wilson_high'])}",
        f"success_rate {_format(summary['success_rate'])}",
        f"n_cap {summary['n_cap']}",
        f"n_wrong_stationary {summary['n_wrong_stationary']}",
        f"n_divergence {summary['n_divergence']}",
        f"n_crash {summary['n_crash']}",
        f"cap {cap}",
        f"force_calls_min {min(calls)}",
        f"force_calls_max {max(calls)}",
        f"force_calls_median {_format(float(np.median(calls)))}",
    ]
    if energies:
        lines.append(f"energy_min {_format(min(energies))}")
        lines.append(f"energy_max {_format(max(energies))}")
    if partial:
        lines.append("partial true")
    dat = directory / f"{stem}.dat"
    jsonl = directory / f"{stem}.starts.jsonl"
    _atomic_write(jsonl, "".join(json.dumps(record) + "\n" for record in records))
    _atomic_write(dat, "\n".join(lines) + "\n")
    return dat, jsonl


def inflight_limit(cap: int, short_remaining: bool) -> int:
    """Keep one worker on a long cap while a shorter problem still has starts."""
    if short_remaining and cap >= LONG_CAP:
        return 1
    return 10**9


def choose_job(
    queues: dict[tuple[str, str], deque],
    inflight: dict[tuple[str, str], int],
    done: dict[tuple[str, str], int],
    caps: dict[tuple[str, str], int],
) -> tuple[str, str] | None:
    short_remaining = any(
        (queues[key] or inflight[key]) and caps[key] < LONG_CAP for key in queues
    )
    ranked = []
    for key, queue in queues.items():
        if not queue:
            continue
        if inflight[key] >= inflight_limit(caps[key], short_remaining):
            continue
        uncovered = inflight[key] == 0 and done[key] == 0
        ranked.append((0 if uncovered else 1, inflight[key], caps[key], key))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][-1]


def _worker(payload: dict) -> dict:
    try:
        return search_one(
            payload["start"],
            problem_id=payload["problem_id"],
            method=payload["method"],
            root=Path(payload["root"]),
        )
    except Exception as exc:
        traceback.print_exc()
        start = payload["start"]
        return {
            "schema": "optbench.start/v2",
            "start": start["label"],
            "problem": payload["problem_id"],
            "template": template_id(payload["problem_id"], payload["method"]),
            "code": "ASE",
            "code_version": CODE_VERSION,
            "commit": COMMIT,
            "tuning": "default",
            "counter": "harness",
            "comparable": True,
            "seed": SEED + int(start["index"]),
            "method": payload["method"],
            "success": False,
            "failure": "crash",
            "reasons": ["crash"],
            "force_calls": 0,
            "energy": None,
            "error": f"{exc.__class__.__name__}: {exc}"[:500],
            "seconds": 0.0,
        }


def _store(campaign_root: Path, spec: dict, records: list[dict], record: dict) -> None:
    records.append(record)
    ordered = sorted(records, key=lambda item: start_index(str(item["start"])))
    partial = len(ordered) < int(spec["starts"])
    write_campaign(campaign_root, ordered, spec=spec, partial=partial)
    records[:] = ordered


def run(
    root: Path,
    jobs: list[tuple[str, str]],
    *,
    workers: int | None = None,
    limit: int | None = None,
    dest: Path | None = None,
    resume: bool = False,
) -> dict[tuple[str, str], list[dict]]:
    campaign_root = root if dest is None else dest
    if workers is None:
        workers = os.cpu_count() or 1
    queues: dict[tuple[str, str], deque] = {}
    inflight = {}
    done_counts = {}
    caps = {}
    specs = {}
    stored: dict[tuple[str, str], list[dict]] = {}
    for problem_id, method in jobs:
        spec = _spec(root, problem_id, method)
        specs[(problem_id, method)] = spec
        caps[(problem_id, method)] = int(spec["cap"])
        problem = load_problem(_problem_path(root, problem_id))
        starts = load_starts(geometry_archive(root, problem), problem_id, int(spec["atoms"]))
        if len(starts) != int(spec["starts"]):
            raise RuntimeError(
                f"{problem_id} archive has {len(starts)} starts, problem states {spec['starts']}"
            )
        jsonl = campaign_root / "results" / "campaign" / f"{campaign_stem(problem_id, method)}.starts.jsonl"
        prior = load_campaign_records(jsonl) if resume else []
        finished = {start_index(str(record["start"])) for record in prior}
        selected = [item for item in starts if item["index"] not in finished]
        if limit is not None:
            selected = selected[:limit]
        queues[(problem_id, method)] = deque(selected)
        inflight[(problem_id, method)] = 0
        done_counts[(problem_id, method)] = len(prior)
        stored[(problem_id, method)] = prior
        print(
            f"queue {problem_id} {method} cap={spec['cap']} pending={len(selected)} prior={len(prior)}",
            flush=True,
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    if all(not queue for queue in queues.values()):
        return stored

    def submit(pool, futures: dict) -> None:
        while len(futures) < workers:
            key = choose_job(queues, inflight, done_counts, caps)
            if key is None:
                return
            start = queues[key].popleft()
            inflight[key] += 1
            payload = {
                "start": start,
                "problem_id": key[0],
                "method": key[1],
                "root": str(root),
            }
            futures[pool.submit(_worker, payload)] = key

    futures: dict = {}
    if workers == 1:
        while True:
            key = choose_job(queues, inflight, done_counts, caps)
            if key is None:
                break
            start = queues[key].popleft()
            record = _worker(
                {"start": start, "problem_id": key[0], "method": key[1], "root": str(root)}
            )
            done_counts[key] += 1
            _store(campaign_root, specs[key], stored[key], record)
        return stored

    with ProcessPoolExecutor(max_workers=workers) as pool:
        submit(pool, futures)
        while futures:
            finished_set, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished_set:
                key = futures.pop(future)
                inflight[key] -= 1
                record = future.result()
                done_counts[key] += 1
                _store(campaign_root, specs[key], stored[key], record)
            submit(pool, futures)
    return stored


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ASE basin hopping and minima hopping")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--problem", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.all:
        jobs = [(problem_id, method) for problem_id in CARTESIAN for method in METHODS]
        for problem_id in RIGID:
            print(
                f"skip {problem_id}: rigid TIP4P is six coordinates per molecule, "
                "and BasinHopping and MinimaHopping move Cartesian atoms",
                flush=True,
            )
    else:
        if not args.problem or not args.method:
            raise SystemExit("pass --all or both --problem and --method")
        jobs = [(args.problem, args.method)]
    if args.dest is not None:
        args.dest.mkdir(parents=True, exist_ok=True)
    temperature = basin_settings()["temperature"]
    print(
        f"basin example temperature={temperature:.16g} dr=0.5 optimizer=LBFGS fmax=0.1",
        flush=True,
    )
    print(
        "minima defaults T0=1000 Ediff0=0.5 fmax=0.05 timestep=1 mdmin=2 optimizer=QuasiNewton",
        flush=True,
    )
    run(
        root,
        jobs,
        workers=args.workers or None,
        limit=args.limit or None,
        dest=args.dest,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
