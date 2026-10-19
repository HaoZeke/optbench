"""Global optimization with GASP-python and LAMMPS ``lj/cut``.

One start is one fixed-composition genetic search. The archived geometry
is the first member of the initial population. The other members and the
offspring are GASP's own defaults. The harness tally is the LAMMPS force
evaluations of ``pair_style lj/cut``: each ``Iterations, force evaluations``
line, plus one for the ``run 0`` that writes the dump GASP parses.

GASP 0.1 does not import on current pymatgen and PyYAML without these
shims: ``yaml.load`` needs a loader, ``Structure.modify_lattice`` was
removed, ``species_and_occu`` is now ``species``, ``Structure.to`` takes
the filename before the format, and ``Lattice.from_lengths_and_angles``
is ``from_parameters``. ``PeriodicSite.__eq__`` is the same tolerance
test without a NumPy call, because GASP looks sites up with
``list.index``. The energy parser
in that revision reads the second thermo line. This runner reads the
final value of ``Energy initial, next-to-last, final`` instead, and the
input sets ``thermo_modify norm no`` so that value is the extensive
Lennard-Jones energy. The example dump interval does not fit in a
32-bit LAMMPS integer, so the dump is written with ``run 0``.
``MoleculeMatcher.fit`` rebuilds an OpenBabel molecule for every pair.
A different InChI means the molecules differ, so that comparison is
cached; equal InChI still uses the original RMSD test. The pymatgen
molecule built for that test is reused while the cell coordinates are
unchanged. Freed allocator pages are returned to the OS after each
calculation.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance

GASP_COMMIT = "0c8d993c82e0e1c69a05b3c34bbb2fcbbdbb7f07"
GASP_VERSION = "0.1"
LAMMPS_COMMIT = "8de817dd79bfe4525d5d39246a212d833e6dee07"
# Wales LJ38 coordinates, quenched with this LAMMPS pin, pair_style lj/cut 40,
# pair_coeff * * 1 1, units lj, thermo_modify norm no. The problem stop is
# 0.001 above this energy.
LJ38_REFERENCE = -173.928426590629
SEED_BASE = 20261015
_FORCE_EVAL = re.compile(r"Iterations, force evaluations\s*=\s*(\d+)\s+(\d+)")
_FINAL_ENERGY = re.compile(
    r"Energy initial, next-to-last, final =\s*\n\s*([-+0-9.eE \t]+)"
)
_RUN0 = re.compile(r"Loop time of .* for 0 steps")


def search_force_cap(num_energy_calcs: int, min_eval: int) -> int:
    """One above the most force evaluations one search can record.

    A finished budget stays strictly under this cap, so exhausting the
    genetic-algorithm calculation limit is not a cap failure. The refused
    harness call is the extra one.
    """
    if num_energy_calcs <= 0 or min_eval <= 0:
        raise ValueError("energy calculations and min_eval must be positive")
    return num_energy_calcs * (min_eval + 1) + 1


def lammps_force_evaluations(log_text: str) -> int:
    """Force evaluations printed by minimize, plus one for a trailing ``run 0``."""
    total = 0
    for match in _FORCE_EVAL.finditer(log_text):
        total += int(match.group(2))
    if _RUN0.search(log_text):
        total += 1
    return total


def lammps_final_energy(log_text: str) -> float | None:
    """Last extensive energy in the minimization stats, or None."""
    matches = _FINAL_ENERGY.findall(log_text)
    if not matches:
        return None
    values = [float(part) for part in matches[-1].split()]
    if not values:
        return None
    return values[-1]


_STRUCTURE_FORMATS = frozenset(
    {
        "cif",
        "poscar",
        "cssr",
        "json",
        "yaml",
        "yml",
        "xsf",
        "mcsqs",
        "res",
        "pwmat",
        "aims",
        "prismatic",
        "fleur-inpgen",
    }
)


def _coords_match(left, right, atol: float) -> bool:
    """``np.allclose(left, right, atol=atol)`` for one length-3 coordinate."""
    rtol = 1e-5
    for index in range(3):
        sample = float(right[index])
        if abs(float(left[index]) - sample) > atol + rtol * abs(sample):
            return False
    return True


def _install_site_equality() -> None:
    """Same site equality GASP's ``list.index`` calls, without a NumPy entry.

    The boolean matches ``PeriodicSite.__eq__`` in pymatgen 2026.9.24.
    """
    from pymatgen.core.sites import PeriodicSite, Site
    from pymatgen.util.misc import is_np_dict_equal

    if getattr(PeriodicSite.__eq__, "_optbench_eq", False):
        return

    def eq(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if not _coords_match(self.coords, other.coords, Site.position_atol):
            return False
        if self.species != other.species or self.lattice != other.lattice:
            return False
        if not self.properties and not other.properties:
            return True
        return is_np_dict_equal(self.properties, other.properties)

    eq._optbench_eq = True
    PeriodicSite.__eq__ = eq


_TOPOLOGY: dict[tuple, str] = {}


def _molecule_key(mol) -> tuple:
    coords = tuple(float(value) for value in mol.cart_coords.reshape(-1))
    species = tuple(str(site.specie) for site in mol)
    return species, coords


def _topology_hash(mapper, mol) -> str:
    key = _molecule_key(mol)
    cached = _TOPOLOGY.get(key)
    if cached is not None:
        return cached
    from pymatgen.io.babel import BabelMolAdaptor

    ob_mol = BabelMolAdaptor(mol).openbabel_mol
    hashed = mapper.get_molecule_hash(ob_mol)
    _TOPOLOGY[key] = hashed
    return hashed


def _install_matcher_cache() -> None:
    """Skip repeated OpenBabel work when the InChI already differs.

    ``IsomorphismMolAtomMapper.uniform_labels`` returns no mapping when
    the InChI strings differ, and ``fit`` is then false. Matching InChI
    strings still go through the original RMSD comparison.
    """
    from pymatgen.analysis.molecule_matcher import IsomorphismMolAtomMapper, MoleculeMatcher

    if getattr(MoleculeMatcher.fit, "_optbench_cache", False):
        return
    original = MoleculeMatcher.fit

    def fit(self, mol1, mol2):
        mapper = getattr(self, "_mapper", None)
        if isinstance(mapper, IsomorphismMolAtomMapper):
            try:
                if _topology_hash(mapper, mol1) != _topology_hash(mapper, mol2):
                    return False
            except Exception:
                return original(self, mol1, mol2)
        return original(self, mol1, mol2)

    fit._optbench_cache = True
    MoleculeMatcher.fit = fit


_MOLECULES: dict[int, tuple] = {}
_RELEASES = 0


def _cached_molecule(cell):
    """The molecule ``match_molecules`` would build, kept while coordinates match."""
    import numpy as np
    from pymatgen.core.structure import Molecule

    coords = np.asarray(cell.cart_coords, dtype=np.float64)
    found = _MOLECULES.get(id(cell))
    if found is not None and found[0] == coords.shape and np.array_equal(found[1], coords):
        return found[2]
    molecule = Molecule(cell.species, coords)
    _MOLECULES[id(cell)] = (coords.shape, coords.copy(), molecule)
    return molecule


def _install_molecule_cache() -> None:
    """Build each redundancy molecule once.

    GASP compares a new structure with every structure it has kept, and
    each comparison constructs two molecules. The coordinates are checked
    before reuse, so a later edit of the cell does not keep a stale geometry.
    """
    import gasp.development as development

    guard = development.RedundancyGuard
    if getattr(guard.match_molecules, "_optbench_mol", False):
        return

    def match_molecules(self, cell1, cell2):
        return self.molecule_matcher.fit(_cached_molecule(cell1), _cached_molecule(cell2))

    match_molecules._optbench_mol = True
    guard.match_molecules = match_molecules


def _release_heap() -> None:
    """Return freed malloc pages so a long search does not keep them."""
    global _RELEASES
    import ctypes
    import gc

    _RELEASES += 1
    if _RELEASES % 10:
        return
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def install_shims() -> None:
    """Make GASP 0.1 importable. Safe to call more than once."""
    import yaml
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.sites import PeriodicSite, Site
    from pymatgen.core.structure import Structure

    _install_matcher_cache()

    if not hasattr(Lattice, "from_lengths_and_angles"):

        def from_lengths_and_angles(cls, lengths, angles, **kwargs):
            a, b, c = (float(value) for value in lengths)
            alpha, beta, gamma = (float(value) for value in angles)
            return cls.from_parameters(a, b, c, alpha, beta, gamma, **kwargs)

        Lattice.from_lengths_and_angles = classmethod(from_lengths_and_angles)
    _install_site_equality()

    if not hasattr(Structure, "modify_lattice"):

        def modify_lattice(self, new_lattice):
            self.lattice = new_lattice

        Structure.modify_lattice = modify_lattice
    if not hasattr(PeriodicSite, "species_and_occu"):
        PeriodicSite.species_and_occu = property(lambda self: self.species)
    if not hasattr(Site, "species_and_occu"):
        Site.species_and_occu = property(lambda self: self.species)
    if not getattr(Structure.to, "_optbench_filename", False):
        original_to = Structure.to

        def to(self, *args, **kwargs):
            # GASP 0.1 calls ``to("poscar", path)``. pymatgen now takes the path first.
            if (
                args
                and isinstance(args[0], str)
                and args[0].lower() in _STRUCTURE_FORMATS
                and "fmt" not in kwargs
            ):
                filename = args[1] if len(args) > 1 else kwargs.pop("filename", "")
                return original_to(self, filename, args[0], *args[2:], **kwargs)
            return original_to(self, *args, **kwargs)

        to._optbench_filename = True
        Structure.to = to
    if not getattr(yaml.load, "_optbench_safe", False):
        original = yaml.load

        def load(stream, Loader=None):
            if Loader is None:
                from yaml import SafeLoader

                Loader = SafeLoader
            return original(stream, Loader=Loader)

        load._optbench_safe = True
        yaml.load = load


class _Tally:
    def __init__(self, cap: int, energy_limit: float | None):
        self.counter = HarnessCounter(cache=False, cap=cap)
        self.cap = cap
        self.energy_limit = energy_limit
        self.best: float | None = None
        self.launches = 0
        self.failed_calcs = 0
        self.stopping = None
        self.capped = False
        self.abort = ""

    def note(self, energy: float) -> None:
        if not math.isfinite(energy):
            return
        if self.best is None or energy < self.best:
            self.best = float(energy)
        if (
            self.energy_limit is not None
            and self.best <= self.energy_limit
            and self.stopping is not None
        ):
            self.stopping.are_satisfied = True

    def charge(self, count: int) -> None:
        for _ in range(count):
            try:
                self.counter.charge((float(self.counter.calls),))
            except ForceCallCap:
                self.capped = True
                if self.stopping is not None:
                    self.stopping.are_satisfied = True
                return


_LAMMPS_BASE = None
_ORIGINAL_MAKE = None


def _counting_calculator():
    """Subclass of GASP's original LAMMPS calculator.

    The replacement is stored on the GASP module. A later search in the
    same process must still subclass the original calculator. Subclassing
    the replacement stacks the stop check, and an earlier energy limit
    then ends the next search on its first calculation.
    """
    global _LAMMPS_BASE
    import gasp.energy_calculators as energy_calculators

    if _LAMMPS_BASE is None:
        _LAMMPS_BASE = energy_calculators.LammpsEnergyCalculator

    class CountingLammps(_LAMMPS_BASE):
        tally: _Tally | None = None

        def get_energy(self, lammps_log_path):
            text = Path(lammps_log_path).read_text(encoding="utf-8", errors="replace")
            energy = lammps_final_energy(text)
            if energy is None:
                raise ValueError(f"no minimization energy in {lammps_log_path}")
            return energy

        def do_energy_calculation(self, organism, dictionary, key, composition_space):
            _release_heap()
            tally = CountingLammps.tally
            if tally is None:
                raise RuntimeError("GASP tally is not set")
            if tally.capped or tally.counter.calls >= tally.cap:
                tally.capped = True
                if tally.stopping is not None:
                    tally.stopping.are_satisfied = True
                dictionary[key] = None
                return
            tally.launches += 1
            try:
                super().do_energy_calculation(organism, dictionary, key, composition_space)
            except Exception:
                dictionary[key] = None
            log_path = Path.cwd() / "temp" / str(organism.id) / "log.lammps"
            if log_path.is_file():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                tally.charge(lammps_force_evaluations(text))
                energy = lammps_final_energy(text)
                if energy is not None:
                    tally.note(energy)
            if dictionary.get(key) is None:
                tally.failed_calcs += 1

    return CountingLammps


class _InlineThread:
    """Run the energy calculation on this thread.

    GASP's loop polls ``is_alive``. An inline start has already finished,
    so the poll does not spin.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = {} if kwargs is None else kwargs
        self._alive = False
        self.daemon = daemon

    def start(self):
        self._alive = True
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        finally:
            self._alive = False

    def is_alive(self):
        return self._alive


def _lammps_script(min_eval: int, cutoff: float) -> str:
    return (
        "units lj\n"
        "atom_style charge\n"
        "boundary f f f\n"
        "read_data in.data\n"
        "\n"
        f"pair_style lj/cut {cutoff}\n"
        "pair_coeff * * 1.0 1.0\n"
        "\n"
        "neigh_modify delay 0 every 1 check yes one 5000\n"
        "thermo_style custom step temp epair emol etotal\n"
        "thermo_modify norm no\n"
        "thermo 100000\n"
        "\n"
        "min_style cg\n"
        f"minimize 0.0 1.0e-8 1000000 {int(min_eval)}\n"
        "\n"
        "dump myDump all atom 1 dump.atom\n"
        "dump_modify myDump sort 1 scale no\n"
        "run 0\n"
    )


def _input_yaml(job: dict, folder: Path, script: Path) -> str:
    lines = [
        "CompositionSpace:",
        f"    - {job['symbol']}",
        "",
        "EnergyCode:",
        "    lammps:",
        f"        input_script: {script}",
        "",
        "Constraints:",
        f"    min_num_atoms: {job['natoms']}",
        f"    max_num_atoms: {job['natoms']}",
        "",
        "Geometry:",
        "    shape: cluster",
        "",
        "InitialPopulation:",
    ]
    random_organisms = int(job["random_organisms"])
    if random_organisms > 0:
        lines.append("    random:")
        lines.append(f"        number: {random_organisms}")
    lines.append("    from_files:")
    lines.append(f"        path_to_folder: {folder}")
    lines.append("")
    lines.append("StoppingCriteria:")
    lines.append(f"    num_energy_calcs: {int(job['num_energy_calcs'])}")
    if job.get("energy_limit") is not None:
        epa = float(job["energy_limit"]) / int(job["natoms"])
        lines.append(f"    epa_achieved: {epa:.16g}")
    lines.append("")
    lines.append("NumCalcsAtOnce: 1")
    lines.append("RunTitle: optbench")
    lines.append("")
    return "\n".join(lines)


def _write_poscar(path: Path, coords: list[list[float]], symbol: str) -> None:
    import numpy as np
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1.0) + 1.0
    shifted = points - low + 0.5
    lattice = Lattice.from_parameters(float(span[0]), float(span[1]), float(span[2]), 90, 90, 90)
    structure = Structure(lattice, [symbol] * len(shifted), shifted, coords_are_cartesian=True)
    structure.to(filename=str(path), fmt="poscar")


def _find_gasp(template: dict) -> Path:
    env = os.environ.get("OPTBENCH_GASP")
    candidates = [Path(env)] if env else []
    candidates.append(Path("/tmp/GASP-python"))
    pinned = str(template.get("commit") or GASP_COMMIT)
    for root in candidates:
        if not (root / "gasp" / "scripts" / "run.py").is_file():
            continue
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != pinned:
            raise RuntimeError(f"{root} is {commit}, template pins {pinned}")
        return root
    raise RuntimeError("GASP checkout not found. Set OPTBENCH_GASP to the pinned source tree.")


def _find_lammps() -> Path:
    env = os.environ.get("OPTBENCH_LAMMPS")
    candidates = [Path(env)] if env else []
    candidates.append(Path("/tmp/lammps/build/lmp"))
    found = shutil.which("lmp")
    if found:
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise RuntimeError("LAMMPS binary not found. Set OPTBENCH_LAMMPS.")


def _prepare_caller(work: Path, lammps: Path) -> Path:
    bindir = work / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    script = bindir / "calllammps"
    script.write_text(
        "#!/bin/sh\n"
        "cd \"$(dirname \"$1\")\" || exit 1\n"
        f"exec {lammps} -in \"$(basename \"$1\")\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bindir


def search_job(job: dict) -> dict:
    """Run one GASP search. ``job`` is plain data so a process pool can carry it."""
    import contextlib

    start_dir = Path(job["work"]) / f"start-{int(job['index']):05d}"
    if start_dir.exists():
        shutil.rmtree(start_dir)
    start_dir.mkdir(parents=True)
    out_path = start_dir / "gasp.out"
    tally = _Tally(int(job["cap"]), job.get("energy_limit"))
    previous = Path.cwd()
    argv = sys.argv[:]
    try:
        gasp_root = Path(job["gasp_root"])
        if str(gasp_root) not in sys.path:
            sys.path.insert(0, str(gasp_root))
        install_shims()
        os.environ["PATH"] = str(job["caller_dir"]) + os.pathsep + os.environ.get("PATH", "")
        os.environ["OMP_NUM_THREADS"] = "1"
        random.seed(int(job["seed"]))
        folder = start_dir / "structs"
        folder.mkdir()
        _write_poscar(folder / "POSCAR", job["coords"], str(job["symbol"]))
        script = start_dir / "in.min"
        script.write_text(_lammps_script(int(job["min_eval"]), float(job["cutoff"])), encoding="utf-8")
        yaml_path = start_dir / "ga.yaml"
        yaml_path.write_text(_input_yaml(job, folder, script), encoding="utf-8")
        counting = _counting_calculator()
        counting.tally = tally
        import gasp.energy_calculators as energy_calculators
        import gasp.objects_maker as objects_maker
        import gasp.scripts.run as run
        import threading

        energy_calculators.LammpsEnergyCalculator = counting
        _install_molecule_cache()
        global _ORIGINAL_MAKE
        if _ORIGINAL_MAKE is None:
            _ORIGINAL_MAKE = objects_maker.make_objects

        def make_objects(parameters):
            made = _ORIGINAL_MAKE(parameters)
            tally.stopping = made["stopping_criteria"]
            return made

        objects_maker.make_objects = make_objects
        run.objects_maker.make_objects = make_objects
        threading.Thread = _InlineThread
        sys.argv = ["run.py", str(yaml_path)]
        os.chdir(start_dir)
        with out_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                run.main()
            except SystemExit:
                pass
        text = out_path.read_text(encoding="utf-8", errors="replace") if out_path.is_file() else ""
        satisfied = tally.stopping is not None and bool(tally.stopping.are_satisfied)
        reached = (
            tally.energy_limit is not None
            and tally.best is not None
            and tally.best <= float(tally.energy_limit)
        )
        if "Quitting..." in text and not satisfied and not reached:
            tally.abort = "gasp quit before the stopping criteria"
    except Exception as exc:
        tally.abort = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(previous)
        sys.argv = argv
    record = _record(job, tally)
    if not tally.abort:
        shutil.rmtree(start_dir, ignore_errors=True)
    return record


def _record(job: dict, tally: _Tally) -> dict:
    calls = int(tally.counter.calls)
    cap = int(job["cap"])
    energy = tally.best
    limit = job.get("energy_limit")
    finite = energy is not None and math.isfinite(energy)
    reasons: list[str] = []
    if tally.abort:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif calls >= cap or tally.capped:
        reasons.append("cap")
    elif limit is not None and float(energy) > float(limit):
        reasons.append("energy")
    failure = None
    success = None
    if limit is None and not tally.abort and finite and calls < cap and not tally.capped:
        success = None
        failure = None
        reasons = []
    else:
        failure = classify_failure(reasons)
        success = failure is None
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": int(job["index"]),
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": limit is not None,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "energy_calcs": int(tally.launches),
        "failed_calcs": int(tally.failed_calcs),
        "seed": int(job["seed"]),
        "cap": cap,
    }
    if tally.abort:
        record["error"] = tally.abort
    if limit is None:
        record["ruled"] = False
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict | None, commit: str, version: str, *, partial: str) -> str:
    lines = [
        "# counter harness. One force call is one LAMMPS lj/cut force evaluation.",
        "# A minimize line contributes its force-evaluation count. The dump's run 0 contributes one.",
        "# GASP 0.1 runs with the yaml, lattice, and species shims, and the energy is the final minimization stat.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    if partial:
        lines.append("# " + partial)
    rows = {
        "force_calls": None if summary is None else summary.get("force_calls"),
        "force_calls_min": None if summary is None else summary.get("force_calls_min"),
        "force_calls_max": None if summary is None else summary.get("force_calls_max"),
        "nfailed": None if summary is None else summary.get("nfailed"),
        "median": None if summary is None else summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
    }
    if metrics is not None:
        rows.update(
            {
                "par2_mean": metrics.get("par2_mean"),
                "wilson_low": metrics.get("wilson_low"),
                "wilson_high": metrics.get("wilson_high"),
                "par2_median": metrics.get("median"),
                "par2_median_low": metrics.get("median_low"),
                "par2_median_high": metrics.get("median_high"),
                "v2_nfailed": metrics.get("nfailed"),
                "v2_successes": metrics.get("successes"),
                "n": metrics.get("n"),
                "cap": metrics.get("cap"),
            }
        )
    if summary is not None and summary.get("energy_min") is not None:
        rows["energy_min"] = summary.get("energy_min")
        rows["energy_max"] = summary.get("energy_max")
    if partial:
        rows["partial"] = "true"
        rows["ranked"] = "false"
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _gasp_section(template: dict) -> dict:
    section = template.get("gasp") or {}
    if "min_eval" not in section or "num_energy_calcs" not in section:
        raise ValueError(f"{template['id']} needs gasp.min_eval and gasp.num_energy_calcs")
    return section


def _energy_limit(problem: dict, template: dict) -> float | None:
    convergence = problem.get("convergence") or {}
    if "energy_above_minimum" not in convergence:
        return None
    section = _gasp_section(template)
    if "energy_minimum" not in section:
        raise ValueError(f"{template['id']} needs gasp.energy_minimum for {problem['id']}")
    return float(section["energy_minimum"]) + float(convergence["energy_above_minimum"])


def _archive_for(root: Path, problem: dict, template: dict) -> Path:
    from optbench.campaign import geometry_archive

    archive_path = (problem.get("input") or {}).get("archive_path")
    if archive_path:
        return geometry_archive(root, problem)
    named = str((_gasp_section(template).get("geometries") or ""))
    if not named:
        raise ValueError(f"{problem['id']} has no archive and {template['id']} has no gasp.geometries")
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == named:
            return root / row["stored"]
    raise FileNotFoundError(f"{named} is not in the geometry manifest")


def _member_key(name: str) -> tuple:
    base = Path(name).name
    stem = Path(base).stem
    if stem.isdigit():
        return (0, int(stem), base)
    tail = base.rsplit(".", 1)[-1]
    if tail.isdigit():
        return (0, int(tail), base)
    return (1, 0, base)


def _start_name(path: Path) -> str:
    """Geometry label. ``12.con`` stays ``12``. ``coords.12`` stays ``12``."""
    stem = path.stem
    if stem.isdigit():
        return stem
    tail = path.name.rsplit(".", 1)[-1]
    if tail.isdigit():
        return tail
    return stem


def _coords_from_text(name: str, text: str) -> list[list[float]]:
    if name.endswith(".con"):
        from optbench.potentials import read_con

        pos, _box = read_con(text)
        return pos.reshape(-1, 3).tolist()
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{name} has no coordinates")
    return rows


def _load_starts(root: Path, problem: dict, template: dict, work: Path) -> list[tuple[str, list[list[float]]]]:
    archive = _archive_for(root, problem, template)
    dest = work / "geometries" / problem["id"].replace("/", "-")
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != archive.name:
        if dest.exists():
            for child in dest.iterdir():
                if child.name == ".extracted":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
        marker.write_text(archive.name, encoding="utf-8")
    members = [
        path
        for path in dest.rglob("*")
        if path.is_file() and path.name != ".extracted" and not path.name.endswith(".xyz")
    ]
    # Prefer .con when both .con and .xyz are present (the lj38 archive has both).
    cons = [path for path in members if path.name.endswith(".con")]
    if cons:
        members = cons
    else:
        members = [path for path in members if path.suffix != ".xyz"]
    members.sort(key=lambda path: _member_key(path.name))
    if not members:
        raise FileNotFoundError(f"{archive} has no geometries")
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(members) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(members)}")
    loaded = []
    for path in members:
        loaded.append((_start_name(path), _coords_from_text(path.name, path.read_text(encoding="utf-8", errors="replace"))))
    return loaded


def _jobs(root: Path, problem: dict, template: dict, work: Path, *, limit: int) -> list[dict]:
    section = _gasp_section(template)
    num_energy_calcs = int(section["num_energy_calcs"])
    min_eval = int(section["min_eval"])
    cap = search_force_cap(num_energy_calcs, min_eval)
    stated = template.get("failure_cap")
    if stated is None or int(stated) != cap:
        raise ValueError(f"{template['id']} failure_cap must be {cap}")
    limit_energy = _energy_limit(problem, template)
    natoms = int(section["natoms"])
    symbol = str(section.get("symbol") or "C")
    random_organisms = int(section.get("random_organisms", 27))
    cutoff = float(section.get("cutoff", 40.0))
    seed_base = int(section.get("seed", SEED_BASE))
    gasp_root = _find_gasp(template)
    lammps = _find_lammps()
    caller = _prepare_caller(work, lammps)
    starts = _load_starts(root, problem, template, work)
    if limit > 0:
        starts = starts[:limit]
    jobs = []
    for index, (name, coords) in enumerate(starts):
        if len(coords) != natoms:
            raise ValueError(f"{name} has {len(coords)} atoms, template says {natoms}")
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords,
                "natoms": natoms,
                "symbol": symbol,
                "work": str(work),
                "seed": seed_base + index,
                "cap": cap,
                "num_energy_calcs": num_energy_calcs,
                "min_eval": min_eval,
                "cutoff": cutoff,
                "random_organisms": random_organisms,
                "energy_limit": limit_energy,
                "gasp_root": str(gasp_root),
                "caller_dir": str(caller),
            }
        )
    return jobs


def _write_outputs(output: Path, template: dict, records: list[dict], *, partial: str, commit: str, version: str) -> dict:
    cap = int(records[0]["cap"]) if records else None
    ruled = any(record.get("success") is not None or record.get("ruled") is False for record in records)
    has_rule = any(record.get("comparable") for record in records)
    metrics = None
    summary: dict = {"starts": len(records), "template": template["id"], "problem": template.get("problem")}
    if has_rule and records:
        metrics = score_v2(
            [
                {
                    "force_calls": record["force_calls"],
                    "success": record["success"],
                    "failure": record["failure"],
                }
                for record in records
            ],
            cap,
        )
        legacy_counts = [
            cap if not record["success"] else int(record["force_calls"]) for record in records
        ]
        summary.update(summarize_capped(legacy_counts, cap))
        successes = [int(record["force_calls"]) for record in records if record["success"]]
        summary["median"] = statistics.median(successes) if successes else None
        summary["par2_mean"] = metrics["par2_mean"]
        summary["v2_nfailed"] = metrics["nfailed"]
        summary["nfailed"] = metrics["nfailed"]
    elif records:
        calls = [int(record["force_calls"]) for record in records]
        summary["force_calls"] = None
        summary["force_calls_min"] = min(calls)
        summary["force_calls_max"] = max(calls)
        summary["median"] = statistics.median(calls)
        summary["nfailed"] = None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output,
        _dat(summary if records else None, template, metrics, commit, version, partial=partial),
    )
    public = []
    for record in records:
        row = dict(record)
        row.pop("index", None)
        if row.get("success") is None:
            row.pop("success", None)
            row.pop("failure", None)
        public.append(row)
    _atomic_text(
        output.with_suffix(".starts.jsonl"),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in public),
    )
    if has_rule and metrics is not None:
        document = {
            "schema": "optbench.result/v2",
            "status": "archive",
            "problem": template.get("problem"),
            "scoring": "par2-wilson-bootstrap",
            "par_factor": 2,
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 20261015,
            "note": "Harness counter. Each call is a LAMMPS lj/cut force evaluation.",
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
                    "parameters": {
                        "num_energy_calcs": int(_gasp_section(template)["num_energy_calcs"]),
                        "min_eval": int(_gasp_section(template)["min_eval"]),
                    },
                    "provenance": collect_provenance(
                        code=str(template.get("code") or "gasp"),
                        code_version=version,
                        commit=commit,
                        parameters={
                            "num_energy_calcs": int(_gasp_section(template)["num_energy_calcs"]),
                            "min_eval": int(_gasp_section(template)["min_eval"]),
                        },
                    ),
                    "metrics": metrics,
                    "starts": public,
                }
            ],
        }
        _atomic_text(
            output.with_suffix(".v2.json"),
            json.dumps(document, indent=2, sort_keys=True) + "\n",
        )
    summary["starts"] = len(records)
    summary["ruled"] = ruled and has_rule
    return summary


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
) -> dict:
    if geometries is not None:
        raise ValueError("gasp reads the problem archive")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    version = str(template.get("code_version") or GASP_VERSION)
    if version != GASP_VERSION:
        raise RuntimeError(f"{template['id']} version {version} is not GASP {GASP_VERSION}")
    commit = str(template.get("commit") or "")
    if commit != GASP_COMMIT:
        raise RuntimeError(f"{template['id']} commit is not {GASP_COMMIT}")
    work_dir = work or output.parent / "gasp-work" / template["id"].replace("/", "-")
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(root, problem, template, work_dir, limit=limit)
    record_dir = work_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    records = []
    for job in jobs:
        cached = record_dir / f"{int(job['index']):05d}.json"
        if cached.is_file():
            records.append(json.loads(cached.read_text(encoding="utf-8")))
        else:
            pending.append(job)

    def emit() -> dict:
        ordered = sorted(records, key=lambda row: int(row["index"]))
        return _write_outputs(
            output,
            template,
            ordered,
            partial=_partial_label(problem, ordered, limit),
            commit=commit,
            version=version,
        )

    if pending:
        done = 0
        total = len(pending)

        def accept(record: dict) -> None:
            nonlocal done
            done += 1
            _store_record(record_dir, record)
            records.append(record)
            emit()
            print(_progress(template["id"], done, total, record), flush=True)

        if workers <= 1:
            for job in pending:
                accept(search_job(job))
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                max_tasks_per_child=1,
            ) as pool:
                futures = {pool.submit(search_job, job): job for job in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = _record(job, _Tally(int(job["cap"]), job.get("energy_limit")))
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        record["success"] = False
                        record["failure"] = "crash"
                        record["reasons"] = ["crash"]
                        record["comparable"] = job.get("energy_limit") is not None
                    accept(record)
    return emit()


def _partial_label(problem: dict, records: list[dict], limit: int) -> str:
    if not records:
        return ""
    if not any(record.get("comparable") for record in records):
        return "no success rule on this problem; rows are the lowest energy and the harness calls"
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(records) < int(expected):
        return f"scored {len(records)} of {int(expected)} starts"
    if limit > 0 and (expected is None or len(records) < int(expected or limit)):
        return f"scored {len(records)} starts"
    return ""


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _store_record(record_dir: Path, record: dict) -> None:
    path = record_dir / f"{int(record['index']):05d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _progress(template_id: str, done: int, total: int, record: dict) -> str:
    energy = record.get("energy")
    shown = "none" if energy is None else f"{float(energy):.6f}"
    return (
        f"{template_id} {done}/{total} start {record.get('start')} "
        f"calls {record.get('force_calls')} energy {shown} "
        f"success {record.get('success')} failure {record.get('failure')}"
    )
