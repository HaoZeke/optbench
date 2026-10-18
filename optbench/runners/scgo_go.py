"""Global optimization with SCGO on the harness Lennard-Jones potential.

SCGO 0.9.1 speaks ASE. The campaign attaches one calculator whose
``calculate`` calls ``optbench.potentials.lennard_jones`` through a
``HarnessCounter``. ASE forces are the negative of that gradient. A
repeated coordinate is one call.

Basin hopping is ``scgo.algorithms.basinhopping_go.bh_go``. For a
38-atom cluster that is the routine that starts from one geometry.
``run_go`` would build a genetic-algorithm population and drop the
archived start. Each file in the problem archive is one start.

The library budget for 38 atoms is 131 hops and 233 FIRE steps. A
start ends early when a quenched energy lies within
``energy_above_minimum`` of the harness minimum. The force-call cap is
the most evaluations that budget can spend. A search that uses the cap
is a cap failure. A quenched energy outside the window is
``wrong_stationary``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import statistics
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

# Cambridge Cluster Database LJ points/38, quenched on
# optbench.potentials.lennard_jones until the force norm is about 1e-5.
# Twelve digits reproduce that quenched energy.
LJ38_MINIMUM_COORDS: tuple[tuple[float, float, float], ...] = (
    (0.194786967488, 0.330631219999, 1.706921685903),
    (1.159223029557, -1.151441027825, -0.625480994283),
    (1.48512027671, -0.067619506026, 0.922326402889),
    (-0.14979442702, 1.442512072747, -0.978550635955),
    (1.427730642954, 0.352999124864, -0.947532465218),
    (-0.688117861633, -1.573698362227, -0.332878499922),
    (-1.427730623949, -0.352999111742, 0.947532447139),
    (0.688117863551, 1.573698355775, 0.332878506111),
    (-1.159223044972, 1.151441012345, 0.625480985936),
    (0.149794441215, -1.442512071987, 0.978550628748),
    (-1.485120278659, 0.067619480609, -0.922326422061),
    (-0.70569983852, 0.620714131272, -1.475649788273),
    (-0.874520727403, 0.46481410202, 1.442221276118),
    (-0.974207418674, -0.883725900769, -1.153598328064),
    (-0.194786989428, -0.330631228708, -1.706921683139),
    (0.376004907876, -1.707230599198, -0.069461554634),
    (-1.712432976066, 0.333627824022, 0.130792177843),
    (0.914328358012, 1.308979885041, -0.715133697329),
    (-0.376004935133, 1.707230606672, 0.06946156128),
    (1.712432984879, -0.33362782162, -0.130792209607),
    (-0.914328349747, -1.308979884463, 0.715133704686),
    (0.974207435629, 0.883725863968, 1.153598338252),
    (0.705699857705, -0.620714146803, 1.475649796843),
    (0.874520698081, -0.464814072988, -1.442221253319),
    (-1.19547839912, -0.617191566375, -0.102143565514),
    (0.091734325894, -1.014479138363, -0.884841348185),
    (0.927620181933, -0.883610130988, 0.423416158256),
    (1.19547840045, 0.61719156649, 0.102143557348),
    (-0.927620183274, 0.883610135126, -0.423416181947),
    (-0.359592538458, -0.486322542431, 1.206113962843),
    (0.359592543928, 0.486322546505, -1.206113944074),
    (-0.091734319868, 1.014479156693, 0.884841361633),
    (0.641069978874, -0.197860956718, -0.389806630013),
    (-0.4162958555, -0.065176629508, -0.651550895411),
    (0.133401297866, 0.747443602305, -0.160003250342),
    (-0.641069973251, 0.197860969204, 0.389806638982),
    (0.416295853479, 0.065176636405, 0.651550900494),
    (-0.133401305406, -0.747443593321, 0.160003256185),
)

CODE_VERSION = "0.9.1"
PINNED_COMMIT = "2338742a434a920ed82e52a72dc1338cf06647c1"


class SearchStop(Exception):
    """The search has a quenched target, or it has spent the force budget.

    ``RuntimeError`` would be swallowed by SCGO's local relaxation and
    turned into a penalty energy. This exception leaves the relaxation.
    """

    def __init__(self, kind: str, energy: float | None, force_norm: float | None):
        super().__init__(kind)
        self.kind = kind
        self.energy = energy
        self.force_norm = force_norm


def lj38_reference_energy() -> tuple[float, float]:
    """Harness energy and force norm of the quenched Cambridge LJ38 geometry."""
    coords = np.asarray(LJ38_MINIMUM_COORDS, dtype=np.float64).reshape(-1)
    energy, gradient = lennard_jones(coords)
    return float(energy), float(np.linalg.norm(gradient))


def default_bh_budget(n_atoms: int) -> tuple[int, int, int]:
    """Hops, FIRE steps, and force-call cap for SCGO 0.9.1 automatic sizes.

    The hop count is ``int(clip(3 + 35 log1p(n), 3, 1000))``. The local
    step count is ``int(clip(50 + 50 log1p(n), 50, 2000))``. Those are
    ``auto_niter`` and ``auto_niter_local_relaxation``. The cap allows
    one force evaluation per FIRE step for the opening quench and every
    hop, plus the evaluation on which the force tolerance is met.
    """
    count = max(int(n_atoms), 1)
    scaled = math.log1p(count)
    niter = int(min(1000, max(3, 3 + 35 * scaled)))
    nlocal = int(min(2000, max(50, 50 + 50 * scaled)))
    cap = (niter + 1) * (nlocal + 1)
    return niter, nlocal, cap


def reference_energy(problem: dict) -> float | None:
    """Known minimum used by ``energy_above_minimum``.

    A number on the problem file wins. Lennard-Jones 38 otherwise uses
    the harness energy of the quenched Cambridge geometry.
    """
    convergence = problem.get("convergence") or {}
    potential = problem.get("potential") or {}
    for source in (convergence, potential):
        for key in ("energy_minimum", "minimum_energy", "known_minimum"):
            if key in source and source[key] not in (None, ""):
                return float(source[key])
    if problem.get("id") == "globalopt/lj38" and "energy_above_minimum" in convergence:
        energy, _norm = lj38_reference_energy()
        return energy
    return None


def _geometry_key(path: Path) -> tuple:
    name = path.name
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), name)
    head, dot, tail = name.rpartition(".")
    if dot and tail.isdigit() and head:
        return (0, int(tail), name)
    return (1, name, name)


def _read_points(text: str) -> list[float]:
    """Whitespace xyz rows. Blank lines are skipped."""
    rows: list[float] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.extend((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if len(rows) < 3 or len(rows) % 3:
        raise ValueError("coordinate file has no xyz rows")
    return rows


def _coordinate_files(dest: Path) -> list[Path]:
    cons = sorted(dest.rglob("*.con"), key=_geometry_key)
    if cons:
        return cons
    found = []
    for path in dest.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        name = path.name
        if name.startswith("coords.") or name.endswith(".xyz"):
            found.append(path)
    return sorted(found, key=_geometry_key)


def load_geometries(
    root: Path,
    problem: dict,
    geometries: Path | None,
    work: Path | None,
    output: Path,
    archive_path: str | None = None,
):
    """Start files in numeric order, flat coordinates, and the box when present.

    ``.con`` files use the eOn reader. A tarball of ``coords.N`` files is
    the Lennard-Jones 75 cluster set stored in the geometry manifest.
    """
    from optbench.campaign import geometry_archive

    if geometries is None:
        source = problem
        if archive_path and not (problem.get("input") or {}).get("archive_path"):
            source = {"id": problem["id"], "input": {"archive_path": archive_path}}
        archive = geometry_archive(root, source)
        dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
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
        cons = _coordinate_files(dest)
        if not cons:
            raise FileNotFoundError(f"{archive} has no start files")
    else:
        cons = _coordinate_files(geometries)
        if not cons:
            raise FileNotFoundError(f"{geometries} has no start files")
    loaded = []
    for path in cons:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".con":
            pos, box = read_con(text)
            coords = pos.reshape(-1).tolist()
            box_list = None if box is None else box.tolist()
            name = path.stem
        else:
            coords = _read_points(text)
            box_list = None
            name = path.name
        loaded.append((name, coords, box_list))
    return loaded


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, *, partial: bool, energy_rule: bool) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
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
        "cap": metrics.get("cap"),
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
    }
    lines = [
        "# counter harness. One force call is one ASE calculate on optbench.potentials.lennard_jones.",
        (
            "# A start succeeds when a quenched energy is within energy_above_minimum of the known minimum."
            if energy_rule
            else "# The problem states no energy_above_minimum. success means a finite quench under the FIRE force tolerance. energy is that quenched energy, or the lowest energy seen when the force never met fmax."
        ),
        "# The force-call cap is the default basin-hopping budget, counted in FIRE evaluations.",
    ]
    if partial:
        if energy_rule:
            lines.append("# partial true. This file has fewer starts than the problem sample.")
        else:
            lines.append(
                "# partial true. The problem states no energy_above_minimum, so this row is not a global-minimum score."
            )
        rows["partial"] = "true"
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _silence_scgo() -> None:
    logging.getLogger("scgo").setLevel(logging.ERROR)
    logging.getLogger("ase").setLevel(logging.ERROR)


def _one_start(job: dict) -> dict:
    """One basin-hopping search. The record is the harness tally."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _silence_scgo()
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    from scgo.algorithms.basinhopping_go import bh_go
    from scgo.utils.helpers import auto_niter, auto_niter_local_relaxation
    from scgo.utils.optimizer_utils import get_optimizer_class
    from scgo.utils.rng_helpers import ensure_rng

    composition = [str(job["symbol"])] * int(job["n_atoms"])
    niter = auto_niter(composition)
    nlocal = auto_niter_local_relaxation(composition)
    expected_niter, expected_local, expected_cap = default_bh_budget(int(job["n_atoms"]))
    if (niter, nlocal) != (expected_niter, expected_local):
        raise RuntimeError(
            f"SCGO auto budget {(niter, nlocal)} does not match the 0.9.1 formula {(expected_niter, expected_local)}"
        )
    cap = int(job["cap"])
    if cap != expected_cap:
        raise RuntimeError(f"template cap {cap} is not the 0.9.1 budget {expected_cap}")

    class _LennardJones(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.counter = HarnessCounter(cache=True, cap=cap)
            self.fmax = float(job["fmax"])
            self.threshold = job["threshold"]
            self.best_energy: float | None = None
            self.best_force_norm: float | None = None
            self.lowest_energy: float | None = None
            self.lowest_force_norm: float | None = None

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            coords = self.atoms.get_positions().reshape(-1)
            key = tuple(float(value) for value in coords)
            if self.counter.calls >= cap:
                raise SearchStop("cap", self.best_energy, self.best_force_norm)
            fresh, cached = self.counter.charge(key)
            if fresh or cached is None:
                energy, gradient = lennard_jones(coords)
                flat = tuple(-float(value) for value in gradient)
                cached = (float(energy), flat)
                self.counter.remember(key, cached)
                force_norm = float(np.linalg.norm(gradient))
                forces = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
                max_atom = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
                if math.isfinite(energy) and (
                    self.lowest_energy is None or energy < self.lowest_energy
                ):
                    self.lowest_energy = float(energy)
                    self.lowest_force_norm = force_norm
                if max_atom <= self.fmax and (self.best_energy is None or energy < self.best_energy):
                    self.best_energy = float(energy)
                    self.best_force_norm = force_norm
                if (
                    self.threshold is not None
                    and energy <= float(self.threshold)
                    and max_atom <= self.fmax
                ):
                    raise SearchStop("target", float(energy), force_norm)
            energy, flat = cached
            self.results["energy"] = float(energy)
            self.results["forces"] = np.asarray(flat, dtype=np.float64).reshape(-1, 3)

    calc = _LennardJones()
    positions = np.asarray(job["coords"], dtype=np.float64).reshape(-1, 3)
    atoms = Atoms(symbols=composition, positions=positions, pbc=False)
    atoms.calc = calc
    work = Path(job["work"])
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    error = ""
    kind = ""
    returned: float | None = None
    try:
        minima = bh_go(
            atoms,
            str(work),
            niter=int(niter),
            fmax=float(job["fmax"]),
            niter_local_relaxation=int(nlocal),
            optimizer=get_optimizer_class(str(job["optimizer"])),
            dr=float(job["dr"]),
            move_fraction=float(job["move_fraction"]),
            move_strategy=str(job["move_strategy"]),
            temperature=float(job["temperature"]),
            deduplicate=True,
            energy_tolerance=float(job["energy_tolerance"]),
            verbosity=0,
            clean=True,
            rng=ensure_rng(int(job["seed"])),
            system_type="gas_cluster",
            connectivity_factor=job["connectivity_factor"],
        )
        if minima:
            returned = float(min(energy for energy, _atoms in minima))
    except SearchStop as stop:
        kind = stop.kind
        if stop.energy is not None and (
            calc.best_energy is None or float(stop.energy) < calc.best_energy
        ):
            calc.best_energy = float(stop.energy)
            calc.best_force_norm = stop.force_norm
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)

    calls = int(calc.counter.calls)
    quenched_point = calc.best_energy is not None and kind != "cap"
    if quenched_point:
        energy = calc.best_energy
        force_norm = calc.best_force_norm
    elif calc.lowest_energy is not None:
        energy = calc.lowest_energy
        force_norm = calc.lowest_force_norm
    else:
        energy = returned
        force_norm = None
    finite = (
        energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    quenched = quenched_point and finite
    has_window = job["threshold"] is not None
    in_window = (
        finite
        and has_window
        and float(energy) <= float(job["threshold"])
        and kind == "target"
    )
    reasons: list[str] = []
    if error:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif kind == "cap" or calls >= cap:
        reasons.append("cap")
    elif not quenched:
        reasons.append("divergence")
    elif has_window and not in_window:
        reasons.append("energy")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "problem": job["problem"],
        "template": job["template"],
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
        "seed": str(job["seed"]),
        "force_calls": calls,
        "verify_calls": int(calc.counter.verify_calls),
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": force_norm,
        "converged": bool(in_window),
        "quenched": bool(quenched),
        "hessian_index": None,
        "expected_index": None,
        "algorithm": "basin hopping",
        "scgo_returned_energy": returned,
        "reference_energy": job["reference_energy"],
        "energy_above_minimum": job["energy_above_minimum"],
    }
    if error:
        record["error"] = error
    if not quenched and failure is None:
        record["success"] = False
        record["failure"] = "divergence"
        record["reasons"] = ["divergence"]
    return record


def _part_path(output: Path, index: int) -> Path:
    return output.parent / f"{output.stem}.parts" / f"{index:04d}.json"


def _write_campaign(output: Path, template: dict, problem: dict, records: list[dict], parameters: dict) -> dict:
    cap = int(parameters["cap"])
    expected = (problem.get("sample") or {}).get("starts")
    energy_rule = parameters.get("energy_above_minimum") is not None
    partial = (not energy_rule) or (expected is not None and len(records) < int(expected))
    legacy_counts = [
        cap if not record["success"] else int(record["force_calls"]) for record in records
    ]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["success"]]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
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
    version = str(template.get("code_version") or CODE_VERSION)
    commit = str(template.get("commit") or "")
    provenance = collect_provenance(
        code=str(template.get("code") or "SCGO"),
        code_version=version,
        commit=commit,
        parameters=parameters,
    )
    stored = []
    for record in records:
        row = dict(record)
        row.pop("index", None)
        stored.append(row)
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter on optbench.potentials.lennard_jones. "
            + (
                "A start succeeds when a quenched energy is within energy_above_minimum of the known minimum. "
                if energy_rule
                else "The problem states no energy_above_minimum. success means a finite quench under the FIRE force tolerance. energy is that quenched energy, or the lowest energy seen when the force never met fmax. "
            )
            + "The cap is the default SCGO basin-hopping budget."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm", "basin hopping"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "partial": partial,
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
                "starts": stored,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _dat(summary, template, metrics, partial=partial, energy_rule=energy_rule),
        encoding="utf-8",
    )
    write_starts(output.with_suffix(".starts.jsonl"), stored)
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary


def bh_settings(problem: dict, template: dict, n_atoms: int) -> dict:
    """Library basin-hopping block, the harness window, and the force-call cap."""
    import scgo

    version = str(getattr(scgo, "__version__", ""))
    if template.get("code_version") and version != str(template["code_version"]):
        raise RuntimeError(f"scgo {version} does not match template {template['code_version']}")
    commit = str(template.get("commit") or "")
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"{template['id']} commit {commit} is not the pinned SCGO revision {PINNED_COMMIT}")
    section = template.get("scgo") or {}
    if str(section.get("algorithm") or "bh") != "bh":
        raise ValueError(f"{template['id']} algorithm must be bh")
    convergence = problem.get("convergence") or {}
    if "energy_above_minimum" in convergence:
        known = reference_energy(problem)
        if known is None:
            raise ValueError(f"{problem['id']} has no known minimum")
        window = float(convergence["energy_above_minimum"])
        threshold = known + window
    else:
        known = None
        window = None
        threshold = None
    from scgo.param_presets import get_default_params

    defaults = get_default_params()
    block = defaults["optimizer_params"]["bh"]
    niter, nlocal, cap = default_bh_budget(n_atoms)
    stated_cap = template.get("failure_cap")
    if stated_cap is not None and int(stated_cap) != cap:
        raise ValueError(f"{template['id']} failure_cap {stated_cap} is not the budget {cap}")
    symbol = str(section.get("symbol") or "C")
    seed = int(section.get("seed") if section.get("seed") is not None else 314159)
    return {
        "algorithm": "bh",
        "optimizer": str(block["optimizer"]),
        "fmax": float(block["fmax"]),
        "dr": float(block["dr"]),
        "move_fraction": float(block["move_fraction"]),
        "move_strategy": str(block["move_strategy"]),
        "temperature": float(block["temperature"]),
        "energy_tolerance": float(block["energy_tolerance"]),
        "niter": niter,
        "niter_local_relaxation": nlocal,
        "cap": cap,
        "symbol": symbol,
        "seed": seed,
        "reference_energy": known,
        "energy_above_minimum": window,
        "threshold": threshold,
        "connectivity_factor": defaults["connectivity_factor"],
        "potential": "lennard-jones",
        "epsilon": 1.0,
        "sigma": 1.0,
        "cutoff": 40.0,
        "commit": commit,
    }


def make_job(
    problem: dict,
    template: dict,
    settings: dict,
    *,
    index: int,
    name: str,
    coords: list[float],
    work: str,
    seed: int | None = None,
) -> dict:
    """One start for ``_one_start``."""
    n_atoms = len(coords) // 3
    return {
        "index": index,
        "start": name,
        "coords": coords,
        "n_atoms": n_atoms,
        "symbol": settings["symbol"],
        "seed": int(settings["seed"] if seed is None else seed),
        "cap": int(settings["cap"]),
        "fmax": settings["fmax"],
        "threshold": settings["threshold"],
        "reference_energy": settings["reference_energy"],
        "energy_above_minimum": settings["energy_above_minimum"],
        "optimizer": settings["optimizer"],
        "dr": settings["dr"],
        "move_fraction": settings["move_fraction"],
        "move_strategy": settings["move_strategy"],
        "temperature": settings["temperature"],
        "energy_tolerance": settings["energy_tolerance"],
        "connectivity_factor": settings["connectivity_factor"],
        "work": work,
        "problem": problem["id"],
        "template": template["id"],
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": settings["commit"],
        "tuning": template.get("tuning", ""),
    }


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
    archive_path = str((template.get("scgo") or {}).get("archive_path") or "") or None
    loaded = load_geometries(root, problem, geometries, work, output, archive_path=archive_path)
    if limit and limit > 0:
        loaded = loaded[: int(limit)]
    if not loaded:
        raise ValueError(f"{problem['id']} has no starts")
    n_atoms = len(loaded[0][1]) // 3
    settings = bh_settings(problem, template, n_atoms)
    parameters = {key: value for key, value in settings.items() if key != "commit"}
    scratch = (work or output.parent / "work") / template["id"].replace("/", "-")
    jobs = []
    cached: dict[int, dict] = {}
    for index, (name, coords, _box) in enumerate(loaded):
        part = _part_path(output, index)
        if resume and part.is_file():
            cached[index] = json.loads(part.read_text(encoding="utf-8"))
            continue
        jobs.append(
            make_job(
                problem,
                template,
                settings,
                index=index,
                name=name,
                coords=coords,
                work=str(scratch / f"{index:04d}-{name}"),
            )
        )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    fresh: list[dict] = []
    if jobs:
        def _store(record: dict) -> None:
            part = _part_path(output, int(record["index"]))
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            fresh.append(record)

        if workers <= 1:
            for job in jobs:
                _store(_one_start(job))
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(_one_start, job): job["index"] for job in jobs}
                total = len(futures)
                for done, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    _store(record)
                    if done == total or done % 10 == 0:
                        print(
                            f"{template['id']} {done}/{total} start {record['start']} "
                            f"calls={record['force_calls']} energy={record['energy']} "
                            f"success={record['success']}",
                            flush=True,
                        )
    by_index = dict(cached)
    for record in fresh:
        by_index[int(record["index"])] = record
    records = [by_index[index] for index in range(len(loaded))]
    return _write_campaign(output, template, problem, records, parameters)
