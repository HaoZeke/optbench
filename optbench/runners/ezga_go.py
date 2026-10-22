"""EZGA 1.2.31 on the harness potential.

The relaxer is FIRE at fmax 0.05 with no MD. ``steps_max`` and the
generation count are the force-call cap. The population is 16 copies of
the archived start. A hit is a quench inside fmax and inside the energy
window.

EZGA's engine catches ``Exception``. ``ForceCallCap`` and ``Hit`` are
exceptions, so they used to stay inside the generation loop and the
search kept running. Both now leave through ``BaseException``, the same
way AGOX does. The physics job runs inline so a later generation is not
already queued when the cap is reached.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import tarfile
import tempfile
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.potentials import binary_lennard_jones, lennard_jones
from optbench.score import _median, half_even, score_v2
from optbench.v2 import collect_provenance

FMAX = 0.05
POPULATION = 16
SEED_BASE = 314159
# problems/globalopt/lj100-binary.toml: unshifted lj/cut, cutoff 20.
BINARY_CUTOFF = 20.0
_INDEX = re.compile(r"(\d+)(?:\.[^.]+)?$")


class Hit(Exception):
    """A converged quench landed inside the global-minimum window."""


class _Halt(BaseException):
    """Leave EZGA. Its engine catches Exception, including the cap and a hit."""


class Budget:
    """Counted ``(energy, gradient)``. A repeated coordinate is free."""

    def __init__(
        self,
        evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
        cap: int,
        *,
        target: float | None = None,
        window: float = 0.0,
    ):
        self.counter = HarnessCounter(cache=True, cap=int(cap))
        self._evaluate = evaluate
        self.cap = int(cap)
        self.target = None if target is None else float(target)
        self.window = float(window)
        self.energy: float | None = None
        self.gradient: np.ndarray | None = None
        self.point: np.ndarray | None = None
        self.hit = False
        self.quenched = False
        self.quench_energy: float | None = None
        self.quench_point: np.ndarray | None = None
        self.best_energy: float | None = None
        self.best_point: np.ndarray | None = None

    @property
    def calls(self) -> int:
        return int(self.counter.calls)

    def fg(self, coords: Sequence[float] | np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(coords, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in x)
        fresh, cached = self.counter.charge(key)
        if not fresh and cached is not None:
            energy, gradient = cached
            gradient_arr = np.asarray(gradient, dtype=np.float64)
            self.energy = float(energy)
            self.gradient = gradient_arr
            self.point = x.copy()
            return float(energy), gradient_arr
        energy, gradient = self._evaluate(x)
        gradient_arr = np.asarray(gradient, dtype=np.float64).reshape(-1)
        stored = (float(energy), tuple(float(value) for value in gradient_arr))
        self.counter.remember(key, stored)
        self.energy = float(energy)
        self.gradient = gradient_arr
        self.point = x.copy()
        if math.isfinite(self.energy) and (self.best_energy is None or self.energy < self.best_energy):
            self.best_energy = self.energy
            self.best_point = x.copy()
        return self.energy, gradient_arr

    def note_quench(
        self,
        point: Sequence[float] | np.ndarray,
        energy: float,
        gradient: Sequence[float] | np.ndarray,
        *,
        converged: bool,
    ) -> None:
        """Record a finished local minimization. A hit raises :class:`Hit`."""
        if not converged:
            return
        point_arr = np.asarray(point, dtype=np.float64).reshape(-1).copy()
        grad_arr = np.asarray(gradient, dtype=np.float64).reshape(-1).copy()
        energy_f = float(energy)
        self.quenched = True
        in_window = (
            self.target is not None
            and energy_f <= self.target + self.window
            and self.calls < self.cap
        )
        better = self.quench_energy is None or energy_f < self.quench_energy
        if better or in_window:
            self.quench_energy = energy_f
            self.quench_point = point_arr
            self.point = point_arr.copy()
            self.energy = energy_f
            self.gradient = grad_arr
        if in_window:
            self.hit = True
            raise Hit()


def observe_point(budget: Budget, point: np.ndarray, *, fmax: float) -> None:
    """One calculator entry.

    The force-call cap and a target hit leave as :class:`_Halt`. EZGA
    catches ``Exception``, so either of those types would be swallowed
    and the generation loop would keep running.
    """
    try:
        energy, gradient = budget.fg(point)
    except ForceCallCap:
        raise _Halt() from None
    forces = -np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
    atomic = float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
    try:
        budget.note_quench(point, float(energy), gradient, converged=atomic < fmax)
    except Hit:
        raise _Halt() from None


def _reference(problem: dict) -> tuple[float | None, float | None]:
    """Stated minimum and the absolute window around it.

    ``convergence.energy_above_minimum`` is the window when the problem
    states one. Otherwise ``reference.window`` is that same number for a
    problem whose convergence block does not carry it.
    """
    convergence = problem.get("convergence") or {}
    reference = problem.get("reference") or {}
    potential = problem.get("potential") or {}
    energy = reference.get("energy")
    if energy in (None, ""):
        energy = potential.get("energy_minimum")
    if energy in (None, ""):
        return None, None
    window = convergence.get("energy_above_minimum")
    if window in (None, ""):
        window = reference.get("window")
    if window in (None, ""):
        return float(energy), None
    return float(energy), float(window)


def _cap(problem: dict) -> int:
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    if stated in (None, ""):
        raise ValueError(f"{problem.get('id', 'problem')} has no max_force_calls")
    return int(stated)


def _read_components(text: str) -> list[tuple[str, np.ndarray]]:
    lines = text.splitlines()
    components: list[tuple[str, np.ndarray]] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("Coordinates of Component"):
            symbol = lines[index - 1].strip()
            index += 1
            rows: list[list[float]] = []
            while index < len(lines) and lines[index][:1].isspace():
                parts = lines[index].split()
                if len(parts) < 3:
                    break
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                index += 1
            components.append((symbol, np.asarray(rows, dtype=np.float64)))
            continue
        index += 1
    if not components or any(len(rows) == 0 for _symbol, rows in components):
        raise ValueError("con file has no coordinates")
    return components


def _read_coords(text: str) -> np.ndarray:
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
        raise ValueError("coordinate file has no rows")
    return np.asarray(rows, dtype=np.float64)


def _file_index(name: str) -> int | None:
    match = _INDEX.search(Path(name).name)
    if match is None:
        return None
    return int(match.group(1))


def _members(archive: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            found[member.name] = extracted.read().decode("utf-8", errors="replace")
    return found


def _choose_files(names: list[str]) -> list[str]:
    grouped: dict[int, list[str]] = {}
    for name in names:
        index = _file_index(name)
        if index is None:
            continue
        grouped.setdefault(index, []).append(name)
    chosen = []
    for index in sorted(grouped):
        group = grouped[index]
        cons = [name for name in group if name.endswith(".con")]
        chosen.append(sorted(cons or group)[0])
    return chosen


def _species_labels(problem: dict, symbols: list[str], counts: list[int]) -> list[int]:
    potential = problem.get("potential") or {}
    symbol_a = str(potential.get("symbol_A") or "A")
    symbol_b = str(potential.get("symbol_B") or "B")
    labels: list[int] = []
    for symbol, count in zip(symbols, counts):
        if symbol == symbol_a:
            labels.extend([0] * count)
        elif symbol == symbol_b:
            labels.extend([1] * count)
        else:
            raise ValueError(f"{problem['id']} has no species for symbol {symbol}")
    return labels


def load_starts(root: Path, problem: dict) -> list[dict]:
    """Archived starts in numeric order. A paired ``.con`` wins."""
    from optbench.campaign import geometry_archive

    archive = geometry_archive(root, problem)
    chosen = _choose_files(list(_members(archive)))
    if not chosen:
        raise FileNotFoundError(f"{archive} has no geometries")
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(chosen) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(chosen)}")
    texts = _members(archive)
    binary = str((problem.get("potential") or {}).get("kind") or "") == "binary-lennard-jones"
    rows = []
    for index, name in enumerate(chosen):
        text = texts[name]
        if name.endswith(".con"):
            components = _read_components(text)
            positions = np.vstack([rows_ for _symbol, rows_ in components])
            species = _species_labels(
                problem,
                [symbol for symbol, _rows in components],
                [len(rows_) for _symbol, rows_ in components],
            ) if binary else None
        else:
            positions = _read_coords(text)
            species = None
            if binary:
                potential = problem["potential"]
                n_b = int(potential["n_B"])
                n_a = int(potential["n_A"])
                if positions.shape[0] != n_a + n_b:
                    raise ValueError(f"{name} has {positions.shape[0]} atoms")
                species = [1] * n_b + [0] * n_a
        rows.append(
            {
                "index": index,
                "start": Path(name).name,
                "positions": positions.reshape(-1).tolist(),
                "species": species,
                "natoms": int(positions.shape[0]),
            }
        )
    return rows


def _evaluate(job: dict) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    species = job.get("species")

    def evaluate(coords: np.ndarray) -> tuple[float, np.ndarray]:
        if species is None:
            return lennard_jones(coords)
        return binary_lennard_jones(
            coords,
            species,
            sigma_aa=float(job["sigma_aa"]),
            sigma_bb=float(job["sigma_bb"]),
            epsilon_aa=float(job["epsilon_aa"]),
            epsilon_bb=float(job["epsilon_bb"]),
            epsilon_ab=float(job["epsilon_ab"]),
            cutoff=float(job["cutoff"]) if job.get("cutoff") is not None else None,
        )

    return evaluate


def _judge(budget: Budget | None, *, crash: str) -> dict:
    calls = 0 if budget is None else budget.calls
    cap = 0 if budget is None else budget.cap
    quench = None if budget is None else budget.quench_energy
    energy = quench if quench is not None else (None if budget is None else budget.best_energy)
    finite = energy is not None and math.isfinite(float(energy))
    in_window = False
    if budget is not None and finite and budget.target is not None and budget.quenched:
        in_window = float(energy) <= budget.target + budget.window and calls < budget.cap
    reasons: list[str] = []
    if crash:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif calls >= cap:
        reasons.append("cap")
    elif not in_window:
        reasons.append("energy" if budget is not None and budget.quenched else "force_norm")
    failure = classify_failure(reasons)
    return {
        "force_calls": int(calls),
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if not finite else float(energy),
        "hit": bool(budget and budget.hit and failure is None),
    }


def _silence() -> None:
    os.environ["TQDM_DISABLE"] = "1"
    warnings.filterwarnings("ignore", message=r"\[EZGA\]")
    for name in ("ezga", "ase", "matplotlib"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _run_ezga(budget: Budget, positions: np.ndarray, job: dict) -> None:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.io import write
    from ezga.evaluator.objective import objective_energy
    from ezga.factory import build_default_engine, load_config
    from ezga.simulator.ase_calculator import ase_calculator
    from ezga.variation.mutation import mutation_rattle

    fmax = float(job["fmax"])
    population = int(job["population"])
    generations = int(job["generations"])

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)
            point = np.asarray(atoms.positions, dtype=np.float64).reshape(-1)
            observe_point(budget, point, fmax=fmax)
            gradient = np.asarray(budget.gradient, dtype=np.float64).reshape(-1, 3)
            self.results["energy"] = float(budget.energy)
            self.results["forces"] = -gradient

    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    frames = [
        Atoms(numbers=[6] * pos.shape[0], positions=pos, cell=[40.0, 40.0, 40.0], pbc=False)
        for _copy in range(population)
    ]
    runner = ase_calculator(
        calculator=_Calc(),
        nvt_steps=0,
        fmax=fmax,
        steps_max=int(job["cap"]),
        device="cpu",
        log_interval=0,
        write_interval=0,
    )
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="ezga-") as tmp:
        os.chdir(tmp)
        try:
            path = os.path.join(tmp, "population.xyz")
            write(path, frames)
            config = {
                "output_path": os.path.join(tmp, "engine"),
                "max_generations": generations,
                "resume": False,
                "rng": int(job["seed"]),
                "population": {
                    "dataset_path": path,
                    "db_path": os.path.join(tmp, "db"),
                    "filter_duplicates": False,
                    "collision_factor": 0.0,
                },
                "multiobjective": {"size": population},
                "variation": {"initial_mutation_rate": 1.0},
                "evaluator": {"objectives_funcs": [objective_energy()]},
                "mutation_funcs": [mutation_rattle(std=0.2, seed=int(job["seed"]))],
                "simulator": {"calculator": runner, "execution": "inline"},
            }
            try:
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    saved_out, saved_err = sys.stdout, sys.stderr
                    sys.stdout = devnull
                    sys.stderr = devnull
                    try:
                        build_default_engine(load_config(config)).run()
                    except _Halt:
                        return
                    finally:
                        sys.stdout, sys.stderr = saved_out, saved_err
            except _Halt:
                return
        finally:
            os.chdir(cwd)


def search_start(job: dict) -> dict:
    """One archived start. The cap and a hit both return."""
    _silence()
    budget = Budget(
        _evaluate(job),
        int(job["cap"]),
        target=None if job.get("target") is None else float(job["target"]),
        window=0.0 if job.get("window") is None else float(job["window"]),
    )
    crash = ""
    try:
        _run_ezga(budget, np.asarray(job["positions"], dtype=np.float64), job)
    except _Halt:
        pass
    except Exception as exc:
        crash = f"{type(exc).__name__}: {exc}"
    judged = _judge(budget, crash=crash)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": int(job["index"]),
        "force_calls": judged["force_calls"],
        "counter": "harness",
        "success": judged["success"],
        "failure": judged["failure"],
        "reasons": judged["reasons"],
        "energy": judged["energy"],
        "hit": judged["hit"],
        "code": job.get("code", "EZGA"),
        "code_version": job.get("code_version", ""),
        "commit": job.get("commit", ""),
        "tuning": job.get("tuning", "default"),
        "template": job.get("template", ""),
        "problem": job.get("problem", ""),
    }
    if crash:
        record["error"] = crash
    return record


def _job(template: dict, problem: dict, start: dict, cap: int, target: float | None, window: float | None) -> dict:
    potential = problem.get("potential") or {}
    binary = str(potential.get("kind") or "") == "binary-lennard-jones"
    return {
        "index": int(start["index"]),
        "start": start["start"],
        "positions": start["positions"],
        "species": start.get("species"),
        "cap": cap,
        "target": target,
        "window": window,
        "fmax": FMAX,
        "population": POPULATION,
        "generations": cap,
        "seed": SEED_BASE + int(start["index"]),
        "sigma_aa": float(potential.get("sigma_AA") or 1.0),
        "sigma_bb": float(potential.get("sigma_BB") or 1.3),
        "epsilon_aa": float(potential.get("epsilon_AA") or 1.0),
        "epsilon_bb": float(potential.get("epsilon_BB") or 1.0),
        "epsilon_ab": float(potential.get("epsilon_AB") or 1.0),
        "cutoff": BINARY_CUTOFF if binary else None,
        "code": template.get("code", "EZGA"),
        "code_version": template.get("code_version", ""),
        "commit": template.get("commit", ""),
        "tuning": template.get("tuning", "default"),
        "template": template.get("id", ""),
        "problem": problem.get("id", ""),
    }


def _starts_path(output: Path) -> Path:
    return output.with_suffix(".starts.jsonl")


def _load_resume(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    found: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("start"):
            found[str(row["start"])] = row
    return found


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict) -> str:
    lines = [
        "# counter harness. One force call is one entry into the harness potential.",
        "# FIRE at fmax 0.05, no MD. steps_max and the generation count are the force-call cap.",
        "# A hit is a quench inside fmax and inside the energy window. The cap and a hit leave the search.",
        f"# The problem force-call cap is {int(summary['cap'])}. "
        "A start at or above it fails. PAR-2 charges each failure twice that cap.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    rows = {
        "counter": "harness",
        "force_calls": summary["force_calls"],
        "force_calls_min": summary["force_calls_min"],
        "force_calls_max": summary["force_calls_max"],
        "force_calls_median": summary["force_calls_median"],
        "nfailed": summary["nfailed"],
        "n": summary["starts"],
        "cap": summary["cap"],
        "par2_mean": metrics.get("par2_mean"),
        "median": metrics.get("median"),
        "median_low": metrics.get("median_low"),
        "median_high": metrics.get("median_high"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "code": template.get("code", "EZGA"),
        "code_version": template.get("code_version", ""),
        "commit": template.get("commit", ""),
        "algorithm": template.get("algorithm", "GA"),
        "tuning": template.get("tuning", "default"),
        "template": template.get("id", ""),
        "problem": template.get("problem", ""),
        "reducer": template.get("reducer", "capped"),
    }
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format(value)}")
    return "\n".join(lines) + "\n"


def _publish(output: Path, records: list[dict], template: dict, problem: dict, cap: int) -> dict:
    from optbench.campaign import write_starts

    calls = [int(row["force_calls"]) for row in records]
    scored = [
        {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
        for row in records
    ]
    metrics = score_v2(scored, cap)
    summary = {
        "force_calls": half_even(sum(calls) / len(calls)),
        "force_calls_min": min(calls),
        "force_calls_max": max(calls),
        "force_calls_median": _median([float(value) for value in calls]),
        "nfailed": sum(1 for row in records if not row["success"]),
        "starts": len(records),
        "cap": cap,
        "template": template["id"],
        "problem": problem["id"],
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
    }
    public = []
    for row in records:
        item = dict(row)
        item.pop("index", None)
        public.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics), encoding="utf-8")
    write_starts(_starts_path(output), public)
    provenance = collect_provenance(
        code=str(template.get("code", "EZGA")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={
            "fmax": FMAX,
            "population": POPULATION,
            "generations": cap,
            "steps_max": cap,
            "seed_base": SEED_BASE,
            "cap": cap,
            "reference_energy": summary.get("reference_energy"),
        },
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "measured",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. EZGA 1.2.31 FIRE at fmax 0.05. "
            f"The problem force-call cap is {cap}."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code", "EZGA"),
                "algorithm": template.get("algorithm", "GA"),
                "code_version": template.get("code_version", ""),
                "commit": template.get("commit", ""),
                "tuning": template.get("tuning", "default"),
                "present_source": True,
                "cap": cap,
                "provenance": provenance,
                "metrics": metrics,
                "starts": public,
            }
        ],
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    resume: bool = False,
) -> dict:
    del geometries, work
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    cap = _cap(problem)
    stop = (template.get("stop") or {}).get("max_force_calls")
    if stop is not None and int(stop) < cap:
        raise ValueError("template max_force_calls is below the problem force-call cap")
    target, window = _reference(problem)
    starts = load_starts(Path(root), problem)
    if limit and limit > 0:
        starts = starts[: int(limit)]
    done = _load_resume(_starts_path(output)) if resume else {}
    pending = [row for row in starts if row["start"] not in done]
    jobs = [_job(template, problem, row, cap, target, window) for row in pending]
    checkpoint = _starts_path(output)
    if workers <= 1:
        for done_n, job in enumerate(jobs, 1):
            record = search_start(job)
            _append(checkpoint, record)
            done[record["start"]] = record
            print(
                f"{template['id']} {len(done)}/{len(starts)} start={record['start']} "
                f"calls={record['force_calls']} energy={record['energy']} success={record['success']}",
                flush=True,
            )
            del done_n
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(search_start, job) for job in jobs]
            for record in (future.result() for future in as_completed(futures)):
                _append(checkpoint, record)
                done[record["start"]] = record
                print(
                    f"{template['id']} {len(done)}/{len(starts)} start={record['start']} "
                    f"calls={record['force_calls']} energy={record['energy']} success={record['success']}",
                    flush=True,
                )
    ordered = []
    for row in starts:
        if row["start"] not in done:
            summary = {
                "force_calls": None,
                "nfailed": None,
                "starts": len(done),
                "template": template["id"],
                "problem": problem["id"],
                "par2_mean": None,
                "median": None,
            }
            return summary
        ordered.append(done[row["start"]])
    for index, row in enumerate(ordered):
        row["index"] = index
    return _publish(output, ordered, template, problem, cap)
