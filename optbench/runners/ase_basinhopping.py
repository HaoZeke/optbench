"""ASE basin hopping on the Lennard-Jones 75 start set.

The potential is the harness Lennard-Jones. The success rule is the problem
file: energy at or below the Cambridge minimum plus 0.001, with the force-call
count below the cap. One calculator entry is one harness force call.

The local minimiser is FIRE. ASE's BasinHopping displaces the coordinates
from before the quench. This runner displaces the quenched coordinates, which
is the Wales and Doye step the ASE docstring cites.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from optbench.campaign import geometry_archive, load_template
from optbench.globalopt import judge_globalopt, reference_energy
from optbench.potentials import lennard_jones
from optbench.problems import load_problem
from optbench.score import score_v2

TEMPLATE_ID = "ase-globalopt/lj75-basinhopping"
PROBLEM_ID = "globalopt/lj75"
COMMIT = "f27c0005ae6a67ea419f996e728668865bfc1f86"
CODE_VERSION = "3.29.0"


class TargetReached(Exception):
    """The quench crossed the published energy stop."""

    def __init__(self, energy: float):
        super().__init__("target energy")
        self.energy = float(energy)


def read_coords(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if len(rows) != 75:
        raise ValueError(f"expected 75 atoms, read {len(rows)}")
    return np.asarray(rows, dtype=np.float64)


def _start_index(name: str) -> int | None:
    base = name.rstrip("/").rsplit("/", 1)[-1]
    prefix = "coords."
    if not base.startswith(prefix):
        return None
    suffix = base[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def load_starts(archive: Path) -> list[tuple[int, np.ndarray]]:
    found: list[tuple[int, np.ndarray]] = []
    with tarfile.open(archive, "r:gz") as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            index = _start_index(member.name)
            if index is None:
                continue
            raw = packed.extractfile(member)
            if raw is None:
                continue
            found.append((index, read_coords(raw.read().decode("utf-8", errors="replace"))))
    found.sort(key=lambda item: item[0])
    return found


@lru_cache(maxsize=1)
def quenched_basin_hopping():
    """BasinHopping whose accepted geometry is the quenched one."""
    from ase.optimize.basin import BasinHopping

    class QuenchedBasinHopping(BasinHopping):
        def __init__(self, atoms, progress=None, **kwargs):
            self._progress = progress
            super().__init__(atoms, **kwargs)

        def log(self, step, En, Emin):
            if self._progress is None:
                super().log(step, En, Emin)
                return
            self._progress(step, En, Emin)

        def get_energy(self, positions):
            energy = super().get_energy(positions)
            quenched = np.array(self.optimizable.get_x(), dtype=np.float64).reshape(-1, 3).copy()
            self.positions = quenched
            return energy

        def run(self, steps):
            ro = np.array(self.positions, dtype=np.float64, copy=True)
            Eo = self.get_energy(ro)
            ro = np.array(self.positions, dtype=np.float64, copy=True)
            for step in range(steps):
                En = None
                while En is None:
                    displaced = self.move(ro)
                    En = self.get_energy(displaced)
                quenched = np.array(self.positions, dtype=np.float64, copy=True)
                if En < self.Emin:
                    self.Emin = En
                    self.rmin = quenched.copy()
                    self.call_observers()
                self.log(step, En, self.Emin)
                accept = np.exp((Eo - En) / self.kT) > np.random.uniform()
                if accept:
                    ro = quenched
                    Eo = En

    return QuenchedBasinHopping


def _spec(root: Path) -> dict:
    problem = load_problem(root / "problems" / "globalopt" / "lj75.toml")
    template = load_template(root, TEMPLATE_ID)
    parameters = template["optimizer"]["parameters"]
    reference = reference_energy(problem)
    tolerance = float(problem["convergence"]["energy_above_minimum"])
    cap = int(problem["convergence"]["max_force_calls"])
    if int(template["stop"]["max_force_calls"]) != cap:
        raise ValueError("template cap differs from the problem cap")
    if abs(float(template["stop"]["energy_above_minimum"]) - tolerance) > 0.0:
        raise ValueError("template energy window differs from the problem")
    return {
        "reference": reference,
        "tolerance": tolerance,
        "target": reference + tolerance,
        "cap": cap,
        "temperature": float(parameters["temperature"]),
        "dr": float(parameters["dr"]),
        "fmax": float(parameters["fmax"]),
        "local_steps": int(parameters["local_steps"]),
        "maxstep": float(parameters["maxstep"]),
        "seed": int(parameters["seed"]),
        "local": str(template["optimizer"].get("local") or "FIRE"),
    }


def _reported_energy(hopper, calculator) -> float | None:
    """Best finished minimum, or the last finite sample if none finished."""
    if hopper is not None:
        emin = getattr(hopper, "Emin", None)
        if emin is not None:
            value = float(emin)
            if np.isfinite(value) and value < 1.0e30:
                return value
    last = getattr(calculator, "last_energy", None)
    if last is None:
        return None
    value = float(last)
    return value if np.isfinite(value) else None


def search_one(
    positions: np.ndarray,
    *,
    index: int,
    root: Path = Path("."),
    hops: int = 10**9,
    cap: int | None = None,
) -> dict:
    """Run one start. ``index`` selects the seed ``seed + index``."""
    spec = _spec(root)
    if cap is not None:
        spec["cap"] = int(cap)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    np.random.seed(spec["seed"] + int(index))

    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.optimize import FIRE

    from optbench.counter import attach_ase_counter

    class LennardJonesCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, target: float):
            super().__init__()
            self.target = float(target)
            self.last_energy: float | None = None

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            energy, gradient = lennard_jones(self.atoms.positions)
            self.last_energy = float(energy)
            self.results["energy"] = self.last_energy
            self.results["forces"] = -gradient.reshape((-1, 3))
            if np.isfinite(self.last_energy) and self.last_energy <= self.target:
                raise TargetReached(self.last_energy)

    if spec["local"] != "FIRE":
        raise ValueError(f"lj75 basin hopping quenches with FIRE, template says {spec['local']}")

    class Quench(FIRE):
        def __init__(self, atoms, logfile=None, **kwargs):
            super().__init__(atoms, logfile=logfile, maxstep=spec["maxstep"], **kwargs)

        def run(self, fmax=0.05, steps=None):
            return super().run(fmax=fmax, steps=spec["local_steps"])

    atoms = Atoms(symbols=["H"] * len(positions), positions=np.asarray(positions, dtype=float), pbc=False)
    calculator = attach_ase_counter(LennardJonesCalculator(spec["target"]), cap=spec["cap"])
    atoms.calc = calculator
    energy = None
    crashed = False
    error = ""
    hopper = None
    started = time.perf_counter()
    try:
        def _log(step, _energy, emin):
            if step == -1 or step % 50 == 0:
                print(
                    f"coords.{index} hop={step} emin={emin:.6f} "
                    f"calls={calculator.optbench_counter['calls']}",
                    flush=True,
                )

        hopper = quenched_basin_hopping()(
            atoms,
            temperature=spec["temperature"],
            optimizer=Quench,
            fmax=spec["fmax"],
            dr=spec["dr"],
            logfile=None,
            trajectory=None,
            optimizer_logfile=None,
            local_minima_trajectory=None,
            adjust_cm=True,
            progress=_log,
        )
        hopper.run(hops)
        energy = _reported_energy(hopper, calculator)
    except TargetReached as exc:
        energy = exc.energy
    except RuntimeError as exc:
        if "force-call cap" not in str(exc):
            crashed = True
            error = exc.__class__.__name__
        energy = _reported_energy(hopper, calculator)
    except Exception as exc:
        crashed = True
        error = exc.__class__.__name__
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
    return {
        "schema": "optbench.start/v2",
        "start": f"coords.{index}",
        "problem": PROBLEM_ID,
        "template": TEMPLATE_ID,
        "code": "ASE",
        "code_version": CODE_VERSION,
        "commit": COMMIT,
        "tuning": "default",
        "counter": "harness",
        "comparable": True,
        "seed": spec["seed"] + int(index),
        "success": judged["success"],
        "failure": judged["failure"],
        "reasons": judged["reasons"],
        "force_calls": calls,
        "energy": judged["energy"],
        "error": error,
        "seconds": time.perf_counter() - started,
    }


def load_campaign_records(path: Path) -> list[dict]:
    """Read a starts jsonl. A missing file is an empty run."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _worker(payload: tuple[int, list[float], str, int]) -> dict:
    index, coords, root, hops = payload
    return search_one(np.asarray(coords, dtype=np.float64), index=index, root=Path(root), hops=hops)


def run(
    root: Path,
    *,
    limit: int | None = None,
    workers: int | None = None,
    hops: int = 10**9,
    dest: Path | None = None,
    prior: list[dict] | None = None,
) -> list[dict]:
    problem = load_problem(root / "problems" / "globalopt" / "lj75.toml")
    starts = load_starts(geometry_archive(root, problem))
    expected = int(problem["sample"]["starts"])
    if len(starts) != expected:
        raise RuntimeError(f"archive has {len(starts)} numbered starts, problem states {expected}")
    records = [dict(record) for record in prior or []]
    done = {int(str(record["start"]).split(".")[1]) for record in records}
    selected = [item for item in (starts if limit is None else starts[:limit]) if item[0] not in done]
    payload = [(index, positions.tolist(), str(root), int(hops)) for index, positions in selected]
    if workers is None:
        workers = os.cpu_count() or 1
    campaign_root = root if dest is None else dest
    cap = int(problem["convergence"]["max_force_calls"])

    def _store(record: dict) -> None:
        records.append(record)
        _report(record)
        ordered = sorted(records, key=lambda item: int(str(item["start"]).split(".")[1]))
        write_campaign(
            campaign_root,
            ordered,
            partial=len(ordered) < expected,
            cap=cap,
            spec=_spec(root),
        )

    if not payload:
        return sorted(records, key=lambda record: int(str(record["start"]).split(".")[1]))
    if workers <= 1 or len(payload) == 1:
        for item in payload:
            _store(_worker(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, item) for item in payload]
            for future in as_completed(futures):
                _store(future.result())
    records.sort(key=lambda record: int(str(record["start"]).split(".")[1]))
    return records


def _report(record: dict) -> None:
    print(
        f"{record['start']} calls={record['force_calls']} energy={record['energy']} "
        f"success={record['success']} failure={record['failure']}",
        flush=True,
    )


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_campaign(
    root: Path,
    records: list[dict],
    *,
    partial: bool,
    cap: int,
    spec: dict | None = None,
) -> tuple[Path, Path]:
    summary = score_v2(records, cap)
    directory = root / "results" / "campaign"
    directory.mkdir(parents=True, exist_ok=True)
    stem = "ase-3.29.0-lj75-basinhopping"
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    calls = [int(record["force_calls"]) for record in records]
    if spec is None:
        quench = "# FIRE quench. The next displacement starts from the quenched coordinates."
    else:
        quench = (
            f"# FIRE maxstep {spec['maxstep']}, fmax {spec['fmax']}, "
            f"local steps {spec['local_steps']}, kT {spec['temperature']}, "
            f"displacement {spec['dr']}. Seed {spec['seed']} plus the coords index."
        )
    lines = [
        "# ASE 3.29.0 BasinHopping on globalopt/lj75.",
        quench,
        "# The next displacement starts from the quenched coordinates.",
        "# The energy stop and the force-call cap are the problem file.",
        f"problem {PROBLEM_ID}",
        f"template {TEMPLATE_ID}",
        "code ASE",
        f"code_version {CODE_VERSION}",
        f"commit {COMMIT}",
        "algorithm BasinHopping",
        "tuning default",
        "counter harness",
        f"n {summary['n']}",
        f"nfailed {summary['nfailed']}",
        f"par2_mean {_format(summary['par2_mean'])}",
        f"median {_format(summary['median'])}",
        f"median_low {_format(summary['median_low'])}",
        f"median_high {_format(summary['median_high'])}",
        f"rate_low {_format(summary['rate_low'])}",
        f"rate_high {_format(summary['rate_high'])}",
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
    _atomic_write(dat, "\n".join(lines) + "\n")
    _atomic_write(jsonl, "".join(json.dumps(record) + "\n" for record in records))
    return dat, jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ASE basin hopping on lj75")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hops", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    limit = args.limit or None
    workers = args.workers or None
    hops = args.hops or 10**9
    if args.dest is not None:
        args.dest.mkdir(parents=True, exist_ok=True)
    campaign_root = root if args.dest is None else args.dest
    prior = None
    if args.resume:
        jsonl = campaign_root / "results" / "campaign" / "ase-3.29.0-lj75-basinhopping.starts.jsonl"
        prior = load_campaign_records(jsonl)
        print(f"resume {len(prior)} starts from {jsonl}", flush=True)
    records = run(root, limit=limit, workers=workers, hops=hops, dest=args.dest, prior=prior)
    problem = load_problem(root / "problems" / "globalopt" / "lj75.toml")
    expected = int(problem["sample"]["starts"])
    partial = len(records) < expected
    summary = score_v2(records, int(problem["convergence"]["max_force_calls"]))
    campaign_root = root if args.dest is None else args.dest
    dat = campaign_root / "results" / "campaign" / "ase-3.29.0-lj75-basinhopping.dat"
    print(
        f"wrote {dat} n={summary['n']} successes={summary['successes']} "
        f"par2_mean={summary['par2_mean']} partial={partial}",
        flush=True,
    )


if __name__ == "__main__":
    main()
