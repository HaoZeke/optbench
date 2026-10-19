"""Basin hopping with the 2013 pele driver for Lennard-Jones 38.

``bin/js850/globalopt.py`` builds ``LJClusterWrap``, sets
``structural_quench_params.tol`` to 1e-3, temperature 1, and steps
with ``RandomDisplacement(0.41)`` inside ``AdaptiveStepsize``
(acceptance 0.5, factor 0.9, interval 100). The quench is
``lbfgs_cpp`` from ``get_minimizer``. A run stops when the Markov
energy is at most ``-173.9284 + 1e-3``, and the check happens after
each hop, including the first hop after the initial quench.

The potential is the harness Lennard-Jones kernel, epsilon 1, sigma 1,
cutoff 40, no shift. Pairs in these clusters stay inside that cutoff,
so the energy matches pele's infinite Lennard-Jones. A force call is
one ``getEnergy`` or ``getEnergyGradient`` on the wrapper in
``tools.LJClusterWrap``. The same coordinates are counted again.

pele 0.1.2 multiplies a rising LBFGS step by 1/2. Commit a68ec5, the
archived code, divides that step by 10. This runner needs the a68ec5
factor. The numpy generator is seeded with the integer in the start
name before the first displacement.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap
from optbench.failures import classify_failure
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance

# A converged energy this far below the published minimum is not that
# surface. The cutoff-40 kernel matches the Cambridge values to 1e-6.
_BELOW_MINIMUM = 0.01


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _archive_path(problem: dict, template: dict) -> str:
    problem_path = str((problem.get("input") or {}).get("archive_path") or "")
    template_path = str((template.get("input") or {}).get("archive_path") or "")
    if problem_path and template_path and problem_path != template_path:
        raise ValueError(f"{template['id']} archive_path is not the problem archive")
    path = problem_path or template_path
    if not path:
        raise ValueError(f"{problem['id']} has no input.archive_path")
    return path


def _stored_archive(root: Path, archive_path: str) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(f"{archive_path} is not in the geometry manifest")


def _seed_of(name: str) -> int:
    tail = name.rsplit(".", 1)[-1]
    if tail.isdigit():
        return int(tail)
    if name.isdigit():
        return int(name)
    raise ValueError(f"{name} has no integer seed")


def _prefer_name(left: str, right: str) -> str:
    def numbered(name: str) -> bool:
        return name.rsplit(".", 1)[-1].isdigit()

    if numbered(left) != numbered(right):
        return left if numbered(left) else right
    return min(left, right)


def _read_plain(text: str):
    import numpy as np

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
        raise ValueError("coordinate file has no rows")
    return np.asarray(rows, dtype=np.float64)


def _load_geometries(root: Path, problem: dict, template: dict, geometries: Path | None, work: Path | None, output: Path):
    import numpy as np

    if geometries is None:
        import tarfile

        archive_path = _archive_path(problem, template)
        archive = _stored_archive(root, archive_path)
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
        cons = [path for path in dest.rglob("*.con") if path.is_file()]
        files = cons or [path for path in dest.rglob("*") if path.is_file() and path.name != ".extracted"]
    else:
        files = [path for path in sorted(geometries.rglob("*")) if path.is_file()]
        if not files:
            files = [path for path in sorted(geometries.glob("*")) if path.is_file()]
    if not files:
        raise FileNotFoundError(f"{problem['id']} has no start files")
    expected_atoms = _atom_count(problem["id"])
    chosen: dict[str, tuple[str, object]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".con":
            pos, _box = read_con(text)
        else:
            pos = _read_plain(text)
        if expected_atoms is not None and pos.shape[0] != expected_atoms:
            raise ValueError(f"{path.name} has {pos.shape[0]} atoms, {problem['id']} has {expected_atoms}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stem = path.stem if path.suffix == ".con" else path.name
        current = chosen.get(digest)
        if current is None or _prefer_name(stem, current[0]) == stem:
            chosen[digest] = (stem, np.asarray(pos, dtype=np.float64).reshape(-1).copy())
    loaded = [(name, coords, _seed_of(name)) for name, coords in chosen.values()]
    loaded.sort(key=lambda item: (item[2], item[0]))
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected not in (None, "", 0, "0") and len(loaded) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(loaded)}")
    return loaded


def _atom_count(problem_id: str) -> int | None:
    name = problem_id.rsplit("/", 1)[-1]
    if name.startswith("lj") and name[2:].isdigit():
        return int(name[2:])
    return None


def _rule(problem: dict, template: dict) -> tuple[float, float, str]:
    """Return ``(minimum, tolerance, partial_reason)``.

    The problem value wins. A missing start count or a missing energy
    rule keeps the row partial. LJ38 states the tolerance and the start
    count. The numerical minimum lives on the template until a problem
    file states it.
    """
    convergence = problem.get("convergence") or {}
    potential = problem.get("potential") or {}
    reference = template.get("reference") or {}
    minimum = convergence.get("energy_minimum", potential.get("energy_minimum", reference.get("energy_minimum")))
    tolerance = convergence.get("energy_above_minimum", reference.get("energy_above_minimum"))
    if minimum is None or tolerance is None:
        raise ValueError(f"{template['id']} needs an energy minimum and energy_above_minimum")
    reasons = []
    if "energy_above_minimum" not in convergence:
        reasons.append("problem file has no energy_above_minimum")
    if not (problem.get("sample") or {}).get("starts"):
        reasons.append("problem file has no start count")
    if problem.get("contract") == "partial":
        reasons.append("problem contract is partial")
    return float(minimum), float(tolerance), "; ".join(reasons)


def _budget(problem: dict, template: dict) -> int:
    from optbench.campaign import _check_stop

    cap = _check_stop(problem, template)
    if cap is None:
        raise ValueError(f"{template['id']} needs stop.max_force_calls")
    return int(cap)


def _options(template: dict) -> dict:
    section = template.get("pele") or {}
    quench = str(section.get("quench") or "lbfgs_cpp")
    if quench != "lbfgs_cpp":
        raise ValueError(f"{template['id']} quench {quench} is not lbfgs_cpp")
    takestep = str(section.get("takestep") or "adaptive")
    if takestep != "adaptive":
        raise ValueError(f"{template['id']} takestep {takestep} is not the archived AdaptiveStepsize")
    return {
        "temperature": float(section.get("temperature", 1.0)),
        "stepsize": float(section.get("stepsize", 0.41)),
        "quench_tol": float(section.get("quench_tol", 1e-3)),
        "quench_maxstep": float(section.get("quench_maxstep", 0.1)),
        "quench_m": int(section.get("quench_m", 4)),
        "quench_nsteps": int(section.get("quench_nsteps", 10000)),
        "acc_ratio": float(section.get("acc_ratio", 0.5)),
        "factor": float(section.get("factor", 0.9)),
        "interval": int(section.get("interval", 100)),
        "takestep": takestep,
    }


# Displacement of start 1 with seed 0. The a68ec5 linesearch (a rising
# step divided by 10) takes 327 gradient evaluations. pele 0.1.2
# multiplies that step by 1/2 and does not land on this count.
_LINESEARCH_CALLS = 327
_LINESEARCH_ENERGY = -163.20543198540588


class _WrapperLJ:
    """Count every ``getEnergy`` and ``getEnergyGradient``, as ``LJClusterWrap`` does."""

    def __init__(self, cap: int):
        self.calls = 0
        self.cap = cap

    def _charge(self) -> None:
        self.calls += 1
        if self.calls > self.cap:
            raise ForceCallCap(f"force-call cap {self.cap}")

    def getEnergy(self, coords):
        self._charge()
        energy, _gradient = lennard_jones(coords)
        return float(energy)

    def getEnergyGradient(self, coords):
        import numpy as np

        self._charge()
        energy, gradient = lennard_jones(coords)
        return float(energy), np.ascontiguousarray(gradient, dtype=np.float64)


def _hop_job(job: dict) -> dict:
    import numpy as np
    from pele.basinhopping import BasinHopping
    from pele.optimize import lbfgs_cpp
    from pele.takestep import AdaptiveStepsize, RandomDisplacement

    cap = int(job["cap"])
    minimum = float(job["energy_minimum"])
    tolerance = float(job["energy_tolerance"])
    seed = int(job["seed"])
    np.random.seed(seed)
    potential = _WrapperLJ(cap)
    error = ""
    steps = 0
    caught_cap = False
    energy = None
    rms = None
    converged = False

    def quench(coords):
        return lbfgs_cpp(
            coords,
            potential,
            tol=float(job["quench_tol"]),
            maxstep=float(job["quench_maxstep"]),
            M=int(job["quench_m"]),
            nsteps=int(job["quench_nsteps"]),
            iprint=-1,
        )

    try:
        step = AdaptiveStepsize(
            RandomDisplacement(stepsize=float(job["stepsize"])),
            acc_ratio=float(job["acc_ratio"]),
            factor=float(job["factor"]),
            interval=int(job["interval"]),
            verbose=False,
        )
        bh = BasinHopping(
            np.asarray(job["coords"], dtype=np.float64),
            potential,
            step,
            temperature=float(job["temperature"]),
            quench=quench,
            outstream=None,
        )
        while potential.calls < cap:
            bh.takeOneStep()
            steps += 1
            energy = float(bh.markovE)
            rms = float(bh.rms) if bh.rms is not None else None
            converged = math.isfinite(energy)
            if energy <= minimum + tolerance:
                break
    except Exception as exc:
        caught_cap = isinstance(exc, ForceCallCap) or potential.calls > cap
        if not caught_cap:
            error = f"{type(exc).__name__}: {exc}"
        if energy is None:
            try:
                energy = float(bh.markovE)
            except Exception:
                energy = None
    calls = int(potential.calls)
    if calls > cap:
        calls = cap
    success = (
        error == ""
        and energy is not None
        and math.isfinite(energy)
        and energy <= minimum + tolerance
        and energy >= minimum - _BELOW_MINIMUM
        and calls < cap
    )
    reasons: list[str] = []
    if error:
        reasons.append("crash")
    elif energy is None or not math.isfinite(energy if energy is not None else math.nan):
        reasons.append("divergence" if not caught_cap and calls < cap else "cap")
    elif energy < minimum - _BELOW_MINIMUM:
        reasons.append("divergence")
    elif calls >= cap or caught_cap:
        reasons.append("cap")
    elif not success:
        reasons.append("cap")
    if success:
        reasons = []
    failure = classify_failure(reasons)
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "seed": seed,
        "force_calls": calls,
        "verify_calls": 0,
        "counter": "pele-wrapper",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "lowest_energy": None if energy is None else float(energy),
        "force_norm": rms,
        "quench_converged": converged,
        "bh_steps": steps,
        "energy_minimum": minimum,
        "energy_tolerance": tolerance,
        "problem": job["problem"],
        "template": job["template"],
        "code": "pele",
        "code_version": job["code_version"],
        "commit": job["commit"],
        "tuning": job["tuning"],
        "error": error,
    }


def _require_a68ec5_linesearch(coords) -> None:
    """Refuse a stock pele 0.1.2 linesearch before any basin hop."""
    import numpy as np
    from pele.optimize import lbfgs_cpp
    from pele.takestep import RandomDisplacement

    np.random.seed(0)
    kicked = np.asarray(coords, dtype=np.float64).reshape(-1).copy()
    RandomDisplacement(stepsize=0.41).takeStep(kicked)
    potential = _WrapperLJ(10**9)
    result = lbfgs_cpp(
        kicked,
        potential,
        tol=1e-3,
        maxstep=0.1,
        M=4,
        nsteps=10000,
        iprint=-1,
    )
    calls = int(result.nfev)
    energy = float(result.energy)
    if calls != _LINESEARCH_CALLS or abs(energy - _LINESEARCH_ENERGY) > 1e-8:
        raise RuntimeError(
            "lbfgs_cpp is not the a68ec5 backtracking quench "
            f"(start 1, seed 0 returned {calls} calls and energy {energy:.12g}; "
            f"expected {_LINESEARCH_CALLS} and {_LINESEARCH_ENERGY:.12g}). "
            "Apply codes/pele-a68ec5-lbfgs-linesearch.patch and rebuild pele."
        )


def _read_done(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("start"):
            rows.append(value)
    return rows


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, *, partial: str, cap: int, minimum: float, tolerance: float) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "force_calls_median": summary.get("median"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "pele-wrapper",
        "cap": cap,
        "partial": "true" if partial else "false",
        "energy_minimum": minimum,
        "energy_tolerance": tolerance,
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "lowest_energy": summary.get("energy_min"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    lines = [
        "# counter pele-wrapper. One force call is one getEnergy or one getEnergyGradient. The same coordinates are counted again.",
        "# The quench is pele.optimize.lbfgs_cpp with the a68ec5 factor (a rising step is divided by 10). The driver is pele.basinhopping.BasinHopping.",
        "# A start succeeds when the Markov energy is within energy_tolerance of energy_minimum and the call count is under the cap.",
        "# The numpy generator is seeded with the integer in the start name.",
    ]
    if partial:
        lines.append("# partial: " + partial)
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_outputs(output: Path, records: list[dict], template: dict, problem: dict, metrics_cap: int, minimum: float, tolerance: float, partial: str, version: str, commit: str, options: dict) -> dict:
    ordered = sorted(records, key=lambda row: (int(row.get("index", 0)), str(row.get("start"))))
    counts = [int(row["force_calls"]) if row["success"] else metrics_cap for row in ordered]
    summary = summarize_capped(counts, metrics_cap)
    successes = [int(row["force_calls"]) for row in ordered if row["success"]]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(row["energy"]) for row in ordered if row.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    score_records = [
        {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
        for row in ordered
    ]
    metrics = score_v2(score_records, metrics_cap)
    passed = dict(options)
    passed["energy_minimum"] = minimum
    passed["energy_tolerance"] = tolerance
    passed["cap"] = metrics_cap
    provenance = collect_provenance(code="pele", code_version=version, commit=commit, parameters=passed)
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": partial or "pele wrapper. The success rule is the Markov energy window.",
        "runs": [
            {
                "entry": output.stem,
                "code": "pele",
                "algorithm": template.get("algorithm", "basin hopping"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": metrics_cap,
                "partial": bool(partial),
                "parameters": passed,
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "force_calls": summary.get("force_calls"),
                    "force_calls_min": summary.get("force_calls_min"),
                    "force_calls_max": summary.get("force_calls_max"),
                    "nfailed": summary.get("nfailed"),
                    "median": summary.get("median"),
                },
                "starts": ordered,
            }
        ],
    }
    public = []
    for row in ordered:
        item = dict(row)
        item.pop("index", None)
        public.append(item)
    _atomic_text(
        output,
        _dat(
            summary,
            template,
            metrics,
            commit,
            version,
            partial=partial,
            cap=metrics_cap,
            minimum=minimum,
            tolerance=tolerance,
        ),
    )
    _atomic_text(
        output.with_suffix(".starts.jsonl"),
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in public),
    )
    _atomic_text(output.with_suffix(".v2.json"), json.dumps(document, indent=2, sort_keys=True) + "\n")
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary


def _pele_revision() -> str:
    import pele.version as pele_version

    revision = str(getattr(pele_version, "git_revision", "") or "")
    if not revision:
        raise RuntimeError("pele.version.git_revision is empty")
    return revision


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
    commit = str(template.get("commit") or "")
    version = str(template.get("code_version") or "")
    if not commit or not version:
        raise RuntimeError(f"{template['id']} needs commit and code_version")
    revision = _pele_revision()
    if revision != commit:
        raise RuntimeError(f"pele revision {revision} is not template commit {commit}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    minimum, tolerance, partial = _rule(problem, template)
    cap = _budget(problem, template)
    options = _options(template)
    loaded = _load_geometries(root, problem, template, geometries, work, output)
    by_start = {name: coords for name, coords, _seed in loaded}
    if "1" in by_start:
        _require_a68ec5_linesearch(by_start["1"])
    starts_path = output.with_suffix(".starts.jsonl")
    done = _read_done(starts_path) if resume else []
    if done:
        sample = done[0]
        if float(sample.get("energy_minimum")) != minimum or float(sample.get("energy_tolerance")) != tolerance:
            raise RuntimeError(f"{starts_path} used a different energy rule")
        if int(sample.get("force_calls", 0)) >= 0 and str(sample.get("commit")) != commit:
            raise RuntimeError(f"{starts_path} is commit {sample.get('commit')}")
    done_names = {str(row["start"]) for row in done}
    jobs = []
    for index, (name, coords, seed) in enumerate(loaded):
        if name in done_names:
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "seed": seed,
                "coords": coords.tolist(),
                "cap": cap,
                "energy_minimum": minimum,
                "energy_tolerance": tolerance,
                "problem": problem["id"],
                "template": template["id"],
                "code_version": version,
                "commit": commit,
                "tuning": str(template.get("tuning") or ""),
                **options,
            }
        )
    if limit > 0:
        jobs = jobs[:limit]
    by_name = {str(row["start"]): row for row in done}
    full = len(loaded)

    def emit() -> dict:
        records = list(by_name.values())
        counted = f"scored {len(records)} of {full} starts"
        if len(records) < full:
            held = f"{partial}; {counted}" if partial else counted
        else:
            held = partial
        return _write_outputs(
            output, records, template, problem, cap, minimum, tolerance, held, version, commit, options
        )

    summary = None
    if jobs:
        if workers <= 1:
            pending = jobs
            for done_count, job in enumerate(pending, 1):
                record = _hop_job(job)
                by_name[str(record["start"])] = record
                summary = emit()
                print(
                    f"{template['id']} {job['start']} calls={record['force_calls']} "
                    f"energy={record['energy']} success={record['success']} {done_count}/{len(pending)}",
                    flush=True,
                )
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(_hop_job, job): job for job in jobs}
                total = len(futures)
                for done_count, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    by_name[str(record["start"])] = record
                    summary = emit()
                    print(
                        f"{template['id']} {record['start']} calls={record['force_calls']} "
                        f"energy={record['energy']} success={record['success']} {done_count}/{total}",
                        flush=True,
                    )
    if not by_name:
        raise RuntimeError(f"{template['id']} produced no starts")
    if summary is None:
        summary = emit()
    return summary
