"""Basin hopping with the archived ``bin/js850/globalopt.py`` settings.

SVN r269 builds ``LJClusterWrap``, sets ``structural_quench_params.tol``
to 1e-3, temperature 1, and steps with ``RandomDisplacement(0.41)``
inside ``AdaptiveStepsize`` (acceptance 0.5, factor 0.9, interval 100).
The quench is whatever ``get_minimizer`` returns in the pele commit
named by the template. A run stops when the Markov energy is at most
``-173.9284 + 1e-3``. The check happens after each hop, including the
first hop after the initial quench. The in-memory database is the one
the script passes into ``get_basinhopping``.

The paper row is unmodified pele ``2d5bc8e3`` (27 Aug 2013). Its
minimizer is ``mylbfgs``, and that linesearch already divides a rising
step by 10. Those numbers were counted by ``LJClusterWrap`` and are
stored with the campaign file. This module does not rebuild that
Python 2 extension.

``current-pele/lj38-basinhopping`` is unpatched pele 0.1.2. Its
minimizer is stock ``lbfgs_cpp``, which multiplies a rising step by
1/2. The ranked count is one ``PeleCounter`` entry per ``getEnergy``
or ``getEnergyGradient``. The numpy generator is seeded with the
integer in the start name before the first displacement.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.failures import classify_failure
from optbench.potentials import read_con
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
    if quench not in {"lbfgs_cpp", "mylbfgs"}:
        raise ValueError(f"{template['id']} quench {quench} is not the archived get_minimizer")
    takestep = str(section.get("takestep") or "adaptive")
    if takestep != "adaptive":
        raise ValueError(f"{template['id']} takestep {takestep} is not the archived AdaptiveStepsize")
    return {
        "temperature": float(section.get("temperature", 1.0)),
        "stepsize": float(section.get("stepsize", 0.41)),
        "quench": quench,
        "quench_tol": float(section.get("quench_tol", 1e-3)),
        "quench_maxstep": float(section.get("quench_maxstep", 0.1)),
        "quench_m": int(section.get("quench_m", 4)),
        "quench_nsteps": int(section.get("quench_nsteps", 10000)),
        "acc_ratio": float(section.get("acc_ratio", 0.5)),
        "factor": float(section.get("factor", 0.9)),
        "interval": int(section.get("interval", 100)),
        "takestep": takestep,
    }


def _counter_name(template: dict) -> str:
    quench = str((template.get("pele") or {}).get("quench") or "")
    if quench == "mylbfgs":
        return "pele-wrapper"
    return "harness"


def _hop_job(job: dict) -> dict:
    import numpy as np
    from pele.potentials import LJ
    from pele.systems import LJCluster
    from pele.takestep import AdaptiveStepsize, RandomDisplacement

    from optbench.counter import PeleCounter

    if str(job["quench"]) != "lbfgs_cpp":
        raise RuntimeError(
            f"{job['template']} uses {job['quench']} from that pele commit. "
            "The 2013 mylbfgs extension is not this interpreter's lbfgs_cpp."
        )
    cap = int(job["cap"])
    minimum = float(job["energy_minimum"])
    tolerance = float(job["energy_tolerance"])
    seed = int(job["seed"])
    np.random.seed(seed)

    class LJClusterWrap(LJCluster):
        def get_potential(self):
            self.counted = PeleCounter(LJ())
            return self.counted

    system = LJClusterWrap(38)
    system.params.structural_quench_params.tol = float(job["quench_tol"])
    system.params.basinhopping.temperature = float(job["temperature"])
    take = RandomDisplacement(stepsize=float(job["stepsize"]))
    adaptive = AdaptiveStepsize(
        take,
        acc_ratio=float(job["acc_ratio"]),
        factor=float(job["factor"]),
        interval=int(job["interval"]),
        verbose=False,
    )
    error = ""
    steps = 0
    caught_cap = False
    energy = None
    rms = None
    converged = False
    opt = None
    try:
        database = system.create_database()
        opt = system.get_basinhopping(
            database=database,
            takestep=adaptive,
            coords=np.asarray(job["coords"], dtype=np.float64),
            outstream=None,
        )
        opt.printfrq = 100
        for steps in range(1, 100001):
            opt.run(1)
            energy = float(opt.markovE)
            rms = None if opt.rms is None else float(opt.rms)
            converged = math.isfinite(energy)
            if energy <= minimum + tolerance:
                break
            if int(system.counted.calls) > cap:
                caught_cap = True
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if energy is None and opt is not None:
            try:
                energy = float(opt.markovE)
            except Exception:
                energy = None
    calls = int(system.counted.calls) if getattr(system, "counted", None) is not None else 0
    if calls > cap:
        calls = cap
        caught_cap = True
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
        "counter": "harness",
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
        "counter": _counter_name(template),
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
    counter = _counter_name(template)
    if counter == "pele-wrapper":
        counted = "# counter pele-wrapper. One force call is one getEnergy or one getEnergyGradient on LJClusterWrap."
        quench = "# The quench is mylbfgs from pele 2d5bc8e3. A rising step is divided by 10."
    else:
        counted = "# counter harness. One force call is one PeleCounter entry, a getEnergy or a getEnergyGradient."
        quench = "# The quench is stock lbfgs_cpp from unpatched pele 0.1.2. A rising step is multiplied by 1/2."
    lines = [
        counted,
        quench,
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
