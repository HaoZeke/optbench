"""Minimize OptBench starts with GeometryOptimization.jl.

The solver is ``minimize_energy!`` with ``OptimLBFGS``, ``OptimCG``, or
``OptimSD``. ``OptimSD`` is Optim.jl ``GradientDescent``. The linesearch
is the package default, BackTracking of order 2 with maxstep 0.8 bohr.
LBFGS also uses the Hager-Zhang initial step.

Each energy or force request enters ``HarnessCounter`` through
``RgpotCounter``. A second request at the same coordinates is the cached
entry. The callback stops when the problem Euclidean force norm is met,
or, on the noise problem, when the noise-aware rule is met. The library
force tolerance stays far below that stop so the max-atom check does not
end the search first. The iteration ceiling is the force-call cap.

Lennard-Jones positions are labeled bohr and the energy hartree, so
``austrip`` leaves the spec numbers unchanged. Morse positions are
angstrom and the energy is eV. The recorded force is the spec gradient.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

CODE = "GeometryOptimization.jl"
VERSION = "0.1.6"
OPTIM_VERSION = "1.13.3"
COMMIT = "5b145886c19042f2ef13e21627f28d86027f719c"
_PREFIX = "OPTBENCH "
_READY = "OPTBENCH ready "
_METHODS = {
    "lbfgs": "OptimLBFGS",
    "cg": "OptimCG",
    "gradientdescent": "OptimSD",
}


def project_dir(root: Path | None = None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[2] / "codes" / "geomoptjl"
    return Path(root) / "codes" / "geomoptjl"


def manifest_pin(root: Path | None = None) -> str:
    """Commit recorded for GeometryOptimization in the Julia manifest."""
    path = project_dir(root) / "Manifest.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    block = data["deps"]["GeometryOptimization"][0]
    return str(block["repo-rev"])


def julia_executable() -> str:
    env = os.environ.get("OPTBENCH_JULIA")
    if env:
        return env
    found = shutil.which("julia")
    if found:
        return found
    candidate = Path.home() / "opt" / "julia" / "bin" / "julia"
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError("julia is not on PATH and OPTBENCH_JULIA is unset")


def _kind(problem_id: str) -> str:
    if problem_id in {"minimization/lj38", "minimization/lj38-noise"}:
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no GeometryOptimization potential")


class JuliaWorker:
    """One long-lived Julia process."""

    def __init__(self, root: Path):
        project = project_dir(root)
        driver = project / "driver.jl"
        if not driver.is_file():
            raise FileNotFoundError(driver)
        env = os.environ.copy()
        env["JULIA_CONDAPKG_BACKEND"] = "Null"
        env["JULIA_PYTHONCALL_EXE"] = sys.executable
        env["JULIA_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        self._stderr: list[str] = []
        self.proc = subprocess.Popen(
            [
                julia_executable(),
                f"--project={project}",
                "--startup-file=no",
                "--quiet",
                "--threads=1",
                str(driver),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._drain = threading.Thread(target=self._read_stderr, daemon=True)
        self._drain.start()
        self.version, self.optim_version = self._ready()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line)

    def _ready(self) -> tuple[str, str]:
        assert self.proc.stdout is not None
        for _ in range(400):
            line = self.proc.stdout.readline()
            if line == "":
                tail = "".join(self._stderr)[-2000:]
                raise RuntimeError(f"julia exited before ready\n{tail}")
            if line.startswith(_READY):
                parts = line[len(_READY) :].split()
                if len(parts) < 2:
                    raise RuntimeError(f"julia ready line is incomplete: {line!r}")
                return parts[0], parts[1]
        raise RuntimeError("julia ready timed out")

    def run(self, job: dict) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"julia exited {self.proc.returncode}")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                tail = "".join(self._stderr)[-2000:]
                raise RuntimeError(f"julia closed the result pipe\n{tail}")
            if line.startswith(_PREFIX) and not line.startswith(_READY):
                payload = json.loads(line[len(_PREFIX) :])
                payload.setdefault("start", job.get("start"))
                payload.setdefault("index", job.get("index"))
                return payload

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write("OPTBENCH quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, sigma: float | None) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "problem": template.get("problem"),
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
    if sigma is not None:
        rows["sigma"] = sigma
    lines = [
        "# counter harness. One force call is one RgpotCounter.calculate entry.",
        "# GeometryOptimization LBFGS, CG, and GradientDescent call the spec potential.",
        "# The callback stops at the problem Euclidean force norm, or the noise-aware rule.",
        "# v2 success is under the cap, at that stop, and Hessian index 0.",
    ]
    if str(template.get("problem")) == "minimization/lj38":
        lines.append(
            "# LJ38 index removes the six rigid modes. A curvature is negative only below "
            "-max(force_norm, 1e-8*||H||_F, 1e-6)."
        )
    if str(template.get("problem")) == "minimization/lj38-noise":
        lines.append("# Noise is added after the clean call. The index uses the clean Hessian.")
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _options(template: dict) -> str:
    section = template.get("geomoptjl") or {}
    method = str(section.get("method") or "")
    if method not in _METHODS:
        raise ValueError(f"{template['id']} needs geomoptjl.method lbfgs, cg, or gradientdescent")
    return method


def _jobs(problem, template, loaded, method: str) -> tuple[list[dict], float | None]:
    from optbench.noise import KAPPA, window_energy_tolerance

    convergence = problem.get("convergence") or {}
    gtol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    kind = _kind(problem["id"])
    noise = template.get("noise") or {}
    sigma = None
    if problem["id"] == "minimization/lj38-noise":
        sigma = float(noise["sigma"])
        if "energy_tol" not in convergence:
            raise ValueError(f"{problem['id']} has no energy_tol for the noise window")
        energy_tol = window_energy_tolerance(
            float(convergence["energy_tol"]),
            sigma,
            float((problem.get("noise") or {}).get("kappa", KAPPA)),
        )
        seed = int(noise.get("seed", (problem.get("noise") or {})["seed"]))
    else:
        energy_tol = 0.0
        seed = 0
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        job = {
            "index": index,
            "start": name,
            "coords": [float(value) for value in coords],
            "box": None if kind == "lj" else [float(value) for value in box],
            "kind": kind,
            "method": method,
            "gtol": gtol,
            "cap": cap,
            "noise": sigma is not None,
            "sigma": 0.0 if sigma is None else sigma,
            "seed": seed,
            "energy_tol": energy_tol,
        }
        jobs.append(job)
    return jobs, sigma


def _run_jobs(root: Path, jobs: list[dict], workers: int, label: str, partial: Path | None = None) -> list[dict]:
    count = max(1, min(int(workers), len(jobs) or 1))
    if not jobs:
        return []
    pool = [JuliaWorker(root) for _ in range(count)]
    replacements: list[JuliaWorker] = []
    try:
        version = pool[0].version
        optim_version = pool[0].optim_version
        if version != VERSION:
            raise RuntimeError(f"GeometryOptimization {version} does not match {VERSION}")
        if optim_version != OPTIM_VERSION:
            raise RuntimeError(f"Optim {optim_version} does not match {OPTIM_VERSION}")
        pending: queue.Queue = queue.Queue()
        for job in jobs:
            pending.put(job)
        for _ in pool:
            pending.put(None)
        records: list[dict] = []
        lock = threading.Lock()
        errors: list[BaseException] = []

        def consume(slot: list[JuliaWorker]) -> None:
            while True:
                job = pending.get()
                if job is None:
                    return
                record = None
                for attempt in range(2):
                    worker = slot[0]
                    try:
                        if worker.proc.poll() is not None:
                            raise RuntimeError(f"julia exited {worker.proc.returncode}")
                        record = worker.run(job)
                        break
                    except Exception as exc:
                        if attempt == 0:
                            try:
                                worker.close()
                            except Exception:
                                pass
                            try:
                                restarted = JuliaWorker(root)
                                slot[0] = restarted
                                with lock:
                                    replacements.append(restarted)
                                continue
                            except Exception as restart_exc:
                                exc = restart_exc
                        record = {
                            "schema": "optbench.start/v2",
                            "start": job.get("start"),
                            "index": job.get("index"),
                            "force_calls": 0,
                            "verify_calls": 0,
                            "counter": "harness",
                            "comparable": True,
                            "success": False,
                            "failure": "crash",
                            "reasons": ["crash"],
                            "converged": False,
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        }
                        with lock:
                            errors.append(exc)
                with lock:
                    records.append(record)
                    if partial is not None:
                        with partial.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                    done = len(records)
                    if done == len(jobs) or done % 25 == 0:
                        print(f"{label} {done}/{len(jobs)}", flush=True)

        threads = [threading.Thread(target=consume, args=([worker],)) for worker in pool]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors and len(records) != len(jobs):
            raise errors[0]
    finally:
        for worker in [*pool, *replacements]:
            worker.close()
    records.sort(key=lambda row: int(row.get("index") or 0))
    return records


def _write(output: Path, problem, template, records, method: str, sigma, version: str, commit: str, root: Path) -> dict:
    for record in records:
        record.pop("index", None)
    convergence = problem.get("convergence") or {}
    cap = int(convergence["max_force_calls"])
    gtol = float(convergence["force_norm"])
    legacy_counts = [cap if not record.get("converged") else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record.get("converged")]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    metrics = score_v2(
        [
            {
                "force_calls": record.get("force_calls") or 0,
                "success": bool(record.get("success")),
                "failure": record.get("failure"),
            }
            for record in records
        ],
        cap,
    )
    passed = {
        "method": method,
        "solver": _METHODS[method],
        "gtol": gtol,
        "maxiters": cap,
        "maxstep_bohr": 0.8,
        "linesearch": "BackTracking",
        "linesearch_order": 2,
    }
    if method == "lbfgs":
        passed["alphaguess"] = "InitialHagerZhang"
    if sigma is not None:
        passed["sigma"] = sigma
    provenance = collect_provenance(
        code=CODE,
        code_version=version,
        commit=commit,
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": "Harness counter. GeometryOptimization calls the spec potential.",
        "runs": [
            {
                "entry": output.stem,
                "code": CODE,
                "algorithm": template.get("algorithm", method),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
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
                "starts": records,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, sigma), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
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
    from optbench.runners.rgmin_min import _load_geometries

    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    commit = str(template.get("commit") or "")
    pinned = manifest_pin(root)
    if commit != pinned:
        raise RuntimeError(f"{template['id']} commit {commit} does not match the manifest pin {pinned}")
    if str(template.get("code_version") or "") != VERSION:
        raise RuntimeError(f"{template['id']} version does not match {VERSION}")
    method = _options(template)
    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    jobs, sigma = _jobs(problem, template, loaded, method)
    done: dict[str, dict] = {}
    partial = output.with_suffix(".partial.jsonl")
    if resume:
        for path in (output.with_suffix(".starts.jsonl"), partial):
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("start"):
                    done[str(row["start"])] = row
    pending = [job for job in jobs if str(job["start"]) not in done]
    fresh = _run_jobs(root, pending, workers, template["id"], partial) if pending else []
    by_start = dict(done)
    for record in fresh:
        by_start[str(record.get("start"))] = record
    ordered = []
    for job in jobs:
        row = by_start.get(str(job["start"]))
        if row is None:
            raise RuntimeError(f"{template['id']} lost start {job['start']}")
        row = dict(row)
        row["index"] = job["index"]
        ordered.append(row)
    summary = _write(output, problem, template, ordered, method, sigma, VERSION, commit, root)
    if partial.is_file() and len(ordered) == len(jobs):
        partial.unlink()
    return summary
