"""Minimize with TorchSim FIRE or gradient descent.

The model is ``SpecPotentialModel``. It evaluates the problem potential
through ``HarnessCounter`` and returns the force. FIRE is the ASE flavor.
Gradient descent uses ``pos_lr``. The cell is held fixed.

Lennard-Jones 38 and the noisy variant use the vibrational index: six
rigid modes are removed, and a curvature is negative only below
``-max(force_norm, 1e-8*||H||_F, 1e-6)``. On the noisy problem that
force norm is the noise-aware stop. Morse bulk uses the full Hessian
and counts a curvature below ``-1e-6``, the same rule as the rgmin
rows on that problem.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.noise import KAPPA, stop_threshold, window_energy_tolerance
from optbench.potentials import (
    lennard_jones,
    lennard_jones_hessian,
    morse_bulk,
    morse_bulk_hessian,
)
from optbench.potserver import RgpotCounter
from optbench.runners.noise_min import _Observed
from optbench.runners.rgmin_min import EIGENVALUE_TOL, _load_geometries
from optbench.score import score_v2, summarize_capped
from optbench.stationary import curvature_tolerance, negative_index, vibrational_eigenvalues
from optbench.v2 import collect_provenance

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

_FIRE_INIT = ("dt_start", "alpha_start", "fire_flavor")
_FIRE_STEP = (
    "dt_max",
    "n_min",
    "f_inc",
    "f_dec",
    "alpha_start",
    "f_alpha",
    "max_step",
    "fire_flavor",
)
_GD_STEP = ("pos_lr",)


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _kind(problem_id: str) -> str:
    if problem_id in {"minimization/lj38", "minimization/lj38-noise"}:
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no TorchSim potential")


def _section(template: dict) -> tuple[str, dict]:
    section = template.get("torchsim") or {}
    if "method" not in section:
        raise ValueError(f"{template['id']} needs torchsim.method")
    method = str(section["method"])
    options = {key: value for key, value in section.items() if key != "method"}
    if method == "fire":
        known = set(_FIRE_INIT) | set(_FIRE_STEP)
    elif method == "gradient_descent":
        known = set(_GD_STEP)
    else:
        raise ValueError(f"{template['id']} method {method} is not fire or gradient_descent")
    extra = set(options) - known
    if extra:
        raise ValueError(f"{template['id']} has unknown torchsim keys {sorted(extra)}")
    if "n_min" in options:
        options["n_min"] = int(options["n_min"])
    return method, options


def _split_options(method: str, options: dict) -> tuple[dict, dict]:
    if method == "fire":
        init_kwargs = {key: options[key] for key in _FIRE_INIT if key in options}
        step_kwargs = {key: options[key] for key in _FIRE_STEP if key in options}
        return init_kwargs, step_kwargs
    return {}, {key: options[key] for key in _GD_STEP if key in options}


def _index_at(kind: str, point: list[float], box, force_stop: float) -> tuple[int, list[float]]:
    import numpy as np

    if kind == "lj":
        matrix = lennard_jones_hessian(point)
        values = vibrational_eigenvalues(matrix, point, rotations=True)
        tolerance = curvature_tolerance(force_stop, float(np.linalg.norm(matrix)))
    else:
        matrix = morse_bulk_hessian(point, box)
        values = np.linalg.eigvalsh(matrix).tolist()
        tolerance = EIGENVALUE_TOL
    return negative_index(values, tolerance), [float(value) for value in values[:8]]


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


class _Trail:
    """One start, its counter, and the noisy force the optimizer sees."""

    def __init__(self, job: dict):
        import numpy as np

        self.index = int(job["index"])
        self.start = str(job["start"])
        self.kind = str(job["kind"])
        self.cap = int(job["cap"])
        self.sigma = float(job["sigma"])
        self.clean_tol = float(job["clean_tol"])
        self.energy_tol = float(job["energy_tol"])
        self.box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
        self.coords0 = np.asarray(job["coords"], dtype=np.float64).reshape(-1, 3)
        self.components = int(self.coords0.size)
        self.pbc = self.kind == "morse"
        counter = HarnessCounter(cache=True, cap=self.cap)

        def evaluate(positions, _numbers=None, _box=None):
            flat = np.asarray(positions, dtype=np.float64).reshape(-1)
            if self.kind == "lj":
                return lennard_jones(flat)
            return morse_bulk(flat, self.box)

        self.potential = RgpotCounter(evaluate, counter)
        self.observed = _Observed(
            self.potential,
            self.sigma,
            __import__("random").Random(int(job["seed"]) + self.index),
        )
        self.point: list[float] | None = None
        self.stopped = False
        self.error = ""
        self.steps = 0

    def evaluate(self, rows):
        import numpy as np

        flat = np.asarray(rows, dtype=np.float64).reshape(-1)
        energy, gradient = self.observed(flat)
        return float(energy), np.asarray(gradient, dtype=np.float64)

    def met(self) -> bool:
        return self.observed.met(self.clean_tol, self.energy_tol, self.components)


def _state(trail: _Trail):
    import torch
    import torch_sim as ts

    device = torch.device("cpu")
    dtype = torch.float64
    pos = torch.tensor(trail.coords0, dtype=dtype, device=device)
    count = int(pos.shape[0])
    if trail.box is None:
        lengths = torch.tensor([40.0, 40.0, 40.0], dtype=dtype, device=device)
    else:
        lengths = torch.tensor(np_box(trail.box), dtype=dtype, device=device)
    cell = torch.diag(lengths).unsqueeze(0)
    return ts.SimState(
        positions=pos,
        masses=torch.ones(count, dtype=dtype, device=device),
        cell=cell,
        atomic_numbers=torch.ones(count, dtype=torch.int64, device=device),
        pbc=trail.pbc,
    )


def np_box(box):
    import numpy as np

    return np.asarray(box, dtype=np.float64).reshape(3).tolist()


def _flat(state) -> list[float]:
    return [float(value) for value in state.positions.detach().cpu().reshape(-1).tolist()]


def _finite_state(state) -> bool:
    import torch

    if not torch.isfinite(state.positions).all():
        return False
    energy = getattr(state, "energy", None)
    if energy is not None and not torch.isfinite(energy).all():
        return False
    forces = getattr(state, "forces", None)
    if forces is not None and not torch.isfinite(forces).all():
        return False
    return True


def _relax(trail: _Trail, method: str, init_kwargs: dict, step_kwargs: dict) -> None:
    import torch
    import torch_sim as ts

    from optbench.torchsim_model import SpecPotentialModel

    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        pass
    state = _state(trail)
    model = SpecPotentialModel([trail.evaluate], device=torch.device("cpu"), dtype=torch.float64)
    try:
        with torch.no_grad():
            if method == "fire":
                state = ts.fire_init(state=state, model=model, cell_filter=None, **init_kwargs)
            elif method == "gradient_descent":
                state = ts.gradient_descent_init(state=state, model=model, cell_filter=None)
            else:
                raise ValueError(f"unknown method {method}")
            while True:
                trail.point = _flat(state)
                if not _finite_state(state):
                    break
                calls = int(trail.potential.counter.calls)
                if calls >= trail.cap:
                    break
                if trail.met():
                    trail.stopped = True
                    break
                if method == "fire":
                    state = ts.fire_step(state=state, model=model, **step_kwargs)
                else:
                    state = ts.gradient_descent_step(state=state, model=model, **step_kwargs)
                trail.steps += 1
    except ForceCallCap as exc:
        trail.error = f"ForceCallCap: {exc}"
        if state is not None:
            trail.point = _flat(state)
    except Exception as exc:
        trail.error = f"{type(exc).__name__}: {exc}"
        if state is not None:
            try:
                trail.point = _flat(state)
            except Exception:
                pass


def _record(trail: _Trail, started: float) -> dict:
    import numpy as np

    calls = int(trail.potential.counter.calls) if trail.potential is not None else 0
    verify_calls = 0
    energy = None
    force_norm = None
    point = trail.point
    if point is not None:
        try:
            with trail.potential.counter.verification():
                energy, gradient = trail.potential.calculate(point)
            force_norm = float(np.linalg.norm(gradient))
            verify_calls = int(trail.potential.counter.verify_calls)
        except Exception as exc:
            if not trail.error:
                trail.error = f"{type(exc).__name__}: {exc}"
    threshold = stop_threshold(trail.clean_tol, trail.sigma, trail.components, KAPPA)
    hessian_index = None
    min_eigenvalues: list[float] = []
    cap_text = "ForceCallCap" in trail.error or "force-call cap" in trail.error
    if point is not None and (not trail.error or cap_text):
        try:
            hessian_index, min_eigenvalues = _index_at(trail.kind, point, trail.box, threshold)
        except Exception as exc:
            if not trail.error:
                trail.error = f"{type(exc).__name__}: {exc}"
    finite = (
        point is not None
        and all(math.isfinite(value) for value in point)
        and energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    calls = int(trail.potential.counter.calls)
    at_force = (
        finite
        and trail.stopped
        and force_norm is not None
        and force_norm <= threshold
        and calls < trail.cap
        and not trail.error
    )
    reasons: list[str] = []
    cap_hit = calls >= trail.cap or cap_text
    if trail.error and not cap_hit:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif cap_hit:
        reasons.append("cap")
    elif not at_force:
        reasons.append("force_norm")
    elif hessian_index != 0:
        reasons.append("hessian_index")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": trail.start,
        "index": trail.index,
        "force_calls": calls,
        "verify_calls": verify_calls,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None else float(energy),
        "force_norm": force_norm,
        "hessian_index": hessian_index,
        "expected_index": 0,
        "converged": at_force,
        "min_eigenvalues": min_eigenvalues,
        "sigma": trail.sigma,
        "steps": trail.steps,
        "seconds": time.perf_counter() - started,
    }
    if trail.error:
        record["error"] = trail.error
    return record


def _crash_record(job: dict, exc: BaseException, started: float) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": str(job.get("start") or ""),
        "index": int(job.get("index") or 0),
        "force_calls": 0,
        "verify_calls": 0,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": 0,
        "converged": False,
        "min_eigenvalues": [],
        "sigma": float(job.get("sigma") or 0.0),
        "steps": 0,
        "seconds": time.perf_counter() - started,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    started = time.perf_counter()
    try:
        import torch_sim

        expected = str(job["code_version"])
        if torch_sim.__version__ != expected:
            raise RuntimeError(f"torch_sim {torch_sim.__version__} does not match template {expected}")
        trail = _Trail(job)
        init_kwargs, step_kwargs = _split_options(str(job["method"]), dict(job["options"]))
        _relax(trail, str(job["method"]), init_kwargs, step_kwargs)
        return _record(trail, started)
    except Exception as exc:
        return _crash_record(job, exc, started)


def _dat(summary: dict, template: dict, metrics: dict, *, partial: bool) -> str:
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
        "cap": summary.get("cap"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "partial": "true" if partial else None,
    }
    if template.get("noise"):
        rows["sigma"] = template["noise"]["sigma"]
    lines = [
        "# counter harness. One force call is one SpecPotentialModel evaluation of the problem potential.",
        "# FIRE is the ASE flavor. Gradient descent uses pos_lr. The cell stays fixed.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# LJ38 removes the six rigid modes. A curvature is negative only below "
        "-max(force_norm, 1e-8*||H||_F, 1e-6). The noise-aware stop supplies that force_norm when sigma is positive.",
        "# Morse bulk uses the full Hessian. A curvature below -1e-6 is negative.",
        "# Energy and Kabsch RMSD were not applied.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    if partial:
        lines.append("# partial true. The start file is shorter than the problem's full set.")
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _document(template, output, metrics, summary, records, passed, version, commit) -> dict:
    provenance = collect_provenance(
        code="TorchSim",
        code_version=version,
        commit=commit,
        parameters=passed,
    )
    return {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": template.get("problem"),
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. SpecPotentialModel evaluates the problem potential. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "TorchSim",
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": summary.get("cap"),
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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_bundle(output: Path, template, records, cap, version, commit, passed, *, full_n: int) -> dict:
    ordered = sorted(records, key=lambda row: int(row["index"]))
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in ordered]
    summary = summarize_capped(legacy_counts, cap)
    summary["cap"] = cap
    kept = [int(record["force_calls"]) for record in ordered if record["converged"]]
    summary["median"] = statistics.median(kept) if kept else None
    energies = [float(record["energy"]) for record in ordered if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in ordered
        ],
        cap,
    )
    partial = len(ordered) < full_n
    _atomic_text(output, _dat(summary, template, metrics, partial=partial))
    body = "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered)
    _atomic_text(output.with_suffix(".starts.jsonl"), body)
    document = _document(template, output, metrics, summary, ordered, passed, version, commit)
    _atomic_text(
        output.with_suffix(".v2.json"),
        json.dumps(document, indent=2, sort_keys=True) + "\n",
    )
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = template.get("problem")
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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
    from optbench.campaign import _check_stop

    _quiet_threads()
    if template.get("engine") != "torchsim":
        raise ValueError(f"{template['id']} is not a torchsim template")
    version = str(template.get("code_version") or "")
    commit = str(template.get("commit") or "")
    if not version or not commit:
        raise RuntimeError(f"{template['id']} needs code_version and commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    _check_stop(problem, template)
    convergence = problem.get("convergence") or {}
    clean_tol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    method, options = _section(template)
    kind = _kind(problem["id"])
    sigma = 0.0
    seed = 0
    if problem["id"] == "minimization/lj38-noise":
        noise = template.get("noise") or {}
        if "sigma" not in noise:
            raise ValueError(f"{template['id']} needs noise.sigma")
        sigma = float(noise["sigma"])
        seed = int(noise["seed"])
        problem_seed = int((problem.get("noise") or {})["seed"])
        if seed != problem_seed:
            raise ValueError(f"{template['id']} seed {seed} is not the problem seed {problem_seed}")
        allowed = [float(value) for value in (problem.get("noise") or {})["sigma"]]
        if not any(math.isclose(sigma, value, rel_tol=0.0, abs_tol=0.0) for value in allowed):
            raise ValueError(f"{template['id']} sigma {sigma} is outside the problem list")
        energy_tol = window_energy_tolerance(float(convergence["energy_tol"]), sigma, KAPPA)
    else:
        if template.get("noise"):
            raise ValueError(f"{template['id']} carries a noise section on {problem['id']}")
        energy_tol = 0.0
    loaded = _load_geometries(root, problem, geometries, work, output)
    if problem["id"] == "minimization/morse-bulk" and geometries is None and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    stated_n = (problem.get("sample") or {}).get("starts")
    if stated_n is not None:
        archive_n = int(stated_n)
    elif problem["id"] == "minimization/morse-bulk":
        archive_n = 100
    else:
        archive_n = len(loaded)
    if limit:
        loaded = loaded[: int(limit)]
    jobs = []
    for index, (name, coords, box) in enumerate(loaded):
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.tolist(),
                "box": None if box is None else box.tolist(),
                "kind": kind,
                "method": method,
                "options": options,
                "sigma": sigma,
                "seed": seed,
                "clean_tol": clean_tol,
                "energy_tol": energy_tol,
                "cap": cap,
                "code_version": version,
            }
        )
    done: dict[int, dict] = {}
    if resume:
        for row in _read_jsonl(output.with_suffix(".starts.jsonl")):
            if "index" not in row:
                continue
            done[int(row["index"])] = row
    pending = [job for job in jobs if job["index"] not in done]
    passed = {"method": method, "gtol": clean_tol, "maxiter": cap, "sigma": sigma, **options}

    def _flush() -> None:
        if not done:
            return
        _write_bundle(
            output,
            template,
            list(done.values()),
            cap,
            version,
            commit,
            passed,
            full_n=archive_n,
        )

    try:
        if pending:
            if workers <= 1:
                for done_n, job in enumerate(pending, 1):
                    done[job["index"]] = _minimize_job(job)
                    if done_n == len(pending) or done_n % 25 == 0:
                        print(f"{template['id']} {len(done)}/{len(jobs)}", flush=True)
                        _flush()
            else:
                import multiprocessing as mp

                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    futures = [pool.submit(_minimize_job, job) for job in pending]
                    finished = 0
                    for future in as_completed(futures):
                        record = future.result()
                        done[int(record["index"])] = record
                        finished += 1
                        if finished == len(futures) or finished % 25 == 0:
                            print(f"{template['id']} {len(done)}/{len(jobs)}", flush=True)
                            _flush()
    finally:
        if done:
            _flush()
    summary = _write_bundle(
        output,
        template,
        list(done.values()),
        cap,
        version,
        commit,
        passed,
        full_n=archive_n,
    )
    return summary
