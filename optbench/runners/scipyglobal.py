"""SciPy global optimizers on the harness Lennard-Jones potential.

One force call is one entry into ``optbench.potentials.lennard_jones``.
``HarnessCounter`` caches a repeated coordinate, so an energy read and a
gradient read of one geometry are one evaluation. ``ForceCallCap`` is a
``RuntimeError``. The optimizers swallow ``Exception``, so a stop raises
``SearchStop``, a ``BaseException``, and the refused call is not counted.

The LJ38 window is the measured energy of the Wales coordinates plus the
problem ``energy_above_minimum``. A hit on the call that fills the cap is
a cap failure. LJ75 stays partial while the problem file states no window.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.lj_minima import LJ38_MINIMUM
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_all, summarize_capped
from optbench.v2 import collect_provenance, write_starts

COORD_BOUND = 6.0
LJ75_ARCHIVE = "website/globalopt/lj75/lj75_clusters.tgz"
SOURCE_TREE = Path("/tmp/scipy-src")
BOUNDED = frozenset({"dual_annealing", "differential_evolution", "shgo"})
CHECKPOINT_ROOT = Path("/tmp/optbench-scipyglobal")
# shgo builds the whole simplicial complex before it calls the potential.
# For these dimensions that complex does not fit. The worker raises
# MemoryError at this address-space limit instead of being killed.
SHGO_ADDRESS_BYTES = 2 * 1024**3


class SearchStop(BaseException):
    """Leave SciPy. ``except Exception`` inside the optimizers must not catch this."""

    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(kind)


class HarnessObjective:
    """Energy and analytic gradient counted on one ``HarnessCounter``."""

    def __init__(self, evaluate, cap: int, target: float | None, label: str = ""):
        self.evaluate = evaluate
        self.counter = HarnessCounter(cache=True, cap=int(cap))
        self.target = None if target is None else float(target)
        self.cap = int(cap)
        self.label = label
        self.best: float | None = None
        self.best_calls: int | None = None
        self.stop: str | None = None
        self.trace: list[list[float | int]] = []

    def _eval(self, x) -> tuple[float, tuple[float, ...]]:
        if self.stop:
            raise SearchStop(self.stop)
        arr = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        key = tuple(float(value) for value in arr)
        try:
            fresh, cached = self.counter.charge(key)
        except ForceCallCap:
            self.stop = "cap"
            raise SearchStop("cap") from None
        if fresh:
            energy, gradient = self.evaluate(arr)
            energy = float(energy)
            grad = tuple(float(value) for value in np.asarray(gradient, dtype=np.float64).reshape(-1))
            self.counter.remember(key, (energy, grad))
        else:
            energy, grad = cached
        calls = int(self.counter.calls)
        if math.isfinite(energy) and (self.best is None or energy < self.best):
            self.best = float(energy)
            self.best_calls = calls
            self.trace.append([calls, self.best])
        if fresh and calls % 50000 == 0:
            shown = "none" if self.best is None else f"{self.best:.8g}"
            print(f"{self.label} calls={calls} best={shown}", flush=True)
        hit = (
            self.target is not None
            and math.isfinite(energy)
            and energy <= self.target
            and calls < self.cap
        )
        if hit:
            self.stop = "hit"
            raise SearchStop("hit")
        if calls >= self.cap:
            self.stop = "cap"
            raise SearchStop("cap")
        return float(energy), grad

    def fun(self, x) -> float:
        energy, _gradient = self._eval(x)
        return energy

    def jac(self, x) -> np.ndarray:
        _energy, gradient = self._eval(x)
        return np.asarray(gradient, dtype=np.float64)


def _ls_maxiter(dim: int) -> int:
    """The local-search iteration cap dual_annealing installs when minimizer_kwargs is empty."""
    return min(max(int(dim) * 6, 100), 1000)


def method_parameters(method: str, cap: int, dim: int, bound: float | None) -> dict:
    """Constructor values for one method. Iteration limits follow the force-call cap."""
    if method == "basinhopping":
        return {
            "method": method,
            "niter": int(cap),
            "T": 1.0,
            "stepsize": 0.5,
            "interval": 50,
            "target_accept_rate": 0.5,
            "stepwise_factor": 0.9,
            "minimizer": "BFGS",
            "jac": True,
        }
    if method == "dual_annealing":
        return {
            "method": method,
            "maxiter": int(cap),
            "maxfun": int(cap),
            "initial_temp": 5230.0,
            "restart_temp_ratio": 2.0e-5,
            "visit": 2.62,
            "accept": -5.0,
            "no_local_search": False,
            "minimizer": "L-BFGS-B",
            "ls_maxiter": _ls_maxiter(dim),
            "jac": True,
            "bound": bound,
        }
    if method == "differential_evolution":
        return {
            "method": method,
            "strategy": "best1bin",
            "maxiter": int(cap),
            "popsize": 15,
            "tol": 0.01,
            "mutation": [0.5, 1.0],
            "recombination": 0.7,
            "polish": True,
            "init": "latinhypercube",
            "atol": 0.0,
            "updating": "immediate",
            "workers": 1,
            "vectorized": False,
            "bound": bound,
        }
    if method == "shgo":
        return {
            "method": method,
            "n": 100,
            "iters": 1,
            "sampling_method": "simplicial",
            "minimizer": "SLSQP",
            "ftol": 1e-12,
            "jac": True,
            "workers": 1,
            "bound": bound,
            "address_space_bytes": SHGO_ADDRESS_BYTES,
        }
    raise ValueError(f"unknown scipy method {method}")


def _invoke(method: str, objective: HarnessObjective, x0: np.ndarray, cap: int, bound: float | None, rng) -> None:
    import scipy.optimize as optimize

    fun = objective.fun
    jac = objective.jac
    if method == "basinhopping":
        optimize.basinhopping(
            fun,
            x0,
            niter=int(cap),
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={"method": "BFGS", "jac": jac},
            interval=50,
            rng=rng,
            target_accept_rate=0.5,
            stepwise_factor=0.9,
            callback=lambda _x, _f, _accept: objective.stop is not None,
        )
        return
    if bound is None:
        raise ValueError(f"{method} needs bounds")
    limits = [(-float(bound), float(bound))] * int(x0.size)
    if method == "dual_annealing":
        optimize.dual_annealing(
            fun,
            limits,
            maxiter=int(cap),
            initial_temp=5230.0,
            restart_temp_ratio=2.0e-5,
            visit=2.62,
            accept=-5.0,
            maxfun=int(cap),
            rng=rng,
            no_local_search=False,
            x0=x0,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "jac": jac,
                "bounds": limits,
                "options": {"maxiter": _ls_maxiter(x0.size)},
            },
            callback=lambda _x, _f, _context: objective.stop is not None,
        )
        return
    if method == "differential_evolution":
        optimize.differential_evolution(
            fun,
            limits,
            strategy="best1bin",
            maxiter=int(cap),
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            rng=rng,
            polish=True,
            init="latinhypercube",
            atol=0.0,
            updating="immediate",
            workers=1,
            x0=x0,
            vectorized=False,
            callback=lambda _xk, convergence=0.0: objective.stop is not None,
        )
        return
    if method == "shgo":
        optimize.shgo(
            fun,
            limits,
            n=100,
            iters=1,
            sampling_method="simplicial",
            workers=1,
            minimizer_kwargs={"jac": jac, "options": {"ftol": 1e-12}},
            callback=lambda _xk: objective.stop is not None,
        )
        return
    raise ValueError(f"unknown scipy method {method}")


def _classify(calls: int, cap: int, best: float | None, target: float | None, stop: str | None, error: str) -> tuple[bool, str | None, list[str]]:
    if error:
        return False, "crash", ["crash"]
    if best is None or not math.isfinite(best):
        return False, "divergence", ["divergence"]
    if calls >= cap or stop == "cap":
        return False, "cap", ["cap"]
    if target is None:
        failure = classify_failure(["no_success_rule"])
        return False, failure, ["no_success_rule"]
    if best <= target and calls < cap:
        return True, None, []
    return False, "wrong_stationary", ["energy"]


def _blank_crash(job: dict, message: str) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": job.get("start"),
        "index": job.get("index"),
        "force_calls": 0,
        "verify_calls": 0,
        "best_calls": None,
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": "crash",
        "reasons": ["crash"],
        "energy": None,
        "seed": job.get("seed"),
        "cap": job.get("cap"),
        "method": job.get("method"),
        "target": job.get("target"),
        "bound": job.get("bound"),
        "commit": job.get("commit"),
        "code_version": job.get("code_version"),
        "parameters": job.get("parameters"),
        "stop": "crash",
        "error": message,
        "trace": [],
    }


def search_start(job: dict) -> dict:
    """Run one start. The returned record includes the checkpoint trace."""
    method = str(job["method"])
    cap = int(job["cap"])
    target = None if job.get("target") is None else float(job["target"])
    bound = None if job.get("bound") is None else float(job["bound"])
    x0 = np.asarray(job["coords"], dtype=np.float64).reshape(-1).copy()
    if bound is not None and (np.any(x0 < -bound) or np.any(x0 > bound)):
        return _blank_crash(job, "start lies outside the bounds")
    if job.get("limit_address_space"):
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (SHGO_ADDRESS_BYTES, SHGO_ADDRESS_BYTES))
    evaluate = job.get("evaluate", lennard_jones)
    label = f"{job.get('template_id', method)}:{job.get('start')}"
    objective = HarnessObjective(evaluate, cap, target, label=label)
    error = ""
    try:
        rng = None if method == "shgo" else np.random.default_rng(int(job["seed"]))
        _invoke(method, objective, x0, cap, bound, rng)
        if objective.stop is None:
            objective.stop = "return"
    except SearchStop:
        pass
    except MemoryError as exc:
        error = f"MemoryError: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error and not objective.stop:
        objective.stop = "crash"
    calls = int(objective.counter.calls)
    success, failure, reasons = _classify(calls, cap, objective.best, target, objective.stop, error)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": 0,
        "best_calls": objective.best_calls,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": objective.best,
        "seed": int(job["seed"]),
        "cap": cap,
        "method": method,
        "target": target,
        "bound": bound,
        "commit": job.get("commit"),
        "code_version": job.get("code_version"),
        "parameters": job.get("parameters"),
        "stop": objective.stop,
        "trace": objective.trace,
    }
    if error:
        record["error"] = error
    _write_checkpoint(job, record)
    return record


def _checkpoint_path(job: dict) -> Path | None:
    raw = job.get("checkpoint")
    if not raw:
        return None
    return Path(raw)


def _write_checkpoint(job: dict, record: dict) -> None:
    path = _checkpoint_path(job)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def _rescore(saved: dict, target: float, cap: int) -> dict:
    """Apply a new energy window to a finished trace.

    The trace records the call count of each new lowest energy. The first
    point at or below the window is the call where an early stop would have
    fired.
    """
    record = dict(saved)
    record["target"] = float(target)
    hit = None
    for calls, energy in saved.get("trace") or []:
        if float(energy) <= float(target) and int(calls) < int(cap):
            hit = int(calls)
            break
    if hit is not None:
        record["force_calls"] = hit
        record["success"] = True
        record["failure"] = None
        record["reasons"] = []
        record["stop"] = "hit"
        record.pop("error", None)
        return record
    calls = int(saved.get("force_calls") or 0)
    success, failure, reasons = _classify(calls, cap, saved.get("energy"), target, saved.get("stop"), "")
    record["success"] = success
    record["failure"] = failure
    record["reasons"] = reasons
    return record


def _reusable(saved: dict, job: dict) -> dict | None:
    if not isinstance(saved, dict):
        return None
    if saved.get("schema") != "optbench.start/v2":
        return None
    if saved.get("parameters") != job.get("parameters"):
        return None
    if int(saved.get("cap") or -1) != int(job["cap"]):
        return None
    if str(saved.get("commit") or "") != str(job.get("commit") or ""):
        return None
    if int(saved.get("seed") or -1) != int(job["seed"]):
        return None
    old = saved.get("target")
    new = job.get("target")
    if old == new:
        return saved
    if new is None or not saved.get("trace"):
        return None
    if saved.get("stop") == "hit" and old is not None and float(new) < float(old):
        energy = saved.get("energy")
        if energy is None or float(energy) > float(new):
            return None
    return _rescore(saved, float(new), int(job["cap"]))


def _load_checkpoint(job: dict) -> dict | None:
    path = _checkpoint_path(job)
    if path is None or not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    found = _reusable(saved, job)
    if found is not None and found is not saved:
        _write_checkpoint(job, found)
    return found


def _pipe_entry(job: dict, connection) -> None:
    try:
        connection.send(search_start(job))
    except BaseException as exc:  # noqa: BLE001
        connection.send(_blank_crash(job, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _isolated(job: dict) -> dict:
    import multiprocessing as mp

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    proc = context.Process(target=_pipe_entry, args=(job, child))
    proc.start()
    child.close()
    proc.join()
    if parent.poll():
        record = parent.recv()
        parent.close()
        return record
    parent.close()
    return _blank_crash(job, f"worker exited {proc.exitcode}")


def energy_target(problem: dict) -> float | None:
    """Energy the start must reach, or None when the problem states no window."""
    convergence = problem.get("convergence") or {}
    if "energy_above_minimum" not in convergence:
        return None
    window = float(convergence["energy_above_minimum"])
    potential = problem.get("potential") or {}
    if "energy_minimum" in potential:
        return float(potential["energy_minimum"]) + window
    if "energy_minimum" in convergence:
        return float(convergence["energy_minimum"]) + window
    if problem.get("id") == "globalopt/lj38":
        energy, _gradient = lennard_jones(np.asarray(LJ38_MINIMUM, dtype=np.float64))
        return float(energy) + window
    return None


def _index_of(name: str) -> int:
    base = name[:-4] if name.endswith(".con") else name
    tail = base.split(".")[-1]
    if not tail.isdigit():
        raise ValueError(f"no start index in {name}")
    return int(tail)


def _read_plain(text: str) -> np.ndarray:
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
        raise ValueError("coordinate file has no positions")
    return np.asarray(rows, dtype=np.float64)


def _archive_for(root: Path, problem: dict) -> Path:
    from optbench.campaign import geometry_archive

    stated = (problem.get("input") or {}).get("archive_path")
    if stated:
        return geometry_archive(root, problem)
    if problem.get("id") == "globalopt/lj75":
        copied = dict(problem)
        copied["input"] = {"archive_path": LJ75_ARCHIVE}
        return geometry_archive(root, copied)
    raise ValueError(f"{problem.get('id')} has no input.archive_path")


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    recorded = archive.name
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == recorded:
        return
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    marker.write_text(recorded, encoding="utf-8")


def load_starts(root: Path, problem: dict, work: Path | None = None) -> list[tuple[str, np.ndarray]]:
    """Centered starts in archive order. Coordinates lie inside the shared bound."""
    dest = Path(work or CHECKPOINT_ROOT) / "geometries" / str(problem["id"]).replace("/", "-")
    _extract(_archive_for(root, problem), dest)
    cons = [path for path in dest.rglob("*") if path.is_file() and path.name.endswith(".con")]
    if cons:
        paths = sorted(cons, key=lambda path: _index_of(path.name))
        loaded = []
        for path in paths:
            pos, _box = read_con(path.read_text(encoding="utf-8", errors="replace"))
            loaded.append((str(_index_of(path.name)), pos))
    else:
        coords = [path for path in dest.rglob("*") if path.is_file() and path.name.startswith("coords.")]
        if not coords:
            raise FileNotFoundError(f"{dest} has no starts")
        paths = sorted(coords, key=lambda path: _index_of(path.name))
        loaded = [(str(_index_of(path.name)), _read_plain(path.read_text(encoding="utf-8"))) for path in paths]
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(loaded) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(loaded)}")
    if problem.get("id") == "globalopt/lj75" and expected is None and len(loaded) != 200:
        raise ValueError(f"globalopt/lj75 expected 200 starts, found {len(loaded)}")
    size = _cluster_size(str(problem.get("id") or ""))
    centered = []
    for name, pos in loaded:
        if size is not None and pos.shape[0] != size:
            raise ValueError(f"{name} has {pos.shape[0]} atoms")
        flat = _center(pos)
        if np.any(np.abs(flat) > COORD_BOUND):
            raise ValueError(f"{name} extends past {COORD_BOUND}")
        centered.append((name, flat))
    return centered


def _cluster_size(problem_id: str) -> int | None:
    name = problem_id.rsplit("/", 1)[-1]
    if name.startswith("lj") and name[2:].isdigit():
        return int(name[2:])
    return None


def _center(pos: np.ndarray) -> np.ndarray:
    placed = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    placed = placed - placed.mean(axis=0)
    return np.ascontiguousarray(placed.reshape(-1))


def _section(template: dict) -> tuple[str, int, float | None]:
    section = template.get("scipy") or {}
    if "method" not in section:
        raise ValueError(f"{template['id']} needs scipy.method")
    method = str(section["method"])
    if "seed" not in section:
        raise ValueError(f"{template['id']} needs scipy.seed")
    seed = int(section["seed"])
    bound = section.get("bound")
    if method in BOUNDED:
        if bound is None:
            raise ValueError(f"{template['id']} needs scipy.bound")
        bound = float(bound)
        if abs(bound - COORD_BOUND) > 1e-12:
            raise ValueError(f"{template['id']} bound is {bound}, the shared bound is {COORD_BOUND}")
    elif bound is not None:
        raise ValueError(f"{template['id']} leaves {method} unbounded")
    return method, seed, None if bound is None else float(bound)


def _cap_of(template: dict) -> int:
    raw = template.get("failure_cap")
    stop = template.get("stop") or {}
    stated = stop.get("max_force_calls", stop.get("max_steps"))
    if raw is None and stated is None:
        raise ValueError(f"{template['id']} needs a force-call cap")
    cap = int(raw if raw is not None else stated)
    if stated is not None and int(stated) != cap:
        raise ValueError(f"{template['id']} stop cap disagrees with failure_cap")
    if cap <= 0:
        raise ValueError(f"{template['id']} cap must be positive")
    return cap


def scipy_install() -> dict:
    import importlib.metadata
    import scipy

    dist = importlib.metadata.distribution("scipy")
    return {
        "version": scipy.__version__,
        "file": str(scipy.__file__),
        "direct_url": dist.read_text("direct_url.json") or "",
    }


def require_source_build(commit: str) -> dict:
    """The campaign binary is the pinned source tree, not a wheel."""
    info = scipy_install()
    if info["version"] != "1.18.1":
        raise RuntimeError(f"scipy {info['version']} is not 1.18.1")
    url = info["direct_url"]
    if "scipy-src" not in url:
        raise RuntimeError("scipy is not the source build of the pinned checkout")
    head = SOURCE_TREE
    if not (head / ".git").exists():
        raise RuntimeError(f"{SOURCE_TREE} is not the SciPy checkout")
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(head), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{SOURCE_TREE} has no commit")
    found = proc.stdout.strip()
    if found != commit:
        raise RuntimeError(f"SciPy checkout {found} is not template commit {commit}")
    return info


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _public_record(record: dict) -> dict:
    """Drop the improvement trace from the committed start row."""
    body = {key: value for key, value in record.items() if key not in {"trace", "index"}}
    return body


def _dat(summary: dict, template: dict, metrics: dict, *, partial: bool, partial_reason: str, target: float | None) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "force_calls_median": summary.get("force_calls_median"),
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
        "cap": summary.get("cap"),
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
        "# counter harness. One force call is one lennard_jones evaluation.",
        "# A repeated coordinate is cached, so the gradient of that geometry is the same call.",
        "# The cap refuses the next search call and does not count it.",
    ]
    if target is None:
        lines.append("# This problem file states no energy window. The row is partial.")
    else:
        lines.append(f"# Success is a finite harness energy at or below {target:.12g} with calls under the cap.")
    if partial:
        lines.append("# " + partial_reason)
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    if partial:
        rows["partial"] = "true"
        rows["partial_reason"] = partial_reason
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _legacy(records: list[dict], cap: int, target: float | None) -> dict:
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if target is None:
        counts = [int(record["force_calls"]) for record in records]
        summary = summarize_all(counts, len(records))
        summary["median"] = summary.get("force_calls_median")
    else:
        counts = [int(record["force_calls"]) if record["success"] else cap for record in records]
        summary = summarize_capped(counts, cap)
        successes = [int(record["force_calls"]) for record in records if record["success"]]
        summary["median"] = statistics.median(successes) if successes else None
        if successes:
            summary["force_calls_median"] = summary["median"]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    summary["cap"] = cap
    return summary


def _partial_reason(problem: dict, records: list[dict], expected: int | None, target: float | None) -> tuple[bool, str]:
    reasons = []
    if target is None:
        reasons.append(
            "no energy window on this problem; each start stores its lowest harness energy and call count"
        )
    if expected is not None and len(records) < int(expected):
        reasons.append(f"scored {len(records)} of {int(expected)} starts")
    if not reasons:
        return False, ""
    return True, "; ".join(reasons)


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
    resume: bool = True,
    check_source: bool = True,
) -> dict:
    del geometries
    version = str(template.get("code_version") or "")
    commit = str(template.get("commit") or "")
    if not commit:
        raise RuntimeError(f"{template['id']} has no commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    if check_source:
        info = require_source_build(commit)
        if info["version"] != version:
            raise RuntimeError(f"scipy {info['version']} does not match template {version}")
    method, base_seed, bound = _section(template)
    cap = _cap_of(template)
    target = energy_target(problem)
    work_dir = Path(work) if work is not None else CHECKPOINT_ROOT
    starts = load_starts(root, problem, work_dir)
    if limit and limit > 0:
        starts = starts[: int(limit)]
    jobs = []
    for index, (name, coords) in enumerate(starts):
        parameters = method_parameters(method, cap, coords.size, bound)
        parameters["seed"] = base_seed + index
        job = {
            "index": index,
            "start": name,
            "coords": coords.tolist(),
            "cap": cap,
            "method": method,
            "seed": base_seed + index,
            "bound": bound,
            "target": target,
            "commit": commit,
            "code_version": version,
            "parameters": parameters,
            "template_id": template["id"],
            "checkpoint": str(work_dir / "checkpoints" / template["id"].replace("/", "-") / f"{index:04d}.json"),
            "limit_address_space": method == "shgo",
        }
        jobs.append(job)
    records = []
    pending = []
    for job in jobs:
        saved = _load_checkpoint(job) if resume else None
        if saved is not None:
            saved["index"] = job["index"]
            records.append(saved)
        else:
            pending.append(job)
    if pending:
        if workers <= 1:
            produced = []
            for done, job in enumerate(pending, 1):
                produced.append(search_start(job))
                if done == len(pending) or done % 10 == 0:
                    print(f"{template['id']} {done}/{len(pending)}", flush=True)
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            pool_workers = min(int(workers), 4) if method == "shgo" else int(workers)
            with ProcessPoolExecutor(
                max_workers=pool_workers,
                mp_context=context,
                max_tasks_per_child=1 if method == "shgo" else None,
            ) as pool:
                futures = [pool.submit(search_start, job) for job in pending]
                produced = []
                total = len(futures)
                for done, future in enumerate(as_completed(futures), 1):
                    produced.append(future.result())
                    if done == total or done % 10 == 0:
                        print(f"{template['id']} {done}/{total}", flush=True)
        records.extend(produced)
    records.sort(key=lambda row: int(row["index"]))
    expected = (problem.get("sample") or {}).get("starts")
    if problem.get("id") == "globalopt/lj75" and expected is None:
        expected = 200
    partial, reason = _partial_reason(problem, records, None if not expected else int(expected), target)
    if limit and limit > 0 and expected is not None and len(records) < int(expected):
        partial = True
        short = f"scored {len(records)} of {int(expected)} starts"
        if short not in reason:
            reason = f"{reason}; {short}" if reason else short
    summary = _legacy(records, cap, target)
    public = [_public_record(record) for record in records]
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in public
        ],
        cap,
    )
    passed = {
        "method": method,
        "cap": cap,
        "seed": base_seed,
        "bound": bound,
        "target": target,
    }
    provenance = collect_provenance(
        code="SciPy",
        code_version=version,
        commit=commit,
        parameters=passed,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": reason or "Harness counter on optbench.potentials.lennard_jones.",
        "runs": [
            {
                "entry": output.stem,
                "code": "SciPy",
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
                "starts": public,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _dat(summary, template, metrics, partial=partial, partial_reason=reason, target=target),
        encoding="utf-8",
    )
    write_starts(output.with_suffix(".starts.jsonl"), public)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["nfailed"] = summary.get("nfailed", metrics["nfailed"])
    return summary
