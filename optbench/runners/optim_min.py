"""Minimize OptBench starts with OPTIM's LBFGS and ``USERPOT``.

The binary is the Wales-group OPTIM built from the pinned svn export.
``BFGSMIN`` is LBFGS. ``USERPOT`` is last in ``odata``, and the
coordinates follow it. Each geometry is one TCP ``calculate`` on
``PotServer``. ``HarnessCounter(cache=True)`` counts that geometry once
when OPTIM asks for the energy and then the gradient.

The RMS stop is the problem force norm divided by ``sqrt(3N)``. The
recorded force norm is a direct clean evaluation and is not a search
call. A v2 success is under the cap, at the force stop, and Hessian
index 0. On LJ38 that index drops the six rigid modes. Noise adds
``N(0, sigma^2)`` after the clean call. A positive sigma also requires
the clean-energy window. OPTIM is started again from the last point
with ``GMAX`` 0 when the RMS stop is met and the window is not.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import HarnessCounter
from optbench.failures import classify_failure
from optbench.noise import KAPPA, WINDOW, converged_under_noise, stop_threshold, window_energy_tolerance
from optbench.potentials import lennard_jones, morse_bulk
from optbench.potserver import PotServer
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

VERSION_RE = re.compile(r"OPTIM version\s+(\S+),")
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_ATTEMPT_LIMIT = 8
_TIMEOUT_S = 3600


class WindowMet(RuntimeError):
    """The noisy stop is met. The evaluation that raised it is counted."""


def fortran_d(value: float) -> str:
    """List-directed real OPTIM's READF accepts. The exponent letter is D."""
    return f"{float(value):.16E}".replace("E", "D")


def gmax(force_norm: float, atoms: int) -> float:
    """RMS convergence. OPTIM compares ``||g|| / sqrt(3N)`` with this value."""
    if atoms < 1:
        raise ValueError("atoms must be positive")
    if force_norm < 0.0:
        raise ValueError("force_norm must be non-negative")
    return float(force_norm) / math.sqrt(3 * atoms)


def optim_binary() -> Path | None:
    """``$OPTIM``, ``OPTIM`` on ``PATH``, or the build-script prefix."""
    env = os.environ.get("OPTIM")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            return path
    found = shutil.which("OPTIM")
    if found:
        return Path(found)
    default = Path(os.environ.get("OPTIM_PREFIX", "/tmp/optbench-optim")) / "OPTIM"
    if default.is_file() and os.access(default, os.X_OK):
        return default
    return None


def write_odata(
    path: Path,
    coords: list[float],
    *,
    gmax_value: float,
    updates: int,
    maxbfgs: float,
    steps: int,
) -> None:
    """``USERPOT`` is the last keyword. FETCHZ reads the coordinates after it."""
    if len(coords) % 3 != 0:
        raise ValueError("coordinates must be a multiple of 3")
    lines = [
        f"BFGSMIN {fortran_d(gmax_value)}",
        f"BFGSSTEPS {int(steps)}",
        f"UPDATES {int(updates)}",
        f"MAXBFGS {fortran_d(maxbfgs)}",
        "USERPOT",
    ]
    for atom in range(len(coords) // 3):
        x, y, z = coords[3 * atom : 3 * atom + 3]
        lines.append(f"X {fortran_d(x)} {fortran_d(y)} {fortran_d(z)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xyz(path: Path) -> list[float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} is not an xyz file")
    count = int(lines[0].split()[0])
    coords: list[float] = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path} has a short atom line")
        coords.extend(float(part) for part in parts[-3:])
    if len(coords) != 3 * count:
        raise ValueError(f"{path} announced {count} atoms")
    return coords


def _quiet_threads() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _box9(box) -> list[float]:
    if box is None:
        return [0.0] * 9
    lengths = [float(value) for value in box]
    if len(lengths) != 3:
        raise ValueError("box is three orthorhombic lengths")
    return [lengths[0], 0.0, 0.0, 0.0, lengths[1], 0.0, 0.0, 0.0, lengths[2]]


def _norm(gradient) -> float:
    return math.sqrt(sum(value * value for value in gradient))


class _Search:
    """Clean potential, optional noise, and the point of the last fresh call."""

    def __init__(self, job: dict, counter: HarnessCounter):
        self.kind = job["kind"]
        self.box = None if job["box"] is None else job["box"]
        self.sigma = float(job["sigma"])
        self.clean_tol = float(job["clean_tol"])
        self.energy_tol = float(job["energy_tol"])
        self.counter = counter
        self.rng = random.Random(int(job["seed"]) + int(job["index"]))
        self.energies: list[float] = []
        self.point: list[float] | None = None
        self.last_norm: float | None = None
        self.met = False

    def evaluate(self, positions, _numbers=None, _box=None):
        coords = [float(value) for value in positions]
        if self.kind == "lj":
            energy, gradient = lennard_jones(coords)
        else:
            energy, gradient = morse_bulk(coords, self.box)
        clean = tuple(float(value) for value in gradient)
        if self.counter._verifying:
            return float(energy), clean
        noisy = clean
        if self.sigma != 0.0:
            noisy = tuple(value + self.rng.gauss(0.0, self.sigma) for value in clean)
        self.point = coords
        self.energies.append(float(energy))
        self.last_norm = _norm(noisy)
        if self.sigma != 0.0 and converged_under_noise(
            force_norm=self.last_norm,
            energies=self.energies,
            clean_tol=self.clean_tol,
            sigma=self.sigma,
            components=len(coords),
            energy_tol=self.energy_tol,
            kappa=KAPPA,
            window=WINDOW,
        ):
            self.met = True
            raise WindowMet("window met")
        return float(energy), noisy


def _run_once(job: dict, binary: str, work: Path, search: _Search, server: PotServer, point: list[float], gmax_value: float) -> tuple[int, str, list[float] | None]:
    work.mkdir(parents=True, exist_ok=True)
    for child in work.iterdir():
        if child.is_file():
            child.unlink()
    write_odata(
        work / "odata",
        point,
        gmax_value=gmax_value,
        updates=int(job["updates"]),
        maxbfgs=float(job["maxbfgs"]),
        steps=int(job["cap"]),
    )
    env = os.environ.copy()
    env.update({name: "1" for name in _THREAD_VARS})
    env["OPTBENCH_POTSERVER"] = server.endpoint()
    env["OPTBENCH_Z"] = "1"
    env["OPTBENCH_BOX"] = " ".join(fortran_d(value) for value in _box9(job["box"]))
    try:
        proc = subprocess.run(
            [binary],
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return 124, stdout + "\n" + stderr, None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    dumped = None
    for name in ("converged.xyz", "unconverged.xyz"):
        path = work / name
        if path.is_file():
            try:
                dumped = read_xyz(path)
            except ValueError:
                dumped = None
            break
    return int(proc.returncode), text, dumped


def _minimize_job(job: dict) -> dict:
    _quiet_threads()
    import time

    started = time.perf_counter()
    binary = str(job["binary"])
    cap = int(job["cap"])
    sigma = float(job["sigma"])
    clean_tol = float(job["clean_tol"])
    components = len(job["coords"])
    atoms = components // 3
    threshold = stop_threshold(clean_tol, sigma, components, KAPPA)
    counter = HarnessCounter(cache=True, cap=cap)
    search = _Search(job, counter)
    server = PotServer(search.evaluate, counter).start()
    error = ""
    exit_code = None
    text = ""
    point = [float(value) for value in job["coords"]]
    optim_converged = False
    attempts = 0
    root = Path(job["work"])
    try:
        opening = gmax(threshold, atoms)
        for attempt in range(_ATTEMPT_LIMIT):
            attempts = attempt + 1
            before = counter.calls
            step_gmax = opening if attempt == 0 else 0.0
            exit_code, text, dumped = _run_once(
                job, binary, root / f"attempt-{attempt}", search, server, point, step_gmax
            )
            optim_converged = "**** CONVERGED ****" in text
            if search.met or exit_code == 5:
                if search.point is not None:
                    point = search.point
                break
            if exit_code == 3:
                if search.point is not None:
                    point = search.point
                break
            if exit_code == 124:
                error = "TimeoutExpired"
                if search.point is not None:
                    point = search.point
                break
            if exit_code not in (0,):
                tail = " ".join(text.strip().split())
                error = f"OPTIM exit {exit_code}: {tail[-400:]}"
                if dumped is not None:
                    point = dumped
                elif search.point is not None:
                    point = search.point
                break
            if dumped is not None:
                point = dumped
            elif search.point is not None:
                point = search.point
            if sigma == 0.0 or search.met or counter.calls >= cap or counter.calls == before:
                break
            if not optim_converged and counter.calls == before:
                break
        else:
            if not search.met and sigma != 0.0 and not error:
                error = ""
    finally:
        server.stop()
    version = None
    match = VERSION_RE.search(text)
    if match:
        version = match.group(1)
    calls = int(counter.calls)
    energy = None
    force_norm = None
    verify_calls = 0
    if point is not None and not error:
        try:
            if job["kind"] == "lj":
                raw_energy, gradient = lennard_jones(point)
            else:
                raw_energy, gradient = morse_bulk(point, job["box"])
            energy = float(raw_energy)
            force_norm = _norm(gradient)
            verify_calls = 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    finite = (
        point is not None
        and all(math.isfinite(value) for value in point)
        and energy is not None
        and math.isfinite(energy)
        and force_norm is not None
        and math.isfinite(force_norm)
    )
    if sigma == 0.0:
        stopped = bool(optim_converged and exit_code == 0)
    else:
        stopped = bool(search.met)
    cap_hit = exit_code == 3 or calls >= cap
    at_force = (
        finite
        and stopped
        and force_norm is not None
        and force_norm <= threshold
        and calls < cap
        and not error
        and not cap_hit
    )
    hessian_index = None
    min_eigenvalues: list[float] = []
    if point is not None and finite and (not error or cap_hit):
        try:
            if job["kind"] == "lj":
                from optbench.runners.noise_min import _index

                hessian_index, min_eigenvalues = _index(point, threshold)
            else:
                from optbench.runners.rgmin_min import _index_at

                hessian_index, min_eigenvalues = _index_at("morse", point, job["box"], clean_tol)
        except Exception as exc:
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    reasons: list[str] = []
    if error and not cap_hit:
        reasons.append("crash")
    elif not finite:
        reasons.append("divergence")
    elif cap_hit or calls >= cap:
        reasons.append("cap")
    elif not at_force:
        reasons.append("force_norm")
    elif hessian_index != 0:
        reasons.append("hessian_index")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "verify_calls": verify_calls,
        "counter": "harness",
        "comparable": True,
        "success": failure is None,
        "failure": failure,
        "reasons": reasons,
        "energy": energy,
        "force_norm": force_norm,
        "hessian_index": hessian_index,
        "expected_index": 0,
        "converged": at_force,
        "hessian_calls": 0,
        "min_eigenvalues": min_eigenvalues,
        "seconds": time.perf_counter() - started,
        "optim_exit": exit_code,
        "optim_converged": optim_converged,
        "optim_version": version,
        "attempts": attempts,
        "sigma": sigma,
        "seed": int(job["seed"]) + int(job["index"]),
    }
    if error:
        record["error"] = error
    path = job.get("record_path")
    if path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(dest)
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, commit: str, version: str, sigma: float) -> str:
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
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "sigma": sigma,
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
    if summary.get("partial"):
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one USERPOT geometry.",
        "# An energy request and a gradient request at that geometry are one call.",
        "# The recorded force norm is a direct clean evaluation and is not a search call.",
        "# v2 success is under the cap, at the force stop, and Hessian index 0.",
        "# energy and Kabsch RMSD were not applied: the problem file has no per-start reference.",
        "# energy_min and energy_max are the final potential values on the recorded points.",
    ]
    if str(template.get("problem", "")).startswith("minimization/lj38"):
        lines.append(
            "# LJ38 index removes the six rigid modes. A curvature is negative only below "
            "-max(force_norm, 1e-8*||H||_F, 1e-6)."
        )
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _options(template: dict) -> tuple[int, float]:
    section = template.get("optim") or {}
    updates = int(section.get("updates", 4))
    maxbfgs = float(section.get("maxbfgs", 0.2))
    if updates < 1:
        raise ValueError(f"{template['id']} updates must be positive")
    if maxbfgs <= 0.0:
        raise ValueError(f"{template['id']} maxbfgs must be positive")
    return updates, maxbfgs


def _kind(problem_id: str) -> str:
    if problem_id in {"minimization/lj38", "minimization/lj38-noise"}:
        return "lj"
    if problem_id == "minimization/morse-bulk":
        return "morse"
    raise ValueError(f"{problem_id} has no OPTIM potential")


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
    _quiet_threads()
    binary = optim_binary()
    if binary is None:
        raise RuntimeError("OPTIM is not on PATH and $OPTIM is unset")
    version = str(template.get("code_version") or "")
    commit = str(template.get("commit") or "")
    if not version or not commit:
        raise RuntimeError(f"{template['id']} has no code_version or commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    convergence = problem.get("convergence") or {}
    clean_tol = float(convergence["force_norm"])
    cap = int(convergence["max_force_calls"])
    updates, maxbfgs = _options(template)
    kind = _kind(problem["id"])
    noise = template.get("noise") or {}
    sigma = float(noise.get("sigma", 0.0))
    if problem["id"] == "minimization/lj38-noise":
        if "energy_tol" not in convergence:
            raise ValueError(f"{problem['id']} has no energy_tol for the noise window")
        kappa = float((problem.get("noise") or {}).get("kappa", KAPPA))
        energy_tol = window_energy_tolerance(float(convergence["energy_tol"]), sigma, kappa)
        seed = int(noise.get("seed", (problem.get("noise") or {})["seed"]))
    else:
        if sigma != 0.0:
            raise ValueError(f"{template['id']} sets sigma on a clean problem")
        energy_tol = 0.0
        seed = 0
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if problem["id"] == "minimization/morse-bulk" and geometries is None and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    if limit:
        loaded = loaded[: int(limit)]
    slug = str(template["id"]).replace("/", "-")
    run_root = Path(work) if work is not None else Path("/tmp/optbench-optim-runs") / slug
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    _require_version(binary, version, run_root / "preflight")
    jobs = []
    cached: list[dict] = []
    for index, (name, coords, box) in enumerate(loaded):
        record_path = records_dir / f"{index:04d}.json"
        if resume and record_path.is_file():
            cached.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": [float(value) for value in coords],
                "box": None if kind == "lj" or box is None else [float(value) for value in box],
                "kind": kind,
                "sigma": sigma,
                "seed": seed,
                "clean_tol": clean_tol,
                "energy_tol": energy_tol,
                "cap": cap,
                "updates": updates,
                "maxbfgs": maxbfgs,
                "binary": str(binary),
                "code_version": version,
                "work": str(run_root / f"{index:04d}"),
                "record_path": str(record_path),
            }
        )
    fresh: list[dict] = []
    if jobs:
        if workers <= 1:
            for done, job in enumerate(jobs, 1):
                fresh.append(_minimize_job(job))
                if done == len(jobs) or done % 25 == 0:
                    print(f"{template['id']} {done + len(cached)}/{len(loaded)}", flush=True)
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_minimize_job, job) for job in jobs]
                total = len(futures)
                for done, future in enumerate(as_completed(futures), 1):
                    fresh.append(future.result())
                    if done == total or done % 25 == 0:
                        print(f"{template['id']} {done + len(cached)}/{len(loaded)}", flush=True)
                        _publish_checkpoint(
                            problem,
                            template,
                            records_dir,
                            output=output,
                            cap=cap,
                            sigma=sigma,
                            updates=updates,
                            maxbfgs=maxbfgs,
                            clean_tol=clean_tol,
                            energy_tol=energy_tol,
                            version=version,
                            commit=commit,
                            expected=None if limit else _expected_starts(problem),
                        )
    records = _read_records(records_dir) if (resume or jobs) else cached + fresh
    if not records:
        records = cached + fresh
    records.sort(key=lambda row: row["index"])
    found = {str(row.get("optim_version")) for row in records if row.get("optim_version")}
    if found and found != {version}:
        raise RuntimeError(f"OPTIM version {sorted(found)} does not match template {version}")
    expected = _expected_starts(problem)
    partial = expected is not None and len(records) < expected
    return _publish(
        problem,
        template,
        records,
        output=output,
        cap=cap,
        sigma=sigma,
        updates=updates,
        maxbfgs=maxbfgs,
        clean_tol=clean_tol,
        energy_tol=energy_tol,
        version=version,
        commit=commit,
        partial=partial,
    )


def _expected_starts(problem: dict) -> int | None:
    expected = (problem.get("sample") or {}).get("starts")
    if expected is None and problem["id"] == "minimization/morse-bulk":
        return 100
    if expected is None:
        return None
    return int(expected)


def _read_records(records_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(records_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: row["index"])
    return rows


def _publish_checkpoint(
    problem,
    template,
    records_dir: Path,
    *,
    output: Path,
    cap: int,
    sigma: float,
    updates: int,
    maxbfgs: float,
    clean_tol: float,
    energy_tol: float,
    version: str,
    commit: str,
    expected: int | None,
) -> None:
    records = _read_records(records_dir)
    if not records:
        return
    partial = expected is not None and len(records) < expected
    _publish(
        problem,
        template,
        records,
        output=output,
        cap=cap,
        sigma=sigma,
        updates=updates,
        maxbfgs=maxbfgs,
        clean_tol=clean_tol,
        energy_tol=energy_tol,
        version=version,
        commit=commit,
        partial=partial,
    )


def _require_version(binary: Path, version: str, work: Path) -> None:
    """One dimer at the Lennard-Jones minimum. The header must name ``version``."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    distance = 2.0 ** (1.0 / 6.0)
    counter = HarnessCounter(cache=True, cap=20)

    def evaluate(positions, _numbers=None, _box=None):
        return lennard_jones(positions)

    server = PotServer(evaluate, counter).start()
    try:
        write_odata(
            work / "odata",
            [0.0, 0.0, 0.0, distance, 0.0, 0.0],
            gmax_value=gmax(0.01, 2),
            updates=4,
            maxbfgs=0.2,
            steps=20,
        )
        env = os.environ.copy()
        env.update({name: "1" for name in _THREAD_VARS})
        env["OPTBENCH_POTSERVER"] = server.endpoint()
        env["OPTBENCH_Z"] = "1"
        env["OPTBENCH_BOX"] = " ".join(fortran_d(0.0) for _ in range(9))
        proc = subprocess.run(
            [str(binary)],
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        server.stop()
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = VERSION_RE.search(text)
    got = match.group(1) if match else ""
    if got != version:
        tail = " ".join(text.strip().split())[-400:]
        raise RuntimeError(f"OPTIM version {got or 'missing'} does not match {version}: {tail}")


def _publish(
    problem,
    template,
    records,
    *,
    output: Path,
    cap: int,
    sigma: float,
    updates: int,
    maxbfgs: float,
    clean_tol: float,
    energy_tol: float,
    version: str,
    commit: str,
    partial: bool,
) -> dict:
    stored = []
    for record in records:
        row = dict(record)
        row.pop("index", None)
        stored.append(row)
    legacy_counts = [cap if not record["converged"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in records if record["converged"]]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    if partial:
        summary["partial"] = True
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
    passed = {
        "updates": updates,
        "maxbfgs": maxbfgs,
        "gmax": "force_norm/sqrt(3N)",
        "bfgssteps": cap,
        "gtol": clean_tol,
        "sigma": sigma,
        "energy_tol": energy_tol,
    }
    provenance = collect_provenance(
        code="optim",
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
        "note": (
            "Harness counter. USERPOT charges one geometry once. "
            "The recorded force norm is a direct clean evaluation. "
            "Energy and Kabsch RMSD were not applied."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "optim",
                "algorithm": template.get("algorithm", "LBFGS"),
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
                "starts": stored,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, sigma), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), stored)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    return summary
