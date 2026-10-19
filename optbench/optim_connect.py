"""OPTIM doubly-nudged elastic band with hybrid eigenvector following.

The binary is the Wales OPTIM build pinned in ``codes/registry.toml``.
Each ``POTENTIAL`` entry is one uncached socket calculate on the spec
potential. The judged point is a refined saddle from that connect run.
Verification Hessian calls are not search cost.

This is the problem-image setup (``optim/dneb-hef``). Image count, step
length, and ``STEPS`` come from the problem. The 2014 preambles live in
``templates/optim/preambles`` and are scored by ``optim/de-lj38``,
``optim/de-pt``, and ``optim/de-2ts``.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import tempfile
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .counter import ForceCallCap, HarnessCounter
from .failures import classify_failure
from .pes import evaluate
from .potserver import PotServer, RgpotCounter
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance, write_starts

_FD_STEP = 1e-5
_CODE_ID = "optim"
_ENGINE = "optim-connect"


def optim_binary() -> Path | None:
    """The OPTIM executable named by ``OPTIM``, or the build on this machine."""
    env = os.environ.get("OPTIM")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            return path
    default = Path("/tmp/optim-build/OPTIM")
    if default.is_file() and os.access(default, os.X_OK):
        return default
    found = shutil.which("OPTIM")
    if found:
        return Path(found)
    return None


def _threads_one() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _registry(root: Path) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == _CODE_ID:
            return row
    raise ValueError("codes/registry.toml has no optim entry")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the optim registry")
    if template.get("engine") != _ENGINE:
        raise ValueError(f"unknown engine {template.get('engine')}")
    band = problem["band"]
    convergence = problem["convergence"]
    stated = template.get("band") or {}
    return {
        "potential": str(band["potential"]),
        "n_interior": int(band["interior_images"]),
        "max_move": float(band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": problem["geometry"]["kind"],
        "iteration_density": int(stated.get("iteration_density", 30)),
        "neb_rms": float(stated.get("neb_rms", 0.01)),
        "hef_rms": float(stated.get("hef_rms", 1.0e-5)),
        "spring": float(stated.get("spring", 1.0)),
        "connect_cycles": int(stated.get("connect_cycles", 1)),
        "connect_tries": int(stated.get("connect_tries", 1)),
    }


def minimum_image_finish(
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
) -> np.ndarray:
    """Product coordinates unwrapped onto the reactant.

    Frozen atoms stay on the reactant. Free atoms move by the
    minimum-image displacement, which is how the band path is built.
    OPTIM then reads Cartesian ``finish`` and does not turn on its own
    periodic keywords.
    """
    start = np.asarray(reactant, dtype=np.float64)
    end = np.asarray(product, dtype=np.float64)
    mask = np.asarray(free, dtype=bool)
    delta = end - start
    if box is not None:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        delta = delta - lengths * np.round(delta / lengths)
    finish = np.array(start, copy=True)
    finish[mask] = start[mask] + delta[mask]
    return finish


def _odata_float(value: float) -> str:
    """Decimal text. OPTIM's reader rejects an exponent."""
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-", "-0"} else "0"


def write_odata(
    path: Path,
    reactant: np.ndarray,
    *,
    interior: int,
    cap: int,
    max_move: float,
    iteration_density: int,
    frozen: list[int],
    neb_rms: float = 0.01,
    spring: float = 1.0,
    hef_rms: float = 1.0e-5,
    connect_cycles: int = 1,
    connect_tries: int = 1,
) -> None:
    """One connect input. Image count, step length, and step budget are the problem.

    ``STEPS`` is that force-call budget. OPTIM's compiled default is one
    iteration, which is the PATHSAMPLE single-step mode. ``STEPS`` also
    copies that limit onto the LBFGS minimizer. ``BFGSMIN`` makes the
    connection quench gradient-only: the default path asks for a Hessian,
    and this potential has none. The spring is OPTIM's own default.
    ``CONVERGE`` is left unset: writing it also tightens ``GMAX``, and
    that number is the endpoint RMS test. The hybrid eigenvector search
    then tightens its own RMS to the compiled ``CONVR`` of 1e-5.
    """
    if float(hef_rms) <= 0:
        raise ValueError("hef_rms must be positive")
    iterations = int(interior) * int(iteration_density)
    step = float(max_move)
    lines = [
        f"NEWCONNECT {int(connect_cycles)} {int(connect_tries)}",
        f"NEWNEB {int(interior)} {iterations} {_odata_float(neb_rms)}",
        "BFGSTS",
        "BFGSMIN",
        "NOHESS",
        "NOFRQS",
        "DUMPALLPATHS",
        f"STEPS {int(cap)}",
        f"NEBK {_odata_float(spring)}",
        f"MAXSTEP {_odata_float(step)}",
        f"MAXMAX {_odata_float(step)}",
        f"MAXBFGS {_odata_float(step)} {_odata_float(step)} {_odata_float(step)} {_odata_float(step)}",
    ]
    if frozen:
        lines.append("FREEZE")
    lines.append("POINTS")
    for xyz in np.asarray(reactant, dtype=np.float64):
        lines.append(f"XX {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_finish(path: Path, coords: np.ndarray) -> None:
    rows = [
        f"{xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}"
        for xyz in np.asarray(coords, dtype=np.float64)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_frozen(path: Path, indices: list[int]) -> None:
    body = [str(len(indices)), *[str(index) for index in indices]]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def read_dumped_points(text: str, n_atoms: int) -> list[np.ndarray]:
    """Blocks written by ``HARNESS_DUMP_TS``: a count, then that many coordinates."""
    values: list[float] = []
    for token in text.split():
        try:
            values.append(_fortran_float(token))
        except ValueError:
            continue
    expected = 3 * int(n_atoms)
    points: list[np.ndarray] = []
    cursor = 0
    while cursor < len(values):
        count = int(round(values[cursor]))
        cursor += 1
        if count != expected or cursor + count > len(values):
            break
        coords = np.asarray(values[cursor : cursor + count], dtype=np.float64).reshape((n_atoms, 3))
        cursor += count
        points.append(coords)
    return points


def read_path_info(text: str, n_atoms: int) -> list[np.ndarray]:
    """Transition-state blocks in a ``NOFRQS`` path.info.

    A stationary point is an energy, a point-group line, and one
    Cartesian line per atom. Minima and saddles alternate, starting
    with a minimum, so the saddles are the even blocks.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[np.ndarray] = []
    index = 0
    while index + 1 < len(lines):
        energy = lines[index].split()
        if len(energy) != 1:
            index += 1
            continue
        try:
            _fortran_float(energy[0])
        except ValueError:
            index += 1
            continue
        index += 2
        coords: list[list[float]] = []
        while index < len(lines) and len(coords) < n_atoms:
            parts = lines[index].split()
            if len(parts) < 3:
                break
            try:
                coords.append([_fortran_float(parts[0]), _fortran_float(parts[1]), _fortran_float(parts[2])])
            except ValueError:
                break
            index += 1
        if len(coords) == n_atoms:
            blocks.append(np.asarray(coords, dtype=np.float64))
    return blocks[1::2]


def _deduped(points: list[np.ndarray]) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for point in points:
        key = tuple(np.round(np.asarray(point, dtype=np.float64).reshape(-1), 8))
        if key in seen:
            continue
        seen.add(key)
        kept.append(np.asarray(point, dtype=np.float64))
    return kept


def _candidate_points(work: Path, n_atoms: int) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    path_info = work / "path.info"
    if path_info.is_file():
        points.extend(read_path_info(path_info.read_text(encoding="utf-8", errors="replace"), n_atoms))
    dumped = work / "optbench.ts"
    if dumped.is_file():
        points.extend(read_dumped_points(dumped.read_text(encoding="utf-8", errors="replace"), n_atoms))
    return _deduped(points)


def _free_force(grad: np.ndarray, free: np.ndarray) -> tuple[np.ndarray, float]:
    vector = np.asarray(grad, dtype=np.float64).reshape((-1, 3)).copy()
    vector[~np.asarray(free, dtype=bool)] = 0.0
    return vector, float(np.linalg.norm(vector))


def _judge(
    points: list[np.ndarray],
    *,
    kind: str,
    free: np.ndarray,
    box: np.ndarray | None,
    counter: HarnessCounter,
    force_norm: float,
    cap: int,
    project_rigid: bool,
) -> dict:
    """Pick a dumped saddle. One search call reads its force. The Hessian is verification.

    OPTIM's ``Converged to TS`` line requires a negative lowest mode.
    The harness index is the free-atom Hessian after rigid projection.
    A higher index is ``wrong_stationary`` when that line is present.
    """
    mask = np.asarray(free, dtype=bool)
    box_flat = None if box is None else tuple(float(value) for value in np.asarray(box, dtype=np.float64).reshape(-1))

    def evaluate_full(coords, _numbers=None, _box=None):
        xyz = np.asarray(coords, dtype=np.float64).reshape((-1, 3))
        energy, grad = evaluate(kind, xyz, None if kind == "lj" else box)
        return float(energy), np.asarray(grad, dtype=np.float64).reshape(-1)

    rgpot = RgpotCounter(evaluate_full, counter)
    best_force = None
    best_point = None
    best_energy = float("nan")
    best_grad = None
    wrong = None
    for point in points:
        if counter.calls >= cap:
            break
        try:
            energy, grad = rgpot.calculate(np.asarray(point, dtype=np.float64).reshape(-1), None, box_flat)
        except ForceCallCap:
            break
        grad_full, force = _free_force(grad, mask)
        if best_force is None or force < best_force:
            best_force = force
            best_point = np.asarray(point, dtype=np.float64)
            best_energy = float(energy)
            best_grad = grad_full
        if not np.isfinite(force) or force > force_norm:
            continue
        hessian_index = _index(point, mask, kind, box, rgpot, box_flat, project_rigid, counter)
        if hessian_index == 1:
            return {
                "point": np.asarray(point, dtype=np.float64),
                "energy": float(energy),
                "force_norm": force,
                "hessian_index": 1,
                "reasons": [],
            }
        wrong = {
            "point": np.asarray(point, dtype=np.float64),
            "energy": float(energy),
            "force_norm": force,
            "hessian_index": hessian_index,
        }
    if counter.calls >= cap:
        reasons = ["cap"]
    elif wrong is not None:
        return {
            "point": wrong["point"],
            "energy": wrong["energy"],
            "force_norm": wrong["force_norm"],
            "hessian_index": wrong["hessian_index"],
            "reasons": ["hessian_index"],
        }
    elif best_point is None:
        reasons = ["divergence"]
    else:
        reasons = ["force_norm"]
    return {
        "point": best_point,
        "energy": best_energy,
        "force_norm": None if best_force is None else best_force,
        "hessian_index": None,
        "reasons": reasons,
    }


def _index(point, free, kind, box, rgpot, box_flat, project_rigid: bool, counter: HarnessCounter) -> int:
    mask = np.asarray(free, dtype=bool)
    base = np.asarray(point, dtype=np.float64)
    free_point = base[mask].reshape(-1)

    def gradient(flat):
        full = np.array(base, copy=True)
        full[mask] = np.asarray(flat, dtype=np.float64).reshape((-1, 3))
        _energy, grad = rgpot.calculate(full.reshape(-1), None, box_flat)
        return np.asarray(grad, dtype=np.float64).reshape((-1, 3))[mask].reshape(-1)

    with counter.verification():
        hessian = numerical_hessian(gradient, free_point.tolist(), step=_FD_STEP)
    return vibrational_index(
        np.asarray(hessian, dtype=np.float64),
        base[mask],
        project_rigid and int(np.count_nonzero(mask)) >= 2,
    )


def _record(
    *,
    name: str,
    calls: int,
    verify_calls: int,
    success: bool,
    failure: str | None,
    reasons: list[str],
    energy: float,
    force: float | None,
    hessian_index: int | None,
    crash: str | None = None,
) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify_calls),
        "counter": "harness",
        "call_unit": "total",
        "comparable": True,
        "success": bool(success),
        "failure": failure,
        "reasons": reasons,
        "energy": None if energy is None or not np.isfinite(energy) else float(energy),
        "force_norm": None if force is None or not np.isfinite(force) else float(force),
        "hessian_index": hessian_index,
        "crash": crash,
    }


def _execute(job: dict) -> dict:
    _threads_one()
    binary = optim_binary()
    name = job["name"]
    if binary is None:
        return _record(
            name=name,
            calls=0,
            verify_calls=0,
            success=False,
            failure="crash",
            reasons=["crash"],
            energy=float("nan"),
            force=None,
            hessian_index=None,
            crash="missing OPTIM binary",
        )
    settings = job["settings"]
    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = np.asarray(job["product"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    kind = settings["potential"]
    cap = int(settings["cap"])
    force_norm = float(settings["force_norm"])
    project_rigid = bool(np.all(free) and kind == "lj")
    frozen = [index + 1 for index, flag in enumerate(free) if not flag]
    counter = HarnessCounter(cache=False, cap=cap)

    def evaluate_server(coords, _numbers, _box):
        xyz = np.asarray(coords, dtype=np.float64).reshape((-1, 3))
        energy, grad = evaluate(kind, xyz, None if kind == "lj" else box)
        return float(energy), np.asarray(grad, dtype=np.float64).reshape(-1)

    server = PotServer(evaluate_server, counter, cap=cap).start()
    work = Path(tempfile.mkdtemp(prefix="optim-connect-"))
    stdout = ""
    stderr = ""
    returncode = None
    try:
        write_odata(
            work / "odata",
            reactant,
            interior=settings["n_interior"],
            cap=cap,
            max_move=settings["max_move"],
            iteration_density=settings["iteration_density"],
            frozen=frozen,
            neb_rms=settings["neb_rms"],
            spring=settings["spring"],
            hef_rms=settings["hef_rms"],
            connect_cycles=int(settings.get("connect_cycles", 1)),
            connect_tries=int(settings.get("connect_tries", 1)),
        )
        write_finish(work / "finish", minimum_image_finish(reactant, product, free, box))
        if frozen:
            write_frozen(work / "frozen", frozen)
        env = os.environ.copy()
        env["OPTBENCH_POTSERVER"] = server.endpoint()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        try:
            completed = subprocess.run(
                [str(binary)],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=max(600, cap),
                check=False,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            returncode = int(completed.returncode)
            (work / "stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
            (work / "stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            returncode = None
            (work / "optbench.timeout").write_text("timeout\n", encoding="utf-8")
        points = _candidate_points(work, reactant.shape[0])
        ran = "OPTIM version" in stdout
        cap_hit = (work / "optbench.cap").is_file() or counter.calls >= cap
        crashed = (work / "optbench.crash").is_file() or (work / "optbench.timeout").is_file() or (
            returncode not in (0, None) and not ran and not points
        )
        if cap_hit:
            judged = {
                "point": points[0] if points else None,
                "energy": float("nan"),
                "force_norm": None,
                "hessian_index": None,
                "reasons": ["cap"],
            }
        elif crashed and not points:
            judged = {
                "point": None,
                "energy": float("nan"),
                "force_norm": None,
                "hessian_index": None,
                "reasons": ["crash"],
            }
        else:
            judged = _judge(
                points,
                kind=kind,
                free=free,
                box=box,
                counter=counter,
                force_norm=force_norm,
                cap=cap,
                project_rigid=project_rigid,
            )
            if counter.calls >= cap and "cap" not in judged["reasons"]:
                judged["reasons"] = ["cap", *judged["reasons"]]
        reasons = list(judged["reasons"])
        if crashed and "crash" not in reasons and not points:
            reasons.insert(0, "crash")
        if counter.calls >= cap and "cap" not in reasons:
            reasons.append("cap")
        failure = classify_failure(reasons)
        if failure == "cap":
            reasons = ["cap"] if "crash" not in reasons else ["crash", "cap"]
            failure = classify_failure(reasons)
        detail = None
        if failure == "crash":
            tail = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), "")
            detail = tail[:240] or "optim"
        return _record(
            name=name,
            calls=counter.calls,
            verify_calls=counter.verify_calls,
            success=failure is None,
            failure=failure,
            reasons=reasons,
            energy=judged["energy"],
            force=judged["force_norm"],
            hessian_index=judged["hessian_index"],
            crash=detail,
        )
    finally:
        server.stop()
        shutil.rmtree(work, ignore_errors=True)


_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    key = (str(Path(root).resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        from .pes import load_pairs

        loaded = load_pairs(geometry_archive(root, problem), settings["geometry"], settings["pairs"])
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def _job(pair, settings: dict) -> dict:
    return {
        "name": pair.name,
        "reactant": pair.reactant,
        "product": pair.product,
        "free": pair.free,
        "box": pair.box,
        "settings": settings,
    }


def search_named(
    root: Path,
    problem_name: str,
    template_name: str,
    start: str,
    seed: str = "0",
) -> dict:
    """One archived pair. ``seed`` is recorded. The connect run does not draw random numbers."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(pairs[start], settings))
    record["seed"] = str(seed)
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict, problem: dict) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness socket calculate, one POTENTIAL entry, no cache",
        "# force_calls count raw potential evaluations and are not divided by the image count",
        "# the mean is round-half-even over successes; cap, force, or Hessian index other than 1 is a failure",
    ]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        images = int(settings["n_interior"])
        if images > 0:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
            lines.append("call_unit total")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row['hessian_index'] == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append(f"problem {problem['id']}")
    lines.append("counter harness")
    lines.append(f"images {settings['n_interior']}")
    return "\n".join(lines) + "\n"


def _legacy(records: list[dict]) -> dict:
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary = {
        "successes": len(successes),
        "nfailed": len(records) - len(successes),
        "n": len(records),
        "transition_states": sum(1 for row in records if row["hessian_index"] == 1),
    }
    if successes:
        summary["force_calls"] = int(round(statistics.fmean(successes)))
        summary["force_calls_min"] = min(successes)
        summary["force_calls_max"] = max(successes)
        summary["force_calls_median"] = statistics.median(successes)
    return summary


def write_outputs(output: Path, records: list[dict], template: dict, settings: dict, problem: dict) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(records, template, settings, problem), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    parameters = dict(settings)
    parameters["counter"] = "harness"
    parameters["cache"] = False
    parameters["seed"] = "0"
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score_v2(
            [{"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]} for row in records],
            settings["cap"],
        ),
        "legacy": _legacy(records),
    }
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    legacy = document["legacy"]
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": legacy.get("force_calls"),
        "nfailed": legacy["nfailed"],
        "starts": legacy["n"],
        "successes": legacy["successes"],
        "output": str(output),
    }


def publish_sidecars(root: Path, directory: Path, output_dir: Path) -> list[dict]:
    """Reduce a complete sidecar tree into one dat per problem."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    paths = sorted(Path(directory).rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no asv sidecar records in {directory}")
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault((row["problem"], row["template"]), []).append(row)
    summaries = []
    commit = _registry(root)["commit"]
    for problem_name, template_name in sorted(grouped):
        records = grouped[(problem_name, template_name)]
        problem = resolve_problem(root, problem_name)
        template = load_template(root, template_name)
        settings = _settings(template, problem, commit)
        order = list(cached_pairs(root, problem, settings))
        found = {row["start"] for row in records}
        if found != set(order) or len(records) != len(order):
            raise ValueError(
                f"{template_name} on {problem_name} has {len(records)} records, the problem has {len(order)} pairs"
            )
        rank = {start: index for index, start in enumerate(order)}
        records.sort(key=lambda row: rank[row["start"]])
        slug = f"{problem['id'].replace('/', '-')}-{template['id'].replace('/', '-')}"
        summaries.append(write_outputs(output_dir / f"{slug}.dat", records, template, settings, problem))
    return summaries


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    """Run every pair in one problem and write the dat, starts, and v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_job(pair, settings) for pair in pairs]
    _threads_one()
    if workers <= 1:
        records = [_execute(job) for job in jobs]
    else:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(_execute, jobs))
    order = {job["name"]: index for index, job in enumerate(jobs)}
    records.sort(key=lambda row: order[row["start"]])
    for record in records:
        record["seed"] = "0"
        record["problem"] = problem["id"]
        record["template"] = template["id"]
        record["code"] = template["code"]
        record["code_version"] = template["code_version"]
        record["commit"] = template["commit"]
        record["algorithm"] = template["algorithm"]
        record["tuning"] = template["tuning"]
    return write_outputs(output, records, template, settings, problem)
