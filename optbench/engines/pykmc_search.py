"""pyKMC event search through ``partn_search``.

The pin is hugomoison/pyKMC, not SourceForge lkmc. One search is
``pykmc.enginemanager.lmpi.lammps_operations.partn_search`` in the
pyKMC virtualenv. LAMMPS holds the free atoms. ``pair_style zero``
contributes nothing. ``fix external pf/callback`` evaluates the spec
potential and ``HarnessCounter`` counts those calls.

``partn_search`` leaves ``converge_property`` unset, so pARTn would
test the maximum force component. The worker sets ``norm`` and
``forc_thr`` to the problem force stop when that parameter is written.
``ninit`` is 0 on a transition-state guess. A search that starts at a
minimum keeps the pyKMC push defaults.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from optbench.pykmc_worker import (
    ARTN_COMMIT,
    IRA_COMMIT,
    LAMMPS_COMMIT,
    PYKMC_COMMIT,
    library_path,
    python_path,
    stack_status,
)

from .common import judge_saddle, pair_seed, registry_pin, require_pin, write_campaign
from .geometries import load_starts

ENGINE = "pykmc"
CODE = "pykmc"
REPOSITORY = "https://github.com/hugomoison/pyKMC"
RELEASE = "0.0.1"
_SEED_BASE = 20261015
# On these slabs a one-atom hop is about 2.7 A, and the mean over every
# free atom stays under 0.5. The protruding atoms sit more than 1.5 A
# above the next layer: the Pt heptamer, or the Al adatom.
_TOP_WINDOW = 1.5
_STATUS: str | None = None
_STATUS_READY = False


def available() -> str | None:
    global _STATUS, _STATUS_READY
    if not _STATUS_READY:
        _STATUS = stack_status()
        _STATUS_READY = True
    return _STATUS


def _section(template: dict) -> dict:
    section = dict(template.get("pykmc") or {})
    section.setdefault("ninit", 2)
    section.setdefault("push_mode", "rad")
    section.setdefault("push_step_size", 0.4)
    section.setdefault("push_dist_thr", 1.0)
    section.setdefault("eigen_step_size", 0.2)
    section.setdefault("lanczos_max_size", 20)
    section.setdefault("lanczos_min_size", 10)
    section.setdefault("forc_thr", 0.001)
    section.setdefault("delr_thr", 1.0e9)
    section.setdefault("converge_property", "norm")
    section.setdefault("evalf_max", 10000)
    section.setdefault("rcut", 6.5)
    section.setdefault("rnei", 3.0)
    section.setdefault("timeout", 1800)
    return section


def _env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = library_path() + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYKMC_STACK"] = str(Path(os.environ.get("PYKMC_STACK", "/opt/pykmc-stack")))
    return env


def _invoke(root: Path, job: dict) -> dict:
    python = python_path()
    with tempfile.TemporaryDirectory(prefix="pykmc-ipc-") as folder:
        spec = Path(folder) / "job.json"
        dest = Path(folder) / "result.json"
        spec.write_text(json.dumps(job), encoding="utf-8")
        try:
            proc = subprocess.run(
                [str(python), "-m", "optbench.pykmc_worker", str(spec), str(dest)],
                cwd=folder,
                env=_env(root),
                capture_output=True,
                text=True,
                timeout=int(job.get("timeout") or 1800),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "calls": 0,
                "capped": False,
                "crashed": True,
                "diverged": False,
                "ok": False,
                "message": f"timeout: {exc}",
                "positions": None,
                "artn_energy": None,
            }
        if dest.is_file():
            payload = json.loads(dest.read_text(encoding="utf-8"))
        else:
            tail = (proc.stderr or proc.stdout or "worker wrote no result").strip().splitlines()
            payload = {
                "calls": 0,
                "capped": False,
                "crashed": True,
                "diverged": False,
                "ok": False,
                "message": tail[-1] if tail else "worker wrote no result",
                "positions": None,
                "artn_energy": None,
            }
        if proc.returncode != 0 and not payload.get("message"):
            payload["crashed"] = True
            payload["message"] = (proc.stderr or "worker failed").strip().splitlines()[-1]
        return payload


def _surface(problem: dict, template: dict) -> dict:
    spec = problem.get("potential") or {}
    kernel = str(template.get("kernel") or spec.get("kind") or "")
    if kernel in {"morse", "morse_pt"} or spec.get("kind") == "morse":
        pbc = spec.get("pbc") or [True, True, False]
        return {
            "kind": "morse",
            "element": "Pt",
            "pbc": [bool(pbc[0]), bool(pbc[1]), bool(pbc[2])],
            "depth": float(spec.get("D", 0.7102)),
            "alpha": float(spec.get("alpha", 1.6047)),
            "r0": float(spec.get("r0", 2.897)),
            "cutoff": float(spec.get("cutoff", 9.5)),
        }
    if kernel == "eam":
        return {
            "kind": "eam",
            "element": "Al",
            "pbc": [True, True, True],
            "depth": 0.0,
            "alpha": 0.0,
            "r0": 0.0,
            "cutoff": 6.5,
        }
    raise ValueError(f"{template['id']} has no Morse or EAM kernel")


def _evaluator(surface: dict, box: np.ndarray):
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3].copy()
    if surface["kind"] == "morse":
        from optbench.surfaces import morse_smooth_linear

        pbc = tuple(surface["pbc"])

        def evaluate(coords, _box):
            return morse_smooth_linear(
                np.asarray(coords, dtype=np.float64).reshape(-1, 3),
                depth=float(surface["depth"]),
                alpha=float(surface["alpha"]),
                r0=float(surface["r0"]),
                cutoff=float(surface["cutoff"]),
                cell=lengths,
                pbc=pbc,
            )

        return evaluate
    from optbench.eon_harness import eam_al

    cell = np.zeros(9, dtype=np.float64)
    cell[0], cell[4], cell[8] = (float(value) for value in lengths)

    def evaluate(coords, _box):
        energy, forces = eam_al(np.asarray(coords, dtype=np.float64).reshape(-1, 3), None, cell)
        gradient = -np.asarray(forces, dtype=np.float64).reshape(-1, 3)
        return float(energy), gradient

    return evaluate


class _Calls:
    def __init__(self, calls: int):
        self.calls = int(calls)
        self.verify_calls = 0

    def verification(self):
        from contextlib import nullcontext

        return nullcontext()


def _stored_text(root: Path, archive_path: str) -> str:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return (root / row["stored"]).read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(archive_path)


def _read_geometry(root: Path, archive_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from optbench.pes import read_con

    positions, free, box = read_con(_stored_text(root, archive_path))
    if box is None:
        raise ValueError(f"{archive_path} has no orthorhombic box")
    return positions, free, np.asarray(box, dtype=np.float64).reshape(-1)[:3]


def _cap(problem: dict, template: dict) -> int:
    stated = (problem.get("convergence") or {}).get("max_force_calls")
    if stated not in (None, ""):
        return int(stated)
    if template.get("failure_cap") not in (None, ""):
        return int(template["failure_cap"])
    raise ValueError(f"{problem['id']} has no force-call cap")


def _force(problem: dict, template: dict) -> float:
    convergence = problem.get("convergence") or {}
    stated = convergence.get("force_norm", convergence.get("gradient_norm"))
    if stated not in (None, ""):
        return float(stated)
    return float(_section(template)["forc_thr"])


def _base_job(name: str, positions, free, box, surface: dict, section: dict, *, cap: int, central: int, zseed: int) -> dict:
    return {
        "name": name,
        "positions": np.asarray(positions, dtype=np.float64).reshape(-1).tolist(),
        "free": np.asarray(free, dtype=bool).reshape(-1).tolist(),
        "box": np.asarray(box, dtype=np.float64).reshape(-1)[:3].tolist(),
        "pbc": list(surface["pbc"]),
        "kind": surface["kind"],
        "element": surface["element"],
        "depth": surface["depth"],
        "alpha": surface["alpha"],
        "r0": surface["r0"],
        "cutoff": surface["cutoff"],
        "cap": int(cap),
        "central": int(central),
        "zseed": int(zseed),
        "ninit": int(section["ninit"]),
        "push_mode": str(section["push_mode"]),
        "push_step_size": float(section["push_step_size"]),
        "push_dist_thr": float(section["push_dist_thr"]),
        "eigen_step_size": float(section["eigen_step_size"]),
        "lanczos_max_size": int(section["lanczos_max_size"]),
        "lanczos_min_size": int(section["lanczos_min_size"]),
        "forc_thr": float(section["forc_thr"]),
        "delr_thr": float(section["delr_thr"]),
        "converge_property": str(section["converge_property"]),
        "evalf_max": int(min(int(section["evalf_max"]), int(cap))),
        "rcut": float(section["rcut"]),
        "rnei": float(section["rnei"]),
        "timeout": int(section["timeout"]),
        "engine_id": 0,
    }


def _prefix(payload: dict) -> list[str]:
    if payload.get("capped"):
        return ["cap"]
    if payload.get("diverged"):
        return ["divergence"]
    if payload.get("crashed"):
        return ["crash"]
    return []


def _judge(name: str, payload: dict, *, cap: int, force_norm: float, kind: str, free, box, evaluator, calls: int) -> dict:
    point = None
    if payload.get("positions"):
        point = np.asarray(payload["positions"], dtype=np.float64).reshape(-1, 3)
    gradient = None
    energy = None
    if point is not None and np.all(np.isfinite(point)):
        energy, gradient = evaluator(point, box)
        energy = float(energy)
        gradient = np.asarray(gradient, dtype=np.float64)
    record = judge_saddle(
        name,
        point,
        energy,
        gradient,
        counter=_Calls(calls),
        cap=cap,
        force_norm=force_norm,
        kind=kind,
        free=np.asarray(free, dtype=bool),
        box=box,
        prefix=_prefix(payload),
        keep_mode=True,
        evaluator=evaluator,
    )
    if point is not None:
        record["positions"] = np.asarray(point, dtype=np.float64)
    if payload.get("message"):
        record["error"] = str(payload["message"])
    record["artn_ok"] = bool(payload.get("ok"))
    return record


def _minimize_free(evaluator, positions, free, box, force_tol: float, budget: int):
    from optbench.rgsaddle_minmode import HarnessSurface, _lbfgs_minimize

    free = np.asarray(free, dtype=bool)

    def call(point):
        full = np.array(positions, dtype=np.float64, copy=True)
        full[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
        energy, grad = evaluator(full, box)
        return float(energy), np.asarray(grad, dtype=np.float64)[free].reshape(-1)

    surface = HarnessSurface(call)
    point, ok = _lbfgs_minimize(surface, positions[free].reshape(-1).copy(), force_tol, budget)
    out = np.array(positions, dtype=np.float64, copy=True)
    out[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    energy, _gradient = evaluator(out, box)
    return out, bool(ok), int(surface.calls), float(energy)


def _rmsd(moved: np.ndarray, reference: np.ndarray, lengths=None, pbc=None) -> float:
    delta = np.asarray(moved, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if lengths is not None and pbc is not None:
        cell = np.asarray(lengths, dtype=np.float64).reshape(-1)[:3]
        for axis, periodic in enumerate(pbc):
            length = float(cell[axis])
            if periodic and length > 0.0:
                delta[..., axis] -= np.round(delta[..., axis] / length) * length
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _event_mask(positions: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Free atoms in the top window. The substrate stays in the frame."""
    free = np.asarray(free, dtype=bool)
    z = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[:, 2]
    top = float(z[free].max())
    mask = free & (z > top - _TOP_WINDOW)
    if not np.any(mask):
        return free
    return mask


def _placed(saddle: np.ndarray, free: np.ndarray, free_flat: np.ndarray) -> np.ndarray:
    out = np.array(saddle, dtype=np.float64, copy=True)
    out[free] = np.asarray(free_flat, dtype=np.float64).reshape(-1, 3)
    return out


def _link(record: dict, *, evaluator, free, box, minimum, min_energy, force_norm: float, barrier_max: float, connect_rmsd: float, found: list[np.ndarray], pbc) -> dict:
    from optbench.failures import classify_failure

    saddle = None if record.get("positions") is None else np.asarray(record["positions"], dtype=np.float64).reshape(-1, 3)
    record["reasons"] = list(record.get("reasons") or [])
    mode = record.get("lowest_mode")
    if not (record["success"] and saddle is not None and mode):
        if record.get("success"):
            record["success"] = False
            record["reasons"].append("divergence")
            record["failure"] = classify_failure(record["reasons"])
        record.pop("positions", None)
        record.pop("lowest_mode", None)
        return record
    axis = np.asarray(record["lowest_mode"], dtype=np.float64)
    scale = float(np.linalg.norm(axis))
    if scale < 1e-14:
        record["success"] = False
        record["reasons"].append("divergence")
        record["failure"] = classify_failure(record["reasons"])
        record.pop("positions", None)
        record.pop("lowest_mode", None)
        return record
    axis = axis / scale
    saddle_free = saddle[free].reshape(-1)
    plus, _plus_ok, plus_calls, _plus_energy = _minimize_free(
        evaluator, _placed(saddle, free, saddle_free + 0.1 * axis), free, box, force_norm, 2000
    )
    minus, _minus_ok, minus_calls, _minus_energy = _minimize_free(
        evaluator, _placed(saddle, free, saddle_free - 0.1 * axis), free, box, force_norm, 2000
    )
    mask = _event_mask(minimum, free)
    image = {"lengths": box, "pbc": pbc}
    plus_rmsd = _rmsd(plus[mask], minimum[mask], **image)
    minus_rmsd = _rmsd(minus[mask], minimum[mask], **image)
    links = (plus_rmsd <= connect_rmsd) ^ (minus_rmsd <= connect_rmsd)
    novel = all(_rmsd(saddle[mask], old, **image) > connect_rmsd for old in found)
    barrier = float(record["energy"] - min_energy)
    record["barrier"] = barrier
    record["rmsd_plus"] = plus_rmsd
    record["rmsd_minus"] = minus_rmsd
    record["rmsd_plus_free"] = _rmsd(plus[free], minimum[free], **image)
    record["rmsd_minus_free"] = _rmsd(minus[free], minimum[free], **image)
    record["rmsd_atoms"] = int(np.count_nonzero(mask))
    record["endpoint_calls"] = int(plus_calls + minus_calls)
    record["force_calls"] = int(record["force_calls"]) + int(plus_calls + minus_calls)
    record["positions"] = saddle.reshape(-1).tolist()
    if novel:
        found.append(saddle[mask].copy())
    if links and novel and 0.0 < barrier < barrier_max:
        record["success"] = True
        record["failure"] = None
        record["reasons"] = []
    else:
        if not links:
            record["reasons"].append("not_connected")
        if not novel:
            record["reasons"].append("duplicate")
        if barrier <= 0.0 or barrier >= barrier_max:
            record["reasons"].append("barrier")
        record["success"] = False
        record["failure"] = classify_failure(record["reasons"])
    record.pop("lowest_mode", None)
    return record


def _parameters(surface: dict, section: dict, cap: int, force_norm: float) -> dict:
    return {
        "kernel": surface["kind"],
        "element": surface["element"],
        "pbc": surface["pbc"],
        "cap": cap,
        "force_norm": force_norm,
        "counter": "harness",
        "pykmc": PYKMC_COMMIT,
        "lammps": LAMMPS_COMMIT,
        "artn": ARTN_COMMIT,
        "ira": IRA_COMMIT,
        "zseed_base": _SEED_BASE,
        **section,
    }


def _one_point(root: Path, job: dict, *, cap: int, force_norm: float, kind: str, free, box, evaluator) -> dict:
    payload = _invoke(root, job)
    calls = int(payload.get("calls") or 0)
    if payload.get("capped"):
        calls = max(calls, cap)
    record = _judge(
        job["name"],
        payload,
        cap=cap,
        force_norm=force_norm,
        kind=kind,
        free=free,
        box=box,
        evaluator=evaluator,
        calls=calls,
    )
    record["zseed"] = int(job["zseed"])
    record["central"] = int(job["central"])
    return record


def run_points(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    surface = _surface(problem, template)
    section = _section(template)
    cap = _cap(problem, template)
    force_norm = _force(problem, template)
    section["forc_thr"] = force_norm
    starts = load_starts(root, problem, output.parent / "work" / template["id"].replace("/", "-"))
    jobs = []
    evaluators = []
    metas = []
    for start in starts:
        positions = np.asarray(start["positions"], dtype=np.float64)
        free = np.asarray([not bool(flag) for flag in start["fixed"]], dtype=bool)
        box = np.asarray(start["box"], dtype=np.float64)
        central = int(np.flatnonzero(free)[0])
        zseed = pair_seed(f"{template['id']}:{start['name']}", _SEED_BASE)
        jobs.append(
            _base_job(
                start["name"],
                positions,
                free,
                box,
                surface,
                section,
                cap=cap,
                central=central,
                zseed=zseed,
            )
        )
        evaluators.append(_evaluator(surface, box))
        metas.append((free, box))
    ckpt = output.parent / "work" / template["id"].replace("/", "-") / "points.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if ckpt.is_file():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["start"])] = row
    write_lock = threading.Lock()

    def finish(index: int) -> dict:
        job = jobs[index]
        if job["name"] in done:
            return done[job["name"]]
        free, box = metas[index]
        print(f"{template['id']} {job['name']}", flush=True)
        record = _one_point(
            root,
            job,
            cap=cap,
            force_norm=force_norm,
            kind="morse_pt" if surface["kind"] == "morse" else surface["kind"],
            free=free,
            box=box,
            evaluator=evaluators[index],
        )
        record.pop("positions", None)
        record.pop("lowest_mode", None)
        with write_lock:
            with ckpt.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_json_default) + "\n")
        print(
            f"{template['id']} {job['name']} calls {record['force_calls']} "
            f"success {int(record['success'])} failure {record['failure']}",
            flush=True,
        )
        return record

    if workers <= 1 or len(jobs) <= 1:
        records = [finish(index) for index in range(len(jobs))]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(finish, range(len(jobs))))
    comment = (
        "# counter harness. One pyKMC partn_search from each transition-state guess.\n"
        "# ninit is 0. The free atoms are the LAMMPS system. Fixed atoms stay in the callback.\n"
        "# pair_style zero. Forces are the smooth-linear Morse gradient through fix external.\n"
        "# A success is under the cap, at the force stop, and Hessian index 1."
    )
    return write_campaign(
        output,
        records,
        template,
        problem,
        {"cap": cap, "force_norm": force_norm, "potential": surface["kind"]},
        _parameters(surface, section, cap, force_norm),
        images=None,
        comment=comment,
    )


def run_catalog(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    del workers
    from optbench.v2 import write_starts

    surface = _surface(problem, template)
    section = _section(template)
    force_norm = _force(problem, template)
    section["forc_thr"] = force_norm
    search_cap = int((template.get("search") or {}).get("search_budget", section["evalf_max"]))
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    search = dict(template.get("search") or {})
    repetitions = int(search.get("repetitions", 1))
    magnitude = float(search.get("displace_magnitude", 0.1))
    connect_rmsd = float(search.get("rmsd_connect", 0.5))
    max_searches = int(search.get("max_searches", 20000))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    failure_cap = int(template.get("failure_cap") or (search_cap + 4000))
    starts = load_starts(root, problem, output.parent / "work" / template["id"].replace("/", "-"))
    base = starts[0]
    positions = np.asarray(base["positions"], dtype=np.float64)
    free = np.asarray([not bool(flag) for flag in base["fixed"]], dtype=bool)
    box = np.asarray(base["box"], dtype=np.float64)
    evaluator = _evaluator(surface, box)
    kind = "morse_pt"
    snap_path = output.parent / "work" / template["id"].replace("/", "-") / "catalog.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    comments = []
    records = []
    failed_reps = 0
    for rep in range(1, repetitions + 1):
        state = None
        if snap_path.is_file():
            state = json.loads(snap_path.read_text(encoding="utf-8"))
        if state and int(state.get("rep") or 0) == rep and state.get("minimum") is not None:
            minimum = np.asarray(state["minimum"], dtype=np.float64).reshape(-1, 3)
            min_energy = float(state["min_energy"])
            minimize_calls = int(state["minimize_calls"])
            min_ok = bool(state["min_ok"])
            connected = int(state["connected"])
            searches = int(state["searches"])
            search_calls = int(state["search_calls"])
            found = [np.asarray(item, dtype=np.float64).reshape(-1, 3) for item in state.get("found") or []]
            records = list(state.get("records") or [])
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng"]
            last_success = 0
            for index, saved in enumerate(records, start=1):
                if saved.get("success"):
                    last_success = index
            stall = searches - last_success
        else:
            minimum, min_ok, minimize_calls, min_energy = _minimize_free(
                evaluator, positions, free, box, force_norm, 5000
            )
            rng = np.random.default_rng(_SEED_BASE + rep)
            found = []
            connected = 0
            searches = 0
            search_calls = 0
            records = []
            stall = 0
        # A flat stretch of 160 searches adds no saddle under 1.5 eV.
        # The per-search score of that finished sample is the row.
        while connected < goal and searches < max_searches and not (searches >= 200 and stall >= 160):
            searches += 1
            atom = int(listed[int(rng.integers(0, len(listed)))])
            if not bool(free[atom]):
                raise RuntimeError(f"displace atom {atom} is not free")
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            kicked = np.array(minimum, copy=True)
            kicked[atom] = kicked[atom] + magnitude * direction
            zseed = pair_seed(f"{template['id']}:rep{rep}:search{searches}", _SEED_BASE)
            climb_cap = min(search_cap, failure_cap)
            job = _base_job(
                f"rep{rep}-search{searches}",
                kicked,
                free,
                box,
                surface,
                section,
                cap=climb_cap,
                central=atom,
                zseed=zseed,
            )
            print(
                f"{template['id']} rep {rep} search {searches} connected {connected}",
                flush=True,
            )
            record = _one_point(
                root,
                job,
                cap=climb_cap,
                force_norm=force_norm,
                kind=kind,
                free=free,
                box=box,
                evaluator=evaluator,
            )
            before = int(record["force_calls"])
            record = _link(
                record,
                evaluator=evaluator,
                free=free,
                box=box,
                minimum=minimum,
                min_energy=min_energy,
                force_norm=force_norm,
                barrier_max=barrier_max,
                connect_rmsd=connect_rmsd,
                found=found,
                pbc=surface["pbc"],
            )
            if record["success"]:
                connected += 1
                stall = 0
            else:
                stall += 1
            if int(record["force_calls"]) >= failure_cap and record["success"]:
                record["success"] = False
                record["failure"] = "cap"
                record["reasons"] = ["cap"]
                connected -= 1
            search_calls += int(record["force_calls"])
            record.pop("positions", None)
            records.append(record)
            print(
                f"{template['id']} search {searches} climb {before} total {record['force_calls']} "
                f"success {int(record['success'])} connected {connected} barrier {record.get('barrier')} "
                f"rmsd {record.get('rmsd_plus')} {record.get('rmsd_minus')}",
                flush=True,
            )
            snap_path.write_text(
                json.dumps(
                    {
                        "rep": rep,
                        "minimum": minimum.reshape(-1).tolist(),
                        "min_energy": min_energy,
                        "minimize_calls": minimize_calls,
                        "min_ok": min_ok,
                        "connected": connected,
                        "searches": searches,
                        "search_calls": search_calls,
                        "found": [item.reshape(-1).tolist() for item in found],
                        "records": records,
                        "rng": rng.bit_generator.state,
                    },
                    default=_json_default,
                ),
                encoding="utf-8",
            )
        total_calls = int(minimize_calls + search_calls)
        if connected < goal:
            failed_reps += 1
        fraction = connected / searches if searches else 0.0
        comments.append(
            f"# rep {rep} force_calls {total_calls} unique_saddles {len(found)} "
            f"fraction_connected {fraction} total_searches {searches} "
            f"connected {connected} minimize_calls {minimize_calls} "
            f"minimum_converged {int(min_ok)} seed {_SEED_BASE + rep}"
        )
        last_at = 0
        last_barrier = None
        for index, saved in enumerate(records, start=1):
            if saved.get("success"):
                last_at = index
                last_barrier = saved.get("barrier")
        if last_at and searches > last_at:
            barrier_text = f"{float(last_barrier):.3f}" if last_barrier is not None else "unknown"
            comments.append(
                "# fraction_connected is connected/total_searches. "
                f"The last connected saddle is search {last_at}, barrier {barrier_text} eV. "
                f"The next {searches - last_at} searches added none. "
                "The runner exits after 160 searches with no new connected saddle "
                f"once 200 searches have run. The goal is {goal}."
            )
        if connected < goal:
            output.parent.mkdir(parents=True, exist_ok=True)
            text = "\n".join(comments) + "\npartial true\n"
            text += (
                f"code {template['code']}\ncommit {template['commit']}\n"
                f"template {template['id']}\nproblem {problem['id']}\n"
                f"counter harness\nn {len(records)}\n"
            )
            output.write_text(text, encoding="utf-8")
            write_starts(output.with_suffix(".starts.jsonl"), records)
    totals = [int(line.split()[line.split().index("force_calls") + 1]) for line in comments]
    text = "\n".join(comments) + "\n"
    if totals:
        text += f"force_calls {statistics.fmean(totals)}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
    text += (
        f"code {template.get('code', CODE)}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"cap {failure_cap}\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"n {len(records)}\n"
        f"goal {goal}\n"
        f"connected {connected}\n"
    )
    # The search sample is finished. A mid-run file still says partial.
    # Reaching the goal is recorded by connected and goal, and the
    # per-search rows are the score either way.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    from optbench.score import score_v2
    from optbench.v2 import collect_provenance

    score = score_v2(
        [
            {"force_calls": row["force_calls"], "success": row["success"], "failure": row.get("failure")}
            for row in records
        ],
        failure_cap,
    )
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template.get("code", CODE),
            code_version=template.get("code_version", ""),
            commit=template.get("commit", ""),
            parameters=_parameters(surface, section, failure_cap, force_norm),
        ),
        "score": score,
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "force_calls": statistics.fmean(totals) if totals else None,
        "nfailed": failed_reps,
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
        "connected": connected,
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def _centers(positions: np.ndarray, free: np.ndarray, count: int) -> list[int]:
    order = sorted(np.flatnonzero(free), key=lambda index: (-float(positions[index, 2]), int(index)))
    return [int(index) for index in order[:count]]


def run_event(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    del workers
    surface = _surface(problem, template)
    section = _section(template)
    cap = _cap(problem, template)
    force_norm = _force(problem, template)
    section["forc_thr"] = force_norm
    geometry = str(template.get("geometry") or "")
    positions, free, box = _read_geometry(root, geometry)
    evaluator = _evaluator(surface, box)
    kind = "morse_pt" if surface["kind"] == "morse" else surface["kind"]
    minimum, min_ok, minimize_calls, min_energy = _minimize_free(
        evaluator, positions, free, box, force_norm, min(5000, cap)
    )
    total = int(minimize_calls)
    barrier_max = float(section.get("barrier_max", 1.5))
    connect_rmsd = float(section.get("rmsd_connect", 0.5))
    nsearch = int(section.get("nsearch", 4))
    center_count = int(section.get("centers", 6))
    per_search = int(section["evalf_max"])
    found: list[np.ndarray] = []
    chosen = None
    attempts = 0
    for center in _centers(minimum, free, center_count):
        for attempt in range(nsearch):
            if total >= cap or chosen is not None:
                break
            attempts += 1
            remaining = cap - total
            climb_cap = min(per_search, remaining)
            zseed = pair_seed(f"{template['id']}:{center}:{attempt}", _SEED_BASE)
            job = _base_job(
                f"center{center}-try{attempt}",
                minimum,
                free,
                box,
                surface,
                section,
                cap=climb_cap,
                central=center,
                zseed=zseed,
            )
            print(
                f"{template['id']} attempt {attempts} center {center} spent {total}",
                flush=True,
            )
            record = _one_point(
                root,
                job,
                cap=climb_cap,
                force_norm=force_norm,
                kind=kind,
                free=free,
                box=box,
                evaluator=evaluator,
            )
            record = _link(
                record,
                evaluator=evaluator,
                free=free,
                box=box,
                minimum=minimum,
                min_energy=min_energy,
                force_norm=force_norm,
                barrier_max=barrier_max,
                connect_rmsd=connect_rmsd,
                found=found,
                pbc=surface["pbc"],
            )
            spent = int(record["force_calls"])
            total += spent
            print(
                f"{template['id']} attempt {attempts} calls {spent} total {total} "
                f"success {int(record['success'])} barrier {record.get('barrier')} "
                f"rmsd {record.get('rmsd_plus')} {record.get('rmsd_minus')}",
                flush=True,
            )
            if record["success"] and total < cap:
                chosen = record
                break
        if chosen is not None or total >= cap:
            break
    if chosen is None:
        payload = {
            "positions": None,
            "capped": total >= cap,
            "crashed": False,
            "diverged": total < cap,
            "ok": False,
            "message": f"{attempts} searches, no connected saddle under {barrier_max} eV",
        }
        record = _judge(
            "reactant",
            payload,
            cap=cap,
            force_norm=force_norm,
            kind=kind,
            free=free,
            box=box,
            evaluator=evaluator,
            calls=max(total, cap if total >= cap else total),
        )
    else:
        record = dict(chosen)
        record["start"] = "reactant"
        record["force_calls"] = int(total)
        if total >= cap:
            record["success"] = False
            record["failure"] = "cap"
            record["reasons"] = ["cap"]
    record.pop("positions", None)
    record.pop("lowest_mode", None)
    record["attempts"] = attempts
    record["minimize_calls"] = minimize_calls
    record["minimum_converged"] = bool(min_ok)
    comment = (
        "# counter harness. pyKMC partn_search from the reactant, highest free atoms first.\n"
        "# The call total is the reactant minimization, the searches, and the two endpoint minimizations.\n"
        "# A success is one index-1 saddle under the cap, force at the stop, barrier in (0, 1.5) eV,\n"
        "# and exactly one endpoint within Cartesian RMSD 0.5 of the minimized reactant\n"
        "# on the protruding atoms (within 1.5 A of the highest free atom).\n"
        "# The 2014 rate and barrier columns are a different observable."
    )
    return write_campaign(
        output,
        [record],
        template,
        problem,
        {"cap": cap, "force_norm": force_norm, "potential": surface["kind"]},
        _parameters(surface, section, cap, force_norm),
        images=None,
        comment=comment,
    )


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    if problem["id"] == "tsopt/pt-island":
        return run_points(root, problem, template, output=output, workers=workers)
    if problem["id"] == "saddle-search/pt-island":
        return run_catalog(root, problem, template, output=output, workers=workers)
    if problem["id"] in {"hTST/Pt", "hTST/Al", "accelerated-md/aladatom", "accelerated-md/standard_aladatom"}:
        return run_event(root, problem, template, output=output, workers=workers)
    raise ValueError(f"{template_name} does not score {problem['id']}")
