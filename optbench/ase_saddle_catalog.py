"""Sella and ASE dimer saddle catalogs on Morse::force.

The geometry and the initial mode come from ``draw_search_start``.
Sella 2.6.0 runs at its saddle defaults, order 1, with ``v0`` equal
to the unit kick. The dimer row uses ``DimerControl`` defaults and
that same kick as the eigenmode.
The link is r2205 and novelty is ``find_repeat``. The climb stops at
1000 force calls. Endpoint minimizations are 2000 each. The PAR-2 cap
is 5064. The initial minimization is outside that count.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from .connection import ProcessCatalog, connected_endpoints, link_tolerance, per_atom_max
from .counter import HarnessCounter, attach_ase_counter
from .eon_displace import draw_search_start, problem_pbc
from .endpoints import EndpointArchive, align_resume, stamp_endpoint, stored_cell
from .failures import classify_failure
from .rgsaddle_minmode import (
    HarnessSurface,
    _bind,
    _cartesian_rmsd,
    _finite_point,
    _hessian_spectrum,
    _json_default,
    _lbfgs_minimize,
    _stored,
    _unit,
)
from .saddle_publish import write_saddle_campaign
from .stationary import negative_index
from .surfaces import load_con, potential_from_problem

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_CLIMB_BUDGET = 1000
_ENDPOINT_BUDGET = 2000
_OVERSHOOT = 64
_SEARCH_CAP = _CLIMB_BUDGET + _OVERSHOOT + 2 * _ENDPOINT_BUDGET


def _checkpoint_paths(output: Path) -> tuple[Path, Path]:
    return output.with_name(output.stem + ".header.json"), output.with_name(output.stem + ".journal.jsonl")


def _load_journal(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _append_journal(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
        handle.flush()


def _ase_cell(cell) -> np.ndarray:
    array = np.asarray(cell, dtype=np.float64)
    if array.size == 3:
        return np.diag(array.reshape(3))
    return array.reshape(3, 3)


def _calculator(potential: dict, cell):
    from ase.calculators.calculator import Calculator

    raw = potential_from_problem({"potential": potential})
    box = np.asarray(cell, dtype=np.float64)

    class _MorseForce(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes or [])
            energy, gradient = raw(atoms.get_positions(), box)
            self.results["energy"] = float(energy)
            self.results["forces"] = -np.asarray(gradient, dtype=np.float64)

    return _MorseForce()


def _atoms(positions, fixed, cell, pbc):
    from ase import Atoms
    from ase.constraints import FixAtoms

    atoms = Atoms(numbers=np.full(len(positions), 78), positions=np.asarray(positions, dtype=np.float64), pbc=list(pbc))
    atoms.set_cell(_ase_cell(cell))
    held = np.flatnonzero(np.asarray(fixed, dtype=bool))
    if held.size:
        atoms.set_constraint(FixAtoms(indices=held.tolist()))
    return atoms


def _catalog_job(payload: dict) -> dict:
    """One climb plus, when the point is index 1, the two endpoint minimizations."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import logging
    import warnings

    warnings.filterwarnings("ignore", message="The system is likely too small")
    logging.getLogger("sella").setLevel(logging.ERROR)
    positions = np.asarray(payload["positions"], dtype=np.float64)
    fixed = np.asarray(payload["fixed"], dtype=bool)
    cell = np.asarray(payload["cell"], dtype=np.float64)
    free = np.flatnonzero(~fixed)
    pbc = payload["pbc"]
    force_tol = float(payload["force_tol"])
    climb_cap = int(payload["climb_cap"])
    minimum = np.asarray(payload["minimum"], dtype=np.float64)
    min_energy = float(payload["min_energy"])
    kick = np.asarray(payload["kick"], dtype=np.float64)
    counter = HarnessCounter(cache=False, cap=climb_cap)
    state = {"calls": 0, "verify_calls": 0, "verifying": False, "cap": climb_cap}
    atoms = _atoms(positions, fixed, cell, pbc)
    calculator = _calculator(payload["potential"], cell)
    attach_ase_counter(calculator, cap=climb_cap, state=state, counter=counter)
    atoms.calc = calculator
    task = payload["task"]
    error = None
    mode = None
    try:
        if task == "sella":
            from sella import Sella

            mode = _unit(kick)
            opt = Sella(
                atoms,
                logfile=None,
                order=1,
                v0=mode,
                max_cpu_threads=1,
                **dict(payload["sella_parameters"]),
            )
            for _step in opt.irun(fmax=0.0, steps=int(payload["maxiter"])):
                if counter.calls >= climb_cap:
                    raise RuntimeError("force-call cap")
                if float(np.linalg.norm(atoms.get_forces())) <= force_tol:
                    break
        elif task == "dimer":
            from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

            full = np.zeros_like(positions)
            full.reshape(-1, 3)[free] = kick.reshape(-1, 3)
            mode = _unit(full.reshape(-1)).reshape(-1, 3)
            control = DimerControl(
                logfile=os.devnull,
                mask=[not bool(flag) for flag in np.asarray(fixed, dtype=bool)],
            )
            mode_atoms = MinModeAtoms(atoms, control, eigenmodes=[mode])
            opt = MinModeTranslate(mode_atoms, logfile=None)
            for _step in opt.irun(fmax=0.0, steps=int(payload["maxiter"])):
                if counter.calls >= climb_cap:
                    raise RuntimeError("force-call cap")
                norm = float(np.linalg.norm(atoms.get_forces()))
                curvature = float(mode_atoms.get_curvature())
                if norm <= force_tol and curvature < 0.0:
                    break
            mode = np.asarray(mode_atoms.get_eigenmode(), dtype=np.float64).reshape(-1)
        else:
            raise ValueError(task)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        if "force-call cap" not in str(exc):
            error = error
        else:
            error = "force-call cap"
    climb_calls = int(counter.calls)
    try:
        point = atoms.get_positions()[free].reshape(-1).copy()
    except Exception:
        point = np.zeros(int(free.size) * 3, dtype=np.float64)
    record = {
        "name": payload["name"],
        "climb_calls": climb_calls,
        "endpoint_calls": 0,
        "force_calls": climb_calls,
        "verify_calls": 0,
        "hessian_index": None,
        "converged": False,
        "error": error,
        "judged": False,
        "linked": False,
        "barrier": None,
        "saddle_x": None,
        "endpoint_plus": None,
        "endpoint_minus": None,
        "reasons": [],
    }
    if error and error != "force-call cap":
        record["reasons"] = ["crash"]
        return record
    if not _finite_point(point):
        record["reasons"] = ["divergence"]
        return record
    try:
        energy = float(atoms.get_potential_energy())
        force_norm = float(np.linalg.norm(atoms.get_forces()))
    except Exception as exc:
        record["reasons"] = ["crash"] if error and error != "force-call cap" else ["force_norm"]
        record["error"] = record["error"] or f"{type(exc).__name__}: {exc}"[:500]
        return record
    record["force_norm"] = force_norm
    record["energy"] = float(energy)
    raw = potential_from_problem({"potential": payload["potential"]})
    surface = HarnessSurface(_bind(raw, positions, free, cell))
    negative = force_norm <= force_tol + 1e-10 and math.isfinite(force_norm)
    if not negative:
        record["reasons"] = ["cap", "force_norm"] if climb_calls >= climb_cap or error == "force-call cap" else ["force_norm"]
        return record
    raw_values, vectors = _hessian_spectrum(surface, point)
    index = negative_index(list(raw_values), 1e-6)
    record["hessian_index"] = int(index)
    record["lowest_eigenvalue"] = float(raw_values[0]) if raw_values else None
    record["verify_calls"] = int(surface.verify_calls)
    record["converged"] = True
    if index != 1:
        record["reasons"] = ["hessian_index"]
        return record
    axis = _unit(np.asarray(vectors[:, 0], dtype=np.float64))
    plus, _plus_ok = _lbfgs_minimize(surface, point + 0.1 * axis, force_tol, _ENDPOINT_BUDGET)
    minus, _minus_ok = _lbfgs_minimize(surface, point - 0.1 * axis, force_tol, _ENDPOINT_BUDGET)
    record["endpoint_calls"] = int(surface.calls)
    record["force_calls"] = climb_calls + int(surface.calls)
    plus_rows = plus.reshape(-1, 3)
    minus_rows = minus.reshape(-1, 3)
    reactant_rows = minimum.reshape(-1, 3)
    record["rmsd_plus"] = _cartesian_rmsd(plus_rows, reactant_rows)
    record["rmsd_minus"] = _cartesian_rmsd(minus_rows, reactant_rows)
    record["per_atom_max_plus"] = per_atom_max(plus_rows, reactant_rows, cell)
    record["per_atom_max_minus"] = per_atom_max(minus_rows, reactant_rows, cell)
    record["barrier"] = float(energy - min_energy)
    record["linked"] = connected_endpoints(
        reactant_rows, plus_rows, minus_rows, cell, float(payload["tolerance"])
    )
    record["judged"] = True
    record["saddle_x"] = point.reshape(-1).tolist()
    record["endpoint_plus"] = plus_rows.reshape(-1).tolist()
    record["endpoint_minus"] = minus_rows.reshape(-1).tolist()
    record["reasons"] = []
    return record


def _public_row(template: dict, problem_id: str, raw: dict, decision: dict) -> dict:
    reasons = list(decision["reasons"]) if raw["judged"] else list(raw["reasons"])
    success = bool(decision["success"]) if raw["judged"] else False
    if not raw["judged"] and not reasons:
        reasons = ["divergence"]
        success = False
    failure = None if success else classify_failure(reasons or ["divergence"])
    return {
        "schema": "optbench.start/v2",
        "start": raw["name"],
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "force_calls": int(raw["force_calls"]),
        "verify_calls": int(raw.get("verify_calls") or 0),
        "endpoint_calls": int(raw.get("endpoint_calls") or 0),
        "climb_calls": int(raw.get("climb_calls") or 0),
        "hessian_index": raw.get("hessian_index"),
        "barrier": raw.get("barrier"),
        "rmsd_plus": raw.get("rmsd_plus"),
        "rmsd_minus": raw.get("rmsd_minus"),
        "per_atom_max_plus": raw.get("per_atom_max_plus"),
        "per_atom_max_minus": raw.get("per_atom_max_minus"),
        "converged": bool(raw.get("converged")),
        "code": template.get("code", ""),
        "code_version": template.get("code_version", ""),
        "commit": template.get("commit", ""),
        "tuning": template.get("tuning", ""),
        "template": template["id"],
        "problem": problem_id,
        "saddle_x": raw.get("saddle_x"),
        "endpoint_plus": raw.get("endpoint_plus"),
        "endpoint_minus": raw.get("endpoint_minus"),
    }


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
) -> dict:
    """Run one Sella or ASE dimer catalog until 27 qualifying saddles."""
    del work
    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = load_con(_stored(root, con_path))
    search = template.get("search") or {}
    goal = int((problem.get("goal") or {}).get("connected_saddles", search.get("connected", 27)))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    repetitions = int(search.get("repetitions", 1))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    magnitude = float(search.get("displace_magnitude", 0.1))
    radius = float(search.get("displace_radius", 3.3))
    max_searches = int(search.get("max_searches", 20000))
    stop_at_goal = str(search.get("stop", "goal")) != "budget"
    pbc = problem_pbc(problem)
    tolerance = link_tolerance(problem)
    force_tol = float((problem.get("convergence") or {}).get("force_norm", 1e-3))
    task = str(template.get("task") or "")
    if task == "sella":
        parameters = dict((template.get("optimizer") or {}).get("parameters") or {})
        parameters.pop("order", None)
        maxiter = int((template.get("sella") or {}).get("maxiter", 1000))
    elif task == "dimer":
        parameters = {}
        maxiter = int((template.get("dimer") or {}).get("maxiter", _CLIMB_BUDGET))
    else:
        raise ValueError(f"{template['id']} task {task} is not a saddle catalog")
    free = np.flatnonzero(~fixed)
    archive = EndpointArchive(output.with_name(output.stem + ".endpoints.npz"), int(free.size))
    raw = potential_from_problem(problem)
    comments: list[str] = []
    all_records: list[dict] = []
    failed_reps = 0
    pool = None
    if int(workers) > 1:
        import multiprocessing

        pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=multiprocessing.get_context("spawn"))
    try:
        for rep in range(1, repetitions + 1):
            header_path, journal_path = _checkpoint_paths(output)
            header = json.loads(header_path.read_text(encoding="utf-8")) if header_path.is_file() else None
            same = bool(
                header
                and header.get("template") == template["id"]
                and int(header.get("rep", 0)) == rep
                and header.get("optimizer_parameters") == parameters
                and int(header.get("maxiter", -1)) == maxiter
            )
            saved = _load_journal(journal_path) if same else []
            if same:
                align_resume(archive, len(all_records) + len(saved))
                surface = HarnessSurface(_bind(raw, positions, free, cell))
                minimum = np.asarray(header["minimum"], dtype=np.float64)
                min_energy = float(header["min_energy"])
                minimize_calls = int(header["minimize_calls"])
                min_ok = bool(header["min_ok"])
                catalog = ProcessCatalog(cell, tolerance=tolerance, barrier_max=barrier_max)
                for row in saved:
                    judged = row.get("per_atom_max_plus") is not None
                    catalog.observe(
                        judged=judged,
                        linked=judged and "not_connected" not in (row.get("reasons") or []),
                        structure=row.get("saddle_x"),
                        barrier=row.get("barrier"),
                    )
                for row in saved:
                    all_records.append({key: value for key, value in row.items() if key not in {"saddle_x", "endpoint_plus", "endpoint_minus"}})
                print(
                    f"{template['id']} rep {rep} resume searches {catalog.searches} connected {catalog.qualifying}",
                    flush=True,
                )
            else:
                if rep == 1:
                    archive.reset(stored_cell(cell), free)
                surface = HarnessSurface(_bind(raw, positions, free, cell))
                minimum, min_ok = _lbfgs_minimize(surface, positions[free].reshape(-1).copy(), force_tol, 5000)
                min_energy, _gradient = surface.energy_forces(minimum)
                minimize_calls = int(surface.calls)
                catalog = ProcessCatalog(cell, tolerance=tolerance, barrier_max=barrier_max)
                header_path.parent.mkdir(parents=True, exist_ok=True)
                header_path.write_text(
                    json.dumps(
                        {
                            "template": template["id"],
                            "rep": rep,
                            "optimizer_parameters": parameters,
                            "maxiter": maxiter,
                            "minimum": minimum.tolist(),
                            "min_energy": float(min_energy),
                            "minimize_calls": minimize_calls,
                            "min_ok": bool(min_ok),
                        }
                    ),
                    encoding="utf-8",
                )
                journal_path.unlink(missing_ok=True)
            reactant = positions.copy()
            reactant[free] = minimum.reshape(-1, 3)
            width = max(1, int(workers))
            next_attempt = catalog.searches + 1
            pending: dict = {}
            payloads: dict = {}
            finished: dict = {}

            def _submit(attempt: int) -> None:
                drawn = draw_search_start(
                    reactant,
                    fixed,
                    cell,
                    pbc,
                    listed,
                    rep=rep,
                    attempt=attempt,
                    radius=radius,
                    magnitude=magnitude,
                )
                item = {
                    "name": f"rep{rep}-search{attempt}",
                    "positions": drawn["start"],
                    "fixed": fixed,
                    "cell": cell,
                    "pbc": pbc,
                    "potential": problem["potential"],
                    "task": task,
                    "sella_parameters": parameters,
                    "maxiter": maxiter,
                    "force_tol": force_tol,
                    "climb_cap": _CLIMB_BUDGET,
                    "minimum": minimum,
                    "min_energy": min_energy,
                    "kick": drawn["kick"],
                    "tolerance": tolerance,
                    "seed": int(drawn["seed"]),
                }
                payloads[attempt] = item
                if pool is None:
                    finished[attempt] = _catalog_job(item)
                else:
                    pending[pool.submit(_catalog_job, item)] = attempt

            while (catalog.qualifying < goal or not stop_at_goal) and catalog.searches < max_searches:
                while next_attempt <= max_searches and len(pending) < width and (catalog.qualifying < goal or not stop_at_goal):
                    _submit(next_attempt)
                    next_attempt += 1
                    if pool is None:
                        break
                if pool is not None and pending and (catalog.searches + 1) not in finished:
                    done, _pending_left = wait(set(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        finished[pending.pop(future)] = future.result()
                attempt = catalog.searches + 1
                if attempt not in finished:
                    continue
                item = payloads.pop(attempt)
                raw_row = finished.pop(attempt)
                judged = bool(raw_row["judged"])
                decision = catalog.observe(
                    judged=judged,
                    linked=bool(raw_row["linked"]),
                    structure=raw_row.get("saddle_x"),
                    barrier=raw_row.get("barrier"),
                )
                record = _public_row(template, problem["id"], raw_row, decision)
                plus = None if raw_row.get("endpoint_plus") is None else np.asarray(raw_row["endpoint_plus"], dtype=np.float64).reshape(-1, 3)
                minus = None if raw_row.get("endpoint_minus") is None else np.asarray(raw_row["endpoint_minus"], dtype=np.float64).reshape(-1, 3)
                status = archive.add(minimum.reshape(-1, 3), plus, minus)
                stamp_endpoint(record, archive.name, archive.count - 1, status)
                stored = dict(record)
                stored["seed"] = int(item["seed"])
                _append_journal(journal_path, stored)
                public = {
                    key: value
                    for key, value in record.items()
                    if key not in {"saddle_x", "endpoint_plus", "endpoint_minus"}
                }
                all_records.append(public)
                if catalog.searches == 1 or catalog.searches % 10 == 0 or catalog.qualifying >= goal:
                    print(
                        f"{template['id']} rep {rep} searches {catalog.searches} "
                        f"connected {catalog.qualifying} unique {catalog.unique} "
                        f"calls {record['force_calls']} failure {record['failure']}",
                        flush=True,
                    )
            if catalog.qualifying < goal:
                failed_reps += 1
            rep_calls = sum(
                int(row["force_calls"]) for row in all_records if str(row["start"]).startswith(f"rep{rep}-")
            )
            comments.append(
                f"# rep {rep} force_calls {rep_calls} "
                f"unique_saddles {catalog.unique} fraction_connected {catalog.fraction_connected} "
                f"total_searches {catalog.searches} connected {catalog.qualifying} "
                f"minimize_calls {minimize_calls} minimum_converged {int(min_ok)} "
                f"search_seed (20261015+{rep})*1000003+attempt"
            )
        metrics = write_saddle_campaign(
            output,
            template,
            problem,
            all_records,
            comments,
            archive,
            search_cap=_SEARCH_CAP,
            goal=goal,
            repetitions=repetitions,
            failed_reps=failed_reps,
            note=(
                "One row is one draw_search_start displacement. The cap is the "
                "climb budget of 1000, 64 calls of overshoot, and the two endpoint "
                "minimizations. The initial minimization is not in that count."
            ),
        )
        header_path, journal_path = _checkpoint_paths(output)
        journal_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
        return {
            "force_calls": None if metrics is None else metrics.get("success_only_mean"),
            "nfailed": failed_reps,
            "starts": len(all_records),
            "template": template["id"],
            "problem": problem["id"],
        }
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
