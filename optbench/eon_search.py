"""Per-search eOn Lanczos score for the Pt island.

Each observation is one displacement search. The harness cap is the
template ``search.per_search_cap``. The reactant minimization is outside
that count. Endpoint minimizations are inside it. A search is a success
only when the saddle is index 1 at the force stop, the barrier is under
the problem goal, and ``optbench.connection`` accepts the two minimized
endpoints. ``failure_cap`` stays off this template: that key stops an
AKMC repetition, and this row is not an AKMC repetition score.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .campaign import format_ini, write_starts
from .connection import ProcessTable, per_atom_max, periodic_lengths, tolerances
from .connection import connected as endpoints_connected
from .eon_akmc import prepare_client_env
from .eon_harness import bridge_binary, kernel_for
from .eon_jobs import _client_env, _read_dat
from .failures import classify_failure
from .potserver import PotServer, remote_count, remote_reset
from .rgsaddle_minmode import HarnessSurface, _bind, _hessian_spectrum, _stored
from .score import score_v2
from .stationary import negative_index
from .surfaces import load_con, potential_from_problem
from .v2 import collect_provenance


def _client_version(client: Path) -> str:
    proc = subprocess.run(
        [str(client), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = proc.stdout or ""
    if "version" not in text.lower() and "(" not in text:
        proc = subprocess.run(
            [str(client), "-v"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        text = proc.stdout or ""
    return text.strip()


def _require_client(client: Path, commit: str) -> None:
    if not client.is_file():
        raise RuntimeError(f"eonclient is not a file: {client}")
    token = commit[:8]
    text = _client_version(client)
    if token not in text:
        raise RuntimeError(f"{client} printed {text!r}, expected commit {token}")


def _job_ini(template: dict, job: str, seed: int, wrapper: Path, *, max_iterations: int | None = None) -> str:
    sections = json.loads(json.dumps(template["ini"]))
    sections.pop("Search", None)
    main = sections.setdefault("Main", {})
    main["job"] = job
    main["random_seed"] = int(seed)
    main["parallel"] = False
    if max_iterations is not None:
        sections.setdefault("Optimizer", {})["max_iterations"] = int(max_iterations)
    sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
    return format_ini(sections)


def _run_job(client: Path, ini: str, files: dict[str, Path], dest: Path, timeout: int) -> dict:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.ini").write_text(ini, encoding="utf-8")
    for name, src in files.items():
        shutil.copy(src, dest / name)
    try:
        proc = subprocess.run(
            [str(client)],
            cwd=dest,
            env=_client_env(str(client)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"fields": {}, "returncode": None, "error": f"timeout {exc.timeout}"}
    return {
        "fields": _read_dat(dest / "results.dat"),
        "returncode": proc.returncode,
        "error": "",
        "stderr": (proc.stderr or "")[-500:],
    }


def _con_positions(path: Path) -> np.ndarray:
    positions, _fixed, _cell = load_con(path)
    return positions


def _shift_con(src: Path, mode: np.ndarray, scale: float, dest: Path) -> None:
    from eon import fileio as io

    atoms = io.loadcon(str(src))
    atoms.r = np.asarray(atoms.r, dtype=np.float64) + float(scale) * np.asarray(mode, dtype=np.float64)
    io.savecon(str(dest), atoms)


def _pack_state(state) -> dict:
    kind, keys, pos, has_gauss, cached = state
    return {
        "kind": kind,
        "keys": np.asarray(keys, dtype=np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached": float(cached),
    }


def _unpack_state(payload: dict):
    keys = np.asarray(payload["keys"], dtype=np.uint32)
    return (payload["kind"], keys, int(payload["pos"]), int(payload["has_gauss"]), float(payload["cached"]))


def _checkpoint_paths(output: Path) -> tuple[Path, Path, Path]:
    return (
        output.with_name(output.stem + ".header.json"),
        output.with_name(output.stem + ".partial.jsonl"),
        output.with_name(output.stem + ".rng.json"),
    )


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _load_partial(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _public(record: dict) -> dict:
    hidden = {"rng_state", "saddle_x"}
    return {key: value for key, value in record.items() if key not in hidden}


def _run_search(payload: dict) -> dict:
    """One displacement search. Novelty stays in the parent, in search order."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    client = Path(payload["client"])
    prepare_client_env(client)
    from eon import fileio as io

    cap = int(payload["cap"])
    work = Path(payload["work"])
    work.mkdir(parents=True, exist_ok=True)
    server = PotServer(kernel_for("morse"), cap=cap).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    reasons = ["force_norm"]
    barrier = None
    per_plus = None
    per_minus = None
    hessian_index = None
    table_point = None
    termination = ""
    calls = 0
    stderr = ""
    try:
        remote_reset(server.host, server.port)
        saddle_job = _run_job(
            client,
            payload["saddle_ini"],
            {
                "pos.con": Path(payload["reactant_con"]),
                "displacement.con": Path(payload["disp_con"]),
                "direction.dat": Path(payload["mode_path"]),
            },
            work / "search",
            timeout=int(payload["timeout"]),
        )
        calls, _verify = remote_count(server.host, server.port)
        fields = saddle_job["fields"]
        termination = str(fields.get("termination_reason", ""))
        stderr = saddle_job.get("stderr") or ""
        search_dir = work / "search"
        if saddle_job["error"]:
            reasons = ["crash"]
        elif calls >= cap:
            reasons = ["cap", "force_norm"]
        elif termination != "0" or not (search_dir / "saddle.con").is_file() or not (search_dir / "mode.dat").is_file():
            reasons = ["divergence"] if termination else ["crash"]
        else:
            saddle_path = work / "saddle.con"
            shutil.copy(search_dir / "saddle.con", saddle_path)
            climbed_mode = io.load_mode(str(search_dir / "mode.dat"))
            plus_path = work / "plus.con"
            minus_path = work / "minus.con"
            offset = float(payload["offset"])
            _shift_con(saddle_path, climbed_mode, offset, plus_path)
            _shift_con(saddle_path, climbed_mode, -offset, minus_path)
            plus_job = _run_job(
                client,
                payload["min_ini"],
                {"pos.con": plus_path},
                work / "min-plus",
                timeout=int(payload["timeout"]),
            )
            minus_job = _run_job(
                client,
                payload["min_ini"],
                {"pos.con": minus_path},
                work / "min-minus",
                timeout=int(payload["timeout"]),
            )
            calls, _verify = remote_count(server.host, server.port)
            plus_con = work / "min-plus" / "min.con"
            minus_con = work / "min-minus" / "min.con"
            if calls >= cap:
                reasons = ["cap", "force_norm"]
            elif plus_job["error"] or minus_job["error"] or not plus_con.is_file() or not minus_con.is_file():
                reasons = ["crash"]
            elif (
                plus_job["fields"].get("termination_reason") != "0"
                or minus_job["fields"].get("termination_reason") != "0"
            ):
                reasons = ["divergence"]
            else:
                reactant_pos = np.asarray(payload["reactant_pos"], dtype=np.float64)
                lengths = np.asarray(payload["lengths"], dtype=np.float64)
                plus = _con_positions(plus_con)
                minus = _con_positions(minus_con)
                links = endpoints_connected(
                    reactant_pos, plus, minus, lengths, float(payload["distance_difference"])
                )
                saddle_energy = fields.get("potential_energy_saddle", fields.get("potential_energy"))
                barrier = float(saddle_energy) - float(payload["min_energy"]) if saddle_energy is not None else None
                per_plus = per_atom_max(plus, reactant_pos, lengths)
                per_minus = per_atom_max(minus, reactant_pos, lengths)
                saddle_pos = _con_positions(saddle_path)
                positions = np.asarray(payload["positions"], dtype=np.float64).reshape(-1, 3)
                fixed = np.asarray(payload["fixed"], dtype=bool)
                free = np.flatnonzero(~fixed)
                raw = potential_from_problem({"potential": payload["problem_potential"]})
                verify_surface = HarnessSurface(
                    _bind(raw, positions, free, np.asarray(payload["cell"], dtype=np.float64))
                )
                flat = saddle_pos[free].reshape(-1)
                _energy, gradient = verify_surface.energy_forces(flat)
                force_norm = float(np.linalg.norm(gradient))
                force_tol = float(payload["force_tol"])
                reasons = []
                if force_norm > force_tol + 1e-10:
                    reasons.append("force_norm")
                else:
                    eigenvalues, _vectors = _hessian_spectrum(verify_surface, flat)
                    hessian_index = int(negative_index(eigenvalues, 1e-6))
                    if hessian_index != 1:
                        reasons.append("hessian_index")
                if not links:
                    reasons.append("not_connected")
                index_ok = hessian_index == 1 and force_norm <= force_tol + 1e-10
                if links and index_ok and barrier is not None:
                    table_point = saddle_pos.reshape(-1).tolist()
                if barrier is None or barrier >= float(payload["barrier_max"]):
                    reasons.append("barrier")
        if not reasons and calls >= cap:
            reasons = ["cap"]
        return {
            "index": int(payload["index"]),
            "calls": int(calls),
            "reasons": reasons,
            "barrier": barrier,
            "hessian_index": hessian_index,
            "per_plus": per_plus,
            "per_minus": per_minus,
            "termination": termination,
            "table_point": table_point,
            "rng_state": payload["rng_state"],
            "stderr": stderr[-200:],
        }
    finally:
        server.stop()


def _judge_search(result: dict, found: ProcessTable) -> tuple[dict, dict, bool]:
    """Apply the process table in order and build the stored row."""
    reasons = list(result["reasons"])
    saved_point = None
    if result.get("table_point") is not None and result.get("barrier") is not None:
        point = np.asarray(result["table_point"], dtype=np.float64)
        barrier = float(result["barrier"])
        if found.repeat(point, barrier):
            if "barrier" in reasons:
                reasons.insert(reasons.index("barrier"), "duplicate")
            else:
                reasons.append("duplicate")
        else:
            found.add(point, barrier)
            saved_point = point
    success = not reasons
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2",
        "start": f"rep1-search{int(result['index'])}",
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "force_calls": int(result["calls"]),
        "barrier": result["barrier"],
        "hessian_index": result["hessian_index"],
        "per_atom_max_plus": result["per_plus"],
        "per_atom_max_minus": result["per_minus"],
        "termination_reason": result["termination"],
    }
    stored = dict(record)
    stored["rng_state"] = result["rng_state"]
    if saved_point is not None:
        stored["saddle_x"] = np.asarray(saved_point, dtype=np.float64).reshape(-1).tolist()
    return record, stored, success


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    client: Path | None,
    work: Path | None,
    workers: int = 1,
    limit: int = 0,
) -> dict:
    """Run displacement searches and write the campaign files."""
    search = template.get("search") or {}
    cap = int(search["per_search_cap"])
    repetitions = int(search.get("repetitions", 1))
    max_searches = int(search.get("max_searches", 20000))
    if limit > 0:
        max_searches = min(max_searches, int(limit))
    if repetitions != 1:
        raise RuntimeError(f"{template['id']} repetitions {repetitions} is not the stock count 1")
    if int(workers) < 1:
        raise RuntimeError(f"{template['id']} workers {workers} is below 1")
    if client is None:
        env = os.environ.get("EONCLIENT", "")
        if not env:
            raise RuntimeError("pass --client or set EONCLIENT to the develop eonclient")
        client = Path(env)
    commit = str(template.get("commit") or "")
    _require_client(client, commit)
    prepare_client_env(client)
    from eon import fileio as io
    from eon.config import ConfigClass
    from eon.displace import DisplacementManager

    paths = (problem.get("input") or {}).get("geometries") or []
    con_path = next(path for path in paths if str(path).endswith(".con"))
    positions, fixed, cell = load_con(_stored(root, con_path))
    distance_difference, energy_difference = tolerances(problem)
    lengths = periodic_lengths(cell, (problem.get("potential") or {}).get("pbc"))
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    force_tol = float((problem.get("convergence") or {}).get("force_norm", 1.0e-3))
    seed = int((template.get("ini") or {}).get("Main", {}).get("random_seed", 1))
    work_dir = work or output.parent / "work" / template["id"].replace("/", "-")
    work_dir.mkdir(parents=True, exist_ok=True)
    header_path, partial_path, rng_path = _checkpoint_paths(output)
    saved = _load_partial(partial_path) if header_path.is_file() and json.loads(header_path.read_text(encoding="utf-8")).get("template") == template["id"] else []
    if saved and not rng_path.is_file():
        raise RuntimeError(f"{partial_path} has searches and {rng_path.name} is missing")

    evaluate = kernel_for("morse")
    server = PotServer(evaluate, cap=cap).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    wrapper = bridge_binary()
    try:
        if saved:
            header = json.loads(header_path.read_text(encoding="utf-8"))
            reactant_con = Path(header["reactant_con"])
            min_energy = float(header["min_energy"])
            minimize_calls = int(header["minimize_calls"])
            np.random.set_state(_unpack_state(json.loads(rng_path.read_text(encoding="utf-8"))))
        else:
            partial_path.unlink(missing_ok=True)
            rng_path.unlink(missing_ok=True)
            np.random.seed(seed)
            remote_reset(server.host, server.port)
            reactant_con = work_dir / "reactant-min.con"
            shutil.copy(_stored(root, con_path), work_dir / "pos-start.con")
            minimized = _run_job(
                client,
                _job_ini(template, "minimization", seed, wrapper, max_iterations=5000),
                {"pos.con": work_dir / "pos-start.con"},
                work_dir / "minimize",
                timeout=600,
            )
            minimize_calls, _verify = remote_count(server.host, server.port)
            min_con = work_dir / "minimize" / "min.con"
            if not min_con.is_file() or minimized["fields"].get("termination_reason") not in {"0", "good"}:
                raise RuntimeError(
                    f"reactant minimization failed rc={minimized['returncode']} "
                    f"fields={minimized['fields']} {minimized.get('stderr', '')}"
                )
            shutil.copy(min_con, reactant_con)
            min_energy = float(minimized["fields"]["potential_energy"])
            remote_reset(server.host, server.port)
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(
                json.dumps(
                    {
                        "template": template["id"],
                        "commit": commit,
                        "reactant_con": str(reactant_con),
                        "min_energy": min_energy,
                        "minimize_calls": int(minimize_calls),
                    }
                ),
                encoding="utf-8",
            )
        reactant = io.loadcon(str(reactant_con))
        config_path = work_dir / "config.ini"
        config_path.write_text(format_ini(template["ini"]), encoding="utf-8")
        previous = Path.cwd()
        try:
            os.chdir(work_dir)
            config = ConfigClass()
            config.init(str(config_path))
            # ConfigClass.init seeds numpy from random_seed. Restore the
            # checkpoint after that, or a resumed run repeats search 1.
            if saved:
                np.random.set_state(_unpack_state(json.loads(rng_path.read_text(encoding="utf-8"))))
            manager = DisplacementManager(reactant, None, config=config)
            offset = float(config.process_search_minimization_offset)
        finally:
            os.chdir(previous)
        reactant_pos = _con_positions(reactant_con)
        found = ProcessTable(lengths, distance_difference, energy_difference)
        connected = 0
        records = []
        for row in saved:
            public = _public(row)
            records.append(public)
            if row.get("saddle_x") is not None:
                found.add(row["saddle_x"], float(row["barrier"]))
            if public.get("success"):
                connected += 1
        print(
            f"{template['id']} resume searches {len(records)} connected {connected}",
            flush=True,
        )
        saddle_ini = _job_ini(template, "saddle_search", seed, wrapper)
        min_ini = _job_ini(template, "minimization", seed, wrapper, max_iterations=2000)
        reactant_flat = reactant_pos.reshape(-1).tolist()
        positions_flat = np.asarray(positions, dtype=np.float64).reshape(-1).tolist()
        fixed_flags = [bool(flag) for flag in fixed]
        cell_flat = np.asarray(cell, dtype=np.float64).reshape(-1).tolist()
        # Displacements are drawn in order. Workers may run ahead, and the
        # process table is applied to finished searches in that same order.
        # A search still running after the goal is left out of the row.
        context = mp.get_context("spawn")
        pending: deque = deque()
        produced = len(records)

        def _submit(pool: ProcessPoolExecutor) -> None:
            nonlocal produced
            index = produced + 1
            disp_atoms, mode = manager.make_displacement()
            job_dir = work_dir / f"job-{index}"
            job_dir.mkdir(parents=True, exist_ok=True)
            disp_path = job_dir / "displacement.con"
            mode_path = job_dir / "direction.dat"
            io.savecon(str(disp_path), disp_atoms)
            io.save_mode(str(mode_path), mode, getattr(reactant, "free", None))
            payload = {
                "index": index,
                "client": str(client),
                "cap": cap,
                "saddle_ini": saddle_ini,
                "min_ini": min_ini,
                "reactant_con": str(reactant_con),
                "disp_con": str(disp_path),
                "mode_path": str(mode_path),
                "work": str(job_dir),
                "offset": offset,
                "timeout": 600,
                "reactant_pos": reactant_flat,
                "lengths": lengths.tolist(),
                "distance_difference": distance_difference,
                "min_energy": min_energy,
                "barrier_max": barrier_max,
                "force_tol": force_tol,
                "problem_potential": problem.get("potential") or {},
                "positions": positions_flat,
                "fixed": fixed_flags,
                "cell": cell_flat,
                "rng_state": _pack_state(np.random.get_state()),
            }
            pending.append(pool.submit(_run_search, payload))
            produced += 1

        with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
            while connected < goal and len(records) < max_searches:
                while len(pending) < int(workers) and produced < max_searches:
                    _submit(pool)
                if not pending:
                    break
                result = pending.popleft().result()
                record, stored, success = _judge_search(result, found)
                if success:
                    connected += 1
                records.append(record)
                _append(partial_path, stored)
                rng_path.write_text(json.dumps(result["rng_state"]), encoding="utf-8")
                index = len(records)
                if index == 1 or index % 25 == 0 or connected >= goal or record["failure"] == "crash":
                    print(
                        f"{template['id']} searches {index} connected {connected} "
                        f"calls {record['force_calls']} failure {record['failure']}",
                        flush=True,
                    )
                    if record["failure"] == "crash" and result.get("stderr"):
                        print(result["stderr"], flush=True)
                if connected >= goal:
                    pending.clear()
                    break
        failed_reps = 0 if connected >= goal else 1
        fraction = connected / len(records) if records else 0.0
        search_calls = sum(int(row["force_calls"]) for row in records)
        text = (
            f"# rep 1 force_calls {search_calls + int(minimize_calls)} unique_saddles {len(found)} "
            f"fraction_connected {fraction} total_searches {len(records)} "
            f"connected {connected} minimize_calls {int(minimize_calls)} "
            f"minimum_converged 1 seed {seed}\n"
            f"force_calls {float(search_calls + int(minimize_calls))}\n"
            f"force_calls_min {search_calls + int(minimize_calls)}\n"
            f"force_calls_max {search_calls + int(minimize_calls)}\n"
            f"code {template.get('code', 'eOn')}\n"
            f"code_version {template.get('code_version', '')}\n"
            f"commit {commit}\n"
            f"algorithm {template.get('algorithm', '')}\n"
            f"template {template['id']}\n"
            f"problem {problem['id']}\n"
            f"counter harness\n"
            f"repetitions {repetitions}\n"
            f"nfailed {failed_reps}\n"
            f"goal {goal}\n"
            f"cap {cap}\n"
            f"n {len(records)}\n"
        )
        metrics = score_v2(
            [
                {"force_calls": row["force_calls"], "success": row["success"], "failure": row["failure"]}
                for row in records
            ],
            cap,
        ) if records else None
        if metrics is not None:
            text += f"par2_mean {metrics['par2_mean']}\n" f"v2_nfailed {metrics['nfailed']}\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        write_starts(output.with_suffix(".starts.jsonl"), records)
        if metrics is not None:
            provenance = collect_provenance(
                code=str(template.get("code", "eOn")),
                code_version=str(template.get("code_version", "")),
                commit=commit,
                parameters={"template": template["id"], "problem": problem["id"], "cap": cap},
            )
            document = {
                "schema": "optbench.result/v2",
                "status": "measured",
                "problem": problem["id"],
                "scoring": "par2-wilson-bootstrap",
                "par_factor": 2,
                "bootstrap_resamples": 2000,
                "bootstrap_seed": 20261015,
                "note": (
                    "One row is one displacement search. The cap is the per-search "
                    "harness budget. The reactant minimization is not in that count."
                ),
                "runs": [
                    {
                        "entry": output.stem,
                        "code": template.get("code", "eOn"),
                        "algorithm": template.get("algorithm", ""),
                        "code_version": template.get("code_version", ""),
                        "commit": commit,
                        "tuning": template.get("tuning", ""),
                        "present_source": True,
                        "cap": cap,
                        "provenance": provenance,
                        "metrics": metrics,
                        "starts": records,
                    }
                ],
            }
            output.with_suffix(".v2.json").write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for path in (header_path, partial_path, rng_path):
            path.unlink(missing_ok=True)
        return {
            "force_calls": float(search_calls + int(minimize_calls)) if records else None,
            "nfailed": failed_reps,
            "starts": len(records),
            "template": template["id"],
            "problem": problem["id"],
            "par2_mean": None if metrics is None else metrics["par2_mean"],
            "connected": connected,
        }
    finally:
        server.stop()
