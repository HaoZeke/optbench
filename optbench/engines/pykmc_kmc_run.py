"""KMC campaigns for the pyKMC hTST and aluminum-adatom rows.

Searches start at the minimized reactant. The push is the published
profile. hTST reports the archive cycle means. The adatom rows report
the same columns as the eOn AKMC reducer. Force calls are harness counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optbench.amd import (
    KELVIN_TO_EV,
    catalogue_accuracy,
    distribution,
    force_calls_per_simulated_time,
    match_product,
    mobile_indices,
    mobile_rmsd,
    old_scheme_confidence,
    parse_con,
    rate_accuracy,
    rate_distribution,
    rate_summary,
    select_thermal,
    simulated_time_seconds,
)
from optbench.pykmc_kmc import HTST, PROFILE, amd_dat, htst_dat, reduce_htst


def _engine():
    from optbench.engines import pykmc_search as engine

    return engine


def _load_reactant(root: Path, template: dict):
    engine = _engine()
    geometry = str(template.get("geometry") or "")
    positions, free, box = engine._read_geometry(root, geometry)
    text = engine._stored_text(root, geometry)
    parsed, fixed = parse_con(text)
    if parsed.shape != positions.shape:
        raise RuntimeError(f"{geometry} parse does not match the search geometry")
    return positions, free, box, parsed, mobile_indices(fixed)


def _catalogue(root: Path) -> list[np.ndarray]:
    from optbench.campaign import geometry_archive

    products = []
    for index in range(24):
        path = geometry_archive(
            root,
            {"id": "accelerated-md", "input": {"archive_path": f"website/accelerated-md/products/product{index}.con"}},
        )
        products.append(parse_con(path.read_text(encoding="utf-8", errors="replace"))[0])
    return products


def _centers(minimum: np.ndarray, free: np.ndarray) -> list[int]:
    mask = _engine()._event_mask(minimum, free)
    return [int(index) for index in np.flatnonzero(mask)]


def _match_process(record: dict, processes: list[dict], connect_rmsd: float, box, pbc) -> int | None:
    """Same saddle and the same barrier.

    The link distance is 0.5 Å on the top window. Two different events
    on that coarse mask can sit inside it, so a repeat also has to agree
    with the stored barrier. 0.05 eV keeps a converged saddle together
    and splits a 0.17 eV hop from a 0.44 eV hop.
    """
    if not record.get("eon_connected"):
        return None
    barrier = record.get("barrier")
    if barrier is None or not np.isfinite(float(barrier)) or float(barrier) <= 0.0:
        return None
    mask = record.get("saddle_mask")
    if not mask:
        return None
    query = np.asarray(mask, dtype=np.float64).reshape(-1, 3)
    image = {"lengths": box, "pbc": pbc}
    rmsd = _engine()._rmsd
    height = float(barrier)
    for process in processes:
        if abs(float(process["barrier"]) - height) > 0.05:
            continue
        if rmsd(query, process["saddle"], **image) <= connect_rmsd:
            return int(process["id"])
    return None


def _prepare(root: Path, problem: dict, template: dict, output: Path):
    engine = _engine()
    surface = engine._surface(problem, template)
    section = engine._section(template)
    force_norm = engine._force(problem, template)
    section["forc_thr"] = force_norm
    positions, free, box, _parsed, mobile = _load_reactant(root, template)
    evaluator = engine._evaluator(surface, box)
    kind = "morse_pt" if surface["kind"] == "morse" else surface["kind"]
    work = output.parent / "work" / template["id"].replace("/", "-")
    work.mkdir(parents=True, exist_ok=True)
    snap_path = work / "kmc.json"
    state = None
    if snap_path.is_file():
        loaded = json.loads(snap_path.read_text(encoding="utf-8"))
        if loaded.get("profile") == PROFILE:
            state = loaded
    if state and state.get("minimum") is not None:
        minimum = np.asarray(state["minimum"], dtype=np.float64).reshape(-1, 3)
        min_energy = float(state["min_energy"])
        minimize_calls = int(state["minimize_calls"])
        min_ok = bool(state["min_ok"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        processes = []
        for item in state.get("processes") or []:
            processes.append(
                {
                    "id": int(item["id"]),
                    "barrier": float(item["barrier"]),
                    "rate": float(item["rate"]) if item.get("rate") is not None else None,
                    "channel": item.get("channel"),
                    "saddle": np.asarray(item["saddle"], dtype=np.float64).reshape(-1, 3),
                    "product": None
                    if item.get("product") is None
                    else np.asarray(item["product"], dtype=np.float64).reshape(-1, 3),
                }
            )
        rows = list(state.get("rows") or [])
        repeats = int(state.get("repeats") or 0)
    else:
        minimum, min_ok, minimize_calls, min_energy = engine._minimize_free(
            evaluator, positions, free, box, force_norm, min(5000, engine._cap(problem, template))
        )
        rng = np.random.default_rng(engine.pair_seed(template["id"], engine._SEED_BASE))
        processes = []
        rows = []
        repeats = 0
        state = None
    centers = _centers(minimum, free)
    if not centers:
        raise RuntimeError(f"{template['id']} has no atom in the top window")
    return {
        "engine": engine,
        "surface": surface,
        "section": section,
        "force_norm": force_norm,
        "free": free,
        "box": box,
        "mobile": mobile,
        "evaluator": evaluator,
        "kind": kind,
        "minimum": minimum,
        "min_energy": min_energy,
        "minimize_calls": minimize_calls,
        "min_ok": min_ok,
        "rng": rng,
        "processes": processes,
        "rows": rows,
        "repeats": repeats,
        "centers": centers,
        "snap_path": snap_path,
        "resumed": state is not None,
    }


def _dump_processes(processes: list[dict]) -> list[dict]:
    dumped = []
    for process in processes:
        dumped.append(
            {
                "id": int(process["id"]),
                "barrier": float(process["barrier"]),
                "rate": process.get("rate"),
                "channel": process.get("channel"),
                "saddle": np.asarray(process["saddle"], dtype=np.float64).reshape(-1).tolist(),
                "product": None
                if process.get("product") is None
                else np.asarray(process["product"], dtype=np.float64).reshape(-1).tolist(),
            }
        )
    return dumped


def _checkpoint(prepared: dict, rows: list[dict], repeats: int) -> None:
    prepared["snap_path"].write_text(
        json.dumps(
            {
                "profile": PROFILE,
                "minimum": np.asarray(prepared["minimum"]).reshape(-1).tolist(),
                "min_energy": prepared["min_energy"],
                "minimize_calls": prepared["minimize_calls"],
                "min_ok": prepared["min_ok"],
                "rows": rows,
                "processes": _dump_processes(prepared["processes"]),
                "repeats": repeats,
                "rng": prepared["rng"].bit_generator.state,
            },
            default=prepared["engine"]._json_default,
        ),
        encoding="utf-8",
    )


def _batch(root: Path, template: dict, prepared: dict, *, number: int, room: int, climb_cap: int, workers: int):
    engine = prepared["engine"]
    jobs = []
    centers = prepared["centers"]
    rng = prepared["rng"]
    for offset in range(room):
        atom = int(centers[int(rng.integers(0, len(centers)))])
        zseed = engine.pair_seed(f"{template['id']}:search{number + offset}", engine._SEED_BASE)
        jobs.append(
            engine._base_job(
                f"search{number + offset}",
                prepared["minimum"],
                prepared["free"],
                prepared["box"],
                prepared["surface"],
                prepared["section"],
                cap=climb_cap,
                central=atom,
                zseed=zseed,
            )
        )
    return engine._judge_batch(
        root,
        jobs,
        workers=workers,
        cap=climb_cap,
        force_norm=prepared["force_norm"],
        kind=prepared["kind"],
        free=prepared["free"],
        box=prepared["box"],
        evaluator=prepared["evaluator"],
    )


def _link_one(prepared: dict, record: dict, *, barrier_max: float, connect_rmsd: float, found: list) -> dict:
    engine = prepared["engine"]
    return engine._link(
        record,
        evaluator=prepared["evaluator"],
        free=prepared["free"],
        box=prepared["box"],
        minimum=prepared["minimum"],
        min_energy=prepared["min_energy"],
        force_norm=prepared["force_norm"],
        barrier_max=barrier_max,
        connect_rmsd=connect_rmsd,
        found=found,
        pbc=prepared["surface"]["pbc"],
    )


def _strip(record: dict) -> dict:
    for key in ("endpoint_plus", "endpoint_minus", "product", "saddle_mask", "positions", "lowest_mode"):
        record.pop(key, None)
    return record


def _start(template: dict, problem: dict, *, name: str, calls: int, success: bool, failure: str | None, **extra) -> dict:
    row = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "counter": "harness",
        "comparable": True,
        "success": bool(success),
        "failure": failure,
        "problem": problem["id"],
        "template": template["id"],
        "code": template.get("code", "pyKMC"),
        "code_version": template.get("code_version", ""),
        "commit": template.get("commit", ""),
        "tuning": template.get("tuning", ""),
    }
    row.update(extra)
    return row


def _write_v2(output: Path, template: dict, problem: dict, records: list[dict], cap: int, prepared: dict) -> None:
    from optbench.score import score_v2
    from optbench.v2 import collect_provenance, write_starts

    engine = prepared["engine"]
    write_starts(output.with_suffix(".starts.jsonl"), records)
    score = score_v2(
        [
            {"force_calls": row["force_calls"], "success": row["success"], "failure": row.get("failure")}
            for row in records
        ],
        cap,
    )
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template.get("code", "pyKMC"),
            code_version=template.get("code_version", ""),
            commit=template.get("commit", ""),
            parameters=engine._parameters(prepared["surface"], prepared["section"], cap, prepared["force_norm"]),
        ),
        "score": score,
    }
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_htst(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    """Stay on the reactant and score archive barrier and rate cycles."""
    spec = HTST[problem["id"]]
    prepared = _prepare(root, problem, template, output)
    section = prepared["section"]
    kmc = dict(template.get("kmc") or {})
    max_searches = int(kmc.get("max_searches", 20000))
    connect_rmsd = float(section.get("rmsd_connect", 0.5))
    barrier_max = float(section.get("barrier_max", 1.5))
    climb_cap = int(section["evalf_max"])
    cap = int(template["failure_cap"])
    rows: list[dict] = list(prepared["rows"])
    found: list[np.ndarray] = [process["saddle"] for process in prepared["processes"]]
    root = Path(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def publish(partial: bool) -> dict:
        summary = reduce_htst(rows, spec)
        done = (
            summary["barrier_cycles"] >= int(spec["barrier_cycles"])
            and summary["rate_cycles"] >= int(spec["rate_cycles"])
        )
        partial = partial or not done
        comments = [
            "# counter harness. pyKMC partn_search on the minimized reactant.",
            f"# Temperature {spec['temperature']:g} K, prefactor {spec['prefactor']:.6g}, from {spec['source']}.",
            "# A barrier cycle ends when every reference barrier, rounded to 3 decimals, has been seen.",
            "# A rate cycle ends at 0.999 of rate_ref. The link test is unchanged.",
            (
                "# The 2014 Pt barriers were measured with shifted Morse. This row uses morse/smooth/linear. "
                "eon-svn hTST Pt uses kernel morse_shift. eon-client hTST Pt uses this smooth-linear kernel."
                if problem["id"] == "hTST/Pt"
                else "# The kernel is the r2549 Aluminum EAM, the same family as the eon-svn and eon-client hTST Al rows."
            ),
            f"# matched {summary['matched_barriers']} of {summary['reference_barriers']} reference barriers "
            f"after {summary['searches']} searches.",
        ]
        if summary["unmatched_barriers"]:
            comments.append("# unmatched reference barriers: " + " ".join(f"{value:.3f}" for value in summary["unmatched_barriers"]))
        seen = sorted(
            {
                round(float(row["barrier"]), 3)
                for row in rows
                if row.get("process_id") is not None and row.get("barrier") is not None
            }
        )
        if seen:
            comments.append("# observed barriers: " + " ".join(f"{value:.3f}" for value in seen))
        text = htst_dat(
            summary,
            template=template,
            problem=problem,
            spec=spec,
            searches=len(rows),
            minimize_calls=int(prepared["minimize_calls"]),
            partial=partial,
            comments=comments,
        )
        output.write_text(text, encoding="utf-8")
        if summary["barrier_call_list"]:
            records = [
                _start(
                    template,
                    problem,
                    name=f"barrier-cycle-{index}",
                    calls=int(calls),
                    success=True,
                    failure=None,
                )
                for index, calls in enumerate(summary["barrier_call_list"], start=1)
            ]
        else:
            spent = sum(int(row["force_calls"]) for row in rows) + int(prepared["minimize_calls"])
            records = [
                _start(
                    template,
                    problem,
                    name="kmc",
                    calls=spent,
                    success=False,
                    failure="divergence",
                    reasons=["divergence"],
                )
            ]
        _write_v2(output, template, problem, records, cap, prepared)
        return summary

    while len(rows) < max_searches:
        summary = reduce_htst(rows, spec)
        if (
            summary["barrier_cycles"] >= int(spec["barrier_cycles"])
            and summary["rate_cycles"] >= int(spec["rate_cycles"])
        ):
            break
        room = min(max(1, int(workers)), max_searches - len(rows))
        print(
            f"{template['id']} searches {len(rows)} barrier_cycles {summary['barrier_cycles']} "
            f"rate_cycles {summary['rate_cycles']} matched {summary['matched_barriers']}",
            flush=True,
        )
        judged = _batch(root, template, prepared, number=len(rows) + 1, room=room, climb_cap=climb_cap, workers=workers)
        for record in judged:
            linked = _link_one(prepared, record, barrier_max=barrier_max, connect_rmsd=connect_rmsd, found=found)
            ident = _match_process(linked, prepared["processes"], connect_rmsd, prepared["box"], prepared["surface"]["pbc"])
            if ident is None and linked.get("eon_connected") and linked.get("saddle_mask") and linked.get("barrier"):
                ident = len(prepared["processes"]) + 1
                prepared["processes"].append(
                    {
                        "id": ident,
                        "barrier": float(linked["barrier"]),
                        "rate": None,
                        "channel": None,
                        "saddle": np.asarray(linked["saddle_mask"], dtype=np.float64).reshape(-1, 3),
                        "product": None,
                    }
                )
                found.append(prepared["processes"][-1]["saddle"])
            rows.append(
                {
                    "force_calls": int(linked["force_calls"]),
                    "process_id": ident,
                    "barrier": linked.get("barrier"),
                    "success": bool(linked["success"]),
                    "eon_connected": bool(linked.get("eon_connected")),
                }
            )
            print(
                f"{template['id']} search {len(rows)} calls {linked['force_calls']} "
                f"barrier {linked.get('barrier')} process {ident} eon {int(bool(linked.get('eon_connected')))}",
                flush=True,
            )
            _strip(linked)
        _checkpoint(prepared, rows, 0)
        publish(partial=True)
    summary = publish(partial=False)
    return {
        "force_calls": summary["barrier_force_calls"],
        "nfailed": 0 if summary["barrier_cycles"] else 1,
        "starts": max(1, int(summary["barrier_cycles"])),
        "template": template["id"],
        "problem": problem["id"],
    }


def run_amd(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    """One reactant catalogue, scored like the eOn AKMC reducer."""
    prepared = _prepare(root, problem, template, output)
    section = prepared["section"]
    kmc = dict(template.get("kmc") or {})
    prefactor = float(kmc.get("prefactor", 1.0e12))
    max_searches = int(kmc.get("max_searches", 5000))
    temperature = float(problem["thermal"]["temperature_K"])
    window = float(problem["convergence"]["thermally_accessible_window"])
    min_barrier = float(problem["convergence"]["min_barrier_eV"])
    confidence_target = float(problem["convergence"]["confidence"])
    tolerance = float(problem["convergence"]["catalogue_rmsd"])
    connect_rmsd = float(section.get("rmsd_connect", 0.5))
    barrier_max = float(section.get("barrier_max", 20.0))
    climb_cap = int(section["evalf_max"])
    cap = int(template["failure_cap"])
    products = _catalogue(Path(root))
    reactant = prepared["minimum"]
    mobile = prepared["mobile"]
    repeats = int(prepared["repeats"])
    found: list[np.ndarray] = [process["saddle"] for process in prepared["processes"]]
    search_calls = sum(int(row["force_calls"]) for row in prepared["rows"])
    searches = len(prepared["rows"])
    root = Path(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def thermal_rate() -> tuple[float, list[dict]]:
        thermal = select_thermal(prepared["processes"], temperature, window, min_barrier)
        rate = sum(float(item["rate"]) for item in thermal)
        return rate, thermal

    def publish(partial: bool) -> dict:
        rate, thermal = thermal_rate()
        confidence = old_scheme_confidence(repeats)
        reached = confidence >= confidence_target and rate > 0.0
        partial = partial or not reached
        total = int(prepared["minimize_calls"] + search_calls)
        reference = float(problem["reference"]["true_rate"])
        probabilities = distribution(problem["catalogue"]["probabilities"])
        summary = {
            "repetitions": 1,
            "nfailed": 0 if reached else 1,
            "true_rate": None,
            "uncertainty": None,
            "percent_uncertainty": None,
            "rate_accuracy": None,
            "catalogue_accuracy": None,
            "force_calls_per_simulated_time": None,
            "force_calls": None,
        }
        if reached:
            rates = rate_summary([rate])
            summary["true_rate"] = rates["true_rate"]
            summary["uncertainty"] = rates["uncertainty"]
            summary["percent_uncertainty"] = rates["percent_uncertainty"]
            summary["rate_accuracy"] = rate_accuracy(float(rate), reference)
            channels = [(item.get("channel"), float(item["rate"])) for item in thermal]
            summary["catalogue_accuracy"] = catalogue_accuracy(rate_distribution(channels), probabilities)
            summary["force_calls"] = total
            summary["force_calls_per_simulated_time"] = force_calls_per_simulated_time(
                total, simulated_time_seconds(rate, int(problem["sample"]["products"]))
            )
        comments = [
            f"# rep 1 force_calls {total} total_searches {searches} repeats {repeats} "
            f"rate {rate:.10g} confidence {confidence:.10g} reached {int(reached)}",
            "# counter harness. The catalogue stays on the minimized reactant.",
            f"# Prefactor {prefactor:g} is the eOn AKMC default_value. Temperature is the problem value.",
            "# Old-scheme confidence is 1 - 1/repeats. The link test is unchanged.",
            "# pyKMC uses the r2549 Aluminum EAM through the harness. "
            "The eOn AKMC row uses the client's built-in eam_al and a client counter.",
        ]
        output.write_text(
            amd_dat(
                summary,
                template=template,
                problem=problem,
                workers=workers,
                partial=partial,
                comments=comments,
            ),
            encoding="utf-8",
        )
        record = _start(
            template,
            problem,
            name="rep-1",
            calls=total,
            success=reached,
            failure=None if reached else "divergence",
            rate=rate,
            confidence=confidence,
            repeats=repeats,
            total_searches=searches,
        )
        if not reached:
            record["reasons"] = ["divergence"]
        _write_v2(output, template, problem, [record], cap, prepared)
        summary["reached"] = reached
        return summary

    while searches < max_searches:
        rate, _thermal = thermal_rate()
        if old_scheme_confidence(repeats) >= confidence_target and rate > 0.0:
            break
        if int(prepared["minimize_calls"] + search_calls) >= cap:
            break
        room = min(max(1, int(workers)), max_searches - searches)
        print(
            f"{template['id']} searches {searches} repeats {repeats} "
            f"confidence {old_scheme_confidence(repeats):.4f} rate {rate:.6g}",
            flush=True,
        )
        judged = _batch(
            root, template, prepared, number=searches + 1, room=room, climb_cap=climb_cap, workers=workers
        )
        for record in judged:
            linked = _link_one(
                prepared, record, barrier_max=barrier_max, connect_rmsd=connect_rmsd, found=found
            )
            searches += 1
            search_calls += int(linked["force_calls"])
            ident = _match_process(linked, prepared["processes"], connect_rmsd, prepared["box"], prepared["surface"]["pbc"])
            product = None if linked.get("product") is None else np.asarray(linked["product"], dtype=np.float64)
            if ident is not None:
                repeats += 1
            elif linked.get("eon_connected") and linked.get("barrier") and product is not None:
                if product.shape == reactant.shape and mobile_rmsd(product, reactant, mobile) < tolerance:
                    ident = None
                else:
                    channel = None
                    if product.shape == reactant.shape:
                        channel = match_product(product, products, mobile, tolerance)
                    ident = len(prepared["processes"]) + 1
                    barrier = float(linked["barrier"])
                    process_rate = prefactor * np.exp(-barrier / (temperature * KELVIN_TO_EV))
                    saddle = np.asarray(linked["saddle_mask"], dtype=np.float64).reshape(-1, 3)
                    prepared["processes"].append(
                        {
                            "id": ident,
                            "barrier": barrier,
                            "rate": float(process_rate),
                            "channel": channel,
                            "saddle": saddle,
                            "product": product,
                        }
                    )
                    found.append(saddle)
            prepared["rows"].append(
                {
                    "force_calls": int(linked["force_calls"]),
                    "process_id": ident,
                    "barrier": linked.get("barrier"),
                }
            )
            print(
                f"{template['id']} search {searches} calls {linked['force_calls']} "
                f"barrier {linked.get('barrier')} process {ident} repeats {repeats}",
                flush=True,
            )
            _strip(linked)
        prepared["repeats"] = repeats
        _checkpoint(prepared, prepared["rows"], repeats)
        publish(partial=True)
    summary = publish(partial=False)
    return {
        "force_calls": summary.get("force_calls"),
        "nfailed": summary["nfailed"],
        "starts": 1,
        "template": template["id"],
        "problem": problem["id"],
    }
