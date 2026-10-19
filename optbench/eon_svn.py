"""Write harness campaign files for the eOn SVN r2549 templates.

The start list is the campaign grid on ``benchmarks.bench_eon_svn``.
A file with fewer starts than that grid sets ``partial true``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from optbench.failures import FAILURES
from optbench.score import score_v2

ROOT = Path(__file__).resolve().parents[1]


def stem_for(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return f"eon-svn-r2549-{slug}"


def _spec_map() -> dict[str, dict]:
    from benchmarks.bench_eon_svn import SPECS

    return {spec["template"]: spec for spec in SPECS}


def _job(spec: tuple) -> dict:
    template_name, kind, seed, start, passes, workers = spec
    from benchmarks.bench_eon_svn import run_case

    try:
        return run_case(template_name, kind, int(seed), str(start), passes=int(passes), workers=int(workers))
    except Exception as exc:
        return {
            "force_calls": 0.0,
            "success": False,
            "failure": "crash",
            "error": f"{type(exc).__name__}: {exc}",
            "seed": str(seed),
            "start": str(start),
            "seconds": 0.0,
            "counter": "harness",
            "template": template_name,
        }


def _key(row: dict) -> tuple[str, str]:
    return str(row.get("seed")), str(row.get("start"))


def _has_energy(row: dict) -> bool:
    energy = row.get("energy")
    return isinstance(energy, (int, float)) and not isinstance(energy, bool)


def endpoint_path(stem: str, root: Path | None = None) -> Path:
    """Sidecar of final coordinates, one JSON object per start."""
    return (root or ROOT) / "results" / "campaign" / f"{stem}.endpoints.jsonl"


def _endpoint_row(row: dict) -> dict | None:
    positions = row.get("positions")
    if not isinstance(positions, list) or len(positions) < 3 or len(positions) % 3 != 0:
        return None
    if not _has_energy(row):
        return None
    return {
        "energy": float(row["energy"]),
        "positions": [float(value) for value in positions],
        "seed": str(row.get("seed")),
        "start": str(row.get("start")),
    }


def _load_endpoints(path: Path) -> dict[tuple[str, str], dict]:
    found = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kept = _endpoint_row(row)
        if kept is None:
            continue
        found[_key(kept)] = kept
    return found


def _load_done(
    path: Path,
    *,
    need_energy: bool = False,
    endpoints: dict[tuple[str, str], dict] | None = None,
) -> dict[tuple[str, str], dict]:
    found = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("force_calls") is None or "success" not in row:
            continue
        key = _key(row)
        if need_energy and (not _has_energy(row) or endpoints is None or key not in endpoints):
            continue
        found[key] = row
    return found


def _public_row(row: dict) -> dict:
    kept = {
        "schema": "optbench.start/v2",
        "start": str(row.get("start")),
        "seed": str(row.get("seed")),
        "force_calls": int(row["force_calls"]) if float(row["force_calls"]).is_integer() else float(row["force_calls"]),
        "counter": "harness",
        "comparable": True,
        "success": bool(row["success"]),
        "template": row.get("template", ""),
        "problem": row.get("problem", ""),
        "code": row.get("code", "Eon svn"),
        "code_version": row.get("code_version", "r2549"),
        "commit": row.get("commit", "r2549"),
        "tuning": row.get("tuning", "default"),
        "seconds": float(row.get("seconds") or 0.0),
        "call_unit": "total",
    }
    if not kept["success"]:
        kept["failure"] = row.get("failure") or "crash"
    if _has_energy(row):
        kept["energy"] = float(row["energy"])
    if row.get("reasons"):
        kept["reasons"] = [str(item) for item in row["reasons"]]
    for name in (
        "client_calls",
        "client_reason",
        "hessian_index",
        "overlap",
        "k",
        "extrema",
        "fraction_connected",
        "passes",
        "qualifying",
        "total_searches",
        "unique_saddles",
        "error",
    ):
        if row.get(name) not in (None, ""):
            kept[name] = row[name]
    return kept


def _dat(template: dict, rows: list[dict], *, partial: bool, cap: int) -> str:
    records = []
    for row in rows:
        item = {"force_calls": row["force_calls"], "success": row["success"]}
        if not row["success"]:
            item["failure"] = row.get("failure") or "crash"
        records.append(item)
    summary = score_v2(records, cap)
    calls = [float(row["force_calls"]) for row in rows]
    lines = [
        f"# eOn SVN r2549. HarnessCounter through ext_pot. Kernel {template['ini']['Potential']['potential']}.",
        f"# {template.get('note', '').splitlines()[0] if template.get('note') else ''}".rstrip(),
    ]
    if any(_has_energy(row) for row in rows):
        lines.append(
            "# Final energy is the clean lj/cut value of min.con. "
            "Coordinates are the endpoint sidecar. "
            "A dissociated cluster is divergence."
        )
    lines.extend([
        "counter harness",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template.get('algorithm', '')}",
        f"tuning {template.get('tuning', 'default')}",
        f"template {template['id']}",
        f"problem {template['problem']}",
        "reducer harness",
        f"cap {cap}",
        f"n {summary['n']}",
        f"par2_mean {summary['par2_mean']}",
        f"nfailed {summary['nfailed']}",
        f"mean_calls {sum(calls) / len(calls)}",
        f"median_calls {statistics.median(calls)}",
        f"min_calls {min(calls)}",
        f"max_calls {max(calls)}",
    ])
    for name in FAILURES:
        lines.append(f"n_{name} {summary[f'n_{name}']}")
    energies = [float(row["energy"]) for row in rows if _has_energy(row)]
    if len(energies) == len(rows) and energies:
        lines.append(f"energy_min {min(energies)}")
        lines.append(f"energy_max {max(energies)}")
    if partial:
        lines.append("partial true")
    return "\n".join(line for line in lines if not line.endswith(" ")) + "\n"


def _write_start(handle, endpoint_handle, raw: dict) -> dict:
    row = _public_row(raw)
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    if endpoint_handle is not None:
        endpoint = _endpoint_row(raw)
        if endpoint is not None:
            endpoint_handle.write(json.dumps(endpoint, sort_keys=True) + "\n")
            endpoint_handle.flush()
    return row


def run_template(template_id: str, *, workers: int = 1, limit: int = 0) -> dict:
    from optbench.campaign import load_template

    spec = _spec_map()[template_id]
    template = load_template(ROOT, template_id)
    if template["ini"]["Potential"]["potential"] == "morse_pt":
        from benchmarks.bench_eon_svn import _morse_lib

        _morse_lib()
    cap = int(template["failure_cap"])
    seeds = list(spec["campaign_seeds"])
    starts = list(spec["campaign_starts"])
    if limit:
        starts = starts[:limit]
        if spec["kind"] == "akmc":
            seeds = seeds[:limit]
    pairs = [(seed, start) for seed in seeds for start in starts]
    stem = stem_for(template_id)
    directory = ROOT / "results" / "campaign"
    directory.mkdir(parents=True, exist_ok=True)
    jsonl = directory / f"{stem}.starts.jsonl"
    noise = spec["kind"] == "noise"
    endpoints_file = endpoint_path(stem)
    stored_endpoints = _load_endpoints(endpoints_file) if noise else {}
    done = _load_done(jsonl, need_energy=noise, endpoints=stored_endpoints if noise else None)
    pending = [pair for pair in pairs if pair not in done]
    passes = int(spec.get("campaign_passes") or 0)
    inside = int(spec.get("workers") or 1) if spec["kind"] == "akmc" else 1
    pool_workers = 1 if spec["kind"] == "akmc" else max(1, workers)
    print(
        f"{template_id}: {len(done)} done, {len(pending)} pending, pool {pool_workers}",
        flush=True,
    )
    if pending:
        jobs = [
            (template_id, spec["kind"], seed, start, passes, inside)
            for seed, start in pending
        ]
        finished = 0

        def consume(raw: dict, handle, endpoint_handle) -> None:
            nonlocal finished
            row = _write_start(handle, endpoint_handle, raw)
            finished += 1
            print(
                f"  {finished}/{len(pending)} {row['start']} seed {row['seed']} "
                f"calls {row['force_calls']} success {row['success']} {row.get('failure') or ''}",
                flush=True,
            )

        with ExitStack() as stack:
            handle = stack.enter_context(jsonl.open("a", encoding="utf-8"))
            endpoint_handle = (
                stack.enter_context(endpoints_file.open("a", encoding="utf-8")) if noise else None
            )
            if pool_workers == 1:
                for job in jobs:
                    consume(_job(job), handle, endpoint_handle)
            else:
                with ProcessPoolExecutor(max_workers=pool_workers) as pool:
                    futures = [pool.submit(_job, job) for job in jobs]
                    for future in as_completed(futures):
                        consume(future.result(), handle, endpoint_handle)
    if noise:
        stored_endpoints = _load_endpoints(endpoints_file)
        done = _load_done(jsonl, need_energy=True, endpoints=stored_endpoints)
    else:
        done = _load_done(jsonl)
    ordered = []
    missing = []
    missing_endpoint = []
    for pair in pairs:
        row = done.get(pair)
        if row is None:
            missing.append(pair)
        else:
            ordered.append(row)
        if noise and pair not in stored_endpoints:
            missing_endpoint.append(pair)
    if noise and not limit and (missing or missing_endpoint):
        raise RuntimeError(
            f"{template_id} is missing {len(missing)} starts and {len(missing_endpoint)} endpoints"
        )
    partial = bool(missing) or len(ordered) < len(spec["campaign_seeds"]) * len(spec["campaign_starts"])
    if limit:
        partial = True
    if not ordered:
        raise RuntimeError(f"{template_id} produced no starts")
    if noise and not limit:
        lines = [json.dumps(stored_endpoints[pair], sort_keys=True) for pair in pairs]
        endpoints_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    text = _dat(template, ordered, partial=partial, cap=cap)
    (directory / f"{stem}.dat").write_text(text, encoding="utf-8")
    body = "\n".join(json.dumps(_public_row(row), sort_keys=True) for row in ordered) + "\n"
    jsonl.write_text(body, encoding="utf-8")
    print(text, flush=True)
    return {"template": template_id, "n": len(ordered), "partial": partial, "stem": stem}


_ASV_SKIP = {"machine.json", "benchmarks.json"}
_TRACKS = (
    "time_search",
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)


def _asv_documents(root: Path) -> list[tuple[Path, dict]]:
    directory = root / "results" / "asv"
    found = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.rglob("*.json")):
        if path.name in _ASV_SKIP or not path.is_file():
            continue
        found.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return found


def _eon_method_count(document: dict) -> int:
    results = document.get("results") or {}
    return sum(1 for key in results if str(key).startswith("bench_eon_svn."))


def _measurement_unchanged(stored, entry: list) -> bool:
    """Values, parameters, and version already match. Keep the old timestamp."""
    if not isinstance(stored, list) or len(stored) < 3 or len(entry) < 3:
        return False
    return stored[0] == entry[0] and stored[1] == entry[1] and stored[2] == entry[2]


def apply_method_pool(documents: list[dict], updates: dict[str, list]) -> tuple[list[int], dict[str, list]]:
    """Update each stored copy of a method. Return changed indexes and leftovers.

    A method that already sits in one or more result documents is replaced in
    every one of those documents. A method that sits in none of them is
    inserted into the document that already holds the most ``bench_eon_svn``
    methods. No method and no document is removed. Leftovers are methods that
    had nowhere to go because no document holds an eOn SVN method.
    """
    if not documents:
        return [], dict(updates)
    owners: dict[str, list[int]] = {}
    for index, document in enumerate(documents):
        results = document.get("results") or {}
        for key in updates:
            if key in results:
                owners.setdefault(key, []).append(index)
    richest = max(range(len(documents)), key=lambda index: (_eon_method_count(documents[index]), -index))
    richest_count = _eon_method_count(documents[richest])
    changed: set[int] = set()
    leftover: dict[str, list] = {}
    for key, entry in updates.items():
        indexes = owners.get(key)
        if not indexes:
            if richest_count == 0:
                leftover[key] = entry
                continue
            indexes = [richest]
        for index in indexes:
            document = documents[index]
            results = document.setdefault("results", {})
            stored = results.get(key)
            if stored == entry or _measurement_unchanged(stored, entry):
                continue
            if isinstance(stored, list) and len(stored) > len(entry):
                merged = list(stored)
                for slot, value in enumerate(entry):
                    merged[slot] = value
                results[key] = merged
            else:
                results[key] = entry
            changed.add(index)
    return sorted(changed), leftover


def _campaign_updates(root: Path, specs: list[dict]) -> dict[str, list]:
    """Asv entries for finished campaigns. Noise rows need energy and an endpoint."""
    updates: dict[str, list] = {}
    now = int(time.time() * 1000)
    for spec in specs:
        stem = stem_for(spec["template"])
        path = root / "results" / "campaign" / f"{stem}.starts.jsonl"
        dat = root / "results" / "campaign" / f"{stem}.dat"
        if not path.is_file() or not dat.is_file():
            continue
        if "partial true" in dat.read_text(encoding="utf-8"):
            continue
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(str(row["seed"]), str(row["start"]))] = row
        combos = [(str(seed), str(start)) for seed in spec["campaign_seeds"] for start in spec["campaign_starts"]]
        if any(combo not in rows for combo in combos):
            continue
        ordered = [rows[combo] for combo in combos]
        noise = spec["kind"] == "noise"
        if noise and any(not _has_energy(row) for row in ordered):
            continue
        if noise:
            endpoints = _load_endpoints(endpoint_path(stem, root))
            if any(combo not in endpoints for combo in combos):
                continue
        param_lists = [
            [repr(str(seed)) for seed in spec["campaign_seeds"]],
            [repr(str(start)) for start in spec["campaign_starts"]],
        ]
        columns = {
            "time_search": [float(row.get("seconds") or 0.0) for row in ordered],
            "track_force_calls": [float(row["force_calls"]) for row in ordered],
            "track_success": [1.0 if row["success"] else 0.0 for row in ordered],
            "track_failure_cap": [1.0 if row.get("failure") == "cap" else 0.0 for row in ordered],
            "track_failure_wrong_stationary": [1.0 if row.get("failure") == "wrong_stationary" else 0.0 for row in ordered],
            "track_failure_divergence": [1.0 if row.get("failure") == "divergence" else 0.0 for row in ordered],
            "track_failure_crash": [1.0 if row.get("failure") == "crash" else 0.0 for row in ordered],
        }
        duration = float(sum(columns["time_search"]))
        for method in _TRACKS:
            name = f"bench_eon_svn.{spec['cls']}.{method}"
            updates[name] = [columns[method], param_lists, "r2549", now, duration]
    return updates


def _reject_mismatched_params(loaded: list[tuple[Path, dict]], updates: dict[str, list]) -> None:
    for path, document in loaded:
        results = document.get("results") or {}
        for key, entry in updates.items():
            stored = results.get(key)
            if not isinstance(stored, list) or len(stored) < 2:
                continue
            if stored[1] != entry[1]:
                raise RuntimeError(f"{path} {key} parameters differ from the campaign grid")


def _write_asv_file(path: Path, document: dict) -> None:
    original = path.read_text(encoding="utf-8")
    if original.startswith("{\n"):
        rendered = json.dumps(document, indent=2) + "\n"
    else:
        rendered = json.dumps(document)
        if original.endswith("\n"):
            rendered += "\n"
    path.write_text(rendered, encoding="utf-8")


def _save_new_asv(root: Path, commit: str, updates: dict[str, list]) -> Path:
    """First eOn SVN result file. Later runs update that file per method."""
    from asv.results import Results

    machine = json.loads((root / "results" / "asv" / "cursor" / "machine.json").read_text(encoding="utf-8"))
    params = {
        "arch": machine.get("arch", "x86_64"),
        "cpu": machine.get("cpu", ""),
        "machine": "cursor",
        "num_cpu": str(machine.get("num_cpu", "8")),
        "os": machine.get("os", ""),
        "ram": str(machine.get("ram", "")),
        "python": "3.12",
    }
    now = int(time.time() * 1000)
    result = Results(params, {}, commit, now, "3.12", "uv-pip-py3.12", {})
    for name, entry in updates.items():
        result._results[name] = entry[0]
        result._benchmark_params[name] = entry[1]
        result._benchmark_version[name] = entry[2]
        result._started_at[name] = entry[3]
        result._duration[name] = entry[4]
        result._samples[name] = None
        result._stats[name] = None
    result.save(str(root / "results" / "asv"))
    return root / "results" / "asv" / "cursor" / result._filename


def write_asv(commit: str) -> Path:
    """Pool finished SVN campaigns into the result files that already store them.

    Each method is updated in every file that has it. A method with no copy
    is added to the file that already holds the most eOn SVN methods. Existing
    result files stay on disk.
    """
    from benchmarks.bench_eon_svn import SPECS

    updates = _campaign_updates(ROOT, SPECS)
    if not updates:
        raise RuntimeError("no finished eOn SVN campaign to store")
    loaded = _asv_documents(ROOT)
    _reject_mismatched_params(loaded, updates)
    documents = [document for _path, document in loaded]
    changed, leftover = apply_method_pool(documents, updates)
    written: list[Path] = []
    for index in changed:
        path, _original = loaded[index]
        _write_asv_file(path, documents[index])
        written.append(path)
    if leftover:
        written.append(_save_new_asv(ROOT, commit, leftover))
    if written:
        classes = len({key.rsplit(".", 1)[0] for key in updates})
        print(f"wrote {written[0]} classes {classes} files {len(written)}", flush=True)
        return written[0]
    holders = [
        path
        for path, document in loaded
        if any(key in (document.get("results") or {}) for key in updates)
    ]
    if not holders:
        raise RuntimeError("no finished eOn SVN campaign to store")
    print(f"asv already matches {holders[0]}", flush=True)
    return holders[0]


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    workers = 8
    limit = 0
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if not args or args[0] in {"-h", "--help"}:
        raise SystemExit("usage: python -m optbench.eon_svn run TEMPLATE | run-all | write-asv COMMIT")
    command = args[0]
    if command == "run":
        run_template(args[1], workers=workers, limit=limit)
        return
    if command == "run-all":
        for template_id in _spec_map():
            run_template(template_id, workers=workers, limit=limit)
        return
    if command == "write-asv":
        write_asv(args[1])
        return
    raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
