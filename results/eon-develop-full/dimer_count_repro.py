"""Reproduce the dimer client/harness gap on binary 7095b23.

SaddleSearchJob writes ``this->pot->forceCallCounter`` as
``total_force_calls``. With ``Main.parallel`` true, ImprovedDimer clones
``ExtPot`` for the forward image. Those evaluations reach the harness and
``PotRegistry::total_force_calls``, and they do not increment the primary
counter. NEB writes the registry total, so a parallel NEB still matches.

The stored develop caches were measured by the harness at optbench
06efc6dd, which left ``parallel`` at the client default. ``run_case`` now
sets it false, and one instance records every ``force()`` in both counters.

This script does not change eOn. It runs one stored start both ways.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = Path("/tmp/optbench-asv/harness-1/7095b23ac17d49f434a5e08ccb072897ceca698e")


def _fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        value, key = line.split(None, 1)
        fields[key] = value
    return fields


def _search(template_name: str, kind: str, start: str, parallel: bool) -> dict:
    from benchmarks.bench_eon_develop import EON_COMMIT, _Bridge, _client, _client_files
    from optbench.campaign import format_ini, load_template
    from optbench.eon_jobs import _execute
    from optbench.potserver import remote_count

    template = load_template(ROOT, template_name)
    potential = template["ini"]["Potential"]["potential"]
    cap = int(template["failure_cap"])
    work = Path(tempfile.mkdtemp(prefix="optbench-count-"))
    try:
        with _Bridge(potential, cap) as (server, wrapper):
            copies, texts = _client_files(kind, start, work)
            sections = json.loads(json.dumps(template["ini"]))
            sections.setdefault("Main", {})["random_seed"] = 1
            sections["Main"]["parallel"] = bool(parallel)
            sections["Potential"] = {"potential": "ext_pot", "ext_pot_path": str(wrapper)}
            found = _execute(
                (
                    _client(),
                    format_ini(sections),
                    str(work / "run"),
                    copies,
                    texts,
                    "total_force_calls",
                )
            )
            harness, verify = remote_count(server.host, server.port)
            dat_path = work / "run" / "results.dat"
            dat = dat_path.read_text(encoding="utf-8") if dat_path.is_file() else ""
        fields = _fields(dat)
        return {
            "template": template_name,
            "kind": kind,
            "start": start,
            "parallel": bool(parallel),
            "eon_commit": EON_COMMIT,
            "harness": int(harness),
            "verify_calls": int(verify),
            "client_total_force_calls": found.get("raw"),
            "client_reason": found.get("reason"),
            "force_calls_saddle": fields.get("force_calls_saddle"),
            "results_total_force_calls": fields.get("total_force_calls"),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _lowest(start: str, parallel: bool) -> dict:
    from benchmarks.bench_eon_develop import EON_COMMIT, _Bridge, _client, _extract, kept_lowest
    from optbench.campaign import format_ini, load_template
    from optbench.eon_jobs import _lowest_once, _numpy_direction, _stored
    from optbench.eon_parse import abs_overlap, flat_floats
    from optbench.potserver import remote_count, remote_reset

    template = load_template(ROOT, "eon/develop-low-lj38-dimer")
    potential = template["ini"]["Potential"]["potential"]
    cap = int(template["failure_cap"])
    limit = int(template.get("iteration_limit", 1000))
    threshold = float(template.get("overlap", 0.99))
    work = Path(tempfile.mkdtemp(prefix="optbench-count-low-"))
    try:
        archive = _stored(ROOT, "website/lowest-eigenvector/lj38/coords-con.tar.gz")
        con = _extract(archive, f"coords-con/{start}.con", work / "src.con")
        mode = _extract(archive, f"coords-con/{start}_mode", work / "ref.mode")
        reference = flat_floats(mode.read_text(encoding="utf-8", errors="replace"))
        direction = _numpy_direction()
        sections = json.loads(json.dumps(template["ini"]))
        sections.setdefault("Main", {})["random_seed"] = 1
        sections["Main"]["parallel"] = bool(parallel)
        sections["Potential"] = {"potential": "ext_pot"}
        records = []
        with _Bridge(potential, cap) as (server, wrapper):
            sections["Potential"]["ext_pot_path"] = str(wrapper)
            for attempt in range(1, limit + 1):
                sections_k = json.loads(json.dumps(sections))
                sections_k.setdefault("Dimer", {})["rotations_max"] = attempt
                remote_reset(server.host, server.port)
                found = _lowest_once(
                    _client(),
                    format_ini(sections_k),
                    str(work / "run"),
                    str(con),
                    direction,
                )
                calls, _verify = remote_count(server.host, server.port)
                overlap = None
                if found["mode_text"]:
                    try:
                        overlap = abs_overlap(flat_floats(found["mode_text"]), reference)
                    except ValueError:
                        overlap = None
                records.append(
                    {
                        "attempt": attempt,
                        "harness": int(calls),
                        "client": found.get("raw"),
                        "overlap": overlap,
                    }
                )
                if overlap is not None and overlap >= threshold:
                    break
            dat_path = work / "run" / "results.dat"
            dat = dat_path.read_text(encoding="utf-8") if dat_path.is_file() else ""
        index = kept_lowest([(row["harness"], row["overlap"]) for row in records], threshold)
        chosen = None if index is None else records[index]
        fields = _fields(dat)
        return {
            "template": "eon/develop-low-lj38-dimer",
            "kind": "lowest",
            "start": start,
            "parallel": bool(parallel),
            "eon_commit": EON_COMMIT,
            "attempts": len(records),
            "kept_attempt": None if chosen is None else chosen["attempt"],
            "harness": None if chosen is None else chosen["harness"],
            "client_total_force_calls": None if chosen is None else chosen["client"],
            "overlap": None if chosen is None else chosen["overlap"],
            "force_calls_saddle": fields.get("force_calls_saddle"),
            "results_total_force_calls": fields.get("total_force_calls"),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _stored_cache(template: str, start: str) -> dict | None:
    path = CACHE / template / "passes-0" / "1" / f"{start}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = [
        _search("eon/develop-ts-lj38-dimer-accept-energy", "ts-lj38", "0000", True),
        _search("eon/develop-ts-lj38-dimer-accept-energy", "ts-lj38", "0000", False),
        _lowest("0000", True),
        _lowest("0000", False),
    ]
    report = {
        "stored_ts_lj38_dimer_0000": _stored_cache(
            "eon-develop-ts-lj38-dimer-accept-energy", "0000"
        ),
        "stored_low_lj38_dimer_0000": _stored_cache("eon-develop-low-lj38-dimer", "0000"),
        "runs": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    out = Path(__file__).resolve().parent / "dimer-count-repro.json"
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
