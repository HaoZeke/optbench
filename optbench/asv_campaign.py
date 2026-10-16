"""Turn an ``asv run`` result file into campaign ``benchmark.dat`` files.

The numbers are the track values asv stored. Nothing here reruns eonclient.
``track_force_calls`` is the client ``total_force_calls`` (energy and
gradient), except the AKMC classes, whose track is ``sad-fcs + mins-fcs``.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
from pathlib import Path

STEMS = {
    "EonDevelopTsLj38Lanczos": "eon-develop-ts-lj38-lanczos",
    "EonDevelopTsLj38Dimer": "eon-develop-ts-lj38-dimer",
    "EonDevelopTsPtLanczos": "eon-develop-ts-pt-lanczos",
    "EonDevelopTsPtDimer": "eon-develop-ts-pt-dimer",
    "EonDevelopDeLj38Ew": "eon-develop-de-lj38-ew",
    "EonDevelopDeLj38Oci": "eon-develop-de-lj38-oci",
    "EonDevelopDeLj38Cap": "eon-develop-de-lj38-cap",
    "EonDevelopDePtEw": "eon-develop-de-pt-ew",
    "EonDevelopDePtOci": "eon-develop-de-pt-oci",
    "EonDevelopDe2ts": "eon-develop-de-2ts",
    "EonDevelopDe2tsOci": "eon-develop-de-2ts-oci",
    "EonDevelopDe2tsEw": "eon-develop-de-2ts-ew",
    "EonDevelopLowLj38Dimer": "eon-develop-low-lj38-dimer",
    "EonDevelopLowLj38Lanczos": "eon-develop-low-lj38-lanczos",
    "EonDevelopSaddlePtDimer": "eon-develop-saddle-pt-dimer",
    "EonDevelopSaddlePtLanczos": "eon-develop-saddle-pt-lanczos",
}


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _reason(success, cause) -> str | None:
    if _missing(cause):
        return None
    code = int(cause)
    if code == 0 or success == 1:
        return "0"
    if code == 100:
        return "no-overlap"
    if code == 101:
        return "incomplete"
    if code == 102:
        return None
    if code == 199:
        return "other"
    return str(code)


def _track(payload: dict, benchmark: str) -> list:
    row = payload["results"][benchmark]
    values = row[0]
    if values is None:
        raise ValueError(f"{benchmark} has no result list")
    return list(values)


def _pairs(spec: dict) -> list[tuple[int, str]]:
    return [(int(seed), str(start)) for seed, start in itertools.product(spec["seeds"], spec["starts"])]


def _append(path: Path, lines: list[str]) -> None:
    text = path.read_text(encoding="utf-8").rstrip("\n")
    path.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def write_one(
    root: Path,
    spec: dict,
    tracks: dict[str, list],
    output: Path,
    *,
    asv_commit: str,
    workers: int,
) -> dict:
    """Write one benchmark.dat and its ``.starts`` from aligned track lists."""
    from .campaign import load_template
    from .eon_akmc import _write_saddle
    from .eon_jobs import _finish, _pair_count

    template = load_template(root, spec["template"])
    pairs = _pairs(spec)
    count = len(pairs)
    for name, values in tracks.items():
        if len(values) != count:
            raise ValueError(f"{spec['cls']} {name} has {len(values)} values, expected {count}")
    calls = tracks["force_calls"]
    success = tracks["success"]
    cause = tracks["failure_cause"]
    seconds = tracks.get("seconds")
    if spec["kind"] == "akmc":
        rows = []
        for index, (seed, _start) in enumerate(pairs):
            raw = None if _missing(calls[index]) else int(round(float(calls[index])))
            reached = (not _missing(success[index])) and float(success[index]) == 1.0
            def optional(name: str):
                value = tracks[name][index]
                return None if _missing(value) else float(value)

            rows.append(
                {
                    "force_calls": raw,
                    "fraction_connected": optional("fraction_connected"),
                    "passes": None,
                    "qualifying": None,
                    "reached": reached,
                    "repetition": index + 1,
                    "seconds": None if seconds is None or _missing(seconds[index]) else float(seconds[index]),
                    "seed": seed,
                    "total_searches": optional("total_searches"),
                    "unique_saddles": optional("unique_saddles"),
                }
            )
        summary = _write_saddle(template, rows, output, workers)
    else:
        rows = []
        for index, (seed, start) in enumerate(pairs):
            raw = None if _missing(calls[index]) else int(round(float(calls[index])))
            row = {
                "count_key": "total_force_calls",
                "geometry": start,
                "raw": raw,
                "reason": _reason(None if _missing(success[index]) else float(success[index]), cause[index]),
                "seed": seed,
            }
            if seconds is not None and not _missing(seconds[index]):
                row["seconds"] = float(seconds[index])
            if "extrema" in tracks and not _missing(tracks["extrema"][index]):
                row["extrema"] = int(round(float(tracks["extrema"][index])))
            rows.append(row)
        extras = _pair_count(rows) if spec["kind"] == "de-2ts" else None
        summary = _finish(template, rows, output, 1, extras)
        _append(
            output,
            [
                "counter total_force_calls",
                "counter_file results.dat",
                "counter_note energy and gradient evaluations, not a Python harness wrap",
            ],
        )
    _append(
        output,
        [
            f"asv_benchmark benchmarks.bench_eon_develop.{spec['cls']}",
            f"asv_commit {asv_commit}",
        ],
    )
    return summary


def _class_result(payload: dict, cls: str, method: str) -> list | None:
    name = f"benchmarks.bench_eon_develop.{cls}.{method}"
    if name not in payload["results"]:
        return None
    values = payload["results"][name][0]
    if values is None:
        return None
    return list(values)


def write_results(root: Path, result_path: Path, campaign: Path, *, workers: int = 8) -> list[str]:
    """Write a dat for every develop class present in one asv result file."""
    from benchmarks.bench_eon_develop import _specs

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    commit = str(payload["commit_hash"])
    written: list[str] = []
    campaign.mkdir(parents=True, exist_ok=True)
    for spec in _specs():
        calls = _class_result(payload, spec["cls"], "track_force_calls")
        success = _class_result(payload, spec["cls"], "track_success")
        cause = _class_result(payload, spec["cls"], "track_failure_cause")
        if calls is None or success is None or cause is None:
            continue
        tracks = {"force_calls": calls, "success": success, "failure_cause": cause}
        seconds = _class_result(payload, spec["cls"], "time_run")
        if seconds is not None:
            tracks["seconds"] = seconds
        if spec["kind"] == "akmc":
            for name in ("unique_saddles", "fraction_connected", "total_searches"):
                values = _class_result(payload, spec["cls"], f"track_{name}")
                if values is None:
                    raise ValueError(f"{spec['cls']} is missing track_{name}")
                tracks[name] = values
        if spec["kind"] == "de-2ts":
            extrema = _class_result(payload, spec["cls"], "track_extrema")
            if extrema is None:
                raise ValueError(f"{spec['cls']} is missing track_extrema")
            tracks["extrema"] = extrema
        stem = STEMS[spec["cls"]]
        write_one(root, spec, tracks, campaign / f"{stem}.dat", asv_commit=commit, workers=workers)
        written.append(stem)
    return written


def copy_asv_tree(asv_results: Path, dest: Path) -> list[Path]:
    """Copy machine.json and result JSON into a tracked directory."""
    copied: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(asv_results.rglob("*.json")):
        target = dest / path.relative_to(asv_results)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied
