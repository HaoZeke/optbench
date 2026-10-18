"""Write asv tracks from a finished asebh campaign.

The force counts and the wall times are the campaign records. A partial
campaign is left out. The result commit is a commit that already contains
the templates.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


TRACKS = (
    "time_search",
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)

FAILURES = ("cap", "wrong_stationary", "divergence", "crash")


def class_name(template_id: str) -> str:
    """Benchmark class for one template id. Matches ``benchmarks.bench_asebh``."""
    rest = template_id.split("/", 1)[1]
    parts = [part.capitalize() for part in rest.replace("-", " ").split()]
    return "Asebh" + "".join(parts)


def _quote(value: str) -> str:
    return repr(str(value))


def _version(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _complete(dat: Path) -> bool:
    if not dat.is_file():
        return False
    for line in dat.read_text(encoding="utf-8").splitlines():
        if line.strip() == "partial true":
            return False
    return True


def finished_groups(root: Path) -> dict[str, list[dict]]:
    """Campaign starts whose dat file is not marked partial."""
    campaign = root / "results" / "campaign"
    groups: dict[str, list[dict]] = {}
    for path in sorted(campaign.glob("asebh-*.starts.jsonl")):
        dat = campaign / (path.name[: -len(".starts.jsonl")] + ".dat")
        if not _complete(dat):
            continue
        template = None
        for line in dat.read_text(encoding="utf-8").splitlines():
            if line.startswith("template "):
                template = line.split(None, 1)[1].strip()
        if not template:
            raise ValueError(f"{dat} has no template")
        groups[template] = _records(path)
    return groups


def document(groups: dict[str, list[dict]], *, commit: str, machine: dict, started_at: int) -> dict:
    """One asv result document. ``groups`` maps a template id to its starts, in order."""
    params = {
        "arch": machine["arch"],
        "cpu": machine["cpu"],
        "machine": machine["machine"],
        "num_cpu": machine["num_cpu"],
        "os": machine["os"],
        "ram": machine["ram"],
        "python": "3.12",
    }
    results = {}
    for template, rows in groups.items():
        if not rows:
            raise ValueError(f"{template} has no starts")
        name = class_name(template)
        seeds = [_quote(str(rows[0]["seed"]))]
        starts = [_quote(str(row["start"])) for row in rows]
        calls = [float(row["force_calls"]) for row in rows]
        success = [1.0 if row["success"] else 0.0 for row in rows]
        failure = {label: [] for label in FAILURES}
        seconds = []
        for row in rows:
            label = row.get("failure")
            for name_ in failure:
                failure[name_].append(1.0 if label == name_ else 0.0)
            seconds.append(float(row["seconds"]))
        payload = {
            "time_search": seconds,
            "track_force_calls": calls,
            "track_success": success,
            "track_failure_cap": failure["cap"],
            "track_failure_wrong_stationary": failure["wrong_stationary"],
            "track_failure_divergence": failure["divergence"],
            "track_failure_crash": failure["crash"],
        }
        duration = sum(seconds)
        for method in TRACKS:
            key = f"bench_asebh.{name}.{method}"
            results[key] = [payload[method], [seeds, starts], _version(key), started_at, duration]
    return {
        "commit_hash": commit,
        "env_name": "uv-pip-py3.12",
        "date": started_at,
        "params": params,
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": ["result", "params", "version", "started_at", "duration"],
        "results": results,
        "durations": {},
        "version": 2,
    }


def write_finished(root: Path, commit: str, *, started_at: int | None = None) -> Path | None:
    """Write asv tracks for finished asebh campaigns. Returns None when none are finished."""
    groups = finished_groups(root)
    if not groups:
        return None
    machine = json.loads((root / "results" / "asv" / "cursor" / "machine.json").read_text(encoding="utf-8"))
    now = int(time.time() * 1000) if started_at is None else int(started_at)
    body = document(groups, commit=commit, machine=machine, started_at=now)
    dest = root / "results" / "asv" / "cursor" / f"{commit[:8]}-uv-pip-py3.12.json"
    dest.write_text(json.dumps(body, separators=(", ", ": ")), encoding="utf-8")
    return dest
