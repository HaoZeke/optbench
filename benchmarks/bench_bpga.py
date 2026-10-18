"""BPGA and PyBCGA global optimization, one class per template.

The default grid is seed ``0`` and start ``0``. ``asv check`` imports
this module and does not run a search. ``OPTBENCH_ASV_FULL=1`` selects
every start index the problem file states. The saved tracks are written
from the campaign records by ``write_results``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "0"

TRACKS = (
    "time_search",
    "track_force_calls",
    "track_success",
    "track_failure_cap",
    "track_failure_wrong_stationary",
    "track_failure_divergence",
    "track_failure_crash",
)


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Bpga" + "".join(part.capitalize() for part in slug.split("-"))


def selected_indices(count: int) -> list[str]:
    """Start indices on the grid. Full when ``OPTBENCH_ASV_FULL`` is set."""
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "bpga":
            continue
        sample = 1
        problem_id = str(template["problem"])
        if os.environ.get("OPTBENCH_ASV_FULL") == "1":
            from optbench.campaign import resolve_problem

            stated = (resolve_problem(ROOT, problem_id).get("sample") or {}).get("starts")
            sample = int(stated) if stated not in (None, "", 0, "0") else 1
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": problem_id,
                "cap": int(template["failure_cap"]),
                "n": sample,
            }
        )
    return found


class _BpgaSearch(TrackedSearch):
    """Track methods read the record ``time_search`` stored."""

    timeout = 7200

    def setup(self, seed, start):
        self.record = None

    def time_search(self, seed, start):
        from optbench.campaign import resolve_problem
        from optbench.runners.bpga_go import _job_from_template, _load_starts, _reference_for, build_bpga, pybcga_source, search_job
        from optbench.campaign import load_template

        if str(seed) != SEED:
            raise ValueError(f"seed {seed} is not {SEED}")
        template = load_template(ROOT, self.template_id)
        problem = resolve_problem(ROOT, self.problem_id)
        loaded = _load_starts(ROOT, problem, None, Path("/tmp/optbench-asv-bpga-geo"), ROOT / "unused.dat")
        index = int(start)
        name, coords = loaded[index]
        reference = _reference_for(problem)
        paths = {"work": f"/tmp/optbench-asv-bpga/{template['id'].replace('/', '-')}"}
        if template["bpga"]["method"] == "pybcga":
            paths["pybcga"] = str(pybcga_source())
        else:
            paths["bpga_ga"] = str(build_bpga())
        self.record = search_job(_job_from_template(template, problem, (index, name, coords), reference, paths))


def _benchmark(spec: dict):
    bound = spec

    class _Benchmark(_BpgaSearch):
        param_names = ["seed", "start"]
        params = [[SEED], selected_indices(spec["n"])]
        timeout = 7200
        cap = bound["cap"]
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]
        spec = bound

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)


def _version(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _quote(value: str) -> str:
    return repr(str(value))


def write_results(root: Path, commit: str, *, started_at: int | None = None) -> Path:
    """asv tracks for finished BPGA campaigns. Partial rows are left out.

    The result commit is the commit that contains ``templates/bpga``.
    ``time_search`` stores the measured per-start seconds.
    """
    campaign = root / "results" / "campaign"
    groups: dict[str, list[dict]] = {}
    for path in sorted(campaign.glob("*.starts.jsonl")):
        dat = campaign / (path.name[: -len(".starts.jsonl")] + ".dat")
        if not dat.is_file():
            continue
        fields = {}
        for line in dat.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            key, _, value = line.partition(" ")
            fields[key] = value.strip()
        template = fields.get("template") or ""
        if not template.startswith("bpga/"):
            continue
        if fields.get("partial", "").lower() == "true":
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"{path} has no starts")
        groups[template] = rows
    if not groups:
        raise ValueError(f"{campaign} has no finished bpga campaigns")
    machine = json.loads((root / "results" / "asv" / "cursor" / "machine.json").read_text(encoding="utf-8"))
    params = {
        "arch": machine["arch"],
        "cpu": machine["cpu"],
        "machine": machine["machine"],
        "num_cpu": machine["num_cpu"],
        "os": machine["os"],
        "ram": machine["ram"],
    }
    now = int(time.time() * 1000) if started_at is None else int(started_at)
    columns = ["result", "params", "version", "started_at", "duration"]
    results = {}
    for template, rows in groups.items():
        class_name = _class_name(template)
        calls = [float(row["force_calls"]) for row in rows]
        success = [1.0 if row["success"] else 0.0 for row in rows]
        failure = {name: [] for name in ("cap", "wrong_stationary", "divergence", "crash")}
        seconds = []
        for row in rows:
            label = row.get("failure")
            for name in failure:
                failure[name].append(1.0 if label == name else 0.0)
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
        seeds = [_quote(SEED)]
        starts = [_quote(str(row["start"])) for row in rows]
        duration = sum(seconds)
        for method in TRACKS:
            key = f"bench_bpga.{class_name}.{method}"
            results[key] = [payload[method], [seeds, starts], _version(key), now, duration]
    document = {
        "commit_hash": commit,
        "env_name": "uv-pip-py3.12",
        "date": now,
        "params": params,
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }
    dest = root / "results" / "asv" / "cursor" / f"{commit[:8]}-uv-pip-py3.12.json"
    dest.write_text(json.dumps(document, separators=(", ", ": ")), encoding="utf-8")
    return dest
