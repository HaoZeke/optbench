"""Write one saddle-search campaign in the harness catalog format.

The ``.dat`` comment for each repetition carries ``unique_saddles``,
``fraction_connected``, and ``total_searches``. ``<stem>.endpoints.npz``
is format 2. The starts file points at that archive.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import date
from pathlib import Path

from .endpoints import EndpointArchive
from .score import score_v2
from .v2 import collect_provenance, write_starts


def write_saddle_campaign(
    output: Path,
    template: dict,
    problem: dict,
    records: list[dict],
    comments: list[str],
    archive: EndpointArchive,
    *,
    search_cap: int,
    goal: int,
    repetitions: int,
    failed_reps: int,
    note: str,
) -> dict | None:
    """Write the dat, the starts, the endpoint archive, and the v2 document."""
    totals = []
    uniques = []
    fractions = []
    searches = []
    for line in comments:
        parts = line.split()
        if "force_calls" not in parts:
            continue
        totals.append(int(parts[parts.index("force_calls") + 1]))
        uniques.append(int(parts[parts.index("unique_saddles") + 1]))
        fractions.append(float(parts[parts.index("fraction_connected") + 1]))
        searches.append(int(parts[parts.index("total_searches") + 1]))
    text = "\n".join(comments) + "\n"
    if totals:
        text += f"force_calls {statistics.fmean(totals)}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
        text += f"unique_saddles {statistics.fmean(uniques)}\n"
        text += f"fraction_connected {statistics.fmean(fractions)}\n"
        text += f"total_searches {statistics.fmean(searches)}\n"
        if len(totals) >= 2:
            text += f"force_calls_mean_error {statistics.stdev(totals)}\n"
    over = [
        record["start"]
        for record in records
        if record.get("success") and int(record["force_calls"]) >= int(search_cap)
    ]
    if over:
        raise RuntimeError(f"{template['id']} success {over[0]} spent {search_cap} or more force calls")
    metrics = (
        score_v2(
            [
                {
                    "force_calls": record["force_calls"],
                    "success": record["success"],
                    "failure": record["failure"],
                }
                for record in records
            ],
            search_cap,
        )
        if records
        else None
    )
    text += (
        f"code {template.get('code', '')}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(records)}\n"
    )
    if metrics is not None:
        text += (
            f"par2_mean {metrics['par2_mean']}\n"
            f"successes {metrics['successes']}\n"
            f"wilson_low {metrics['wilson_low']}\n"
            f"wilson_high {metrics['wilson_high']}\n"
            f"median {metrics['median']}\n"
            f"median_low {metrics['median_low']}\n"
            f"median_high {metrics['median_high']}\n"
            f"n_cap {metrics['n_cap']}\n"
            f"n_wrong_stationary {metrics['n_wrong_stationary']}\n"
            f"n_divergence {metrics['n_divergence']}\n"
            f"n_crash {metrics['n_crash']}\n"
            f"v2_nfailed {metrics['nfailed']}\n"
        )
    text += f"contributor Rohit Goswami\ndate {date.today().isoformat()}\nhost {os.uname().nodename}\n"
    if template.get("note"):
        text += f"note {template['note']}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.save()
    output.write_text(text, encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    if metrics is None:
        archive.drop_log()
        return None
    provenance = collect_provenance(
        code=str(template.get("code", "")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters={"template": template["id"], "problem": problem["id"], "cap": search_cap},
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "measured",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": note,
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code", ""),
                "algorithm": template.get("algorithm", ""),
                "code_version": template.get("code_version", ""),
                "commit": template.get("commit", ""),
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": search_cap,
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
    archive.drop_log()
    return metrics
