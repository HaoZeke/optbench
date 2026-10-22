"""Pt-island dimer and dynamics rows next to the archive.

Each code has one row per method. The mean is runs 1-9, the same sample
``report.sh`` wrote into the archive rows.
"""

from __future__ import annotations

import html
from pathlib import Path

from .dat import parse

SVN_LABEL = "EON (SVN r2549)"
DEVELOP_LABEL = "eOn (develop 500350d9)"
EXACT_LABELS = {SVN_LABEL, DEVELOP_LABEL}

METHODS = (
    ("Dimer", "eon-dimer"),
    ("Dynamics", "eon-md"),
)

METRICS = (
    "unique_saddles",
    "fraction_connected",
    "total_searches",
    "force_calls",
    "force_calls_mean_error",
)

DIFFERENCE_METRICS = (
    "unique_saddles",
    "fraction_connected",
    "total_searches",
    "force_calls",
)


def label_of(fields: dict[str, str]) -> str:
    code = str(fields.get("code") or "").strip()
    if code in EXACT_LABELS:
        return code
    version = str(fields.get("code_version") or "").strip()
    commit = str(fields.get("commit") or "").strip()
    if version == "r2549" or commit == "r2549":
        return SVN_LABEL
    if version.startswith("500350d9") or commit.startswith("500350d9"):
        return DEVELOP_LABEL
    return f"{code} {version}".strip() if version else code


def scored_fields(fields: dict[str, str]) -> dict[str, str]:
    """Runs 1-9 when the file records that sample, otherwise the file mean."""
    if fields.get("report_runs") == "9":
        return {key: fields[f"{key}_9"] for key in METRICS}
    return {key: fields[key] for key in METRICS}


def _archive_rows(oracle: list[dict]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for row in oracle:
        if row.get("suite") != "saddle-search" or row.get("problem") != "pt-island":
            continue
        entry = str(row.get("entry") or "")
        for _algorithm, archive_entry in METHODS:
            if entry == archive_entry:
                found[archive_entry] = row
    return found


def _campaign_rows(root: Path) -> list[dict[str, str]]:
    directory = root / "results" / "campaign"
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*.dat")):
        fields = parse(path.read_text(encoding="utf-8"))
        if fields.get("problem") != "saddle-search/pt-island":
            continue
        if fields.get("reducer") != "akmc-until-27":
            continue
        if fields.get("algorithm") not in {name for name, _entry in METHODS}:
            continue
        # report.sh wrote the archive from runs 1-9. A file without that
        # sample is a partial campaign and stays out of the rank.
        if fields.get("report_runs") != "9":
            continue
        rows.append(fields)
    return rows


def comparison_rows(root: Path, oracle: list[dict]) -> list[dict]:
    """Ranked rows, one per code per method, fewest force calls first."""
    archive = _archive_rows(oracle)
    campaign = _campaign_rows(root)
    grouped: dict[str, list[dict]] = {name: [] for name, _entry in METHODS}
    for algorithm, entry in METHODS:
        archived = archive.get(entry)
        if archived is None:
            continue
        fields = dict(archived["fields"])
        grouped[algorithm].append(
            {
                "method": algorithm,
                "label": label_of(fields),
                "metrics": {key: fields.get(key, "") for key in METRICS},
                "archive": True,
            }
        )
    seen: set[tuple[str, str]] = set()
    for fields in campaign:
        algorithm = str(fields.get("algorithm") or "")
        label = label_of(fields)
        key = (algorithm, label)
        if key in seen:
            raise ValueError(f"more than one row for {label} {algorithm}")
        seen.add(key)
        grouped[algorithm].append(
            {
                "method": algorithm,
                "label": label,
                "metrics": scored_fields(fields),
                "archive": False,
            }
        )
    ranked: list[dict] = []
    for algorithm, _entry in METHODS:
        rows = grouped[algorithm]
        if not rows:
            continue
        base = next((row for row in rows if row["archive"]), None)
        base_numbers = None
        if base is not None:
            base_numbers = {key: float(base["metrics"][key]) for key in DIFFERENCE_METRICS}
        rows.sort(key=lambda row: (float(row["metrics"]["force_calls"]), row["label"]))
        for index, row in enumerate(rows, 1):
            if base_numbers is None:
                differences = {key: None for key in DIFFERENCE_METRICS}
            else:
                differences = {
                    key: float(row["metrics"][key]) - base_numbers[key] for key in DIFFERENCE_METRICS
                }
            ranked.append({**row, "rank": index, "difference": differences})
    return ranked


def _plain(value: float) -> str:
    return f"{value:+.10g}"


def saddle_pt_html(root: Path, oracle: list[dict]) -> str:
    """Table of archive and later runs. Empty when neither method has a row."""
    rows = comparison_rows(root, oracle)
    if not rows:
        return ""
    lines = [
        '<section class="bench" id="pt-island-dimer-dynamics">',
        "<h3>Dimer and dynamics</h3>",
        "<p>Runs 1-9. Fewer force calls rank first. "
        "A difference is this row minus the archive row.</p>",
    ]
    for algorithm, _entry in METHODS:
        method_rows = [row for row in rows if row["method"] == algorithm]
        if not method_rows:
            continue
        lines.append(f"<h4>{html.escape(algorithm)}</h4>")
        lines.append('<div class="bench-table"><table class="bench">')
        lines.append(
            "<caption>Runs 1-9. Fewer force calls rank first. "
            "A difference is this row minus the archive row.</caption>"
        )
        headers = ["Rank", "Code"]
        for key in METRICS:
            headers.append(key.replace("_", " "))
            if key in DIFFERENCE_METRICS:
                headers.append(f"{key.replace('_', ' ')} difference")
        lines.append("<thead><tr>")
        for header in headers:
            lines.append(f"<th>{html.escape(header)}</th>")
        lines.append("</tr></thead><tbody>")
        for row in method_rows:
            cells = [str(row["rank"]), row["label"]]
            for key in METRICS:
                cells.append(row["metrics"][key])
                if key in DIFFERENCE_METRICS:
                    gap = row["difference"][key]
                    if row["archive"]:
                        cells.append("0")
                    elif gap is None:
                        cells.append("")
                    else:
                        cells.append(_plain(gap))
            lines.append("<tr>")
            for cell in cells:
                lines.append(f"<td>{html.escape(str(cell))}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table></div>")
    lines.append("</section>")
    return "\n".join(lines)
