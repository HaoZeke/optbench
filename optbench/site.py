"""Render the public site from the problem files and the oracle.

Prose stays in ``docs/orgmode``. This module writes the comparison
tables those pages include.
"""

from __future__ import annotations

import html
from pathlib import Path

from .check import load_oracle
from .dat import normalize, parse_lenient
from .problems import load_all

SUITE_ORDER = (
    "minimization",
    "saddle-search",
    "double-ended-saddle-search",
    "globalopt",
    "lowest-eigenvector",
    "tsopt",
    "mlopt",
    "neb",
    "hTST",
    "accelerated-md",
)

SUITE_TITLE = {
    "minimization": "Minimization",
    "saddle-search": "Saddle search",
    "double-ended-saddle-search": "Double-ended saddle search",
    "globalopt": "Global optimization",
    "lowest-eigenvector": "Lowest eigenvector",
    "tsopt": "Transition state optimization",
    "mlopt": "Machine learning optimization",
    "neb": "Nudged elastic band",
    "hTST": "Harmonic transition state theory",
    "accelerated-md": "Accelerated molecular dynamics",
}

PRIMARY = (
    "force_calls",
    "mean_epoch",
    "mean_epochs",
    "avg_energy",
    "barrier_force_calls",
    "true_rate",
)


def suite_title(suite: str) -> str:
    return SUITE_TITLE.get(suite, suite)


def _float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _held(fields: dict[str, str]) -> bool:
    for key in ("hidden", "hiden", "hide"):
        if fields.get(key, "").lower() == "true":
            return True
    return False


def _primary(problem: dict) -> str | None:
    required = list(problem["metrics"]["required"])
    for key in PRIMARY:
        if key in required:
            return key
    return required[0] if required else None


def _bits(problem: dict) -> list[tuple[str, str]]:
    bits: list[tuple[str, str]] = []
    convergence = problem.get("convergence") or {}
    if "force_norm" in convergence:
        unit = convergence.get("force_unit", "")
        kind = "L2 norm" if str(convergence.get("norm", "")).lower() == "l2" else "norm"
        bits.append(("Stop", f"{kind} of the force at or below {convergence['force_norm']} {unit}".strip()))
    if "gradient_norm" in convergence:
        bits.append(("Stop", f"gradient norm below {convergence['gradient_norm']}"))
    if "max_force_calls" in convergence:
        bits.append(("Cap", f"{convergence['max_force_calls']} force calls"))
    if "energy_above_minimum" in convergence:
        bits.append(("Stop", f"within {convergence['energy_above_minimum']} of the known minimum"))
    if "max_epochs" in convergence:
        bits.append(("Cap", f"{convergence['max_epochs']} epochs"))
    goal = problem.get("goal") or {}
    if goal.get("connected_saddles"):
        barrier = goal.get("barrier_below_eV")
        text = f"{goal['connected_saddles']} connected saddles"
        if barrier is not None:
            text += f" below {barrier} eV"
        bits.append(("Goal", text))
    sample = problem.get("sample") or {}
    if sample.get("starts"):
        bits.append(("Starts", str(sample["starts"])))
    archive = (problem.get("scoring") or {}).get("archive") or {}
    if archive.get("failure"):
        bits.append(("Archived rule", str(archive["failure"])))
    if problem.get("remark"):
        bits.append(("Note", str(problem["remark"])))
    if problem.get("contract") == "partial":
        bits.append(("Contract", "Partly stated. The columns are the keys shared by every archived file."))
    return bits


def _rows_for(problem: dict, oracle: list[dict]) -> list[dict]:
    ident = problem["id"]
    rows = [row for row in oracle if f"{row['suite']}/{row['problem']}" == ident]
    key = _primary(problem)

    def sort_key(row: dict) -> tuple:
        canonical, _renamed = normalize(row["fields"])
        value = _float(canonical.get(key, "")) if key else None
        missing = value is None
        return (missing, value if value is not None else 0.0, row["entry"])

    return sorted(rows, key=sort_key)


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def _caption(primary: str | None) -> str:
    captions = {
        "force_calls": "Archived scores, fewest force calls first.",
        "mean_epoch": "Archived scores, fewest epochs first.",
        "mean_epochs": "Archived scores, fewest epochs first.",
        "avg_energy": "Archived scores, lowest energy first.",
        "barrier_force_calls": "Archived scores, fewest barrier force calls first.",
        "true_rate": "Archived scores, lowest rate first.",
    }
    return captions.get(primary or "", "Archived scores.")


def _marked_failed(value: object) -> bool:
    try:
        return int(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _plot(points: list[dict], primary: str | None, caption: str | None = None) -> str:
    numbers = [point["value"] for point in points if point["value"] is not None and point["value"] > 0]
    if len(numbers) < 2:
        return ""
    scale = "linear"
    if numbers and min(numbers) > 0 and max(numbers) / min(numbers) >= 8:
        scale = "log"
    if any(point["value"] is not None and point["value"] <= 0 for point in points):
        scale = "linear"
    payload = {
        "metric": primary or "",
        "scale": scale,
        "points": [
            {
                "id": point["id"],
                "entry": point["entry"],
                "code": point["code"],
                "value": point["value"],
                "best": point["best"],
                "held": point["held"],
                "failed": bool(point.get("failed")),
            }
            for point in points
            if point["value"] is not None
        ],
    }
    import json

    data = json.dumps(payload, ensure_ascii=True)
    label = caption or (primary or "score").replace("_", " ")
    rows = len(payload["points"])
    return (
        '<figure class="bench-plot">'
        f"<figcaption>{html.escape(label)}. Fewer calls is to the left. Same order as the table.</figcaption>"
        f'<div class="bench-plot-mount" style="--bench-rows:{rows}"></div>'
        f'<script type="application/json" class="bench-plot-data">{data}</script>'
        "</figure>"
    )


def _code_label(fields: dict[str, str]) -> str:
    """Code column. Sella 0.1.1 stays on the 2014 table and is marked legacy."""
    code = fields.get("code", "")
    version = fields.get("code_version", "")
    label = f"{code} {version}".strip() if version else code
    if code == "Sella" and version == "0.1.1":
        label = f"{label} legacy"
    return label


def campaign_problem(stem: str) -> str | None:
    """Map a results/campaign file name to a problem id."""
    if "morse" in stem:
        return "minimization/morse-bulk"
    if stem.startswith("eon-3.5.0-neb-pt"):
        return "neb/Pt-heptamer"
    if stem.startswith("eon-3.5.0-de"):
        return "double-ended-saddle-search/lj38"
    if stem.startswith("eon-3.5.0-saddle-pt"):
        return "saddle-search/pt-island"
    if stem == "sella-2.6.0-ts-lj38":
        return "tsopt/LJ38"
    if stem.startswith(("pele-", "eon-client-", "eon-svn-")):
        return "minimization/lj38"
    return None


def _mean_key(row: dict) -> tuple[int, float]:
    try:
        return (0, float(row["mean"]))
    except (TypeError, ValueError):
        return (1, 0.0)


def _load_present(root: Path) -> dict[str, list[dict]]:
    """Present runs from campaign files, plus JSON rows those files do not already name."""
    grouped: dict[str, list[dict]] = {}
    seen: set[str] = set()
    campaign = root / "results" / "campaign"
    if campaign.is_dir():
        for path in sorted(campaign.glob("*.dat")):
            fields, _duplicates = parse_lenient(path.read_text(encoding="utf-8"))
            if fields.get("partial") == "true" or fields.get("ranked") == "false":
                continue
            problem = fields.get("problem") or campaign_problem(path.stem)
            if problem is None:
                continue
            seen.add(path.stem)
            grouped.setdefault(problem, []).append(
                {
                    "entry": path.stem,
                    "code": fields.get("code", ""),
                    "version": fields.get("code_version", ""),
                    "mean": fields.get("force_calls", ""),
                    "min": fields.get("force_calls_min", ""),
                    "max": fields.get("force_calls_max", ""),
                    "failed": fields.get("nfailed", ""),
                }
            )
    import json

    for path in sorted((root / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        problem = record.get("problem")
        if not problem:
            continue
        for run in record.get("runs") or []:
            entry = str(run.get("entry") or "")
            if not entry or entry in seen:
                continue
            seen.add(entry)
            metrics = run.get("metrics") or {}
            grouped.setdefault(str(problem), []).append(
                {
                    "entry": entry,
                    "code": str(run.get("code") or ""),
                    "version": str(run.get("code_version") or ""),
                    "mean": metrics.get("force_calls", ""),
                    "min": metrics.get("force_calls_min", ""),
                    "max": metrics.get("force_calls_max", ""),
                    "failed": metrics.get("nfailed", run.get("nfailed", "")),
                }
            )
    for rows in grouped.values():
        rows.sort(key=_mean_key)
    return grouped


def _present_block(rows: list[dict], heading: str | None = "Present runs") -> str:
    lines = []
    if heading:
        lines.append(f"<h3>{html.escape(heading)}</h3>")
    if not rows:
        lines.append("<p>No present run is recorded.</p>")
        return "\n".join(lines)
    lines.append('<div class="bench-table"><table>')
    lines.append("<caption>Present runs. Fewer force calls first. The 2014 scores follow.</caption>")
    lines.append(
        "<thead><tr>"
        '<th class="text" scope="col">Entry</th>'
        '<th class="text" scope="col">Code</th>'
        '<th class="text" scope="col">Version</th>'
        '<th class="num" scope="col">Mean (force calls)</th>'
        '<th class="num" scope="col">Min</th>'
        '<th class="num" scope="col">Max</th>'
        '<th class="num" scope="col">Failed</th>'
        "</tr></thead><tbody>"
    )
    for row in rows:
        failed = str(row.get("failed") or "0")
        mark = ' class="bench-row-fail"' if failed not in ("", "0") else ""
        lines.append(f"<tr{mark}>")
        for key, kind in (
            ("entry", "text"),
            ("code", "text"),
            ("version", "text"),
            ("mean", "num"),
            ("min", "num"),
            ("max", "num"),
            ("failed", "num"),
        ):
            lines.append(f'<td class="{kind}">{html.escape(str(row.get(key, "")))}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    points = []
    values = []
    for row in rows:
        try:
            value = float(row["mean"])
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        values.append(value)
        points.append(row)
    best = min(values) if values else None
    plot_points = [
        {
            "id": str(row["entry"]),
            "entry": f"{row['code']} {row['version']} {row['entry']}".strip(),
            "code": str(row["code"]),
            "value": float(row["mean"]),
            "best": best is not None and float(row["mean"]) == best,
            "held": False,
            "failed": _marked_failed(row.get("failed")),
        }
        for row in points
    ]
    lines.append(_plot(plot_points, "force_calls", "Present codes"))
    return "\n".join(lines)


def _collection_html(order: list[str], by_suite: dict[str, list[dict]], present: dict[str, list[dict]]) -> str:
    lines = [
        '<section class="bench" id="collection">',
        "<p>Each suite is a heading. Each problem under it is a subheading. "
        "A present run is a score recorded in this repository. "
        "The 2014 table for that problem is on the suite page.</p>",
    ]
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        lines.append(f'<h2 id="suite-{html.escape(suite)}">{html.escape(suite_title(suite))}</h2>')
        for problem in sorted(group, key=lambda item: str(item.get("title") or item["problem"]).lower()):
            title = str(problem.get("title") or problem["problem"])
            rows = present.get(problem["id"], [])
            lines.append(f"<h3>{html.escape(title)}</h3>")
            count = "no present run" if not rows else f"{len(rows)} present run" + ("s" if len(rows) != 1 else "")
            href = f"generated/suites/{suite}.html#{problem['problem']}"
            lines.append(
                f'<p>{html.escape(count)}. '
                f'<a href="{html.escape(href)}">2014 scores and the present table</a>.</p>'
            )
            if rows:
                lines.append(_present_block(rows, heading=None))
            else:
                lines.append("<p>No present run is recorded.</p>")
    lines.append("</section>")
    return "\n".join(lines)


def problem_html(
    problem: dict,
    oracle: list[dict],
    smoke_note: str = "",
    present: list[dict] | None = None,
) -> str:
    """One problem: the stopping rule, the bar chart, then the rank."""
    canonical_rows = []
    for row in _rows_for(problem, oracle):
        fields, _renamed = normalize(row["fields"])
        canonical_rows.append((row, fields))
    metrics = list(problem["metrics"]["required"])
    primary = _primary(problem)
    best = None
    if primary:
        numbers = [
            value
            for value in (_float(fields.get(primary, "")) for _row, fields in canonical_rows)
            if value is not None
        ]
        best = min(numbers) if numbers else None

    lines = [f'<section class="bench" id="{html.escape(problem["problem"])}">']
    lines.append(f"<h2>{html.escape(str(problem.get('title') or problem['problem']))}</h2>")
    bits = _bits(problem)
    if bits:
        lines.append('<dl class="bench-rule">')
        for label, text in bits:
            lines.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(text)}</dd>")
        lines.append("</dl>")
    if smoke_note:
        lines.append(f'<p class="bench-smoke">{html.escape(smoke_note)}</p>')
    if present is not None:
        lines.append(_present_block(present))

    prepared = []
    for rank, (row, fields) in enumerate(canonical_rows, 1):
        value = _float(fields.get(primary, "")) if primary else None
        held = _held(fields)
        code = _code_label(fields)
        row_id = _slug(f"{problem['id']}-{row['entry']}")
        prepared.append(
            {
                "id": row_id,
                "rank": rank,
                "entry": row["entry"],
                "label": f"{row['entry']} (held out of the original table)" if held else row["entry"],
                "code": code,
                "fields": fields,
                "value": value,
                "best": best is not None and value == best,
                "held": held,
                "current": bool(row.get("current")),
            }
        )
    lines.append('<div class="bench-table"><table>')
    lines.append(f"<caption>{html.escape(_caption(primary))}</caption>")
    headers = ["", "Entry", "Code", *metrics, "Contributor", "Date"]
    lines.append("<thead><tr>")
    for index, header in enumerate(headers):
        kind = "num" if index == 0 or header in metrics else "text"
        lines.append(f'<th class="{kind}" scope="col">{html.escape(header.replace("_", " "))}</th>')
    lines.append("</tr></thead><tbody>")
    for point in prepared:
        classes = []
        if point["best"]:
            classes.append("best")
        if point["held"]:
            classes.append("held")
        if point.get("current"):
            classes.append("current")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        cells = [str(point["rank"]), point["label"], point["code"]]
        for metric in metrics:
            cells.append(point["fields"].get(metric, ""))
        cells.append(point["fields"].get("contributor", ""))
        cells.append(point["fields"].get("date", ""))
        lines.append(f'<tr id="{html.escape(point["id"])}"{class_attr}>')
        for index, cell in enumerate(cells):
            kind = "num" if index == 0 or headers[index] in metrics else "text"
            lines.append(f'<td class="{kind}">{html.escape(cell)}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div>")
    lines.append(_plot(prepared, primary, "2014 scores"))
    lines.append("</section>")
    return "\n".join(lines)


def _raw(fragment: str) -> str:
    lines = ["#+begin_export rst", ".. raw:: html", ""]
    for line in fragment.splitlines():
        lines.append("   " + line)
    lines.append("#+end_export")
    lines.append("")
    return "\n".join(lines)


def _sota_html(root: Path) -> str:
    accepted = []
    for path in sorted((root / "results").glob("*.json")):
        import json

        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "sota":
            continue
        if record.get("problem") not in (None, "", "minimization/lj38"):
            continue
        accepted.extend(record.get("runs") or [])
    chart_runs = list(accepted)
    for path in sorted((root / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("on_chart") and record.get("problem") in (None, "", "minimization/lj38"):
            chart_runs.extend(record.get("runs") or [])
    lines = [
        '<section class="bench" id="sota">',
        "<h2>Lennard-Jones 38</h2>",
    ]
    rule_path = root / "problems" / "minimization" / "lj38.toml"
    if rule_path.is_file():
        from .problems import load_problem

        bits = _bits(load_problem(rule_path))
        if bits:
            lines.append('<dl class="bench-rule">')
            for label, text in bits:
                lines.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(text)}</dd>")
            lines.append("</dl>")
    structure = root / "docs" / "source" / "_static" / "lj38.svg"
    if structure.is_file():
        lines.append(
            '<figure class="bench-structure">'
            '<img src="_static/lj38.svg" width="176" height="155" '
            'alt="Thirty-eight particles in Lennard-Jones start cluster_0000.">'
            "<figcaption>Start cluster_0000, one of the 1,000 geometries. "
            "Particles only. The potential has no bonds.</figcaption>"
            "</figure>"
        )
    if not accepted:
        lines.append(
            "<p class=\"bench-smoke\">No run of a present source has been accepted. "
            "The contract is contract/problems/minimization/lj38.json. "
            "A result is optbench.result/v1 with status sota.</p>"
        )
    else:
        ordered = sorted(
            accepted,
            key=lambda run: float((run.get("metrics") or {}).get("force_calls") or 1e99),
        )
        points = []
        numbered = []
        chart_ordered = sorted(
            chart_runs,
            key=lambda run: float((run.get("metrics") or {}).get("force_calls") or 1e99),
        )
        for run in chart_ordered:
            metrics = run.get("metrics") or {}
            value = metrics.get("force_calls")
            if isinstance(value, (int, float)):
                numbered.append(float(value))
                points.append(
                    {
                        "id": _slug(str(run.get("entry", ""))),
                        "entry": str(run.get("entry", "")),
                        "code": str(run.get("code", "")),
                        "value": float(value),
                        "best": False,
                        "held": False,
                        "failed": _marked_failed(metrics.get("nfailed")),
                    }
                )
        best = min(numbered) if numbered else None
        for point in points:
            point["best"] = best is not None and point["value"] == best
        lines.append(
            f"<p class=\"bench-count\">{len(ordered)} methods. Lower force calls is better.</p>"
        )
        lines.append(
            "<table><caption>Present sources. The cell is the record.</caption>"
            "<thead><tr><th>Entry</th><th>Code</th>"
            "<th class=\"num\">Mean (force calls)</th>"
            "<th class=\"num\">Min (force calls)</th>"
            "<th class=\"num\">Max (force calls)</th>"
            "<th class=\"num\">Failed</th>"
            "<th>Commit</th></tr></thead><tbody>"
        )
        for run in ordered:
            metrics = run.get("metrics") or {}
            commit = str(run.get("commit") or run.get("code_version") or "")
            nfailed = metrics.get("nfailed", run.get("nfailed", 0))
            fail_mark = ""
            try:
                failed_n = int(nfailed)
            except (TypeError, ValueError):
                failed_n = 0
            if failed_n > 0:
                fail_mark = " bench-row-fail"
            lines.append(
                '<tr id="{}" class="{}">'.format(
                    html.escape(_slug(str(run.get("entry", "")))),
                    fail_mark.strip(),
                )
                + "<td>{}</td><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>"
                "<td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td></tr>".format(
                    html.escape(str(run.get("entry", ""))),
                    html.escape(str(run.get("code", ""))),
                    html.escape(str(metrics.get("force_calls", ""))),
                    html.escape(str(metrics.get("force_calls_min", ""))),
                    html.escape(str(metrics.get("force_calls_max", ""))),
                    html.escape(str(nfailed)),
                    html.escape(commit[:12]),
                )
            )
        lines.append("</tbody></table>")
        lines.append(_plot(points, "force_calls"))
    lines.append("</section>")
    return "\n".join(lines)


def _submission_html(root: Path) -> str:
    import json

    lines = ['<section class="bench" id="recorded-passes">', "<h2>Other recorded runs</h2>"]
    lines.append(
        "<p class=\"bench-smoke\">These numbers are real and they are not the SOTA board. "
        "Each run missed the present-source rule in the Lennard-Jones 38 contract.</p>"
    )
    lines.append(
        "<table><caption>Mean force calls on 1000 starts. None failed the force cap.</caption>"
        "<thead><tr><th>Entry</th><th>Code</th><th class=\"num\">Mean</th>"
        "<th class=\"num\">Min</th><th class=\"num\">Max</th><th>Why it stays here</th></tr></thead><tbody>"
    )
    for path in sorted((root / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "archive":
            continue
        for run in record.get("runs") or []:
            metrics = run.get("metrics") or {}
            lines.append("<tr>")
            lines.append(f"<td>{html.escape(str(run.get('entry', '')))}</td>")
            lines.append(f"<td>{html.escape(str(run.get('code', '')))}</td>")
            lines.append(f"<td class=\"num\">{html.escape(str(metrics.get('force_calls', '')))}</td>")
            lines.append(f"<td class=\"num\">{html.escape(str(metrics.get('force_calls_min', '')))}</td>")
            lines.append(f"<td class=\"num\">{html.escape(str(metrics.get('force_calls_max', '')))}</td>")
            lines.append(f"<td>{html.escape(str(run.get('notes', '')))}</td>")
            lines.append("</tr>")
    lines.append("</tbody></table></section>")
    return "\n".join(lines)


def _codes_html(path: Path) -> str:
    import tomllib

    record = tomllib.loads(path.read_text(encoding="utf-8"))
    contributor = record.get("benchmark_contributor", "")
    lines = [
        '<section class="bench" id="current-codes">',
        "<h2>Current codes</h2>",
        "<p class=\"bench-smoke\">A score from one of these pins is SOTA only when the result "
        "names this commit, the problem geometries, and the problem stop. "
        f"The benchmark contributor is {html.escape(contributor)}.</p>",
        '<div class="bench-table"><table>',
        "<caption>No force-call number is listed until a run writes benchmark.dat under the problem rule.</caption>",
        "<thead><tr><th>Code</th><th>Role</th><th>Source</th><th>Commit</th></tr></thead><tbody>",
    ]
    for code in record.get("code", []):
        commit = code.get("commit") or code.get("ref", "")
        short = commit[:12]
        lines.append("<tr>")
        lines.append(f"<td>{html.escape(code.get('name', ''))}</td>")
        lines.append(f"<td>{html.escape(code.get('role', ''))}</td>")
        repo = code.get("repository", "")
        ref = code.get("ref", "")
        lines.append(
            f'<td><a href="{html.escape(repo)}">{html.escape(repo.split("/")[-1])} {html.escape(ref)}</a></td>'
        )
        lines.append(f'<td class="num">{html.escape(short)}</td>')
        lines.append("</tr>")
    lines.append("</tbody></table></div></section>")
    return "\n".join(lines)


def render(root: Path) -> None:
    """Write ``docs/orgmode/generated`` from ``problems`` and ``oracle``."""
    problems = load_all(root / "problems")
    oracle = load_oracle(root / "oracle" / "entries.jsonl")
    smoke = ""
    smoke_path = root / "softsaddle" / "entry.toml"
    if smoke_path.exists():
        import tomllib

        record = tomllib.loads(smoke_path.read_text(encoding="utf-8"))
        archived = record.get("archived") or {}
        smoke_run = record.get("smoke") or {}
        if archived and smoke_run:
            smoke = (
                f"A {smoke_run.get('searches')}-search SoftSaddle smoke is not in this rank. "
                f"The {archived.get('date')} {archived.get('entry')} entry records "
                f"{archived.get('force_calls')} force calls."
            )
    out = root / "docs" / "orgmode" / "generated"
    suite_dir = out / "suites"
    suite_dir.mkdir(parents=True, exist_ok=True)
    by_suite: dict[str, list[dict]] = {}
    for problem in problems.values():
        by_suite.setdefault(problem["suite"], []).append(problem)

    include = root / "docs" / "include"
    include.mkdir(parents=True, exist_ok=True)
    (include / "codes.org").write_text(_raw(_codes_html(root / "codes" / "registry.toml")), encoding="utf-8")

    featured = problems["minimization/lj38"]
    featured_html = problem_html(featured, oracle)
    (include / "featured.org").write_text(_raw(featured_html), encoding="utf-8")
    (include / "sota.org").write_text(_raw(_sota_html(root)), encoding="utf-8")
    (include / "terra-pass.org").write_text(_raw(_submission_html(root)), encoding="utf-8")
    present = _load_present(root)

    order = list(SUITE_ORDER) + sorted(set(by_suite) - set(SUITE_ORDER))
    (include / "collection.org").write_text(
        _raw(_collection_html(order, by_suite, present)),
        encoding="utf-8",
    )
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        sections = []
        for problem in sorted(group, key=lambda item: item["problem"].lower()):
            note = smoke if problem["id"] == "saddle-search/pt-island" else ""
            sections.append(problem_html(problem, oracle, note, present.get(problem["id"], [])))
        body = [
            f"#+TITLE: {suite_title(suite)}",
            "#+OPTIONS: toc:nil num:nil author:nil",
            "",
            _raw("".join(sections)).rstrip(),
            "",
        ]
        (suite_dir / f"{suite}.org").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    render(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
