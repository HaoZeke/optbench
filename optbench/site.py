"""Render the public site from the problem files and the oracle.

Prose stays in ``docs/orgmode``. This module writes the comparison
tables those pages include.
"""

from __future__ import annotations

import html
from pathlib import Path

from .check import load_oracle
from .dat import normalize
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


def problem_html(problem: dict, oracle: list[dict], smoke_note: str = "") -> str:
    """One problem: the stopping rule, then the rank."""
    canonical_rows = []
    for row in _rows_for(problem, oracle):
        fields, _renamed = normalize(row["fields"])
        canonical_rows.append((row, fields))
    metrics = list(problem["metrics"]["required"])
    primary = _primary(problem)
    best = None
    if primary:
        values = [_float(fields.get(primary, "")) for _row, fields in canonical_rows]
        numbers = [value for value in values if value is not None]
        best = min(numbers) if numbers else None

    parts = ["<section class=\"bench\" id=\"" + html.escape(problem["problem"]) + "\">"]
    parts.append(f"<h2>{html.escape(str(problem.get('title') or problem['problem']))}</h2>")
    bits = _bits(problem)
    if bits:
        parts.append("<dl class=\"bench-rule\">")
        for label, text in bits:
            parts.append(f"<dt>{html.escape(label)}</dt><dd>{html.escape(text)}</dd>")
        parts.append("</dl>")
    if smoke_note:
        parts.append(f"<p class=\"bench-smoke\">{html.escape(smoke_note)}</p>")
    captions = {
        "force_calls": "Archived scores, fewest force calls first.",
        "mean_epoch": "Archived scores, fewest epochs first.",
        "mean_epochs": "Archived scores, fewest epochs first.",
        "avg_energy": "Archived scores, lowest energy first.",
        "barrier_force_calls": "Archived scores, fewest barrier force calls first.",
        "true_rate": "Archived scores, lowest rate first.",
    }
    parts.append("<table>")
    parts.append(f"<caption>{html.escape(captions.get(primary or '', 'Archived scores.'))}</caption>")
    headers = ["", "Entry", "Code", *metrics, "Contributor", "Date"]
    parts.append("<thead><tr>")
    for index, header in enumerate(headers):
        kind = "num" if index == 0 or header in metrics else "text"
        parts.append(f"<th class=\"{kind}\" scope=\"col\">{html.escape(header.replace('_', ' '))}</th>")
    parts.append("</tr></thead><tbody>")
    for rank, (row, fields) in enumerate(canonical_rows, 1):
        value = _float(fields.get(primary, "")) if primary else None
        classes = []
        if best is not None and value == best:
            classes.append("best")
        if _held(fields):
            classes.append("held")
        class_attr = f" class=\"{' '.join(classes)}\"" if classes else ""
        entry = row["entry"]
        if _held(fields):
            entry = f"{entry} (held out of the original table)"
        code = fields.get("code", "")
        version = fields.get("code_version", "")
        if version:
            code = f"{code} {version}".strip()
        cells = [str(rank), entry, code]
        for metric in metrics:
            cells.append(fields.get(metric, ""))
        cells.append(fields.get("contributor", ""))
        cells.append(fields.get("date", ""))
        parts.append(f"<tr{class_attr}>")
        for index, cell in enumerate(cells):
            kind = "num" if index == 0 or headers[index] in metrics else "text"
            parts.append(f"<td class=\"{kind}\">{html.escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></section>")
    return "".join(parts)


def _raw(fragment: str) -> str:
    lines = ["#+begin_export rst", ".. raw:: html", ""]
    for line in fragment.splitlines():
        lines.append("   " + line)
    lines.append("#+end_export")
    lines.append("")
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

    featured = problems["minimization/lj38"]
    featured_html = problem_html(featured, oracle)
    include = root / "docs" / "include"
    include.mkdir(parents=True, exist_ok=True)
    (include / "featured.org").write_text(_raw(featured_html), encoding="utf-8")

    order = list(SUITE_ORDER) + sorted(set(by_suite) - set(SUITE_ORDER))
    for suite in order:
        group = by_suite.get(suite)
        if not group:
            continue
        sections = []
        for problem in sorted(group, key=lambda item: item["problem"].lower()):
            note = smoke if problem["id"] == "saddle-search/pt-island" else ""
            sections.append(problem_html(problem, oracle, note))
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
