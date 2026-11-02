"""Embed the paper figures above the tables.

The figures are drawn by the scripts in paper-figures/. This module only
reads the manifest those scripts write.
"""

from __future__ import annotations

import html
import json
from pathlib import Path


def load_manifest(root: Path) -> dict:
    path = root / "paper-figures" / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _figure(item: dict, prefix: str) -> str:
    caption = str(item.get("caption") or "")
    svg = html.escape(f"{prefix}/{item['svg']}", quote=True)
    pdf = html.escape(f"{prefix}/{item['pdf']}", quote=True)
    safe = html.escape(caption)
    return (
        '<figure class="paper-figure">'
        f'<img src="{svg}" alt="{safe}">'
        f"<figcaption>{safe} <a href=\"{pdf}\">PDF</a>.</figcaption>"
        "</figure>"
    )


def problem_charts(root: Path, problem_id: str, prefix: str) -> str:
    manifest = load_manifest(root)
    items = list((manifest.get("problems") or {}).get(problem_id) or [])
    if not items:
        return ""
    kinds = ("success", "family", "distribution", "history")
    parts = ['<div class="chart-stack">']
    for kind in kinds:
        chosen = [item for item in items if item.get("kind") == kind]
        if not chosen:
            continue
        if kind == "family":
            parts.append('<div class="paper-grid">')
            parts.extend(_figure(item, prefix) for item in chosen)
            parts.append("</div>")
        else:
            parts.extend(_figure(item, prefix) for item in chosen)
    parts.append("</div>")
    return "\n".join(parts)


def code_charts(root: Path, codes: list[str], prefix: str) -> str:
    manifest = load_manifest(root)
    items = []
    seen: set[str] = set()
    for code in codes:
        for item in (manifest.get("codes") or {}).get(code) or []:
            name = str(item.get("svg") or "")
            if name in seen:
                continue
            seen.add(name)
            items.append(item)
    if not items:
        return ""
    parts = ['<div class="chart-stack">', "<h3>Version history</h3>"]
    parts.extend(_figure(item, prefix) for item in items)
    parts.append("</div>")
    return "\n".join(parts)
