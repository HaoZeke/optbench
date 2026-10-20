"""Paper-revision eOn templates match the archived config.ini files."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

from optbench.campaign import _check_stop, list_templates, resolve_problem

ROOT = Path(__file__).resolve().parents[1]
STAMPS = {"r2025", "r2030", "r2090", "r2156", "r2205"}
SKIP_PARTS = {"hTST", "accelerated-md"}


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Same merge the eOn reader uses: the first section of a name keeps the keys."""
    sections: dict[str, dict[str, str]] = {}
    order: list[str] = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            if current not in sections:
                sections[current] = {}
                order.append(current)
            continue
        key, value = line.split("=", 1)
        sections[current][key.strip()] = value.strip()
    return {name: sections[name] for name in order}


def _report_cap(text: str) -> int | None:
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        match = re.search(r"\$1\s*<\s*(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _canon(value: object) -> tuple:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return ("bool", text.lower() == "true")
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", text):
        return ("num", float(text))
    return ("str", text)


def _paper_entries() -> list[Path]:
    found = []
    root = ROOT / "archive" / "entries"
    for config in sorted(root.rglob("config.ini")):
        relative = config.relative_to(root)
        if any(part in SKIP_PARTS or "bowl" in part for part in relative.parts):
            continue
        if not any(part.startswith("eon") for part in relative.parts):
            continue
        benchmark = config.parent / "benchmark.dat"
        version = ""
        for line in benchmark.read_text(encoding="utf-8").splitlines():
            if line.startswith("code_version"):
                version = line.split()[1]
        if version in STAMPS:
            found.append(config.parent)
    return found


def _paper_templates() -> list[dict]:
    return [
        row
        for row in list_templates(ROOT)
        if str(row.get("id", "")).startswith("eon-svn-r")
    ]


def test_every_paper_config_has_one_template():
    templates = _paper_templates()
    by_entry = {}
    for template in templates:
        entry = template["entry"]
        assert entry not in by_entry, entry
        by_entry[entry] = template
    archived = {
        f"archive/entries/{path.relative_to(ROOT / 'archive' / 'entries')}"
        for path in _paper_entries()
    }
    assert set(by_entry) == archived
    assert len(archived) == 40


def test_template_ini_matches_the_archived_config():
    for template in _paper_templates():
        folder = ROOT / template["entry"]
        archived = _parse_ini((folder / "config.ini").read_text(encoding="utf-8"))
        rendered = template["ini"]
        assert set(rendered) == set(archived), template["id"]
        for section, values in archived.items():
            got = rendered[section]
            assert set(got) == set(values), (template["id"], section, set(got) ^ set(values))
            for key, raw in values.items():
                assert _canon(got[key]) == _canon(raw), (template["id"], section, key, got[key], raw)
        cap = _report_cap((folder / "report.sh").read_text(encoding="utf-8"))
        if cap is None:
            assert "failure_cap" not in template, template["id"]
        else:
            assert template["failure_cap"] == cap, template["id"]
        assert template["code"] == "Eon svn"
        assert template["commit"] == template["code_version"]
        assert template["engine"] == "eon"
        _check_stop(resolve_problem(ROOT, template["problem"]), template)


def test_stamp_pins_keep_r2549_and_name_the_paper_revisions():
    data = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in data["code"]}
    current = by_id["eon-svn"]
    assert current["commit"] == "r2549"
    assert current["name"] == "Eon svn"
    assert "forbidden" not in current["note"].lower()
    expected = {
        "eon-svn-r2025",
        "eon-svn-r2030",
        "eon-svn-r2090",
        "eon-svn-r2091",
        "eon-svn-r2156",
        "eon-svn-r2158",
        "eon-svn-r2159",
        "eon-svn-r2169",
        "eon-svn-r2204",
        "eon-svn-r2205",
        "eon-svn-r2222",
    }
    for ident in expected:
        row = by_id[ident]
        assert row["name"] == "Eon svn"
        assert row["repository"] == current["repository"]
        assert row["commit"] == ident.removeprefix("eon-svn-")
        assert "forbidden" not in row["note"].lower()
