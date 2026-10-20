"""Check archived benchmark.dat rows against problem contracts."""

from __future__ import annotations

import json
from pathlib import Path

from .dat import normalize
from .problems import load_all


def load_oracle(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = lineno
        rows.append(row)
    return rows


def missing_keys(problem: dict, fields: dict[str, str]) -> list[str]:
    canonical, _renamed = normalize(fields)
    required = problem["metrics"]["required"]
    return [key for key in required if key not in canonical]


def archive_gaps(problems: dict[str, dict], rows: list[dict]) -> list[str]:
    """Ways an archived file departs from its problem contract."""
    gaps: list[str] = []
    for row in rows:
        ident = f"{row['suite']}/{row['problem']}"
        problem = problems.get(ident)
        if problem is None:
            continue
        _canonical, renamed = normalize(row["fields"])
        for old, new in renamed:
            gaps.append(f"spelling {row['path']} {old} -> {new}")
        if problem.get("contract") == "stated":
            for key in missing_keys(problem, row["fields"]):
                gaps.append(f"missing {row['path']} {key}")
        for item in row.get("duplicates") or []:
            gaps.append(
                f"duplicate {row['path']} {item['key']} kept {item['kept']!r} also {item['also']!r}"
            )
    return gaps


def check(problems_dir: Path, oracle_path: Path) -> int:
    """Return 0 when archive gaps match oracle/archive_gaps.txt."""
    problems = load_all(problems_dir)
    rows = load_oracle(oracle_path)
    counts: dict[str, int] = {}
    structural: list[str] = []
    for row in rows:
        ident = f"{row['suite']}/{row['problem']}"
        counts[ident] = counts.get(ident, 0) + 1
        problem = problems.get(ident)
        if problem is None:
            structural.append(f"{ident} has an oracle row and no problem file")
            continue
        if problem.get("contract") == "partial":
            missing = missing_keys(problem, row["fields"])
            for key in missing:
                structural.append(f"partial {row['path']} missing {key}")
    for ident, problem in problems.items():
        expected = problem.get("entry_count")
        got = counts.get(ident, 0)
        if expected is not None and expected != got:
            structural.append(f"{ident} entry_count is {expected}, oracle has {got}")
    gaps = archive_gaps(problems, rows)
    known_path = oracle_path.parent / "archive_gaps.txt"
    known = known_path.read_text(encoding="utf-8").splitlines() if known_path.exists() else []
    print(f"entries {len(rows)}")
    print(f"problems {len(problems)}")
    print(f"archive_gaps {len(gaps)}")
    if structural:
        print(f"structural {len(structural)}")
        for line in structural:
            print(line)
        return 1
    if gaps != known:
        print("archive gaps do not match oracle/archive_gaps.txt")
        for line in gaps:
            if line not in known:
                print(f"+ {line}")
        for line in known:
            if line not in gaps:
                print(f"- {line}")
        return 1
    from .site_check import check_site

    root = oracle_path.resolve().parent.parent
    site_errors = check_site(root)
    if site_errors:
        print(f"site {len(site_errors)}")
        for line in site_errors:
            print(line)
        return 1
    print("site tables match the board")
    from .findings import findings_errors

    finding_errors = findings_errors(root)
    if finding_errors:
        print(f"findings {len(finding_errors)}")
        for line in finding_errors:
            print(line)
        return 1
    print("findings match the archive rows")
    return 0
