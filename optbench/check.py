"""Check archived benchmark.dat rows against problem contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dat import normalize, parse_lenient
from .endpoints import archive_error
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


def _dat_for(starts: Path) -> Path:
    name = starts.name
    if name.endswith(".starts.jsonl"):
        stem = name[: -len(".starts.jsonl")]
    elif name.endswith(".starts"):
        stem = name[: -len(".starts")]
    else:
        stem = starts.stem
    return starts.with_name(stem + ".dat")


def _json_records(path: Path) -> list[dict]:
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"missing endpoints {path.name} line {lineno}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"missing endpoints {path.name} line {lineno}")
        records.append(payload)
    return records


def _problem_of(starts: Path) -> str:
    dat = _dat_for(starts)
    if dat.is_file():
        fields, _duplicates = parse_lenient(dat.read_text(encoding="utf-8"))
        if fields.get("problem"):
            return str(fields["problem"])
    try:
        records = _json_records(starts)
    except ValueError:
        return ""
    if records and records[0].get("problem"):
        return str(records[0]["problem"])
    return ""


def _start_files(campaign: Path) -> list[Path]:
    found = [
        path
        for pattern in ("*.starts.jsonl", "*.starts")
        for path in campaign.glob(pattern)
        if path.is_file()
    ]
    return sorted(found, key=lambda path: path.name)


def _legacy_hashes(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"oracle/saddle_endpoint_legacy.txt has {stripped!r}")
        name, digest = parts
        found[name] = digest
    return found


def saddle_search_starts(root: Path) -> list[Path]:
    """Campaign starts files whose problem id is ``saddle-search/...``."""
    campaign = Path(root) / "results" / "campaign"
    if not campaign.is_dir():
        return []
    return [path for path in _start_files(campaign) if _problem_of(path).startswith("saddle-search/")]


def saddle_endpoint_gaps(root: Path) -> list[str]:
    """Saddle-search starts files that do not carry both endpoint minima.

    ``oracle/saddle_endpoint_legacy.txt`` lists the sha256 of starts files
    scored before those coordinates were stored. A file whose bytes still
    match is exempt. A replaced file has a new hash and is checked.
    """
    legacy = _legacy_hashes(Path(root) / "oracle" / "saddle_endpoint_legacy.txt")
    gaps: list[str] = []
    for starts in saddle_search_starts(root):
        digest = hashlib.sha256(starts.read_bytes()).hexdigest()
        if legacy.get(starts.name) == digest:
            continue
        try:
            records = _json_records(starts)
        except ValueError as exc:
            gaps.append(str(exc))
            continue
        name = str(records[0].get("endpoints") or "") if records else ""
        archive = starts.parent / name if name else starts.with_name("missing.endpoints.npz")
        error = archive_error(archive, records, starts.name)
        if error:
            gaps.append(error)
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
    root = oracle_path.resolve().parent.parent
    endpoint_gaps = saddle_endpoint_gaps(root)
    if endpoint_gaps:
        print(f"endpoint_gaps {len(endpoint_gaps)}")
        for line in endpoint_gaps:
            print(line)
        return 1
    from .site_check import check_site

    site_errors = check_site(root)
    if site_errors:
        print(f"site {len(site_errors)}")
        for line in site_errors:
            print(line)
        return 1
    print("site tables match the board")
    return 0
