"""Copy scores and geometries from a bare clone of the public archive.

Code snapshots under website/codes/ are left behind. Problem files that
already exist are not rewritten.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .dat import SPELLING, parse_lenient

ARCHIVE_COMMIT = "0f26b796"
IDENTITY = {
    "algorithm",
    "code",
    "code_file",
    "code_version",
    "contributor",
    "contributer",
    "date",
    "hidden",
    "hide",
    "hiden",
}
TEXT_SUFFIXES = {".sh", ".py", ".ini", ".lammps", ".dat", ".txt", ".md"}
SKIP_TEXT_NAMES = {"search_results.txt"}
MAX_TEXT_BYTES = 200_000
GEOM_SUFFIXES = {".tgz", ".tar.gz", ".tar.bz2", ".traj", ".con", ".xyz"}


def _git(bare: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(bare), *args],
        text=True,
        errors="replace",
    )


def _ls(bare: Path, rev: str) -> list[tuple[str, str, int]]:
    rows = []
    for line in _git(bare, "ls-tree", "-r", "-l", rev).splitlines():
        meta, path = line.split("\t", 1)
        _mode, _kind, sha, size = meta.split()
        rows.append((path, sha, int(size)))
    return rows


def _blob(bare: Path, sha: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(bare), "cat-file", "blob", sha])


def _keep_geometry(raw: bytes, name: str) -> bool:
    """Structures stay. Code snapshots and path stubs do not."""
    if name.startswith("eon_"):
        return False
    if len(raw) < 128:
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            return True
        if text.startswith("../") or text.endswith((".tgz", ".tar.gz", ".tar.bz2")):
            return False
    return True


def _is_geometry(path: str) -> bool:
    if path.startswith("website/codes/"):
        return False
    name = path.rsplit("/", 1)[-1]
    if name.startswith("POSCAR"):
        return True
    return any(name.endswith(suffix) for suffix in GEOM_SUFFIXES)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _write_partial(path: Path, suite: str, problem: str, count: int, required: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ", ".join(_toml_string(key) for key in required)
    path.write_text(
        "\n".join(
            [
                f"id = {_toml_string(f'{suite}/{problem}')}",
                f"title = {_toml_string(problem)}",
                f"suite = {_toml_string(suite)}",
                f"problem = {_toml_string(problem)}",
                'contract = "partial"',
                f"entry_count = {count}",
                "",
                "[metrics]",
                f"required = [{keys}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def import_archive(bare: Path, dest: Path) -> None:
    """Fill ``dest`` from the main branch, plus geometries that exist only elsewhere."""
    if not (bare / "HEAD").exists() and not (bare / "objects").exists():
        raise FileNotFoundError(f"{bare} is not a bare git directory")
    rows = _ls(bare, "main")
    by_problem: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
    geometries: list[tuple[str, str, int, str]] = []
    for path, sha, size in rows:
        if _is_geometry(path):
            geometries.append((path, sha, size, "main"))
        if not path.endswith("/benchmark.dat") or not path.startswith("website/"):
            continue
        parts = path.split("/")
        if len(parts) < 4:
            continue
        suite, problem = parts[1], parts[2]
        by_problem.setdefault((suite, problem), []).append((path, sha, size))

    oracle_path = dest / "oracle" / "entries.jsonl"
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_rows = []
    text_jobs: list[tuple[str, str, str]] = []
    for (suite, problem), entries in sorted(by_problem.items()):
        key_sets: list[set[str]] = []
        for path, sha, _size in sorted(entries):
            raw = _blob(bare, sha)
            text = raw.decode("utf-8", errors="replace")
            entry = path[len(f"website/{suite}/{problem}/") : -len("/benchmark.dat")]
            fields, duplicates = parse_lenient(text)
            repeated = {item["key"] for item in duplicates}
            row = {
                "suite": suite,
                "problem": problem,
                "entry": entry,
                "path": path,
                "commit": ARCHIVE_COMMIT,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "text": text,
                "fields": fields,
                "duplicates": duplicates,
            }
            oracle_rows.append(row)
            canonical = {SPELLING.get(key, key) for key in fields}
            key_sets.append(canonical - repeated)
            entry_dir = "/".join(path.split("/")[:-1])
            for sib_path, sib_sha, sib_size in rows:
                if not sib_path.startswith(entry_dir + "/"):
                    continue
                if sib_path.count("/") != entry_dir.count("/") + 1:
                    continue
                name = sib_path.rsplit("/", 1)[-1]
                if name in SKIP_TEXT_NAMES or sib_size > MAX_TEXT_BYTES:
                    continue
                if not any(name.endswith(suffix) for suffix in TEXT_SUFFIXES):
                    continue
                rel = f"archive/entries/{suite}/{problem}/{entry}/{name}"
                text_jobs.append((rel, sib_sha, name))
        if key_sets:
            required = sorted(set.intersection(*key_sets) - IDENTITY)
        else:
            required = []
        _write_partial(
            dest / "problems" / suite / f"{problem}.toml",
            suite,
            problem,
            len(entries),
            required,
        )

    oracle_rows.sort(key=lambda row: (row["suite"], row["problem"], row["entry"]))
    oracle_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in oracle_rows),
        encoding="utf-8",
    )

    for rel, sha, _name in text_jobs:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(_blob(bare, sha))

    extra = _ls(bare, "small_molecules")
    main_paths = {path for path, _sha, _size in rows}
    for path, sha, size in extra:
        if path in main_paths or not _is_geometry(path):
            continue
        geometries.append((path, sha, size, "small_molecules"))

    manifest = dest / "inputs" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    lines = []
    for path, sha, size, branch in sorted(geometries, key=lambda item: (item[3], item[0])):
        raw = _blob(bare, sha)
        name = path.rsplit("/", 1)[-1]
        if not _keep_geometry(raw, name):
            continue
        digest = hashlib.sha256(raw).hexdigest()
        stored = dest / "inputs" / "sha256" / digest / name
        if digest not in seen:
            stored.parent.mkdir(parents=True, exist_ok=True)
            if not stored.exists():
                stored.write_bytes(raw)
            seen[digest] = stored
        lines.append(
            json.dumps(
                {
                    "branch": branch,
                    "bytes": size,
                    "path": path,
                    "sha256": digest,
                    "stored": str(seen[digest].relative_to(dest)),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    manifest.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    print(f"oracle {len(oracle_rows)}")
    print(f"geometries {len(seen)}")
    print(f"manifest {len(lines)}")
