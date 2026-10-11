"""benchmark.dat is one key, then whitespace, then the rest of the line."""

from __future__ import annotations

SPELLING = {
    "contributer": "contributor",
    "uncertanity": "uncertainty",
    "hiden": "hidden",
}


class DatError(ValueError):
    """A benchmark.dat line does not match the archive rule."""


def _rows(text: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[1].strip():
            raise DatError(f"line {lineno} has no value")
        rows.append((lineno, parts[0], parts[1].strip()))
    return rows


def parse(text: str) -> dict[str, str]:
    """Return the fields of a benchmark.dat document.

    The first whitespace separates the key from the value. Later whitespace
    stays in the value. A repeated key is an error.
    """
    fields, duplicates = parse_lenient(text)
    if duplicates:
        first = duplicates[0]
        raise DatError(f"duplicate key {first['key']} on line {first['line']}")
    return fields


def parse_lenient(text: str) -> tuple[dict[str, str], list[dict[str, str | int]]]:
    """Keep the first value of a repeated key and list the later ones."""
    fields: dict[str, str] = {}
    duplicates: list[dict[str, str | int]] = []
    for lineno, key, value in _rows(text):
        if key in fields:
            duplicates.append({"key": key, "line": lineno, "kept": fields[key], "also": value})
            continue
        fields[key] = value
    return fields, duplicates


def normalize(fields: dict[str, str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Apply spelling aliases. The second item lists each rename."""
    out: dict[str, str] = {}
    renamed: list[tuple[str, str]] = []
    for key, value in fields.items():
        canonical = SPELLING.get(key, key)
        if canonical in out:
            raise DatError(f"{key} collides with {canonical}")
        if canonical != key:
            renamed.append((key, canonical))
        out[canonical] = value
    return out, renamed
