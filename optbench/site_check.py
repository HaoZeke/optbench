"""Fail when a rendered results table disagrees with the v2 board."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from .board import MissingAsvCommit, _show_per_image, build_board, ranked_cells
from .problems import load_all

STALE_CAPTIONS = (
    "Present runs. Fewer force calls first.",
    "Present sources. The cell is the record.",
    "Present codes. Fewer calls is to the left.",
)

CODE_FIELDS = (
    "template",
    "version",
    "commit",
    "par2_mean",
    "mean_calls",
    "median_calls",
    "failures",
    "n",
)


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self._table: dict | None = None
        self._row: dict | None = None
        self._cell: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        if tag == "table":
            self._table = {"attrs": attr, "rows": []}
        elif tag == "tr" and self._table is not None:
            self._row = {"attrs": attr, "cells": []}
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {"attrs": attr, "text": []}

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._cell["text"] = "".join(self._cell["text"]).strip()
            self._row["cells"].append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table["rows"].append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _parse_tables(text: str) -> list[dict]:
    parser = _Tables()
    parser.feed(text)
    return parser.tables


def _cell_map(row: dict) -> dict[str, str]:
    found: dict[str, str] = {}
    for cell in row["cells"]:
        field = cell["attrs"].get("data-field")
        if field:
            found[field] = cell["text"]
    return found


def _ranked_body(table: dict) -> list[dict]:
    return [row for row in table["rows"] if _cell_map(row)]


def _check_ranked(label: str, table: dict, document: dict) -> list[str]:
    problem_id = table["attrs"].get("data-problem", "")
    problem = (document.get("problems") or {}).get(problem_id)
    ranked = list(problem.get("ranked") or []) if problem else []
    body = _ranked_body(table)
    if len(body) != len(ranked):
        return [f"{label} {problem_id} ranked rows {len(body)} != board {len(ranked)}"]
    per_image = _show_per_image(problem) if problem else False
    errors: list[str] = []
    for index, (got_row, board_row) in enumerate(zip(body, ranked), start=1):
        got = _cell_map(got_row)
        expected = dict(ranked_cells(board_row, per_image))
        for field, value in expected.items():
            if got.get(field) != value:
                errors.append(
                    f"{label} {problem_id} row {index} {field} is {got.get(field)!r}, board has {value!r}"
                )
    return errors


def _check_headline(label: str, table: dict, document: dict) -> list[str]:
    problem_id = table["attrs"].get("data-problem", "")
    problem = (document.get("problems") or {}).get(problem_id)
    ranked = list(problem.get("ranked") or []) if problem else []
    body = _ranked_body(table)
    if not ranked:
        if body:
            return [f"{label} {problem_id} headline has a row and the board has no full run"]
        return []
    if len(body) != 1:
        return [f"{label} {problem_id} headline rows {len(body)} != 1"]
    per_image = _show_per_image(problem) if problem else False
    got = _cell_map(body[0])
    expected = dict(ranked_cells(ranked[0], per_image))
    errors = []
    for field, value in expected.items():
        if got.get(field) != value:
            errors.append(f"{label} {problem_id} headline {field} is {got.get(field)!r}, board has {value!r}")
    return errors


def _check_code_best(label: str, table: dict, document: dict) -> list[str]:
    errors: list[str] = []
    for row in table["rows"]:
        problem_id = row["attrs"].get("data-problem", "")
        template = row["attrs"].get("data-template", "")
        if not problem_id:
            continue
        problem = (document.get("problems") or {}).get(problem_id)
        ranked = list(problem.get("ranked") or []) if problem else []
        board_row = next(
            (item for item in ranked if str(item.get("template") or item.get("entry") or "") == template),
            None,
        )
        got = _cell_map(row)
        if board_row is None:
            errors.append(f"{label} {problem_id} code row {template!r} is not a ranked board row")
            continue
        expected = dict(ranked_cells(board_row, False))
        for field in CODE_FIELDS:
            if field not in got:
                errors.append(f"{label} {problem_id} {template} code row is missing {field}")
            elif got[field] != expected[field]:
                errors.append(
                    f"{label} {problem_id} {template} {field} is {got[field]!r}, board has {expected[field]!r}"
                )
    return errors


def _page_texts(root: Path) -> list[tuple[str, str]]:
    pages: list[Path] = []
    include = root / "docs" / "include"
    for name in ("sota.org", "v2-board.org", "collection.org", "codes.org"):
        pages.append(include / name)
    suite_dir = root / "docs" / "orgmode" / "generated" / "suites"
    if suite_dir.is_dir():
        pages.extend(sorted(suite_dir.glob("*.org")))
    org = root / "docs" / "orgmode"
    for name in ("index.org", "codes.org", "archive.org", "board.org", "run.org"):
        pages.append(org / name)
    build = root / "docs" / "build"
    if build.is_dir():
        pages.extend(sorted(build.rglob("*.html")))
    found: list[tuple[str, str]] = []
    for path in pages:
        if path.is_file():
            found.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
    return found


def _require(text: str, needle: str, label: str, errors: list[str]) -> None:
    if needle not in text:
        errors.append(f"{label} is missing {needle}")


def check_site(root: Path) -> list[str]:
    """Compare rendered v2 tables with a fresh board.

    ``docs/build`` is included when that directory exists.
    """
    root = Path(root)
    try:
        document = build_board(root)
    except MissingAsvCommit as exc:
        return [str(exc)]
    problems = load_all(root / "problems") if (root / "problems").is_dir() else {}
    errors: list[str] = []
    pages = _page_texts(root)
    if not pages:
        return ["site pages are missing"]
    texts = {label: text for label, text in pages}
    for label, text in pages:
        for caption in STALE_CAPTIONS:
            if caption in text:
                errors.append(f"{label} still has the old caption {caption!r}")
        for table in _parse_tables(text):
            kind = table["attrs"].get("data-board-table", "")
            if kind == "ranked":
                errors.extend(_check_ranked(label, table, document))
            elif kind == "headline":
                errors.extend(_check_headline(label, table, document))
            elif kind == "code-best":
                errors.extend(_check_code_best(label, table, document))

    for problem in problems.values():
        suite_label = f"docs/orgmode/generated/suites/{problem['suite']}.org"
        suite = texts.get(suite_label, "")
        if not suite:
            errors.append(f"{suite_label} is missing")
            continue
        _require(
            suite,
            f'data-board-table="ranked" data-problem="{problem["id"]}"',
            suite_label,
            errors,
        )
        board_problem = (document.get("problems") or {}).get(problem["id"]) or {}
        slug = str(problem["id"]).replace("/", "-")
        if board_problem.get("profile"):
            _require(suite, f"{slug}-profile.svg", suite_label, errors)
        if board_problem.get("ranked"):
            _require(suite, f"{slug}-cost.svg", suite_label, errors)
        for name in ("docs/include/sota.org", "docs/include/collection.org"):
            page = texts.get(name, "")
            if not page:
                errors.append(f"{name} is missing")
                continue
            _require(
                page,
                f'data-board-table="headline" data-problem="{problem["id"]}"',
                name,
                errors,
            )

    build = root / "docs" / "build"
    if build.is_dir():
        index = texts.get("docs/build/index.html", "")
        if not index:
            errors.append("docs/build/index.html is missing")
        elif 'data-board-table="headline"' not in index:
            errors.append("docs/build/index.html has no v2 headline table")
        for problem in problems.values():
            built = f"docs/build/generated/suites/{problem['suite']}.html"
            page = texts.get(built, "")
            if not page:
                errors.append(f"{built} is missing")
                continue
            _require(
                page,
                f'data-board-table="ranked" data-problem="{problem["id"]}"',
                built,
                errors,
            )
    return errors
