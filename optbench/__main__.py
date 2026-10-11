"""Command line for the OptBench replay tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="optbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check", help="compare the oracle with the problem files")
    check.add_argument("--problems", type=Path, default=Path("problems"))
    check.add_argument("--oracle", type=Path, default=Path("oracle/entries.jsonl"))

    importer = sub.add_parser(
        "import-archive",
        help="copy scores and geometries out of a bare clone of the public archive",
    )
    importer.add_argument("--bare", type=Path, required=True)
    importer.add_argument("--dest", type=Path, default=Path("."))

    args = parser.parse_args(argv)
    if args.cmd == "check":
        from .check import check as run_check

        return run_check(args.problems, args.oracle)
    if args.cmd == "import-archive":
        from .import_archive import import_archive

        import_archive(args.bare, args.dest)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
