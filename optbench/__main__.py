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

    sub.add_parser("templates", help="list run templates")

    campaign = sub.add_parser(
        "run",
        help="run one problem template and write benchmark.dat",
    )
    campaign.add_argument("problem")
    campaign.add_argument("template")
    campaign.add_argument("--root", type=Path, default=Path("."))
    campaign.add_argument("--client", type=Path, default=None)
    campaign.add_argument("--geometries", type=Path, default=None)
    campaign.add_argument("--output", type=Path, default=Path("benchmark.dat"))
    campaign.add_argument("--workers", type=int, default=1)
    campaign.add_argument("--work", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "check":
        from .check import check as run_check

        return run_check(args.problems, args.oracle)
    if args.cmd == "import-archive":
        from .import_archive import import_archive

        import_archive(args.bare, args.dest)
        return 0
    if args.cmd == "templates":
        from .campaign import list_templates

        for template in list_templates(Path(".")):
            print(template["id"])
        return 0
    if args.cmd == "run":
        from .campaign import main_run

        forwarded = [
            args.problem,
            args.template,
            "--root",
            str(args.root),
            "--output",
            str(args.output),
            "--workers",
            str(args.workers),
        ]
        if args.client is not None:
            forwarded.extend(["--client", str(args.client)])
        if args.geometries is not None:
            forwarded.extend(["--geometries", str(args.geometries)])
        if args.work is not None:
            forwarded.extend(["--work", str(args.work)])
        return main_run(forwarded)
    return 2


if __name__ == "__main__":
    sys.exit(main())
