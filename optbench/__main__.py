"""Command line for the OptBench replay tree."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cyclopts

app = cyclopts.App(
    name="optbench",
    help="Run one OptBench problem and write benchmark.dat.",
)


def _eon_client(client: Path | None) -> Path:
    if client is not None:
        return client
    found = shutil.which("eonclient")
    if found:
        return Path(found)
    env = os.environ.get("EONCLIENT")
    if env:
        return Path(env)
    raise SystemExit("pass --client, set EONCLIENT, or put eonclient on PATH")


@app.command
def check(
    problems: Path = Path("problems"),
    oracle: Path = Path("oracle/entries.jsonl"),
) -> None:
    """Compare the oracle with the problem files."""
    from optbench.check import check as run_check

    raise SystemExit(run_check(problems, oracle))


@app.command(name="import-archive")
def import_archive(
    bare: Path,
    dest: Path = Path("."),
) -> None:
    """Copy scores and geometries out of a bare clone of the public archive."""
    from optbench.import_archive import import_archive as copy_archive

    copy_archive(bare, dest)


@app.command
def templates(root: Path = Path(".")) -> None:
    """List run templates."""
    from optbench.campaign import list_templates

    for template in list_templates(root):
        print(template["id"])


@app.command
def run(
    problem: str,
    template: str,
    *,
    root: Path = Path("."),
    client: Path | None = None,
    geometries: Path | None = None,
    output: Path = Path("benchmark.dat"),
    workers: int = 1,
    work: Path | None = None,
) -> None:
    """Run one problem template and write benchmark.dat."""
    from optbench.campaign import run as run_campaign

    summary = run_campaign(
        root,
        problem,
        template,
        client=_eon_client(client),
        geometries=geometries,
        output=output,
        workers=workers,
        work=work,
    )
    print(
        f"{summary['problem']} {summary['template']} "
        f"force_calls={summary['force_calls']} nfailed={summary['nfailed']} "
        f"starts={summary['starts']}"
    )


@app.command(name="eon-job")
def eon_job(
    name: str,
    *,
    root: Path = Path("."),
    client: Path | None = None,
    output: Path = Path("benchmark.dat"),
    workers: int = 1,
    work: Path | None = None,
) -> None:
    """Run one named eOn band or saddle job and write benchmark.dat."""
    from optbench.eon_jobs import run_named

    run_named(
        root,
        name,
        _eon_client(client),
        output,
        workers=workers,
        work=work,
    )


@app.command
def neb(
    entry: Path,
    minima: Path,
    *,
    client: Path | None = None,
    work: Path,
    output: Path,
    workers: int = 1,
    code: str = "Eon",
    version: str = "3.5.0",
) -> None:
    """Run an archived minima-directory NEB entry."""
    from optbench.archive_eon import run_minima

    text = run_minima(
        entry,
        minima,
        _eon_client(client),
        work,
        output,
        workers=workers,
        code=code,
        version=version,
    )
    print(text, end="")


@app.command
def pairs(
    entry: Path,
    coords: Path,
    *,
    client: Path | None = None,
    work: Path,
    output: Path,
    workers: int = 1,
    count: int = 50,
    code: str = "Eon",
    version: str = "3.5.0",
) -> None:
    """Run an archived reactant/product saddle or band entry."""
    from optbench.archive_eon import run_pairs

    text = run_pairs(
        entry,
        coords,
        _eon_client(client),
        work,
        output,
        workers=workers,
        count=count,
        code=code,
        version=version,
    )
    print(text, end="")


def main(argv: list[str] | None = None) -> None:
    app(argv)


if __name__ == "__main__":
    main()
