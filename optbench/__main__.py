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
    limit: int = 0,
    resume: bool = False,
) -> None:
    """Run one problem template and write benchmark.dat."""
    from optbench.campaign import load_template
    from optbench.campaign import run as run_campaign

    loaded = load_template(root, template)
    client_path = _eon_client(client) if loaded["engine"] == "eon-client" else client
    summary = run_campaign(
        root,
        problem,
        template,
        client=client_path,
        geometries=geometries,
        output=output,
        workers=workers,
        work=work,
        limit=limit,
        resume=resume,
    )
    print(
        f"{summary['problem']} {summary['template']} "
        f"force_calls={summary['force_calls']} nfailed={summary['nfailed']} "
        f"starts={summary['starts']}"
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


def _head(root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


@app.command(name="campaign-ase")
def campaign_ase(
    *,
    root: Path = Path("."),
    output_dir: Path = Path("results/campaign"),
    workers: int = 1,
    limit: int = 0,
    match: str = "",
    ranked_only: bool = True,
    resume: bool = False,
) -> None:
    """Run the ranked ASE and Sella templates."""
    from optbench.ase_sella import run_many

    summaries = run_many(
        root,
        output_dir,
        workers=workers,
        limit=limit,
        match=match,
        ranked_only=ranked_only,
        resume=resume,
    )
    for summary in summaries:
        print(
            f"{summary['problem']} {summary['template']} "
            f"force_calls={summary['force_calls']} nfailed={summary['nfailed']} "
            f"starts={summary['starts']}"
        )


@app.command
def smoke(
    output: Path = Path("results/smoke/quadratic.json"),
    root: Path = Path("."),
    commit: str = "",
) -> None:
    """Run the quadratic v2 smoke. This is not a campaign."""
    from optbench.smoke import write_smoke

    document = write_smoke(output, commit=commit or _head(root))
    print(f"wrote {output} runs={len(document['runs'])}")


@app.command
def score(
    starts: Path,
    cap: int | None = None,
    output: Path = Path("score.json"),
) -> None:
    """Reduce a per-start JSONL file to a v2 score."""
    import json

    from optbench.score import score_v2

    records = [json.loads(line) for line in starts.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = score_v2(records, cap)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"successes={summary['successes']}/{summary['n']} "
        f"par2_mean={summary['par2_mean']} median={summary['median']}"
    )


@app.command
def statistics(
    output: Path = Path("results/validation/statistics.json"),
    root: Path = Path("."),
    results: Path = Path("results/asv"),
    commit: str = "",
) -> None:
    """Score the quadratic sample from saved asv tracks. This does not rerun it."""
    from optbench.asv_score import write_statistics_from_asv

    document = write_statistics_from_asv(output, root=root, results_dir=root / results)
    recorded = document["provenance"]["commit"]
    if commit and not recorded.startswith(commit):
        raise SystemExit(f"asv results are commit {recorded}, not {commit}")
    displacement = document["displacement"]["metrics"]
    dimer = document["dimer"]["metrics"]
    print(
        f"wrote {output} commit={recorded} displacement_median={displacement['median']} "
        f"dimer_median={dimer['median']} "
        f"displacement_failed={displacement['nfailed']} dimer_failed={dimer['nfailed']}"
    )


@app.command(name="validate-counter")
def validate_counter(
    output: Path = Path("results/validation/lj38-counter.dat"),
    *,
    root: Path = Path("."),
    client: Path | None = None,
    limit: int = 3,
    work: Path | None = None,
) -> None:
    """Compare the harness counter with eonclient on an LJ38 subset."""
    from optbench.validate_counter import validate_subset

    summary = validate_subset(root, _eon_client(client), output, limit=limit, work=work)
    print(
        f"wrote {output} version={summary['version']} commit={summary['commit']} "
        f"matched_client={summary['matched_client']}/{limit} "
        f"matched_builtin={summary['matched_builtin']}/{limit}"
    )


def main(argv: list[str] | None = None) -> None:
    app(argv)


if __name__ == "__main__":
    main()
