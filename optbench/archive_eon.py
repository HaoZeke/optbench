"""Run one archived eOn entry with a current client.

The archived config.ini is the template. QuickMin and FIRE timesteps are
copied into the sections version 3.5.0 reads, and the numbers stay the
archived ones. The archived report.sh reduces the starts.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .dat import parse_lenient


def _sections(text: str) -> dict[str, dict[str, str]]:
    current = ""
    found: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip()
            found.setdefault(current, {})
            continue
        if not current or not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        found[current][key.strip()] = value.strip()
    return found


def adapt_optimizer_sections(text: str) -> str:
    """Put an archived timestep where version 3.5.0 reads it.

    Version 3.5.0 reads ``time_step`` from ``[QuickMin]`` or ``[FIRE]``.
    An archived file that stored the same number under ``[Optimizer]`` keeps
    that number and gains the section the client reads.
    """
    sections = _sections(text)
    optimizer = sections.get("Optimizer") or {}
    method = optimizer.get("opt_method", "").lower()
    extra: list[str] = []
    if method == "qm" and "QuickMin" not in sections and "time_step" in optimizer:
        extra.append(f"[QuickMin]\ntime_step = {optimizer['time_step']}")
    if method == "fire" and "FIRE" not in sections:
        lines = ["[FIRE]"]
        if "time_step" in optimizer:
            lines.append(f"time_step = {optimizer['time_step']}")
        if "time_step_max" in optimizer:
            lines.append(f"time_step_max = {optimizer['time_step_max']}")
        if len(lines) > 1:
            extra.append("\n".join(lines))
    if not extra:
        return text if text.endswith("\n") else text + "\n"
    body = text if text.endswith("\n") else text + "\n"
    return body + "\n" + "\n\n".join(extra) + "\n"


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")


def _client(job: tuple[str, str, str]) -> str:
    client, ini, work = job
    dest = Path(work)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.ini").write_text(ini, encoding="utf-8")
    for name in ("reactant.con", "product.con", "in.lammps", "pos.con"):
        source = Path(str(dest) + "." + name)
        if source.is_file():
            shutil.move(source, dest / name)
    proc = subprocess.run(
        [client],
        cwd=dest,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    (dest / "client.err").write_bytes(proc.stderr)
    dat = dest / "results.dat"
    if dat.is_file():
        return "ok"
    (dest / "results.dat").write_text(
        "1 termination_reason\n10000 force_calls_neb\n10000 total_force_calls\n",
        encoding="utf-8",
    )
    return "missing"


def _prepare_min(src: Path, dest_marker: Path) -> None:
    """Stage reactant and product next to the work directory marker."""
    shutil.copy(src / "a.con", Path(str(dest_marker) + ".reactant.con"))
    shutil.copy(src / "b.con", Path(str(dest_marker) + ".product.con"))


def _reduce(report: Path, cwd: Path, code: str, version: str, template: str) -> str:
    subprocess.run(["bash", str(report)], cwd=cwd, check=True)
    fields, _duplicates = parse_lenient((cwd / "benchmark.dat").read_text(encoding="utf-8"))
    kept = [
        "force_calls",
        "force_calls_min",
        "force_calls_max",
        "force_calls_median",
        "nfailed",
    ]
    lines = [f"{key} {fields[key]}" for key in kept if key in fields and fields[key] not in ("", "nan")]
    lines.append(f"code {code}")
    lines.append(f"code_version {version}")
    lines.append(f"template {template}")
    text = "\n".join(lines) + "\n"
    (cwd / "benchmark.dat").write_text(text, encoding="utf-8")
    return text


def run_minima(
    entry: Path,
    minima_archive: Path,
    client: Path,
    work: Path,
    output: Path,
    *,
    workers: int,
    code: str,
    version: str,
) -> str:
    """Replicate an archived ``minima/min??`` NEB entry."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    _extract(minima_archive, work / "unpacked")
    minima = work / "unpacked" / "minima"
    if not minima.is_dir():
        found = list((work / "unpacked").rglob("min01"))
        if not found:
            raise FileNotFoundError(f"{minima_archive} has no min01")
        minima = found[0].parent
    ini = adapt_optimizer_sections((entry / "config.ini").read_text(encoding="utf-8"))
    report = entry / "report.sh"
    staged = work / "staged"
    staged.mkdir()
    shutil.copy(report, staged / "report.sh")
    (staged / "minima").mkdir()
    jobs = []
    for src in sorted(path for path in minima.iterdir() if path.is_dir() and path.name.startswith("min")):
        dest = staged / "minima" / src.name
        _prepare_min(src, dest)
        jobs.append((str(client), ini, str(dest)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        states = list(pool.map(_client, jobs))
    if states and all(state == "missing" for state in states):
        err = staged / "minima" / "min01" / "client.err"
        detail = err.read_text(encoding="utf-8", errors="replace") if err.is_file() else "no results.dat"
        raise RuntimeError(detail[-2000:])
    text = _reduce(staged / "report.sh", staged, code, version, entry.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def run_pairs(
    entry: Path,
    coords_archive: Path,
    client: Path,
    work: Path,
    output: Path,
    *,
    workers: int,
    count: int,
    code: str,
    version: str,
) -> str:
    """Replicate an archived reactant/product loop, ``start_NN`` / ``end_NN``."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    _extract(coords_archive, work / "unpacked")
    coords = work / "unpacked" / "coords-con"
    if not coords.is_dir():
        found = list((work / "unpacked").rglob("start_00.con"))
        if not found:
            raise FileNotFoundError(f"{coords_archive} has no start_00.con")
        coords = found[0].parent
    ini = adapt_optimizer_sections((entry / "config.ini").read_text(encoding="utf-8"))
    lammps = entry / "in.lammps"
    staged = work / "staged"
    runs = staged / "runs"
    runs.mkdir(parents=True)
    shutil.copy(entry / "report.sh", staged / "report.sh")
    jobs = []
    for index in range(count):
        dest = runs / f"run-{index}"
        shutil.copy(coords / f"start_{index:02d}.con", Path(str(dest) + ".reactant.con"))
        shutil.copy(coords / f"end_{index:02d}.con", Path(str(dest) + ".product.con"))
        if lammps.is_file():
            shutil.copy(lammps, Path(str(dest) + ".in.lammps"))
        jobs.append((str(client), ini, str(dest)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_client, jobs))
    text = _reduce(staged / "report.sh", staged, code, version, entry.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="optbench.archive_eon")
    sub = parser.add_subparsers(dest="kind", required=True)
    neb = sub.add_parser("neb")
    neb.add_argument("--entry", type=Path, required=True)
    neb.add_argument("--minima", type=Path, required=True)
    neb.add_argument("--client", type=Path, required=True)
    neb.add_argument("--work", type=Path, required=True)
    neb.add_argument("--output", type=Path, required=True)
    neb.add_argument("--workers", type=int, default=1)
    neb.add_argument("--code", default="Eon")
    neb.add_argument("--version", default="3.5.0")
    pairs = sub.add_parser("pairs")
    pairs.add_argument("--entry", type=Path, required=True)
    pairs.add_argument("--coords", type=Path, required=True)
    pairs.add_argument("--client", type=Path, required=True)
    pairs.add_argument("--work", type=Path, required=True)
    pairs.add_argument("--output", type=Path, required=True)
    pairs.add_argument("--workers", type=int, default=1)
    pairs.add_argument("--count", type=int, default=50)
    pairs.add_argument("--code", default="Eon")
    pairs.add_argument("--version", default="3.5.0")
    args = parser.parse_args(argv)
    if args.kind == "neb":
        text = run_minima(
            args.entry,
            args.minima,
            args.client,
            args.work,
            args.output,
            workers=args.workers,
            code=args.code,
            version=args.version,
        )
    else:
        text = run_pairs(
            args.entry,
            args.coords,
            args.client,
            args.work,
            args.output,
            workers=args.workers,
            count=args.count,
            code=args.code,
            version=args.version,
        )
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
