"""GMIN basin-hopping with the archived symmetrise data file.

The binary is the optbench.org ``wales-27058.tar.bz2``. One start is one
xyz geometry from the LJ38 archive. The score is GMIN's own function-call
count on ``Target hit after``, written with the 2014 ``getdata.py`` rounding.
"""

from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from optbench.campaign import geometry_archive, resolve_problem, write_starts

ENGINE = "gmin"
SHA256 = "144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"
SNAPSHOT = "https://optbench.org/codes/wales-27058.tar.bz2"
_TARGET = re.compile(
    r"Target hit after\s+(\d+)\s+quenches\s+(\d+)\s+function calls"
)


def binary_path() -> Path:
    env = os.environ.get("GMIN_BIN")
    if env:
        return Path(env)
    return Path("/tmp/gmin-build/GMIN")


def available() -> str | None:
    binary = binary_path()
    if binary.is_file():
        return None
    return (
        "GMIN is not built. "
        f"The pin is the sha256 {SHA256} of {SNAPSHOT}. "
        f"{binary} is not present. "
        "codes/gmin/build.sh writes that binary."
    )


def _paper_number(value: float, digits: int) -> str:
    """``getdata.py`` writes ``round`` of the statistic in scientific form."""
    return f"{round(float(value)):.{digits}e}"


def _coords(xyz: str) -> str:
    lines = []
    for line in xyz.splitlines()[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        coords = " ".join(part.replace("e", "D").replace("E", "D") for part in parts[1:4])
        lines.append(coords)
    return "\n".join(lines) + "\n"


def _one(job: tuple[str, str, str, str]) -> dict:
    binary, data, xyz, folder = job
    dest = Path(folder)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "data").write_text(data, encoding="utf-8")
    (dest / "coords").write_text(_coords(xyz), encoding="utf-8")
    proc = subprocess.run(
        [binary],
        cwd=dest,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    blob = proc.stdout
    output = dest / "output"
    if output.is_file():
        blob = output.read_text(encoding="utf-8", errors="replace") + "\n" + blob
    found = _TARGET.findall(blob)
    if not found or proc.returncode != 0:
        return {"quenches": None, "force_calls": None, "success": False}
    quenches, calls = found[-1]
    return {"quenches": int(quenches), "force_calls": int(calls), "success": True}


def _xyz_members(archive: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            name = Path(member.name).name
            if not member.isfile() or not name.endswith(".xyz"):
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            found.append((name, extracted.read().decode("utf-8")))
    found.sort(key=lambda item: int(Path(item[0]).stem))
    return found


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    data_rel = str((template.get("gmin") or {}).get("data") or "")
    data_path = root / data_rel
    if not data_path.is_file():
        raise FileNotFoundError(f"{data_rel} is not the archived data file")
    data = data_path.read_text(encoding="utf-8")
    if "SYMMETRISE 1 0.04 0.03 0.09 0.09 0.4 18 0.4 1.7" not in data:
        raise RuntimeError("data file is not the archived symmetrise input")
    geometries = _xyz_members(geometry_archive(root, problem))
    if limit:
        geometries = geometries[:limit]
    if len(geometries) != 100 and not limit:
        raise RuntimeError(f"LJ38 archive has {len(geometries)} xyz starts")
    scratch = work or output.parent / "work" / str(template["id"]).replace("/", "-")
    jobs = [
        (str(binary_path()), data, text, str(scratch / Path(name).stem))
        for name, text in geometries
    ]
    if workers <= 1:
        details = [_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            details = list(pool.map(_one, jobs))
    calls = [int(row["force_calls"]) for row in details if row["success"] and row["force_calls"] is not None]
    failed = len(details) - len(calls)
    if not calls:
        raise RuntimeError("GMIN reported no target hit")
    mean = statistics.fmean(calls)
    median = statistics.median(calls)
    lines = [
        "# counter is GMIN's Target-hit function calls, the 2014 getdata.py score.",
        f"# unrounded_mean {mean:.2f}",
        f"force_calls {_paper_number(mean, 5)}",
        f"force_calls_median {_paper_number(median, 5)}",
        f"force_calls_min {_paper_number(min(calls), 4)}",
        f"force_calls_max {_paper_number(max(calls), 4)}",
        f"nfailed {failed}",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"tuning {template.get('tuning', 'archive')}",
        f"problem {problem['id']}",
        "counter harness",
        "reducer harness",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records = []
    for (name, _text), detail in zip(geometries, details):
        records.append(
            {
                "algorithm": template["algorithm"],
                "code": template["code"],
                "code_version": template["code_version"],
                "commit": template["commit"],
                "comparable": True,
                "counter": "harness",
                "failure": None if detail["success"] else "crash",
                "force_calls": detail["force_calls"] if detail["success"] else 0,
                "problem": problem["id"],
                "quenches": detail["quenches"],
                "schema": "optbench.start/v2",
                "start": Path(name).stem,
                "success": bool(detail["success"]),
                "template": template["id"],
                "tuning": template.get("tuning", "archive"),
            }
        )
    write_starts(output.with_suffix(".starts.jsonl"), records)
    return {
        "force_calls": round(mean),
        "nfailed": failed,
        "starts": len(details),
        "template": template["id"],
        "problem": problem["id"],
    }


def run(root, problem_name, template_name, *, output, workers):
    from optbench.campaign import load_template

    template = load_template(root, template_name)
    problem = resolve_problem(root, problem_name)
    return run_template(root, problem, template, output=output, workers=workers)
