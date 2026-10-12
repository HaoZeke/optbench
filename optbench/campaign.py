"""One command from a problem file and a template to benchmark.dat.

The template is the client configuration. The problem file is the stop
and the geometries. The starts stay in this loop. The record is one
benchmark.dat.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .problems import load_problem
from .score import summarize_capped

_INI_ORDER = ("Main", "Potential", "Optimizer")


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_ini(sections: dict) -> str:
    names = [name for name in _INI_ORDER if name in sections]
    names.extend(sorted(name for name in sections if name not in _INI_ORDER))
    blocks = []
    for name in names:
        lines = [f"[{name}]"]
        for key, value in sections[name].items():
            lines.append(f"{key} = {format_value(value)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def load_template(root: Path, name: str) -> dict:
    path = Path(name)
    if not path.is_file():
        path = root / "templates" / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no template {name}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "id" not in data or "engine" not in data or "ini" not in data:
        raise ValueError(f"{path} needs id, engine, and ini")
    data["_path"] = path
    return data


def list_templates(root: Path) -> list[dict]:
    found = []
    for path in sorted((root / "templates").rglob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        found.append(data)
    return found


def resolve_problem(root: Path, name: str) -> dict:
    path = Path(name)
    if path.is_file():
        return load_problem(path)
    candidate = root / "problems" / f"{name}.toml"
    if not candidate.is_file():
        raise FileNotFoundError(f"no problem {name}")
    return load_problem(candidate)


def geometry_archive(root: Path, problem: dict) -> Path:
    archive_path = (problem.get("input") or {}).get("archive_path")
    if not archive_path:
        raise ValueError(f"{problem['id']} has no input.archive_path")
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(f"{archive_path} is not in the geometry manifest")


def iter_geometries(path: Path) -> list[Path]:
    if path.is_dir():
        found = sorted(path.glob("*.con"))
    elif tarfile.is_tarfile(path):
        raise ValueError("pass a directory of .con files, or use the problem archive through run()")
    else:
        found = [path]
    if not found:
        raise FileNotFoundError(f"no .con files in {path}")
    return found


def _check_stop(problem: dict, template: dict) -> int:
    convergence = problem.get("convergence") or {}
    cap = int(convergence["max_force_calls"])
    optimizer = (template.get("ini") or {}).get("Optimizer") or {}
    if "max_iterations" in optimizer and int(optimizer["max_iterations"]) < cap:
        raise ValueError("template max_iterations is below the problem force-call cap")
    if "force_norm" in convergence and "converged_force" in optimizer:
        if abs(float(optimizer["converged_force"]) - float(convergence["force_norm"])) > 1e-12:
            raise ValueError("template converged_force is not the problem force norm")
    return cap


def parse_client_dat(text: str) -> tuple[int | None, str | None]:
    """Read the eon client results.dat. The value is the first field."""
    calls = None
    reason = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[1] == "total_force_calls":
            calls = int(float(parts[0]))
        elif parts[1] == "termination_reason":
            reason = parts[0]
    return calls, reason


def _one(job: tuple[str, str, str, str]) -> int | None:
    client, ini, con, work = job
    dest = Path(work)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.ini").write_text(ini)
    shutil.copy(con, dest / "pos.con")
    subprocess.run(
        [client],
        cwd=dest,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    dat = dest / "results.dat"
    if not dat.is_file():
        return None
    calls, reason = parse_client_dat(dat.read_text(encoding="utf-8", errors="replace"))
    if calls is None or (reason is not None and reason != "0"):
        return None
    return calls


def benchmark_text(summary: dict, template: dict) -> str:
    lines = []
    for key in ("force_calls", "force_calls_min", "force_calls_max", "nfailed"):
        value = summary.get(key)
        if value is None:
            continue
        lines.append(f"{key} {value}")
    if template.get("code"):
        lines.append(f"code {template['code']}")
    if template.get("code_version"):
        lines.append(f"code_version {template['code_version']}")
    if template.get("algorithm"):
        lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    return "\n".join(lines) + "\n"


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    client: Path,
    geometries: Path | None = None,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
) -> dict:
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != "eon-client":
        raise ValueError(f"unknown engine {template['engine']}")
    cap = _check_stop(problem, template)
    ini = format_ini(template["ini"])
    if geometries is None:
        archive = geometry_archive(root, problem)
        dest = work or output.parent / "geometries"
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
        cons = sorted(dest.rglob("*.con"))
        if not cons:
            raise FileNotFoundError(f"{archive} has no .con files")
    else:
        cons = iter_geometries(geometries)
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    jobs = [
        (str(client), ini, str(con), str(scratch / con.stem))
        for con in cons
    ]
    if workers <= 1:
        calls = [_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            calls = list(pool.map(_one, jobs))
    # A missing result is a failure at the cap, so it stays out of the mean.
    counts = [cap if call is None else call for call in calls]
    summary = summarize_capped(counts, cap)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(benchmark_text(summary, template), encoding="utf-8")
    summary["starts"] = len(counts)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    return summary


