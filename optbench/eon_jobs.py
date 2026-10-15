"""eOn jobs whose starts are not one minimization geometry.

Minimization uses :func:`optbench.campaign.run`. The jobs here follow the
archived ``report.sh`` rule. A ``total_force_calls`` or ``force_calls_neb``
value under the cap stays in the mean. ``nfailed`` counts a
``termination_reason`` other than 0, including a missing ``results.dat``.
The Pt-heptamer mean divides the band counter by the image count, which is
what ``archive/entries/neb/Pt-heptamer/eon-lbfgs/report.sh`` does.
"""

from __future__ import annotations

import json
import random
import shutil
import statistics
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .campaign import (
    benchmark_text,
    client_result,
    format_ini,
    geometry_archive,
    load_template,
    write_starts,
)


def report_summary(rows: list[dict], cap: int, images: int = 1) -> dict:
    """Reduce one archived report.

    ``rows`` carry ``raw`` (the counter named by the report) and ``reason``.
    Values at or above ``cap`` stay out of the mean. The heptamer report
    divides by ``images`` after the sum. Min and max are the per-start
    values on that same scale, rounded with half-even when images are
    divided out.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    if images <= 0:
        raise ValueError("images must be positive")
    failed = 0
    kept: list[float] = []
    for row in rows:
        raw = row.get("raw")
        reason = row.get("reason")
        if raw is None or (reason is not None and reason != "0"):
            failed += 1
        if raw is not None and float(raw) < cap:
            kept.append(float(raw))
    if not kept:
        return {
            "force_calls": None,
            "force_calls_min": None,
            "force_calls_max": None,
            "force_calls_median": None,
            "nfailed": failed,
        }
    scaled = [value / images for value in kept]
    if images == 1:
        low = int(min(kept))
        high = int(max(kept))
    else:
        low = round(min(scaled))
        high = round(max(scaled))
    return {
        "force_calls": round(sum(kept) / len(kept) / images),
        "force_calls_min": low,
        "force_calls_max": high,
        "force_calls_median": statistics.median(scaled),
        "nfailed": failed,
    }


def _read_dat(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in found:
            found[parts[1]] = parts[0]
    return found


def _execute(spec: tuple) -> dict:
    client, ini, work, copies, texts, count_key = spec
    dest = Path(work)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "config.ini").write_text(ini, encoding="utf-8")
    for src, name in copies:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, target)
    for name, text in texts:
        (dest / name).write_text(text, encoding="utf-8")
    proc = subprocess.run(
        [client],
        cwd=dest,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0 or not (dest / "results.dat").is_file():
        (dest / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    fields = _read_dat(dest / "results.dat")
    base = client_result(dest)
    raw_text = fields.get(count_key)
    raw = int(float(raw_text)) if raw_text is not None else None
    return {
        "geometry": dest.name,
        "force_calls": base["force_calls"],
        "raw": raw,
        "reason": base["reason"] if fields else None,
        "count_key": count_key,
    }


def _run_specs(
    client: Path,
    ini: str,
    specs: list[tuple],
    workers: int,
) -> list[dict]:
    jobs = [
        (str(client), ini, work, copies, texts, count_key)
        for work, copies, texts, count_key in specs
    ]
    if workers <= 1:
        return [_execute(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_execute, jobs))


def _stored(root: Path, archive_path: str) -> Path:
    return geometry_archive(root, {"id": archive_path, "input": {"archive_path": archive_path}})


def _extract(archive: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    return dest


def _coords(text: str) -> list[tuple[float, float, float]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower().startswith("coordinates"))
    points: list[tuple[float, float, float]] = []
    for line in lines[start + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            if points:
                break
            continue
        try:
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            if points:
                break
    if not points:
        raise ValueError("no coordinates")
    return points


def _fmt_mode(rows: list[tuple[float, float, float]]) -> str:
    return "".join(f"{x:.16e} {y:.16e} {z:.16e}\n" for x, y, z in rows)


def _seeded_mode(natoms: int, seed: int) -> str:
    """Python ``random.Random(seed)``, three draws in [0, 1) per atom."""
    rng = random.Random(seed)
    rows = [(rng.random(), rng.random(), rng.random()) for _ in range(natoms)]
    return _fmt_mode(rows)


def _difference_mode(reactant: Path, displacement: Path) -> str:
    left = _coords(reactant.read_text(encoding="utf-8", errors="replace"))
    right = _coords(displacement.read_text(encoding="utf-8", errors="replace"))
    if len(left) != len(right):
        raise ValueError(f"{displacement} has {len(right)} atoms, reactant has {len(left)}")
    rows = [(b[0] - a[0], b[1] - a[1], b[2] - a[2]) for a, b in zip(left, right)]
    norm = sum(x * x + y * y + z * z for x, y, z in rows) ** 0.5
    if norm == 0:
        raise ValueError(f"{displacement} matches the reactant")
    return _fmt_mode([(x / norm, y / norm, z / norm) for x, y, z in rows])


def _finish(
    template: dict,
    rows: list[dict],
    output: Path,
    images: int,
) -> dict:
    cap = int(template["failure_cap"])
    summary = report_summary(rows, cap, images)
    summary_for_text = dict(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(benchmark_text(summary_for_text, template), encoding="utf-8")
    start_rows = []
    for row in rows:
        raw = row["raw"]
        reason = row["reason"]
        bad = raw is None or (reason is not None and reason != "0") or (raw is not None and raw >= cap)
        recorded = None if raw is None else raw / images
        start_rows.append(
            {
                "failed": bad,
                "force_calls": recorded,
                "geometry": row["geometry"],
                "raw": raw,
                "reason": reason,
                "count_key": row.get("count_key"),
            }
        )
    write_starts(output.with_suffix(".starts"), start_rows)
    summary["starts"] = len(rows)
    summary["template"] = template["id"]
    summary["problem"] = template.get("problem")
    return summary


def _template(root: Path, name: str) -> tuple[dict, str]:
    template = load_template(root, name)
    return template, format_ini(template["ini"])


def run_neb_pt(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    images = int(template["ini"]["Nudged Elastic Band"]["images"])
    unpacked = _extract(_stored(root, "website/neb/Pt-heptamer/eon-lbfgs/minima.tgz"), work / "unpacked")
    minima = unpacked / "minima"
    specs = []
    for src in sorted(path for path in minima.iterdir() if path.is_dir() and path.name.startswith("min")):
        specs.append(
            (
                str(work / "runs" / src.name),
                [(str(src / "a.con"), "reactant.con"), (str(src / "b.con"), "product.con")],
                [],
                "force_calls_neb",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, images)


def run_de_pt(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(
        _stored(root, "website/double-ended-saddle-search/pt-island/pt-island-con.tgz"),
        work / "unpacked",
    )
    folder = unpacked / "pt-island-con"
    specs = []
    for index in range(59):
        product = folder / f"product_{index}.con"
        specs.append(
            (
                str(work / "runs" / f"product_{index}"),
                [(str(folder / "reactant.con"), "reactant.con"), (str(product), "product.con")],
                [],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


def run_de_lj38(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(
        _stored(root, "website/double-ended-saddle-search/lj38/coords-con.tgz"),
        work / "unpacked",
    )
    folder = unpacked / "coords-con"
    lammps = root / "archive/entries/double-ended-saddle-search/lj38/eon-ci-neb/in.lammps"
    specs = []
    for index in range(50):
        specs.append(
            (
                str(work / "runs" / f"pair_{index:02d}"),
                [
                    (str(folder / f"start_{index:02d}.con"), "reactant.con"),
                    (str(folder / f"end_{index:02d}.con"), "product.con"),
                    (str(lammps), "in.lammps"),
                ],
                [],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


def run_ts_lj38(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(_stored(root, "website/tsopt/LJ38/coords-con.tar.gz"), work / "unpacked")
    folder = unpacked / "coords-con"
    specs = []
    for index in range(200):
        con = folder / f"{index:04d}.con"
        text = con.read_text(encoding="utf-8", errors="replace")
        mode = _seeded_mode(len(_coords(text)), index)
        specs.append(
            (
                str(work / "runs" / f"{index:04d}"),
                [(str(con), "pos.con"), (str(con), "displacement.con")],
                [("direction.dat", mode)],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


def run_ts_pt(root: Path, template_name: str, client: Path, output: Path, work: Path, workers: int) -> dict:
    template, ini = _template(root, template_name)
    unpacked = _extract(_stored(root, "website/tsopt/pt-island/pt-island-con.tgz"), work / "unpacked")
    folder = unpacked / "pt-island-con"
    reactant = folder / "reactant.con"
    specs = []
    for index in range(49):
        initial = folder / f"initial_{index}.con"
        mode = _difference_mode(reactant, initial)
        specs.append(
            (
                str(work / "runs" / f"initial_{index}"),
                [(str(reactant), "pos.con"), (str(initial), "displacement.con")],
                [("direction.dat", mode)],
                "total_force_calls",
            )
        )
    rows = _run_specs(client, ini, specs, workers)
    return _finish(template, rows, output, 1)


JOBS = {
    "neb-pt-lbfgs": ("eon/neb-pt-lbfgs", run_neb_pt),
    "neb-pt-lbfgs-m100": ("eon/neb-pt-lbfgs-m100", run_neb_pt),
    "de-pt-lbfgs": ("eon/de-pt-lbfgs", run_de_pt),
    "de-pt-lbfgs-m100": ("eon/de-pt-lbfgs-m100", run_de_pt),
    "de-lj38-lbfgs": ("eon/de-lj38-lbfgs", run_de_lj38),
    "de-lj38-lbfgs-m100": ("eon/de-lj38-lbfgs-m100", run_de_lj38),
    "ts-lj38-dimer": ("eon/ts-lj38-dimer", run_ts_lj38),
    "ts-lj38-dimer-m100": ("eon/ts-lj38-dimer-m100", run_ts_lj38),
    "ts-pt-dimer": ("eon/ts-pt-dimer", run_ts_pt),
    "ts-pt-dimer-m100": ("eon/ts-pt-dimer-m100", run_ts_pt),
}


def run_named(
    root: Path,
    name: str,
    client: Path,
    output: Path,
    *,
    workers: int = 1,
    work: Path | None = None,
) -> dict:
    if name not in JOBS:
        known = ", ".join(sorted(JOBS))
        raise SystemExit(f"unknown eon job {name}. known: {known}")
    template_name, fn = JOBS[name]
    scratch = work or output.parent / "work" / name
    summary = fn(root, template_name, client, output, scratch, workers)
    print(
        f"{summary['problem']} {summary['template']} "
        f"force_calls={summary['force_calls']} nfailed={summary['nfailed']} "
        f"starts={summary['starts']}"
    )
    return summary


def result_document(template: dict, summary: dict, starts_path: Path, host: str, date: str) -> dict:
    """One optbench.result/v1 document for a finished job."""
    rows = [json.loads(line) for line in starts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = {
        key: summary[key]
        for key in ("force_calls", "force_calls_min", "force_calls_max", "force_calls_median", "nfailed")
        if summary.get(key) is not None
    }
    return {
        "schema": "optbench.result/v1",
        "status": "sota",
        "problem": template.get("problem"),
        "submitter": "Rohit Goswami",
        "date": date,
        "host": host,
        "runs": [
            {
                "entry": starts_path.with_suffix("").name,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
                "present_source": True,
                "starts": len(rows),
                "nfailed": int(summary.get("nfailed") or 0),
                "method": {
                    "client": "eonclient",
                    "reducer": template.get("reducer"),
                    "ini": template.get("ini"),
                    "note": template.get("note"),
                },
                "metrics": metrics,
                "per_start": rows,
                "notes": template.get("note", ""),
            }
        ],
    }
