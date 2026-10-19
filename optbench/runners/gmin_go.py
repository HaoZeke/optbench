"""GMIN basin hopping on the global-optimisation problems.

The keywords are the Lennard-Jones and TIP4P examples in the GMIN user
guide, together with the temperature and step sizes from Wales and Doye,
J. Phys. Chem. A 1997, 101, 5111. The local minimiser is LBFGS, the
current GMIN default. EDIFF is the problem window, 0.001, and TARGET is
the Cambridge minimum, so a hit is the harness success rule.

The potential is GMIN's native Lennard-Jones, BLJCLUSTER_NOCUT, or TIP 4
with the DEFTIP4 parameters compiled into TIPID 4. NPCALL counts each
entry to POTENTIAL. The scored force-call count is that number on the
target hit or miss line, which is the first-encounter count. A matched
run writes every entry and replays it through the harness kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from optbench.campaign import geometry_archive, load_template
from optbench.failures import classify_failure
from optbench.globalopt import judge_globalopt, reference_energy
from optbench.problems import load_problem
from optbench.score import score_v2

REVISION = "df8da599267534fbe43a250e578f7396a7946311"
VERSION = REVISION[:12]
TARBALL_SHA = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
CODE = "GMIN"

SLUGS = (
    "lj38",
    "lj75",
    "lj98",
    "lj150",
    "lj100-binary",
    "tip4p10",
    "tip4p15",
    "tip4p20",
)
# Long searches first so a full queue spends its cores on them.
RUN_ORDER = (
    "tip4p20",
    "lj98",
    "tip4p15",
    "lj150",
    "lj100-binary",
    "lj75",
    "tip4p10",
    "lj38",
)

_HIT = re.compile(
    r"mc> target (hit|missed), quenches, function calls, time:\s*(\d+)\s+(\d+)\s+(\d+)"
)
_LOWEST = re.compile(r"Lowest Minimum\s+1\s+Energy=\s*(\S+)")
_QUENCH = re.compile(r"^\s*Qu\s+\d+\s+E=\s*(\S+)", re.MULTILINE)
_TOTAL = re.compile(r"finalio> Number of potential calls\s+(\d+)")


def template_id(slug: str) -> str:
    return f"gmin-globalopt/{slug}"


def stem_for(slug: str) -> str:
    return f"gmin-{VERSION}-{slug}-basinhopping"


def gmin_binary() -> Path | None:
    """The built GMIN executable, or None when this machine has not built it."""
    candidates = []
    env = os.environ.get("GMIN_BIN")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/opt/gmin/bin/GMIN"),
            Path("/tmp/gmin-build/GMIN"),
            Path(__file__).resolve().parents[2] / "codes" / "gmin" / "build" / "GMIN",
        ]
    )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def _fort_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def parse_gmin_output(text: str) -> dict:
    """Pull the first-encounter count and the tightest saved energy."""
    return _parse_gmin_lines(text.splitlines())


def _parse_gmin_lines(lines) -> dict:
    hit = None
    quenches = None
    force_calls = None
    energy = None
    potential_calls = None
    quench_energy = None
    for line in lines:
        matched = _HIT.search(line)
        if matched:
            hit = matched.group(1) == "hit"
            quenches = int(matched.group(3))
            force_calls = int(matched.group(4))
            continue
        lowest = _LOWEST.search(line)
        if lowest:
            energy = _fort_float(lowest.group(1))
            continue
        quench = _QUENCH.search(line)
        if quench:
            value = _fort_float(quench.group(1))
            quench_energy = value if quench_energy is None else min(quench_energy, value)
            continue
        total = _TOTAL.search(line)
        if total:
            potential_calls = int(total.group(1))
    if energy is None:
        energy = quench_energy
    return {
        "hit": hit,
        "quenches": quenches,
        "force_calls": force_calls,
        "energy": energy,
        "potential_calls": potential_calls,
    }


def _coordinate_text(text: str, lines: int) -> str:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and not _is_float(parts[0]):
            parts = parts[1:4]
        if len(parts) < 3 or not all(_is_float(part) for part in parts[:3]):
            continue
        rows.append(" ".join(f"{float(part):.16e}" for part in parts[:3]))
    if len(rows) != lines:
        raise ValueError(f"expected {lines} coordinate lines, read {len(rows)}")
    return "\n".join(rows) + "\n"


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _member_index(name: str, kind: str) -> int | None:
    base = name.rstrip("/").rsplit("/", 1)[-1]
    if kind == "xyz":
        if not base.endswith(".xyz"):
            return None
        stem = base[: -len(".xyz")]
    elif kind == "coords":
        prefix = "coords."
        if not base.startswith(prefix):
            return None
        stem = base[len(prefix) :]
    elif kind == "aa":
        if not base.endswith(".aa"):
            return None
        stem = base[: -len(".aa")]
    else:
        raise ValueError(kind)
    if not stem.isdigit():
        return None
    return int(stem)


def load_starts(archive: Path, *, kind: str, lines: int) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    with tarfile.open(archive, "r:gz") as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            index = _member_index(member.name, kind)
            if index is None:
                continue
            raw = packed.extractfile(member)
            if raw is None:
                continue
            text = raw.read().decode("utf-8", errors="replace")
            found.append((index, _coordinate_text(text, lines)))
    found.sort(key=lambda item: item[0])
    return found


def data_text(gmin: dict, *, index: int, reference: float, cap: int | None) -> str:
    """One GMIN data file. The keywords are the published example for this potential."""
    lines = [
        f"SLOPPYCONV {gmin['sloppy_rms']}",
        f"TIGHTCONV {gmin['tight_rms']}",
    ]
    potential = str(gmin["potential"])
    if potential == "blj":
        lines.append(
            "BLJCLUSTER_NOCUT "
            f"{int(gmin['n_A'])} {gmin['epsilon_ab']} {gmin['epsilon_bb']} "
            f"{gmin['sigma_ab']} {gmin['sigma_bb']}"
        )
    elif potential == "tip4p":
        lines.append("TIP 4")
    lines.append("CENTRE")
    if potential != "tip4p":
        lines.append("SORT")
    lines.extend(
        [
            f"MAXIT {int(gmin['maxit_sloppy'])} {int(gmin['maxit_tight'])}",
            f"STEPS {int(gmin['steps'])} {gmin['tfac']}",
        ]
    )
    if potential == "tip4p":
        lines.append(
            f"STEP {gmin['step']} 0.0 {gmin['orient_step']} {int(gmin['block'])}"
        )
    else:
        lines.append(f"STEP {gmin['step']} {gmin['angular_step']}")
    lines.extend(
        [
            f"TEMPERATURE {gmin['temperature']}",
            f"EDIFF {gmin['ediff']}",
            f"TARGET {reference:.12f}",
            f"RANSEED {int(index)}",
            "SUPPRESS",
        ]
    )
    if cap is not None:
        lines.append(f"MAX_NPCALL {int(cap)}")
    return "\n".join(lines) + "\n"


def _spec(root: Path, slug: str) -> dict:
    problem = load_problem(root / "problems" / "globalopt" / f"{slug}.toml")
    template = load_template(root, template_id(slug))
    if template.get("problem") != problem["id"]:
        raise ValueError(f"{template_id(slug)} does not belong to {problem['id']}")
    gmin = template["gmin"]
    reference = reference_energy(problem)
    tolerance = float(problem["convergence"]["energy_above_minimum"])
    stated = problem["convergence"].get("max_force_calls")
    cap = None if stated in (None, "") else int(stated)
    stop = template.get("stop") or {}
    if abs(float(stop["energy_above_minimum"]) - tolerance) > 0.0:
        raise ValueError("template energy window differs from the problem")
    if cap is None and "max_force_calls" in stop:
        raise ValueError("lj38 states no force-call cap")
    if cap is not None and int(stop["max_force_calls"]) != cap:
        raise ValueError("template cap differs from the problem cap")
    if abs(float(gmin["ediff"]) - tolerance) > 0.0:
        raise ValueError("EDIFF is the problem energy window")
    return {
        "slug": slug,
        "problem": problem,
        "template": template,
        "gmin": gmin,
        "reference": reference,
        "tolerance": tolerance,
        "cap": cap,
        "kind": str(gmin["archive"]),
        "lines": int(gmin["coordinate_lines"]),
        "atoms": int(gmin["atoms"]),
    }


def _judge(energy: float | None, calls: int, spec: dict, *, crashed: bool) -> dict:
    if crashed:
        finite = energy is not None and np.isfinite(energy)
        return {
            "success": False,
            "failure": "crash",
            "reasons": ["crash"],
            "energy": float(energy) if finite else None,
            "force_calls": float(calls),
        }
    if spec["cap"] is None:
        reasons: list[str] = []
        finite = energy is not None and np.isfinite(float(energy))
        if not finite:
            reasons.append("divergence")
        elif float(energy) > spec["reference"] + spec["tolerance"]:
            reasons.append("energy")
        failure = classify_failure(reasons)
        return {
            "success": failure is None,
            "failure": failure,
            "reasons": reasons,
            "energy": float(energy) if finite else None,
            "force_calls": float(calls),
        }
    return judge_globalopt(
        energy,
        calls,
        reference=spec["reference"],
        tolerance=spec["tolerance"],
        cap=spec["cap"],
    )


def search_one(coords: str, *, index: int, slug: str, root: Path = Path(".")) -> dict:
    """Run one archived start. ``index`` is the GMIN RANSEED."""
    spec = _spec(root, slug)
    binary = gmin_binary()
    if binary is None:
        raise RuntimeError("GMIN is not built; run codes/gmin/build.sh")
    work = Path(os.environ.get("TMPDIR", "/tmp")) / f"gmin-go-{slug}-{index}-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "coords").write_text(coords, encoding="utf-8")
    (work / "data").write_text(
        data_text(spec["gmin"], index=index, reference=spec["reference"], cap=spec["cap"]),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.pop("OPTBENCH_GMIN_TRACE", None)
    started = time.perf_counter()
    crashed = False
    error = ""
    log_path = work / "gmin.out"
    parsed = {
        "hit": None,
        "quenches": None,
        "force_calls": None,
        "energy": None,
        "potential_calls": None,
    }
    try:
        import subprocess

        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(binary)],
                cwd=work,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            error = f"exit {completed.returncode}"
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            parsed = _parse_gmin_lines(handle)
    except Exception as exc:
        crashed = True
        error = exc.__class__.__name__
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if parsed["force_calls"] is None:
        crashed = True
        error = error or "no target line"
        calls = int(parsed["potential_calls"] or 0)
    else:
        calls = int(parsed["force_calls"])
        if error and not crashed:
            error = ""
    judged = _judge(parsed["energy"], calls, spec, crashed=crashed)
    return {
        "schema": "optbench.start/v2",
        "start": f"coords.{index}",
        "problem": spec["problem"]["id"],
        "template": template_id(slug),
        "code": CODE,
        "code_version": VERSION,
        "commit": REVISION,
        "tuning": "default",
        "counter": "harness",
        "comparable": True,
        "seed": int(index),
        "success": judged["success"],
        "failure": judged["failure"],
        "reasons": judged["reasons"],
        "force_calls": calls,
        "potential_calls": parsed["potential_calls"],
        "energy": judged["energy"],
        "gmin_hit": parsed["hit"],
        "error": error,
        "seconds": time.perf_counter() - started,
    }


def load_campaign_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _worker(payload: tuple[str, int, str, str]) -> dict:
    coords, index, slug, root = payload
    return search_one(coords, index=index, slug=slug, root=Path(root))


def run(
    root: Path,
    slug: str,
    *,
    limit: int | None = None,
    workers: int | None = None,
    dest: Path | None = None,
    prior: list[dict] | None = None,
) -> list[dict]:
    spec = _spec(root, slug)
    problem = spec["problem"]
    starts = load_starts(geometry_archive(root, problem), kind=spec["kind"], lines=spec["lines"])
    expected = int(problem["sample"]["starts"])
    if len(starts) != expected:
        raise RuntimeError(f"{slug} archive has {len(starts)} starts, problem states {expected}")
    records = [dict(record) for record in prior or []]
    done = {int(str(record["start"]).split(".")[1]) for record in records}
    selected = [item for item in (starts if limit is None else starts[:limit]) if item[0] not in done]
    payload = [(coords, index, slug, str(root)) for index, coords in selected]
    if workers is None:
        workers = os.cpu_count() or 1
    campaign_root = root if dest is None else dest
    jsonl = campaign_root / "results" / "campaign" / f"{stem_for(slug)}.starts.jsonl"
    if prior is None:
        jsonl.unlink(missing_ok=True)
        jsonl.with_name(f"{stem_for(slug)}.dat").unlink(missing_ok=True)
    # A full PAR-2 bootstrap on every start dominates the short searches.
    # The jsonl is appended immediately; the summary is rewritten in batches.
    summary_every = 50

    def _store(record: dict) -> None:
        records.append(record)
        _report(record)
        _append_jsonl(jsonl, record)
        if len(records) % summary_every == 0 or len(records) >= expected:
            _rewrite(records)

    def _rewrite(rows: list[dict]) -> None:
        ordered = sorted(rows, key=lambda item: int(str(item["start"]).split(".")[1]))
        try:
            write_campaign(campaign_root, slug, ordered, partial=len(ordered) < expected, spec=spec)
        except ValueError as exc:
            print(f"summary deferred for {slug}: {exc}", flush=True)

    if not payload:
        return sorted(records, key=lambda record: int(str(record["start"]).split(".")[1]))
    if workers <= 1 or len(payload) == 1:
        for item in payload:
            _store(_worker(item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, item) for item in payload]
            for future in as_completed(futures):
                _store(future.result())
    records.sort(key=lambda record: int(str(record["start"]).split(".")[1]))
    _rewrite(records)
    return records


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    slug = str(problem["id"]).split("/", 1)[1]
    if template.get("id") != template_id(slug):
        raise ValueError(f"{template.get('id')} is not {template_id(slug)}")
    prior = None
    if resume:
        prior = load_campaign_records(root / "results" / "campaign" / f"{stem_for(slug)}.starts.jsonl")
    records = run(root, slug, limit=limit or None, workers=workers, prior=prior)
    spec = _spec(root, slug)
    expected = int(problem["sample"]["starts"])
    dat, _jsonl = write_campaign(root, slug, records, partial=len(records) < expected, spec=spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dat.read_text(encoding="utf-8"), encoding="utf-8")
    summary = score_v2(records, spec["cap"])
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    return summary


def _report(record: dict) -> None:
    print(
        f"{record['problem']} {record['start']} calls={record['force_calls']} "
        f"energy={record['energy']} success={record['success']} failure={record['failure']}",
        flush=True,
    )


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_campaign(
    root: Path,
    slug: str,
    records: list[dict],
    *,
    partial: bool,
    spec: dict,
) -> tuple[Path, Path]:
    directory = root / "results" / "campaign"
    directory.mkdir(parents=True, exist_ok=True)
    jsonl = directory / f"{stem_for(slug)}.starts.jsonl"
    _atomic_write(jsonl, "".join(json.dumps(record) + "\n" for record in records))
    summary = score_v2(records, spec["cap"])
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    calls = [int(record["force_calls"]) for record in records]
    gmin = spec["gmin"]
    cap_line = "no cap" if spec["cap"] is None else str(spec["cap"])
    lines = [
        f"# GMIN {VERSION} basin hopping on {spec['problem']['id']}.",
        "# Wales and Doye, J. Phys. Chem. A 1997, 101, 5111, and the GMIN user-guide examples.",
        f"# TEMPERATURE {gmin['temperature']}, STEP {gmin['step']}, "
        f"SLOPPYCONV {gmin['sloppy_rms']}, TIGHTCONV {gmin['tight_rms']}, "
        f"MAXIT {gmin['maxit_sloppy']} {gmin['maxit_tight']}, STEPS {gmin['steps']} {gmin['tfac']}.",
        "# force_calls is NPCALL on the target hit or miss line.",
        f"# The problem cap is {cap_line}. EDIFF and the energy window are {spec['tolerance']}.",
        f"problem {spec['problem']['id']}",
        f"template {template_id(slug)}",
        f"code {CODE}",
        f"code_version {VERSION}",
        f"commit {REVISION}",
        "algorithm basin-hopping",
        "tuning default",
        "counter harness",
        f"n {summary['n']}",
        f"nfailed {summary['nfailed']}",
    ]
    if summary["par2_mean"] is not None:
        lines.append(f"par2_mean {_format(summary['par2_mean'])}")
    lines.extend(
        [
        f"median {_format(summary['median'])}",
        f"median_low {_format(summary['median_low'])}",
        f"median_high {_format(summary['median_high'])}",
        f"wilson_low {_format(summary['wilson_low'])}",
        f"wilson_high {_format(summary['wilson_high'])}",
        f"success_rate {_format(summary['success_rate'])}",
        f"n_cap {summary['n_cap']}",
        f"n_wrong_stationary {summary['n_wrong_stationary']}",
        f"n_divergence {summary['n_divergence']}",
        f"n_crash {summary['n_crash']}",
        f"force_calls_min {min(calls)}",
        f"force_calls_max {max(calls)}",
        f"force_calls_median {_format(float(np.median(calls)))}",
        ]
    )
    if spec["cap"] is not None:
        lines.append(f"cap {spec['cap']}")
    if energies:
        lines.append(f"energy_min {_format(min(energies))}")
        lines.append(f"energy_max {_format(max(energies))}")
    if partial:
        lines.append("partial true")
    dat = directory / f"{stem_for(slug)}.dat"
    _atomic_write(dat, "\n".join(lines) + "\n")
    return dat, jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GMIN basin hopping")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--problem", default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    slugs = list(RUN_ORDER) if args.problem == "all" else [args.problem]
    limit = args.limit or None
    workers = args.workers or None
    for slug in slugs:
        jsonl = root / "results" / "campaign" / f"{stem_for(slug)}.starts.jsonl"
        prior = load_campaign_records(jsonl) if args.resume else None
        if prior:
            print(f"resume {slug} {len(prior)} starts from {jsonl}", flush=True)
        records = run(root, slug, limit=limit, workers=workers, prior=prior)
        spec = _spec(root, slug)
        expected = int(spec["problem"]["sample"]["starts"])
        summary = score_v2(records, spec["cap"])
        print(
            f"wrote {stem_for(slug)} n={summary['n']} successes={summary['successes']} "
            f"par2_mean={summary['par2_mean']} partial={len(records) < expected}",
            flush=True,
        )


if __name__ == "__main__":
    main()
