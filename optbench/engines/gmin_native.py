"""GMIN basin-hopping for the LJ98 core-orbits symmetrisation row.

The binary is GMIN from the Wales snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. ``GMIN/source/SVNREV`` is the version string. The potential
is GMIN's native Lennard-Jones, not the harness socket. The score is
the function-call count on the target-hit line, the counter
``run_gmin.sh`` stores as energy calls.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import tarfile
import tempfile
from pathlib import Path

from .common import run_jobs, write_campaign

ENGINE = "gmin"
TARBALL = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
TARBALL_SHA256 = "738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
TARBALL_BYTES = 122737951
VERSION = "df8da599267534fbe43a250e578f7396a7946311"
# Published STEPS 100000 times MAXIT 1000. GMIN stops on the target, not on this ceiling.
HARNESS_CAP = 100_000_000
RANKED_TEMPLATE = "gmin/lj98-symmetrise"
CAMPAIGN_NAME = "gmin-df8da599-lj98-symmetrise"
_DATA = Path(__file__).resolve().parents[2] / "templates" / "gmin" / "preambles" / "lj98-symmetrise.data"
_HIT = re.compile(r"Target hit after\s+(\d+)\s+quenches\s+(\d+)\s+function calls")
_CPU = re.compile(r"time elapsed\s+([0-9.]+)")
_ARCHIVE_FIELDS = (
    "basin_hopping_steps_mean",
    "basin_hopping_steps_stddev",
    "cpu_time_mean",
    "cpu_time_stddev",
    "energy_calls_mean",
    "energy_calls_stddev",
    "nfailed",
)


def binary_path() -> Path:
    env = os.environ.get("GMIN_BIN")
    if env:
        return Path(env)
    return Path("/tmp/gmin-build/GMIN")


def _missing(binary: Path) -> str:
    return (
        "GMIN is not available. "
        f"The pin is the sha256 {TARBALL_SHA256} of {TARBALL} "
        f"({TARBALL_BYTES} bytes), SVNREV {VERSION}. "
        f"{binary} is not present. "
        "The LJ98 symmetrise score is GMIN's function calls."
    )


def available() -> str | None:
    binary = binary_path()
    if binary.is_file() and _DATA.is_file():
        return None
    missing = binary if not binary.is_file() else _DATA
    return _missing(missing)


def paper_calls(log: str) -> tuple[int | None, int | None, float | None]:
    """``run_gmin.sh``: last target-hit quenches and function calls, and CPU_TIME."""
    quenches = None
    calls = None
    cpu = None
    for line in log.splitlines():
        hit = _HIT.search(line)
        if hit:
            quenches = int(hit.group(1))
            calls = int(hit.group(2))
        elapsed = _CPU.search(line)
        if elapsed:
            cpu = float(elapsed.group(1))
    return quenches, calls, cpu


def record_from_log(name: str, log: str, cap: int, *, returncode: int = 0) -> dict:
    """One start. A target-hit line under the cap is a success."""
    quenches, calls, cpu = paper_calls(log)
    success = calls is not None and quenches is not None and calls < cap
    if success:
        failure = None
    elif calls is None:
        failure = "crash" if returncode else "divergence"
    else:
        failure = "cap"
    record = {
        "start": name,
        "force_calls": int(calls) if calls is not None else int(cap),
        "success": success,
        "failure": failure,
        "counter": "harness",
        "converged": success,
    }
    if quenches is not None:
        record["basin_hopping_steps"] = int(quenches)
    if cpu is not None:
        record["cpu_time"] = float(cpu)
    if calls is not None:
        record["gmin_function_calls"] = int(calls)
    return record


def _number(value: float) -> str:
    return format(float(value), ".16g")


def comparison_fields(records: list[dict]) -> dict[str, str]:
    """Archive columns. Standard deviations divide by n, the published set."""
    hits = [row for row in records if row["success"]]
    if any("basin_hopping_steps" not in row or "cpu_time" not in row for row in hits):
        raise ValueError("a target hit is missing quenches or CPU_TIME")
    steps = [int(row["basin_hopping_steps"]) for row in hits]
    calls = [int(row["gmin_function_calls"]) for row in hits]
    cpus = [float(row["cpu_time"]) for row in hits]
    failed = len(records) - len(hits)
    return {
        "basin_hopping_steps_mean": _number(statistics.fmean(steps)),
        "basin_hopping_steps_stddev": _number(statistics.pstdev(steps)),
        "cpu_time_mean": _number(statistics.fmean(cpus)),
        "cpu_time_stddev": _number(statistics.pstdev(cpus)),
        "energy_calls_mean": _number(statistics.fmean(calls)),
        "energy_calls_stddev": _number(statistics.pstdev(calls)),
        "nfailed": str(failed),
    }


def comparison_text(records: list[dict]) -> str:
    fields = comparison_fields(records)
    lines = [
        f"# GMIN SVNREV {VERSION}. Native LJ, published symmetrise data, 100 lj98 coords.",
        "# Population standard deviation over the starts that hit the target.",
        f"# Ranked harness campaign: results/campaign/{CAMPAIGN_NAME}.dat",
    ]
    for name in _ARCHIVE_FIELDS:
        lines.append(f"{name} {fields[name]}")
    lines.append(f"code_version {VERSION}")
    lines.append(f"commit {TARBALL_SHA256}")
    lines.append(f"n {len(records)}")
    return "\n".join(lines) + "\n"


def _coord_index(name: str) -> int | None:
    base = Path(name).name
    if not base.startswith("coords."):
        return None
    try:
        return int(base.split(".", 1)[1])
    except ValueError:
        return None


def read_coords(archive: Path) -> list[tuple[str, bytes]]:
    """Numbered ``coords.N`` members, in start order."""
    found: list[tuple[int, str, bytes]] = []
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            index = _coord_index(member.name)
            if index is None:
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            found.append((index, f"coords.{index}", extracted.read()))
    found.sort(key=lambda item: item[0])
    return [(name, payload) for _index, name, payload in found]


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name = str(job["name"])
    cap = int(job["cap"])
    binary = Path(job["binary"])
    log = ""
    returncode = 1
    try:
        with tempfile.TemporaryDirectory(prefix="gmin-") as raw:
            folder = Path(raw)
            (folder / "data").write_bytes(job["data"])
            (folder / "coords").write_bytes(job["coords"])
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.run(
                    [str(binary)],
                    cwd=folder,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=7200,
                    check=False,
                )
            log = stdout_path.read_text(encoding="utf-8", errors="replace")
            returncode = int(proc.returncode)
    except Exception as exc:
        log = f"{type(exc).__name__}: {exc}"
        returncode = 1
    return record_from_log(name, log, cap, returncode=returncode)


def _comment() -> str:
    return (
        f"# GMIN SVNREV {VERSION}, svn.tar.bz2 sha256 {TARBALL_SHA256}.\n"
        "# Native Lennard-Jones, the published symmetrise data file, the 100 lj98 coords.\n"
        "# The count is GMIN's function calls on the target-hit line."
    )


def _publish(output: Path, records: list[dict], template: dict, problem: dict, cap: int) -> dict:
    published = dict(template)
    published["code"] = "GMIN"
    summary = write_campaign(
        output,
        records,
        published,
        problem,
        {"cap": cap},
        {
            "method": "symmetrise",
            "keyword": "SYMMETRISE",
            "counter": "gmin function calls",
            "potential": "lennard-jones",
            "version": VERSION,
            "tarball_sha256": TARBALL_SHA256,
            "ediff": 0.003,
            "target": -543.665361,
            "sloppyconv": 0.007,
            "tightconv": 1e-5,
        },
        images=None,
        comment=_comment(),
    )
    comparison = output.parents[1] / "repro" / f"{CAMPAIGN_NAME}.dat"
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_text(comparison_text(records), encoding="utf-8")
    summary["comparison"] = str(comparison)
    return summary


def publish_logs(root: Path, log_dir: Path, output: Path) -> dict:
    """Score stdout logs already produced by this GMIN binary."""
    from optbench.campaign import _check_stop, load_template, resolve_problem
    from optbench.engines.common import registry_pin, require_pin

    root = Path(root)
    template = load_template(root, RANKED_TEMPLATE)
    problem = resolve_problem(root, str(template["problem"]))
    require_pin(template, registry_pin(root, "gmin"))
    cap = _check_stop(problem, template)
    if cap is None:
        raise RuntimeError(f"{RANKED_TEMPLATE} has no harness cap")
    records = []
    for index in range(1, 101):
        path = Path(log_dir) / f"run-{index:03d}" / "stdout.txt"
        log = path.read_text(encoding="utf-8", errors="replace")
        records.append(record_from_log(f"coords.{index}", log, int(cap)))
    if len(records) != 100:
        raise RuntimeError(f"{log_dir} produced {len(records)} starts")
    return _publish(output, records, template, problem, int(cap))


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    """Run the latest GMIN once per archived coordinate file."""
    from optbench.campaign import _check_stop, geometry_archive, load_template, resolve_problem
    from optbench.engines.common import registry_pin, require_pin

    root = Path(root)
    template = load_template(root, template_name)
    problem = resolve_problem(root, problem_name)
    require_pin(template, registry_pin(root, "gmin"))
    if template.get("engine") != "gmin":
        raise ValueError(f"{template_name} engine is {template.get('engine')}")
    stated = template.get("problem")
    if stated and stated != problem_name:
        raise ValueError(f"{template_name} belongs to {stated}")
    binary = binary_path()
    if not binary.is_file() or not _DATA.is_file():
        raise RuntimeError(_missing(binary if not binary.is_file() else _DATA))
    cap = _check_stop(problem, template)
    if cap is None:
        raise RuntimeError(f"{template_name} has no harness cap")
    data = _DATA.read_bytes()
    coords = read_coords(geometry_archive(root, problem))
    sample = int((problem.get("sample") or {}).get("starts") or 0)
    if sample and len(coords) != sample:
        raise RuntimeError(f"{problem['id']} has {len(coords)} coords and sample.starts {sample}")
    jobs = [
        {
            "name": name,
            "cap": int(cap),
            "binary": str(binary),
            "data": data,
            "coords": payload,
        }
        for name, payload in coords
    ]
    records = run_jobs(jobs, _execute, workers)
    return _publish(output, records, template, problem, int(cap))
