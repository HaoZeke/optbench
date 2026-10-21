"""Basin-hopping and parallel-tempering basin-hopping with GMIN.

GMIN's own Lennard-Jones stays in the binary for an unset
``OPTBENCH_POTPORT``. A scored run sets the port. ``LJ`` then returns the
energy and gradient of the problem potential and each request is one
harness call. The server is the call count. GMIN's NPCALL is not the
score. The problem ``[potential]`` picks the kernel: ``lennard-jones``
is the compiled harness LJ, and ``binary-lennard-jones`` is
``optbench.potentials.binary_lennard_jones`` with the first ``n_A``
atoms of each start as species A and the rest as species B.

``failure_cap`` is the problem ``max_force_calls`` and TARGET is the
problem reference energy. A start succeeds when a harness evaluation is
at or below the reference energy plus ``energy_above_minimum`` with a
call count under the cap. The count at that first evaluation is the
start's force calls, the same rule as ``optbench.runners.anneal_go``.
The Cambridge points are checked to be a minimum of the harness kernel
inside that window before any start runs.
"""

from __future__ import annotations

import json
import math
import os
import functools
import shutil
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import HarnessCounter
from optbench.failures import classify_failure
from optbench.globalopt import energy_tolerance, force_cap, judge_globalopt, reference_energy
from optbench.gmin_pot import lennard_jones_compiled
from optbench.potentials import binary_lennard_jones, lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

TARBALL_SHA256 = "0e4a022bcdd43cef9dfdda9dad7fab69e1f6211b8917d96b360d6ce3862d0d9c"
PIN_REF = "0e4a022bcdd4"
CODE_VERSION = "df8da599"
CODE_NAME = "GMIN"


def gmin_binary() -> Path:
    env = os.environ.get("OPTBENCH_GMIN", "")
    path = Path(env) if env else Path(shutil.which("GMIN") or "")
    if not path.is_file():
        raise RuntimeError("set OPTBENCH_GMIN to the GMIN binary built by codes/gmin/build-bh.sh")
    return path


def potential_spec(problem: dict) -> dict:
    """The harness kernel named by the problem ``[potential]`` table."""
    section = problem.get("potential") or {}
    kind = str(section.get("kind") or "lennard-jones")
    if kind == "lennard-jones":
        return {"kind": kind, "epsilon": 1.0, "sigma": 1.0, "cutoff": 40.0}
    if kind == "binary-lennard-jones":
        return {
            "kind": kind,
            "n_A": int(section["n_A"]),
            "n_B": int(section["n_B"]),
            "epsilon_AA": float(section["epsilon_AA"]),
            "epsilon_AB": float(section["epsilon_AB"]),
            "epsilon_BB": float(section["epsilon_BB"]),
            "sigma_AA": float(section["sigma_AA"]),
            "sigma_BB": float(section["sigma_BB"]),
            "sigma_AB": "arithmetic mean",
            "cutoff": None,
        }
    raise ValueError(f"{problem['id']} potential {kind} has no GMIN harness kernel")


def _binary(coords: np.ndarray, *, spec: dict, species: np.ndarray) -> tuple[float, np.ndarray]:
    return binary_lennard_jones(
        coords,
        species,
        sigma_aa=spec["sigma_AA"],
        sigma_bb=spec["sigma_BB"],
        epsilon_aa=spec["epsilon_AA"],
        epsilon_bb=spec["epsilon_BB"],
        epsilon_ab=spec["epsilon_AB"],
    )


def evaluator(spec: dict):
    """``coords -> (energy, gradient)`` for one potential spec."""
    if spec["kind"] == "lennard-jones":
        return lennard_jones_compiled
    species = np.asarray([0] * int(spec["n_A"]) + [1] * int(spec["n_B"]), dtype=np.int64)
    return functools.partial(_binary, spec=spec, species=species)


def reference_evaluation(root: Path, natoms: int, spec: dict | None = None) -> tuple[float, float]:
    """Harness energy and RMS of the Cambridge points for this problem."""
    spec = spec or {"kind": "lennard-jones"}
    if spec["kind"] == "binary-lennard-jones":
        from optbench.ccd import binary_minimum

        record = binary_minimum()
        pos = np.asarray(record["positions"], dtype=np.float64)
        expected = np.asarray([0] * int(spec["n_A"]) + [1] * int(spec["n_B"]), dtype=np.int64)
        if not np.array_equal(np.asarray(record["species"]), expected):
            raise ValueError("Cambridge binary points do not list the A atoms first")
        energy, grad = evaluator(spec)(pos)
    else:
        path = root / "codes" / "gmin" / "reference" / f"lj{natoms}.points"
        if not path.is_file():
            raise FileNotFoundError(f"no reference points for {natoms} atoms")
        pos = np.loadtxt(path)
        if pos.shape != (natoms, 3):
            raise ValueError(f"{path} has shape {pos.shape}")
        energy, grad = lennard_jones(pos)
    grad = np.asarray(grad, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(float(np.dot(grad, grad)) / grad.size))
    return float(energy), rms


def _center(pos: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) - np.mean(pos, axis=0)


def _read_geometry(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    # The lj100-binary starts are .con files named coords.N.
    if path.suffix == ".con" or "Coordinates of Component" in text:
        pos, _box = read_con(text)
        return pos
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"{path} has no coordinates")
    return np.asarray(rows, dtype=np.float64)


def _start_paths(dest: Path, problem_id: str) -> list[Path]:
    if problem_id == "globalopt/lj38":
        files = list(dest.rglob("*.con"))
        return sorted(files, key=lambda path: int(path.stem))
    numbered = []
    for path in dest.rglob("coords.*"):
        suffix = path.name.split(".", 1)[1]
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    numbered.sort()
    return [path for _index, path in numbered]


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    recorded = archive.name
    if marker.is_file():
        previous = marker.read_text(encoding="utf-8").strip()
        if previous != recorded:
            raise RuntimeError(f"{dest} was extracted from {previous}, not {recorded}")
        return
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    marker.write_text(recorded, encoding="utf-8")


# Numbered coords.N from the 2014 website tarballs: 200 for LJ75 and 100
# for LJ98 and LJ150. Used when a problem file leaves archive_path unset.
_ARCHIVE_FALLBACK = {
    "globalopt/lj75": "website/globalopt/lj75/lj75_clusters.tgz",
    "globalopt/lj98": "website/globalopt/lj98/lj98_clusters.tgz",
    "globalopt/lj150": "website/globalopt/lj150/lj150_clusters.tgz",
}


def _problem_for_archive(problem: dict) -> dict:
    if (problem.get("input") or {}).get("archive_path"):
        return problem
    fallback = _ARCHIVE_FALLBACK.get(str(problem.get("id") or ""))
    if not fallback:
        return problem
    copied = dict(problem)
    copied["input"] = {"archive_path": fallback}
    return copied


def load_geometries(root: Path, problem: dict, work: Path, *, center: bool = True) -> list[tuple[str, np.ndarray]]:
    from optbench.campaign import geometry_archive

    archive = geometry_archive(root, _problem_for_archive(problem))
    dest = work / "geometries" / problem["id"].replace("/", "-")
    _extract(archive, dest)
    paths = _start_paths(dest, problem["id"])
    if not paths:
        raise FileNotFoundError(f"{archive} has no starts")
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(paths) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(paths)}")
    loaded = []
    for path in paths:
        pos = _read_geometry(path)
        if center:
            pos = _center(pos)
        # coords.1 has stem "coords". Keep the numeric name. A .con stem is the start index.
        label = path.stem if path.suffix == ".con" else path.name
        loaded.append((label, pos))
    return loaded


def _section(template: dict) -> dict:
    section = template.get("gmin") or {}
    method = str(section.get("method") or "")
    if method not in {"bh", "bhpt"}:
        raise ValueError(f"{template['id']} gmin.method must be bh or bhpt")
    replicas = int(section.get("replicas") or (4 if method == "bhpt" else 1))
    if method == "bh" and replicas != 1:
        raise ValueError(f"{template['id']} serial basin-hopping uses one replica")
    if method == "bhpt" and replicas < 2:
        raise ValueError(f"{template['id']} BHPT needs at least two replicas")
    return section


def _archive_lines(job: dict) -> list[str]:
    """Archive move keywords. Absent keys leave the generic data file unchanged."""
    lines = []
    if job.get("fixboth"):
        lines.append("FIXBOTH")
    if job.get("permdist") not in (None, ""):
        lines.append(f"PERMDIST {float(job['permdist']):.8f}")
    if job.get("newrestart") not in (None, ""):
        lines.append(f"NEWRESTART {int(job['newrestart'])}")
    if job.get("sort"):
        lines.append("SORT")
    symmetrise = str(job.get("symmetrise") or "").strip()
    if symmetrise:
        lines.append(f"SYMMETRISE {symmetrise}")
    avoid = str(job.get("avoid") or "").strip()
    if avoid:
        lines.append(f"AVOID {avoid}")
    symmetrisecsm = str(job.get("symmetrisecsm") or "").strip()
    if symmetrisecsm:
        lines.append(f"SYMMETRISECSM {symmetrisecsm}")
    return lines


def _data_text(job: dict) -> str:
    # This binary is built with GMIN_MPI. mc.F90 sets the replica index only
    # inside IF (MPIT), and the serial DO loop is compiled out. MPI is required
    # for one replica as well as for BHPT. It resets STEP and TEMP, so it is first.
    lines = ["MPI"]
    if int(job["replicas"]) > 1:
        lines.append(
            f"BHPT {job['pttmin']:.8f} {job['pttmax']:.8f} {job['exchprob']:.8f}"
        )
    lines.extend(
        [
            f"SLOPPYCONV {job['sloppyconv']:.8f}",
            f"TIGHTCONV {job['tightconv']:.8f}",
            f"EDIFF {job['ediff']:.8f}",
            f"MAXIT {int(job['maxit'])} {int(job['maxit2'])}",
            f"SAVE {int(job.get('save') or 2)}",
            "CENTRE",
            *( [f"RADIUS {float(job['radius']):.8f}"] if float(job.get("radius") or 0) > 0 else [] ),
            f"TEMPERATURE {job['temperature']:.8f}",
            f"STEP {job['step']:.8f} {job['astep']:.8f}",
            *_archive_lines(job),
            (
                f"STEPS {int(job['steps'])} {float(job['tfac']):.1f}"
                if job.get("tfac") not in (None, "")
                else f"STEPS {int(job['steps'])}"
            ),
            *([f"RANSEED {int(job['seed'])}"] if job.get("emit_seed", True) else []),
            f"TARGET {job['reference']:.16f}",
            f"MAX_NPCALL {int(job['cap'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def _coords_text(coords: list[list[float]], replicas: int) -> str:
    block = "\n".join(
        f"{row[0]:20.10f} {row[1]:20.10f} {row[2]:20.10f}" for row in coords
    )
    return ("\n".join([block] * replicas)) + "\n"


def _command(binary: str, replicas: int) -> list[str]:
    # mpirun -np 1 as well. A direct launch of the MPI binary leaves the
    # replica index unset unless OpenMPI singleton initialises the same way.
    # Yield while a replica waits on the harness socket. Otherwise the ranks
    # spin and the potential server loses the core.
    mpirun = shutil.which("mpirun")
    if not mpirun:
        raise RuntimeError("GMIN MPI binary needs mpirun")
    return [
        mpirun,
        "--oversubscribe",
        "--mca",
        "mpi_yield_when_idle",
        "1",
        "-np",
        str(replicas),
        "--bind-to",
        "none",
        "-x",
        "OPTBENCH_POTPORT",
        "-x",
        "OPTBENCH_POTHOST",
        "-x",
        "OMP_NUM_THREADS",
        binary,
    ]


def _classify(job: dict, snapshot: dict, returncode: int | None, error: str) -> dict:
    """Judge one start with ``judge_globalopt``.

    A timeout, an exception, a signal, or a nonzero exit before the cap is
    a crash, also after a hit. The hook halts GMIN with exit 2 at the cap.
    """
    calls = int(snapshot.get("calls") or 0)
    cap = int(job["cap"])
    cap_hit = bool(snapshot.get("cap_hit")) or calls >= cap
    hit_calls = snapshot.get("hit_calls")
    if hit_calls is not None:
        energy = snapshot.get("hit_energy")
        counted = int(hit_calls)
    else:
        energy = snapshot.get("lowest_energy")
        counted = calls
    judged = judge_globalopt(
        energy,
        counted,
        reference=float(job["reference"]),
        tolerance=float(job["tolerance"]),
        cap=cap,
    )
    failed_run = bool(error) or (returncode is not None and returncode != 0)
    reasons = list(judged["reasons"])
    failure = judged["failure"]
    if failed_run and not cap_hit:
        reasons = ["crash"]
        failure = classify_failure(reasons)
    success = failure is None
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": counted if success else calls,
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": judged["energy"],
        "lowest_energy": snapshot.get("lowest_energy"),
        "hit_calls": hit_calls,
        "total_calls": calls,
        "quench_energy": snapshot.get("quench_energy"),
        "quench_rms": snapshot.get("quench_rms"),
        "quench_calls": snapshot.get("quench_calls"),
        "reference_energy": float(job["reference"]),
        "tolerance": float(job["tolerance"]),
        "cap_hit": cap_hit,
        "returncode": returncode,
        "replicas": int(job["replicas"]),
        "server_errors": int(snapshot.get("errors") or 0),
    }
    if error:
        record["error"] = error
    return record


def run_one(job: dict) -> dict:
    """One GMIN start. The harness server lives in this process."""
    from optbench.gmin_pot import GminServer

    work = Path(job["work"])
    stored = work / "record.json"
    if job.get("resume") and stored.is_file():
        return json.loads(stored.read_text(encoding="utf-8"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").write_text(_data_text(job), encoding="utf-8")
    coords = job["coords"]
    (work / "coords").write_text(_coords_text(coords, int(job["replicas"])), encoding="utf-8")
    counter = HarnessCounter(cache=False, cap=int(job["cap"]))
    server = GminServer(
        evaluator(job["potential"]),
        counter,
        sloppy_rms=float(job["sloppyconv"]),
        target=float(job["reference"]) + float(job["tolerance"]),
    )
    server.start()
    env = os.environ.copy()
    env["OPTBENCH_POTHOST"] = "127.0.0.1"
    env["OPTBENCH_POTPORT"] = str(server.port)
    env["OMP_NUM_THREADS"] = "1"
    error = ""
    returncode: int | None = None
    try:
        with (work / "stdout.txt").open("w", encoding="utf-8") as stdout, (
            work / "stderr.txt"
        ).open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                _command(job["binary"], int(job["replicas"])),
                cwd=work,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                returncode = int(proc.wait(timeout=float(job["timeout"])))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
                returncode = -9
                error = "timeout"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        returncode = returncode if returncode is not None else -1
    snapshot = server.snapshot()
    server.stop()
    record = _classify(job, snapshot, returncode, error)
    stored.write_text(json.dumps(record), encoding="utf-8")
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(summary: dict, template: dict, metrics: dict, partial: bool) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": metrics.get("nfailed"),
        "median": summary.get("median"),
        "code": CODE_NAME,
        "code_version": CODE_VERSION,
        "commit": TARBALL_SHA256,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "problem": template.get("problem"),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "rate_low": metrics.get("rate_low"),
        "rate_high": metrics.get("rate_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "cap": summary.get("cap"),
    }
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one GMIN potential request to lennard_jones.",
        "# The binary was built from the Wales tarball named by commit. The pin is that sha256.",
        "# energy is the lowest quench at or below SLOPPYCONV. lowest_energy on each start is the lowest finite evaluation.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _write_result(output: Path, problem: dict, template: dict, records: list[dict], partial: bool, passed: dict) -> dict:
    cap = int(passed["cap"])
    ordered = sorted(records, key=lambda row: int(row["index"]))
    for record in ordered:
        record.pop("index", None)
    capped_counts = [cap if not record["success"] else int(record["force_calls"]) for record in ordered]
    summary = summarize_capped(capped_counts, cap)
    successes = [int(record["force_calls"]) for record in ordered if record["success"]]
    import statistics

    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in ordered if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    summary["cap"] = cap
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in ordered
        ],
        cap,
    )
    provenance = collect_provenance(
        code=CODE_NAME,
        code_version=CODE_VERSION,
        commit=TARBALL_SHA256,
        parameters=passed,
    )
    rule = (
        "A start succeeds at the first harness evaluation at or below the problem reference energy "
        "plus energy_above_minimum, under the problem max_force_calls."
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "measured",
        "problem": problem["id"],
        "scoring": "par2-rate-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": rule,
        "runs": [
            {
                "entry": output.stem,
                "code": CODE_NAME,
                "algorithm": template.get("algorithm", ""),
                "code_version": CODE_VERSION,
                "commit": TARBALL_SHA256,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": passed,
                "provenance": provenance,
                "metrics": metrics,
                "starts": ordered,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, partial), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), ordered)
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["nfailed"] = metrics["nfailed"]
    return summary


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    if str(template.get("code_version") or "") != CODE_VERSION:
        raise RuntimeError(f"{template['id']} code_version is not {CODE_VERSION}")
    if str(template.get("commit") or "") != TARBALL_SHA256:
        raise RuntimeError(f"{template['id']} commit is not the tarball sha256")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    section = _section(template)
    reference = reference_energy(problem)
    tolerance = energy_tolerance(problem)
    cap = force_cap(problem)
    ediff = float(section["ediff"])
    if abs(tolerance - ediff) > 1e-12:
        raise ValueError("template ediff is not the problem energy_above_minimum")
    if template.get("failure_cap") is None or int(template["failure_cap"]) != cap:
        raise ValueError("template failure_cap is not the problem max_force_calls")
    spec = potential_spec(problem)
    binary = gmin_binary()
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    loaded = load_geometries(root, problem, scratch, center=bool(section.get("center", True)))
    if limit > 0:
        loaded = loaded[: int(limit)]
    natoms = int(loaded[0][1].shape[0])
    if spec["kind"] == "binary-lennard-jones" and natoms != spec["n_A"] + spec["n_B"]:
        raise ValueError(f"{problem['id']} starts have {natoms} atoms, not n_A + n_B")
    points_energy, reference_rms = reference_evaluation(root, natoms, spec)
    sloppy = float(section["sloppyconv"])
    if reference_rms > sloppy:
        raise RuntimeError(f"reference {problem['id']} RMS {reference_rms} is above sloppyconv {sloppy}")
    if points_energy > reference + tolerance:
        raise RuntimeError(f"reference {problem['id']} harness energy {points_energy} is outside the window")
    replicas = int(section["replicas"])
    timeout = float(section.get("timeout") or (cap * 0.005 + 900))
    passed = {
        "method": str(section["method"]),
        "replicas": replicas,
        "temperature": float(section["temperature"]),
        "pttmin": float(section["pttmin"]),
        "pttmax": float(section["pttmax"]),
        "exchprob": float(section["exchprob"]),
        "step": float(section["step"]),
        "astep": float(section["astep"]),
        "sloppyconv": sloppy,
        "tightconv": float(section["tightconv"]),
        "ediff": ediff,
        "maxit": int(section["maxit"]),
        "maxit2": int(section["maxit2"]),
        "radius": float(section["radius"]),
        "steps": int(section["steps"]),
        "seed_base": int(section["seed"]),
        "cap": cap,
        "reference_energy": reference,
        "points_energy": points_energy,
        "reference_rms": reference_rms,
        "tolerance": tolerance,
        "potential": spec,
        "fixboth": bool(section.get("fixboth", False)),
        "permdist": section.get("permdist"),
        "newrestart": section.get("newrestart"),
        "sort": bool(section.get("sort", False)),
        "symmetrise": str(section.get("symmetrise") or ""),
        "avoid": str(section.get("avoid") or ""),
        "symmetrisecsm": str(section.get("symmetrisecsm") or ""),
        "save": int(section.get("save") or 2),
        "tfac": section.get("tfac"),
        "emit_seed": bool(section.get("emit_seed", True)),
    }
    jobs = []
    lennard_jones_compiled(np.zeros((1, 3)))
    for index, (name, pos) in enumerate(loaded):
        if pos.shape != (natoms, 3):
            raise ValueError(f"{name} has {pos.shape[0]} atoms, not {natoms}")
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": pos.tolist(),
                "work": str(scratch / "starts" / f"{index:04d}"),
                "binary": str(binary),
                "replicas": replicas,
                "cap": cap,
                "sloppyconv": sloppy,
                "reference": reference,
                "tolerance": tolerance,
                "potential": spec,
                "resume": bool(resume),
                "timeout": timeout,
                "pttmin": passed["pttmin"],
                "pttmax": passed["pttmax"],
                "exchprob": passed["exchprob"],
                "temperature": passed["temperature"],
                "step": passed["step"],
                "astep": passed["astep"],
                "tightconv": passed["tightconv"],
                "ediff": ediff,
                "maxit": passed["maxit"],
                "maxit2": passed["maxit2"],
                "radius": passed["radius"],
                "steps": passed["steps"],
                "seed": int(section["seed"]) if not passed["emit_seed"] else int(section["seed"]) + index,
                "save": passed["save"],
                "tfac": passed["tfac"],
                "emit_seed": passed["emit_seed"],
                "fixboth": passed["fixboth"],
                "permdist": passed["permdist"],
                "newrestart": passed["newrestart"],
                "sort": passed["sort"],
                "symmetrise": passed["symmetrise"],
                "avoid": passed["avoid"],
                "symmetrisecsm": passed["symmetrisecsm"],
            }
        )
    if workers <= 1:
        records = []
        for done, job in enumerate(jobs, 1):
            records.append(run_one(job))
            print(f"{template['id']} {done}/{len(jobs)}", flush=True)
    else:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            records = []
            total = len(futures)
            for done, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                print(f"{template['id']} {done}/{total}", flush=True)
    stated_n = (problem.get("sample") or {}).get("starts")
    complete = (
        stated_n is not None
        and len(records) == int(stated_n)
        and int(limit) <= 0
    )
    return _write_result(output, problem, template, records, partial=not complete, passed=passed)
