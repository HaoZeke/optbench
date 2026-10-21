"""Minima hopping for Lennard-Jones clusters with FLAME minhocao.

FLAME is the msock server. This runner connects as the i-PI force side
and evaluates ``optbench.potentials.lennard_jones``. The counter is a
``HarnessCounter`` with caching off: one ``POSDATA`` is one force call.
A call at the cap raises ``ForceCallCap`` and is not counted.

minhocao does not stop on a target energy. After each relaxed minimum is
written to ``global.mon``, the next request checks that file before it
charges a call. When the enthalpy, converted with FLAME's Hartree
constant, is within the problem tolerance of the reference energy, the
socket closes and the recorded count is the calls already charged.
``CPUlimit`` is written as well so a later hop would exit at its boundary.

The reference for ``globalopt/lj38`` is the Cambridge cluster minimum
-173.928427. The problem tolerance is 0.001, so the ceiling is
-173.927427, the same stop the 2014 eOn archive used. Problems with no
energy rule record the lowest relaxed energy and the calls that first
reached it, and the row stays partial.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.flame_ipi import BOHR_ANG, HA_EV, SearchStopped, lowest_relaxed_lj, serve
from optbench.potentials import lennard_jones, read_con
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

# Side length of the orthorhombic cell. The cluster is translated to the
# centre. Lennard-Jones has no periodic images. minhocao still wraps atoms
# into the cell on every evaluation, so the side has to be large enough
# that a hot hop cannot walk an atom onto the boundary. A wrapped atom
# reappears on the far side and the free cluster is split.
CELL_ANG = 10_000_000.0
# Documented minhocao starting line: ediff (Ha), temperature (K), maximum temperature (K).
IOPUT = "1.d-2 5.d2 1.d5\n"
# Documented earr.dat tolerances: delta enthalpy (Ha), delta fingerprint.
EARR_TOLERANCE = "  2.000000E-03   1.000000E-01\n"
DEFAULT_NLMINX = 100000
# 2014 eOn basin-hopping step cap. Successes that finish under it are not cap failures.
ARCHIVE_FORCE_CAP = 10_000_000
LJ38_REFERENCE = -173.928427
# Cambridge minima (Doye and Wales). The same 0.001 window as globalopt/lj38.
# Problems that do not state energy_above_minimum use it only to end the
# search. Those rows stay partial and do not count as ranked successes.
REFERENCE_WINDOW = 0.001
CAMBRIDGE = {
    "globalopt/lj38": LJ38_REFERENCE,
    "globalopt/lj75": -397.492331,
    "globalopt/lj98": -543.665361,
    "globalopt/lj150": -893.310258,
}
ARCHIVES = {
    "globalopt/lj38": "website/globalopt/lj38/100_lj38.tar.gz",
    "globalopt/lj75": "website/globalopt/lj75/lj75_clusters.tgz",
    "globalopt/lj98": "website/globalopt/lj98/lj98_clusters.tgz",
    "globalopt/lj150": "website/globalopt/lj150/lj150_clusters.tgz",
}


def flame_binary() -> Path:
    env = os.environ.get("FLAME")
    if env:
        return Path(env)
    return Path("/tmp/flame-build/src/flame")


def port_base(problem_id: str) -> int:
    """Preferred TCP port for the first start of a problem.

    Each start scans 64 ports upward. The blocks are 300 apart so lj75
    (200 starts), lj98, and lj150 can run at the same time.
    """
    block = {
        "globalopt/lj38": 0,
        "globalopt/lj75": 0,
        "globalopt/lj98": 1,
        "globalopt/lj150": 2,
    }.get(str(problem_id), 3)
    return 25000 + 300 * block


def cap_of(problem: dict, template: dict) -> int | None:
    """The one force-call cap for this problem.

    A problem ``max_force_calls`` is that cap. The template ``failure_cap``
    has to repeat it. When the problem states no cap, the template value is
    the cap. The harness stop, the PAR-2 penalty, and the dat ``cap`` field
    all use this number.
    """
    convergence = problem.get("convergence") or {}
    stated = convergence.get("max_force_calls")
    failure = template.get("failure_cap")
    if stated not in (None, "") and failure not in (None, ""):
        if int(stated) != int(failure):
            raise ValueError(
                f"{template.get('id', 'template')} failure_cap {failure} "
                f"disagrees with {problem.get('id', 'problem')} max_force_calls {stated}"
            )
    if stated not in (None, ""):
        return int(stated)
    if failure not in (None, ""):
        return int(failure)
    stop = template.get("stop") or {}
    for key in ("max_force_calls", "max_steps"):
        if stop.get(key) not in (None, ""):
            return int(stop[key])
    return None


def reference_energy(problem: dict, template: dict) -> float | None:
    """Known minimum in Lennard-Jones units, if this run has one."""
    blocks = [problem]
    for name in ("reference", "known", "convergence"):
        block = problem.get(name)
        if isinstance(block, dict):
            blocks.append(block)
    flame = template.get("flame") or {}
    blocks.append(flame)
    keys = ("energy", "minimum", "global_minimum", "minimum_energy", "known_minimum", "reference_energy")
    for block in blocks:
        for key in keys:
            if key in block and block[key] not in (None, ""):
                return float(block[key])
    if problem.get("id") in CAMBRIDGE:
        return CAMBRIDGE[str(problem["id"])]
    return None


def search_target(problem: dict, template: dict) -> float | None:
    """Energy that ends the search.

    A stated ``energy_above_minimum`` is a ranked success. A Cambridge
    reference with no stated rule still ends the search, and the start
    stays a partial observation.
    """
    ceiling = success_ceiling(problem, template)
    if ceiling is not None:
        return ceiling
    reference = reference_energy(problem, template)
    if reference is None:
        return None
    return float(reference) + REFERENCE_WINDOW


def success_ceiling(problem: dict, template: dict) -> float | None:
    """Highest energy that counts as the known minimum, or None."""
    convergence = problem.get("convergence") or {}
    if "energy_above_minimum" not in convergence:
        return None
    reference = reference_energy(problem, template)
    if reference is None:
        raise ValueError(f"{problem['id']} states energy_above_minimum and has no reference energy")
    return float(reference) + float(convergence["energy_above_minimum"])


def _archive_path(problem: dict) -> str:
    stated = (problem.get("input") or {}).get("archive_path")
    if stated:
        return str(stated)
    found = ARCHIVES.get(str(problem["id"]))
    if not found:
        raise FileNotFoundError(f"{problem['id']} has no geometry archive")
    return found


def _geometry_archive(root: Path, problem: dict) -> Path:
    from optbench.campaign import geometry_archive

    view = dict(problem)
    view["input"] = {"archive_path": _archive_path(problem)}
    return geometry_archive(root, view)


def _sort_key(name: str) -> tuple:
    digits = []
    chunk = ""
    for char in Path(name).name:
        if char.isdigit():
            chunk += char
        elif chunk:
            digits.append(int(chunk))
            chunk = ""
    if chunk:
        digits.append(int(chunk))
    return (digits or [10**12], Path(name).name)


def _read_coords(text: str):
    import numpy as np

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    if not rows:
        raise ValueError("coordinate file has no positions")
    return np.asarray(rows, dtype=np.float64)


def _place(coords):
    import numpy as np

    pos = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    centre = pos.mean(axis=0)
    placed = pos - centre + (CELL_ANG / 2.0)
    box = np.full(3, CELL_ANG, dtype=np.float64)
    return placed, box


def load_geometries(root: Path, problem: dict, geometries: Path | None, work: Path | None, output: Path):
    """Starts as ``(name, positions (n, 3), box)``, sorted by the numeric stem."""
    import tarfile

    if geometries is None:
        archive = _geometry_archive(root, problem)
        dest = (work or output.parent / "geometries") / problem["id"].replace("/", "-")
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".extracted"
        recorded = archive.name
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == recorded:
            pass
        else:
            if dest.exists():
                for child in dest.iterdir():
                    if child.name == ".extracted":
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            with tarfile.open(archive) as packed:
                packed.extractall(dest, filter="data")
            marker.write_text(recorded, encoding="utf-8")
        paths = [
            path
            for path in dest.rglob("*")
            if path.is_file() and (path.suffix == ".con" or path.name.startswith("coords."))
        ]
    else:
        root_path = Path(geometries)
        if root_path.is_file():
            paths = [root_path]
        else:
            paths = [
                path
                for path in root_path.rglob("*")
                if path.is_file() and (path.suffix == ".con" or path.name.startswith("coords."))
            ]
    if not paths:
        raise FileNotFoundError(f"{problem['id']} has no .con or coords.* starts")
    paths.sort(key=lambda path: _sort_key(path.name))
    expected = (problem.get("sample") or {}).get("starts")
    if geometries is None and expected is not None and len(paths) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(paths)}")
    loaded = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".con":
            pos, _box = read_con(text)
        else:
            pos = _read_coords(text)
        placed, box = _place(pos)
        loaded.append((path.name, placed, box))
    return loaded


def fire_dt_of(template: dict) -> dict:
    """Optional FIRE timestep lines. Empty means the params_new defaults."""
    flame = template.get("flame") or {}
    found = {}
    for source, dest in (
        ("geofiredtinit", "init"),
        ("geofiredtmin", "min"),
        ("geofiredtmax", "max"),
    ):
        value = flame.get(source)
        if value not in (None, ""):
            found[dest] = str(value).strip()
    return found


def md_lines_of(template: dict) -> list[str]:
    """Optional params_new lines. Empty keeps the minhocao defaults."""
    flame = template.get("flame") or {}
    lines = []
    for source, label in (
        ("mddtinit", "MDDTINIT"),
        ("softnit", "SOFTNIT"),
        ("mdmininit", "MDMININIT"),
        ("geotolmxf", "GEOTOLMXF"),
    ):
        value = flame.get(source)
        if value not in (None, ""):
            lines.append(f"{label} {str(value).strip()}")
    return lines


def ioput_text(template: dict) -> str:
    """The three-number ioput line. The default is the documented 16-atom start."""
    raw = (template.get("flame") or {}).get("ioput")
    if raw in (None, ""):
        return IOPUT
    text = str(raw).strip()
    return text if text.endswith("\n") else text + "\n"


def earr_text(template: dict) -> str:
    """The earr tolerance line. The default is the documented fingerprint pair."""
    raw = (template.get("flame") or {}).get("earr_tolerance")
    if raw in (None, ""):
        return EARR_TOLERANCE
    text = str(raw).strip("\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def _write_case(
    work: Path,
    positions,
    port: int,
    seed: int,
    nlminx: int,
    fire_dt: dict | None = None,
    md_lines: list[str] | None = None,
    ioput: str | None = None,
    earr: str | None = None,
) -> None:
    nat = int(positions.shape[0])
    work.mkdir(parents=True, exist_ok=True)
    (work / "flame_in.yaml").write_text(
        "main:\n"
        "  task: minhocao\n"
        "  types: C\n"
        f"  nat: {nat}\n"
        f"  typat: {nat}*1\n"
        "  params_new: True\n"
        f"  seed: {int(seed)}\n",
        encoding="utf-8",
    )
    (work / "params_new.in").write_text(
        "CODE msock\n"
        "BOUNDARY 2\n"
        # Verbosity 0 skips per-step geometry dumps. global.mon is still
        # written on every hop, and the hopping parameters are unchanged.
        "VERBOSE 0\n"
        f"NAT {nat}\n"
        "NTYPE 1\n"
        "ZNUCL 6\n"
        f"TYPAT {nat}*1\n"
        "SOCKINET 1\n"
        f"SOCKPORT {int(port)}\n"
        "SOCKHOST 127.0.0.1\n"
        "FINDSYM F\n",
        encoding="utf-8",
    )
    labels = {"init": "GEOFIREDTINIT", "min": "GEOFIREDTMIN", "max": "GEOFIREDTMAX"}
    lines = []
    for key, label in labels.items():
        value = None if not fire_dt else fire_dt.get(key)
        if value not in (None, ""):
            lines.append(f"{label} {value}")
    lines.extend(line.strip() for line in (md_lines or []) if str(line).strip())
    if lines:
        with (work / "params_new.in").open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    (work / "ioput").write_text(ioput if ioput not in (None, "") else IOPUT, encoding="utf-8")
    (work / "earr.dat").write_text(
        f"{0:10d}{int(nlminx):10d}\n" + (earr if earr not in (None, "") else EARR_TOLERANCE),
        encoding="utf-8",
    )
    # poscur.ascii stores the projective cell (dxx, dyx, dyy) and
    # (dzx, dzy, dzz), not lengths and angles. A cube of side L is
    # ``L 0 L`` / ``0 0 L``. Lengths and 90 degree angles are read as
    # off-diagonal projections and the atoms leave the input geometry.
    side = f"{CELL_ANG:.10f}"
    lines = [
        str(nat),
        f"{side} 0.0 {side}",
        f"0.0 0.0 {side}",
        "#keywords: fixlat T T T T T T F",
    ]
    for row in positions:
        lines.append(f"{row[0]:.16f} {row[1]:.16f} {row[2]:.16f} C")
    (work / "poscur.ascii").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _free_port(preferred: int) -> int:
    for port in range(int(preferred), int(preferred) + 64):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            continue
        else:
            probe.close()
            return port
        finally:
            probe.close()
    raise RuntimeError(f"no free TCP port at or above {preferred}")


def _connect(proc: subprocess.Popen, port: int, log_path: Path, timeout: float = 60.0) -> socket.socket:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _log_tail(log_path, 2000)
            raise RuntimeError(f"flame exited {proc.returncode} before accept\n{tail}")
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=2.0)
        except OSError as exc:
            last = str(exc)
            time.sleep(0.05)
    tail = _log_tail(log_path, 2000)
    raise RuntimeError(f"flame did not accept on {port}: {last}\n{tail}")


def _log_tail(path: Path, limit: int = 1500) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - limit * 4))
        return handle.read().decode("utf-8", "replace")[-limit:]


def _run_start(job: dict) -> dict:
    """One minhocao search. The job dict is plain data so a process pool can ship it."""
    import numpy as np

    work = Path(job["work"])
    if work.exists():
        shutil.rmtree(work)
    positions = np.asarray(job["coords"], dtype=np.float64).reshape(-1, 3)
    port = _free_port(int(job["port"]))
    _write_case(
        work,
        positions,
        port,
        int(job["seed"]),
        int(job["nlminx"]),
        job.get("fire_dt") or None,
        job.get("md_lines") or None,
        job.get("ioput"),
        job.get("earr"),
    )
    cap = None if job["cap"] is None else int(job["cap"])
    target = None if job["target"] is None else float(job["target"])
    counter = HarnessCounter(cache=False, cap=cap)
    state = {
        "best": None,
        "best_calls": None,
        "stop": "",
        "error": "",
        "first_energy": None,
        "mon_size": -1,
    }
    mon = work / "global.mon"
    log_path = work / "flame_log.yaml"

    def note():
        if not mon.is_file():
            return state["best"]
        size = mon.stat().st_size
        if size == state["mon_size"]:
            return state["best"]
        state["mon_size"] = size
        energy = lowest_relaxed_lj(mon)
        if energy is None:
            return state["best"]
        if state["best"] is None or energy < state["best"]:
            state["best"] = energy
            state["best_calls"] = counter.calls
        return energy

    def before():
        energy = note()
        if target is None or energy is None or energy > target:
            return
        if cap is not None and counter.calls >= cap:
            state["stop"] = "cap"
            return
        (work / "CPUlimit").write_text("0\n", encoding="utf-8")
        state["stop"] = "success"
        raise SearchStopped("success")

    def evaluate(flat):
        pos = np.asarray(flat, dtype=np.float64).reshape(-1)
        margin = CELL_ANG * 0.05
        if np.any(pos < margin) or np.any(pos > CELL_ANG - margin):
            state["stop"] = "divergence"
            raise SearchStopped("divergence")
        key = tuple(float(value) for value in pos)
        _fresh, _cached = counter.charge(key)
        if counter.calls % 1000 == 0:
            try:
                (work / "calls.progress").write_text(
                    f"{counter.calls} {state['best']}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        energy, gradient = lennard_jones(pos)
        if state["first_energy"] is None:
            state["first_energy"] = float(energy)
        gradient = np.asarray(gradient, dtype=np.float64)
        counter.remember(key, (float(energy), tuple(float(value) for value in gradient)))
        return float(energy), gradient

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    # Socket traces are one line per force call. The yaml log is the diagnostic.
    log = open(os.devnull, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [job["binary"]],
        cwd=work,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    try:
        conn = _connect(proc, port, log_path)
        conn.settimeout(float(job.get("idle_timeout", 300)))
        try:
            serve(conn, evaluate, before_eval=before)
        except SearchStopped:
            pass
        except ForceCallCap:
            state["stop"] = "cap"
        except (OSError, TimeoutError) as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            if not state["stop"]:
                state["stop"] = "crash"
        finally:
            try:
                conn.close()
            except OSError:
                pass
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["stop"] = "crash"
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, 15)
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, 9)
                except OSError:
                    proc.kill()
                proc.wait(timeout=15)
        log.close()
    note()
    total = int(counter.calls)
    best = state["best"]
    best_calls = state["best_calls"]
    success = False
    failure = None
    calls = total
    claim = bool(job.get("claim"))
    reached = (
        target is not None
        and best is not None
        and best <= target
        and (cap is None or (best_calls if best_calls is not None else total) < cap)
    )
    reference_bound = False
    if claim and (state["stop"] == "success" or reached):
        success = True
        calls = int(best_calls if best_calls is not None else total)
        if cap is not None and calls >= cap:
            success = False
            failure = "cap"
            calls = total
    elif (not claim) and reached:
        reference_bound = True
        calls = int(best_calls if best_calls is not None else total)
    elif state["stop"] == "divergence":
        failure = "divergence"
        calls = total
    elif state["stop"] == "cap" or (cap is not None and total >= cap):
        failure = "cap"
        calls = total
    elif state["stop"] == "crash" or state["error"]:
        failure = "crash"
    elif target is None:
        failure = "cap" if cap is not None else "divergence"
    else:
        failure = "wrong_stationary"
    if not success and failure is None and not reference_bound:
        failure = "divergence"
    if target is None and best_calls is not None and failure != "crash":
        # No success rule: the reported count is the first sighting of the lowest relaxed energy.
        calls = int(best_calls)
    finite = best is not None and math.isfinite(best)
    record = {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": int(job["index"]),
        "force_calls": int(calls),
        "calls_total": total,
        "calls_to_best": None if best_calls is None else int(best_calls),
        "counter": "harness",
        "comparable": True,
        "success": success,
        "failure": None if success else failure,
        "energy": float(best) if finite else None,
        "first_energy": state["first_energy"],
        "converged": success,
        "seed": int(job["seed"]),
    }
    if reference_bound:
        record["reference_bound"] = True
    if state["error"] and not success:
        record["error"] = state["error"]
        record["log_tail"] = _log_tail(log_path)
    elif failure == "crash":
        record["log_tail"] = _log_tail(log_path)
    shutil.rmtree(work, ignore_errors=True)
    return record


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _dat(
    summary: dict,
    template: dict,
    metrics: dict,
    commit: str,
    version: str,
    *,
    partial: bool,
    ceiling,
    cap,
) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "calls_to_best_mean": summary.get("calls_to_best_mean"),
        "cap": cap,
        "n_cap": metrics.get("n_cap"),
        "n_divergence": metrics.get("n_divergence"),
        "n_crash": metrics.get("n_crash"),
        "n_wrong_stationary": metrics.get("n_wrong_stationary"),
    }
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one minhocao POSDATA evaluated by lennard_jones.",
        "# Positions cross the socket in Bohr and return as Hartree and Ha/Bohr with FLAME Bohr_Ang and Ha_eV.",
        "# The cluster is translated to the centre of a 1e7 Angstrom cell. The potential has no periodic images.",
        "# ioput is " + ioput_text(template).strip() + ". earr tolerances are " + earr_text(template).strip() + ".",
        "# Local optimisation is the code default FIRE tolerance 2.d-4 Ha/Bohr unless GEOTOLMXF is set.",
    ]
    if cap is not None:
        lines.append(
            f"# cap is {int(cap)}. The harness stops at this value and PAR-2 charges 2*cap for every failure class. "
            "nfailed counts only starts whose call count reached cap. v2_nfailed counts every failure class."
        )
    fire = fire_dt_of(template)
    if fire:
        lines.append(
            "# GEOFIREDTINIT "
            + fire.get("init", "10")
            + " and GEOFIREDTMIN "
            + fire.get("min", "1")
            + " replace the code defaults 10 and 1. "
            "Pairs in this archive sit near 0.02 sigma, and the default first step "
            "moves those atoms by more than the cell."
        )
    if ceiling is None:
        lines.append(
            "# No energy_above_minimum is stated. partial is true and the row is not ranked. "
            "A Cambridge reference ends the search at that energy plus 0.001. "
            "reference_bound marks a start that reached it. force_calls is that count. "
            "The start is not a ranked success."
        )
    else:
        lines.append(
            f"# Success is a relaxed enthalpy at or below {ceiling:.12g}, "
            "the reference plus energy_above_minimum. The count stops before the next hop is charged."
        )
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _publish(output: Path, records: list[dict], problem: dict, template: dict, *, expected: int, ceiling) -> dict:
    ordered = sorted(records, key=lambda row: int(row["index"]))
    public = []
    for record in ordered:
        row = dict(record)
        row.pop("index", None)
        row.pop("log_tail", None)
        public.append(row)
    cap = cap_of(problem, template)
    partial = ceiling is None or len(public) < expected
    metrics = score_v2(
        [
            {
                "force_calls": row["force_calls"],
                "success": row["success"],
                "failure": row["failure"],
                "reference_bound": bool(row.get("reference_bound")),
            }
            for row in public
        ],
        cap,
    )
    counted = [int(row["calls_total"]) for row in public]
    legacy = summarize_capped(counted, cap if cap is not None else max(counted))
    successes = [int(row["force_calls"]) for row in public if row["success"]]
    legacy["median"] = statistics.median(successes) if successes else statistics.median(
        [int(row["force_calls"]) for row in public]
    )
    energies = [float(row["energy"]) for row in public if row.get("energy") is not None]
    if energies:
        legacy["energy_min"] = min(energies)
        legacy["energy_max"] = max(energies)
    to_best = [int(row["calls_to_best"]) for row in public if row.get("calls_to_best") is not None]
    if to_best:
        legacy["calls_to_best_mean"] = statistics.fmean(to_best)
    version = str(template.get("code_version") or "")
    commit = str(template.get("commit") or "")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _dat(legacy, template, metrics, commit, version, partial=partial, ceiling=ceiling, cap=cap),
        encoding="utf-8",
    )
    write_starts(output.with_suffix(".starts.jsonl"), public)
    provenance = collect_provenance(
        code=str(template.get("code") or "FLAME"),
        code_version=version,
        commit=commit,
        parameters={
            "seed": int((template.get("flame") or {}).get("seed", 0)),
            "cap": cap,
            "ceiling": ceiling,
            "cell_angstrom": CELL_ANG,
            "ioput": ioput_text(template).strip(),
            "earr_tolerance": earr_text(template).strip(),
            "mddtinit": (template.get("flame") or {}).get("mddtinit"),
            "softnit": (template.get("flame") or {}).get("softnit"),
            "mdmininit": (template.get("flame") or {}).get("mdmininit"),
            "geotolmxf": (template.get("flame") or {}).get("geotolmxf"),
            "geofiredtinit": (template.get("flame") or {}).get("geofiredtinit"),
            "geofiredtmin": (template.get("flame") or {}).get("geofiredtmin"),
        },
    )
    published_metrics = dict(metrics)
    published_metrics["scoring"] = "par2-bootstrap"
    document = {
        "schema": "optbench.result/v2",
        "status": "partial" if partial else "archive",
        "problem": problem["id"],
        "scoring": "par2-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "partial": partial,
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "provenance": provenance,
                "metrics": published_metrics,
                "starts": public,
            }
        ],
    }
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "force_calls": legacy.get("force_calls"),
        "nfailed": metrics["nfailed"],
        "starts": len(public),
        "par2_mean": metrics["par2_mean"],
        "template": template["id"],
        "problem": problem["id"],
        "partial": partial,
        "successes": metrics["successes"],
        "energy_min": legacy.get("energy_min"),
    }


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None = None,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    binary = flame_binary()
    if not binary.is_file():
        raise RuntimeError(f"FLAME binary is missing at {binary}. Set FLAME to the built src/flame.")
    commit = str(template.get("commit") or "")
    if not commit:
        raise RuntimeError(f"{template['id']} has no commit")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    ceiling = success_ceiling(problem, template)
    cap = cap_of(problem, template)
    flame = template.get("flame") or {}
    seed = int(flame.get("seed", 314159))
    nlminx = int(flame.get("nlminx", DEFAULT_NLMINX))
    fire_dt = fire_dt_of(template)
    md_lines = md_lines_of(template)
    ioput = ioput_text(template)
    earr = earr_text(template)
    loaded = load_geometries(root, problem, geometries, work, output)
    if limit > 0:
        loaded = loaded[: int(limit)]
    expected = int((problem.get("sample") or {}).get("starts") or len(loaded))
    done: dict[str, dict] = {}
    starts_path = output.with_suffix(".starts.jsonl")
    if resume and starts_path.is_file():
        for line in starts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[str(row["start"])] = row
    scratch = work or output.parent / "work" / template["id"].replace("/", "-")
    jobs = []
    for index, (name, coords, _box) in enumerate(loaded):
        if name in done:
            done[name]["index"] = index
            continue
        jobs.append(
            {
                "index": index,
                "start": name,
                "coords": coords.reshape(-1).tolist(),
                "port": port_base(str(problem["id"])) + index,
                "work": str(scratch / f"{index:04d}"),
                "binary": str(binary),
                "seed": seed,
                "nlminx": nlminx,
                "fire_dt": fire_dt,
                "md_lines": md_lines,
                "ioput": ioput,
                "earr": earr,
                "cap": cap,
                "target": search_target(problem, template),
                "claim": ceiling is not None,
                "idle_timeout": 1800,
            }
        )
    records = list(done.values())
    if jobs:
        if workers <= 1:
            for job in jobs:
                records.append(_run_start(job))
                _publish(output, records, problem, template, expected=expected, ceiling=ceiling)
                print(
                    f"{template['id']} {len(records)}/{len(loaded)} "
                    f"start={records[-1]['start']} calls={records[-1]['force_calls']} "
                    f"energy={records[-1]['energy']} success={records[-1]['success']}",
                    flush=True,
                )
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_run_start, job) for job in jobs]
                total = len(loaded)
                for future in as_completed(futures):
                    records.append(future.result())
                    _publish(output, records, problem, template, expected=expected, ceiling=ceiling)
                    last = records[-1]
                    print(
                        f"{template['id']} {len(records)}/{total} "
                        f"start={last['start']} calls={last['force_calls']} "
                        f"energy={last['energy']} success={last['success']}",
                        flush=True,
                    )
    if not records:
        raise RuntimeError(f"{template['id']} produced no starts")
    return _publish(output, records, problem, template, expected=expected, ceiling=ceiling)


# Re-export the monitor parser's unit constant for callers that print energies.
HARTREE_EV = HA_EV
BOHR = BOHR_ANG
