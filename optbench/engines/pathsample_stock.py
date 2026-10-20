"""PATHSAMPLE double-ended search for the LJ38 two-transition pairs.

PATHSAMPLE does not call the potential. It quenches the two endpoints
with OPTIM, then launches one child OPTIM connect. The score is the sum
of those children's ``energy+gradient calls`` lines. The connect deck is
the archived odata, copied as written. ``NEWCONNECT`` stays at 20 cycles,
``NEWNEB`` stays commented, and ``STEPS`` stays 200. ``CONNECTIONS 1 1``
and ``CYCLES 1`` are the archived attempt limits. Endpoint ``BFGSSTEPS``
stays 10000, the archived LJ38 minimizer. ``DUMPALLPATHS`` is appended
because this PATHSAMPLE reads ``path.info`` triples and otherwise stops
after a child that already found a path. It does not change the
energy+gradient count. The problem force-call cap is only the outer
budget on that sum.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from optbench.native_calls import connected_path, energy_gradient_calls

from .common import run_jobs, write_campaign
from .optim_dneb import VERSION, read_xyz

SOURCE_COMMIT = "df8da599267534fbe43a250e578f7396a7946311"
_ROOT = Path(__file__).resolve().parents[2]
_PREAMBLE = _ROOT / "templates" / "optim" / "preambles" / "de-2ts.txt"
_ENDPOINT = _ROOT / "templates" / "optim" / "preambles" / "min-lj.txt"
_CHILD = Path(__file__).resolve().parent / "pathsample_child.py"
_OPTIM = Path("/tmp/wales-latest/build/OPTIM")
_PATHSAMPLE = Path("/tmp/wales-latest/build-pathsample/PATHSAMPLE")


def optim_binary() -> Path:
    env = os.environ.get("OPTIM_DNEB_BIN") or os.environ.get("OPTIM_BIN")
    if env:
        return Path(env)
    return _OPTIM


def pathsample_binary() -> Path:
    env = os.environ.get("PATHSAMPLE_BIN")
    if env:
        return Path(env)
    return _PATHSAMPLE


def _points(coords: np.ndarray, symbol: str) -> list[str]:
    lines = ["POINTS"]
    for row in np.asarray(coords, dtype=np.float64):
        lines.append(f"{symbol} {row[0]:.16f} {row[1]:.16f} {row[2]:.16f}")
    return lines


def endpoint_text(coords: np.ndarray, cap: int, symbol: str = "AX") -> str:
    """Archived minimizer keywords plus POINTS. ``cap`` is not written here."""
    del cap
    preamble = _ENDPOINT.read_text(encoding="utf-8")
    if not preamble.endswith("\n"):
        preamble += "\n"
    lines = [preamble.rstrip("\n"), *_points(coords, symbol)]
    return "\n".join(lines) + "\n"


def connect_preamble_text() -> str:
    """Archived odata plus ``DUMPALLPATHS``.

    The archived deck does not set ``DUMPALLPATHS``. This PATHSAMPLE reads
    min-sad-min triples from ``path.info.<pid>`` and stops with
    ``no path.info file`` when that file is absent, including after the
    child has printed ``Connected path found``. The keyword opens that
    file. It is not an iteration limit, and it is not written into the
    standalone OPTIM deck.
    """
    text = _PREAMBLE.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    if not any(line.strip() == "DUMPALLPATHS" for line in text.splitlines()):
        text += "DUMPALLPATHS\n"
    return text


def pathdata_text(natoms: int, cap: int, executable: str, symbol: str = "AX") -> str:
    """PATHSAMPLE keywords. Attempt limits are the archived deck, not the cap."""
    del cap
    lines = [
        f"NATOMS {int(natoms)}",
        f"SYSTEM {symbol}",
        "PERMDIST",
        "MAKESTART",
        "DIJINITSTART",
        # CONNECTIONS above 2 selects the older CYCLE loop. The archived
        # deck asks for one connection and one transition-state attempt.
        "CONNECTIONS 1 1",
        "CYCLES 1",
        "CPUS 1",
        "ETOL 1.0D-6",
        "GEOMDIFFTOL 0.05D0",
        "SEED 1",
        "DIRECTION AB",
        "TEMPERATURE 0.1D0",
        f"EXEC {executable}",
    ]
    return "\n".join(lines) + "\n"


def classify_pair(calls: int | None, connected: bool, cap: int, missing: bool) -> tuple[int, bool, str | None]:
    """Summed child calls, success, and the failure name.

    A missing child summary makes the sum incomplete. That pair is a crash
    charged at the cap, unless the children that did report already reached
    the cap.
    """
    if calls is not None and calls >= cap:
        return int(calls), False, "cap"
    if calls is None or missing:
        return int(cap), False, "crash"
    if not connected:
        return int(calls), False, "divergence"
    return int(calls), True, None


def _child_logs(folder: Path) -> list[Path]:
    logs: list[Path] = []
    for name in ("output.start", "output.finish"):
        path = folder / name
        if path.is_file():
            logs.append(path)
    for path in sorted(folder.glob("OPTIM.*")):
        if path.is_file():
            logs.append(path)
    return logs


def _sum_children(folder: Path) -> tuple[int | None, bool, list[str]]:
    """Sum of child counters, and the log names that never printed one."""
    total = 0
    seen = False
    missing: list[str] = []
    for path in _child_logs(folder):
        text = path.read_text(encoding="utf-8", errors="replace")
        calls = energy_gradient_calls(text)
        if calls is None:
            missing.append(path.name)
            continue
        seen = True
        total += int(calls)
    return (total if seen else None), seen, missing


def child_terminal_line(text: str) -> str:
    """A child line that actually stopped the search.

    ``Bad endpoints`` is printed before ``REOPTIMISEENDPOINTS`` quenches
    the end and continues. It is the stop only when that quench did not run.
    """
    quenched = "Reoptimised" in text or "Connected path found" in text
    chosen = ""
    for line in text.splitlines():
        stripped = line.strip()
        if "no connections left" in stripped:
            chosen = stripped
        elif not quenched and "Bad endpoints" in stripped:
            chosen = stripped
    return chosen


def _stop_reason(log: str, connected: bool, error: str) -> str:
    if error:
        return error
    if connected:
        return "Connected path found"
    markers = (
        "no path.info",
        "ERROR",
        "Bad endpoints",
        "no connections left",
        "Dijinit>",
        "cycle2> ERROR",
        "setup> ERROR",
        "keywords> ERROR",
    )
    chosen = ""
    for line in log.splitlines():
        text = line.strip()
        if any(marker in text for marker in markers):
            chosen = text
    return chosen or "PATHSAMPLE finished without a connected path"


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name = str(job["name"])
    cap = int(job["cap"])
    calls = None
    connected = False
    missing: list[str] = []
    error = ""
    driver = ""
    children: list[dict] = []
    child_connected = False
    child_stop = ""
    kept = str(job.get("keep") or "")
    folder_cm = None
    try:
        if kept:
            folder = Path(kept)
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
        else:
            folder_cm = tempfile.TemporaryDirectory(prefix="pathsample-stock-")
            folder = Path(folder_cm.name)
        try:
            symbol = str(job["symbol"])
            (folder / "odata.start").write_text(job["start_odata"], encoding="utf-8")
            (folder / "odata.finish").write_text(job["finish_odata"], encoding="utf-8")
            preamble_path = folder / "connect.preamble"
            preamble_path.write_text(job["connect_preamble"], encoding="utf-8")
            (folder / "odata.connect").write_text(job["connect_preamble"], encoding="utf-8")
            (folder / "pathdata").write_text(
                pathdata_text(int(job["natoms"]), cap, str(job["child"]), symbol),
                encoding="utf-8",
            )
            counter = folder / "child_calls.txt"
            counter.write_text("0\n", encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["OPTIM_CHILD_BIN"] = str(job["optim"])
            env["OPTBENCH_CONNECT_PREAMBLE"] = str(preamble_path)
            env["OPTBENCH_CAP"] = str(cap)
            env["OPTBENCH_CALLS_FILE"] = str(counter)
            env["OPTBENCH_SYMBOL"] = symbol
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(
                    [str(job["pathsample"])],
                    cwd=folder,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=28800)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=30)
                    returncode = proc.returncode if proc.returncode is not None else -1
                    error = "PATHSAMPLE exceeded the wall limit"
            driver = stdout_path.read_text(encoding="utf-8", errors="replace")
            if returncode not in (0, -15, -signal.SIGTERM) and error == "" and "Connected path found" not in driver:
                tail = stderr_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                detail = tail[-1] if tail else f"exit {returncode}"
                if "ERROR" in driver or returncode not in (0, None):
                    error = f"PATHSAMPLE exit {returncode}: {detail}"
            calls, _seen, missing = _sum_children(folder)
            connected = connected_path(driver)
            child_connected = False
            child_stop = ""
            for path in _child_logs(folder):
                text = path.read_text(encoding="utf-8", errors="replace")
                child_connected = child_connected or connected_path(text)
                children.append({"log": path.name, "calls": energy_gradient_calls(text)})
                terminal = child_terminal_line(text)
                if terminal:
                    child_stop = terminal
        finally:
            if folder_cm is not None:
                folder_cm.cleanup()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    force_calls, success, failure = classify_pair(calls, connected, cap, bool(missing))
    if error and failure not in {"cap", None} and (calls is None or missing):
        failure = "crash"
        success = False
        force_calls = cap
    elif error and not success and failure != "cap":
        failure = "crash"
        success = False
        if calls is not None:
            force_calls = int(calls)
    if failure == "cap":
        reason = (
            f"sum of child OPTIM energy+gradient calls is {int(force_calls)}, "
            f"at or above the cap {cap}"
        )
        if child_connected or connected:
            reason += "; a child OPTIM reported Connected path found"
    elif success:
        reason = "Connected path found"
    elif child_stop:
        reason = child_stop
    else:
        reason = _stop_reason(driver, False, error)
    if missing and failure == "crash":
        missing_names = ", ".join(missing)
        reason = f"{reason}; no energy+gradient line in {missing_names}"
    return {
        "start": name,
        "force_calls": int(force_calls),
        "success": success,
        "failure": failure,
        "counter": "harness",
        "converged": bool(connected and success),
        "pathsample_child_calls": None if calls is None else int(calls),
        "pathsample_children": children,
        "pathsample_stop": reason,
        **({"error": error} if error else {}),
    }


def run_pathsample_stock(root: Path, problem: dict, template: dict, output: Path, workers: int) -> dict:
    optim = optim_binary()
    pathsample = pathsample_binary()
    if not optim.is_file():
        raise RuntimeError(f"OPTIM {VERSION} is not available. {optim} is not present.")
    if not pathsample.is_file():
        raise RuntimeError(f"PATHSAMPLE {VERSION} is not available. {pathsample} is not present.")
    if not _PREAMBLE.is_file() or not _ENDPOINT.is_file() or not _CHILD.is_file():
        raise RuntimeError("PATHSAMPLE stock keywords or the OPTIM child launcher is missing")
    from optbench.engines.geometries import _extract, _stored

    archive = _stored(Path(root), str((problem.get("input") or {})["archive_path"]))
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    folder = _extract(archive, scratch / "geometries")
    coords = folder / "coords"
    if not coords.is_dir():
        found = list(folder.rglob("start_0.xyz"))
        if not found:
            raise FileNotFoundError(f"{archive} has no start_0.xyz")
        coords = found[0].parent
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    sample = int((problem.get("sample") or {}).get("pairs") or 51)
    symbol = str((template.get("pathsample") or {}).get("symbol") or "AX")
    connect = connect_preamble_text()
    jobs = []
    for index in range(sample):
        reactant = read_xyz(coords / f"start_{index}.xyz")
        product = read_xyz(coords / f"end_{index}.xyz")
        jobs.append(
            {
                "name": f"pair_{index:02d}",
                "cap": cap,
                "natoms": int(reactant.shape[0]),
                "symbol": symbol,
                "optim": str(optim),
                "pathsample": str(pathsample),
                "child": str(_CHILD),
                "connect_preamble": connect,
                "start_odata": endpoint_text(reactant, cap, symbol),
                "finish_odata": endpoint_text(product, cap, symbol),
            }
        )
    records = run_jobs(jobs, _execute, workers)
    comment = (
        f"# PATHSAMPLE {VERSION}, PATHSAMPLE/source/SVNREV in the current public svn.tar.bz2. "
        "Child OPTIM is the same tree. The connect deck is the archived odata. "
        "DUMPALLPATHS is appended so path.info triples exist. "
        "CONNECTIONS 1 1 and CYCLES 1 are the archived attempt limits. "
        "NEWCONNECT stays at 20 cycles. Endpoint BFGSSTEPS stays 10000. "
        f"The outer force-call budget on the sum is {cap}.\n"
        "# Archive pairs 0-50. The count is the sum of child OPTIM energy+gradient calls. "
        "A success is Connected path found under that budget."
    )
    published = dict(template)
    published["code"] = "PATHSAMPLE"
    images = (problem.get("band") or {}).get("interior_images")
    return write_campaign(
        output,
        records,
        published,
        problem,
        {"cap": cap},
        {
            "method": "pathsample-stock",
            "counter": "sum of child OPTIM energy+gradient calls",
            "potential": "LJDIFF",
            "version": VERSION,
            "keywords": "archived odata",
            "cycles": 1,
            "connections": 1,
            "maxtsattempts": 1,
            "newconnect_cycles": 20,
            "steps": 200,
            "endpoint_bfgssteps": 10000,
            "dumpallpaths": True,
            "pairs": sample,
        },
        images=int(images) if images else None,
        comment=comment,
    )
