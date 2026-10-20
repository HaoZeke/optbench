#!/usr/bin/env python3
"""One OPTIM child launched by PATHSAMPLE.

Native LJ OPTIM reads the reactant from POINTS in odata. PATHSAMPLE's
MAKESTART writes those coordinates to start.<extension> and the product
to finish.<extension>. When start.<extension> is present, this rewrites
odata.<extension> with the stock connect keywords and those POINTS, then
runs the df8da599 OPTIM binary.

The pair total is the sum of the child energy+gradient lines. Once that
sum reaches the problem cap, PATHSAMPLE is stopped. The child that
crossed the cap is allowed to print its summary first.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from optbench.native_calls import energy_gradient_calls  # noqa: E402


def connect_odata(preamble: str, start_text: str, symbol: str) -> str:
    """Stock connect keywords plus POINTS taken from a MAKESTART file."""
    coords: list[tuple[float, float, float]] = []
    for line in start_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not coords:
        raise ValueError("start file has no coordinates")
    lines = [preamble.rstrip("\n"), "POINTS"]
    for x_coord, y_coord, z_coord in coords:
        lines.append(f"{symbol} {x_coord:.16f} {y_coord:.16f} {z_coord:.16f}")
    return "\n".join(lines) + "\n"


def _counter(path: Path) -> int:
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8").strip()
    return int(text or "0")


def _write_counter(path: Path, total: int) -> None:
    path.write_text(f"{int(total)}\n", encoding="utf-8")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pathsample_ancestors() -> list[int]:
    """PATHSAMPLE pids above this process, nearest first."""
    pid = os.getpid()
    found: list[int] = []
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            comm = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError:
            break
        ppid = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
                break
        if comm == "PATHSAMPLE":
            found.append(pid)
        pid = ppid
    return found


def _stop_driver() -> None:
    """Stop the PATHSAMPLE that launched this child. The root dies first."""
    import signal

    ancestors = _pathsample_ancestors()
    for pid in reversed(ancestors):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue


def driver_should_stop(calls: int | None, total: int, cap: int, log_text: str) -> bool:
    """True when another OPTIM child cannot change this pair.

    A missing summary, a sum at the force cap, or an endpoint RMS refusal
    all end the run. The refusal is identical on every later cycle.
    """
    if calls is None or total >= cap:
        return True
    return "Bad endpoints" in log_text


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("OPTIM extension missing")
    extension = sys.argv[1]
    cap = int(os.environ["OPTBENCH_CAP"])
    counter = Path(os.environ["OPTBENCH_CALLS_FILE"])
    optim = os.environ["OPTIM_CHILD_BIN"]
    preamble = Path(os.environ["OPTBENCH_CONNECT_PREAMBLE"]).read_text(encoding="utf-8")
    symbol = os.environ.get("OPTBENCH_SYMBOL", "AX")
    start = Path(f"start.{extension}")
    if start.is_file():
        text = connect_odata(preamble, start.read_text(encoding="utf-8", errors="replace"), symbol)
        Path(f"odata.{extension}").write_text(text, encoding="utf-8")
    if _counter(counter) >= cap:
        _stop_driver()
        return
    calls = None
    log_text = ""
    proc = subprocess.Popen(
        [optim, extension],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        sys.stdout.buffer.write(raw)
        decoded = raw.decode("utf-8", "replace")
        log_text += decoded
        if "energy+gradient calls" not in decoded:
            continue
        found = energy_gradient_calls(decoded)
        if found is not None:
            calls = found
    sys.stdout.buffer.flush()
    returncode = proc.wait()
    total = 0 if calls is None else _counter(counter) + int(calls)
    if calls is not None:
        _write_counter(counter, total)
    if driver_should_stop(calls, total, cap, log_text):
        # The child that crossed the cap, or that refused the endpoints,
        # has already printed its summary. Later cycles would repeat it.
        _stop_driver()
    sys.exit(returncode)


if __name__ == "__main__":
    main()
