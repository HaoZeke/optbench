"""PATHSAMPLE stock keywords and the child-call score. The binary is not required."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np

from optbench.engines.pathsample_child import connect_odata, driver_should_stop
from optbench.engines.pathsample_stock import (
    apply_endpoint_cap,
    classify_pair,
    endpoint_text,
    pathdata_text,
)
from optbench.native_calls import energy_gradient_calls

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "optbench" / "engines" / "pathsample_child.py"


def test_iteration_limits_are_the_problem_cap():
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    endpoint = endpoint_text(coords, 250000)
    assert "bfgssteps 250000\n" in endpoint
    assert "DUMPDATA\n" in endpoint
    assert "NOFRQS\n" in endpoint
    assert endpoint.rstrip().endswith("AX 4.0000000000000000 5.0000000000000000 6.0000000000000000")
    raised = apply_endpoint_cap("bfgssteps 10000\nDUMPDATA\n", 250000)
    assert raised.startswith("bfgssteps 250000\n")
    pathdata = pathdata_text(38, 250000, "/tmp/optim-child")
    assert "CONNECTIONS 1 250000\n" in pathdata
    assert "CYCLES 250000\n" in pathdata
    assert "CPUS 1\n" in pathdata
    assert "EXEC /tmp/optim-child\n" in pathdata
    assert "DIJINITSTART\n" in pathdata
    assert "MAKESTART\n" in pathdata


def test_connect_odata_uses_the_makestart_coordinates():
    text = connect_odata("BFGSSTEPS 250000\nSTEPS    200\n", "1.5 2.5 3.5\n", "AX")
    assert text.startswith("BFGSSTEPS 250000\nSTEPS    200\nPOINTS\n")
    assert "AX 1.5000000000000000 2.5000000000000000 3.5000000000000000\n" in text


def test_the_sum_succeeds_only_under_the_cap():
    assert classify_pair(1200, True, 250000, False) == (1200, True, None)
    assert classify_pair(250000, True, 250000, False) == (250000, False, "cap")
    assert classify_pair(400000, False, 250000, False) == (400000, False, "cap")
    assert classify_pair(80, False, 250000, False) == (80, False, "divergence")
    assert classify_pair(None, False, 250000, True) == (250000, False, "crash")
    assert classify_pair(40, False, 250000, True) == (250000, False, "crash")
    assert classify_pair(250000, False, 250000, True) == (250000, False, "cap")


def test_hessian_line_is_not_the_child_counter():
    hessian = "OPTIM> # of energy+gradient+Hessian calls=              99 time= 0.1\n"
    assert energy_gradient_calls(hessian) is None
    both = hessian + "OPTIM> # of energy+gradient calls=              12 time= 0.1 %= 1.0\n"
    assert energy_gradient_calls(both) == 12


def test_child_adds_the_optim_count_and_rewrites_points(tmp_path: Path):
    fake = tmp_path / "OPTIM"
    fake.write_text(
        "#!/bin/sh\n"
        "echo 'OPTIM> # of energy+gradient calls=              10 time=           0.01 %=  1.0'\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    preamble = tmp_path / "preamble.txt"
    preamble.write_text("BFGSSTEPS 250000\nSTEPS    200\n", encoding="utf-8")
    (tmp_path / "start.7").write_text("1.0 2.0 3.0\n", encoding="utf-8")
    counter = tmp_path / "calls.txt"
    counter.write_text("20\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "OPTBENCH_CAP": "250000",
            "OPTBENCH_CALLS_FILE": str(counter),
            "OPTIM_CHILD_BIN": str(fake),
            "OPTBENCH_CONNECT_PREAMBLE": str(preamble),
            "OPTBENCH_SYMBOL": "AX",
        }
    )
    proc = subprocess.run(
        [sys.executable, str(CHILD), "7"],
        cwd=tmp_path,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "energy+gradient calls" in proc.stdout
    odata = (tmp_path / "odata.7").read_text(encoding="utf-8")
    assert "AX 1.0000000000000000 2.0000000000000000 3.0000000000000000" in odata
    assert "BFGSSTEPS 250000" in odata
    assert int(counter.read_text(encoding="utf-8").strip()) == 30


def test_bad_endpoints_stop_later_cycles():
    line = (
        "checkendpoints> Bad endpoints supplied - initial, final and maximum RMS: "
        "0.4688286703E-05 0.5691061065E-08 0.1000000000E-05\n"
        "OPTIM> # of energy+gradient calls=                 2 time=           0.00 %=  5.1\n"
    )
    assert driver_should_stop(2, 4, 250000, line)
    assert not driver_should_stop(1200, 1202, 250000, "Connected path found\n")
    assert driver_should_stop(None, 0, 250000, "decide> There appear to be no connections left")
    assert driver_should_stop(10, 250000, 250000, "ok")


def test_child_does_not_launch_optim_after_the_cap(tmp_path: Path):
    fake = tmp_path / "OPTIM"
    marker = tmp_path / "ran"
    fake.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    preamble = tmp_path / "preamble.txt"
    preamble.write_text("BFGSSTEPS 250000\n", encoding="utf-8")
    counter = tmp_path / "calls.txt"
    counter.write_text("250000\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "OPTBENCH_CAP": "250000",
            "OPTBENCH_CALLS_FILE": str(counter),
            "OPTIM_CHILD_BIN": str(fake),
            "OPTBENCH_CONNECT_PREAMBLE": str(preamble),
        }
    )
    proc = subprocess.run(
        [sys.executable, str(CHILD), "3"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert not marker.exists()
