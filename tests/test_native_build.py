"""Two processes can publish one shared library."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_two_processes_publish_one_shared_library(tmp_path: Path):
    if shutil.which("gcc") is None:
        pytest.skip("gcc is not installed")
    source = Path(__file__).resolve().parents[1] / "codes" / "gmin" / "lj40.c"
    dest = tmp_path / "optbench-lj40.so"
    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import subprocess
            import sys
            from pathlib import Path

            from optbench.native_build import exclusive_publish

            source = Path(sys.argv[1])
            dest = Path(sys.argv[2])

            def build(temporary: Path) -> bool:
                subprocess.check_call(
                    ["gcc", "-O3", "-shared", "-fPIC", str(source), "-o", str(temporary)]
                )
                return True

            exclusive_publish(source, dest, build)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    processes = [
        subprocess.Popen([sys.executable, str(worker), str(source), str(dest)])
        for _ in range(2)
    ]
    codes = [process.wait(timeout=120) for process in processes]
    assert codes == [0, 0]
    assert dest.is_file()
    assert dest.stat().st_size > 0
    assert list(tmp_path.glob("*.tmp")) == []
    loaded = ctypes.CDLL(str(dest))
    assert hasattr(loaded, "optbench_lj40")
