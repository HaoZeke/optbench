"""Builds and rewrites of shared files survive concurrent callers."""

from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from optbench import fastpot


def _together(call, workers: int = 6) -> list:
    gate = threading.Barrier(workers)

    def run(_index):
        gate.wait()
        return call()

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(run, range(workers)))


@pytest.fixture
def gcc():
    if shutil.which("gcc") is None:
        pytest.skip("gcc is not installed")


def test_concurrent_fastpot_compiles_publish_one_library(gcc, tmp_path):
    dest = tmp_path / "kernels.so"
    assert all(_together(lambda: fastpot._compile(fastpot._KERNEL_SRC, dest, ["-fPIC", "-shared"])))
    assert [entry.name for entry in tmp_path.iterdir()] == [dest.name]
    assert ctypes.CDLL(str(dest)).optbench_lj_cut


def test_concurrent_fastpot_compiles_publish_one_bridge(gcc, tmp_path):
    dest = tmp_path / "extpot-bridge"
    assert all(_together(lambda: fastpot._compile(fastpot._BRIDGE_SRC, dest, [])))
    assert [entry.name for entry in tmp_path.iterdir()] == [dest.name]
    assert os.access(dest, os.X_OK)


def test_concurrent_gmin_builds_share_one_library(gcc, tmp_path, monkeypatch):
    from optbench.gmin_pot import _lj40_path

    monkeypatch.setenv("OPTBENCH_FASTPOT_DIR", str(tmp_path))
    paths = set(_together(_lj40_path))
    assert len(paths) == 1
    (library,) = paths
    assert [entry.name for entry in tmp_path.iterdir()] == [library.name]
    assert ctypes.CDLL(str(library)).optbench_lj40


def test_concurrent_surface_builds_share_one_library(gcc, tmp_path, monkeypatch):
    from optbench import surfaces

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    source = Path(surfaces.__file__).with_name("morse_pair.c")
    paths = set(_together(lambda: surfaces._morse_path(source)))
    assert len(paths) == 1
    (library,) = paths
    assert [entry.name for entry in tmp_path.iterdir()] == [library.name]
    assert ctypes.CDLL(str(library)).morse_pair_energy
