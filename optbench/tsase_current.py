"""Current TSASE main, stock Lanczos, on the LJ38 lowest-eigenvector sample.

The class is ``lanczos_atoms`` from TSASE commit 98aac60 (setup.py
version 1.0, current main). The constructor is called with the geometry
only. ``phi_tol``, ``rotationMax``, ``dR``, ``ss``, and the initial mode
stay at the class defaults, and ``minmodesearch`` is the library method.
``numpy.random.seed(0)`` is set once so the stock ``vrand`` draw is
repeatable; it is not a search parameter.

The harness stop is the archive rule: absolute overlap 0.99 with the
reference mode, and the problem cap of 10000. The search is not continued
to chase that overlap.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from optbench.campaign import load_template
from optbench.counter import HarnessCounter, attach_ase_counter
from optbench.engines.common import judge_mode, write_campaign
from optbench.engines.geometries import load_starts, point_settings
from optbench.engines.tsase_search import _cell, _surface
from optbench.problems import load_problem

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "98aac60b5d9905e81ee4e6b8c385f169258d0218"
TEMPLATE_ID = "tsase/lj38-lanczos"
PROBLEM_ID = "lowest-eigenvector/lj38"
SEED = 0


def source_root() -> Path:
    env = os.environ.get("TSASE_SRC")
    if env:
        return Path(env)
    checkout = Path("/tmp/tsase-main")
    if (checkout / "tsase" / "dimer" / "lanczos.py").is_file():
        return checkout
    return Path("/tmp/src/tsase")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lanczos_class():
    """``lanczos_atoms`` loaded without ``tsase/__init__.py``."""
    root = source_root() / "tsase"
    lanczos_path = root / "dimer" / "lanczos.py"
    if not lanczos_path.is_file():
        raise RuntimeError(f"TSASE lanczos.py is not under {source_root()}. Set TSASE_SRC.")
    if "tsase" not in sys.modules:
        package = types.ModuleType("tsase")
        package.__path__ = [str(root)]
        package.__file__ = str(root / "__init__.py")
        sys.modules["tsase"] = package
    neb_name = "tsase.neb"
    if neb_name not in sys.modules:
        neb = types.ModuleType(neb_name)
        neb.__path__ = [str(root / "neb")]
        neb.__file__ = str(root / "neb" / "__init__.py")
        sys.modules[neb_name] = neb
        sys.modules["tsase"].neb = neb
    _module("tsase.neb.util", root / "neb" / "util.py")
    if "tsase.io" not in sys.modules:
        io_mod = types.ModuleType("tsase.io")
        io_mod.read_con = None
        sys.modules["tsase.io"] = io_mod
        sys.modules["tsase"].io = io_mod
    dimer_name = "tsase.dimer"
    if dimer_name not in sys.modules:
        dimer = types.ModuleType(dimer_name)
        dimer.__path__ = [str(root / "dimer")]
        dimer.__file__ = str(root / "dimer" / "__init__.py")
        sys.modules[dimer_name] = dimer
        sys.modules["tsase"].dimer = dimer
    _module("tsase.dimer.ssdimer", root / "dimer" / "ssdimer.py")
    loaded = _module("tsase.dimer.lanczos", lanczos_path)
    return loaded.lanczos_atoms


def _overlap(mode: np.ndarray, reference: np.ndarray, n_atoms: int) -> float:
    piece = np.asarray(mode, dtype=np.float64).reshape(-1, 3)[:n_atoms].reshape(-1)
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    scale = float(np.linalg.norm(piece))
    target_scale = float(np.linalg.norm(target))
    if scale < 1e-14 or target_scale < 1e-14:
        return 0.0
    return abs(float(np.dot(piece / scale, target / target_scale)))


def search_start(cls, start: dict, settings: dict) -> dict:
    """One stock ``get_forces`` call, then the archive overlap test."""
    from ase import Atoms

    positions = np.asarray(start["positions"], dtype=np.float64)
    free = np.array([not bool(flag) for flag in start["fixed"]], dtype=bool)
    cap = int(settings["cap"])
    counter = HarnessCounter(cache=False, cap=cap)
    error = ""
    overlap = 0.0
    try:
        atoms = Atoms(positions=positions, cell=_cell(positions, None), pbc=False)
        atoms.calc = attach_ase_counter(
            _surface(settings["potential"], free, None)(),
            cap=cap,
            counter=counter,
        )
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            search = cls(R0=atoms)
            search.get_forces()
            overlap = _overlap(search.get_mode(), start["reference"], positions.shape[0])
    except Exception as exc:
        message = str(exc)
        if "force-call cap" in message or type(exc).__name__ == "ForceCallCap":
            error = f"force-call cap: {message}"
        else:
            error = f"{type(exc).__name__}: {exc}"
    record = judge_mode(start["name"], int(counter.calls), cap, overlap, float(settings["overlap"]), error)
    record["seed"] = str(SEED)
    return record


def run(root: Path | None = None, output: Path | None = None) -> dict:
    root = Path(root or ROOT)
    template = load_template(root, TEMPLATE_ID)
    problem = load_problem(root / "problems" / "lowest-eigenvector" / "lj38.toml")
    settings = point_settings(problem)
    scratch = root / "results" / "campaign" / "work" / "tsase-lj38-lanczos"
    starts = load_starts(root, problem, scratch)
    cls = lanczos_class()
    np.random.seed(SEED)
    records = [search_start(cls, start, settings) for start in starts]
    destination = output or (root / "results" / "campaign" / "tsase-1.0-lj38-lanczos.dat")
    comment = (
        "# counter harness. One force call is one ASE calculate inside stock lanczos_atoms.\n"
        "# TSASE 1.0 commit 98aac60, constructor defaults only. numpy.random.seed(0) pins vrand.\n"
        "# Success is absolute overlap 0.99. The search is not extended to reach it."
    )
    return write_campaign(
        destination,
        records,
        template,
        problem,
        settings,
        {"constructor": "defaults", "random_seed": SEED},
        images=None,
        comment=comment,
    )


def main() -> None:
    summary = run()
    print(summary)


if __name__ == "__main__":
    main()
