"""One pyKMC ``partn_search`` in the pyKMC virtualenv.

The parent process is the optbench environment. It does not import
LAMMPS. This module runs under ``/opt/pykmc-stack/pykmc_env`` with
``PYTHONPATH`` pointing at the repository. ``partn_search`` is the
pyKMC function. The LAMMPS pair is ``zero``. Forces come from
``fix external pf/callback`` and are counted by ``HarnessCounter``.

Fixed atoms are left out of the LAMMPS system. The callback puts the
free atoms back into the full geometry, evaluates the spec potential,
and returns forces on the free atoms. The substrate therefore cannot
drift.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

STACK = Path(os.environ.get("PYKMC_STACK", "/opt/pykmc-stack"))
PYKMC_COMMIT = "9fd1f6e9085ef92642142406f3dd74e574209898"
LAMMPS_COMMIT = "9f06a79b033159c606ca880ee63f0ec82b37dc3a"
ARTN_COMMIT = "9b753517c6c0eccae51425b83eb4e04ef554550b"
IRA_COMMIT = "4bb1319042d22181a7d86e342f809c055122865f"


def library_path() -> str:
    env = os.environ.get("PYKMC_LD_LIBRARY_PATH")
    if env:
        return env
    return os.pathsep.join(
        [
            str(STACK / "lammps" / "build"),
            str(STACK / "pykmc_env" / "pypARTn" / "lib"),
        ]
    )


def python_path() -> Path:
    env = os.environ.get("PYKMC_PYTHON")
    if env:
        return Path(env)
    return STACK / "pykmc_env" / "bin" / "python"


def _git(folder: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(folder), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def stack_status() -> str | None:
    """None when the pinned checkout imports. Otherwise the reason."""
    python = python_path()
    if not python.is_file():
        return f"pyKMC python is not at {python}. Set PYKMC_PYTHON."
    expected = {
        STACK / "pyKMC": PYKMC_COMMIT,
        STACK / "lammps": LAMMPS_COMMIT,
        STACK / "artn-plugin": ARTN_COMMIT,
        STACK / "IterativeRotationsAssignments": IRA_COMMIT,
    }
    for folder, commit in expected.items():
        if not folder.is_dir():
            return f"{folder} is not present. pyKMC was not built in {STACK}."
        found = _git(folder)
        if not found.startswith(commit):
            return f"{folder} is {found or 'unreadable'}, the pin is {commit}."
    import subprocess

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = library_path() + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "1"
    probe = subprocess.run(
        [str(python), "-c", "import lammps, pypARTn, pykmc, ira_mod"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "import failed").strip().splitlines()
        return "pyKMC stack import failed: " + (detail[-1] if detail else "no stderr")
    return None


def _potential(job: dict):
    kind = str(job["kind"])
    lengths = np.asarray(job["box"], dtype=np.float64).reshape(-1)[:3].copy()
    if kind == "morse":
        from optbench.surfaces import morse_smooth_linear

        pbc = tuple(bool(flag) for flag in job["pbc"])

        def evaluate(coords: np.ndarray):
            return morse_smooth_linear(
                coords,
                depth=float(job["depth"]),
                alpha=float(job["alpha"]),
                r0=float(job["r0"]),
                cutoff=float(job["cutoff"]),
                cell=lengths,
                pbc=pbc,
            )

        return evaluate
    if kind == "eam":
        from optbench.eon_harness import eam_al

        box = np.zeros(9, dtype=np.float64)
        box[0], box[4], box[8] = (float(value) for value in lengths)

        def evaluate(coords: np.ndarray):
            energy, forces = eam_al(coords, None, box)
            gradient = -np.asarray(forces, dtype=np.float64).reshape(-1, 3)
            return float(energy), gradient

        return evaluate
    raise ValueError(f"pyKMC has no harness potential {kind}")


def _wrap(positions: np.ndarray, lengths: np.ndarray, pbc: tuple[bool, bool, bool]) -> np.ndarray:
    out = np.array(positions, dtype=np.float64, copy=True)
    for axis, periodic in enumerate(pbc):
        length = float(lengths[axis])
        if periodic and length > 0.0:
            out[:, axis] = np.mod(out[:, axis], length)
    return out


def _config(job: dict):
    from pykmc.config import Config

    return Config.model_validate(
        {
            "control": {
                "initial_config": "unused.xyz",
                "n_steps": 1,
                "engine": "lammps",
                "active_volume": False,
                "verbosity": 0,
            },
            "atomicenvironment": {
                "style": "cna",
                "rnei": float(job.get("rnei", 3.0)),
                "rcut": float(job.get("rcut", 6.5)),
            },
            "eventsearch": {
                "style": "partn",
                "nsearch": 1,
                "delr_thr": float(job["delr_thr"]),
            },
            "rateconstant": {"style": "constant", "k0": 1.0, "T": float(job.get("temperature", 300.0))},
            "psr": {"style": "ira"},
            "lammps": {"pair_style": "zero 9.5", "pair_coeff": "* *"},
            "partn": {
                "verbosity": 0,
                "ninit": int(job["ninit"]),
                "zseed": int(job["zseed"]),
                "evalf_max": int(job["evalf_max"]),
                "forc_thr": float(job["forc_thr"]),
                "push_mode": str(job["push_mode"]),
                "push_step_size": float(job["push_step_size"]),
                "push_dist_thr": float(job["push_dist_thr"]),
                "eigen_step_size": float(job["eigen_step_size"]),
                "eigval_thr": float(job["eigval_thr"]),
                "lanczos_max_size": int(job["lanczos_max_size"]),
                "lanczos_min_size": int(job["lanczos_min_size"]),
                "lanczos_disp": float(job["lanczos_disp"]),
                "lanczos_eval_conv_thr": float(job["lanczos_eval_conv_thr"]),
                "nsmooth": int(job["nsmooth"]),
                "nnewchance": int(job["nnewchance"]),
                "neigen": int(job["neigen"]),
                "nperp": int(job["nperp"]),
                "push_over": float(job["push_over"]),
                "dmax": float(job["dmax"]),
            },
            "ira": {},
        }
    )


def _pin_artn(converge_property: str, nevalf_max: int) -> None:
    """partn_search does not forward the force-norm test. Set it when forc_thr is set."""
    import pypARTn

    real = pypARTn.artn

    class _Artn(real):
        def set(self, name, val):
            super().set(name, val)
            if name == "forc_thr":
                super().set("converge_property", converge_property)
                super().set("nevalf_max", int(nevalf_max))

    pypARTn.artn = _Artn


class _Engine:
    def __init__(self, command, engine_id: int):
        self.lmp = command
        self.rank = 0
        self.engine_id = int(engine_id)

    def command(self, text: str) -> None:
        self.lmp.command(text)


def search(job: dict) -> dict:
    """Run one event search. The dict is JSON-safe."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    holder = {
        "calls": 0,
        "capped": False,
        "crashed": False,
        "diverged": False,
        "message": "",
        "ok": False,
        "positions": None,
        "artn_energy": None,
    }
    saved_out = os.dup(1)
    previous = Path.cwd()
    try:
        _search_body(job, holder)
    except Exception as exc:
        holder["crashed"] = True
        if not holder["message"]:
            holder["message"] = f"{type(exc).__name__}: {exc}"
        holder["trace"] = traceback.format_exc(limit=8)
    finally:
        try:
            os.dup2(saved_out, 1)
        finally:
            os.close(saved_out)
        try:
            os.chdir(previous)
        except OSError:
            pass
    payload = {
        "calls": int(holder["calls"]),
        "capped": bool(holder["capped"]),
        "crashed": bool(holder["crashed"]),
        "diverged": bool(holder["diverged"]),
        "ok": bool(holder["ok"]),
        "message": str(holder["message"]),
        "positions": holder["positions"],
        "artn_energy": holder["artn_energy"],
    }
    return payload


def _search_body(job: dict, holder: dict) -> None:
    import tempfile

    from optbench.counter import ForceCallCap, HarnessCounter

    positions = np.asarray(job["positions"], dtype=np.float64).reshape(-1, 3)
    free = np.asarray(job["free"], dtype=bool).reshape(-1)
    if positions.shape[0] != free.shape[0]:
        raise RuntimeError("positions and the free mask have different lengths")
    if not np.any(free):
        raise RuntimeError("the geometry has no free atom")
    lengths = np.asarray(job["box"], dtype=np.float64).reshape(-1)[:3].copy()
    pbc = tuple(bool(flag) for flag in job["pbc"])
    wrapped = _wrap(positions, lengths, pbc)
    free_index = np.flatnonzero(free)
    central = int(job["central"])
    if central < 0 or central >= len(free) or not bool(free[central]):
        raise RuntimeError(f"central atom {central} is not a free atom")
    local_central = int(np.flatnonzero(free_index == central)[0])
    mobile = np.ascontiguousarray(wrapped[free], dtype=np.float64)
    nfree = int(mobile.shape[0])
    cap = int(job["cap"])
    evaluate = _potential(job)
    counter = HarnessCounter(cache=True, cap=cap)
    reference = wrapped
    last = {"energy": None, "gradient": None}

    def callback(_caller, _step, nlocal, tag, coords, forces):
        if int(nlocal) != nfree:
            holder["diverged"] = True
            holder["message"] = f"LAMMPS exposed {nlocal} atoms, the free set has {nfree}"
            forces[:] = 0.0
            return
        ids = np.asarray(tag, dtype=np.int64) - 1
        if np.any(ids < 0) or np.any(ids >= nfree):
            holder["diverged"] = True
            holder["message"] = "LAMMPS atom id is outside the free set"
            forces[:] = 0.0
            return
        trial = np.array(reference, copy=True)
        placed = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        trial[free_index[ids]] = placed
        key = tuple(float(value) for value in trial.reshape(-1))
        try:
            fresh, cached = counter.charge(key)
            if fresh:
                energy, gradient = evaluate(trial)
                gradient = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
                counter.remember(key, (float(energy), tuple(float(value) for value in gradient.reshape(-1))))
            else:
                energy, flat = cached
                gradient = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
            local = -gradient[free_index[ids]]
            forces[:] = np.ascontiguousarray(local, dtype=np.float64)
            holder["lmp"].fix_external_set_energy_global("ext", float(energy))
            last["energy"] = float(energy)
            last["gradient"] = gradient
        except ForceCallCap as exc:
            holder["capped"] = True
            holder["message"] = f"force-call cap: {exc}"
            forces[:] = 0.0
        except Exception as exc:
            holder["crashed"] = True
            holder["message"] = f"{type(exc).__name__}: {exc}"
            forces[:] = 0.0

    _pin_artn(str(job.get("converge_property") or "norm"), int(job["evalf_max"]))
    import lammps
    from ase.data import atomic_masses, atomic_numbers
    from pykmc.enginemanager.lmpi.lammps_operations import partn_search

    element = str(job.get("element") or "Pt")
    mass = float(atomic_masses[atomic_numbers[element]])
    zlo = float(np.min(wrapped[:, 2]) - 20.0)
    zhi = float(np.max(wrapped[:, 2]) + 20.0)
    if pbc[2]:
        zlo, zhi = 0.0, float(lengths[2])
    boundary = " ".join("p" if flag else "f" for flag in pbc)
    with tempfile.TemporaryDirectory(prefix="pykmc-") as folder:
        os.chdir(folder)
        command = lammps.lammps(cmdargs=["-screen", "none", "-log", "none", "-nocite"])
        holder["lmp"] = command
        engine = _Engine(command, int(job.get("engine_id") or 0))
        xhi, yhi = float(lengths[0]), float(lengths[1])
        lines = [
            "units metal",
            "atom_style atomic",
            "dimension 3",
            f"boundary {boundary}",
            "atom_modify map array",
            "atom_modify sort 0 0.0",
            f"region box block 0.0 {xhi} 0.0 {yhi} {zlo} {zhi}",
            "create_box 1 box",
            f"mass 1 {mass}",
            "pair_style zero 9.5",
            "pair_coeff * *",
            "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check no binsize 1000",
            "fix ext all external pf/callback 1 1",
            "fix_modify ext energy yes",
        ]
        command.commands_string("\n".join(lines))
        count = nfree
        ident = np.arange(1, count + 1, dtype=np.int32)
        types = np.ones(count, dtype=np.int32)
        command.create_atoms(
            count,
            ident,
            types,
            np.ascontiguousarray(mobile.reshape(-1), dtype=np.float64),
        )
        command.set_fix_external_callback("ext", callback)
        config = _config(job)
        outcome = partn_search(engine, config, local_central, positions=mobile)
        command.close()
    holder["calls"] = int(counter.calls)
    if holder["capped"]:
        holder["ok"] = False
        return
    if holder["crashed"] or holder["diverged"]:
        holder["ok"] = False
        return
    if outcome is None or not outcome.is_ok():
        holder["ok"] = False
        if outcome is not None:
            err = outcome.err_value()
            holder["message"] = f"{getattr(err, 'type', 'error')}: {getattr(err, 'message', err)}"
        elif not holder["message"]:
            holder["message"] = "partn_search returned no result"
        return
    value = outcome.ok_value()
    saddle = np.asarray(value.saddle_positions, dtype=np.float64).reshape(-1, 3)
    if saddle.shape != mobile.shape:
        holder["crashed"] = True
        holder["message"] = f"saddle shape {saddle.shape} is not {mobile.shape}"
        return
    full = np.array(reference, copy=True)
    full[free] = saddle
    holder["positions"] = full.reshape(-1).tolist()
    holder["ok"] = True
    try:
        holder["artn_energy"] = float(value.dE_forward)
    except (TypeError, ValueError):
        holder["artn_energy"] = None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m optbench.pykmc_worker JOB.json RESULT.json", file=sys.stderr)
        return 2
    spec = Path(argv[1])
    dest = Path(argv[2])
    job = json.loads(spec.read_text(encoding="utf-8"))
    payload = search(job)
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
