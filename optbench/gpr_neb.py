"""GP-accelerated NEB through GPR calculator 0.0.4.

The search follows ``gpr_calc.NEB.neb_calc`` and the cluster scripts in
``examples/Pd4/gp_neb.py`` and ``examples/H2S/gp_neb.py`` at commit
c4d63271: one shared zero-mean GP, an IDPP initial path, ASE NEB, and
FIRE. ``GPR.calculate`` keeps the library uncertainty gate. The base
calculator is called when that gate asks for one. ``HarnessCounter``
sits on that base calculator. A surrogate prediction does not enter the
potential, so it is not a force call. Hessian finite differences use
``verification`` and the spec potential, and they are not search cost.

The problem file supplies the image count, the harness force-call cap,
the success force norm, and the maximum move. The optimizer stops on
the template force residual, or on the library step limit, or on the
cap. ``classify_outcome`` is the class. A finite stop above the force
tolerance is ``not_converged``. That class is a finished failure and
costs twice the force-call cap, the same as every other failure.
``cap`` is only the harness force-call cap. ``SEARCH_TIME_LIMIT`` is
one infrastructure slice. Crossing it writes a resume checkpoint and
is not a score. Only that slice continues. A method stop does not.
The measured duration is ``search_seconds``, which the benchmark
writes as ``time_search``.

Section 6 of Onyango, Kang, and Zhu, Comput. Phys. Commun. 316 (2025)
109781, arXiv:2504.07319, writes one numerical set because those
examples are in eV and Å. The template keeps that set. ``_unit_scales``
converts it into the potential's units.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import tomllib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem, write_starts
from .counter import ForceCallCap, HarnessCounter, attach_ase_counter
from .failures import classify_outcome
from .pes import evaluate, load_pairs
from .rgsaddle_band import vibrational_index
from .score import score_v2
from .stationary import numerical_hessian
from .v2 import collect_provenance

GPR_VERSION = "0.0.4"
GPR_COMMIT = "c4d63271a6e87a14171ed43fe85a56b38ee1bea1"
CODE_ID = "gprcalc"
CODE_NAME = "GPR calculator"
_FD_STEP = 1e-5
_SYMBOL = {"lj": "Ar", "morse_pt": "Pt"}
# Morse platinum in ``pes.py`` is the board's eV/Å metal potential.
# LJ is matched to it at the pair minimum: ε ↔ D0 and 2**(1/6) σ ↔ r0.
_MORSE_D0 = 0.7102
_MORSE_R0 = 2.897
_LJ_REQ_OVER_SIGMA = 2.0 ** (1.0 / 6.0)
# One infrastructure slice. An LJ38 initial fit at the scaled cutoff is
# about two seconds, and one GPR NEB in examples/emt-serial.py finished
# on this machine in 26.818827 s. Six hours lets a slice take many GPR
# steps. Crossing it writes ``resume/`` and is not a scored pair. The
# next slice continues from that band and that training set.
SEARCH_TIME_LIMIT = 6 * 60 * 60


class _SearchTime(Exception):
    """The infrastructure clock elapsed. The class is decided afterwards."""


def _threads_one() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _registry(root: Path) -> dict:
    data = tomllib.loads((root / "codes" / "registry.toml").read_text(encoding="utf-8"))
    for row in data["code"]:
        if row["id"] == CODE_ID:
            return row
    raise ValueError("codes/registry.toml has no gprcalc entry")


def _unit_scales(kind: str) -> dict[str, float]:
    """eV per energy unit and Å per length unit.

    Section 6 uses one set of numbers for every example because those
    calculators are in eV and Å. Morse platinum on this board is already
    those units, so both factors are 1. Lennard-Jones is ε = σ = 1.
    The bridge is the pair minimum of Morse platinum: one ε is one D0
    (0.7102 eV) and one σ is r0 / 2**(1/6) (2.897 Å / 2**(1/6)). A paper
    value with dimensions energy^a * length^b is divided by
    ``energy**a * length**b``. The kernel length is the descriptor length
    the library uses at a 5.0 Å cutoff, so it is divided by the length
    factor along with the cutoff.
    """
    if kind == "morse_pt":
        return {"energy": 1.0, "length": 1.0}
    if kind == "lj":
        return {"energy": _MORSE_D0, "length": _MORSE_R0 / _LJ_REQ_OVER_SIGMA}
    raise ValueError(f"no GPR unit scale for potential {kind}")


def _settings(template: dict, problem: dict, commit: str) -> dict:
    if template.get("commit") != commit:
        raise ValueError(f"{template['id']} commit does not match the gprcalc registry")
    if template.get("code_version") != GPR_VERSION:
        raise ValueError(f"{template['id']} version is not {GPR_VERSION}")
    band = template["band"]
    problem_band = problem["band"]
    convergence = problem["convergence"]
    kind = str(problem_band["potential"])
    scales = _unit_scales(kind)
    energy = float(scales["energy"])
    length = float(scales["length"])
    return {
        "potential": kind,
        "n_interior": int(problem_band["interior_images"]),
        "max_move": float(problem_band["max_move"]),
        "force_norm": float(convergence["force_norm"]),
        "cap": int(convergence["max_force_calls"]),
        "pairs": int(problem["sample"]["pairs"]),
        "geometry": str(problem["geometry"]["kind"]),
        "climbing": bool(band["climbing"]),
        "algo": str(band["algo"]),
        "spring_k": float(band["spring_k"]),
        "steps": int(band["steps"]),
        "passes": int(band["passes"]),
        "kernel": str(band["kernel"]),
        "zeta": float(band["zeta"]),
        "noise_e_structure": float(band["noise_e_structure"]) / energy,
        "noise_f": float(band["noise_f"]) * length / energy,
        "noise_e_floor": float(band["noise_e_floor"]) / energy,
        "optimizer_fmax": float(band["optimizer_fmax"]) * length / energy,
        "initial_path": str(band["initial_path"]),
        "lmax": int(band["lmax"]),
        "nmax": int(band["nmax"]),
        "rcut": float(band["rcut"]) / length,
        "freq": int(band["freq"]),
        "neb_method": str(band["neb_method"]),
        "sigma0": float(band["sigma0"]) / energy,
        "length0": float(band["length0"]) / length,
        "sigma_min": float(band["sigma_min"]) / energy,
        "sigma_max": float(band["sigma_max"]) / energy,
        "length_min": float(band["length_min"]) / length,
        "length_max": float(band["length_max"]) / length,
        "energy_uncertainty": bool(band.get("energy_uncertainty", False)),
        "energy_unit_eV": energy,
        "length_unit_Ang": length,
        "time_limit_seconds": SEARCH_TIME_LIMIT,
    }


def _noise(settings: dict, n_atoms: int) -> tuple[float, float]:
    """Per-atom energy noise and force noise, already in problem units.

    ``examples/Pd4/gp_neb.py`` divides the structure energy noise by the
    atom count and applies ``noise_e_min``.
    """
    structure = float(settings["noise_e_structure"])
    floor = float(settings["noise_e_floor"])
    per_atom = structure / int(n_atoms)
    return max(floor, per_atom), float(settings["noise_f"])


def _gpr():
    from .gpr_compat import install

    install()
    from gpr_calc.calculator import GPR
    from gpr_calc.gaussianprocess import GP

    return GP, GPR


def spec_calculator(kind: str, box: np.ndarray | None):
    """ASE calculator for one spec potential. Forces are the negative gradient."""
    from ase.calculators.calculator import Calculator
    from ase.calculators.calculator import all_changes

    class SpecPotential(Calculator):
        implemented_properties = ["energy", "forces"]
        nolabel = True

        def __init__(self):
            super().__init__()
            self.kind = kind
            self.box = None if box is None else np.asarray(box, dtype=np.float64).reshape(-1)
            self.last_positions = None
            self.last_energy = None
            self.last_gradient = None

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            coords = np.array(atoms.positions, dtype=np.float64, copy=True)
            pes_box = None if self.kind == "lj" else self.box
            energy, gradient = evaluate(self.kind, coords, pes_box)
            self.last_positions = coords
            self.last_energy = float(energy)
            self.last_gradient = np.array(gradient, dtype=np.float64, copy=True)
            self.results["energy"] = float(energy)
            self.results["free_energy"] = float(energy)
            self.results["forces"] = -self.last_gradient.copy()

    return SpecPotential()


def _cap_error(exc: BaseException) -> bool:
    if isinstance(exc, ForceCallCap):
        return True
    text = str(exc).lower()
    return "force-call cap" in text or "force call cap" in text


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _blank(name: str, calls: int, verify: int) -> dict:
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": int(verify),
        "counter": "harness",
        "comparable": True,
        "success": False,
        "failure": None,
        "reasons": [],
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "converged": False,
        "candidate": "highest",
        "ci_index": None,
        "base_calls": 0,
        "surrogate_calls": 0,
        "fits": 0,
        "n_energy": 0,
        "n_forces": 0,
        "optimizer_steps": 0,
        "noise_e": None,
        "noise_f": None,
        "kernel_sigma": None,
        "kernel_length": None,
        "optimizer_fmax": None,
        "force_tol": None,
        "search_seconds": 0.0,
        "time_limited": False,
        "incomplete": False,
    }


def _crash_record(name: str, calls: int, verify: int, crash: str) -> dict:
    record = _blank(name, calls, verify)
    record["failure"] = "crash"
    record["reasons"] = ["crash"]
    record["candidate"] = "crash"
    record["crash"] = crash
    return record


def _frame(coords: np.ndarray, free: np.ndarray, box: np.ndarray | None, symbol: str):
    from ase import Atoms
    from ase.constraints import FixAtoms

    atoms = Atoms(symbols=[symbol] * len(coords), positions=np.asarray(coords, dtype=np.float64))
    if box is not None:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        atoms.set_cell(np.diag(lengths))
        atoms.set_pbc(True)
    fixed = [index for index, flag in enumerate(np.asarray(free, dtype=bool)) if not flag]
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def _mic_product(reactant: np.ndarray, product: np.ndarray, free: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    out = np.array(product, dtype=np.float64, copy=True)
    if box is not None:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        delta = out - reactant
        delta -= lengths * np.round(delta / lengths)
        out = reactant + delta
    out[~np.asarray(free, dtype=bool)] = reactant[~np.asarray(free, dtype=bool)]
    return out


def _images(
    reactant,
    product,
    free,
    box,
    symbol: str,
    n_interior: int,
    spring_k: float,
    neb_method: str,
    initial_path: str,
):
    from ase.mep import NEB

    path = str(initial_path)
    if path not in {"idpp", "linear"}:
        raise ValueError(f"unknown initial path {path}")
    left = _frame(reactant, free, box, symbol)
    right = _frame(_mic_product(reactant, product, free, box), free, box, symbol)
    images = [left] + [left.copy() for _ in range(n_interior)] + [right]
    band = NEB(images, k=spring_k, parallel=False, method=neb_method)
    band.interpolate(method=path, apply_constraint=True, mic=box is not None)
    return images


def _true_gradient(image, base, counter: HarnessCounter):
    coords = np.array(image.positions, dtype=np.float64, copy=True)
    if base.last_positions is not None and np.array_equal(base.last_positions, coords):
        return base.last_energy, np.array(base.last_gradient, copy=True)
    cap = counter.cap
    if cap is not None and counter.calls >= cap:
        return None, None
    probe = image.copy()
    probe.calc = base
    probe.get_potential_energy()
    probe.get_forces()
    return base.last_energy, None if base.last_gradient is None else np.array(base.last_gradient, copy=True)


def _band_energies(images, gp) -> list[float]:
    energies = []
    train = list(gp.train_y["energy"])
    last = len(images) - 1
    for index, image in enumerate(images):
        if index in (0, last) and index < len(train):
            energies.append(float(train[index]) * len(image))
            continue
        image.calc.freeze()
        try:
            energies.append(float(image.get_potential_energy()))
        finally:
            image.calc.unfreeze()
    return energies


def _candidate_index(energies: list[float], climbing: bool) -> tuple[int, str]:
    interior = energies[1:-1]
    finite = [value for value in interior if math.isfinite(value)]
    if not finite:
        return 1, "climbing" if climbing else "highest"
    best = max(range(len(interior)), key=lambda index: interior[index] if math.isfinite(interior[index]) else -math.inf)
    return 1 + best, "climbing" if climbing else "highest"


def _judge(
    name: str,
    *,
    counter: HarnessCounter,
    image,
    base,
    free: np.ndarray,
    kind: str,
    box: np.ndarray | None,
    force_norm: float,
    cap: int,
    project_rigid: bool,
    candidate: str,
    index: int,
    climbing: bool,
    optimizer_converged: bool,
    stats: dict,
) -> dict:
    record = _blank(name, counter.calls, counter.verify_calls)
    record.update(stats)
    record["candidate"] = candidate
    record["ci_index"] = index if climbing else None
    record["converged"] = bool(optimizer_converged)
    record["force_tol"] = float(force_norm)
    energy, gradient = _true_gradient(image, base, counter)
    record["force_calls"] = int(counter.calls)
    record["verify_calls"] = int(counter.verify_calls)
    reasons: list[str] = []
    hessian_index = None
    force = None
    if counter.calls >= cap:
        reasons.append("cap")
    if gradient is None:
        force = None
    else:
        free_grad = np.asarray(gradient, dtype=np.float64)[np.asarray(free, dtype=bool)]
        force = float(np.linalg.norm(free_grad))
        record["energy"] = None if energy is None or not math.isfinite(float(energy)) else float(energy)
        record["force_norm"] = None if not math.isfinite(force) else float(force)
        if not math.isfinite(force) or force > force_norm:
            reasons.append("force_norm")
    if "cap" not in reasons and "force_norm" not in reasons and gradient is not None and math.isfinite(force):
        coords = np.array(image.positions, dtype=np.float64, copy=True)
        mask = np.asarray(free, dtype=bool)
        point = coords[mask].reshape(-1)
        pes_box = None if kind == "lj" else None if box is None else np.asarray(box, dtype=np.float64).reshape(-1)

        def gradient_free(flat):
            trial = coords.copy()
            trial[mask] = np.asarray(flat, dtype=np.float64).reshape((-1, 3))
            _energy, grad = evaluate(kind, trial, pes_box)
            counter.charge(tuple(float(value) for value in np.asarray(flat, dtype=np.float64).reshape(-1)))
            return np.asarray(grad, dtype=np.float64)[mask].reshape(-1).tolist()

        with counter.verification():
            hessian = numerical_hessian(gradient_free, point.tolist(), step=_FD_STEP)
        hessian_index = vibrational_index(
            np.asarray(hessian, dtype=np.float64),
            point.reshape((-1, 3)),
            project_rigid,
        )
        record["hessian_index"] = int(hessian_index)
        record["verify_calls"] = int(counter.verify_calls)
        record["force_calls"] = int(counter.calls)
        if hessian_index != 1:
            reasons.append("hessian_index")
    unmet = gradient is None and counter.calls < cap
    failure = classify_outcome(
        energy=None if record["energy"] is None else float(record["energy"]),
        force=None if force is None else float(force),
        calls=float(counter.calls),
        cap=int(cap),
        force_tol=float(force_norm),
        hessian_index=None if hessian_index is None else int(hessian_index),
        expected_index=1,
        stationary_rejected=hessian_index is not None and int(hessian_index) != 1,
        unmet=unmet,
    )
    record["reasons"] = reasons
    record["failure"] = failure
    record["success"] = failure is None
    return record


def _stats(gp, noise_e: float, noise_f: float, nsteps: int, optimizer_fmax: float) -> dict:
    kernel = getattr(gp, "kernel", None)
    sigma = getattr(kernel, "sigma", None) if kernel is not None else None
    length = getattr(kernel, "l", None) if kernel is not None else None
    return {
        "base_calls": int(getattr(gp, "use_base", 0) or 0),
        "surrogate_calls": int(getattr(gp, "use_surrogate", 0) or 0),
        "fits": int(getattr(gp, "fits", 0) or 0),
        "n_energy": int(getattr(gp, "N_energy", 0) or 0),
        "n_forces": int(getattr(gp, "N_forces", 0) or 0),
        "optimizer_steps": int(nsteps),
        "noise_e": float(noise_e),
        "noise_f": float(noise_f),
        "kernel_sigma": None if sigma is None else float(sigma),
        "kernel_length": None if length is None else float(length),
        "optimizer_fmax": float(optimizer_fmax),
    }


def _resume_dir(scratch: Path) -> Path:
    return Path(scratch) / "resume"


def _has_resume(scratch: Path | None) -> bool:
    return scratch is not None and (_resume_dir(scratch) / "state.json").is_file()


def _scored(row: dict) -> bool:
    """A wall-clock slice is a checkpoint. It is not a finished pair."""
    return not bool(row.get("incomplete")) and not bool(row.get("time_limited"))


def _unscored_slice(name: str, calls: int, verify: int, seconds: float) -> dict:
    record = _blank(name, calls, verify)
    record["incomplete"] = True
    record["time_limited"] = True
    record["failure"] = None
    record["success"] = False
    record["search_seconds"] = float(seconds)
    return record


def _discard_resume(scratch: Path) -> None:
    dest = _resume_dir(scratch)
    if not dest.is_dir():
        return
    for path in dest.iterdir():
        if path.is_file():
            path.unlink()
    dest.rmdir()


def _save_resume(
    scratch: Path,
    images,
    gp,
    counter: HarnessCounter,
    *,
    pass_index: int,
    steps_in_pass: int,
    optimizer_steps: int,
    search_seconds: float,
) -> None:
    """Write the band and the library model. ``state.json`` is last."""
    dest = _resume_dir(scratch)
    dest.mkdir(parents=True, exist_ok=True)
    positions = np.asarray([np.asarray(image.positions, dtype=np.float64) for image in images])
    np.save(dest / "positions.npy", positions)
    gp.save(str(dest / "model.json"), str(dest / "train.db"), verbose=False)
    payload = {
        "fits": int(getattr(gp, "fits", 0) or 0),
        "force_calls": int(counter.calls),
        "optimizer_steps": int(optimizer_steps),
        "pass_index": int(pass_index),
        "search_seconds": float(search_seconds),
        "steps_in_pass": int(steps_in_pass),
        "use_base": int(getattr(gp, "use_base", 0) or 0),
        "use_surrogate": int(getattr(gp, "use_surrogate", 0) or 0),
        "verify_calls": int(counter.verify_calls),
    }
    text = json.dumps(payload, sort_keys=True) + "\n"
    temporary = dest / "state.json.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(dest / "state.json")


def _load_resume(scratch: Path, images, counter: HarnessCounter):
    """Restore the band and rebuild the kernel weights without a new fit."""
    dest = _resume_dir(scratch)
    payload = json.loads((dest / "state.json").read_text(encoding="utf-8"))
    positions = np.load(dest / "positions.npy")
    if len(positions) != len(images):
        raise RuntimeError("resume band does not match the image count")
    for image, pos in zip(images, positions):
        image.positions[:] = np.asarray(pos, dtype=np.float64)
    from gpr_calc.gaussianprocess import GP

    gp = GP.load(str(dest / "model.json"))
    gp.fit(opt=False, show=False)
    gp.fits = int(payload["fits"])
    gp.use_base = int(payload["use_base"])
    gp.use_surrogate = int(payload["use_surrogate"])
    counter.calls = int(payload["force_calls"])
    counter.verify_calls = int(payload["verify_calls"])
    return gp, payload


def run_gp_neb(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    settings: dict,
    scratch: Path | None = None,
) -> dict:
    """One reactant/product pair. The record uses the harness counter."""
    state: dict = {"counter": None, "nsteps": 0, "gp": None, "noise_e": None, "noise_f": None}
    try:
        return _run_gp_neb(
            name=name,
            reactant=reactant,
            product=product,
            free=free,
            box=box,
            settings=settings,
            scratch=scratch,
            state=state,
        )
    except (Exception, SystemExit) as exc:
        counter = state["counter"]
        calls = 0 if counter is None else int(counter.calls)
        verify = 0 if counter is None else int(counter.verify_calls)
        if _cap_error(exc):
            record = _blank(name, calls, verify)
            record["force_tol"] = float(settings["force_norm"])
            record["force_calls"] = calls
            failure = classify_outcome(
                calls=float(calls),
                cap=int(settings["cap"]),
                force_tol=float(settings["force_norm"]),
                unmet=calls < int(settings["cap"]),
            )
            record["reasons"] = ["cap"] if failure == "cap" else ["solver"]
            record["failure"] = failure
            record["success"] = False
            gp = state["gp"]
            if gp is not None and state["noise_e"] is not None:
                record.update(
                    _stats(
                        gp,
                        state["noise_e"],
                        state["noise_f"],
                        state["nsteps"],
                        float(state["optimizer_fmax"]),
                    )
                )
            return record
        record = _crash_record(name, calls, verify, type(exc).__name__)
        gp = state["gp"]
        if gp is not None and state["noise_e"] is not None:
            record.update(
                _stats(
                    gp,
                    state["noise_e"],
                    state["noise_f"],
                    state["nsteps"],
                    float(state["optimizer_fmax"]),
                )
            )
            record["failure"] = "crash"
            record["reasons"] = ["crash"]
            record["success"] = False
            record["candidate"] = "crash"
        return record


def _run_gp_neb(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    settings: dict,
    scratch: Path | None,
    state: dict,
) -> dict:
    warnings.filterwarnings("ignore", message="Casting complex values")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"gpr_calc")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ase")
    previous = None
    if scratch is not None:
        previous = Path.cwd()
        Path(scratch).mkdir(parents=True, exist_ok=True)
        os.chdir(scratch)
    try:
        return _search_gp_neb(
            name=name,
            reactant=reactant,
            product=product,
            free=free,
            box=box,
            settings=settings,
            scratch=scratch,
            state=state,
        )
    finally:
        if previous is not None:
            os.chdir(previous)


def _new_gp(images, base, settings: dict, noise_e: float, noise_f: float):
    """One zero-mean RBF model. Bounds are set before the first fit."""
    from gpr_calc.SO3 import SO3
    from gpr_calc.gaussianprocess import GP
    from gpr_calc.kernels.RBF_mb import RBF_mb

    if str(settings["kernel"]) != "RBF":
        raise ValueError(f"unknown kernel {settings['kernel']}")
    if "sigma0" in settings:
        kernel = RBF_mb(
            para=[float(settings["sigma0"]), float(settings["length0"])],
            bounds=[
                [float(settings["sigma_min"]), float(settings["sigma_max"])],
                [float(settings["length_min"]), float(settings["length_max"])],
            ],
            zeta=float(settings["zeta"]),
        )
    else:
        kernel = RBF_mb(para=[1.0, 0.1], zeta=float(settings["zeta"]))
    gp = GP(
        kernel=kernel,
        descriptor=SO3(
            nmax=int(settings["nmax"]),
            lmax=int(settings["lmax"]),
            rcut=float(settings["rcut"]),
        ),
        noise_e=float(noise_e),
        noise_f=float(noise_f),
    )
    gp.train_images(images, base)
    return gp


def _search_gp_neb(
    *,
    name: str,
    reactant: np.ndarray,
    product: np.ndarray,
    free: np.ndarray,
    box: np.ndarray | None,
    settings: dict,
    scratch: Path | None,
    state: dict,
) -> dict:
    import time

    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE

    state["started"] = time.perf_counter()
    limit = float(settings.get("time_limit_seconds", SEARCH_TIME_LIMIT))
    state["time_limit"] = limit
    state["optimizer_fmax"] = float(settings["optimizer_fmax"])

    def _over() -> bool:
        return time.perf_counter() - float(state["started"]) >= float(state["time_limit"])

    kind = settings["potential"]
    symbol = _SYMBOL.get(kind)
    if symbol is None:
        raise ValueError(f"no GPR element for potential {kind}")
    free = np.asarray(free, dtype=bool)
    reactant = np.asarray(reactant, dtype=np.float64)
    product = np.asarray(product, dtype=np.float64)
    n_free = int(np.count_nonzero(free))
    project_rigid = bool(np.all(free) and n_free >= 2 and kind == "lj")
    cap = int(settings["cap"])
    force_norm = float(settings["force_norm"])
    if _over() and not _has_resume(scratch):
        return _unscored_slice(name, 0, 0, 0.0)
    _gp_mod, GPR = _gpr()
    noise_e, noise_f = _noise(settings, len(reactant))
    state["noise_e"] = noise_e
    state["noise_f"] = noise_f
    images = _images(
        reactant,
        product,
        free,
        box,
        symbol,
        int(settings["n_interior"]),
        float(settings["spring_k"]),
        str(settings["neb_method"]),
        str(settings["initial_path"]),
    )
    counter = HarnessCounter(cache=False, cap=cap)
    state["counter"] = counter
    base = spec_calculator(kind, box)
    attach_ase_counter(base, cap=cap, counter=counter)
    carried_seconds = 0.0
    start_pass = 0
    resume_steps_in_pass = 0
    total_steps = 0
    if _has_resume(scratch):
        gp, meta = _load_resume(scratch, images, counter)
        carried_seconds = float(meta["search_seconds"])
        start_pass = int(meta["pass_index"])
        resume_steps_in_pass = int(meta["steps_in_pass"])
        total_steps = int(meta["optimizer_steps"])
    else:
        gp = _new_gp(images, base, settings, noise_e, noise_f)
    state["gp"] = gp
    state["nsteps"] = total_steps
    calc = GPR(base=base, ff=gp, save=False, freq=int(settings["freq"]))
    # Section 3.3 uses only the force uncertainty. ``ignore_E_std`` is the
    # library switch for the energy gate: true skips it. A template sets
    # ``energy_uncertainty`` to turn that gate on.
    calc.ignore_E_std = not bool(settings.get("energy_uncertainty", False))
    train_index = len(images) - 2
    for index, image in enumerate(images):
        image.calc = copy(calc)
        image.calc.update_gpr = index == train_index
        image.calc.ignore_E_std = calc.ignore_E_std
    algo = str(settings["algo"])
    logfile = None if scratch is None else str(Path(scratch) / "optimizer.log")
    optimizer_fmax = float(settings["optimizer_fmax"])
    optimizer_converged = False
    time_limited = False
    pass_index = start_pass
    steps_before_pass = max(0, total_steps - resume_steps_in_pass)
    steps_in_pass = resume_steps_in_pass

    def _elapsed() -> float:
        return carried_seconds + (time.perf_counter() - float(state["started"]))

    try:
        for pass_index in range(start_pass, int(settings["passes"])):
            already = resume_steps_in_pass if pass_index == start_pass else 0
            steps_before_pass = total_steps - already
            steps_in_pass = already
            if _over():
                time_limited = True
                break
            remaining = int(settings["steps"]) - already
            if remaining <= 0:
                continue
            neb = NEB(
                images,
                k=float(settings["spring_k"]),
                parallel=False,
                method=str(settings["neb_method"]),
            )
            neb.climb = bool(settings["climbing"])
            if algo == "BFGS":
                opt = BFGS(neb, logfile=logfile, maxstep=float(settings["max_move"]))
            elif algo == "FIRE":
                opt = FIRE(neb, logfile=logfile, maxstep=float(settings["max_move"]))
            else:
                raise ValueError(f"unknown NEB algorithm {algo}")

            def _observe(opt=opt, neb=neb, offset=steps_before_pass + already):
                state["nsteps"] = offset + int(opt.nsteps)
                if scratch is not None:
                    residual = getattr(neb, "residuals", None)
                    fmax = ""
                    if residual is not None and len(residual) > 0:
                        fmax = f" fmax {float(np.max(residual)):.6f}"
                    line = (
                        f"step {state['nsteps']} calls {counter.calls} base {gp.use_base} "
                        f"surrogate {gp.use_surrogate} energy {gp.N_energy} forces {gp.N_forces} "
                        f"fits {gp.fits} sigma {gp.kernel.sigma:.6f} length {gp.kernel.l:.6f}"
                        f"{fmax} "
                        f"elapsed {time.perf_counter() - float(state['started']):.3f}\n"
                    )
                    with (Path(scratch) / "steps.log").open("a", encoding="utf-8") as handle:
                        handle.write(line)
                if _over():
                    raise _SearchTime()

            opt.attach(_observe)
            opt.run(fmax=optimizer_fmax, steps=remaining)
            total_steps = steps_before_pass + already + int(opt.nsteps)
            steps_in_pass = already + int(opt.nsteps)
            state["nsteps"] = total_steps
            try:
                optimizer_converged = bool(opt.converged())
            except Exception:
                optimizer_converged = False
            if optimizer_converged:
                break
    except _SearchTime:
        time_limited = True
        # The observer counted the in-progress pass. ``total_steps`` does
        # not include it until ``opt.run`` returns.
        if state.get("nsteps"):
            total_steps = int(state["nsteps"])
        steps_in_pass = total_steps - steps_before_pass
    if time_limited:
        if scratch is not None:
            _save_resume(
                scratch,
                images,
                gp,
                counter,
                pass_index=pass_index,
                steps_in_pass=steps_in_pass,
                optimizer_steps=total_steps,
                search_seconds=_elapsed(),
            )
        record = _unscored_slice(name, counter.calls, counter.verify_calls, _elapsed())
        record.update(_stats(gp, noise_e, noise_f, total_steps, optimizer_fmax))
        record["force_calls"] = int(counter.calls)
        record["incomplete"] = True
        record["time_limited"] = True
        record["failure"] = None
        record["success"] = False
        return record
    energies = _band_energies(images, gp)
    index, candidate = _candidate_index(energies, bool(settings["climbing"]))
    record = _judge(
        name,
        counter=counter,
        image=images[index],
        base=base,
        free=free,
        kind=kind,
        box=box,
        force_norm=force_norm,
        cap=cap,
        project_rigid=project_rigid,
        candidate=candidate,
        index=index,
        climbing=bool(settings["climbing"]),
        optimizer_converged=optimizer_converged,
        stats=_stats(gp, noise_e, noise_f, total_steps, optimizer_fmax),
    )
    record["time_limited"] = False
    record["search_seconds"] = _elapsed()
    return record


def _job(pair, settings: dict, scratch: str) -> dict:
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64),
        "product": np.asarray(pair.product, dtype=np.float64),
        "free": np.asarray(pair.free, dtype=bool),
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64),
        "settings": settings,
        "scratch": scratch,
    }


def _execute(job: dict) -> dict:
    _threads_one()
    scratch = Path(job["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    stale = scratch / "steps.log"
    # A resume keeps the log of the slices already run. A fresh pair
    # does not inherit a log from an earlier campaign in this directory.
    if stale.exists() and not _has_resume(scratch):
        stale.unlink()
    previous = Path.cwd()
    handle = (scratch / "run.log").open("a" if _has_resume(scratch) else "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = handle
    sys.stderr = handle
    try:
        os.chdir(scratch)
        previous_signature = None
        while True:
            record = run_gp_neb(
                name=job["name"],
                reactant=job["reactant"],
                product=job["product"],
                free=job["free"],
                box=job["box"],
                settings=job["settings"],
                scratch=scratch,
            )
            if _scored(record):
                return record
            # The slice made no further calls or steps, so another
            # immediate slice would repeat it. Leave the checkpoint.
            signature = (int(record.get("force_calls") or 0), int(record.get("optimizer_steps") or 0))
            if signature == previous_signature:
                return record
            previous_signature = signature
    finally:
        os.chdir(previous)
        sys.stdout = old_out
        sys.stderr = old_err
        handle.close()


_PAIR_CACHE: dict[tuple[str, str], dict] = {}


def cached_pairs(root: Path, problem: dict, settings: dict) -> dict:
    key = (str(Path(root).resolve()), problem["id"])
    found = _PAIR_CACHE.get(key)
    if found is None:
        loaded = load_pairs(geometry_archive(root, problem), settings["geometry"], settings["pairs"])
        found = {pair.name: pair for pair in loaded}
        _PAIR_CACHE[key] = found
    return found


def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".10g")


def _dat(records: list[dict], template: dict, settings: dict) -> str:
    successes = [row for row in records if row["success"]]
    calls = [int(row["force_calls"]) for row in successes]
    lines = [
        "# counter harness base-calculator evaluations",
        "# surrogate predictions are not force calls",
        "# force_calls are the band total and are not divided by the image count",
        "# the mean is round-half-even over successes; cap, force, or Hessian index other than 1 is a failure",
    ]
    if calls:
        lines.append(f"force_calls {_number(round(statistics.fmean(calls)))}")
        lines.append(f"force_calls_min {min(calls)}")
        lines.append(f"force_calls_max {max(calls)}")
        lines.append(f"force_calls_median {_number(statistics.median(calls))}")
        images = int(settings["n_interior"])
        if images > 0:
            per_image = [count / images for count in calls]
            lines.append(f"force_calls_per_image {_number(round(statistics.fmean(per_image)))}")
            lines.append("call_unit total")
    lines.append(f"successes {len(calls)}")
    lines.append(f"nfailed {len(records) - len(calls)}")
    lines.append(f"n {len(records)}")
    lines.append(f"transition_states {sum(1 for row in records if row['hessian_index'] == 1)}")
    lines.append(f"code {template['code']}")
    lines.append(f"code_version {template['code_version']}")
    lines.append(f"commit {template['commit']}")
    lines.append(f"algorithm {template['algorithm']}")
    lines.append(f"template {template['id']}")
    lines.append(f"tuning {template['tuning']}")
    lines.append("counter harness")
    lines.append(f"images {settings['n_interior']}")
    return "\n".join(lines) + "\n"


def _legacy(records: list[dict]) -> dict:
    successes = [int(row["force_calls"]) for row in records if row["success"]]
    summary = {
        "successes": len(successes),
        "nfailed": len(records) - len(successes),
        "n": len(records),
        "transition_states": sum(1 for row in records if row["hessian_index"] == 1),
    }
    if successes:
        summary["force_calls"] = int(round(statistics.fmean(successes)))
        summary["force_calls_min"] = min(successes)
        summary["force_calls_max"] = max(successes)
        summary["force_calls_median"] = statistics.median(successes)
    return summary


def _annotate(record: dict, problem: dict, template: dict) -> dict:
    record["seed"] = "0"
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def write_outputs(output: Path, records: list[dict], template: dict, settings: dict, problem: dict) -> dict:
    """Write ``benchmark.dat``, the per-start file, and the v2 score."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    starts = output.with_suffix(".starts.jsonl")
    write_starts(starts, records)
    output.write_text(_dat(records, template, settings), encoding="utf-8")
    parameters = dict(settings)
    parameters["counter"] = "harness"
    parameters["seed"] = "0"
    parameters["counting"] = "base calculator calculate; surrogate predictions are not force calls"
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score_v2(records, settings["cap"]),
        "legacy": _legacy(records),
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy = document["legacy"]
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": legacy.get("force_calls"),
        "nfailed": legacy["nfailed"],
        "starts": legacy["n"],
        "successes": legacy["successes"],
        "output": str(output),
    }


def _read_starts(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def output_path(root: Path, problem_id: str, template_id: str) -> Path:
    """Campaign file for one problem and one GPR template."""
    stem = f"gprcalc-{GPR_VERSION}-{problem_id.replace('/', '-')}-{template_id.split('/', 1)[-1]}"
    return Path(root) / "results" / "campaign" / f"{stem}.dat"


def search_named(root: Path, problem_name: str, template_name: str, start: str, seed: str = "0") -> dict:
    """One archived pair. ``seed`` is recorded. The search does not draw random numbers."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    scratch = Path(os.environ.get("OPTBENCH_GPR_WORK", "/tmp/optbench-gprcalc")) / template["id"].replace("/", "-") / start
    record = _execute(_job(pairs[start], settings, str(scratch)))
    record["seed"] = str(seed)
    return _annotate(record, problem, template)


def run(
    root: Path,
    problem_name: str,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Run the pairs in one problem and write the dat, starts, and v2 score."""
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    if template["engine"] != "gprcalc":
        raise ValueError(f"unknown engine {template['engine']}")
    commit = _registry(root)["commit"]
    settings = _settings(template, problem, commit)
    pairs = list(cached_pairs(root, problem, settings).values())
    if limit:
        pairs = pairs[: int(limit)]
    output = Path(output)
    if work is None:
        work = Path("/tmp/optbench-gprcalc") / problem["id"].replace("/", "-") / template["id"].replace("/", "-")
    work.mkdir(parents=True, exist_ok=True)
    scratch_starts = work / "starts.jsonl"
    if resume and scratch_starts.is_file():
        done = {row["start"]: row for row in _read_starts(scratch_starts) if _scored(row)}
    elif resume:
        done = {row["start"]: row for row in _read_starts(output.with_suffix(".starts.jsonl")) if _scored(row)}
    else:
        done = {}
    pending = [pair for pair in pairs if pair.name not in done]
    jobs = [_job(pair, settings, str(work / pair.name)) for pair in pending]
    _threads_one()
    records_by_name = dict(done)

    def _store(record: dict) -> dict:
        _annotate(record, problem, template)
        records_by_name[record["start"]] = record
        ordered = [records_by_name[pair.name] for pair in pairs if pair.name in records_by_name]
        write_starts(scratch_starts, ordered)
        full = len(ordered) == int(settings["pairs"]) and len(pairs) == int(settings["pairs"])
        if full:
            summary = write_outputs(output, ordered, template, settings, problem)
            summary["published"] = True
        else:
            summary = {
                "problem": problem["id"],
                "template": template["id"],
                "force_calls": None,
                "nfailed": sum(1 for row in ordered if not row["success"]),
                "starts": len(ordered),
                "successes": sum(1 for row in ordered if row["success"]),
                "output": "",
                "published": False,
            }
        print(
            f"{record['start']} calls={record['force_calls']} success={record['success']} "
            f"failure={record['failure']} seconds={record.get('search_seconds')} "
            f"published={summary['published']}",
            file=sys.stderr,
            flush=True,
        )
        _discard_resume(work / record["start"])
        return summary

    def _checkpoint(record: dict) -> dict:
        ordered = [records_by_name[pair.name] for pair in pairs if pair.name in records_by_name]
        print(
            f"{record['start']} checkpoint calls={record.get('force_calls')} "
            f"steps={record.get('optimizer_steps')} scored=False",
            file=sys.stderr,
            flush=True,
        )
        return {
            "problem": problem["id"],
            "template": template["id"],
            "force_calls": None,
            "nfailed": sum(1 for row in ordered if not row["success"]),
            "starts": len(ordered),
            "successes": sum(1 for row in ordered if row["success"]),
            "output": "",
            "published": False,
        }

    def _accept(record: dict) -> dict:
        if not _scored(record):
            return _checkpoint(record)
        return _store(record)

    summary = None
    if workers <= 1:
        for job in jobs:
            summary = _accept(_execute(job))
    else:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_execute, job): job["name"] for job in jobs}
            for future in as_completed(futures):
                summary = _accept(future.result())
    if summary is None:
        ordered = [records_by_name[pair.name] for pair in pairs if pair.name in records_by_name]
        if not ordered:
            raise ValueError(f"{template['id']} on {problem['id']} has no starts")
        full = len(ordered) == int(settings["pairs"])
        if full:
            summary = write_outputs(output, ordered, template, settings, problem)
            summary["published"] = True
        else:
            summary = {
                "problem": problem["id"],
                "template": template["id"],
                "force_calls": None,
                "nfailed": sum(1 for row in ordered if not row["success"]),
                "starts": len(ordered),
                "successes": sum(1 for row in ordered if row["success"]),
                "output": "",
                "published": False,
            }
    return summary
