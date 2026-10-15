"""ASE and Sella campaigns with one force-call counter.

A force call is one calculator evaluation. ASE optimizers, Sella, NEB
images, and the dimer all increment that counter. An evaluation that
returns both energy and forces counts once.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from .campaign import load_template, resolve_problem
from .score import summarize_all, summarize_capped, summarize_divided

CONTRIBUTOR = "Rohit Goswami"

ASE_DEFAULTS = {
    "BFGS": {"maxstep": 0.2, "alpha": 70.0},
    "LBFGS": {
        "maxstep": 0.2,
        "memory": 100,
        "damping": 1.0,
        "alpha": 70.0,
        "use_line_search": False,
    },
    "FIRE": {
        "dt": 0.1,
        "maxstep": 0.2,
        "dtmax": 1.0,
        "Nmin": 5,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.1,
        "fa": 0.99,
        "a": 0.1,
        "downhill_check": False,
    },
    "BFGSLineSearch": {
        "maxstep": 0.2,
        "c1": 0.23,
        "c2": 0.46,
        "alpha": 10.0,
        "stpmax": 50.0,
    },
    "GPMin": {"update_prior_strategy": "maximum", "update_hyperparams": False},
    "PreconLBFGS": {
        "maxstep": 0.04,
        "memory": 100,
        "damping": 1.0,
        "alpha": 70.0,
        "precon": "auto",
        "use_armijo": True,
        "c1": 0.23,
        "c2": 0.46,
        "variable_cell": False,
    },
    "PreconFIRE": {
        "dt": 0.1,
        "maxmove": 0.2,
        "dtmax": 1.0,
        "Nmin": 5,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.1,
        "fa": 0.99,
        "a": 0.1,
        "theta": 0.1,
        "precon": "none",
        "use_armijo": True,
        "variable_cell": False,
    },
    "PreconODE12r": {"alpha": 1.0, "precon": "Exp", "rtol": 0.01, "verbose": 0},
}

SELLA_MINIMUM = {
    "order": 0,
    "eta": 1e-4,
    "gamma": 0.1,
    "delta0": 0.1,
    "sigma_inc": 1.15,
    "sigma_dec": 0.90,
    "rho_inc": 1.035,
    "rho_dec": 100.0,
    "method": "qn",
    "eig": False,
    "internal": False,
    "threepoint": False,
}
SELLA_SADDLE = {
    "order": 1,
    "eta": 1e-4,
    "gamma": 0.1,
    "delta0": 0.1,
    "sigma_inc": 1.15,
    "sigma_dec": 0.65,
    "rho_inc": 1.035,
    "rho_dec": 5.0,
    "method": "prfo",
    "eig": True,
    "internal": False,
    "threepoint": False,
}

DIMER_DEFAULTS = {
    "eigenmode_method": "dimer",
    "f_rot_min": 0.1,
    "f_rot_max": 1.0,
    "max_num_rot": 1,
    "trial_angle": 0.7853981633974483,
    "trial_trans_step": 0.001,
    "maximum_translation": 0.1,
    "cg_translation": True,
    "use_central_forces": True,
    "dimer_separation": 0.0001,
    "initial_eigenmode_method": "gauss",
    "extrapolate_forces": False,
    "displacement_method": "gauss",
    "gauss_std": 0.1,
    "order": 1,
}

_SCALAR_ATTRS = (
    "maxstep",
    "alpha",
    "memory",
    "damping",
    "dt",
    "dtmax",
    "Nmin",
    "finc",
    "fdec",
    "astart",
    "fa",
    "a",
    "use_line_search",
    "use_armijo",
    "c1",
    "c2",
    "stpmax",
    "downhill_check",
    "theta",
    "rtol",
    "maxmove",
    "update_hyperparams",
    "update_prior_strategy",
    "verbose",
)


class ForceCap(RuntimeError):
    """The harness force-call cap was reached."""


def entry_id(template: dict) -> str:
    prefix = "ase" if template["engine"] == "ase" else "sella"
    rest = template["id"].split("/", 1)[1]
    return f"{prefix}-{template['code_version']}-{rest}"


def raw_cap(template: dict) -> int:
    cap = int(template["stop"]["max_force_calls"])
    if template.get("reducer") == "per_image":
        return cap * int(template["neb"]["moving_images"])
    return cap


def _plain(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return value
    return str(value)


def format_metric(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def _lookup_stored(root: Path, archive_path: str) -> Path:
    manifest = root / "inputs" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("path") == archive_path:
            return root / row["stored"]
    raise FileNotFoundError(archive_path)


def materialize(root: Path, template: dict, problem: dict, cache: Path) -> Path:
    archive_path = (template.get("input") or {}).get("archive_path")
    if not archive_path:
        archive_path = (problem.get("input") or {}).get("archive_path")
    if not archive_path:
        raise FileNotFoundError(f"{template['id']} has no geometry archive")
    stored = _lookup_stored(root, archive_path)
    if not tarfile.is_tarfile(stored):
        return stored
    dest = cache / stored.parent.name
    marker = dest / ".complete"
    if not marker.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(stored) as packed:
            packed.extractall(dest, filter="data")
        marker.write_text("ok\n", encoding="utf-8")
    return dest


def _prefer_con(paths: list[Path]) -> Path:
    for path in paths:
        if path.suffix == ".con":
            return path
    return paths[0]


def discover(material: Path, template: dict) -> list[tuple[str, list[Path], int | None]]:
    """Return (start name, geometry paths, seed) in a stable order."""
    task = template["task"]
    if task == "dimer":
        path = material if material.is_file() else _prefer_con(sorted(material.rglob("*.con")))
        searches = int((template.get("dimer") or {}).get("searches", 1))
        return [(f"seed-{seed:04d}", [path], seed) for seed in range(searches)]
    if task == "neb":
        return _discover_neb(material, template["neb"])
    select = (template.get("input") or {}).get("select", "con")
    if material.is_file():
        return [(material.name, [material], None)]
    files = [path for path in material.rglob("*") if path.is_file() and path.suffix == ".con"]
    if select == "cluster":
        files = [path for path in files if path.name.startswith("cluster_")]
    elif select == "initial":
        files = [path for path in files if path.name.startswith("initial_")]
    else:
        files = [path for path in files if path.name != "reactant.con"]
    return [(path.name, [path], None) for path in sorted(files, key=_nat_key)]


def _discover_neb(material: Path, neb: dict) -> list[tuple[str, list[Path], int | None]]:
    layout = neb["layout"]
    if layout == "minima":
        rows = []
        directories = sorted(
            (path for path in material.rglob("*") if path.is_dir() and path.name.startswith("min")),
            key=_nat_key,
        )
        for directory in directories:
            left = directory / "a.con"
            right = directory / "b.con"
            if left.is_file() and right.is_file():
                rows.append((directory.name, [left, right], None))
        return rows
    grouped: dict[tuple[str, int | None], list[Path]] = {}
    import re

    pattern = re.compile(r"^(start|end|product)_(\d+)\.(con|xyz)$")
    for path in material.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "reactant.con":
            grouped.setdefault(("reactant", None), []).append(path)
            continue
        match = pattern.match(path.name)
        if match:
            grouped.setdefault((match.group(1), int(match.group(2))), []).append(path)
    if layout == "reactant-products":
        reactant = _prefer_con(grouped[("reactant", None)])
        products = sorted(
            (
                (index, _prefer_con(paths))
                for (role, index), paths in grouped.items()
                if role == "product" and index is not None
            ),
            key=lambda item: item[0],
        )
        return [(f"product_{index}", [reactant, path], None) for index, path in products]
    starts = {
        index: _prefer_con(paths)
        for (role, index), paths in grouped.items()
        if role == "start" and index is not None
    }
    ends = {
        index: _prefer_con(paths)
        for (role, index), paths in grouped.items()
        if role == "end" and index is not None
    }
    indices = sorted(set(starts) & set(ends))
    count = neb.get("count")
    if count is not None:
        indices = [index for index in indices if index < int(count)]
    return [(f"{index:04d}", [starts[index], ends[index]], None) for index in indices]


def _nat_key(path: Path) -> list[int | str]:
    import re

    pieces = re.split(r"(\d+)", path.name)
    return [int(piece) if piece.isdigit() else piece for piece in pieces]


def _read_positions(path: Path) -> list[list[float]]:
    atoms = _read_atoms(path)
    return atoms.positions.tolist()


def _read_xyz(path: Path):
    """Read an xyz file whose symbols need not be chemical elements."""
    from ase import Atoms
    from ase.data import atomic_numbers

    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].split()[0])
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append(parts)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"{path} has {len(rows)} coordinates, expected {count}")
    symbols = []
    positions = []
    for parts in rows:
        symbol = parts[0]
        if symbol not in atomic_numbers:
            symbol = "H"
        symbols.append(symbol)
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return Atoms(symbols=symbols, positions=positions)


def _read_atoms(path: Path):
    from ase.io import read

    if path.suffix == ".xyz":
        try:
            return read(path, format="xyz")
        except Exception:
            return _read_xyz(path)
    try:
        return read(path, format="eon")
    except Exception:
        return read(str(path))


def _apply_pbc(atoms, potential: dict):
    pbc = potential.get("pbc")
    if pbc is not None:
        atoms.set_pbc(list(pbc))
    return atoms


class _Morse:
    """Pt Morse used by the eOn morse_pt and smooth-linear archives."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, D: float, alpha: float, r0: float, rc: float, form: str):
        from ase.calculators.calculator import Calculator

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self):
                super().__init__()
                self.D = D
                self.alpha = alpha
                self.r0 = r0
                self.rc = rc
                self.form = form

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
                from ase.calculators.calculator import all_changes

                changes = all_changes if system_changes is None else system_changes
                super().calculate(atoms, properties, changes)
                energy, forces = _morse_arrays(
                    atoms.positions,
                    atoms.cell.array,
                    atoms.pbc,
                    self.D,
                    self.alpha,
                    self.r0,
                    self.rc,
                    self.form,
                )
                self.results = {"energy": energy, "forces": forces}

        self.calc = _Calc()


def _morse_arrays(positions, cell, pbc, D, alpha, r0, rc, form):
    import numpy as np

    delta = positions[:, None, :] - positions[None, :, :]
    if bool(np.any(pbc)):
        frac = delta @ np.linalg.inv(cell)
        shift = np.round(frac)
        for axis, periodic in enumerate(pbc):
            if not periodic:
                shift[..., axis] = 0.0
        delta = (frac - shift) @ cell
    dist = np.linalg.norm(delta, axis=-1)
    mask = (dist > 1e-12) & (dist < rc)
    x = np.exp(-alpha * (dist - r0))
    energy_pair = D * (x * x - 2.0 * x)
    slope = 2.0 * alpha * D * x * (1.0 - x)
    x_cut = np.exp(-alpha * (rc - r0))
    energy_cut = D * (x_cut * x_cut - 2.0 * x_cut)
    slope_cut = 2.0 * alpha * D * x_cut * (1.0 - x_cut)
    if form == "shift":
        energy_pair = energy_pair - energy_cut
    elif form == "smooth":
        linear = -slope_cut
        energy_pair = energy_pair + linear * dist - energy_cut - linear * rc
        slope = slope + linear
    else:
        raise ValueError(form)
    energy_pair = np.where(mask, energy_pair, 0.0)
    slope = np.where(mask, slope, 0.0)
    energy = float(0.5 * energy_pair.sum())
    hat = np.zeros_like(delta)
    np.divide(delta, dist[..., None], out=hat, where=mask[..., None])
    forces = -np.sum(slope[..., None] * hat, axis=1)
    return energy, forces


class _Counter:
    def __init__(self, inner, bucket: list[int]):
        from ase.calculators.calculator import Calculator

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self):
                super().__init__()
                self.inner = inner
                self.bucket = bucket

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
                from ase.calculators.calculator import all_changes

                changes = all_changes if system_changes is None else system_changes
                super().calculate(atoms, properties, changes)
                if self.bucket[0] + 1 > self.bucket[1]:
                    raise ForceCap()
                self.inner.calculate(atoms, ("energy", "forces"), changes)
                self.bucket[0] += 1
                self.results = dict(self.inner.results)

        self.calc = _Calc()


def _inner_calculator(potential: dict):
    if potential["kind"] == "lennard-jones":
        from ase.calculators.lj import LennardJones

        return LennardJones(
            epsilon=float(potential["epsilon"]),
            sigma=float(potential["sigma"]),
            rc=float(potential["rc"]),
        )
    if potential["kind"] == "morse":
        return _Morse(
            float(potential["D"]),
            float(potential["alpha"]),
            float(potential["r0"]),
            float(potential["rc"]),
            str(potential["form"]),
        ).calc
    raise ValueError(potential["kind"])


def _attach(atoms, potential: dict, bucket: list[int]):
    _apply_pbc(atoms, potential)
    atoms.calc = _Counter(_inner_calculator(potential), bucket).calc
    return atoms


def _cached_l2(calc, atoms=None) -> float | None:
    """Euclidean norm of the forces the optimizer is allowed to move.

    Fixed atoms keep a reaction force in the raw calculator result. The stop
    uses the constrained forces, which is what ``atoms.get_forces`` returns
    from the cached evaluation.
    """
    import numpy as np

    if atoms is not None and getattr(atoms, "calc", None) is not None:
        results = getattr(atoms.calc, "results", None) or {}
        if "forces" in results:
            return float(np.linalg.norm(np.asarray(atoms.get_forces(), dtype=float)))
    results = getattr(calc, "results", None) or {}
    forces = results.get("forces")
    if forces is None:
        inner = getattr(calc, "inner", None)
        if inner is not None:
            forces = getattr(inner, "results", {}).get("forces")
    if forces is None:
        return None
    return float(np.linalg.norm(np.asarray(forces, dtype=float)))


def _build_optimizer(spec: dict, target):
    from ase.optimize import BFGS, BFGSLineSearch, FIRE, GPMin, LBFGS
    from ase.optimize.precon import Exp, PreconFIRE, PreconLBFGS, PreconODE12r

    classes = {
        "BFGS": BFGS,
        "LBFGS": LBFGS,
        "FIRE": FIRE,
        "BFGSLineSearch": BFGSLineSearch,
        "GPMin": GPMin,
        "PreconLBFGS": PreconLBFGS,
        "PreconFIRE": PreconFIRE,
        "PreconODE12r": PreconODE12r,
    }
    kwargs = dict(spec.get("parameters") or {})
    precon = kwargs.pop("precon", None)
    if isinstance(precon, dict):
        kwargs["precon"] = Exp(A=float(precon.get("A", 3.0)))
    elif precon is not None:
        kwargs["precon"] = precon
    if spec["name"] == "PreconFIRE" and "maxstep" in kwargs:
        kwargs["maxmove"] = kwargs.pop("maxstep")
    cls = classes[spec["name"]]
    try:
        return cls(target, logfile=None, **kwargs)
    except TypeError:
        return cls(target, **kwargs)


def _describe_precon(precon) -> str:
    if precon is None:
        return "none"
    if isinstance(precon, str):
        return precon
    text = type(precon).__name__
    amplitude = getattr(precon, "A", None)
    if isinstance(amplitude, (int, float)):
        text += f" A={float(amplitude)}"
    return text


def _requested_parameters(spec: dict) -> dict:
    requested = dict(spec.get("parameters") or {})
    if spec.get("name") == "PreconFIRE" and "maxstep" in requested:
        requested["maxmove"] = requested.pop("maxstep")
    return requested


def _snapshot_optimizer(opt, spec: dict) -> dict:
    name = spec["name"]
    params = {key: _plain(value) for key, value in ASE_DEFAULTS.get(name, {}).items()}
    requested = _requested_parameters(spec)
    precon = requested.pop("precon", None)
    for key, value in requested.items():
        params[key] = _plain(value)
    if precon is not None:
        params["precon"] = _describe_precon(precon) if not isinstance(precon, dict) else (
            f"{precon.get('name', 'Exp')} A={float(precon.get('A', 3.0))}"
        )
    for key in _SCALAR_ATTRS:
        if hasattr(opt, key):
            value = getattr(opt, key)
            if isinstance(value, (int, float, str, bool)):
                params[key] = _plain(value)
    if hasattr(opt, "precon"):
        params["precon"] = _describe_precon(opt.precon)
    return params


def _snapshot_sella(opt, order: int, requested: dict) -> dict:
    params = dict(SELLA_MINIMUM if order == 0 else SELLA_SADDLE)
    params.update(requested)
    params["order"] = int(getattr(opt, "ord", order))
    for key in ("eta", "sigma_inc", "sigma_dec", "rho_inc", "rho_dec", "method", "eig"):
        if hasattr(opt, key):
            params[key] = _plain(getattr(opt, key))
    if hasattr(opt, "delta"):
        params["delta0"] = float(opt.delta)
    diag = getattr(opt, "diagkwargs", {}) or {}
    if "gamma" in diag:
        params["gamma"] = float(diag["gamma"])
    if "threepoint" in diag:
        params["threepoint"] = bool(diag["threepoint"])
    params["internal"] = bool(getattr(opt, "user_internal", False))
    return params


def _drive_l2(opt, calc, bucket: list[int], tol: float, max_steps: int, atoms=None) -> tuple[bool, float | None]:
    from ase.optimize.optimize import Optimizer

    if type(opt).step is Optimizer.step:
        return _drive_l2_scipy(opt, calc, bucket, tol, max_steps, atoms)
    norm = None
    try:
        for _converged in opt.irun(fmax=0.0, steps=max_steps):
            norm = _cached_l2(calc, atoms)
            if bucket[0] >= bucket[1]:
                raise ForceCap()
            if norm is not None and norm <= tol:
                return False, norm
    except ForceCap:
        return True, _cached_l2(calc, atoms)
    norm = _cached_l2(calc, atoms)
    failed = norm is None or norm > tol or bucket[0] >= bucket[1]
    return failed, norm


def _drive_l2_scipy(opt, calc, bucket: list[int], tol: float, max_steps: int, atoms=None) -> tuple[bool, float | None]:
    def gradient_converged(_gradient) -> bool:
        if bucket[0] >= bucket[1]:
            raise ForceCap()
        norm = _cached_l2(calc, atoms)
        return norm is not None and norm <= tol

    opt.gradient_converged = gradient_converged
    try:
        opt.run(fmax=tol, steps=max_steps)
    except ForceCap:
        return True, _cached_l2(calc, atoms)
    norm = _cached_l2(calc, atoms)
    return norm is None or norm > tol or bucket[0] >= bucket[1], norm


def _run_minimize(payload: dict, bucket: list[int]) -> dict:
    atoms = _attach(_read_atoms(Path(payload["paths"][0])), payload["potential"], bucket)
    opt = _build_optimizer(payload["optimizer"], atoms)
    row = {"failed": True, "force_norm": None}
    if payload.get("snapshot"):
        row["parameters"] = _snapshot_optimizer(opt, payload["optimizer"])
    try:
        failed, norm = _drive_l2(opt, atoms.calc, bucket, payload["tol"], payload["max_steps"], atoms)
    except ForceCap:
        row["error"] = "force-call cap"
        row["force_norm"] = _cached_l2(atoms.calc, atoms)
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return row
    row["failed"] = failed
    row["force_norm"] = norm
    return row


def _reactant_mode(atoms, reference) -> object:
    """Unit displacement from the reactant, in the free-atom coordinates."""
    import numpy as np
    from ase.constraints import FixAtoms

    if reference is None:
        return None
    reference = np.asarray(reference, dtype=float)
    if reference.shape != atoms.positions.shape:
        return None
    displacement = atoms.positions - reference
    free = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            free[np.asarray(constraint.index, dtype=int)] = False
    vector = displacement[free].ravel()
    scale = float(np.linalg.norm(vector))
    if scale == 0.0:
        return None
    return vector / scale


def _run_sella(payload: dict, bucket: list[int]) -> dict:
    from sella import Sella

    atoms = _attach(_read_atoms(Path(payload["paths"][0])), payload["potential"], bucket)
    requested = dict(payload["optimizer"].get("parameters") or {})
    order = int(requested.pop("order", payload.get("sella", {}).get("order", 1)))
    v0 = _reactant_mode(atoms, payload.get("reactant_positions"))
    kwargs = {key: value for key, value in requested.items() if value is not None}
    opt = Sella(atoms, logfile=None, order=order, v0=v0, **kwargs)
    row = {"failed": True, "force_norm": None}
    if payload.get("snapshot"):
        snapped = _snapshot_sella(opt, order, requested)
        if payload.get("sella", {}).get("v0") not in (None, "none"):
            snapped["v0"] = payload["sella"]["v0"]
        row["parameters"] = snapped
    try:
        failed, norm = _drive_l2(opt, atoms.calc, bucket, payload["tol"], payload["max_steps"], atoms)
    except ForceCap:
        row["error"] = "force-call cap"
        row["force_norm"] = _cached_l2(atoms.calc, atoms)
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return row
    row["failed"] = failed
    row["force_norm"] = norm
    library = getattr(opt, "pes", None)
    if library is not None and hasattr(library, "neval"):
        row["library_neval"] = int(library.neval)
    return row


def _run_neb(payload: dict, bucket: list[int]) -> dict:
    from ase.mep import NEB

    neb_spec = payload["neb"]
    moving = int(neb_spec["moving_images"])
    left = _apply_pbc(_read_atoms(Path(payload["paths"][0])), payload["potential"])
    right = _apply_pbc(_read_atoms(Path(payload["paths"][1])), payload["potential"])
    images = [left]
    for _ in range(moving):
        images.append(left.copy())
    images.append(right)
    band = NEB(
        images,
        k=float(neb_spec["spring"]),
        climb=bool(neb_spec["climb"]),
        method=str(neb_spec.get("method", "improvedtangent")),
    )
    band.interpolate()
    for image in images:
        _attach(image, payload["potential"], bucket)
    opt = _build_optimizer(payload["optimizer"], band)
    converged = False
    try:
        for converged in opt.irun(fmax=payload["tol"], steps=payload["max_steps"]):
            if bucket[0] >= bucket[1]:
                converged = False
                raise ForceCap()
    except ForceCap:
        converged = False
    residuals = [float(value) for value in getattr(band, "residuals", []) or []]
    failed = (not converged) or bucket[0] >= bucket[1]
    row = {"failed": failed, "force_norm": max(residuals) if residuals else None}
    if payload.get("snapshot"):
        params = _snapshot_optimizer(opt, payload["optimizer"])
        params["spring"] = float(neb_spec["spring"])
        params["climb"] = bool(neb_spec["climb"])
        params["moving_images"] = moving
        params["method"] = str(neb_spec.get("method", "improvedtangent"))
        row["parameters"] = params
    return row


def _free_mask(atoms) -> list[bool]:
    """True where the dimer may displace an atom. Fixed atoms stay put."""
    import numpy as np
    from ase.constraints import FixAtoms

    free = np.ones(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            free[np.asarray(constraint.index, dtype=int)] = False
    return [bool(value) for value in free]


def _run_dimer(payload: dict, bucket: list[int]) -> dict:
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

    atoms = _attach(_read_atoms(Path(payload["paths"][0])), payload["potential"], bucket)
    mask = _free_mask(atoms)
    control = DimerControl(
        logfile=os.devnull,
        mask=mask,
        **dict(payload["optimizer"].get("parameters") or {}),
    )
    mode_atoms = MinModeAtoms(atoms, control, random_seed=int(payload["seed"]))
    opt = MinModeTranslate(mode_atoms, logfile=None)
    norm = None
    curvature = None
    try:
        # ASE stops on the largest force component. The problem stop is the
        # Euclidean norm, so the fmax gate stays open and this loop applies it.
        for _ase_converged in opt.irun(fmax=0.0, steps=payload["max_steps"]):
            if bucket[0] >= bucket[1]:
                raise ForceCap()
            norm = _cached_l2(atoms.calc, atoms)
            curvature = float(mode_atoms.get_curvature())
            if norm is not None and norm <= payload["tol"] and curvature < 0.0:
                break
    except ForceCap:
        pass
    if curvature is None:
        curvature = float(mode_atoms.get_curvature())
    if norm is None:
        norm = _cached_l2(atoms.calc, atoms)
    saddle = curvature < 0.0 and norm is not None and norm <= payload["tol"]
    failed = (not saddle) or bucket[0] >= bucket[1]
    row = {
        "failed": failed,
        "force_norm": norm,
        "curvature": curvature,
        "saddle": saddle,
    }
    if payload.get("snapshot"):
        params = dict(DIMER_DEFAULTS)
        params.update(payload["optimizer"].get("parameters") or {})
        params["mask"] = "free" if not all(mask) else "all"
        row["parameters"] = params
    return row


def _run_start(payload: dict) -> dict:
    import logging
    import warnings

    warnings.filterwarnings("ignore", message="The system is likely too small")
    logging.getLogger("sella").setLevel(logging.ERROR)
    bucket = [0, int(payload["raw_cap"])]
    row = {
        "start": payload["start"],
        "force_calls": 0,
        "failed": True,
        "force_norm": None,
    }
    try:
        task = payload["task"]
        if task == "minimize":
            extra = _run_minimize(payload, bucket)
        elif task == "sella":
            extra = _run_sella(payload, bucket)
        elif task == "neb":
            extra = _run_neb(payload, bucket)
        elif task == "dimer":
            extra = _run_dimer(payload, bucket)
        else:
            raise ValueError(task)
        row.update(extra)
    except ForceCap:
        row["failed"] = True
        row["error"] = "force-call cap"
    except Exception as exc:
        row["failed"] = True
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    row["force_calls"] = int(bucket[0])
    if row["force_calls"] >= bucket[1]:
        row["failed"] = True
    return row


def _load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["start"])] = row
    return rows


def _summarize(rows: list[dict], template: dict) -> dict:
    cap = int(template["stop"]["max_force_calls"])
    reducer = template["reducer"]
    if reducer == "capped":
        counts = [
            cap if row["failed"] or int(row["force_calls"]) >= cap else int(row["force_calls"])
            for row in rows
        ]
        summary = summarize_capped(counts, cap)
        ok = [count for count in counts if count < cap]
        summary["force_calls_median"] = float(statistics.median(ok)) if ok else None
        return summary
    if reducer == "all":
        return summarize_all(
            [int(row["force_calls"]) for row in rows],
            sum(1 for row in rows if row["failed"]),
        )
    if reducer == "per_image":
        return summarize_divided(
            [int(row["force_calls"]) for row in rows],
            [bool(row["failed"]) for row in rows],
            images=int(template["neb"]["moving_images"]),
            cap=cap,
        )
    raise ValueError(reducer)


def _baseline_parameters(template: dict) -> dict:
    task = template["task"]
    requested = dict((template.get("optimizer") or {}).get("parameters") or {})
    if task == "sella":
        order = int(requested.get("order", template.get("sella", {}).get("order", 1)))
        params = dict(SELLA_MINIMUM if order == 0 else SELLA_SADDLE)
        params.update(requested)
        v0 = (template.get("sella") or {}).get("v0")
        if v0 not in (None, "none"):
            params["v0"] = v0
        return params
    if task == "dimer":
        params = dict(DIMER_DEFAULTS)
        params.update(requested)
        return params
    name = template["optimizer"]["name"]
    params = dict(ASE_DEFAULTS.get(name, {}))
    if name == "PreconFIRE" and "maxstep" in requested:
        requested["maxmove"] = requested.pop("maxstep")
    precon = requested.pop("precon", None)
    params.update(requested)
    if isinstance(precon, dict):
        params["precon"] = f"{precon.get('name', 'Exp')} A={float(precon.get('A', 3.0))}"
    if task == "neb":
        neb = template["neb"]
        params["spring"] = float(neb["spring"])
        params["climb"] = bool(neb["climb"])
        params["moving_images"] = int(neb["moving_images"])
        params["method"] = str(neb.get("method", "improvedtangent"))
    return params


def _write_dat(
    path: Path,
    template: dict,
    summary: dict,
    parameters: dict,
    *,
    starts: int,
    partial: bool,
) -> None:
    method = {
        "profile": template["profile"],
        "parameter_source": template.get("parameter_source", ""),
        "counter": "harness calculator force evaluations",
        "reducer": template["reducer"],
        "parameters": parameters,
        "potential": template.get("potential") or {},
        "stop": template.get("stop") or {},
    }
    fields = [
        ("force_calls", summary.get("force_calls")),
        ("force_calls_min", summary.get("force_calls_min")),
        ("force_calls_median", summary.get("force_calls_median")),
        ("force_calls_max", summary.get("force_calls_max")),
        ("nfailed", summary.get("nfailed")),
        ("force_calls_raw", summary.get("force_calls_raw")),
        ("starts", starts),
        ("code", template["code"]),
        ("code_version", template["code_version"]),
        ("commit", template.get("commit", "")),
        ("algorithm", template["algorithm"]),
        ("profile", template["profile"]),
        ("template", template["id"]),
        ("problem", template["problem"]),
        ("counter", "harness"),
        ("reducer", template["reducer"]),
        ("parameter_source", template.get("parameter_source", "")),
        ("method", json.dumps(_jsonable(method), sort_keys=True, separators=(",", ":"))),
        ("note", template.get("note", "")),
        ("host", socket.gethostname()),
        ("date", date.today().isoformat()),
        ("contributor", CONTRIBUTOR),
    ]
    if partial:
        fields.append(("partial", "true"))
    if template.get("ranked") is False:
        fields.append(("ranked", "false"))
    if template["problem"] == "double-ended-saddle-search/lj38_2ts_separation":
        kept = starts - int(summary.get("nfailed") or 0)
        fields.append(("transition_states", kept))
    lines = []
    for key, value in fields:
        text = format_metric(value)
        if text is None or text == "":
            continue
        lines.append(f"{key} {text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_template(
    root: Path,
    template_name: str,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
    resume: bool = False,
    cache: Path | None = None,
) -> dict:
    """Run one ASE or Sella template and write benchmark.dat plus per-start rows."""
    template = load_template(root, template_name)
    if template["engine"] not in {"ase", "sella"}:
        raise ValueError(f"{template_name} is not an ASE or Sella template")
    problem = resolve_problem(root, template["problem"])
    cache = cache or Path("/tmp/optbench-geometry")
    material = materialize(root, template, problem, cache)
    jobs = discover(material, template)
    if limit > 0:
        jobs = jobs[:limit]
    if not jobs:
        raise FileNotFoundError(f"{template['id']} found no geometries")
    reactant_positions = None
    if (template.get("sella") or {}).get("v0") == "reactant":
        reactant = next(path for path in material.rglob("reactant.con") if path.is_file())
        reactant_positions = _read_positions(reactant)
    starts_path = output.with_name(output.stem + ".starts.jsonl")
    done = _load_rows(starts_path) if resume else {}
    if not resume and starts_path.exists():
        starts_path.unlink()
        done = {}
    cap = raw_cap(template)
    tol = float(template["stop"].get("force_norm", template["stop"].get("gradient_norm")))
    payloads = []
    for name, paths, seed in jobs:
        if name in done:
            continue
        payloads.append(
            {
                "start": name,
                "task": template["task"],
                "paths": [str(path) for path in paths],
                "potential": template["potential"],
                "optimizer": template["optimizer"],
                "neb": template.get("neb"),
                "sella": template.get("sella"),
                "reactant_positions": reactant_positions,
                "seed": 0 if seed is None else seed,
                "tol": tol,
                "max_steps": int(template["stop"]["max_steps"]),
                "raw_cap": cap,
                "snapshot": False,
            }
        )
    if payloads:
        payloads[0]["snapshot"] = True
    fresh: list[dict] = []
    if payloads:
        starts_path.parent.mkdir(parents=True, exist_ok=True)
        with starts_path.open("a", encoding="utf-8") as handle:
            if workers <= 1:
                iterator = (_run_start(payload) for payload in payloads)
                total = len(payloads)
                for index, row in enumerate(iterator, 1):
                    fresh.append(row)
                    handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                    handle.flush()
                    if index == 1 or index == total or index % 25 == 0:
                        print(f"{template['id']} {index}/{total}", flush=True)
            else:
                import multiprocessing as mp

                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    futures = [pool.submit(_run_start, payload) for payload in payloads]
                    total = len(futures)
                    for index, future in enumerate(as_completed(futures), 1):
                        row = future.result()
                        fresh.append(row)
                        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        handle.flush()
                        if index == 1 or index == total or index % 25 == 0:
                            print(f"{template['id']} {index}/{total}", flush=True)
    by_start = dict(done)
    for row in fresh:
        by_start[row["start"]] = row
    ordered = [by_start[name] for name, _paths, _seed in jobs]
    starts_path.write_text(
        "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    errors = [row for row in ordered if row.get("error")]
    for row in errors[:5]:
        print(f"error {template['id']} {row['start']}: {row['error']}", flush=True)
    summary = _summarize(ordered, template)
    parameters = _baseline_parameters(template)
    for row in ordered:
        if row.get("parameters"):
            parameters = row["parameters"]
            break
    _write_dat(
        output,
        template,
        summary,
        parameters,
        starts=len(ordered),
        partial=limit > 0,
    )
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = template["problem"]
    summary["entry"] = entry_id(template)
    summary["output"] = str(output)
    return summary


def run_many(
    root: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    limit: int = 0,
    match: str = "",
    ranked_only: bool = True,
    resume: bool = False,
) -> list[dict]:
    """Run every ranked ASE and Sella template, light problems first."""
    from .campaign import list_templates

    found = []
    for template in list_templates(root):
        if template.get("engine") not in {"ase", "sella"}:
            continue
        if ranked_only and not template.get("ranked", True):
            continue
        if match and match not in template["id"]:
            continue
        found.append(template)

    def weight(template: dict) -> tuple:
        ident = template["id"]
        score = 0
        if any(token in ident for token in ("gpmin", "precon", "de-pt", "neb-pt", "pt-island")):
            score += 2
        if "lj38" in ident:
            score += 1
        return (score, ident)

    summaries = []
    cache = Path("/tmp/optbench-geometry")
    for template in sorted(found, key=weight):
        output = output_dir / f"{entry_id(template)}.dat"
        print(f"run {template['id']} -> {output.name}", flush=True)
        summaries.append(
            run_template(
                root,
                template["id"],
                output=output,
                workers=workers,
                limit=limit,
                resume=resume,
                cache=cache,
            )
        )
    return summaries


def write_problem_record(root: Path, campaign: Path, dest: Path, problem: str, status: str) -> None:
    """Write one optbench.result/v1 file from campaign dat files for ``problem``."""
    from .dat import parse_lenient

    runs = []
    for path in sorted(campaign.glob("*.dat")):
        fields, _duplicates = parse_lenient(path.read_text(encoding="utf-8"))
        if fields.get("problem") != problem or fields.get("partial") == "true":
            continue
        metrics = {}
        for key in (
            "force_calls",
            "force_calls_min",
            "force_calls_max",
            "force_calls_median",
            "force_calls_raw",
            "nfailed",
        ):
            if key not in fields:
                continue
            text = fields[key]
            metrics[key] = float(text) if "." in text else int(text)
        method = json.loads(fields["method"]) if fields.get("method") else {}
        runs.append(
            {
                "entry": path.stem,
                "code": fields.get("code", ""),
                "algorithm": fields.get("algorithm", ""),
                "code_version": fields.get("code_version", ""),
                "commit": fields.get("commit", ""),
                "present_source": True,
                "starts": int(fields.get("starts", "0")),
                "nfailed": int(fields.get("nfailed", "0")),
                "method": method,
                "metrics": metrics,
                "starts_file": str(path.with_name(path.stem + ".starts.jsonl").relative_to(root)),
                "notes": fields.get("note", ""),
            }
        )
    record = {
        "schema": "optbench.result/v1",
        "status": status,
        "problem": problem,
        "submitter": CONTRIBUTOR,
        "date": date.today().isoformat(),
        "host": socket.gethostname(),
        "runs": runs,
    }
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
