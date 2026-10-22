"""pARTn through ``libartn.so`` and ``next_displ``.

The plugin and this standalone loop are the same library. Each step
evaluates the harness potential, passes the force (the negative
gradient), and applies the returned displacement. ``nevalf`` is recorded
and is not the score. When ``has_sad`` is set, the scored geometry is
``tau_sad`` and the loop stops. The displacement from that call is the
push off the saddle. With ``lpush_final`` the later position is a minimum.
The product of a pair chooses the initial push.
The search does not build a path to that product.

``engine_units`` is ``lammps/metal`` on Morse platinum and ``lammps/lj``
on Lennard-Jones, so a push length is an angstrom or a sigma. k-ART is
not this library.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from ctypes import POINTER, byref, c_char_p
from pathlib import Path

import numpy as np

from optbench.connection import ProcessCatalog, connected_endpoints, link_tolerance, per_atom_max
from optbench.endpoints import STATUS_STORED, EndpointArchive, stamp_endpoint, stored_cell
from optbench.eon_displace import draw_search_start, problem_pbc

from .common import (
    cached_pairs,
    initial_push,
    judge_mode,
    judge_saddle,
    lattice,
    pair_seed,
    prepared,
    run_jobs,
    write_campaign,
)
from .geometries import load_starts, point_settings

ENGINE = "partn"
CODE = "artn"
_SEED_BASE = 20261015


def library_path() -> Path:
    env = os.environ.get("ARTN_LIB")
    if env:
        return Path(env)
    return Path("/tmp/src/artn/lib/libartn.so")


def interface_dir() -> Path:
    env = os.environ.get("ARTN_INTERFACE")
    if env:
        return Path(env)
    return Path("/tmp/src/artn/interface")


def available() -> str | None:
    library = library_path()
    if not library.is_file():
        return f"libartn.so is not at {library}. Set ARTN_LIB to the built library."
    interface = interface_dir() / "pypARTn.py"
    if not interface.is_file():
        return f"pypARTn.py is not at {interface}. Set ARTN_INTERFACE to the interface directory."
    return None


def _units(kind: str) -> str:
    if kind == "morse_pt":
        return "lammps/metal"
    if kind == "lj":
        return "lammps/lj"
    raise ValueError(f"{kind} has no pARTn engine_units")


def _section(template: dict) -> dict:
    section = dict(template.get("partn") or {})
    section.setdefault("push_step_size", 0.1)
    section.setdefault("eigen_step_size", 0.15)
    section.setdefault("ninit", 3)
    section.setdefault("lanczos_max_size", 16)
    section.setdefault("lanczos_disp", None)
    section.setdefault("lanczos_min_size", None)
    section.setdefault("lanczos_eval_conv_thr", None)
    section.setdefault("etot_diff_limit", None)
    section.setdefault("lpush_final", False)
    section.setdefault("converge_property", "norm")
    section.setdefault("nsmooth", None)
    section.setdefault("nnewchance", None)
    section.setdefault("eigval_thr", None)
    section.setdefault("push_over", None)
    section.setdefault("nperp_limitation", None)
    section.setdefault("fire_dt_max_f", None)
    return section


def _knobs(section: dict) -> dict:
    """Documented pARTn inputs. Absent keys keep the library default."""
    return {
        "push_step_size": float(section["push_step_size"]),
        "eigen_step_size": float(section["eigen_step_size"]),
        "ninit": int(section["ninit"]),
        "lanczos_max_size": int(section["lanczos_max_size"]),
        "lanczos_disp": section.get("lanczos_disp"),
        "lanczos_min_size": section.get("lanczos_min_size"),
        "lanczos_eval_conv_thr": section.get("lanczos_eval_conv_thr"),
        "etot_diff_limit": section.get("etot_diff_limit"),
        "lpush_final": bool(section["lpush_final"]),
        "converge_property": str(section["converge_property"]),
        "nsmooth": section.get("nsmooth"),
        "nnewchance": section.get("nnewchance"),
        "eigval_thr": section.get("eigval_thr"),
        "push_over": section.get("push_over"),
        "nperp_limitation": section.get("nperp_limitation"),
        "fire_dt_max_f": section.get("fire_dt_max_f"),
    }


def _job(pair, settings: dict, template: dict, seed: str) -> dict:
    section = _section(template)
    label = f"{seed}:{pair.name}"
    return {
        "name": pair.name,
        "reactant": np.asarray(pair.reactant, dtype=np.float64).tolist(),
        "product": np.asarray(pair.product, dtype=np.float64).tolist(),
        "free": np.asarray(pair.free, dtype=bool).tolist(),
        "box": None if pair.box is None else np.asarray(pair.box, dtype=np.float64).reshape(-1).tolist(),
        "kind": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
        **_knobs(section),
        "fire_dt": _fire_dt(section, settings["potential"]),
        "task": "saddle",
        "reference": None,
        "overlap": 0.99,
        "seed": str(seed),
        "zseed": pair_seed(label, _SEED_BASE),
        "shlib": str(library_path()),
        "interface": str(interface_dir()),
        "commit": template["commit"],
    }


def _fire_dt(section: dict, kind: str) -> float | None:
    """FIRE ``dt_init`` in engine units.

    The library default is 20, which is an atomic unit. On ``lammps/lj``
    that step is large enough to leave the cluster. Lennard-Jones uses
    0.05 unless the template sets ``fire_dt_init``. Morse keeps the default.
    """
    if "fire_dt_init" in section and section["fire_dt_init"] is not None:
        return float(section["fire_dt_init"])
    if kind == "lj":
        return 0.05
    return None


def _mode_overlap(art, reference, free: np.ndarray) -> float | None:
    """Absolute overlap of the pARTn eigenvector with the free-atom reference."""
    if reference is None:
        return None
    try:
        raw = art.get_runparam("eigenvec")
    except Exception:
        return None
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    nat = int(len(free))
    if values.size != nat * 3:
        return None
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    mask = np.asarray(free, dtype=bool)
    best = 0.0
    for shaped in (values.reshape(nat, 3), values.reshape(3, nat).T):
        piece = shaped[mask].reshape(-1)
        scale = float(np.linalg.norm(piece))
        if scale < 1e-14 or piece.size != target.size:
            continue
        best = max(best, abs(float(np.dot(piece / scale, target / np.linalg.norm(target)))))
    return best


def _gitinfo(art) -> str:
    """Read ``artn_gitinfo``. ``pypARTn.gitinfo`` passes the wrong pointer depth."""
    library = art.lib
    library.artn_gitinfo.restype = None
    library.artn_gitinfo.argtypes = [POINTER(c_char_p)]
    holder = c_char_p()
    library.artn_gitinfo(byref(holder))
    if not holder.value:
        return ""
    return holder.value.decode()


def _set(art, name: str, value) -> None:
    try:
        art.set(name, value)
    except Exception as exc:
        raise RuntimeError(f"pARTn set {name} failed: {exc}") from exc


def _atom_rows(raw, nat: int) -> np.ndarray:
    """``(nat, 3)`` positions. ``get_data`` already reverses a Fortran ``(3, nat)`` size."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape == (nat, 3):
        return np.array(arr, copy=True)
    if arr.shape == (3, nat):
        return np.array(arr.T, copy=True)
    raise RuntimeError(f"pARTn coordinate array has shape {arr.shape}, expected {(nat, 3)}")


def _stored_saddle(art, nat: int, free: np.ndarray, reactant: np.ndarray, kind: str, box, counter):
    """Energy and gradient at ``tau_sad``. The search already paid for that geometry."""
    from optbench.pes import evaluate

    if not bool(art.extract("has_sad")):
        return None, None, None
    tau = _atom_rows(art.extract("tau_sad"), nat)
    if not np.all(np.isfinite(tau)):
        raise RuntimeError("pARTn tau_sad is non-finite")
    tau[~free] = reactant[~free]
    with counter.verification():
        energy, gradient = evaluate(kind, tau, box)
    return tau, float(energy), np.asarray(gradient, dtype=np.float64)


def _execute(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    interface = job["interface"]
    if interface not in sys.path:
        sys.path.insert(0, interface)
    import pypARTn

    from optbench.counter import ForceCallCap, HarnessCounter
    from optbench.pes import evaluate

    reactant = np.asarray(job["reactant"], dtype=np.float64)
    product = None if job.get("product") is None else np.asarray(job["product"], dtype=np.float64)
    free = np.asarray(job["free"], dtype=bool)
    task = str(job.get("task") or "saddle")
    kind = job["kind"]
    cap = int(job["cap"])
    box = None if job["box"] is None else np.asarray(job["box"], dtype=np.float64)
    counter = HarnessCounter(cache=True, cap=cap)
    pos = reactant.copy()
    energy = None
    gradient = None
    error = ""
    lconv = False
    overlap = 0.0
    max_step = 0.0
    has_sad = None
    has_error = None
    artn_nevalf = None
    art = None
    saddle_pos = None
    saddle_energy = None
    saddle_gradient = None
    previous = Path.cwd()
    try:
        version = None
        with tempfile.TemporaryDirectory(prefix="partn-") as folder:
            os.chdir(folder)
            art = pypARTn.artn(engine="other", shlib=job["shlib"])
            version = art.version()
            if version != (1, 1, 0):
                raise RuntimeError(f"libartn version is {version}, the pin is 1.1.0")
            gitinfo = _gitinfo(art)
            token = str(job["commit"])[:7]
            if token not in gitinfo:
                raise RuntimeError(f"libartn gitinfo {gitinfo!r} does not contain {token}")
            nat = int(pos.shape[0])
            _set(art, "engine_units", _units(kind))
            _set(art, "verbose", 0)
            _set(art, "forc_thr", float(job["force_norm"]))
            _set(art, "converge_property", job["converge_property"])
            _set(art, "ninit", int(job["ninit"]))
            _set(art, "push_step_size", float(job["push_step_size"]))
            _set(art, "eigen_step_size", float(job["eigen_step_size"]))
            _set(art, "lanczos_max_size", int(job["lanczos_max_size"]))
            if job.get("lanczos_disp") is not None:
                _set(art, "lanczos_disp", float(job["lanczos_disp"]))
            if job.get("etot_diff_limit") is not None:
                _set(art, "etot_diff_limit", float(job["etot_diff_limit"]))
            _set(art, "lpush_final", bool(job["lpush_final"]))
            _set(art, "struc_format_out", "none")
            _set(art, "nevalf_max", int(cap))
            _set(art, "zseed", int(job["zseed"]))
            _apply_documented(art, job)
            if job.get("fire_dt") is not None:
                art.fire_set("dt_init", float(job["fire_dt"]))
            if job.get("fire_dt_max_f") is not None:
                art.fire_set("dt_max_f", float(job["fire_dt_max_f"]))
            if job.get("push_init") is not None:
                push = np.asarray(job["push_init"], dtype=np.float64)
            elif product is None:
                push = np.zeros_like(reactant)
                push[int(np.flatnonzero(free)[0]), 0] = float(job["push_step_size"])
            else:
                push = initial_push(reactant, product, free, float(job["push_step_size"]))
            _set(art, "push_init", np.ascontiguousarray(push, dtype=np.float64))
            typ = np.ones(nat, dtype=np.int32)
            if_pos = np.ones((nat, 3), dtype=np.int32)
            if_pos[~free] = 0
            cell = np.ascontiguousarray(lattice(box, pos), dtype=np.float64)
            lanczos_code = int(art.get_runparam("LANC")) if task == "mode" else None
            stalled = 0
            for _step in range(cap + 2):
                if counter.calls >= cap:
                    break
                key = tuple(float(value) for value in pos.reshape(-1))
                fresh, cached = counter.charge(key)
                if fresh:
                    energy, gradient = evaluate(kind, pos, box)
                    gradient = np.asarray(gradient, dtype=np.float64)
                    counter.remember(key, (float(energy), gradient))
                else:
                    energy, gradient = cached
                    gradient = np.asarray(gradient, dtype=np.float64)
                force = np.ascontiguousarray(-gradient, dtype=np.float64)
                dr, lconv = art.next_displ(
                    nat,
                    float(energy),
                    force,
                    typ,
                    np.ascontiguousarray(pos, dtype=np.float64),
                    cell,
                    if_pos,
                )
                if task != "mode":
                    try:
                        stored = bool(art.extract("has_sad"))
                    except Exception:
                        stored = False
                    if stored:
                        # The displacement from this call is the push off the
                        # saddle (block_pushover when lpush_final is set).
                        # tau_sad is the geometry just passed in. The final
                        # minima are the harness connection test.
                        break
                if task == "mode":
                    found = _mode_overlap(art, job.get("reference"), free)
                    if found is not None:
                        overlap = max(overlap, found)
                    # The Lanczos vector is the mode. The next phase pushes
                    # away from this geometry, which is not the mode score.
                    if overlap >= float(job.get("overlap", 0.99)) or int(art.disp_code()) != lanczos_code:
                        lconv = overlap >= float(job.get("overlap", 0.99))
                        break
                if lconv:
                    break
                step = np.array(dr, dtype=np.float64, copy=True)
                if not np.all(np.isfinite(step)):
                    raise RuntimeError("pARTn returned a non-finite displacement")
                step_norm = float(np.linalg.norm(step))
                if step_norm > max_step:
                    max_step = step_norm
                if step_norm < 1e-14:
                    stalled += 1
                else:
                    stalled = 0
                if stalled >= 3:
                    raise RuntimeError("pARTn displacement stayed at zero")
                pos = pos + step
                pos[~free] = reactant[~free]
            if counter.calls < cap and np.all(np.isfinite(pos)):
                key = tuple(float(value) for value in pos.reshape(-1))
                fresh, cached = counter.charge(key)
                if fresh:
                    energy, gradient = evaluate(kind, pos, box)
                    gradient = np.asarray(gradient, dtype=np.float64)
                    counter.remember(key, (float(energy), gradient))
                else:
                    energy, gradient = cached
                    gradient = np.asarray(gradient, dtype=np.float64)
            try:
                has_error = bool(art.extract("has_error"))
                has_sad = bool(art.extract("has_sad"))
                artn_nevalf = int(art.extract("nevalf"))
            except Exception:
                pass
            if has_error and not has_sad:
                ierr, message = art.get_error()
                raise RuntimeError(f"pARTn error {ierr}: {message}")
    except ForceCallCap as exc:
        error = f"force-call cap: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if art is not None:
            try:
                has_error = bool(art.extract("has_error"))
                has_sad = bool(art.extract("has_sad"))
                artn_nevalf = int(art.extract("nevalf"))
                if has_sad:
                    saddle_pos, saddle_energy, saddle_gradient = _stored_saddle(
                        art, int(reactant.shape[0]), free, reactant, kind, box, counter
                    )
            except Exception:
                pass
            try:
                art.destroy()
            except Exception:
                pass
        os.chdir(previous)
    prefix: list[str] = []
    if error:
        if "force-call cap" in error or "ForceCallCap" in error:
            prefix.append("cap")
        else:
            prefix.append("crash")
    if task == "mode":
        record = judge_mode(
            job["name"],
            int(counter.calls),
            cap,
            overlap,
            float(job.get("overlap", 0.99)),
            error,
        )
        record["seed"] = job["seed"]
        record["zseed"] = int(job["zseed"])
        record["lconv"] = bool(lconv)
        record["artn_has_sad"] = has_sad
        record["artn_nevalf"] = artn_nevalf
        record["max_step"] = float(max_step)
        record["geometry"] = "final"
        return record
    if saddle_pos is not None:
        pos = saddle_pos
        energy = saddle_energy
        gradient = saddle_gradient
        # A library error after the saddle is stored belongs to the final
        # push. The scored point is tau_sad.
        if prefix == ["crash"]:
            prefix = []
            error = ""
    record = judge_saddle(
        job["name"],
        pos,
        None if energy is None else float(energy),
        None if gradient is None else np.asarray(gradient, dtype=np.float64),
        counter=counter,
        cap=cap,
        force_norm=float(job["force_norm"]),
        kind=kind,
        free=free,
        box=box,
        prefix=prefix,
        keep_mode=bool(job.get("keep_mode")),
    )
    if job.get("keep_positions"):
        record["positions"] = np.asarray(pos, dtype=np.float64).reshape(-1).tolist()
    if error:
        record["error"] = error
    record["seed"] = job["seed"]
    record["zseed"] = int(job["zseed"])
    record["lconv"] = bool(lconv)
    record["artn_has_sad"] = has_sad
    record["artn_nevalf"] = artn_nevalf
    record["max_step"] = float(max_step)
    record["geometry"] = "tau_sad" if saddle_pos is not None else "final"
    return record


def search_named(root: Path, problem_name: str, template_name: str, start: str, seed: str = "0") -> dict:
    """One pair. ``seed`` enters the pARTn ``zseed`` together with the pair name."""
    reason = available()
    if reason:
        raise RuntimeError(reason)
    problem, template, settings = prepared(root, problem_name, template_name, CODE)
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    pairs = cached_pairs(root, problem, settings)
    if start not in pairs:
        raise KeyError(f"{problem['id']} has no start {start}")
    record = _execute(_job(pairs[start], settings, template, seed))
    record["problem"] = problem["id"]
    record["template"] = template["id"]
    record["code"] = template["code"]
    record["code_version"] = template["code_version"]
    record["commit"] = template["commit"]
    record["algorithm"] = template["algorithm"]
    record["tuning"] = template["tuning"]
    return record


def _apply_documented(art, job: dict) -> None:
    """Knobs from the pARTn input and the authors' Pt / OptBench scripts.

    ``fire_set`` is separate: ``dt_init`` is the library value (atomic
    time on ``lammps/metal``, reduced time on ``lammps/lj``) and
    ``dt_max_f`` is the factor in ``dt_max = dt_init * dt_max_f``.
    """
    integers = ("nsmooth", "nnewchance", "lanczos_min_size")
    reals = ("eigval_thr", "push_over", "lanczos_eval_conv_thr")
    for name in integers:
        if job.get(name) is not None:
            _set(art, name, int(job[name]))
    for name in reals:
        if job.get(name) is not None:
            _set(art, name, float(job[name]))
    limits = job.get("nperp_limitation")
    if limits:
        values = np.asarray([int(item) for item in limits], dtype=np.int32)
        _set(art, "lnperp_limitation", True)
        _set(art, "nperp_limitation", values)


def _point_job(start: dict, settings: dict, template: dict) -> dict:
    section = _section(template)
    task = "mode" if settings.get("task") == "mode" else "saddle"
    label = f"0:{start['name']}"
    return {
        "name": start["name"],
        "reactant": start["positions"],
        "product": None,
        "free": [not bool(flag) for flag in start["fixed"]],
        "box": start["box"],
        "kind": settings["potential"],
        "cap": settings["cap"],
        "force_norm": settings["force_norm"],
        **_knobs(section),
        "fire_dt": _fire_dt(section, settings["potential"]),
        "push_init": start.get("push_init"),
        "task": task,
        "reference": start.get("reference"),
        "overlap": float(settings["overlap"]),
        "seed": "0",
        "zseed": pair_seed(label, _SEED_BASE),
        "shlib": str(library_path()),
        "interface": str(interface_dir()),
        "commit": template["commit"],
    }


def run_points(root: Path, problem: dict, template: dict, settings: dict, *, output: Path, workers: int) -> dict:
    """One ARTn search from each sample geometry."""
    scratch = output.parent / "work" / template["id"].replace("/", "-")
    starts = load_starts(root, problem, scratch)
    jobs = [_point_job(start, settings, template) for start in starts]
    records = run_jobs(jobs, _execute, workers)
    section = _section(template)
    parameters = {
        **{key: value for key, value in settings.items() if key != "task"},
        **section,
        "task": settings.get("task"),
        "engine_units": "lammps/metal for morse_pt, lammps/lj for lj",
        "counter": "harness",
        "seed": "0",
        "zseed_base": _SEED_BASE,
        "fire_dt_init": _fire_dt(section, settings["potential"]),
    }
    if settings.get("task") == "mode":
        comment = (
            "# counter harness. The search stops when the pARTn eigenvector "
            "overlaps the reference mode by the problem threshold.\n"
            "# The displacement that would leave that geometry is not applied after the overlap is met.\n"
            "# nevalf is the pARTn tally and is not this count."
        )
    else:
        comment = (
            "# counter harness. The start is the sample geometry, not a reactant/product pair.\n"
            "# ninit initial pushes use push_init along x of the first free atom.\n"
            "# A success is under the cap, at the force stop, and Hessian index 1.\n"
            "# nevalf is the pARTn tally and is not this count."
        )
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )


def _minimize_free(kind: str, positions: np.ndarray, free: np.ndarray, box, force_tol: float, budget: int):
    """L-BFGS on the free atoms. The returned call count is the harness count."""
    from optbench.pes import evaluate
    from optbench.rgsaddle_minmode import HarnessSurface, _lbfgs_minimize

    def call(point):
        full = np.array(positions, dtype=np.float64, copy=True)
        full[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
        energy, grad = evaluate(kind, full, box)
        return float(energy), np.asarray(grad, dtype=np.float64)[free].reshape(-1)

    surface = HarnessSurface(call)
    point, ok = _lbfgs_minimize(
        surface,
        positions[free].reshape(-1).copy(),
        force_tol,
        budget,
    )
    energy, _gradient = surface.energy_forces(point)
    out = np.array(positions, dtype=np.float64, copy=True)
    out[free] = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    return out, bool(ok), int(surface.calls), float(energy)


def _rmsd(moved: np.ndarray, reference: np.ndarray) -> float:
    delta = np.asarray(moved, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def run_catalog(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    """Displace the Pt island until the goal count of connected index-1 saddles.

    Each search is one pARTn run from a kick off the minimized reactant.
    The harness count for that search is the pARTn count plus the two
    endpoint minimizations. The repetition total also includes the
    initial minimization. A batch of ``workers`` kicks is started
    together. Connection and ``find_repeat`` checks still run in kick order,
    and kicks past the goal are not charged. The shipped template pushes
    the island in plane. The atom-list path calls ``draw_search_start``
    with ``search_seed(rep, attempt)``, the same displacement the other
    harness saddle runners use. ``<stem>.endpoints.npz`` stores float16
    free-atom deltas from the minimized reactant.
    """
    import statistics

    from optbench.failures import classify_failure
    from optbench.v2 import write_starts

    settings = point_settings(problem)
    goal = int((problem.get("goal") or {}).get("connected_saddles", 27))
    barrier_max = float((problem.get("goal") or {}).get("barrier_below_eV", 1.5))
    search = dict(template.get("search") or {})
    repetitions = int(search.get("repetitions", 1))
    magnitude = float(search.get("displace_magnitude", 0.1))
    tolerance = link_tolerance(problem)
    max_searches = int(search.get("max_searches", 20000))
    budget = int(search.get("search_budget", 10000))
    listed = [int(part) for part in str(search.get("displace_atoms", "0,1,2,3,4,5,6")).split(",") if part.strip()]
    pbc = problem_pbc(problem)
    displace_radius = float(search.get("displace_radius", 3.3))
    surface = str(search.get("push", "atom")) == "surface"
    dmin = float(search.get("dmin", 0.05))
    dmax = float(search.get("dmax", 0.5))
    min_atoms = int(search.get("min_atoms", 1))
    max_atoms = int(search.get("max_atoms", 7))
    starts = load_starts(root, problem, output.parent / "work" / template["id"].replace("/", "-"))
    base = starts[0]
    positions = np.asarray(base["positions"], dtype=np.float64)
    free = np.asarray([not bool(flag) for flag in base["fixed"]], dtype=bool)
    box = None if base["box"] is None else np.asarray(base["box"], dtype=np.float64)
    comments = [
        "# counter harness. Each search is one pARTn run. The geometry is tau_sad.",
        "# The displacement returned when the saddle is stored is the push off the saddle and is not applied.",
        "# Endpoint minima use the eOn r2205 link: perAtomNorm strictly below 0.1 angstrom.",
        "# A repeat is AKMCState.find_repeat on the saddle: barrier within 0.01 eV and the same per-atom test.",
        "# fraction_connected counts every search that is not Not Connected. unique_saddles counts good saddles.",
        "# Both minima are float16 free-atom deltas from the reactant in the sibling endpoints npz.",
    ]
    archive = EndpointArchive(output.with_name(output.stem + ".endpoints.npz"), int(np.count_nonzero(free)))
    archive.reset(stored_cell(box), np.flatnonzero(free))
    records = []
    failed_reps = 0
    for rep in range(1, repetitions + 1):
        minimum, min_ok, minimize_calls, min_energy = _minimize_free(
            settings["potential"], positions, free, box, settings["force_norm"], 5000
        )
        rng = np.random.default_rng(20261015 + rep)
        catalog = ProcessCatalog(box, tolerance=tolerance, barrier_max=barrier_max)
        search_calls = 0
        width = max(1, int(workers))
        while catalog.qualifying < goal and catalog.searches < max_searches:
            batch = []
            while len(batch) < width and catalog.searches + len(batch) < max_searches:
                index = catalog.searches + len(batch) + 1
                if surface:
                    count = int(rng.integers(min_atoms, max_atoms + 1))
                    chosen = [
                        int(value)
                        for value in rng.choice(np.asarray(listed, dtype=int), size=count, replace=False)
                    ]
                    push = _surface_push(positions.shape[0], chosen, dmin, dmax, rng)
                    origin = minimum
                else:
                    drawn = draw_search_start(
                        minimum,
                        ~free,
                        box,
                        pbc,
                        listed,
                        rep=rep,
                        attempt=index,
                        radius=displace_radius,
                        magnitude=magnitude,
                    )
                    origin = drawn["start"]
                    push = None
                job = _point_job(
                    {
                        "name": f"rep{rep}-search{index}",
                        "positions": np.asarray(origin, dtype=np.float64).tolist(),
                        "fixed": (~free).tolist(),
                        "box": base["box"],
                        "reference": None,
                        "push_init": None if push is None else push.tolist(),
                    },
                    {**settings, "cap": budget, "task": "saddle", "overlap": 0.99},
                    template,
                )
                job["keep_mode"] = True
                job["keep_positions"] = True
                job["ninit"] = int(_section(template)["ninit"])
                batch.append(job)
            print(
                f"{template['id']} rep {rep} searches {catalog.searches + 1} connected {catalog.qualifying}",
                flush=True,
            )
            for record in run_jobs(batch, _execute, width):
                if catalog.qualifying >= goal:
                    break
                climb = int(record["force_calls"])
                saddle = None if record.get("positions") is None else np.asarray(record["positions"], dtype=np.float64).reshape(-1, 3)
                record["reasons"] = list(record.get("reasons") or [])
                plus_ep = None
                minus_ep = None
                judged = False
                links = False
                structure = None
                barrier_value = None
                if record["success"] and saddle is not None and record.get("lowest_mode"):
                    axis = np.asarray(record["lowest_mode"], dtype=np.float64)
                    axis = axis / np.linalg.norm(axis)
                    saddle_free = saddle[free].reshape(-1)
                    plus, _plus_ok, plus_calls, _plus_energy = _minimize_free(
                        settings["potential"],
                        _placed(saddle, free, saddle_free + 0.1 * axis),
                        free,
                        box,
                        settings["force_norm"],
                        2000,
                    )
                    minus, _minus_ok, minus_calls, _minus_energy = _minimize_free(
                        settings["potential"],
                        _placed(saddle, free, saddle_free - 0.1 * axis),
                        free,
                        box,
                        settings["force_norm"],
                        2000,
                    )
                    plus_rmsd = _rmsd(plus[free], minimum[free])
                    minus_rmsd = _rmsd(minus[free], minimum[free])
                    links = connected_endpoints(minimum, plus, minus, box, tolerance)
                    barrier_value = float(record["energy"] - min_energy)
                    record["barrier"] = barrier_value
                    record["rmsd_plus"] = plus_rmsd
                    record["rmsd_minus"] = minus_rmsd
                    record["per_atom_max_plus"] = per_atom_max(plus, minimum, box)
                    record["per_atom_max_minus"] = per_atom_max(minus, minimum, box)
                    plus_ep = plus[free]
                    minus_ep = minus[free]
                    record["endpoint_calls"] = int(plus_calls + minus_calls)
                    record["force_calls"] = climb + int(plus_calls + minus_calls)
                    structure = saddle[free]
                    judged = True
                decision = catalog.observe(
                    judged=judged,
                    linked=links,
                    structure=structure,
                    barrier=barrier_value,
                )
                if judged:
                    record["reasons"] = list(decision["reasons"])
                    record["success"] = bool(decision["success"])
                    record["failure"] = None if record["success"] else classify_failure(record["reasons"])
                search_calls += int(record["force_calls"])
                record.pop("positions", None)
                record.pop("lowest_mode", None)
                status = archive.add(minimum[free], plus_ep, minus_ep)
                stamp_endpoint(record, archive.name, archive.count - 1, status)
                records.append(record)
        total_calls = int(minimize_calls + search_calls)
        if catalog.qualifying < goal:
            failed_reps += 1
        comments.append(
            f"# rep {rep} force_calls {total_calls} unique_saddles {catalog.unique} "
            f"fraction_connected {catalog.fraction_connected} total_searches {catalog.searches} "
            f"connected {catalog.qualifying} minimize_calls {minimize_calls} "
            f"minimum_converged {int(min_ok)} "
            + (
                f"seed {20261015 + rep}"
                if surface
                else f"search_seed (20261015+{rep})*1000003+attempt"
            )
        )
    totals = [
        int(line.split()[line.split().index("force_calls") + 1])
        for line in comments
        if "force_calls" in line.split()
    ]
    # The problem states no cap. A missed link is still a scored search.
    # The cap is the pARTn budget plus the two endpoint minimizations, the
    # same shape as the rgsaddle saddle catalog. A success stays under it.
    endpoint_budget = 2000
    search_cap = int(budget) + 2 * endpoint_budget
    from optbench.score import score_v2
    from optbench.v2 import collect_provenance

    metrics = (
        score_v2(
            [
                {
                    "force_calls": record["force_calls"],
                    "success": record["success"],
                    "failure": record["failure"],
                }
                for record in records
            ],
            search_cap,
        )
        if records
        else None
    )
    text = "\n".join(comments) + "\n"
    if totals:
        text += f"force_calls {statistics.fmean(totals)}\n"
        text += f"force_calls_min {min(totals)}\n"
        text += f"force_calls_max {max(totals)}\n"
    text += (
        f"code {template.get('code', CODE)}\n"
        f"code_version {template.get('code_version', '')}\n"
        f"commit {template.get('commit', '')}\n"
        f"algorithm {template.get('algorithm', '')}\n"
        f"tuning {template.get('tuning', '')}\n"
        f"template {template['id']}\n"
        f"problem {problem['id']}\n"
        f"counter harness\n"
        f"repetitions {repetitions}\n"
        f"nfailed {failed_reps}\n"
        f"goal {goal}\n"
        f"cap {search_cap}\n"
        f"n {len(records)}\n"
    )
    if metrics is not None:
        text += f"par2_mean {metrics['par2_mean']}\nv2_nfailed {metrics['nfailed']}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.save()
    output.write_text(text, encoding="utf-8")
    # The board scores one row per search. The repetition total above is
    # the cost of the searches that were run, including misses.
    output.with_suffix(".searches.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    write_starts(output.with_suffix(".starts.jsonl"), records)
    if metrics is not None:
        successes = [int(row["force_calls"]) for row in records if row["success"]]
        success_block = {
            "successes": len(successes),
            "nfailed": len(records) - len(successes),
            "n": len(records),
            "transition_states": sum(1 for row in records if row.get("endpoint_status") == STATUS_STORED),
        }
        if successes:
            success_block["force_calls"] = int(round(statistics.fmean(successes)))
            success_block["force_calls_min"] = min(successes)
            success_block["force_calls_max"] = max(successes)
            success_block["force_calls_median"] = statistics.median(successes)
        document = {
            "schema": "optbench.score/v2",
            "problem": problem["id"],
            "template": template["id"],
            "counter": "harness",
            "provenance": collect_provenance(
                code=str(template.get("code", CODE)),
                code_version=str(template.get("code_version", "")),
                commit=str(template.get("commit", "")),
                parameters={
                    "cap": search_cap,
                    "counter": "harness",
                    "geometry": "tau_sad",
                    "goal": goal,
                    "max_searches": max_searches,
                    "search_budget": budget,
                    "distance_difference": tolerance,
                },
            ),
            "score": metrics,
            "success_block": success_block,
        }
        output.with_suffix(".v2.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    archive.drop_log()
    return {
        "force_calls": statistics.fmean(totals) if totals else None,
        "nfailed": failed_reps,
        "starts": len(records),
        "template": template["id"],
        "problem": problem["id"],
    }


def _surface_push(natoms: int, indices: list[int], dmin: float, dmax: float, rng) -> np.ndarray:
    """In-plane push used by ``examples/OptBench_SaddleSearch.d``.

    Lengths stay in angstroms. ``engine_units = lammps/metal`` reads
    ``push_init`` in those units. The z component is zero so the
    heptamer is not kicked off the slab.
    """
    vec = np.zeros((natoms, 3), dtype=np.float64)
    for index in indices:
        direction = rng.random(3) - np.array([0.5, 0.5, 0.0])
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        vec[int(index)] = direction / norm * float(rng.uniform(dmin, dmax))
    return vec


def _placed(saddle: np.ndarray, free: np.ndarray, free_flat: np.ndarray) -> np.ndarray:
    out = np.array(saddle, dtype=np.float64, copy=True)
    out[free] = np.asarray(free_flat, dtype=np.float64).reshape(-1, 3)
    return out


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    reason = available()
    if reason:
        raise RuntimeError(reason)
    root = Path(root)
    from optbench.campaign import load_template, resolve_problem

    from .common import registry_pin, require_pin

    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, CODE))
    if template["engine"] != ENGINE:
        raise ValueError(f"{template_name} engine is {template['engine']}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template_name} belongs to {stated}")
    if problem.get("suite") in {"tsopt", "lowest-eigenvector"}:
        settings = point_settings(problem)
        if settings["cap"] is None:
            raise ValueError(f"{problem['id']} has no force-call cap")
        settings["task"] = "mode" if problem["suite"] == "lowest-eigenvector" else "saddle"
        if problem["suite"] == "tsopt":
            # The geometry is already a transition-state guess. Skip the
            # blind push and let Lanczos follow the lowest mode.
            template = dict(template)
            partn = dict(template.get("partn") or {})
            partn.setdefault("ninit", 0)
            template["partn"] = partn
        return run_points(root, problem, template, settings, output=output, workers=workers)
    if problem["id"] == "saddle-search/pt-island":
        return run_catalog(root, problem, template, output=output, workers=workers)
    problem, template, settings = prepared(root, problem_name, template_name, CODE)
    pairs = list(cached_pairs(root, problem, settings).values())
    jobs = [_job(pair, settings, template, "0") for pair in pairs]
    records = run_jobs(jobs, _execute, workers)
    for record in records:
        record["problem"] = problem["id"]
        record["template"] = template["id"]
    section = _section(template)
    parameters = {
        **settings,
        **section,
        "engine_units": "lammps/metal for morse_pt, lammps/lj for lj",
        "push": "one free atom from the reactant toward the product",
        "counter": "harness",
        "seed": "0",
        "zseed_base": _SEED_BASE,
    }
    comment = (
        "# counter harness. One force call is one potential evaluation in the next_displ loop.\n"
        "# nevalf is the pARTn tally and is not this count.\n"
        "# The product sets the initial push. This search does not build a band to the product.\n"
        "# A success is under the cap, at the force stop, and Hessian index 1."
    )
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )
