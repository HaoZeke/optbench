"""Train the mlopt networks with the aenet 2.0.5 optimizers.

The trajectory, fingerprints, architecture, and squared-error loss are
the problem. aenet supplies L-BFGS-B (``setulb``), Adam, online steepest
descent, Levenberg-Marquardt, and the extended Kalman filter. One
harness call is one full-dataset loss and gradient. The parent builds
the fingerprints once. Forked workers share those arrays and each loads
``setulb`` after the fork.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.aenet_ml import (
    FINGERPRINT_REVISION,
    INIT_SEED,
    LBFGS_FACTR,
    LBFGS_M,
    LBFGS_PGTOL,
    LossModel,
    Layout,
    build_descriptors,
    ensure_lbfgsb_built,
    initial_weights,
    parse_fpparas,
    train_start,
)
from optbench.campaign import geometry_archive
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

CODE_VERSION = "2.0.5"
CODE_COMMIT = "8f35990f5e3dd6cb296b953448cb11694d0cff3d"
FULL_STARTS = 50
CACHE_ROOT = Path(os.environ.get("OPTBENCH_AENET_CACHE", "/tmp/optbench-aenet-fp"))

PROTOCOL = {
    "energy": {"cap": 10000, "tolerance": 0.02, "c_energy": 1.0, "c_force": 0.0},
    "force": {"cap": 30000, "tolerance": 0.1, "c_energy": 1.0, "c_force": 0.1},
}
ALGORITHMS = {
    "bfgs": "L-BFGS-B",
    "adam": "Adam",
    "online_sd": "online_sd",
    "lm": "LM",
    "ekf": "EKF",
}

_MODEL: LossModel | None = None


def campaign_stem(template: dict) -> str:
    slug = str(template["id"]).split("/", 1)[1]
    return f"aenet-{CODE_VERSION}-{slug}"


def _finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value == -1.0:
            return "-1.0"
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _checked(problem: dict, template: dict) -> dict:
    if template.get("code") != "aenet":
        raise ValueError(f"{template['id']} is not an aenet template")
    if str(template.get("code_version")) != CODE_VERSION:
        raise ValueError(f"{template['id']} version is not {CODE_VERSION}")
    if str(template.get("commit")) != CODE_COMMIT:
        raise ValueError(f"{template['id']} commit is not the aenet pin")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    block = template.get("aenet") or {}
    method = str(block.get("method") or "")
    target = str(block.get("target") or "")
    if method not in ALGORITHMS:
        raise ValueError(f"{template['id']} has no aenet trainer")
    if target not in PROTOCOL:
        raise ValueError(f"{template['id']} target must be energy or force")
    if str(template.get("algorithm")) != ALGORITHMS[method]:
        raise ValueError(f"{template['id']} algorithm does not match {method}")
    protocol = PROTOCOL[target]
    cap = int(template["failure_cap"])
    if cap != protocol["cap"]:
        raise ValueError(f"{template['id']} cap is not {protocol['cap']}")
    stop = template.get("stop") or {}
    tolerance = float(stop["gradient_norm"])
    if abs(tolerance - protocol["tolerance"]) > 1e-12:
        raise ValueError(f"{template['id']} gradient stop is not {protocol['tolerance']}")
    c_energy = float(block["c_energy"])
    c_force = float(block["c_force"])
    if abs(c_energy - protocol["c_energy"]) > 1e-12 or abs(c_force - protocol["c_force"]) > 1e-12:
        raise ValueError(f"{template['id']} coefficients are not the {target} protocol")
    fp_parameters = str(block.get("fp_parameters") or "")
    if not fp_parameters:
        raise ValueError(f"{template['id']} has no fingerprint file")
    convergence = problem.get("convergence") or {}
    if convergence.get("max_force_calls") not in (None, ""):
        raise ValueError(f"{problem['id']} max_force_calls would replace the epoch cap")
    if "max_epochs" in convergence and int(convergence["max_epochs"]) != PROTOCOL["energy"]["cap"]:
        raise ValueError(f"{problem['id']} energy cap is not 10000")
    if "force_training_max_epochs" in convergence:
        if int(convergence["force_training_max_epochs"]) != PROTOCOL["force"]["cap"]:
            raise ValueError(f"{problem['id']} force cap is not 30000")
    if "energy_only_grad_norm" in convergence:
        if abs(float(convergence["energy_only_grad_norm"]) - PROTOCOL["energy"]["tolerance"]) > 1e-12:
            raise ValueError(f"{problem['id']} energy stop is not 0.02")
    if "force_training_grad_norm" in convergence:
        if abs(float(convergence["force_training_grad_norm"]) - PROTOCOL["force"]["tolerance"]) > 1e-12:
            raise ValueError(f"{problem['id']} force stop is not 0.1")
    sample = problem.get("sample") or {}
    full_n = int(sample.get("starts") or sample.get("initializations") or 0)
    if full_n != FULL_STARTS:
        raise ValueError(f"{problem['id']} states {full_n} starts, not {FULL_STARTS}")
    return {
        "method": method,
        "target": target,
        "cap": cap,
        "tolerance": tolerance,
        "c_energy": c_energy,
        "c_force": c_force,
        "fp_parameters": fp_parameters,
        "full_n": full_n,
    }


def _read_trajectory(path: Path):
    from ase.io import iread

    positions = []
    forces = []
    energies = []
    symbols = None
    cell = None
    for atoms in iread(str(path), index=":"):
        current = list(atoms.get_chemical_symbols())
        frame_cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if symbols is None:
            symbols = current
            cell = frame_cell
        else:
            if current != symbols:
                raise ValueError(f"{path} changes the atom order")
            if not np.allclose(frame_cell, cell):
                raise ValueError(f"{path} has more than one cell")
        positions.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
    if not positions or symbols is None or cell is None:
        raise ValueError(f"{path} has no frames")
    return (
        np.stack(positions),
        np.asarray(energies, dtype=np.float64),
        np.stack(forces),
        symbols,
        cell,
    )


def _cache_dir(problem: dict, spec: dict, fp_hash: str) -> Path:
    slug = problem["id"].replace("/", "-")
    kind = "force" if spec["c_force"] else "energy"
    return CACHE_ROOT / f"{slug}-{kind}-{fp_hash[:12]}"


def _load_cache(cache: Path, elements: list[str], derivatives: bool):
    meta_path = cache / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("revision") != FINGERPRINT_REVISION or meta.get("derivatives") != derivatives:
        return None
    if meta.get("elements") != elements:
        return None
    fp = {element: np.load(cache / f"fp-{element}.npy") for element in elements}
    dG = None
    if derivatives:
        dG = {element: np.load(cache / f"dG-{element}.npy") for element in elements}
    atoms = {element: np.load(cache / f"atoms-{element}.npy") for element in elements}
    energy = np.load(cache / "energy.npy")
    forces = np.load(cache / "forces.npy")
    return fp, dG, atoms, energy, forces


def _save_cache(cache: Path, elements: list[str], fp, dG, atoms, energy, forces, fp_hash: str, derivatives: bool) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for element in elements:
        np.save(cache / f"fp-{element}.npy", fp[element])
        np.save(cache / f"atoms-{element}.npy", atoms[element])
        if derivatives:
            np.save(cache / f"dG-{element}.npy", dG[element])
    np.save(cache / "energy.npy", energy)
    np.save(cache / "forces.npy", forces)
    meta = {
        "revision": FINGERPRINT_REVISION,
        "fp_sha256": fp_hash,
        "derivatives": derivatives,
        "elements": elements,
    }
    (cache / "meta.json").write_text(json.dumps(meta, sort_keys=True) + "\n", encoding="utf-8")


def prepare_model(root: Path, problem: dict, template: dict) -> LossModel:
    """Fingerprints and the loss. A repeated call reloads the cache."""
    spec = _checked(problem, template)
    fp_path = root / spec["fp_parameters"]
    text = fp_path.read_text(encoding="utf-8")
    fp_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    elements, specs = parse_fpparas(text)
    derivatives = spec["c_force"] > 0.0
    cache = _cache_dir(problem, spec, fp_hash)
    loaded = _load_cache(cache, elements, derivatives)
    if loaded is None:
        trajectory = geometry_archive(root, problem)
        positions, energy, forces, symbols, cell = _read_trajectory(trajectory)
        fp, dG, atoms = build_descriptors(
            positions,
            cell,
            symbols,
            elements,
            specs,
            derivatives=derivatives,
        )
        _save_cache(cache, elements, fp, dG, atoms, energy, forces, fp_hash, derivatives)
    else:
        fp, dG, atoms, energy, forces = loaded
    n_fp = {element: len(specs[element]) for element in elements}
    for element, array in fp.items():
        if array.shape[-1] != n_fp[element]:
            raise RuntimeError(f"{element} fingerprint width is not {n_fp[element]}")
    layout = Layout(elements, n_fp)
    return LossModel(
        fp=fp,
        atoms=atoms,
        energy=energy,
        forces=forces,
        layout=layout,
        c_energy=spec["c_energy"],
        c_force=spec["c_force"],
        dG=dG,
    )


def _train_index(index: int) -> dict:
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if _MODEL is None:
        raise RuntimeError("the aenet worker has no model")
    spec = _SPEC
    weights = initial_weights(_MODEL.layout, index)
    record = train_start(
        _MODEL,
        weights,
        spec["method"],
        cap=spec["cap"],
        tolerance=spec["tolerance"],
    )
    record["schema"] = "optbench.start/v2"
    record["start"] = f"{index:04d}"
    record["counter"] = "harness"
    record["comparable"] = True
    record["energy_rmse"] = _finite(record.get("energy_rmse"))
    record["force_rmse"] = _finite(record.get("force_rmse"))
    if record.get("max_group_norm") is not None:
        record["max_group_norm"] = _finite(record["max_group_norm"])
    if record["success"]:
        record["failure"] = None
        record["reasons"] = []
    else:
        record["reasons"] = [str(record["failure"])]
    return record


_SPEC: dict = {}


def _dat(summary: dict, template: dict, metrics: dict, spec: dict, partial: bool) -> str:
    rows = {
        "problem": template.get("problem"),
        "mean_epoch": summary.get("force_calls"),
        "median_epoch": summary.get("median"),
        "mean_energy_rmse": summary.get("mean_energy_rmse"),
        "median_energy_rmse": summary.get("median_energy_rmse"),
        "mean_force_rmse": summary.get("mean_force_rmse"),
        "median_force_rmse": summary.get("median_force_rmse"),
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": CODE_VERSION,
        "commit": CODE_COMMIT,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "n_cap": metrics.get("n_cap"),
        "n_divergence": metrics.get("n_divergence"),
        "n_crash": metrics.get("n_crash"),
        "n_wrong_stationary": metrics.get("n_wrong_stationary"),
        "cap": spec["cap"],
    }
    if partial:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One call is one full-dataset loss and gradient.",
        "# An L-BFGS-B line-search request and an LM error retry each count.",
        "# A repeated weight vector is cached and is not counted again.",
        "# Success requires every parameter-tensor gradient L2 at or below the stop, with the call count under the cap.",
        "# pyamff 8037eb98 stopped on the max absolute entry of each tensor. This row uses the L2 test.",
        "# There is no weight-space Hessian, so wrong_stationary is unused.",
        "# mean_epoch is the half-even mean of the successful call counts.",
        "# RMSE means are arithmetic means of the successes. force_rmse is -1 when the force coefficient is 0.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _rmse_stats(records: list[dict], key: str) -> tuple[float | None, float | None]:
    values = [float(record[key]) for record in records if record["success"] and record.get(key) is not None]
    if not values:
        return None, None
    return sum(values) / len(values), statistics.median(values)


def _parameters(spec: dict) -> dict:
    parameters = {
        "method": spec["method"],
        "target": spec["target"],
        "cap": spec["cap"],
        "gradient_norm": spec["tolerance"],
        "c_energy": spec["c_energy"],
        "c_force": spec["c_force"],
        "init_seed": INIT_SEED,
        "initialization": "uniform[-1, 1)",
        "hidden": [10, 7],
        "activation": "tanh",
        "fingerprint_revision": FINGERPRINT_REVISION,
        "fp_parameters": spec["fp_parameters"],
        "neighbor_cutoff": 6.0,
    }
    if spec["method"] == "bfgs":
        parameters.update({"memory": LBFGS_M, "factr": LBFGS_FACTR, "pgtol": LBFGS_PGTOL})
    elif spec["method"] == "adam":
        parameters.update({"mu": 0.01, "b1": 0.9, "b2": 0.999, "eps": 1.0e-8, "batch": "all structures"})
    elif spec["method"] == "online_sd":
        parameters.update({"alpha": 0.01, "gamma": 0.01, "update": "full gradient"})
    elif spec["method"] == "lm":
        parameters.update({"learnrate": 0.01, "adjust": 10, "batch": "all structures"})
    elif spec["method"] == "ekf":
        parameters.update(
            {
                "lambda": 0.999,
                "lambda0": 0.97,
                "state": 10,
                "pnoise": 0.0,
                "wgroup": 100,
                "covariance": "0.1 on the diagonal and 0 off the diagonal",
            }
        )
    return parameters


def _publish(output: Path, records: list[dict], problem: dict, template: dict, spec: dict) -> dict:
    ordered = sorted(records, key=lambda row: str(row["start"]))
    cap = spec["cap"]
    legacy_counts = [cap if not record["success"] else int(record["force_calls"]) for record in ordered]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record in ordered if record["success"]]
    summary["median"] = statistics.median(successes) if successes else None
    mean_energy, median_energy = _rmse_stats(ordered, "energy_rmse")
    if spec["c_force"] == 0.0:
        mean_force, median_force = -1.0, -1.0
    else:
        mean_force, median_force = _rmse_stats(ordered, "force_rmse")
    summary["mean_energy_rmse"] = mean_energy
    summary["median_energy_rmse"] = median_energy
    summary["mean_force_rmse"] = mean_force
    summary["median_force_rmse"] = median_force
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in ordered
        ],
        cap,
    )
    partial = len(ordered) < spec["full_n"]
    provenance = collect_provenance(
        code="aenet",
        code_version=CODE_VERSION,
        commit=CODE_COMMIT,
        parameters=_parameters(spec),
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "campaign",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter. One call is one full-dataset loss and gradient. "
            "The stop is the L2 norm of each parameter tensor."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": "aenet",
                "algorithm": template.get("algorithm"),
                "code_version": CODE_VERSION,
                "commit": CODE_COMMIT,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": _parameters(spec),
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "mean_epoch": summary.get("force_calls"),
                    "median_epoch": summary.get("median"),
                    "mean_energy_rmse": summary.get("mean_energy_rmse"),
                    "median_energy_rmse": summary.get("median_energy_rmse"),
                    "mean_force_rmse": summary.get("mean_force_rmse"),
                    "median_force_rmse": summary.get("median_force_rmse"),
                    "nfailed": summary.get("nfailed"),
                },
                "starts": ordered,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".dat.tmp")
    temporary.write_text(_dat(summary, template, metrics, spec, partial), encoding="utf-8")
    temporary.replace(output)
    write_starts(output.with_suffix(".starts.jsonl"), ordered)
    v2_path = output.with_suffix(".v2.json")
    v2_tmp = v2_path.with_suffix(".json.tmp")
    v2_tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    v2_tmp.replace(v2_path)
    summary["starts"] = len(ordered)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["partial"] = partial
    return summary


def _read_done(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row["start"])
        if name in rows:
            raise RuntimeError(f"duplicate start {name} in {path}")
        rows[name] = row
    return rows


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    spec = _checked(problem, template)
    count = spec["full_n"] if limit <= 0 else min(int(limit), spec["full_n"])
    wanted = [f"{index:04d}" for index in range(count)]
    done = _read_done(output.with_suffix(".starts.jsonl")) if resume else {}
    missing = [index for index, name in enumerate(wanted) if name not in done]
    global _MODEL, _SPEC
    if missing:
        started = time.perf_counter()
        print(f"{template['id']} fingerprints", flush=True)
        _MODEL = prepare_model(root, problem, template)
        _SPEC = spec
        if spec["method"] == "bfgs":
            ensure_lbfgsb_built()
        print(
            f"{template['id']} {len(missing)} starts, {workers} workers, "
            f"weights {_MODEL.layout.n}, images {_MODEL.n_images}",
            flush=True,
        )
        fresh: list[dict] = []
        if workers <= 1:
            for index in missing:
                fresh.append(_train_index(index))
                _publish(output, list(done.values()) + fresh, problem, template, spec)
                row = fresh[-1]
                print(
                    f"{template['id']} {row['start']} calls={row['force_calls']} "
                    f"{'ok' if row['success'] else row['failure']}",
                    flush=True,
                )
        else:
            import multiprocessing as mp

            context = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(_train_index, index): index for index in missing}
                finished = 0
                for future in as_completed(futures):
                    row = future.result()
                    fresh.append(row)
                    finished += 1
                    _publish(output, list(done.values()) + fresh, problem, template, spec)
                    print(
                        f"{template['id']} {finished}/{len(missing)} {row['start']} "
                        f"calls={row['force_calls']} {'ok' if row['success'] else row['failure']}",
                        flush=True,
                    )
        done.update({row["start"]: row for row in fresh})
        elapsed = time.perf_counter() - started
        print(f"{template['id']} finished in {elapsed:.1f}s", flush=True)
    records = [done[name] for name in wanted]
    return _publish(output, records, problem, template, spec)
