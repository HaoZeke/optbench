"""Write a harness campaign for one lane-B template.

The start set, the force cap, and the success rule come from the
problem file. The template names the client and the library knobs.
"""

from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.lane_load import load_starts
from optbench.lane_methods import run_job
from optbench.noise import KAPPA, WINDOW
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

_PINS = {
    "scipy": ("1.18.1", "e4e854eaa8f18d807cd3496028e257e36caa93cc"),
    "pele": ("0.1.2", "b7081b932a060be11342fbe8287ac35da8ada879"),
    "sqnm": ("1.3", "4241692173cccb42a9ae05974bfd62285ec2cd7b"),
    "geometric": ("1.1.1", "f657e69bae9c81bf7a23e7375db98502ca5c8c30"),
    "ase-ga": ("1.0.3", "f3ecaa80915d5ddbf2298e17702e6df21f5e7b9c"),
    "ASE": ("3.29.0", "f27c0005ae6a67ea419f996e728668865bfc1f86"),
    "nlopt": ("2.11.0", "88c424d4f458412787df96fcc95218acbca224fd"),
    "optking": ("767e2c0c", "767e2c0c13ff2b851ac82101ccd5aab9a558a4f1"),
    "pybcga": ("fb646c19", "fb646c19b1829656bf16cc72ca318bef2d37bfeb"),
    "ase-mh": ("1.1.2", "6baabf94c48fd4f673bcef5307915e49c2c906b2"),
    "jax-md": ("0.2.29", "a41c7d19f6468f4e5263c32c12c9ed6cba26ebff"),
    "scgo": ("0.9.1", "2338742a434a920ed82e52a72dc1338cf06647c1"),
    "torch-sim": ("0.6.2", "2e9c16f5afe354282279630a57885f46521a1cc0"),
    "ezga": ("1.2.31", "1395e8de01f003e40a3352987377d5f863a2e388"),
    "agox": ("0.0.1.dev1+g22d94a6fa", "22d94a6fae8cf2d08d51ecc8b9291a47667aa95c"),
}


def _one_thread() -> None:
    """Keep one BLAS thread per worker. Eight workers each opening a full pool stall."""
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def preflight(template: dict) -> str:
    """Import the client and return the version string the dat will record.

    A missing import raises here, before any result file is opened.
    """
    method = str((template.get("search") or {}).get("method") or "")
    code = str(template.get("code") or "")
    if method == "gmin" or code == "gmin":
        from optbench.lane_methods import _gmin_reason

        raise RuntimeError(_gmin_reason())
    version = _import_version(code, method)
    expected = str(template.get("code_version") or "")
    if expected and version != expected:
        raise RuntimeError(f"{code} {version} does not match template {expected}")
    return version


def _import_version(code: str, method: str) -> str:
    if code == "scipy" or method.startswith("scipy"):
        import scipy

        return scipy.__version__
    if code == "pele" or method.startswith("pele"):
        import importlib.metadata as metadata

        import pele.basinhopping  # noqa: F401
        import pele.optimize  # noqa: F401

        return metadata.version("pele")
    if code == "sqnm" or method == "sqnm":
        import sqnm.free_or_fixed_cell_sqnm  # noqa: F401

        # setup.py on tag v1.3 still says 1.2. The registry pin is the tag.
        return "1.3"
    if code == "geometric" or method == "geometric":
        import geometric

        return geometric.__version__
    if code == "ase-ga" or method == "ase-ga":
        import importlib.metadata as metadata

        import ase_ga.particle_mutations  # noqa: F401

        return metadata.version("ase-ga")
    if code == "ASE" or method.startswith("ase-"):
        import ase

        return ase.__version__
    if code == "nlopt" or method.startswith("nlopt"):
        import nlopt

        return nlopt.__version__
    if code == "optking" or method == "optking":
        import importlib.metadata as metadata

        import optking.opt_helper  # noqa: F401

        installed = metadata.version("optking")
        if "767e2c0c" not in installed:
            raise RuntimeError(f"optking {installed} is not commit 767e2c0c")
        return "767e2c0c"
    if code == "pybcga" or method == "pybcga":
        import bcga.genetic_algorithm  # noqa: F401
        import bcga.pele_interface  # noqa: F401

        return "fb646c19"
    if code == "ase-mh" or method == "goedecker-mh":
        import importlib.metadata as metadata

        import minimahopping.minhop  # noqa: F401

        return metadata.version("minimahopping")
    if code == "jax-md" or method == "jaxmd-fire":
        import importlib.metadata as metadata

        import jax_md.minimize  # noqa: F401

        return metadata.version("jax-md")
    if code == "scgo" or method == "scgo-bh":
        import importlib.metadata as metadata

        import scgo.algorithms.basinhopping_go  # noqa: F401

        return metadata.version("scgo")
    if code == "torch-sim" or method == "torchsim-fire":
        import importlib.metadata as metadata

        import torch_sim  # noqa: F401

        return metadata.version("torch-sim-atomistic")
    if code == "ezga" or method == "ezga":
        import importlib.metadata as metadata

        import ezga.factory  # noqa: F401

        return metadata.version("ezga-lib")
    if code == "agox" or method == "agox-bh":
        import importlib.metadata as metadata

        import agox  # noqa: F401

        return metadata.version("agox")
    raise RuntimeError(f"no preflight for {code} method {method}")


def _checkpoint_path(output: Path) -> Path:
    """Per-start rows kept beside the campaign directory.

    The board reads only files directly in ``results/campaign``. A checkpoint
    next to that directory is not a row.
    """
    if output.parent.name == "campaign":
        return output.parent.parent / "checkpoints" / f"{output.stem}.starts.jsonl"
    return output.with_suffix(".checkpoint.jsonl")


def _load_checkpoint(path: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        start = row.get("start")
        if start:
            found[str(start)] = row
    return found


def _dump_checkpoint(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    if geometries is not None:
        raise ValueError("the harness runner reads the problem archive")
    _one_thread()
    version = preflight(template)
    commit = str(template.get("commit") or "")
    pinned = _PINS.get(str(template.get("code") or ""))
    if pinned and not commit.startswith(pinned[1]):
        raise RuntimeError(f"{template['id']} commit is not the registry pin")
    search = dict(template.get("search") or {})
    if "method" not in search or "family" not in search:
        raise ValueError(f"{template['id']} needs search.method and search.family")
    convergence = problem.get("convergence") or {}
    potential = problem.get("potential") or {}
    gtol = float(convergence.get("force_norm") or convergence.get("gradient_norm") or 0.0)
    cap = convergence.get("max_force_calls")
    if cap is None and search["family"] != "minimize":
        cap = (template.get("stop") or {}).get("max_force_calls")
    if cap is None:
        raise ValueError(f"{problem['id']} has no force-call cap")
    cap = int(cap)
    stop = template.get("stop") or {}
    if "max_force_calls" in stop and int(stop["max_force_calls"]) < cap:
        raise ValueError("template max_force_calls is below the problem force-call cap")
    if "force_norm" in convergence and "force_norm" in stop:
        if abs(float(stop["force_norm"]) - float(convergence["force_norm"])) > 1e-12:
            raise ValueError("template force_norm is not the problem force norm")
    loaded = load_starts(root, problem, work=work, output=output, limit=limit)
    expected = (problem.get("sample") or {}).get("starts")
    target = convergence.get("energy_minimum", potential.get("energy_minimum"))
    window = float(convergence.get("energy_above_minimum") or 0.001)
    if search["family"] == "globalopt" and target is None:
        raise ValueError(f"{problem['id']} has no energy minimum")
    noise = problem.get("noise") or {}
    sigma = search.get("sigma")
    if sigma is not None:
        allowed = [float(value) for value in (noise.get("sigma") or [])]
        if float(sigma) not in allowed:
            raise ValueError(f"{template['id']} sigma {sigma} is not in the problem list")
    kind = _kind(problem)
    layout = str((problem.get("sample") or {}).get("layout") or "")
    jobs = []
    for row in loaded:
        seed = 314159 + int(row["index"])
        noise_seed = int(noise.get("seed") or 0) + int(row["index"]) + int(float(sigma or 0.0) * 1e6)
        energy_tol = max(0.01, float(noise.get("kappa") or KAPPA) * float(sigma or 0.0))
        jobs.append(
            {
                "index": int(row["index"]),
                "start": row["start"],
                "coords": row["coords"].tolist(),
                "box": None if row["box"] is None else row["box"].tolist(),
                "species": None if row["species"] is None else row["species"].tolist(),
                "mode": None if row["mode"] is None else row["mode"].tolist(),
                "kind": kind,
                "layout": layout,
                "family": search["family"],
                "method": search["method"],
                "scipy_method": search.get("scipy_method"),
                "profile": template.get("profile") or "default",
                "cap": cap,
                "gtol": gtol,
                "seed": seed,
                "target": None if target is None else float(target),
                "energy_window": window,
                "sigma": float(sigma or 0.0),
                "kappa": float(noise.get("kappa") or KAPPA),
                "noise_window": int(noise.get("window") or WINDOW),
                "energy_tol": energy_tol,
                "noise_seed": noise_seed,
                "coordsys": search.get("coordsys"),
                "nlopt_algorithm": search.get("nlopt_algorithm"),
                "overlap": (problem.get("sample") or {}).get("overlap", 0.99),
            }
        )
    checkpoint = _checkpoint_path(output)
    finished = _load_checkpoint(checkpoint) if resume else {}
    if not resume and checkpoint.exists():
        checkpoint.unlink()
    pending = [job for job in jobs if str(job["start"]) not in finished]
    fresh: list[dict] = []

    def _store(record: dict) -> None:
        fresh.append(record)
        by_start = dict(finished)
        for row in fresh:
            by_start[str(row["start"])] = row
        ordered_now = [by_start[str(job["start"])] for job in jobs if str(job["start"]) in by_start]
        _dump_checkpoint(checkpoint, ordered_now)
        print(
            f"{template['id']} {record['start']} {len(ordered_now)}/{len(jobs)} "
            f"calls={record['force_calls']} success={record['success']}",
            flush=True,
        )

    if pending and workers <= 1:
        for job in pending:
            _store(run_job(job))
    elif pending:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(run_job, job) for job in pending]
            for future in as_completed(futures):
                _store(future.result())
    by_start = dict(finished)
    for row in fresh:
        by_start[str(row["start"])] = row
    ordered = [by_start[str(job["start"])] for job in jobs if str(job["start"]) in by_start]
    expected_n = len(jobs) if expected is None else int(expected)
    complete = limit <= 0 and len(ordered) == len(jobs) and len(ordered) >= expected_n
    if not complete:
        failed = sum(1 for row in ordered if not row.get("success"))
        return {
            "force_calls": None,
            "nfailed": failed,
            "starts": len(ordered),
            "template": template["id"],
            "problem": problem["id"],
            "complete": False,
        }
    for record in ordered:
        record.pop("index", None)
        record.pop("_point", None)
    summary = _write(problem, template, ordered, cap, version, commit, False, output)
    if checkpoint.exists():
        checkpoint.unlink()
    summary["complete"] = True
    return summary


def _kind(problem: dict) -> str:
    if problem["id"] == "globalopt/lj100-binary":
        return "binary"
    if problem["id"] == "minimization/morse-bulk":
        return "morse"
    return "lj"


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _write(problem, template, records, cap, version, commit, partial, output: Path) -> dict:
    family = str((template.get("search") or {}).get("family") or "")
    legacy_ok = [bool(record["converged"]) if family == "minimize" else bool(record["success"]) for record in records]
    legacy_counts = [cap if not ok else int(record["force_calls"]) for record, ok in zip(records, legacy_ok, strict=True)]
    summary = summarize_capped(legacy_counts, cap)
    successes = [int(record["force_calls"]) for record, ok in zip(records, legacy_ok, strict=True) if ok]
    summary["median"] = statistics.median(successes) if successes else None
    energies = [float(record["energy"]) for record in records if record.get("energy") is not None]
    if energies:
        summary["energy_min"] = min(energies)
        summary["energy_max"] = max(energies)
    metrics = score_v2(
        [
            {"force_calls": record["force_calls"], "success": record["success"], "failure": record["failure"]}
            for record in records
        ],
        cap,
    )
    search = dict(template.get("search") or {})
    provenance = collect_provenance(
        code=str(template.get("code")),
        code_version=version,
        commit=commit,
        parameters=search,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "archive",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": "Harness counter. One fresh coordinate is one force call.",
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning") or template.get("profile") or "",
                "present_source": True,
                "cap": cap,
                "parameters": search,
                "provenance": provenance,
                "metrics": metrics,
                "legacy": {
                    "force_calls": summary.get("force_calls"),
                    "force_calls_min": summary.get("force_calls_min"),
                    "force_calls_max": summary.get("force_calls_max"),
                    "nfailed": summary.get("nfailed"),
                    "median": summary.get("median"),
                },
                "starts": records,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics, commit, version, problem, partial, cap), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    summary["par2_mean"] = metrics["par2_mean"]
    summary["v2_nfailed"] = metrics["nfailed"]
    summary["nfailed"] = summary.get("nfailed")
    return summary


def _dat(summary, template, metrics, commit, version, problem, partial, cap) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "force_calls_median": summary.get("median"),
        "nfailed": summary.get("nfailed"),
        "median": summary.get("median"),
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": problem["id"],
        "tuning": template.get("tuning") or template.get("profile") or "",
        "counter": "harness",
        "cap": cap,
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
        "n_wrong_stationary": metrics.get("n_wrong_stationary"),
        "n_divergence": metrics.get("n_divergence"),
        "n_crash": metrics.get("n_crash"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
    }
    if partial:
        rows["partial"] = "true"
    if template.get("ranked") is False:
        rows["ranked"] = "false"
    lines = [
        "# counter harness. One fresh coordinate is one force call.",
        "# A repeated coordinate is free. A call at the cap is refused and is not counted.",
    ]
    family = str((template.get("search") or {}).get("family") or "")
    if family == "minimize":
        lines.append("# The force stop is the Euclidean force. v2 success also requires Hessian index 0.")
        if problem["id"] == "minimization/lj38-noise":
            lines.append(
                "# sigma 0 uses the clean force and Hessian rule. Positive sigma compares the clean "
                "energy and Kabsch RMSD with an uncounted clean minimization. The tolerance is max(0.01, kappa * sigma)."
            )
    elif family == "globalopt":
        lines.append("# A hit is a converged quench inside the energy window, with the call count strictly under the cap.")
    elif family == "eigen":
        lines.append("# One matvec is a central difference, two force calls. Success is absolute overlap of at least 0.99.")
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"
