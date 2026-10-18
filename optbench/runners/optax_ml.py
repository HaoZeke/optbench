"""Campaign driver for the Optax mlopt rows.

The trajectory and the fingerprint file are the ones named by the problem
and the template. Fingerprints are cached under ``OPTBENCH_MLOPT_CACHE``.
A start is one draw of uniform[-1, 1) weights. The draw for index ``i``
uses ``seed + i``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from optbench.campaign import geometry_archive, load_template, resolve_problem
from optbench.ml_bp import load_trajectory
from optbench.ml_optax import TrainSettings, train
from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts


def _settings(template: dict) -> TrainSettings:
    section = template.get("optax") or {}
    if "method" not in section:
        raise ValueError(f"{template['id']} needs optax.method")
    hidden = tuple(int(size) for size in section.get("hidden_layers") or ())
    rate = section.get("learning_rate")
    return TrainSettings(
        method=str(section["method"]),
        learning_rate=None if rate in (None, "") else float(rate),
        hidden=hidden,
        seed=int(section["seed"]),
        energy_coefficient=float(section["energy_coefficient"]),
        force_coefficient=float(section["force_coefficient"]),
        grad_tol=float(section["grad_tol"]),
        energy_rmse_tol=float(section["energy_rmse_tol"]),
        force_rmse_tol=float(section["force_rmse_tol"]),
        cap=int(template["failure_cap"]),
        memory_size=int(section.get("memory_size") or 10),
        max_linesearch_steps=int(section.get("max_linesearch_steps") or 20),
    )


def _installed_commit() -> str:
    import json
    from importlib.metadata import distribution

    text = distribution("optax").read_text("direct_url.json")
    if not text:
        raise RuntimeError("optax was not installed from a pinned URL")
    info = json.loads(text).get("vcs_info") or {}
    commit = str(info.get("commit_id") or "")
    if not commit:
        raise RuntimeError("optax direct_url.json has no commit")
    return commit


def prepare(root: Path, problem: dict, template: dict) -> dict:
    section = template["optax"]
    traj = geometry_archive(root, problem)
    fingerprints = root / str(section["fingerprints"])
    if not fingerprints.is_file():
        raise FileNotFoundError(fingerprints)
    cutoff = float(section.get("neighbor_cutoff") or 6.0)
    return load_trajectory(traj, fingerprints, cutoff=cutoff)


def _checkpoint(output: Path) -> Path:
    return Path("/tmp/optbench-mlopt/checkpoints") / f"{output.stem}.pkl"


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def render_dat(
    records: list[dict],
    *,
    problem: dict,
    template: dict,
    metrics: dict,
    full_n: int,
    commit: str,
    version: str,
) -> str:
    cap = int(template["failure_cap"])
    successes = [record for record in records if record["success"]]
    settings = _settings(template)
    legacy_counts = [cap if not record["success"] else int(record["force_calls"]) for record in records]
    summary = summarize_capped(legacy_counts, cap)
    force_reported = float(settings.force_coefficient) == 0.0
    rows = {
        "problem": problem["id"],
        "template": template["id"],
        "cap": cap,
        "counter": "harness",
        "code": template.get("code"),
        "code_version": version,
        "commit": commit,
        "tuning": template.get("tuning", ""),
        "algorithm": template.get("algorithm"),
        "n": len(records),
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": _median([float(record["force_calls"]) for record in successes]),
        "mean_epoch": _mean([float(record["epoch"]) for record in successes]),
        "median_epoch": _median([float(record["epoch"]) for record in successes]),
        "mean_energy_rmse": _mean([float(record["energy_rmse"]) for record in successes]),
        "median_energy_rmse": _median([float(record["energy_rmse"]) for record in successes]),
        "mean_force_rmse": -1.0 if force_reported else _mean([float(record["force_rmse"]) for record in successes]),
        "median_force_rmse": -1.0
        if force_reported
        else _median([float(record["force_rmse"]) for record in successes]),
        "par2_mean": metrics.get("par2_mean"),
        "wilson_low": metrics.get("wilson_low"),
        "wilson_high": metrics.get("wilson_high"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
    }
    if len(records) < full_n:
        rows["partial"] = "true"
    lines = [
        "# counter harness. One force call is one evaluation of the squared-error loss and its gradient.",
        "# The network is the PyAMFF 10-7 tanh model. Initial weights are uniform[-1, 1).",
        "# A start that first meets the stop on the call at the cap is a cap failure.",
    ]
    if settings.method == "lbfgs":
        lines.append("# L-BFGS counts each zoom line-search evaluation. The RMSE at the accepted point is a verification call.")
    else:
        lines.append("# Adam and Rprop take one evaluation per epoch, including the initial weights.")
    if force_reported:
        lines.append("# mean_force_rmse and median_force_rmse are -1 because the force coefficient is 0. The stop still reads the force RMSE.")
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _document(problem, template, records, metrics, cap, commit, version, output: Path) -> dict:
    settings = _settings(template)
    parameters = {
        "method": settings.method,
        "learning_rate": settings.learning_rate,
        "hidden_layers": list(settings.hidden),
        "seed": settings.seed,
        "energy_coefficient": settings.energy_coefficient,
        "force_coefficient": settings.force_coefficient,
        "grad_tol": settings.grad_tol,
        "energy_rmse_tol": settings.energy_rmse_tol,
        "force_rmse_tol": settings.force_rmse_tol,
        "cap": cap,
        "memory_size": settings.memory_size,
        "max_linesearch_steps": settings.max_linesearch_steps,
    }
    provenance = collect_provenance(
        code=str(template.get("code") or "Optax"),
        code_version=version,
        commit=commit,
        parameters=parameters,
    )
    return {
        "schema": "optbench.result/v2",
        "status": "measured",
        "problem": problem["id"],
        "scoring": "par2-wilson-bootstrap",
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": (
            "Harness counter on the PyAMFF squared-error loss. "
            "One call evaluates that loss and the parameter gradient."
        ),
        "runs": [
            {
                "entry": output.stem,
                "code": template.get("code"),
                "algorithm": template.get("algorithm"),
                "code_version": version,
                "commit": commit,
                "tuning": template.get("tuning", ""),
                "present_source": True,
                "cap": cap,
                "parameters": parameters,
                "provenance": provenance,
                "metrics": metrics,
                "starts": records,
            }
        ],
    }


def _log(text: str) -> None:
    print(text, flush=True)


def train_indices(root: Path, template_id: str, indices: list[int], *, resume: bool = False, output: Path | None = None) -> list[dict]:
    """Train the given start indices of one template."""
    root = Path(root)
    template = load_template(root, template_id)
    problem = resolve_problem(root, str(template["problem"]))
    _check_pin(template)
    prepared = prepare(root, problem, template)
    settings = _settings(template)
    checkpoint = None if output is None else _checkpoint(output)
    return train(
        prepared,
        settings,
        indices,
        checkpoint=checkpoint,
        resume=resume,
        log=_log,
    )


def _check_pin(template: dict) -> str:
    import optax

    version = str(optax.__version__)
    if version != str(template["code_version"]):
        raise RuntimeError(f"optax {version} does not match template {template['code_version']}")
    commit = _installed_commit()
    pinned = str(template["commit"])
    if not commit.startswith(pinned) and not pinned.startswith(commit):
        raise RuntimeError(f"optax commit {commit} does not match template {pinned}")
    return version


def run(
    root: Path,
    problem: str | dict,
    template: dict,
    *,
    output: Path,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    root = Path(root)
    output = Path(output)
    if isinstance(problem, str):
        loaded = resolve_problem(root, problem)
    else:
        loaded = problem
    stated = template.get("problem")
    if stated and stated != loaded["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {loaded['id']}")
    version = _check_pin(template)
    commit = str(template["commit"])
    from optbench.board import stated_full_n

    full_n = stated_full_n(root, loaded)
    if not full_n:
        raise RuntimeError(f"{loaded['id']} does not state a full start count")
    count = full_n if int(limit) <= 0 else min(int(limit), full_n)
    indices = list(range(count))
    prepared = prepare(root, loaded, template)
    settings = _settings(template)
    records = train(
        prepared,
        settings,
        indices,
        checkpoint=_checkpoint(output),
        resume=resume,
        log=_log,
    )
    for record in records:
        record["problem"] = loaded["id"]
        record["template"] = template["id"]
        record["code"] = template.get("code")
        record["code_version"] = version
        record["commit"] = commit
        record["tuning"] = template.get("tuning", "")
    cap = int(template["failure_cap"])
    metrics = score_v2(
        [
            {
                "force_calls": record["force_calls"],
                "success": record["success"],
                "failure": record["failure"],
            }
            for record in records
        ],
        cap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_dat(
            records,
            problem=loaded,
            template=template,
            metrics=metrics,
            full_n=full_n,
            commit=commit,
            version=version,
        ),
        encoding="utf-8",
    )
    write_starts(output.with_suffix(".starts.jsonl"), records)
    document = _document(loaded, template, records, metrics, cap, commit, version, output)
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "force_calls": metrics.get("success_only_mean"),
        "nfailed": metrics["nfailed"],
        "starts": len(records),
        "template": template["id"],
        "problem": loaded["id"],
        "par2_mean": metrics["par2_mean"],
        "v2_nfailed": metrics["nfailed"],
    }
    return summary
