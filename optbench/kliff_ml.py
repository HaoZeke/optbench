"""Train the problem network with a KLIFF PyTorch optimizer.

KLIFF's ``LossNeuralNetworkModel`` names the optimizers and builds them
with ``getattr(torch.optim, method)(parameters, **kwargs)``. The closure
passed to ``optimizer.step`` evaluates the problem squared-error loss and
writes the parameter gradient. One entry into that evaluation is one
``HarnessCounter`` call. A repeated parameter vector is a cache hit.

The search stops when the infinity norm of the parameter gradient is at
or below the template stop, when the next evaluation would pass the call
budget, or when the optimizer-step budget is exhausted. Success requires
a finite loss, that gradient norm, and a call count strictly under the
budget. A non-finite loss or gradient is divergence. An exception is a
crash. The parameter-gradient stop is the only stationary test, so
``wrong_stationary`` stays at 0.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import statistics
from pathlib import Path

from optbench.counter import CountedPotential, ForceCallCap
from optbench.failures import FAILURES
from optbench.ml_loss import (
    ACTIVATION,
    ENERGY_COEFFICIENT,
    HIDDEN,
    MISSING_FORCE_RMSE,
    SLOPE,
    ensure_features,
    parameter_count,
    parse_fp,
)
from optbench.score import score_v2

SEED_BASE = 20261015
KIM_LIB = "/home/ubuntu/kim-api/lib"


class SearchStop(Exception):
    """The evaluation met a stop rule. The call is already counted."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def _ensure_kim() -> None:
    lib = os.environ.get("KIM_API_LIB", KIM_LIB)
    if not Path(lib).is_dir():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(":") if part]
    if lib not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([lib, *parts])


def _kliff_loss():
    _ensure_kim()
    # Importing KLIFF opens kliff.log in the working directory.
    previous = Path.cwd()
    os.chdir("/tmp")
    try:
        from kliff.legacy.loss import LossNeuralNetworkModel
    finally:
        os.chdir(previous)
    return LossNeuralNetworkModel


def _hidden(template: dict) -> tuple[int, int]:
    raw = (template.get("model") or {}).get("hidden")
    if raw is None:
        raise ValueError(f"{template['id']} needs model.hidden")
    hidden = tuple(int(item) for item in raw)
    if hidden != HIDDEN:
        raise ValueError(f"{template['id']} hidden layers are {hidden}")
    activation = str((template.get("model") or {}).get("activation") or "")
    if activation != ACTIVATION:
        raise ValueError(f"{template['id']} activation is {activation}")
    coefficient = float((template.get("model") or {}).get("energy_coefficient"))
    if coefficient != ENERGY_COEFFICIENT:
        raise ValueError(f"{template['id']} energy coefficient is {coefficient}")
    return hidden


def _expect_stop(problem: dict, force_coefficient: float) -> tuple[float, int]:
    convergence = problem.get("convergence") or {}
    ident = problem["id"]
    if ident == "mlopt/emt-pt38":
        cap = int(convergence["max_epochs"])
        if force_coefficient == 0.0:
            return float(convergence["energy_only_grad_norm"]), cap
        return float(convergence["force_training_grad_norm"]), cap
    if ident == "mlopt/C2H4":
        if force_coefficient == 0.0:
            return float(convergence["energy_only_grad_norm"]), int(convergence["energy_only_max_epochs"])
        return float(convergence["force_training_grad_norm"]), int(convergence["force_training_max_epochs"])
    raise ValueError(f"{ident} has no KLIFF loss stop")


def _settings(template: dict, problem: dict) -> dict:
    section = template.get("kliff") or {}
    if "method" not in section or "force_coefficient" not in section:
        raise ValueError(f"{template['id']} needs kliff.method and kliff.force_coefficient")
    method = str(section["method"])
    force_coefficient = float(section["force_coefficient"])
    stop = template.get("stop") or {}
    grad_tol = float(stop["gradient_norm"])
    cap = int(stop["max_force_calls"])
    expect_grad, expect_cap = _expect_stop(problem, force_coefficient)
    if abs(grad_tol - expect_grad) > 1e-12 or cap != expect_cap:
        raise ValueError(f"{template['id']} stop does not match {problem['id']}")
    learning_rate = section.get("learning_rate")
    hidden = _hidden(template)
    fp_path = template.get("fp_parameters")
    if not fp_path:
        raise ValueError(f"{template['id']} needs fp_parameters")
    return {
        "method": method,
        "force_coefficient": force_coefficient,
        "grad_tol": grad_tol,
        "cap": cap,
        "max_epochs": cap,
        "learning_rate": None if learning_rate in (None, "") else float(learning_rate),
        "hidden": hidden,
        "seed_base": int(section.get("seed_base", SEED_BASE)),
        "fp_parameters": str(fp_path),
    }


def _element_index(symbols: list[str], elements: tuple[str, ...]) -> dict[str, list[int]]:
    groups = {element: [] for element in elements}
    for index, symbol in enumerate(symbols):
        if symbol not in groups:
            raise ValueError(f"atom {symbol} has no fingerprint")
        groups[symbol].append(index)
    return groups


def _build_model(elements: tuple[str, ...], width: int, hidden: tuple[int, int]):
    import torch
    from torch import nn

    class ElementNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            first, second = hidden
            self.fc1 = nn.Linear(width, first, dtype=torch.float64)
            self.fc2 = nn.Linear(first, second, dtype=torch.float64)
            self.out = nn.Linear(second, 1, dtype=torch.float64)

        def forward(self, flat):
            hidden_state = torch.tanh(self.fc1(flat))
            hidden_state = torch.tanh(self.fc2(hidden_state))
            return self.out(hidden_state).squeeze(-1)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nets = nn.ModuleDict({element: ElementNet() for element in elements})

        def energies(self, fps, groups):
            total = fps.new_zeros(fps.shape[0])
            for element, index in groups.items():
                block = fps[:, index, :]
                pred = self.nets[element](block.reshape(-1, block.shape[-1]))
                total = total + pred.reshape(block.shape[0], block.shape[1]).sum(dim=1)
            return total

    return Model()


def _init_uniform(model, seed: int) -> None:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for param in model.parameters():
            draw = torch.rand(param.shape, generator=generator, dtype=torch.float64)
            param.copy_(2.0 * draw - 1.0)


def _flatten(params) -> list[float]:
    import torch

    if not params:
        return []
    flat = torch.cat([param.detach().reshape(-1) for param in params])
    return flat.tolist()


def _scatter(params, values) -> None:
    import torch

    flat = torch.as_tensor(values, dtype=torch.float64)
    offset = 0
    with torch.no_grad():
        for param in params:
            count = param.numel()
            param.copy_(flat[offset : offset + count].reshape(param.shape))
            offset += count


def _write_grad(params, values) -> None:
    import torch

    flat = torch.as_tensor(values, dtype=torch.float64)
    offset = 0
    for param in params:
        count = param.numel()
        param.grad = flat[offset : offset + count].reshape(param.shape).clone()
        offset += count


def _finite(value: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def train_start(features: dict, job: dict) -> dict:
    """One initialization. ``features`` holds the cached fingerprints."""
    import torch

    torch.set_num_threads(1)
    loss_type = _kliff_loss()
    method = job["method"]
    if method not in loss_type.torch_minimize_methods:
        raise ValueError(f"{method} is not a KLIFF neural-network optimizer")
    symbols = [str(symbol) for symbol in features["symbols"].tolist()]
    elements = tuple(job["elements"])
    groups = {element: index for element, index in _element_index(symbols, elements).items()}
    model = _build_model(elements, int(job["width"]), tuple(job["hidden"]))
    _init_uniform(model, int(job["seed"]))
    params = list(model.parameters())
    expected = sum(parameter_count(int(job["width"]), tuple(job["hidden"])) for _ in elements)
    if sum(param.numel() for param in params) != expected:
        raise RuntimeError("parameter count drifted")
    kwargs = {}
    if job["learning_rate"] is not None:
        kwargs["lr"] = float(job["learning_rate"])
    optimizer = getattr(torch.optim, method)(params, **kwargs)
    fps = torch.tensor(features["fps"], dtype=torch.float64)
    ref_energy = torch.tensor(features["energy"], dtype=torch.float64)
    force_coefficient = float(job["force_coefficient"])
    jac = None
    ref_forces = None
    if force_coefficient != 0.0:
        jac = torch.tensor(features["jacobian"], dtype=torch.float64)
        ref_forces = torch.tensor(features["forces"], dtype=torch.float64)
    n_atoms = int(fps.shape[1])
    n_images = int(fps.shape[0])
    energy_coefficient = float(job["energy_coefficient"])
    grad_tol = float(job["grad_tol"])
    cap = int(job["cap"])
    holder: dict = {}
    state: dict = {}

    def evaluate(coords):
        _scatter(params, coords)
        model.zero_grad(set_to_none=True)
        active = fps.detach().requires_grad_(force_coefficient != 0.0)
        energies = model.energies(active, groups)
        energyloss = ((energies - ref_energy) / n_atoms).pow(2).sum()
        if force_coefficient == 0.0:
            forceloss = energies.new_zeros(())
            loss = energy_coefficient * energyloss
        else:
            d_energy = torch.autograd.grad(energies.sum(), active, create_graph=True)[0]
            forces = -SLOPE * torch.einsum("iaf,iafbk->ibk", d_energy, jac)
            per_image = (forces - ref_forces).pow(2).sum(dim=(1, 2)) / (3.0 * n_atoms)
            forceloss = per_image.sum()
            loss = energy_coefficient * energyloss + force_coefficient * forceloss
        loss.backward()
        grad_inf = 0.0
        pieces = []
        for param in params:
            if param.grad is None:
                raise SearchStop("crash")
            grad_inf = max(grad_inf, float(param.grad.detach().abs().max()))
            pieces.append(param.grad.detach().reshape(-1))
        flat_grad = torch.cat(pieces)
        loss_value = float(loss.detach())
        energy_rmse = float(torch.sqrt(energyloss.detach() / n_images))
        if force_coefficient == 0.0:
            force_rmse = MISSING_FORCE_RMSE
        else:
            force_rmse = float(torch.sqrt(forceloss.detach() / n_images))
        holder["loss"] = loss_value
        holder["grad_norm"] = grad_inf
        holder["energy_rmse"] = energy_rmse
        holder["force_rmse"] = force_rmse
        calls = int(state["potential"].calls)
        if not math.isfinite(loss_value) or not math.isfinite(grad_inf):
            raise SearchStop("divergence")
        if grad_inf <= grad_tol and calls < cap:
            raise SearchStop("success")
        if calls >= cap:
            raise SearchStop("cap")
        return loss_value, tuple(float(item) for item in flat_grad.tolist())

    state["potential"] = CountedPotential(evaluate)
    state["potential"]._counter.cap = cap

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss_value, grad = state["potential"].energy_forces(_flatten(params))
        _write_grad(params, grad)
        return torch.tensor(loss_value, dtype=torch.float64)

    epochs = 0
    kind = None
    error = ""
    try:
        while epochs < int(job["max_epochs"]):
            optimizer.step(closure)
            epochs += 1
    except SearchStop as exc:
        kind = exc.kind
    except ForceCallCap:
        kind = "cap"
    except Exception as exc:
        kind = "crash"
        error = f"{type(exc).__name__}: {exc}"
    calls = int(state["potential"].calls)
    if kind is None:
        kind = "cap" if calls >= cap else "divergence"
    success = kind == "success" and calls < cap
    if not success and kind == "success":
        kind = "cap"
    failure = None if success else kind
    if failure not in (None, *FAILURES):
        failure = "crash"
    record = {
        "schema": "optbench.start/v2",
        "start": str(job["index"]),
        "seed": int(job["seed"]),
        "force_calls": calls,
        "success": success,
        "failure": failure,
        "counter": "harness",
        "comparable": True,
        "grad_norm": _finite(holder.get("grad_norm")),
        "force_norm": _finite(holder.get("grad_norm")),
        "energy_rmse": _finite(holder.get("energy_rmse")),
        "force_rmse": holder.get("force_rmse", MISSING_FORCE_RMSE),
        "loss": _finite(holder.get("loss")),
        "epochs": epochs,
        "reasons": [] if success else [failure],
    }
    if error:
        record["error"] = error
    return record


def _train_job(job: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _ensure_kim()
    import numpy as np

    stored = np.load(job["cache"], allow_pickle=False)
    features = {"fps": stored["fps"], "energy": stored["energy"], "symbols": stored["symbols"]}
    if float(job["force_coefficient"]) != 0.0:
        features["jacobian"] = stored["jacobian"]
        features["forces"] = stored["forces"]
    record = train_start(features, job)
    print(
        f"{job['template']} start {job['index']} calls {record['force_calls']} "
        f"{'ok' if record['success'] else record['failure']}",
        flush=True,
    )
    return record


def _format(value: object) -> str:
    if isinstance(value, float):
        return format(value, ".16g")
    return str(value)


def _dat(template: dict, settings: dict, metrics: dict, records: list[dict], *, partial: bool) -> str:
    calls = [float(row["force_calls"]) for row in records]
    lines = [
        "# counter harness. One force call is one loss and parameter-gradient evaluation.",
        "# The loss is the pyamff squared-error sum on Behler-Parrinello fingerprints.",
        "# The network is 10-7 tanh, one net per element, uniform[-1, 1) at the start seed.",
        "# Success is a finite loss and a parameter-gradient infinity norm at or below the stop, with the call count under the cap.",
        "# force_norm on a start is that infinity norm. wrong_stationary is unused.",
        "# A force coefficient of 0 reports force_rmse -1 and leaves forces out of the graph.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    rows = {
        "force_calls": statistics.fmean(calls),
        "force_calls_min": min(calls),
        "force_calls_max": max(calls),
        "nfailed": metrics["nfailed"],
        "median": statistics.median(calls),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "problem": template.get("problem"),
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "cap": settings["cap"],
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
        "grad_norm": settings["grad_tol"],
        "force_coefficient": settings["force_coefficient"],
    }
    if settings["learning_rate"] is not None:
        rows["learning_rate"] = settings["learning_rate"]
    if partial:
        rows["partial"] = "true"
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format(value)}")
    return "\n".join(lines) + "\n"


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: int(row["start"]))
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
    path.write_text(body, encoding="utf-8")


def _load_done(path: Path) -> dict[int, dict]:
    found: dict[int, dict] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[int(row["start"])] = row
    return found


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None = None,
    output: Path,
    workers: int,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Run every initialization and write the campaign files."""
    del geometries, work
    settings = _settings(template, problem)
    loss_type = _kliff_loss()
    if settings["method"] not in loss_type.torch_minimize_methods:
        raise ValueError(f"{settings['method']} is not a KLIFF neural-network optimizer")
    spec = parse_fp(root / settings["fp_parameters"])
    full_n = int((problem.get("sample") or {})["initializations"])
    count = full_n if limit <= 0 else min(full_n, int(limit))
    built = ensure_features(
        root,
        problem,
        spec,
        need_jacobian=settings["force_coefficient"] != 0.0,
    )
    del built
    from .ml_loss import cache_file

    cache = cache_file(problem["id"])
    starts_path = output.with_suffix(".starts.jsonl")
    done = _load_done(starts_path) if resume else {}
    jobs = []
    for index in range(count):
        if index in done:
            continue
        jobs.append(
            {
                "cache": str(cache),
                "index": index,
                "seed": settings["seed_base"] + index,
                "template": template["id"],
                "method": settings["method"],
                "learning_rate": settings["learning_rate"],
                "force_coefficient": settings["force_coefficient"],
                "energy_coefficient": ENERGY_COEFFICIENT,
                "grad_tol": settings["grad_tol"],
                "cap": settings["cap"],
                "max_epochs": settings["max_epochs"],
                "hidden": list(settings["hidden"]),
                "elements": list(spec.elements),
                "width": spec.width,
            }
        )
    records = [done[index] for index in range(count) if index in done]
    if jobs:
        workers = max(1, min(int(workers), len(jobs)))
        if workers == 1:
            for job in jobs:
                records.append(_train_job(job))
                _write_records(starts_path, records)
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(workers) as pool:
                for record in pool.imap_unordered(_train_job, jobs):
                    records.append(record)
                    _write_records(starts_path, records)
    records = sorted(records, key=lambda row: int(row["start"]))
    if len(records) != count:
        raise RuntimeError(f"{template['id']} wrote {len(records)} of {count} starts")
    metrics = score_v2(records, settings["cap"])
    partial = count < full_n
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(template, settings, metrics, records, partial=partial), encoding="utf-8")
    _write_records(starts_path, records)
    document = {
        "schema": "optbench.result/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "cap": settings["cap"],
        "partial": partial,
        "metrics": {key: metrics[key] for key in metrics if key != "n"},
        "n": metrics["n"],
        "starts": records,
    }
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "template": template["id"],
        "problem": problem["id"],
        "n": metrics["n"],
        "successes": metrics["successes"],
        "par2_mean": metrics["par2_mean"],
        "median": metrics["median"],
        "nfailed": metrics["nfailed"],
        "partial": partial,
        "failures": {name: metrics[f"n_{name}"] for name in FAILURES},
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    from optbench.campaign import load_template, resolve_problem

    parser = argparse.ArgumentParser(description="Run one KLIFF mlopt template")
    parser.add_argument("--template", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    template = load_template(root, args.template)
    problem = resolve_problem(root, template["problem"])
    if args.output is None:
        slug = template["id"].split("/", 1)[1]
        commit = str(template["commit"])[:12]
        args.output = root / "results" / "campaign" / f"kliff-{commit}-{slug}.dat"
    summary = run_template(
        root,
        problem,
        template,
        output=args.output,
        workers=args.workers,
        limit=args.limit,
        resume=args.resume,
    )
    print(
        f"{summary['template']} n={summary['n']} successes={summary['successes']} "
        f"par2={summary['par2_mean']} median={summary['median']} partial={summary['partial']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
