"""Paper-reproduction rows for PyAMFF 8037eb98.

The 2022 archive ran this commit's ``pyamff`` entry point. That process
reads the archived ``config.ini``, builds Behler-Parrinello fingerprints
with this commit's Fortran engine, and trains with this commit's
``Trainer``. Rprop, ADAM, and LBFGS in that trainer are ``torch.optim``
objects constructed by ``training.py``. The epoch written to
``benchmark.dat`` is the integer in the last training line of
``pyamff.log``, which is what ``mlrf0_optb.py`` parsed.

A harness call here is that logged epoch. The stop is the one in the
archived source: every parameter gradient has ``max(abs(grad))`` strictly
below ``loss_grad_tol``. The torch.optim rows at 240f51d stay on their
own templates and stop on the L2 norm instead.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from optbench.score import score_v2, summarize_capped
from optbench.v2 import collect_provenance, write_starts

COMMIT = "8037eb98e2288c132c112aa6c2032f00cd5c5277"
SOURCE = Path(os.environ.get("OPTBENCH_PYAMFF_8037", "/tmp/pyamff-8037eb98"))
SHIM = Path(os.environ.get("OPTBENCH_PYAMFF_8037_SHIM", "/tmp/pyamff-8037-shim"))
WORK_ROOT = Path(os.environ.get("OPTBENCH_PYAMFF_8037_WORK", "/tmp/optbench-pyamff-archive"))
ALGORITHMS = {"Rprop": "Rprop", "ADAM": "Adam", "LBFGS": "LBFGS"}


def archive_tail(text: str) -> tuple[str, str]:
    """Return the two lines ``tail -2 pyamff.log`` gave ``mlrf0_optb.py``.

    The archive script splits that command's output on ``\\n`` and reads
    the epoch from the first piece and the status from the second. A
    trailing newline from ``tail`` becomes an empty last piece, so the
    status is the last non-empty line and the epoch is the line before it.
    """
    if not text.strip():
        raise ValueError("pyamff.log is empty")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if len(lines) < 2:
        raise ValueError("pyamff.log has no status line")
    return lines[-2], lines[-1]


def parse_training_line(epoch_line: str, status_line: str) -> dict:
    """Epoch and RMSE columns from one logged training line."""
    fields = epoch_line.split()
    if len(fields) < 6:
        raise ValueError(f"training line has {len(fields)} fields")
    return {
        "logged_epoch": int(fields[0]),
        "energy_rmse": float(fields[4]),
        "logged_force_rmse": float(fields[5]),
        "status": status_line.strip(),
        "hit_max_epoch": status_line.lstrip().startswith("Max"),
    }


def with_unique_port(text: str, port: int) -> str:
    """Copy an archived config and give this start its own gloo port.

    Every other line stays as archived. The port is not a training
    parameter. Parallel starts cannot share ``12355``.
    """
    if port <= 0 or port > 65535:
        raise ValueError(f"port {port} is outside 1..65535")
    lines = text.splitlines()
    replaced = False
    rewritten: list[str] = []
    for line in lines:
        if line.strip().lower().startswith("master_port"):
            rewritten.append(f"master_port = {port}")
            replaced = True
        else:
            rewritten.append(line)
    if replaced:
        return "\n".join(rewritten) + "\n"
    inserted: list[str] = []
    placed = False
    for line in rewritten:
        inserted.append(line)
        if not placed and line.strip().lower().startswith("master_addr"):
            inserted.append(f"master_port = {port}")
            placed = True
    if not placed:
        inserted.append(f"master_port = {port}")
    return "\n".join(inserted) + "\n"


def _port(template_id: str, index: int) -> int:
    slot = sum(ord(char) for char in template_id) % 400
    return 20000 + slot * 100 + int(index)


def _source_commit(source: Path) -> str:
    if not (source / "pyamff" / "training.py").is_file():
        raise RuntimeError(f"{source} is not a PyAMFF checkout")
    proc = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    sha = proc.stdout.strip()
    if not sha.startswith(COMMIT[:12]):
        raise RuntimeError(f"{source} is {sha}, not {COMMIT}")
    return sha


def _ensure_shim(path: Path) -> None:
    """Let ``import torchvision.models`` succeed.

    ``pyamff/main.py`` imports torchvision and never calls it. The
    training numerics do not use that package.
    """
    package = path / "torchvision"
    package.mkdir(parents=True, exist_ok=True)
    init = package / "__init__.py"
    models = package / "models.py"
    if not init.is_file():
        init.write_text(
            '"""Import shim. PyAMFF 8037eb98 imports torchvision and does not call it."""\n',
            encoding="utf-8",
        )
    if not models.is_file():
        models.write_text("", encoding="utf-8")


def _fortran_engine_ok(log_text: str, stdout: str) -> str | None:
    """Reject the Python fingerprint path. Return an error string or None."""
    blob = f"{log_text}\n{stdout}"
    if "Calculating FPs for Image " in blob or "Proceeding with Python" in blob:
        return "PyAMFF used the Python fingerprint engine"
    if "Calculating FPs for image " not in blob:
        return "PyAMFF log has no Fortran fingerprint line"
    return None


def _launcher() -> str:
    return (
        "import sys\n"
        f"sys.path = [p for p in sys.path if p not in {{'/tmp/pyamff'}}]\n"
        f"sys.path.insert(0, {str(SHIM)!r})\n"
        f"sys.path.insert(0, {str(SOURCE)!r})\n"
        "import pyamff\n"
        "import pyamff.fmodules\n"
        "print('PYAMFF', pyamff.__file__)\n"
        "print('FMODULES', pyamff.fmodules.__file__)\n"
        "from pyamff.main import main\n"
        "main()\n"
    )


def _run_one(job: dict) -> dict:
    """Run one archived config in its own directory."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    index = int(job["index"])
    work = Path(job["work"])
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    config = with_unique_port(Path(job["config"]).read_text(encoding="utf-8"), int(job["port"]))
    (work / "config.ini").write_text(config, encoding="utf-8")
    shutil.copy(job["fp_paras"], work / "fpParas.dat")
    shutil.copy(job["trajectory"], work / job["trajectory_name"])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SHIM), str(SOURCE)])
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", _launcher()],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )
    seconds = time.perf_counter() - started
    log_path = work / "pyamff.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    record = {
        "schema": "optbench.start/v2",
        "start": str(index),
        "index": index,
        "problem": job["problem"],
        "template": job["template"],
        "code": job["code"],
        "code_version": job["code_version"],
        "commit": job["commit"],
        "algorithm": job["algorithm"],
        "tuning": job["tuning"],
        "counter": "harness",
        "comparable": True,
        "seconds": seconds,
        "returncode": int(proc.returncode),
        "port": int(job["port"]),
    }
    if "PYAMFF " in stdout:
        record["pyamff_file"] = stdout.split("PYAMFF ", 1)[1].splitlines()[0].strip()
    if "FMODULES " in stdout:
        record["fmodules_file"] = stdout.split("FMODULES ", 1)[1].splitlines()[0].strip()
    cap = int(job["cap"])
    coeff = float(job["force_coefficient"])
    try:
        parsed = parse_training_line(*archive_tail(log_text))
    except (ValueError, IndexError) as exc:
        record.update(
            {
                "success": False,
                "failure": "crash",
                "force_calls": cap,
                "energy_rmse": None,
                "force_rmse": -1.0,
                "logged_epoch": None,
                "status": "",
                "error": f"{exc}; returncode {proc.returncode}",
            }
        )
        _store(job, record, log_text, stdout, stderr)
        shutil.rmtree(work, ignore_errors=True)
        return record
    engine_error = _fortran_engine_ok(log_text, stdout)
    epoch = int(parsed["logged_epoch"])
    energy = float(parsed["energy_rmse"])
    logged_force = float(parsed["logged_force_rmse"])
    finite = math.isfinite(energy) and math.isfinite(logged_force)
    if proc.returncode != 0 or engine_error or not finite:
        failure = "divergence" if not finite or "nan" in (stderr + stdout).lower() else "crash"
        record.update(
            {
                "success": False,
                "failure": failure,
                "force_calls": epoch if epoch > 0 else cap,
                "energy_rmse": energy if finite else None,
                "force_rmse": logged_force if coeff > 0 and finite else -1.0,
                "logged_epoch": epoch,
                "logged_force_rmse": logged_force,
                "status": parsed["status"],
                "error": engine_error or f"returncode {proc.returncode}",
            }
        )
        _store(job, record, log_text, stdout, stderr)
        shutil.rmtree(work, ignore_errors=True)
        return record
    hit_max = bool(parsed["hit_max_epoch"]) or epoch >= cap
    record.update(
        {
            "success": not hit_max,
            "failure": "cap" if hit_max else None,
            "force_calls": epoch,
            "energy_rmse": energy,
            "force_rmse": logged_force if coeff > 0 else -1.0,
            "logged_epoch": epoch,
            "logged_force_rmse": logged_force,
            "status": parsed["status"],
            "hit_max_epoch": hit_max,
        }
    )
    _store(job, record, log_text, stdout, stderr)
    shutil.rmtree(work, ignore_errors=True)
    return record


def _store(job: dict, record: dict, log_text: str, stdout: str, stderr: str) -> None:
    path = Path(job["record_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".log").write_text(log_text, encoding="utf-8")
    tail = path.with_suffix(".out")
    tail.write_text(stdout[-4000:] + "\n" + stderr[-4000:], encoding="utf-8")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fmt(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _mean(values: list[float]):
    return sum(values) / len(values) if values else None


def _median(values: list[float]):
    return float(statistics.median(values)) if values else None


def archive_aggregates(records: list[dict]) -> dict[str, float | int | None]:
    """Means the archive script wrote: every parsed log, failures included."""
    parsed = [row for row in records if row.get("logged_epoch") is not None]
    epochs = [float(row["logged_epoch"]) for row in parsed]
    energy = [float(row["energy_rmse"]) for row in parsed if row.get("energy_rmse") is not None]
    force = [float(row["logged_force_rmse"]) for row in parsed if row.get("logged_force_rmse") is not None]
    nfailed = sum(1 for row in parsed if str(row.get("status") or "").lstrip().startswith("Max"))
    return {
        "archive_mean_epoch": _mean(epochs),
        "archive_median_epoch": _median(epochs),
        "archive_mean_energy_rmse": _mean(energy),
        "archive_median_energy_rmse": _median(energy),
        "archive_mean_force_rmse": _mean(force),
        "archive_median_force_rmse": _median(force),
        "archive_nfailed": nfailed,
        "archive_parsed": len(parsed),
    }


def _write_dat(path: Path, template: dict, problem: dict, records: list[dict], cap: int, full_n: int) -> dict:
    legacy_counts = [int(row["force_calls"]) if row["success"] else cap for row in records]
    legacy = summarize_capped(legacy_counts, cap)
    metrics = score_v2(
        [
            {
                "force_calls": row["force_calls"],
                "success": row["success"],
                "failure": row["failure"],
            }
            for row in records
        ],
        cap,
    )
    successes = [row for row in records if row["success"]]
    epochs = [float(row["force_calls"]) for row in successes]
    energy = [float(row["energy_rmse"]) for row in successes if row.get("energy_rmse") is not None]
    force = [float(row["force_rmse"]) for row in successes if row.get("force_rmse") is not None]
    lines = []
    for key in ("force_calls", "force_calls_min", "force_calls_max"):
        text = _fmt(legacy.get(key))
        if text is not None:
            lines.append(f"{key} {text}")
    if epochs:
        lines.append(f"force_calls_median {_fmt(_median(epochs))}")
    lines.append(f"nfailed {metrics['nfailed']}")
    lines.append(f"starts {len(records)}")
    lines.append(f"code {template.get('code', 'pyamff-8037eb98')}")
    lines.append(f"code_version {template.get('code_version', '8037eb98')}")
    lines.append(f"commit {template.get('commit', COMMIT)}")
    lines.append(f"algorithm {template.get('algorithm', '')}")
    lines.append(f"tuning {template.get('tuning', 'paper')}")
    lines.append(f"template {template['id']}")
    lines.append(f"problem {problem['id']}")
    lines.append("counter harness")
    lines.append(f"cap {cap}")
    for key in (
        "par2_mean",
        "median",
        "median_low",
        "median_high",
        "wilson_low",
        "wilson_high",
        "n_cap",
        "n_divergence",
        "n_wrong_stationary",
        "n_crash",
    ):
        text = _fmt(metrics.get(key))
        if text is not None:
            lines.append(f"{key} {text}")
    if epochs:
        lines.append(f"mean_epoch {_fmt(_mean(epochs))}")
        lines.append(f"median_epoch {_fmt(_median(epochs))}")
    if energy:
        lines.append(f"mean_energy_rmse {_fmt(_mean(energy))}")
        lines.append(f"median_energy_rmse {_fmt(_median(energy))}")
    if force:
        lines.append(f"mean_force_rmse {_fmt(_mean(force))}")
        lines.append(f"median_force_rmse {_fmt(_median(force))}")
    for key, value in archive_aggregates(records).items():
        text = _fmt(value)
        if text is not None:
            lines.append(f"{key} {text}")
    if len(records) < full_n:
        lines.append("partial true")
    lines.append(
        "note Paper reproduction of PyAMFF 8037eb98. The logged epoch is one "
        "Trainer iteration. Rprop and ADAM log that iteration after one "
        "optimizer step and one full-batch loss gradient. The stop is "
        "max(abs(parameter gradient)) below loss_grad_tol. Fingerprints are "
        "this commit's Fortran engine."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return metrics


def _publish(output: Path, template: dict, problem: dict, records: list[dict], cap: int, full_n: int) -> dict:
    ordered = sorted(records, key=lambda row: int(row["start"]))
    metrics = _write_dat(output, template, problem, ordered, cap, full_n)
    public = []
    for row in ordered:
        item = dict(row)
        item.pop("index", None)
        public.append(item)
    write_starts(output.with_suffix(".starts.jsonl"), public)
    provenance = collect_provenance(
        code=str(template.get("code") or "pyamff-8037eb98"),
        code_version=str(template.get("code_version") or "8037eb98"),
        commit=str(template.get("commit") or COMMIT),
        parameters={
            "optimizer": template.get("algorithm"),
            "archive": template.get("archive"),
            "pyamff": template.get("pyamff") or {},
            "stop": template.get("stop") or {},
        },
    )
    document = {
        "schema": "optbench.result/v2",
        "problem": problem["id"],
        "template": template["id"],
        "scoring": "par2-wilson-bootstrap",
        "cap": cap,
        "provenance": provenance,
        "metrics": metrics,
        "runs": [{"starts": public, "metrics": metrics}],
    }
    v2_path = output.with_suffix(".v2.json")
    temporary = v2_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(v2_path)
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": metrics.get("success_only_mean"),
        "nfailed": metrics["nfailed"],
        "starts": len(public),
        "par2_mean": metrics.get("par2_mean"),
    }


def _full_n(problem: dict) -> int:
    sample = problem.get("sample") or {}
    value = sample.get("initializations")
    if value in (None, "", 0, "0"):
        raise ValueError(f"{problem['id']} has no sample.initializations")
    return int(value)


def _stop(problem: dict, template: dict) -> tuple[int, float, float]:
    section = template.get("pyamff") or {}
    stop = template.get("stop") or {}
    cap = int(stop["max_steps"])
    tol = float(stop["gradient_norm"])
    coeff = float(section["force_coefficient"])
    convergence = problem.get("convergence") or {}
    if coeff == 0.0:
        stated_tol = convergence.get("energy_only_grad_norm")
        stated_cap = convergence.get("max_epochs")
    else:
        stated_tol = convergence.get("force_training_grad_norm")
        stated_cap = convergence.get("force_max_epochs")
    if stated_tol not in (None, "") and abs(float(stated_tol) - tol) > 1e-12:
        raise ValueError(f"{template['id']} gradient_norm is not the problem stop")
    if stated_cap not in (None, "") and int(stated_cap) != cap:
        raise ValueError(f"{template['id']} max_steps is not the problem epoch cap")
    return cap, tol, coeff


def _record_dir(output: Path) -> Path:
    return WORK_ROOT / "records" / output.stem


def _load_done(output: Path) -> dict[int, dict]:
    found: dict[int, dict] = {}
    directory = _record_dir(output)
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            found[int(row["start"])] = row
    return found


def _trajectory_name(config_text: str) -> str:
    for line in config_text.splitlines():
        if line.strip().lower().startswith("trajectory_file"):
            return line.split("=", 1)[1].strip()
    raise ValueError("archived config has no Trajectory_file")


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    work: Path | None = None,
    limit: int = 0,
    resume: bool = False,
) -> dict:
    """Train every start with the archived PyAMFF 8037eb98 loop."""
    del work
    _source_commit(SOURCE)
    _ensure_shim(SHIM)
    module = SOURCE / "pyamff" / "fmodules.cpython-312-x86_64-linux-gnu.so"
    if not module.is_file() and not list((SOURCE / "pyamff").glob("fmodules*.so")):
        raise RuntimeError(
            "PyAMFF 8037eb98 has no fmodules extension. Build it with f2py and gfortran."
        )
    cap, _tol, coeff = _stop(problem, template)
    full_n = _full_n(problem)
    count = full_n if limit <= 0 else min(full_n, int(limit))
    archive = root / "archive" / "entries" / str(template["archive"])
    config_path = archive / "config.ini"
    paras_path = archive / "fpParas.dat"
    if not config_path.is_file() or not paras_path.is_file():
        raise FileNotFoundError(f"{template['id']} archive entry is incomplete")
    from optbench.campaign import geometry_archive

    trajectory = geometry_archive(root, problem)
    trajectory_name = _trajectory_name(config_path.read_text(encoding="utf-8"))
    done = _load_done(output) if resume else {}
    directory = _record_dir(output)
    pending = []
    for index in range(count):
        if index in done:
            continue
        pending.append(
            {
                "index": index,
                "port": _port(str(template["id"]), index),
                "work": str(WORK_ROOT / "work" / output.stem / f"{index:04d}"),
                "config": str(config_path),
                "fp_paras": str(paras_path),
                "trajectory": str(trajectory),
                "trajectory_name": trajectory_name,
                "record_path": str(directory / f"{index:04d}.json"),
                "cap": cap,
                "force_coefficient": coeff,
                "problem": problem["id"],
                "template": template["id"],
                "code": str(template.get("code") or "pyamff-8037eb98"),
                "code_version": str(template.get("code_version") or "8037eb98"),
                "commit": str(template.get("commit") or COMMIT),
                "algorithm": str(template.get("algorithm") or ""),
                "tuning": str(template.get("tuning") or "paper"),
            }
        )
    if pending:
        if workers <= 1:
            fresh = [_run_one(job) for job in pending]
            _publish(output, template, problem, list(done.values()) + fresh, cap, full_n)
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_run_one, job) for job in pending]
                fresh = []
                total = len(futures)
                for finished, future in enumerate(as_completed(futures), 1):
                    fresh.append(future.result())
                    _publish(output, template, problem, list(done.values()) + fresh, cap, full_n)
                    print(f"{template['id']} {finished}/{total} new starts", flush=True)
        for row in fresh:
            done[int(row["start"])] = row
    records = [done[index] for index in range(count) if index in done]
    if len(records) != count:
        missing = [index for index in range(count) if index not in done]
        raise RuntimeError(f"{template['id']} missing starts {missing[:8]}")
    return _publish(output, template, problem, records, cap, full_n)
