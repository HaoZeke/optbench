"""Native GPRidge bands on the archived Pt endpoint pairs.

The native provider tally is a client counter, not a HarnessCounter.
Search and verification costs remain separate. The final energy, gradient,
fixed atoms and Hessian index are checked against the problem surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, load_template, resolve_problem
from .engines.common import band_settings, require_pin, registry_pin, run_jobs, write_campaign
from .failures import classify_failure
from .pes import load_pairs, morse_pt
from .rgsaddle_band import vibrational_index


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime(template: dict) -> dict:
    """Resolve the executable through a verified native-validation manifest."""
    binary = Path(os.environ["GPRIDGE_GPRN"]).resolve(strict=True)
    manifest = Path(os.environ["GPRIDGE_VALIDATION"]).resolve(strict=True)
    result = json.loads((manifest / "result.json").read_text())
    if result.get("exit_code") != 0:
        raise ValueError("native validation did not succeed")
    provenance = json.loads((manifest / "provenance.json").read_text())
    revision = provenance.get("source_revision")
    if not revision and (manifest / "source-provenance.json").is_file():
        revision = json.loads((manifest / "source-provenance.json").read_text())["source_revision"]
    if revision != template["commit"]:
        raise ValueError(f"runtime revision {revision} does not match {template['commit']}")
    expected = {
        Path(line.split(maxsplit=1)[1]).name: line.split(maxsplit=1)[0]
        for line in (manifest / "programs.sha256").read_text().splitlines()
    }
    checksum = digest(binary)
    if expected.get("gprn") != checksum:
        raise ValueError("gprn does not match the native-validation binary")
    return {"binary": str(binary), "sha256": checksum, "revision": revision,
            "manifest": str(manifest)}


def command(binary: str, directory: Path, settings: dict, options: dict) -> list[str]:
    """Explicit physical and numerical settings for a Pt band."""
    return [
        binary, "--reactant", str(directory / "reactant.con"),
        "--product", str(directory / "product.con"),
        "--config", str(directory / "potential.toml"),
        "--nimages", str(settings["n_interior"]),
        "--nebmode", str(options.get("mode", "gp_oie")),
        "--converged-force", str(settings["force_norm"]),
        "--convergence-force-norm", "l2",
        "--max-iterations", str(settings["cap"]),
        "--neb-lbfgs-max-move", str(settings["max_move"]),
        "--climbing-image", "true", "--climbing-image-converged-only", "false",
        "--spring-constant", "5", "--spring-constant-min", "5",
        "--idpp", "false", "--sidpp", "false", "--endpoint-assign", "false",
        "--gp-nonperiodic", "false", "--gp-zbl-prior-mean", "false",
        "--gp-reactive-types", "false",
        "--gp-environment-types", str(directory / "environment-types.txt"),
        "--linalg-threads", str(options.get("threads", 1)),
        "--linalg-mode", "reproducible",
        "--output", str(directory / "band.h5"),
        "--write-con", str(directory / "band.con"),
        "--gp-observation-archive", str(directory / "observations"),
    ]


def write_inputs(directory: Path, pair) -> None:
    """ASE's CON writer preserves the cell and the fixed-atom mask."""
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.io import write

    directory.mkdir(parents=True, exist_ok=False)
    for name, positions in (("reactant", pair.reactant), ("product", pair.product)):
        atoms = Atoms("Pt" * len(positions), positions=positions, cell=pair.box, pbc=True)
        atoms.set_constraint(FixAtoms(mask=~pair.free))
        write(directory / f"{name}.con", atoms, format="eon")
    (directory / "potential.toml").write_text(
        '[oracle]\npotential = "morse"\n'
        '[oracle.parameters.morse]\nde = 0.7102\na = 1.6047\nre = 2.897\ncutoff = 9.5\n'
    )
    (directory / "environment-types.txt").write_text(
        "".join(f"{index} 78 0\n" for index in range(len(pair.reactant)))
    )


def parse_calls(log: str) -> tuple[int | None, int | None]:
    """Read the reconciled physical evaluation count, including endpoints."""
    search = re.findall(r"^Oracle calls \(to convergence\): (\d+)", log, re.MULTILINE)
    total = re.findall(r"^Oracle calls \(with validation\): (\d+)", log, re.MULTILINE)
    return (int(search[-1]) if search else None, int(total[-1]) if total else None)


def verify_band(path: Path, pair, settings: dict) -> dict:
    """Check the actual native output on the independent Morse surface.

    The HDF5 coordinates contain the moving degrees of freedom; the full
    substrate is restored from the endpoint before evaluating the surface.
    Hessian differences displace only the free atoms.
    """
    import h5py

    verification_calls = 0

    def evaluate(point):
        nonlocal verification_calls
        verification_calls += 1
        energy, gradient = morse_pt(point, pair.box)
        if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
            raise ValueError("non-finite independent energy or gradient")
        if np.shape(gradient) != point.shape:
            raise ValueError("independent gradient has an unexpected dimension")
        return energy, gradient

    with h5py.File(path, "r") as handle:
        raw_index = float(np.asarray(handle["summary/max_energy_image"]).item())
        if not np.isfinite(raw_index) or not raw_index.is_integer():
            raise ValueError("invalid maximum-energy image index")
        index = int(raw_index)
        if not 0 < index < settings["n_interior"] + 1:
            raise ValueError("reported saddle is not an interior image")
        image = handle[f"converged/{index}"]
        raw = np.asarray(image["positions"]).reshape(-1, 3)
        point = np.array(pair.reactant, copy=True)
        if len(raw) == len(point):
            if not np.array_equal(raw[~pair.free], point[~pair.free]):
                raise ValueError("fixed substrate atoms moved")
            point = raw
        elif len(raw) == int(pair.free.sum()):
            point[pair.free] = raw
        else:
            raise ValueError("native result has an unexpected coordinate dimension")
        gradient = np.asarray(image["gradient"]).reshape(-1, 3)
        if len(gradient) == len(point):
            gradient = gradient[pair.free]
        native_energy = float(np.asarray(image["energy"]).item())
    if not np.all(np.isfinite(point)):
        raise ValueError("non-finite native geometry")
    if gradient.shape != (int(pair.free.sum()), 3):
        raise ValueError("native gradient has an unexpected coordinate dimension")
    if not np.isfinite(native_energy) or not np.all(np.isfinite(gradient)):
        raise ValueError("non-finite native energy or gradient")
    energy, full_gradient = evaluate(point)
    free_gradient = full_gradient[pair.free]
    energy_error = abs(energy - native_energy)
    gradient_error = float(np.max(np.abs(free_gradient - gradient)))
    reasons = []
    if energy_error > 1e-7 or gradient_error > 1e-7:
        reasons.append("potential_mismatch")
    force = float(np.linalg.norm(free_gradient))
    index = None
    if force > settings["force_norm"]:
        reasons.append("force_norm")
    elif not reasons:
        step = 1e-5
        free_positions = point[pair.free].reshape(-1)
        hessian = np.empty((len(free_positions), len(free_positions)))
        for column in range(len(free_positions)):
            plus, minus = free_positions.copy(), free_positions.copy()
            plus[column] += step
            minus[column] -= step
            upper, lower = point.copy(), point.copy()
            upper[pair.free] = plus.reshape(-1, 3)
            lower[pair.free] = minus.reshape(-1, 3)
            hessian[:, column] = (
                evaluate(upper)[1][pair.free].reshape(-1)
                - evaluate(lower)[1][pair.free].reshape(-1)
            ) / (2 * step)
        index = vibrational_index((hessian + hessian.T) / 2, point[pair.free], False)
        if index != 1:
            reasons.append("hessian_index")
    return {"energy": energy, "force_norm": force, "hessian_index": index,
            "expected_index": 1, "verify_calls": verification_calls,
            "energy_error": energy_error, "gradient_error": gradient_error,
            "reasons": reasons}


def run_search(argv: list[str], directory: Path, env: dict, log, timeout: float) -> int:
    """Run one search, terminating its process group when the budget expires."""
    with subprocess.Popen(argv, cwd=directory, env=env, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True) as proc:
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return 124


def _execute(job: dict) -> dict:
    pair, settings = job["pair"], job["settings"]
    directory = Path(job["directory"]).resolve()
    write_inputs(directory, pair)
    argv = command(job["runtime"]["binary"], directory, settings, job["options"])
    inputs = {name: digest(directory / name) for name in
              ("reactant.con", "product.con", "potential.toml", "environment-types.txt")}
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(job["options"].get("threads", 1))
    (directory / "command.json").write_text(json.dumps({
        "argv": argv, "runtime": job["runtime"], "inputs": inputs,
        "environment": {key: env[key] for key in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }, indent=2) + "\n")
    began = time.perf_counter()
    reasons = []
    with (directory / "search.log").open("w") as log:
        returncode = run_search(argv, directory, env, log,
                                float(job["options"].get("timeout", 1800)))
        if returncode == 124:
            reasons.append("timeout")
    search_wall = time.perf_counter() - began
    text = (directory / "search.log").read_text(errors="replace")
    calls, total = parse_calls(text)
    if calls is None or total is None:
        reasons.append("missing_count")
    if returncode not in (0, 2, 124):
        reasons.append("crash")
    if returncode != 0:
        reasons.append("not_converged")
    if calls is not None and calls >= settings["cap"]:
        reasons.append("cap")
    if not re.search(r"^Converged: yes", text, re.MULTILINE):
        reasons.append("not_converged")
    checked = {}
    verification_start = time.perf_counter()
    if (directory / "band.h5").is_file():
        try:
            checked = verify_band(directory / "band.h5", pair, settings)
            reasons.extend(checked.pop("reasons"))
        except (OSError, KeyError, ValueError) as exc:
            reasons.append("invalid_output")
            checked["error"] = str(exc)
    else:
        reasons.append("missing_output")
    failure = classify_failure(reasons)
    record = {
        "schema": "optbench.start/v2", "start": pair.name,
        "counter": "client", "comparable": False,
        "force_calls": calls if calls is not None else 0,
        "count_available": calls is not None, "total_native_calls": total,
        "success": failure is None, "failure": failure, "reasons": reasons,
        "search_wall_seconds": search_wall,
        "verification_wall_seconds": time.perf_counter() - verification_start,
        "wall_seconds": time.perf_counter() - began,
        "returncode": returncode, "work": str(directory), **checked,
    }
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run(root: Path, problem_name: str, template_name: str, *, output: Path,
        workers: int = 1, work: Path | None = None, limit: int = 0) -> dict:
    problem = resolve_problem(root, problem_name)
    template = load_template(root, template_name)
    require_pin(template, registry_pin(root, "gpridge"))
    if template.get("problem") != problem["id"]:
        raise ValueError("template does not name this problem")
    settings = band_settings(problem)
    if settings["potential"] != "morse_pt" or problem["convergence"]["norm"] != "l2":
        raise ValueError("the native Pt runner requires Morse and the Euclidean force norm")
    if settings["geometry"] not in ("con-minima-pairs", "con-reactant-products"):
        raise ValueError("the native Pt runner requires CON endpoint pairs")
    identity = runtime(template)
    archive = geometry_archive(root, problem)
    pairs = load_pairs(archive, settings["geometry"], settings["pairs"])
    if limit > 0:
        pairs = pairs[:limit]
    work = (work or output.parent / output.stem).resolve()
    jobs = [{"pair": pair, "name": pair.name, "settings": settings,
             "runtime": identity, "options": template.get("gpridge", {}),
             "directory": str(work / pair.name)} for pair in pairs]
    records = run_jobs(jobs, _execute, workers)
    return write_campaign(output, records, template, problem, settings,
                          {**settings, "runtime": identity, "archive_sha256": digest(archive)},
                          images=settings["n_interior"], counter="client",
                          comment="# Native provider count; independent surface and stationary-point checks.")
