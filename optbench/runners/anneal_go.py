"""Anneal cluster search on the harness Lennard-Jones potential.

``cluster_search`` is the point-set driver. The two ranked presets are
the stock constructors ``Config.communicating`` and ``Config.recommended``.
Quenches go through the rgmin build anneal links. Energy and gradient
at the same coordinates are one harness call, the same counter a GMIN
row uses. A start stops when the energy is inside the problem bound
or the harness count reaches the problem cap.
"""

from __future__ import annotations

import json
import math
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from optbench.counter import CountedPotential
from optbench.globalopt import energy_tolerance, force_cap, judge_globalopt, reference_energy
from optbench.potentials import lennard_jones, read_con
from optbench.score import half_even, score_v2
from optbench.v2 import collect_provenance

ANNEAL_VERSION = "0.11.1"
ANNEAL_COMMIT = "db22b99d09a0cdf1d4f004136f9ecb04104fcb39"
RGMIN_VERSION = "0.2.0"
RGMIN_COMMIT = "20832fd169932a1cd0d47ac84dc77fa9b435f940"
SEED_BASE = 20261018
# A quench step stays under this per-atom displacement. A larger jump is a hop.
HOP_DISPLACEMENT = 0.5
# The binding spends three evaluations on a gradient-orientation probe.
PROBE_CALLS = 3


class SearchStop(KeyboardInterrupt):
    """Stop ``cluster_search``. The binding re-raises ``KeyboardInterrupt`` from a callback."""


def preset_flags(name: str) -> dict[str, bool]:
    """Flags for one stock preset.

    ``recommended=False`` selects ``Config.for_cluster``. ``derived=True``
    selects ``Config.derived``. Neither is a ranked row.
    """
    if name == "recommended":
        return {"recommended": True, "derived": False, "communicating": False, "ras": False}
    if name == "communicating":
        return {"recommended": True, "derived": False, "communicating": True, "ras": False}
    raise ValueError(f"unknown anneal preset {name}")


def version_label(preset: str) -> str:
    preset_flags(preset)
    return f"{ANNEAL_VERSION} ({preset})"


def installed_rgmin_pin() -> tuple[str, str]:
    """Version and commit of the rgmin crate linked into this anneal wheel."""
    import anneal

    if getattr(anneal, "__version__", None) != ANNEAL_VERSION:
        found = getattr(anneal, "__version__", None)
        raise RuntimeError(f"anneal {found} is not {ANNEAL_VERSION}")
    root = Path(anneal.__file__).resolve().parent.parent
    matches = sorted(root.glob("anneal-*.dist-info/sboms/anneal-core.cyclonedx.json"))
    if not matches:
        raise RuntimeError("anneal wheel has no rgmin SBOM")
    document = json.loads(matches[-1].read_text(encoding="utf-8"))
    for component in document.get("components") or []:
        if component.get("name") != "rgmin":
            continue
        version = str(component.get("version") or "")
        commit = ""
        purl = str(component.get("purl") or "")
        marker = "%40"
        if marker in purl:
            commit = purl.split(marker, 1)[1]
        if not commit:
            for ref in component.get("externalReferences") or []:
                url = str(ref.get("url") or "")
                if "/commit/" in url:
                    commit = url.rsplit("/", 1)[-1]
        if version != RGMIN_VERSION or not commit.startswith(RGMIN_COMMIT):
            raise RuntimeError(f"anneal linked rgmin {version} {commit}, expected {RGMIN_VERSION} {RGMIN_COMMIT}")
        return version, RGMIN_COMMIT
    raise RuntimeError("anneal SBOM has no rgmin component")


def _center(pos: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) - np.mean(pos, axis=0)


def _read_points(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
        except ValueError:
            continue
    coords = np.asarray(rows, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError("coordinate file has no points")
    return coords


def _read_geometry(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".con":
        pos, _box = read_con(text)
        return pos
    return _read_points(text)


def _start_paths(dest: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    cons: list[tuple[int, Path]] = []
    for path in dest.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("coords."):
            suffix = name.split(".", 1)[1]
            if suffix.isdigit():
                numbered.append((int(suffix), path))
        elif path.suffix == ".con":
            stem = path.stem
            if stem.isdigit():
                cons.append((int(stem), path))
    if numbered:
        numbered.sort()
        return [path for _index, path in numbered]
    cons.sort()
    return [path for _index, path in cons]


def load_geometries(root: Path, problem: dict, work: Path) -> list[tuple[str, np.ndarray]]:
    from optbench.campaign import geometry_archive

    archive = geometry_archive(root, problem)
    dest = work / "geometries"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if not marker.is_file():
        with tarfile.open(archive) as packed:
            packed.extractall(dest, filter="data")
        marker.write_text(archive.name, encoding="utf-8")
    paths = _start_paths(dest)
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(paths) != int(expected):
        raise ValueError(f"{problem['id']} expected {expected} starts, found {len(paths)}")
    if not paths:
        raise FileNotFoundError(f"{archive} has no starts")
    loaded = []
    for path in paths:
        pos = _center(_read_geometry(path))
        if path.suffix == ".con":
            label = path.stem
        else:
            label = path.name
        loaded.append((label, pos))
    return loaded


def _blank_profile() -> dict:
    return {
        "quenches": 0,
        "quench_calls": 0,
        "quench_calls_max": 0,
        "improvements": 0,
        "probe_calls": 0,
        "obj_calls": 0,
        "grad_calls": 0,
    }


def _close_quench(state: dict) -> None:
    calls = int(state["seg_calls"])
    if calls <= 0:
        return
    state["quenches"] += 1
    state["quench_calls"] += calls
    state["quench_calls_max"] = max(int(state["quench_calls_max"]), calls)
    landed = state["seg_best"]
    if math.isfinite(landed) and landed < state["incumbent"]:
        state["improvements"] += 1
        state["incumbent"] = landed
    state["seg_calls"] = 0
    state["seg_best"] = math.inf


def _observe(state: dict, pos: np.ndarray, energy: float, fresh: bool) -> None:
    if not fresh:
        if math.isfinite(energy) and energy < state["best_energy"]:
            state["best_energy"] = float(energy)
            state["best_x"] = pos.reshape(-1).copy()
        return
    seen = int(state["seen"])
    state["seen"] = seen + 1
    if seen < PROBE_CALLS:
        state["probe_calls"] = seen + 1
        state["prev"] = pos.copy()
        return
    previous = state["prev"]
    hopped = previous is None
    if previous is not None and previous.shape == pos.shape:
        hopped = float(np.max(np.linalg.norm(pos - previous, axis=1))) > HOP_DISPLACEMENT
    if hopped:
        _close_quench(state)
    state["seg_calls"] += 1
    if math.isfinite(energy):
        state["seg_best"] = min(float(state["seg_best"]), float(energy))
        if energy < state["best_energy"]:
            state["best_energy"] = float(energy)
            state["best_x"] = pos.reshape(-1).copy()
    state["prev"] = pos.copy()


def bind_potential(cap: int, target: float):
    """Harness-counted energy and gradient callables for one start."""
    potential = CountedPotential(lennard_jones)
    state = _blank_profile()
    state.update(
        {
            "seen": 0,
            "seg_calls": 0,
            "seg_best": math.inf,
            "incumbent": math.inf,
            "best_energy": math.inf,
            "best_x": None,
            "prev": None,
            "hit": False,
            "cap_hit": False,
            "stop": False,
        }
    )

    def _touch(values, kind: str) -> tuple[float, np.ndarray]:
        if kind == "obj":
            state["obj_calls"] += 1
        else:
            state["grad_calls"] += 1
        before = potential.calls
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        energy, gradient = potential.energy_forces(flat)
        pos = flat.reshape(-1, 3)
        fresh = potential.calls > before
        _observe(state, pos, float(energy), fresh)
        if math.isfinite(energy) and float(energy) <= target and potential.calls < cap:
            state["hit"] = True
            state["stop"] = True
            raise SearchStop()
        if potential.calls >= cap:
            state["cap_hit"] = True
            state["stop"] = True
            raise SearchStop()
        return float(energy), np.asarray(gradient, dtype=np.float64)

    def objective(values):
        energy, _gradient = _touch(values, "obj")
        return energy

    def gradient(values):
        _energy, grad = _touch(values, "grad")
        return grad

    return potential, state, objective, gradient


def _profile_summary(state: dict) -> dict:
    _close_quench(state)
    quenches = int(state["quenches"])
    calls = int(state["quench_calls"])
    mean = (calls / quenches) if quenches else 0.0
    return {
        "quenches": quenches,
        "quench_calls": calls,
        "mean_quench_calls": mean,
        "max_quench_calls": int(state["quench_calls_max"]),
        "improvements": int(state["improvements"]),
        "probe_calls": int(state["probe_calls"]),
        "obj_calls": int(state["obj_calls"]),
        "grad_calls": int(state["grad_calls"]),
    }


def search_start(job: dict) -> dict:
    """One archived start. The geometry is already centred at the origin."""
    from anneal import cluster_search

    installed_rgmin_pin()
    flags = preset_flags(str(job["preset"]))
    coords = np.asarray(job["coords"], dtype=np.float64)
    natoms = int(coords.shape[0])
    cap = int(job["cap"])
    target = float(job["target"])
    potential, state, objective, gradient = bind_potential(cap, target)
    returned = None
    error = ""
    try:
        returned = cluster_search(
            objective,
            gradient,
            natoms,
            int(job["budget"]),
            int(job["seed"]),
            recommended=flags["recommended"],
            derived=flags["derived"],
            communicating=flags["communicating"],
            ras=flags["ras"],
            start=coords.reshape(-1),
        )
    except KeyboardInterrupt:
        if not state["stop"]:
            raise
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
    profile = _profile_summary(state)
    if returned is not None:
        profile["hops"] = int(returned.get("hops") or 0)
    energy = state["best_energy"]
    if not math.isfinite(energy):
        energy = None
    judged = judge_globalopt(
        energy,
        potential.calls,
        reference=float(job["reference"]),
        tolerance=float(job["tolerance"]),
        cap=cap,
    )
    if error:
        judged = {
            "success": False,
            "failure": "crash",
            "reasons": ["crash"],
            "energy": judged["energy"],
            "force_calls": float(potential.calls),
        }
    return {
        "index": int(job["index"]),
        "start": str(job["start"]),
        "seed": int(job["seed"]),
        "force_calls": int(potential.calls),
        "success": bool(judged["success"]),
        "failure": judged["failure"],
        "reasons": judged["reasons"],
        "energy": judged["energy"],
        "counter": "harness",
        "preset": str(job["preset"]),
        "profile": profile,
        "error": error,
    }


def _format_field(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def _read_done(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _dat(summary: dict, template: dict, metrics: dict) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "median": metrics.get("median"),
        "code": template.get("code"),
        "code_version": template.get("code_version"),
        "commit": template.get("commit"),
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "problem": template.get("problem"),
        "counter": "harness",
        "par2_mean": metrics.get("par2_mean"),
        "par2_median": metrics.get("median"),
        "par2_median_low": metrics.get("median_low"),
        "par2_median_high": metrics.get("median_high"),
        "v2_nfailed": metrics.get("nfailed"),
        "v2_successes": metrics.get("successes"),
        "n": metrics.get("n"),
        "energy_min": summary.get("energy_min"),
        "energy_max": summary.get("energy_max"),
        "cap": summary.get("cap"),
        "rgmin_version": RGMIN_VERSION,
        "rgmin_commit": RGMIN_COMMIT,
    }
    lines = [
        "# counter harness. One force call is one Lennard-Jones energy and gradient at the same coordinates.",
        "# cluster_search quenches through the rgmin build linked by anneal.",
        "# A start succeeds inside the problem energy bound and under the problem cap.",
    ]
    note = str(template.get("note") or "").strip()
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {_format_field(value)}")
    return "\n".join(lines) + "\n"


def _write_result(output: Path, problem: dict, template: dict, records: list[dict]) -> dict:
    from optbench.campaign import write_starts

    cap = force_cap(problem)
    ordered = sorted(records, key=lambda row: int(row["index"]))
    for row in ordered:
        row.pop("index", None)
    calls = [int(row["force_calls"]) for row in ordered]
    successes = [row for row in ordered if row["success"]]
    energies = [float(row["energy"]) for row in ordered if row.get("energy") is not None]
    if successes:
        mean_calls = half_even(sum(int(row["force_calls"]) for row in successes) / len(successes))
    else:
        mean_calls = half_even(sum(calls) / len(calls))
    summary = {
        "force_calls": mean_calls,
        "force_calls_min": min(calls),
        "force_calls_max": max(calls),
        "nfailed": sum(1 for row in ordered if not row["success"]),
        "energy_min": min(energies) if energies else None,
        "energy_max": max(energies) if energies else None,
        "cap": cap,
    }
    scored = [
        {
            "force_calls": row["force_calls"],
            "success": row["success"],
            "failure": row["failure"],
            "reasons": row.get("reasons") or [],
        }
        for row in ordered
    ]
    metrics = score_v2(scored, cap)
    preset = str(template.get("preset") or "")
    parameters = {
        "preset": preset,
        "cap": cap,
        "reference_energy": reference_energy(problem),
        "tolerance": energy_tolerance(problem),
        "seed_base": SEED_BASE,
        "potential": "lennard_jones",
        "epsilon": 1.0,
        "sigma": 1.0,
        "cutoff": 40.0,
        "centered": True,
        "rgmin_version": RGMIN_VERSION,
        "rgmin_commit": RGMIN_COMMIT,
        "budget_factor": 3,
    }
    provenance = collect_provenance(
        code=str(template.get("code", "anneal")),
        code_version=str(template.get("code_version", "")),
        commit=str(template.get("commit", "")),
        parameters=parameters,
    )
    document = {
        "schema": "optbench.result/v2",
        "status": "measured",
        "problem": problem["id"],
        "scoring": problem["scoring"]["v2"]["version"],
        "par_factor": 2,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20261015,
        "note": "The problem energy bound is the success rule. The harness counter is the call count.",
        "runs": [
            {
                "problem": problem["id"],
                "entry": output.stem,
                "code": template.get("code"),
                "code_version": template.get("code_version"),
                "commit": template.get("commit"),
                "algorithm": template.get("algorithm"),
                "template": template["id"],
                "tuning": template.get("tuning", ""),
                "cap": cap,
                "present_source": True,
                "provenance": provenance,
                "parameters": parameters,
                "metrics": metrics,
                "starts": ordered,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, metrics), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), ordered)
    output.with_suffix(".v2.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    summary["problem"] = problem["id"]
    summary["template"] = template["id"]
    summary["starts"] = len(ordered)
    summary["par2_mean"] = metrics["par2_mean"]
    summary["nfailed"] = metrics["nfailed"]
    summary["successes"] = metrics["successes"]
    return summary


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
) -> dict:
    del geometries
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} belongs to {stated}")
    if template.get("commit") != ANNEAL_COMMIT:
        raise RuntimeError(f"{template['id']} commit is not anneal {ANNEAL_VERSION}")
    if template.get("code_version") != version_label(str(template.get("preset") or "")):
        raise RuntimeError(f"{template['id']} version is not the preset label")
    preset_flags(str(template.get("preset") or ""))
    cap = force_cap(problem)
    if template.get("failure_cap") not in (None, "") and int(template["failure_cap"]) != cap:
        raise ValueError(f"{template['id']} failure_cap is not the problem cap")
    reference = reference_energy(problem)
    tolerance = energy_tolerance(problem)
    target = reference + tolerance
    scratch = work or output.parent / f".work-{output.stem}"
    scratch.mkdir(parents=True, exist_ok=True)
    loaded = load_geometries(root, problem, scratch)
    done_path = output.with_suffix(".starts.jsonl")
    done = {str(row["start"]): row for row in _read_done(done_path)}
    jobs = []
    records = []
    for index, (label, pos) in enumerate(loaded):
        if label in done:
            row = dict(done[label])
            row["index"] = index
            records.append(row)
            continue
        jobs.append(
            {
                "index": index,
                "start": label,
                "coords": pos.tolist(),
                "cap": cap,
                "budget": cap * 3,
                "target": target,
                "reference": reference,
                "tolerance": tolerance,
                "seed": SEED_BASE + index,
                "preset": str(template["preset"]),
            }
        )
    if jobs:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        installed_rgmin_pin()
        total = len(loaded)

        def _finish(record: dict) -> None:
            records.append(record)
            done_path.parent.mkdir(parents=True, exist_ok=True)
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"{template['id']} {len(records)}/{total} start={record['start']} "
                f"calls={record['force_calls']} energy={record['energy']} success={record['success']}",
                flush=True,
            )

        if workers <= 1:
            for job in jobs:
                _finish(search_start(job))
        else:
            import multiprocessing as mp

            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(search_start, job) for job in jobs]
                for future in as_completed(futures):
                    _finish(future.result())
    expected = (problem.get("sample") or {}).get("starts")
    if expected is not None and len(records) != int(expected):
        raise RuntimeError(f"{template['id']} scored {len(records)} of {expected} starts")
    return _write_result(output, problem, template, records)
