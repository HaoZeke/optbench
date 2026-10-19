"""A minimization success is a bound cluster, not only a small force.

The problem files do not give an energy window or a per-start reference.
``[bound]`` states the neighbour distance: ``1.5`` times ``sigma`` for
Lennard-Jones, and the Morse bond length for the bulk cell. Every atom
has to sit in one component. A dissociated end state is ``divergence``.

Campaign records store the energy and not the coordinates. A recorded
energy above the spanning-tree ceiling cannot be that component: each of
the ``atoms - 1`` contacts inside the neighbour shell is at most the pair
energy at the shell edge, and a force-converged point has no repulsive
contact. The shifted Morse energy is zero when every pair is outside the
potential cutoff, which is outside the bond length.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from optbench.failures import FAILURES, failure_tally
from optbench.potentials import LJ_SIGMA, MORSE_R0
from optbench.score import score_v2

LJ_KIND = "lj"
MORSE_KIND = "morse"
_KINDS = {
    "lj": LJ_KIND,
    "lennard-jones": LJ_KIND,
    "morse": MORSE_KIND,
}


def distance_for_kind(kind: str) -> float:
    """Neighbour distance declared for this potential kind."""
    name = _KINDS.get(kind)
    if name == LJ_KIND:
        return 1.5 * LJ_SIGMA
    if name == MORSE_KIND:
        return float(MORSE_R0)
    raise ValueError(f"no bound distance for {kind}")


def bound_distance(problem: dict) -> float:
    """Distance from the problem ``[bound]`` table."""
    bound = problem.get("bound") or {}
    scale = float(bound["scale"])
    length = str(bound["length"])
    if length == "sigma":
        return scale * float(problem["potential"]["sigma"])
    if length == "bond":
        return scale * float(bound["bond"])
    raise ValueError(f"{problem.get('id')} bound length {length}")


def _positions(positions) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    return pos


def _lengths(box) -> np.ndarray | None:
    if box is None:
        return None
    lengths = np.asarray(box, dtype=np.float64).reshape(-1)
    if lengths.size == 3:
        return lengths
    if lengths.size == 9:
        cell = lengths.reshape(3, 3)
        return np.array([cell[0, 0], cell[1, 1], cell[2, 2]], dtype=np.float64)
    raise ValueError("box must be three lengths or a diagonal cell")


def connected_cluster(positions, distance: float, box=None) -> bool:
    """True when every atom belongs to one component.

    ``box`` is three orthorhombic lengths. Pairs use the minimum image.
    A pair at ``distance`` counts as a neighbour.
    """
    pos = _positions(positions)
    count = int(pos.shape[0])
    if count <= 1:
        return True
    cutoff = float(distance)
    if cutoff <= 0.0:
        raise ValueError("bound distance must be positive")
    # A contact written at the declared distance survives binary rounding.
    cutoff *= 1.0 + 1e-8
    parent = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    diff = pos[:, None, :] - pos[None, :, :]
    lengths = _lengths(box)
    if lengths is not None:
        diff -= lengths * np.round(diff / lengths)
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    limit = cutoff * cutoff
    iu, ju = np.triu_indices(count, k=1)
    close = r2[iu, ju] <= limit
    for left, right in zip(iu[close], ju[close], strict=True):
        a = find(int(left))
        b = find(int(right))
        if a != b:
            parent[a] = b
    root = find(0)
    return all(find(index) == root for index in range(count))


def lj_pair_energy(distance: float) -> float:
    """Unshifted Lennard-Jones pair energy at ``distance`` with sigma 1."""
    sr = LJ_SIGMA / float(distance)
    sr6 = sr ** 6
    return 4.0 * (sr6 * sr6 - sr6)


def energy_proves_dissociated(kind: str, energy: float, atoms: int, distance: float) -> bool:
    """True when ``energy`` cannot be one connected cluster of ``atoms``.

    Lennard-Jones uses the spanning tree of neighbour contacts. Morse uses
    the shifted zero: a non-negative energy has no pair inside the cutoff.
    """
    if not math.isfinite(float(energy)) or int(atoms) < 2:
        return False
    name = _KINDS.get(kind)
    if name == LJ_KIND:
        ceiling = (int(atoms) - 1) * lj_pair_energy(distance)
        return float(energy) > ceiling
    if name == MORSE_KIND:
        return float(energy) >= 0.0
    return False


def note_unbound(reasons: list[str], positions, *, kind: str, box=None) -> None:
    """Append ``divergence`` when the end state is dissociated.

    A crash or a cap stays that class. ``classify_failure`` ranks
    ``divergence`` ahead of a wrong Hessian index.
    """
    if "crash" in reasons or "cap" in reasons:
        return
    cell = box if _KINDS.get(kind) == MORSE_KIND else None
    if connected_cluster(positions, distance_for_kind(kind), cell):
        return
    if "divergence" not in reasons:
        reasons.append("divergence")


def archived_dissociated(problem: dict, record: dict) -> bool:
    """Apply the energy certificate to one archived start."""
    energy = record.get("energy")
    if not isinstance(energy, (int, float)) or isinstance(energy, bool):
        return False
    bound = problem.get("bound") or {}
    atoms = int(bound["atoms"])
    kind = str((problem.get("potential") or {}).get("kind") or "")
    if problem["id"] == "minimization/morse-bulk":
        kind = MORSE_KIND
    elif kind in {"lennard-jones", "lj", ""}:
        kind = LJ_KIND
    return energy_proves_dissociated(kind, float(energy), atoms, bound_distance(problem))


def rescore_record(problem: dict, record: dict) -> bool:
    """Mark a dissociated archived start as ``divergence``. Return whether it changed."""
    if not archived_dissociated(problem, record):
        return False
    failure = record.get("failure")
    if failure in {"crash", "cap"}:
        return False
    reasons = [str(item) for item in record.get("reasons") or []]
    if record.get("success") is False and failure == "divergence" and "divergence" in reasons:
        return False
    if "divergence" not in reasons:
        reasons.append("divergence")
    record["success"] = False
    record["failure"] = "divergence"
    record["reasons"] = reasons
    return True


def _problem_of_dat(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("problem "):
            return line.split(" ", 1)[1].strip()
    return None


def _load_problems(root: Path) -> dict[str, dict]:
    from optbench.problems import load_all

    return load_all(root / "problems")


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _summary_lines(metrics: dict) -> dict[str, str]:
    def num(value) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        number = float(value)
        if number == 0.0:
            return "0"
        text = f"{number:.12g}"
        return text

    return {
        "par2_mean": num(metrics["par2_mean"]),
        "wilson_low": num(metrics["wilson_low"]),
        "wilson_high": num(metrics["wilson_high"]),
        "par2_median": num(metrics["median"]),
        "par2_median_low": num(metrics["median_low"]),
        "par2_median_high": num(metrics["median_high"]),
        "v2_nfailed": str(int(metrics["nfailed"])),
        "v2_successes": str(int(metrics["successes"])),
    }


def _patch_dat(path: Path, metrics: dict) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    updates = _summary_lines(metrics)
    seen = set()
    rewritten = []
    note = (
        "# A returned geometry is one cluster inside the problem bound distance. "
        "A dissociated geometry is divergence."
    )
    inserted = any(line == note for line in lines)
    for line in lines:
        key = line.split(" ", 1)[0] if line and not line.startswith("#") else ""
        if key in updates:
            rewritten.append(f"{key} {updates[key]}")
            seen.add(key)
        else:
            rewritten.append(line)
        if not inserted and line.startswith("# Energy and Kabsch"):
            rewritten.append(note)
            inserted = True
    for key, value in updates.items():
        if key not in seen:
            rewritten.append(f"{key} {value}")
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _patch_v2(path: Path, rows: list[dict], metrics: dict, cap: int) -> None:
    if not path.is_file():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    tally = failure_tally(
        [{"force_calls": row["force_calls"], "success": row["success"], "failure": row.get("failure")} for row in rows],
        cap,
    )
    for run in document.get("runs") or []:
        starts = run.get("starts")
        if isinstance(starts, list) and len(starts) == len(rows):
            for stored, fresh in zip(starts, rows, strict=True):
                stored["success"] = fresh["success"]
                stored["failure"] = fresh.get("failure")
                if "reasons" in fresh:
                    stored["reasons"] = fresh["reasons"]
        block = run.get("metrics")
        if isinstance(block, dict):
            block["nfailed"] = metrics["nfailed"]
            block["n_cap"] = tally["cap"]
            block["n_crash"] = tally["crash"]
            block["n_divergence"] = tally["divergence"]
            block["n_wrong_stationary"] = tally["wrong_stationary"]
            block["par2_mean"] = metrics["par2_mean"]
            block["mean"] = metrics["par2_mean"]
            for name in ("median", "median_low", "median_high", "wilson_low", "wilson_high", "success_rate"):
                if name in metrics:
                    block[name] = metrics[name]
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _cap_of(problem: dict) -> int:
    return int((problem.get("convergence") or {})["max_force_calls"])


def rescore_campaigns(root: Path) -> list[dict]:
    """Rewrite minimization campaign records that the energy certificate rejects.

    The optimizer is not run again. The returned rows are the templates whose
    success count changed, with the old and new tallies.
    """
    root = Path(root)
    problems = _load_problems(root)
    campaign = root / "results" / "campaign"
    changed: list[dict] = []
    vectors: list[dict] = []
    for dat in sorted(campaign.glob("*.dat")):
        problem_id = _problem_of_dat(dat)
        problem = problems.get(problem_id or "")
        if problem is None or not str(problem_id).startswith("minimization/"):
            continue
        if "bound" not in problem:
            continue
        starts = dat.with_suffix(".starts.jsonl")
        if not starts.is_file():
            continue
        rows = _jsonl(starts)
        before_success = sum(1 for row in rows if row.get("success"))
        old_flags = [bool(row.get("success")) for row in rows]
        template = ""
        old_par2 = None
        for line in dat.read_text(encoding="utf-8").splitlines():
            if line.startswith("template "):
                template = line.split(" ", 1)[1].strip()
            elif line.startswith("par2_mean "):
                old_par2 = float(line.split(" ", 1)[1])
        touched = sum(1 for row in rows if rescore_record(problem, row))
        if touched == 0:
            continue
        _write_jsonl(starts, rows)
        if not _recorded_success(rows):
            continue
        vectors.append({"template": template, "old": old_flags, "rows": rows})
        cap = _cap_of(problem)
        scored = [
            {"force_calls": row["force_calls"], "success": bool(row["success"]), "failure": row.get("failure")}
            for row in rows
        ]
        metrics = score_v2(scored, cap)
        _patch_dat(dat, metrics)
        _patch_v2(dat.with_suffix(".v2.json"), rows, metrics, cap)
        after_success = int(metrics["successes"])
        changed.append(
            {
                "problem": problem_id,
                "template": template,
                "file": starts.name,
                "n": len(rows),
                "old_successes": before_success,
                "new_successes": after_success,
                "old_par2": old_par2,
                "new_par2": metrics["par2_mean"],
                "failures": failure_tally(scored, cap),
            }
        )
    if vectors:
        _sync_asv(root, vectors)
    return changed


def _sync_asv(root: Path, vectors: list[dict]) -> None:
    """Copy rescored success bits into asv tracks that stored the old bits."""
    from optbench.board import _benchmark_index

    index = _benchmark_index()
    by_template: dict[str, list[dict]] = {}
    for item in vectors:
        by_template.setdefault(item["template"], []).append(item)
    directory = root / "results" / "asv"
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*.json")):
        if path.name in {"machine.json", "benchmarks.json"}:
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        results = document.get("results") or {}
        dirty = False
        keys = [key for key in results if key.endswith(".track_success")]
        for key in keys:
            class_name = key.rsplit(".", 1)[0]
            described = index.get(class_name) or {}
            template = str(described.get("template") or "")
            candidates = by_template.get(template) or []
            if not candidates:
                continue
            entry = results[key]
            if not isinstance(entry, list) or not entry or not isinstance(entry[0], list):
                continue
            params = entry[1] if len(entry) > 1 else []
            order = _param_order(params)
            calls_key = f"{class_name}.track_force_calls"
            calls_entry = results.get(calls_key)
            if not isinstance(calls_entry, list) or not calls_entry or not isinstance(calls_entry[0], list):
                continue
            stored_calls = [float(value) for value in calls_entry[0]]
            match = None
            for candidate in candidates:
                if not _recorded_success(candidate["rows"]):
                    continue
                aligned = _align(candidate["rows"], order, len(stored_calls))
                if aligned is None:
                    continue
                if [float(row["force_calls"]) for row in aligned] == stored_calls:
                    match = aligned
                    break
            if match is None:
                continue
            if _write_tracks(results, class_name, match):
                dirty = True
        if dirty:
            _write_asv(path, document)


def _recorded_success(rows: list[dict]) -> bool:
    """v2 campaign rows carry ``success``. Legacy rows only carry ``failed``."""
    return bool(rows) and all("success" in row for row in rows)


def _write_asv(path: Path, document: dict) -> None:
    original = path.read_text(encoding="utf-8")
    if original.startswith("{\n"):
        rendered = json.dumps(document, indent=2) + "\n"
    else:
        rendered = json.dumps(document)
        if original.endswith("\n"):
            rendered += "\n"
    path.write_text(rendered, encoding="utf-8")


def _param_order(params) -> list[str] | None:
    """Start labels in asv product order.

    The long axis is the start. A one-value seed shares digits with start
    ``0``, so the seed is not used as an index.
    """
    if not isinstance(params, list) or not params:
        return None
    import itertools

    from optbench.asv_score import _unwrap

    axes = []
    for axis in params:
        if not isinstance(axis, list):
            return None
        axes.append([str(_unwrap(value)) for value in axis])
    if not axes:
        return None
    width = max(len(axis) for axis in axes)
    if width <= 1:
        return None
    slot = next(index for index, axis in enumerate(axes) if len(axis) == width)
    return [combo[slot] for combo in itertools.product(*axes)]


def _align(rows: list[dict], order: list[str] | None, count: int) -> list[dict] | None:
    if order is None or len(order) != count or len(rows) != count:
        return None
    by_index = {}
    by_start = {}
    for row in rows:
        if "index" in row:
            by_index[str(row["index"])] = row
        if row.get("start") is not None:
            by_start[str(row["start"])] = row
    aligned = []
    for label in order:
        found = by_index.get(label) or by_start.get(label)
        if found is None:
            return None
        aligned.append(found)
    return aligned


def _write_tracks(results: dict, class_name: str, rows: list[dict]) -> bool:
    success = [1.0 if row.get("success") else 0.0 for row in rows]
    classes = {name: [] for name in FAILURES}
    for row in rows:
        failure = row.get("failure") if not row.get("success") else None
        for name in FAILURES:
            classes[name].append(1.0 if failure == name else 0.0)
    mapping = {
        "track_success": success,
        "track_failure_cap": classes["cap"],
        "track_failure_wrong_stationary": classes["wrong_stationary"],
        "track_failure_divergence": classes["divergence"],
        "track_failure_crash": classes["crash"],
    }
    changed = False
    for method, values in mapping.items():
        key = f"{class_name}.{method}"
        entry = results.get(key)
        if isinstance(entry, list) and entry and isinstance(entry[0], list) and len(entry[0]) == len(values):
            if entry[0] != values:
                entry[0] = values
                changed = True
    return changed
