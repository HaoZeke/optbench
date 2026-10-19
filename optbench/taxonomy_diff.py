"""Rescore stored starts and write the failure-class diff.

The old class is the label the board on main reported: a stored failure
in the four paper-era bins, or the old reason order when that label was
missing. The new class is :func:`optbench.failures.apply_taxonomy`, which
reads the final energy, the geometry, the true force, and the call count.

PAR-2 is ``score_v2``'s mean of the penalised costs. The bootstrap does
not enter that mean, so one resample is the same number the site prints.
"""

from __future__ import annotations

import json
from pathlib import Path

from optbench.board import (
    _benchmark_index,
    _campaign_groups,
    _cap_of,
    _commit_divides_band,
    _first_raw,
    _force_tol,
    _image_count,
    _json_lines,
    _record_from_raw,
    _template_at_commit,
)
from optbench.dat import DatError, parse_lenient
from optbench.failures import FAILURES, failure_of
from optbench.problems import load_all
from optbench.score import normalize_start, score_v2
from optbench.site import campaign_problem
from optbench.v2 import parse_repetitions

OLD_FAILURES = ("crash", "divergence", "cap", "wrong_stationary")


def legacy_classify(reasons: list[str]) -> str | None:
    """The reason order on main, before the central classifier."""
    if not reasons:
        return None
    if "crash" in reasons:
        return "crash"
    if "divergence" in reasons:
        return "divergence"
    if "cap" in reasons:
        return "cap"
    if "force_norm" in reasons:
        return "divergence"
    if any(name in reasons for name in ("hessian_index", "energy", "rmsd")):
        return "wrong_stationary"
    return "divergence"


def legacy_failure_of(record: dict, cap: int | None) -> str | None:
    """Class the board on main tallied for one start."""
    if record.get("success"):
        label = record.get("failure")
        if label not in (None, "", "none"):
            raise ValueError("a success cannot carry a failure class")
        return None
    label = record.get("failure")
    if label in OLD_FAILURES:
        return str(label)
    reasons = [str(reason) for reason in record.get("reasons") or []]
    if reasons:
        classified = legacy_classify(reasons)
        if classified is not None:
            return classified
    calls = record.get("force_calls")
    if cap is not None and calls is not None and float(calls) >= float(cap):
        return "cap"
    return "divergence"


def legacy_record_from_raw(raw: dict, cap: int | None, force_tol: float | None) -> dict | None:
    """One start as ``optbench.board._record_from_raw`` built it on main."""
    if not isinstance(raw, dict):
        return None
    calls = raw.get("force_calls", raw.get("raw", raw.get("per_search")))
    if calls is None:
        return None
    try:
        calls = float(calls)
    except (TypeError, ValueError):
        return None
    if calls != calls or calls in (float("inf"), float("-inf")):
        return None
    failure = raw.get("failure")
    if "success" in raw and raw.get("success") is not None:
        success = bool(raw["success"])
    elif "failed" in raw:
        success = not bool(raw["failed"])
    elif "reason" in raw:
        success = str(raw.get("reason")) in {"0", "0.0"}
        failure = None
    else:
        return None
    if cap is not None and calls >= cap:
        success = False
    record: dict = {"force_calls": calls, "success": success, "start": raw.get("start") or raw.get("geometry")}
    if success:
        return record
    if failure in OLD_FAILURES:
        record["failure"] = failure
        return record
    reasons = raw.get("reasons")
    if isinstance(reasons, list) and reasons:
        record["reasons"] = [str(item) for item in reasons]
        return record
    if raw.get("error"):
        record["failure"] = "crash"
    elif cap is not None and calls >= cap:
        record["failure"] = "cap"
    elif force_tol is not None and raw.get("force_norm") not in (None, ""):
        try:
            above = float(raw["force_norm"]) > force_tol
        except (TypeError, ValueError):
            above = True
        record["failure"] = "divergence" if above else "wrong_stationary"
    return record


def _counts(records: list[dict], cap: int | None, *, legacy: bool) -> dict[str, int]:
    counts = {name: 0 for name in FAILURES}
    for record in records:
        if legacy:
            label = legacy_failure_of(record, cap)
        else:
            label = failure_of(record, cap)
        if label is None:
            continue
        if label not in counts:
            raise ValueError(f"unknown failure class {label}")
        counts[str(label)] += 1
    return counts


def par2_mean(records: list[dict], cap: int | None) -> float | None:
    """PAR-2 mean from ``score_v2``. ``None`` when the problem states no cap."""
    if not records:
        raise ValueError("records is empty")
    summary = score_v2(records, cap, resamples=1, seed=1)
    value = summary["par2_mean"]
    if value is None:
        return None
    return float(value)


def _pair_lines(
    group: dict[str, Path],
    cap: int | None,
    force_tol: float | None,
    images: int | None,
    problem_id: str | None = None,
):
    """Stored and rescored starts from one campaign group, in file order."""
    kinds = ("starts_jsonl", "starts", "v2")
    for kind in kinds:
        path = group.get(kind)
        if path is None:
            continue
        if kind == "v2":
            document = json.loads(path.read_text(encoding="utf-8"))
            raws = []
            for run in document.get("runs") or []:
                raws.extend(run.get("starts") or [])
        else:
            raws = list(_json_lines(path))
        old_rows = []
        new_rows = []
        for raw in raws:
            scaled = normalize_start(raw, images)
            old = legacy_record_from_raw(scaled, cap, force_tol)
            new = _record_from_raw(scaled, cap, force_tol, problem_id)
            if old is None and new is None:
                continue
            if old is None or new is None:
                raise ValueError(f"{path} dropped a start on one side of the rescore")
            old_rows.append(old)
            new_rows.append(new)
        if old_rows:
            return old_rows, new_rows
    if "dat" in group:
        text = group["dat"].read_text(encoding="utf-8")
        fields, _duplicates = parse_lenient(text)
        failed = int(float(fields.get("nfailed") or 0))
        if failed:
            return [], []
        rows = [
            {
                "force_calls": float(row["per_search"]),
                "success": True,
                "failure": None,
                "start": str(row["repetition"]),
            }
            for row in parse_repetitions(text)
        ]
        if rows:
            return list(rows), list(rows)
    return [], []


def _row(
    *,
    problem: str,
    template: str,
    source: str,
    path: str,
    cap: int | None,
    old_records: list[dict],
    new_records: list[dict],
) -> dict:
    old_counts = _counts(old_records, cap, legacy=True)
    new_counts = _counts(new_records, cap, legacy=False)
    old_par2, old_error = _safe_par2(old_records, cap)
    new_par2, new_error = _safe_par2(new_records, cap)
    moves: dict[str, int] = {}
    if len(old_records) == len(new_records):
        for old, new in zip(old_records, new_records, strict=True):
            before = legacy_failure_of(old, cap) or "success"
            after = "success" if new.get("success") else str(new.get("failure") or "not_converged")
            if before != after:
                key = f"{before}->{after}"
                moves[key] = moves.get(key, 0) + 1
    changed = old_counts != new_counts or old_par2 != new_par2 or bool(moves)
    return {
        "problem": problem,
        "template": template,
        "source": source,
        "path": path,
        "n": len(new_records),
        "cap": cap,
        "old_counts": old_counts,
        "new_counts": new_counts,
        "old_par2": old_par2,
        "new_par2": new_par2,
        "moves": moves,
        "changed": changed,
        "old_error": old_error,
        "new_error": new_error,
    }


def _safe_par2(records: list[dict], cap: int | None) -> tuple[float | None, str]:
    try:
        return par2_mean(records, cap), ""
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)


def _campaign(root: Path, problems: dict, templates: dict) -> list[dict]:
    rows = []
    for stem, group in _campaign_groups(root / "results" / "campaign").items():
        path = group.get("dat") or group.get("v2") or group.get("starts_jsonl") or group.get("starts")
        fields: dict[str, str] = {}
        if "dat" in group:
            try:
                fields, _duplicates = parse_lenient(group["dat"].read_text(encoding="utf-8"))
            except DatError:
                continue
        probe = _first_raw(group)
        template_id = fields.get("template") or str(probe.get("template") or "")
        template = templates.get(template_id)
        problem_id = (
            fields.get("problem")
            or str(probe.get("problem") or "")
            or (template or {}).get("problem")
            or campaign_problem(stem)
            or ""
        )
        problem = problems.get(problem_id)
        cap = _cap_of(problem, template, fields)
        force_tol = _force_tol(problem, template)
        images = _image_count(problem, template, fields)
        try:
            old_records, new_records = _pair_lines(group, cap, force_tol, images, problem_id)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if not old_records:
            continue
        rows.append(
            _row(
                problem=problem_id,
                template=template_id,
                source="campaign",
                path=str(path.relative_to(root)) if path else stem,
                cap=cap,
                old_records=old_records,
                new_records=new_records,
            )
        )
    return rows


def _asv(root: Path, problems: dict, templates: dict) -> list[dict]:
    results_dir = root / "results" / "asv"
    if not results_dir.is_dir():
        return []
    from asv.results import iter_results

    from optbench.asv_score import track_rows

    index = _benchmark_index()
    rows = []
    for result in iter_results(str(results_dir)):
        keys = [key for key in result.get_all_result_keys() if str(key).endswith("track_force_calls")]
        seen: set[str] = set()
        for key in keys:
            head = str(key)[: -len(".track_force_calls")] if str(key).endswith(".track_force_calls") else str(key)
            parts = head.split(".")
            class_name = parts[-1]
            if class_name in seen:
                continue
            seen.add(class_name)
            module_name = parts[-2] if len(parts) >= 2 else ""
            described = index.get(f"{module_name}.{class_name}") or index.get(class_name) or {}
            template_id = described.get("template") or ""
            template = templates.get(template_id)
            pinned = _template_at_commit(root, str(result.commit_hash), template_id) if template_id else None
            source_template = pinned or template
            problem_id = (
                described.get("problem")
                or (source_template or {}).get("problem")
                or (template or {}).get("problem")
                or ""
            )
            if not problem_id or not template_id:
                continue
            problem = problems.get(problem_id)
            cap = described.get("cap")
            if cap is None:
                cap = _cap_of(problem, source_template or template, {})
            try:
                tracked = track_rows(result, class_name)
            except SystemExit:
                continue
            reducer = str((source_template or template or {}).get("reducer") or "")
            images = _image_count(problem, source_template or template, {})
            per_image_track = reducer == "per_image" and _commit_divides_band(root, str(result.commit_hash))
            old_records = []
            new_records = []
            try:
                for item in tracked:
                    raw = {
                        "force_calls": item["force_calls"],
                        "success": item["success"],
                        "failure": item["failure"],
                    }
                    scaled = normalize_start(raw, images, per_image_track=per_image_track)
                    old = legacy_record_from_raw(scaled, cap, _force_tol(problem, source_template or template))
                    new = _record_from_raw(
                        scaled,
                        cap,
                        _force_tol(problem, source_template or template),
                        problem_id,
                    )
                    if old is None or new is None:
                        continue
                    old_records.append(old)
                    new_records.append(new)
            except ValueError:
                continue
            if not old_records:
                continue
            rows.append(
                _row(
                    problem=problem_id,
                    template=template_id,
                    source="asv",
                    path=f"results/asv/{result.commit_hash}",
                    cap=cap,
                    old_records=old_records,
                    new_records=new_records,
                )
            )
    return rows


def collect(root: Path) -> dict:
    """Every scored campaign file and asv track under ``root``."""
    root = Path(root)
    from optbench.campaign import list_templates

    problems = load_all(root / "problems")
    templates = {item["id"]: item for item in list_templates(root)}
    rows = _campaign(root, problems, templates) + _asv(root, problems, templates)
    rows.sort(key=lambda row: (row["problem"], row["template"], row["source"], row["path"]))
    changed = [row for row in rows if row["changed"]]
    return {
        "scored": len(rows),
        "changed": len(changed),
        "starts": sum(int(row["n"]) for row in rows),
        "rows": rows,
    }


def _number(value) -> str:
    if value is None:
        return "null"
    return json.dumps(value)


def _count_text(counts: dict) -> str:
    return " ".join(f"{name}={counts.get(name, 0)}" for name in FAILURES)


def render_markdown(document: dict) -> str:
    """Markdown table of templates whose class counts or PAR-2 changed."""
    lines = [
        "# Failure taxonomy rescore",
        "",
        "Each row is one stored result. The old class is the label the board on main tallied. "
        "The new class is `classify_outcome` on the stored energy, geometry, force, and call count. "
        "PAR-2 is the `score_v2` mean of the penalised costs.",
        "",
        f"Scored templates: {document['scored']}. Changed: {document['changed']}. Starts: {document['starts']}.",
        "",
        "The 2014 paper reports `nfailed`. A run fails when it exceeds the maximum number of force calls "
        "or fails to converge for any other reason. `cap` is the first clause. `not_converged` is the second "
        "clause when the energy and the geometry stayed finite and the cap was not hit. `divergence` stays "
        "the blow-up. `wrong_stationary` is a point that met the force stop and was rejected. `crash` is a "
        "dead process.",
        "",
        "| problem | template | source | n | old counts | new counts | old PAR-2 | new PAR-2 | moves |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in document["rows"]:
        if not row["changed"]:
            continue
        moves = ", ".join(f"{name} {count}" for name, count in sorted(row["moves"].items()))
        lines.append(
            "| {problem} | {template} | {source} | {n} | {old} | {new} | {old_par2} | {new_par2} | {moves} |".format(
                problem=row["problem"],
                template=row["template"] or row["path"],
                source=row["source"],
                n=row["n"],
                old=_count_text(row["old_counts"]),
                new=_count_text(row["new_counts"]),
                old_par2=_number(row["old_par2"]),
                new_par2=_number(row["new_par2"]),
                moves=moves,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_diff(root: Path, output: Path) -> dict:
    """Write the markdown diff and a JSON file with the same numbers."""
    document = collect(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(document), encoding="utf-8")
    payload = dict(document)
    payload["rows"] = document["rows"]
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document
