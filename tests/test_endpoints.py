"""Endpoint minima stored beside a saddle-search campaign.

A saddle-search starts file carries both minimized endpoints, or an
explicit ``not_minimized`` row, in ``<stem>.endpoints.npz``. ``optbench
check`` rejects a campaign record that omits them. Starts files listed
by sha256 in ``oracle/saddle_endpoint_legacy.txt`` were scored before
those coordinates existed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import resolve_problem
from optbench.check import check, saddle_endpoint_gaps
from optbench.connection import connected_endpoints, same_minimum
from optbench.endpoints import (
    FORMAT_DELTA,
    STATUS_NOT_MINIMIZED,
    STATUS_STORED,
    EndpointArchive,
    align_resume,
    archive_error,
    load_endpoints,
    stamp_endpoint,
)
from optbench.engines.geometries import load_starts

ROOT = Path(__file__).resolve().parents[1]


def _campaign(root: Path) -> Path:
    shutil.copytree(ROOT / "problems", root / "problems")
    shutil.copytree(ROOT / "oracle", root / "oracle")
    campaign = root / "results" / "campaign"
    campaign.mkdir(parents=True)
    return campaign


def _dat(path: Path, problem: str) -> None:
    path.write_text(f"problem {problem}\n", encoding="utf-8")


def _island(work: Path):
    problem = resolve_problem(ROOT, "saddle-search/pt-island")
    start = load_starts(ROOT, problem, work)[0]
    positions = np.asarray(start["positions"], dtype=np.float64)
    free = np.flatnonzero(~np.asarray(start["fixed"], dtype=bool))
    cell = np.asarray(start["box"], dtype=np.float64)
    return positions, free, cell


def test_saddle_search_without_endpoints_fails_check(tmp_path, monkeypatch, capsys):
    campaign = _campaign(tmp_path)
    _dat(campaign / "bare.dat", "saddle-search/pt-island")
    (campaign / "bare.starts.jsonl").write_text(
        json.dumps({"start": "rep1-search1", "success": False}) + "\n",
        encoding="utf-8",
    )
    _dat(campaign / "min.dat", "minimization/lj38")
    (campaign / "min.starts.jsonl").write_text(
        json.dumps({"start": "seed-0", "success": True}) + "\n",
        encoding="utf-8",
    )
    _dat(campaign / "band.dat", "double-ended-saddle-search/pt-island")
    (campaign / "band.starts.jsonl").write_text(
        json.dumps({"start": "pair-0", "success": True}) + "\n",
        encoding="utf-8",
    )
    assert saddle_endpoint_gaps(tmp_path) == ["missing endpoints bare.starts.jsonl rep1-search1"]
    monkeypatch.setattr("optbench.site_check.check_site", lambda _root: [])
    assert check(tmp_path / "problems", tmp_path / "oracle" / "entries.jsonl") == 1
    output = capsys.readouterr().out
    assert "endpoint_gaps 1" in output
    assert "missing endpoints bare.starts.jsonl rep1-search1" in output
    assert "min.starts" not in output
    assert "band.starts" not in output


def test_a_legacy_hash_exempts_the_unchanged_starts_file(tmp_path):
    campaign = _campaign(tmp_path)
    _dat(campaign / "bare.dat", "saddle-search/pt-island")
    starts = campaign / "bare.starts.jsonl"
    starts.write_text(json.dumps({"start": "rep1-search1", "success": False}) + "\n", encoding="utf-8")
    digest = hashlib.sha256(starts.read_bytes()).hexdigest()
    legacy = tmp_path / "oracle" / "saddle_endpoint_legacy.txt"
    previous = legacy.read_text(encoding="utf-8") if legacy.is_file() else ""
    if previous and not previous.endswith("\n"):
        previous += "\n"
    legacy.write_text(previous + f"bare.starts.jsonl {digest}\n", encoding="utf-8")
    assert saddle_endpoint_gaps(tmp_path) == []
    starts.write_bytes(starts.read_bytes() + b"\n")
    assert saddle_endpoint_gaps(tmp_path) == ["missing endpoints bare.starts.jsonl rep1-search1"]


def test_a_stored_archive_passes_check(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    archive = EndpointArchive(campaign / "kept.endpoints.npz", 1)
    archive.reset([19.2088, 19.0118, 30.0], [0])
    reactant = np.zeros((1, 3))
    hopped = reactant.copy()
    hopped[0, 0] = 2.71
    relaxed = reactant.copy()
    relaxed[0, 0] = 0.05
    first = {"start": "rep1-search1", "success": True}
    second = {"start": "rep1-search2", "success": False}
    stamp_endpoint(first, archive.name, 0, archive.add(reactant, hopped, relaxed))
    stamp_endpoint(second, archive.name, 1, archive.add(reactant, None, None))
    archive.save()
    _dat(campaign / "kept.dat", "saddle-search/pt-island")
    (campaign / "kept.starts.jsonl").write_text(
        json.dumps(first, sort_keys=True) + "\n" + json.dumps(second, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert archive_error(archive.path, [first, second], "kept.starts.jsonl") is None
    assert saddle_endpoint_gaps(tmp_path) == []
    monkeypatch.setattr("optbench.site_check.check_site", lambda _root: [])
    assert check(tmp_path / "problems", tmp_path / "oracle" / "entries.jsonl") == 0


def test_float16_deltas_keep_the_hop_and_the_relaxation(tmp_path):
    positions, free, cell = _island(tmp_path / "work")
    reactant = positions[free]
    hopped = reactant.copy()
    hopped[0, 0] += 2.71
    relaxed = reactant.copy()
    relaxed[0, 0] += 0.05
    archive = EndpointArchive(tmp_path / "island.endpoints.npz", reactant.shape[0])
    archive.reset(cell, free)
    assert archive.add(reactant, hopped, relaxed) == STATUS_STORED
    assert archive.add(reactant, None, None) == STATUS_NOT_MINIMIZED
    archive.save()
    with np.load(archive.path) as packed:
        assert int(packed["format"]) == FORMAT_DELTA
        assert "plus" not in packed.files
        assert "minus" not in packed.files
        assert packed["plus_delta"].dtype == np.float16
        assert not np.isfinite(packed["plus_delta"][1]).all()
    stored_reactant, plus, minus, stored_cell, _free_index = load_endpoints(archive.path)
    assert stored_cell == pytest.approx(cell)
    assert not same_minimum(plus[0], stored_reactant[0], stored_cell)
    assert same_minimum(minus[0], stored_reactant[0], stored_cell)
    assert connected_endpoints(stored_reactant[0], plus[0], minus[0], stored_cell)
    assert not np.isfinite(plus[1]).all()
    assert archive.bin_path.is_file()
    archive.drop_log()
    assert not archive.bin_path.exists()


def test_an_old_absolute_archive_still_loads(tmp_path):
    path = tmp_path / "legacy.endpoints.npz"
    reactant = np.zeros((1, 1, 3), dtype=np.float32)
    plus = reactant.copy()
    plus[0, 0, 0] = np.float32(2.71)
    minus = reactant.copy()
    np.savez_compressed(
        path,
        reactant=reactant,
        plus=plus,
        minus=minus,
        cell=np.array([19.2088, 19.0118, 30.0]),
        free_index=np.array([0], dtype=np.int32),
    )
    loaded_reactant, loaded_plus, loaded_minus, cell, free_index = load_endpoints(path)
    assert free_index.tolist() == [0]
    assert not same_minimum(loaded_plus[0], loaded_reactant[0], cell)
    assert same_minimum(loaded_minus[0], loaded_reactant[0], cell)
    record = {"start": "rep1-search1"}
    stamp_endpoint(record, path.name, 0, STATUS_STORED)
    assert archive_error(path, [record], "legacy.starts.jsonl") is None


def test_resume_drops_the_row_that_missed_the_partial_file(tmp_path):
    path = tmp_path / "run.endpoints.npz"
    archive = EndpointArchive(path, 1)
    reactant = np.zeros((1, 3))
    archive.reset([1.0, 2.0, 3.0], [0])
    archive.add(reactant, reactant + [1.0, 0.0, 0.0], reactant + [0.0, 1.0, 0.0])
    archive.add(reactant, None, None)
    restored = EndpointArchive(path, 1)
    align_resume(restored, 1)
    assert restored.count == 1
    assert restored.add(reactant, reactant + [0.2, 0.0, 0.0], reactant + [0.0, 0.2, 0.0]) == STATUS_STORED
    restored.save()
    records = []
    for index, start in enumerate(("rep1-search1", "rep1-search2")):
        record = {"start": start}
        status = STATUS_STORED
        stamp_endpoint(record, restored.name, index, status)
        records.append(record)
    assert archive_error(path, records, "run.starts.jsonl") is None
    mismatched = EndpointArchive(path, 1)
    with pytest.raises(RuntimeError, match="endpoint rows"):
        align_resume(mismatched, 5)


def test_shipped_saddle_campaigns_meet_the_endpoint_rule():
    assert saddle_endpoint_gaps(ROOT) == []
