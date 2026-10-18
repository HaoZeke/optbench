"""Harness counting for a PANNA training loop, without the Adam libraries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optbench.ml_landscape import build_landscape, parse_fingerprints
from optbench.panna_train import TrainingPaused, _clear_checkpoint, train_starts


def _landscape():
    text = "\n".join(["BP", "A", "0.0", "1 0", "G1 A A 0.2 0.0 4.0"])
    positions = np.array([[[0.0, 0.0, 0.0], [1.4, 0.2, 0.0]]], dtype=np.float64)
    return build_landscape(
        positions,
        ("A", "A"),
        np.eye(3) * 20.0,
        np.array([False, False, False]),
        np.array([0.5]),
        np.zeros((1, 2, 3)),
        parse_fingerprints(text),
        with_forces=False,
        hidden=(2, 2),
    )


def _template() -> dict:
    return {
        "id": "panna/toy",
        "problem": "mlopt/emt-pt38",
        "code": "PANNA",
        "code_version": "2.1.0",
        "commit": "132a2df662598ecbbb2c9b8d37b992bb6ada6348",
        "tuning": "default",
    }


def _hold(parameters, gradient, active):
    return parameters


def _nudge(parameters, gradient, active):
    updated = np.array(parameters, copy=True)
    updated[active, 0] += 1.0e-3
    return updated


def _identity(row: dict) -> tuple:
    return (
        row["start"],
        row["force_calls"],
        row["success"],
        row["failure"],
        row["schema"],
        row["counter"],
    )


def test_a_start_under_the_stop_is_one_harness_call():
    records = train_starts(
        _landscape(),
        _template(),
        cap=5,
        tolerance=1.0e9,
        energy_coefficient=1.0,
        force_coefficient=0.0,
        stepper=_hold,
        starts=[0, 1],
        seed=20261015,
        progress=False,
    )
    assert [row["force_calls"] for row in records] == [1, 1]
    assert all(row["success"] for row in records)
    assert all(row["failure"] is None for row in records)
    assert all(row["counter"] == "harness" for row in records)
    assert all(row["schema"] == "optbench.start/v2" for row in records)


def test_a_start_that_never_stops_is_a_cap_failure():
    records = train_starts(
        _landscape(),
        _template(),
        cap=3,
        tolerance=-1.0,
        energy_coefficient=1.0,
        force_coefficient=0.0,
        stepper=_nudge,
        starts=[4],
        seed=20261015,
        progress=False,
    )
    assert records[0]["force_calls"] == 3
    assert records[0]["success"] is False
    assert records[0]["failure"] == "cap"
    assert records[0]["schema"] == "optbench.start/v2"


def test_a_shrinking_batch_still_counts_the_cap():
    records = train_starts(
        _landscape(),
        _template(),
        cap=3,
        tolerance=-1.0,
        energy_coefficient=1.0,
        force_coefficient=0.0,
        stepper=_nudge,
        starts=[0, 1, 2],
        seed=20261015,
        progress=False,
    )
    assert [row["force_calls"] for row in records] == [3, 3, 3]
    assert [row["failure"] for row in records] == ["cap", "cap", "cap"]


def test_a_repeated_parameter_vector_is_divergence():
    records = train_starts(
        _landscape(),
        _template(),
        cap=5,
        tolerance=-1.0,
        energy_coefficient=1.0,
        force_coefficient=0.0,
        stepper=_hold,
        starts=[0],
        seed=20261015,
        progress=False,
    )
    assert records[0]["force_calls"] == 1
    assert records[0]["success"] is False
    assert records[0]["failure"] == "divergence"


def test_a_checkpoint_resumes_the_unfinished_cap(tmp_path: Path):
    landscape = _landscape()
    template = _template()
    kwargs = dict(
        cap=4,
        tolerance=-1.0,
        energy_coefficient=1.0,
        force_coefficient=0.0,
        stepper=_nudge,
        starts=[0, 2],
        seed=20261015,
        progress=False,
    )
    full = train_starts(landscape, template, **kwargs)
    checkpoint = tmp_path / "run.npz"
    with pytest.raises(TrainingPaused):
        train_starts(
            landscape,
            template,
            checkpoint=checkpoint,
            checkpoint_every=1,
            pause_after=2,
            **kwargs,
        )
    resumed = train_starts(landscape, template, checkpoint=checkpoint, **kwargs)
    assert [_identity(row) for row in resumed] == [_identity(row) for row in full]
    _clear_checkpoint(checkpoint)
    assert not checkpoint.exists()
