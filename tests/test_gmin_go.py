"""GMIN basin-hopping templates and the native-potential matched run."""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

import pytest

from optbench.campaign import list_templates
from optbench.runners.gmin_go import (
    REVISION,
    SLUGS,
    TARBALL_SHA,
    VERSION,
    _spec,
    data_text,
    gmin_binary,
    load_starts,
    parse_gmin_output,
    template_id,
)
from optbench.campaign import geometry_archive

ROOT = Path(__file__).resolve().parents[1]


def test_registry_pins_the_svn_revision_and_the_tarball():
    import tomllib

    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "gmin")
    assert pinned["name"] == "GMIN"
    assert pinned["commit"] == REVISION
    assert pinned["committed"] == "2026-10-18"
    assert TARBALL_SHA in pinned["note"]
    assert "NPCALL" in pinned["note"]


def test_templates_use_the_published_keywords():
    found = [row for row in list_templates(ROOT) if row.get("engine") == "gmin-go"]
    assert {row["id"] for row in found} == {template_id(slug) for slug in SLUGS}
    for template in found:
        assert template["code"] == "GMIN"
        assert template["code_version"] == VERSION
        assert template["commit"] == REVISION
        assert template["tuning"] == "default"
        gmin = template["gmin"]
        assert float(gmin["sloppy_rms"]) == 0.01
        assert float(gmin["ediff"]) == 0.001
        assert float(gmin["tfac"]) == 1.0
        spec = _spec(ROOT, template["id"].split("/", 1)[1])
        text = data_text(gmin, index=7, reference=spec["reference"], cap=spec["cap"])
        assert "SLOPPYCONV 0.01" in text
        assert "EDIFF 0.001" in text
        assert "RANSEED 7" in text
        assert "SUPPRESS" in text
        assert f"TARGET {spec['reference']:.12f}" in text
        if gmin["potential"] == "tip4p":
            assert "TIP 4" in text
            assert "TEMPERATURE 5.0" in text
            assert "STEP 0.6 0.0 0.9 200" in text
            assert "TIGHTCONV 0.0001" in text
            assert "SORT" not in text
        else:
            assert "TEMPERATURE 0.8" in text
            assert "STEP 0.36 0.4" in text
            assert "TIGHTCONV 0.001" in text
            assert "SORT" in text
        if gmin["potential"] == "blj":
            assert "BLJCLUSTER_NOCUT 42 1.0 1.0 1.15 1.3" in text
        if spec["cap"] is None:
            assert "MAX_NPCALL" not in text
        else:
            assert f"MAX_NPCALL {spec['cap']}" in text


def test_archives_have_the_problem_start_count():
    for slug in SLUGS:
        spec = _spec(ROOT, slug)
        starts = load_starts(
            geometry_archive(ROOT, spec["problem"]),
            kind=spec["kind"],
            lines=spec["lines"],
        )
        assert len(starts) == int(spec["problem"]["sample"]["starts"])
        assert starts[0][0] == 0 or slug in {"lj75", "lj98", "lj150"}


def test_parser_reads_the_first_encounter_line():
    text = """
mc> target hit, quenches, function calls, time: 1 12 3456         0.1000000000
Lowest Minimum      1 Energy=     -173.9284270
finalio> Number of potential calls               3500
"""
    parsed = parse_gmin_output(text)
    assert parsed["hit"] is True
    assert parsed["force_calls"] == 3456
    assert parsed["quenches"] == 12
    assert parsed["energy"] == pytest.approx(-173.928427)
    assert parsed["potential_calls"] == 3500


def _trace_records(path: Path):
    blob = path.read_bytes()
    offset = 0
    records = []
    while offset + 12 <= len(blob):
        count = struct.unpack_from("<i", blob, offset)[0]
        energy = struct.unpack_from("<d", blob, offset + 4)[0]
        need = 4 + 8 + 16 * count + 4
        if count <= 0 or offset + need > len(blob):
            break
        import numpy as np

        coords = np.frombuffer(blob, dtype="<f8", count=count, offset=offset + 12).copy()
        gradient = np.frombuffer(blob, dtype="<f8", count=count, offset=offset + 12 + 8 * count).copy()
        # The trailing integer is 1 when POTENTIAL was asked for a gradient.
        gradt = struct.unpack_from("<i", blob, offset + 12 + 16 * count)[0]
        records.append((energy, coords, gradient, bool(gradt)))
        offset += need
    return records, len(blob) - offset


def _traced(tmp_path: Path, data: str, coords: str) -> tuple[dict, list]:
    binary = gmin_binary()
    if binary is None:
        pytest.skip("GMIN is not built")
    (tmp_path / "data").write_text(data, encoding="utf-8")
    (tmp_path / "coords").write_text(coords, encoding="utf-8")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPTBENCH_GMIN_TRACE"] = str(tmp_path / "trace.bin")
    completed = subprocess.run(
        [str(binary)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    text = completed.stdout.decode("utf-8", errors="replace")
    assert completed.returncode == 0, text[-500:]
    parsed = parse_gmin_output(text)
    records, leftover = _trace_records(tmp_path / "trace.bin")
    assert leftover == 0
    assert parsed["potential_calls"] == len(records)
    assert parsed["force_calls"] <= len(records)
    return parsed, records


def test_lennard_jones_calls_match_the_harness(tmp_path: Path):
    import numpy as np

    from optbench.counter import HarnessCounter
    from optbench.potentials import lennard_jones

    rng = np.random.default_rng(2)
    coords = "\n".join(" ".join(f"{value:.16e}" for value in row) for row in rng.normal(size=(13, 3)))
    data = """SLOPPYCONV 0.01
TIGHTCONV 0.001
CENTRE
SORT
MAXIT 250 500
STEPS 4 1.0
STEP 0.36 0.4
TEMPERATURE 0.8
EDIFF 0.001
TARGET -44.326801
RANSEED 3
SUPPRESS
"""
    parsed, records = _traced(tmp_path, data, coords + "\n")
    counter = HarnessCounter(cache=False)
    worst_energy = 0.0
    worst_gradient = 0.0
    gradient_calls = 0
    for energy, coords_row, gradient, gradt in records:
        counter.charge(tuple(float(value) for value in coords_row))
        harness_energy, harness_gradient = lennard_jones(coords_row)
        worst_energy = max(worst_energy, abs(float(harness_energy) - energy))
        if gradt:
            gradient_calls += 1
            worst_gradient = max(worst_gradient, float(np.max(np.abs(harness_gradient - gradient))))
    assert gradient_calls
    assert counter.calls == len(records)
    assert counter.calls == parsed["potential_calls"]
    assert worst_energy < 1e-8
    assert worst_gradient < 1e-6


def test_binary_lennard_jones_calls_match_the_harness(tmp_path: Path):
    import numpy as np

    from optbench.counter import HarnessCounter
    from optbench.potentials import binary_lennard_jones

    rng = np.random.default_rng(4)
    positions = rng.normal(size=(8, 3))
    species = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    coords = "\n".join(" ".join(f"{value:.16e}" for value in row) for row in positions)
    data = """SLOPPYCONV 0.01
TIGHTCONV 0.001
BLJCLUSTER_NOCUT 3 1.0 1.0 1.15 1.3
CENTRE
SORT
MAXIT 250 500
STEPS 2 1.0
STEP 0.36 0.4
TEMPERATURE 0.8
EDIFF 0.001
TARGET -100
RANSEED 4
SUPPRESS
"""
    _parsed, records = _traced(tmp_path, data, coords + "\n")
    counter = HarnessCounter(cache=False)
    worst = 0.0
    worst_gradient = 0.0
    gradient_calls = 0
    for energy, coords_row, gradient, gradt in records:
        counter.charge(tuple(float(value) for value in coords_row))
        harness_energy, harness_gradient = binary_lennard_jones(coords_row, species)
        worst = max(worst, abs(float(harness_energy) - energy))
        if gradt:
            gradient_calls += 1
            worst_gradient = max(worst_gradient, float(np.max(np.abs(harness_gradient - gradient))))
    assert gradient_calls
    assert counter.calls == len(records)
    assert worst < 1e-8
    assert worst_gradient < 1e-6


def test_tip4p_calls_match_the_harness_constants(tmp_path: Path):
    import math

    import numpy as np

    from optbench.counter import HarnessCounter
    from optbench.tip4p import TIP4P_ANGLE_DEG, TIP4P_ROH, TIP4P_ROM, _rotate, site_energy_forces

    half = math.radians(TIP4P_ANGLE_DEG) / 2.0
    hydrogen_1 = np.array([math.sin(half) * TIP4P_ROH, 0.0, -math.cos(half) * TIP4P_ROH])
    hydrogen_2 = np.array([-math.sin(half) * TIP4P_ROH, 0.0, -math.cos(half) * TIP4P_ROH])
    sites = np.stack(
        [
            np.zeros(3),
            hydrogen_1,
            hydrogen_2,
            np.array([0.0, 0.0, -TIP4P_ROM]),
        ]
    )
    rng = np.random.default_rng(5)
    molecules = rng.normal(size=(4, 6))
    rows = []
    centres = molecules[:, :3]
    axes = molecules[:, 3:]
    blocked = np.concatenate([centres.reshape(-1), axes.reshape(-1)])
    # GMIN reads centres first, then angle axes, three components per line.
    values = blocked.reshape(-1, 3)
    coords = "\n".join(" ".join(f"{value:.16e}" for value in row) for row in values)

    def harness_energy(flat):
        count = flat.size // 6
        centre = flat[: 3 * count].reshape(count, 3)
        axis = flat[3 * count :].reshape(count, 3)
        placed = np.zeros((count, 4, 3))
        for index in range(count):
            rotated, _derivative = _rotate(axis[index], sites)
            placed[index] = centre[index] + rotated
        energy, _gradient = site_energy_forces(placed)
        return energy

    data = """SLOPPYCONV 0.01
TIGHTCONV 0.0001
TIP 4
CENTRE
MAXIT 1000 1000
STEPS 2 1.0
STEP 0.6 0.0 0.9 200
TEMPERATURE 5.0
EDIFF 0.001
TARGET -1000000
RANSEED 5
SUPPRESS
"""
    _parsed, records = _traced(tmp_path, data, coords + "\n")
    counter = HarnessCounter(cache=False)
    worst = 0.0
    for energy, coords_row, _gradient, _gradt in records:
        counter.charge(tuple(float(value) for value in coords_row))
        worst = max(worst, abs(harness_energy(coords_row) - energy))
    assert counter.calls == len(records)
    assert worst < 1e-5
