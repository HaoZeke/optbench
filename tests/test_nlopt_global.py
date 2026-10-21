"""NLopt global-optimization templates and the Lennard-Jones 38 judge."""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from optbench.campaign import list_templates, load_template
from optbench.potentials import lennard_jones, read_con
from optbench.runners.nlopt_global import (
    judge_global,
    objective_report,
    reference_minimum,
    search_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "88c424d4f458412787df96fcc95218acbca224fd"
CUBE_TEMPLATES = tuple(
    f"nloptglobal/{size}-{method}"
    for size in ("lj38", "lj75", "lj98", "lj150")
    for method in ("mlsl-lbfgs", "crs2", "isres", "stogo")
)
HOP_CAPS = {
    "nloptglobal/lj38-lbfgs-bh": 2_000_000,
    "nloptglobal/lj75-lbfgs-bh": 2_000_000,
    "nloptglobal/lj98-lbfgs-bh": 2_000_000,
    "nloptglobal/lj150-lbfgs-bh": 2_000_000,
}


def test_templates_pin_the_registry_commit():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "nloptglobal")
    assert pinned["commit"] == COMMIT
    assert pinned["name"] == "NLopt"
    assert pinned["ref"] == "v2.11.0"
    ids = {row["id"] for row in registry["template"]}
    for name in (*CUBE_TEMPLATES, *HOP_CAPS):
        template = load_template(ROOT, name)
        assert template["commit"] == COMMIT
        assert template["code_version"] == "2.11.0"
        assert template["code"] == "NLopt"
        assert template["engine"] == "nlopt"
        assert float(template["stop"]["energy_above_minimum"]) == 0.001
        assert name in ids
        if name in HOP_CAPS:
            assert int(template["stop"]["max_force_calls"]) == HOP_CAPS[name]
            assert float(template["nlopt"]["displacement"]) == 0.5
            assert float(template["nlopt"]["temperature"]) == 0.8
            assert float(template["nlopt"]["max_move"]) == 0.2
            assert float(template["nlopt"]["converged_force"]) == 0.001
            assert int(template["nlopt"]["max_iterations"]) == 1000
            assert int(template["nlopt"]["local_maxeval"]) == 4000
            assert "maxstep" not in template["nlopt"]
            assert "core_radius" not in template["nlopt"]
            assert "tolg" not in template["nlopt"]
            assert "max_move" in template["note"]
            assert "1e-12" in template["note"]
        else:
            size = name.split("/")[1].split("-", 1)[0]
            assert template["problem"] == f"globalopt/{size}"
            assert int(template["stop"]["max_force_calls"]) == 2_000_000
            assert float(template["domain"]["half_width"]) == 6.0


def test_nlopt_templates_are_the_files_on_disk():
    files = sorted("nloptglobal/" + path.stem for path in (ROOT / "templates" / "nloptglobal").glob("*.toml"))
    found = [row["id"] for row in list_templates(ROOT) if row.get("engine") == "nlopt"]
    assert found == files
    assert set(CUBE_TEMPLATES).issubset(files)
    assert set(HOP_CAPS).issubset(files)


def test_reference_minimum_matches_the_polished_cluster():
    energy, force, coords = reference_minimum()
    assert energy == pytest.approx(-173.92842659062865, abs=1e-9)
    assert force < 1e-4
    assert coords.size == 114
    again, gradient = lennard_jones(coords)
    assert again == pytest.approx(energy)
    assert float(np.linalg.norm(gradient)) == pytest.approx(force)


def test_every_archived_start_and_the_minimum_fit_in_the_cube():
    import tarfile

    from optbench.campaign import geometry_archive, resolve_problem

    _energy, _force, minimum = reference_minimum()
    minimum = minimum.reshape(-1, 3)
    minimum = minimum - minimum.mean(axis=0)
    archive = geometry_archive(ROOT, resolve_problem(ROOT, "globalopt/lj38"))
    with tarfile.open(archive) as packed:
        names = sorted(
            (name for name in packed.getnames() if name.endswith(".con")),
            key=lambda name: int(Path(name).stem),
        )
        assert len(names) == 100
        for name in names:
            text = packed.extractfile(name).read().decode("utf-8", errors="replace")
            pos, _box = read_con(text)
            flat = np.asarray(pos, dtype=np.float64).reshape(-1)
            lower, upper = search_bounds(flat, 6.0)
            assert np.all(flat >= lower - 1e-8)
            assert np.all(flat <= upper + 1e-8)
            center = flat.reshape(-1, 3).mean(axis=0)
            placed = (minimum + center).reshape(-1)
            assert np.all(placed >= lower - 1e-8)
            assert np.all(placed <= upper + 1e-8)


def test_the_cambridge_rule_is_at_or_below_the_minimum():
    reference = -173.928427
    tolerance = 0.001
    success, failure, reasons = judge_global(
        calls=12,
        cap=100,
        energy=reference + tolerance,
        finite=True,
        nlopt_code=2,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert success is True
    assert failure is None
    assert reasons == []
    below, below_failure, _reasons = judge_global(
        calls=12,
        cap=100,
        energy=reference - 1.0,
        finite=True,
        nlopt_code=2,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert below is True
    assert below_failure is None


def test_the_budget_is_a_cap_and_a_high_energy_is_the_wrong_point():
    reference = -173.928427
    tolerance = 0.001
    success, failure, _reasons = judge_global(
        calls=100,
        cap=100,
        energy=-160.0,
        finite=True,
        nlopt_code=5,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert success is False
    assert failure == "cap"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-164.0,
        finite=True,
        nlopt_code=4,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert failure == "wrong_stationary"
    success, failure, _reasons = judge_global(
        calls=40,
        cap=100,
        energy=-80.0,
        finite=True,
        nlopt_code=1,
        error="",
        reference=reference,
        tolerance=tolerance,
    )
    assert failure == "wrong_stationary"
    success, failure, _reasons = judge_global(
        calls=3,
        cap=100,
        energy=None,
        finite=False,
        nlopt_code=-1,
        error="RuntimeError: boom",
        reference=reference,
        tolerance=tolerance,
    )
    assert failure == "crash"


def test_a_coincident_pair_is_the_true_singularity():
    stacked = np.zeros(6)
    raw, grad = lennard_jones(stacked)
    assert math.isinf(raw) and raw > 0.0
    assert np.all(grad == 0.0)
    reported, reported_grad, usable = objective_report(float(raw), grad, stacked)
    assert usable is False
    assert math.isinf(reported) and reported > 0.0
    assert np.all(reported_grad == 0.0)
    energy, _force, coords = reference_minimum()
    kept, kept_grad, keep = objective_report(energy, np.zeros(coords.size), coords)
    assert keep is True
    assert kept == pytest.approx(energy)
    assert kept_grad.shape == (coords.size,)


def test_a_short_mlsl_start_is_counted_by_the_harness():
    nlopt = pytest.importorskip("nlopt")
    assert (nlopt.version_major(), nlopt.version_minor(), nlopt.version_bugfix()) == (2, 11, 0)
    import tarfile

    from optbench.campaign import geometry_archive, resolve_problem
    from optbench.runners.nlopt_global import _settings, run_job

    template = load_template(ROOT, "nloptglobal/lj38-mlsl-lbfgs")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    settings = _settings(problem, template)
    archive = geometry_archive(ROOT, problem)
    with tarfile.open(archive) as packed:
        text = packed.extractfile("100_lj38/0.con").read().decode("utf-8", errors="replace")
    pos, _box = read_con(text)
    from optbench.globalopt import reference_energy

    reference = reference_energy(problem)
    job = {
        "index": 0,
        "start": "0",
        "coords": np.asarray(pos, dtype=np.float64).reshape(-1).tolist(),
        "seed": 0,
        "algorithm": "GN_MLSL",
        "local": "LD_LBFGS",
        "local_ftol_rel": 1e-15,
        "local_xtol_rel": 1e-7,
        "population": None,
        "half_width": settings["half_width"],
        "cap": 30,
        "reference": reference,
        "tolerance": settings["window"],
    }
    record = run_job(job)
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["schema"] == "optbench.start/v2"
    assert 0 < record["force_calls"] <= 30
    assert record["failure"] in {"cap", "divergence", "crash", "wrong_stationary"}
    assert record["success"] is False


def test_asv_classes_follow_the_templates():
    import benchmarks.bench_nloptglobal as bench

    found = [row for row in list_templates(ROOT) if row.get("engine") == "nlopt"]
    assert {row["id"] for row in found} >= set(CUBE_TEMPLATES) | set(HOP_CAPS)
    for template in found:
        slug = str(template["id"]).split("/", 1)[1]
        name = "Nlopt" + "".join(part.capitalize() for part in slug.split("-"))
        cls = getattr(bench, name)
        assert cls.template_id == template["id"]
        assert cls.problem_id == template["problem"]
        assert cls.cap == int(template["stop"]["max_force_calls"])
        assert cls.params == [["0"], ["0"]]


def test_a_close_pair_is_the_true_repulsion():
    pair = np.zeros(6)
    pair[3] = 0.4
    energy, grad = lennard_jones(pair)
    assert math.isfinite(energy)
    assert energy > 1000.0
    eps = 1e-7
    finite = np.zeros_like(grad)
    for index in range(pair.size):
        pair[index] += eps
        above, _grad = lennard_jones(pair)
        pair[index] -= 2.0 * eps
        below, _grad = lennard_jones(pair)
        pair[index] += eps
        finite[index] = (above - below) / (2.0 * eps)
    scale = max(1.0, float(np.max(np.abs(grad))))
    assert np.max(np.abs(finite - grad)) / scale < 1e-4


def test_a_polished_basin_is_a_cambridge_success():
    pytest.importorskip("nlopt")
    from optbench.campaign import resolve_problem
    from optbench.globalopt import reference_energy
    from optbench.runners.nlopt_global import _job, _settings, run_job

    template = load_template(ROOT, "nloptglobal/lj38-lbfgs-bh")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    settings = _settings(problem, template)
    _energy, _force, coords = reference_minimum()
    rng = np.random.default_rng(2)
    trial = coords + rng.normal(scale=0.05, size=coords.size)
    trial = trial.reshape(-1, 3)
    trial -= trial.mean(axis=0)
    reference = reference_energy(problem)
    job = _job(0, "gm", trial.reshape(-1), settings, reference, settings["window"])
    job["cap"] = 300
    record = run_job(job)
    assert record["success"] is True
    assert record["failure"] is None
    assert record["energy"] <= reference + 0.001
    assert record["energy"] == pytest.approx(-173.92842659062865, abs=1e-6)
    assert 0 < record["force_calls"] < 300
    assert record["verify_calls"] == 1
    assert record["counter"] == "harness"


def test_the_native_kernel_matches_the_harness_potential():
    from optbench.fastpot import lj_gradient

    energy, _force, coords = reference_minimum()
    got = lj_gradient(coords)
    if got is None:
        pytest.skip("native Lennard-Jones kernel is unavailable")
    value, grad, minimum = got
    raw, raw_grad = lennard_jones(coords)
    assert minimum > 1.0
    assert value == pytest.approx(raw, abs=1e-9)
    assert value == pytest.approx(energy, abs=1e-9)
    assert np.max(np.abs(grad - raw_grad)) < 1e-8
    pair = np.zeros(6)
    pair[3] = 0.4
    got = lj_gradient(pair)
    raw, raw_grad = lennard_jones(pair)
    assert math.isfinite(got[0]) and got[0] > 1000.0
    assert got[0] == pytest.approx(raw, rel=1e-9, abs=1e-6)
    scale = max(1.0, float(np.max(np.abs(raw_grad))))
    assert np.max(np.abs(got[1] - raw_grad)) / scale < 1e-8
    coincident = np.zeros(6)
    got = lj_gradient(coincident)
    raw, raw_grad = lennard_jones(coincident)
    assert got[2] == pytest.approx(0.0, abs=1e-15)
    assert math.isinf(got[0]) and got[0] > 0.0
    assert math.isinf(raw) and raw > 0.0
    assert np.all(got[1] == 0.0)
    assert np.all(raw_grad == 0.0)


def test_cambridge_minima_match_the_harness_potential():
    from optbench.ccd import LJ_SIZES, lennard_jones_minimum

    for size in LJ_SIZES:
        record = lennard_jones_minimum(size)
        harness, _gradient = lennard_jones(record["positions"])
        positions = np.asarray(record["positions"], dtype=np.float64).reshape(-1, 3)
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(distances, np.inf)
        assert float(distances.min()) > 0.9
        assert math.isfinite(harness)
        assert harness == pytest.approx(float(record["energy"]), abs=1e-6)


def test_the_c_quench_matches_the_python_quench(monkeypatch):
    from optbench.counter import HarnessCounter
    from optbench.fastpot import quench_library
    from optbench.runners.nlopt_global import _quench

    if quench_library() is None:
        pytest.skip("native quench is unavailable")
    _energy, _force, coords = reference_minimum()
    rng = np.random.default_rng(2)
    trial = coords + rng.normal(scale=0.05, size=coords.size)
    trial = trial.reshape(-1, 3)
    trial -= trial.mean(axis=0)
    trial = trial.reshape(-1)
    kwargs = {
        "cap": 20000,
        "local_max": 4000,
        "converged_force": 0.001,
        "max_move": 0.2,
        "max_iterations": 1000,
    }
    monkeypatch.setenv("OPTBENCH_PYTHON_QUENCH", "1")
    python_counter = HarnessCounter(cache=True, cap=20000)
    python_point, python_energy, python_code = _quench(python_counter, trial, **kwargs)
    monkeypatch.setenv("OPTBENCH_PYTHON_QUENCH", "0")
    native_counter = HarnessCounter(cache=True, cap=20000)
    native_point, native_energy, native_code = _quench(native_counter, trial, **kwargs)
    assert native_counter.calls == python_counter.calls
    assert native_energy == python_energy
    assert native_code == python_code
    assert np.array_equal(native_point, python_point)


def test_a_checkpoint_resumes_the_same_trajectory(tmp_path: Path):
    pytest.importorskip("nlopt")
    from optbench.campaign import resolve_problem
    from optbench.globalopt import reference_energy
    from optbench.runners.nlopt_global import _job, _settings, run_job

    template = load_template(ROOT, "nloptglobal/lj38-lbfgs-bh")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    settings = _settings(problem, template)
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(38, 3))
    coords -= coords.mean(axis=0)
    reference = reference_energy(problem)
    job = _job(0, "cloud", coords.reshape(-1), settings, reference, settings["window"])
    job["cap"] = 2500
    job["local_maxeval"] = 400
    full = run_job(dict(job))
    partial_job = dict(job)
    partial_job["checkpoint"] = str(tmp_path / "progress.json")
    partial_job["checkpoint_calls"] = 0
    partial_job["max_hops"] = 1
    partial = run_job(partial_job)
    assert partial["force_calls"] < full["force_calls"]
    saved = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "optbench.bh-progress/v3"
    assert saved["potential"] == "lj-cut"
    assert int(saved["calls"]) == partial["force_calls"]
    resumed = run_job(dict(partial_job, max_hops=None))
    assert resumed["force_calls"] == full["force_calls"]
    assert resumed["energy"] == full["energy"]
    assert resumed["nlopt_code"] == full["nlopt_code"]
    assert resumed["failure"] == full["failure"]
    assert resumed["success"] == full["success"]


def test_a_killed_lj98_start_resumes_its_checkpoint(tmp_path: Path):
    """SIGKILL leaves the last finished hop, and that file resumes exactly."""
    pytest.importorskip("nlopt")
    import os
    import signal
    import subprocess
    import sys
    import time

    from optbench.campaign import resolve_problem
    from optbench.globalopt import energy_tolerance, reference_energy
    from optbench.runners.nlopt_global import _job, _load_geometries, _restore_basin, _settings, run_job

    shard = tmp_path / "shard"
    shard.mkdir()
    child = tmp_path / "child.py"
    child.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "os.environ.setdefault('OMP_NUM_THREADS', '1')",
                "from optbench.campaign import load_template, resolve_problem",
                "from optbench.runners.nlopt_global import run_basin_shard",
                f"root = Path({str(ROOT)!r})",
                "problem = resolve_problem(root, 'globalopt/lj98')",
                "template = load_template(root, 'nloptglobal/lj98-lbfgs-bh')",
                "run_basin_shard(",
                "    root, problem, template, [90], workers=1,",
                f"    shard_dir=Path({str(shard)!r}), cap=50_000_000, checkpoint_calls=0,",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(child)],
        cwd=ROOT,
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    progress = shard / "090.progress.json"
    deadline = time.perf_counter() + 90.0
    killed = None
    while time.perf_counter() < deadline:
        if progress.is_file():
            try:
                killed = json.loads(progress.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                killed = None
            if killed and int(killed.get("calls", 0)) >= 15_000:
                break
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    assert killed is not None and int(killed["calls"]) >= 15_000
    killed = json.loads(progress.read_text(encoding="utf-8"))
    assert killed["schema"] == "optbench.bh-progress/v3"
    assert killed["potential"] == "lj-cut"
    assert killed["start"] == "coords.91"
    assert killed["index"] == 90

    problem = resolve_problem(ROOT, "globalopt/lj98")
    template = load_template(ROOT, "nloptglobal/lj98-lbfgs-bh")
    settings = _settings(problem, template)
    loaded = _load_geometries(
        ROOT,
        problem,
        None,
        ROOT / "results" / "campaign" / "geometries",
        shard / "geometries.dat",
    )
    name, coords = loaded[90]
    base = _job(90, name, coords, settings, reference_energy(problem), energy_tolerance(problem))
    base["cap"] = 50_000_000
    base["checkpoint_calls"] = 0
    restored = _restore_basin(killed, base, len(base["coords"]))
    assert restored is not None
    assert restored["calls"] == int(killed["calls"])
    assert restored["rng"].bit_generator.state == killed["rng"]
    assert np.array_equal(restored["current"], np.asarray(killed["current"], dtype=np.float64))
    assert np.array_equal(restored["best"], np.asarray(killed["best"], dtype=np.float64))

    def resume(path: Path, hops: int) -> dict:
        path.write_text(json.dumps(killed) + "\n", encoding="utf-8")
        job = dict(base)
        job["checkpoint"] = str(path)
        job["max_hops"] = hops
        return run_job(job)

    resume(shard / "idle.progress.json", 0)
    idle = json.loads((shard / "idle.progress.json").read_text(encoding="utf-8"))
    for key in ("calls", "code", "current", "current_energy", "best", "best_energy", "rng", "seed", "start"):
        assert idle[key] == killed[key]
    resume(shard / "left.progress.json", 2)
    resume(shard / "right.progress.json", 2)
    left = json.loads((shard / "left.progress.json").read_text(encoding="utf-8"))
    right = json.loads((shard / "right.progress.json").read_text(encoding="utf-8"))
    assert int(left["calls"]) > int(killed["calls"])
    for key in ("calls", "code", "current", "current_energy", "best", "best_energy", "rng"):
        assert left[key] == right[key]


def test_a_shard_keeps_finished_starts(tmp_path: Path):
    pytest.importorskip("nlopt")
    from optbench.campaign import resolve_problem
    from optbench.runners.nlopt_global import assemble_basin_shard, run_basin_shard

    problem = resolve_problem(ROOT, "globalopt/lj38")
    template = load_template(ROOT, "nloptglobal/lj38-lbfgs-bh")
    shard = tmp_path / "shard"
    output = tmp_path / "row.dat"
    first = run_basin_shard(ROOT, problem, template, [0, 1], workers=2, shard_dir=shard, cap=400)
    assert sorted(path.name for path in shard.glob("*.done.json")) == ["000.done.json", "001.done.json"]
    assert list(shard.glob("*.progress.json")) == []
    second = run_basin_shard(ROOT, problem, template, [0, 1], workers=1, shard_dir=shard, cap=400)
    assert [row["force_calls"] for row in sorted(second, key=lambda row: row["index"])] == [
        row["force_calls"] for row in sorted(first, key=lambda row: row["index"])
    ]
    assert assemble_basin_shard(ROOT, problem, template, output, shard_dir=shard, expected=100) is None
    assert not output.exists()
    summary = assemble_basin_shard(ROOT, problem, template, output, shard_dir=shard, expected=2)
    assert summary is not None
    assert summary["starts"] == 2
    assert output.is_file()
    assert output.with_suffix(".starts.jsonl").is_file()
    assert output.with_suffix(".v2.json").is_file()


def test_wales_cluster_archives_load_numbered_coords(tmp_path: Path):
    from optbench.campaign import resolve_problem
    from optbench.runners.nlopt_global import _load_geometries

    expected = {
        "globalopt/lj75": (200, 75),
        "globalopt/lj98": (100, 98),
        "globalopt/lj150": (100, 150),
    }
    for problem_id, (count, atoms) in expected.items():
        loaded = _load_geometries(
            ROOT,
            resolve_problem(ROOT, problem_id),
            None,
            tmp_path / problem_id.replace("/", "-"),
            tmp_path / f"{problem_id.replace('/', '-')}.dat",
        )
        assert len(loaded) == count
        assert loaded[0][0] == "coords.1"
        assert loaded[-1][0] == f"coords.{count}"
        assert loaded[0][1].size == atoms * 3
