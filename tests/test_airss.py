"""AIRSS cell generation and the harness two-point step."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_airss import AirssLj38, selected_starts
from optbench.potentials import lennard_jones
from optbench.runners.airss_go import (
    TARBALL_SHA256,
    cell_text,
    cluster_lennard_jones,
    find_buildcell,
    load_wales_coordinates,
    positions_from_cell,
    reached_minimum,
    reference_energy,
    relax_positions,
    seed_words,
)

ROOT = Path(__file__).resolve().parents[1]


def test_wales_lj38_energy_is_the_published_minimum():
    coords = load_wales_coordinates()
    assert coords.shape == (38, 3)
    energy = reference_energy(coords)
    assert energy == pytest.approx(-173.92842631234794, abs=1e-9)
    assert energy == pytest.approx(-173.9284263, abs=5e-7)
    direct, _gradient = lennard_jones(coords)
    assert direct == pytest.approx(energy)


def test_energy_window_accepts_the_minimum_and_rejects_the_icosahedron():
    reference = reference_energy()
    assert reached_minimum(reference, reference, 0.001)
    assert reached_minimum(reference - 1e-4, reference, 0.001)
    assert reached_minimum(reference + 0.001, reference, 0.001)
    assert reached_minimum(reference + 0.001 + 1e-9, reference, 0.001) is False
    assert reached_minimum(-173.252378, reference, 0.001) is False
    assert reached_minimum(float("nan"), reference, 0.001) is False


def test_seed_words_are_eight_positive_integers_and_depend_on_the_start():
    first = seed_words(0, 0)
    assert len(first) == 8
    assert all(1 <= word < 2147483647 for word in first)
    assert seed_words(0, 0) == first
    assert seed_words(1, 0) != first
    assert seed_words(0, 1) != first


def test_cell_text_carries_the_example_directives_and_eight_seeds():
    text = cell_text(
        natoms=38,
        box=100,
        posamp=1.875,
        minsep=0.75,
        symmops="2-5",
        nform=1,
        seed=seed_words(3, 4),
    )
    assert "#CLUSTER" in text
    assert "#POSAMP=1.875" in text
    assert "#MINSEP=0.75" in text
    assert "#SYMMOPS=2-5" in text
    assert "#NFORM=1" in text
    assert "NUM=38" in text
    seed_line = next(line for line in text.splitlines() if line.startswith("#SEED="))
    assert len(seed_line.split("=", 1)[1].split()) == 8


def test_positions_from_cell_scales_fractional_coordinates():
    text = """%BLOCK LATTICE_ABC
    100.00000    100.00000    100.00000
     90.00000     90.00000     90.00000
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_FRAC
  Al        0.5000000        0.5000000        0.5000000
  Al        0.5100000        0.5000000        0.5000000
%ENDBLOCK POSITIONS_FRAC
"""
    pos = positions_from_cell(text, 2)
    assert pos == pytest.approx(np.array([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0]]))


def test_two_point_step_relaxes_a_dimer_to_the_pair_minimum():
    pos = np.array([[50.0, 50.0, 50.0], [51.5, 50.0, 50.0]])
    point, energy, calls, converged = relax_positions(pos, 100.0)
    distance = float(np.linalg.norm(point[0] - point[1]))
    assert converged
    assert calls > 0
    assert energy == pytest.approx(-1.0, abs=1e-8)
    assert distance == pytest.approx(2.0 ** (1.0 / 6.0), abs=1e-5)


def test_two_point_step_keeps_the_wales_minimum_inside_the_window():
    coords = load_wales_coordinates()
    point = coords - coords.mean(axis=0) + 50.0
    _point, energy, calls, converged = relax_positions(point, 100.0)
    reference = reference_energy(coords)
    assert converged
    assert calls < 500
    assert reached_minimum(energy, reference, 0.001)
    assert energy == pytest.approx(reference, abs=1e-3)


def test_template_pins_the_tarball_and_the_asv_grid_is_one_start(monkeypatch, tmp_path):
    import tomllib

    template = tomllib.loads((ROOT / "templates" / "airss" / "lj38.toml").read_text(encoding="utf-8"))
    assert template["commit"] == TARBALL_SHA256
    assert template["engine"] == "airss"
    assert template["problem"] == "globalopt/lj38"
    assert template["airss"]["posamp"] == 1.875
    assert template["airss"]["symmops"] == "2-5"
    assert template["failure_cap"] == 2000000
    assert AirssLj38.template_id == "airss/lj38"
    assert AirssLj38.problem_id == "globalopt/lj38"
    assert AirssLj38.param_names == ["seed", "start"]
    assert AirssLj38.params[0] == ["0"]
    assert AirssLj38.params[1] == ["0"]
    assert AirssLj38.cap == 2000000
    assert AirssLj38.repeat == 1
    assert AirssLj38.track_force_calls.unit == "force calls"
    monkeypatch.setenv("OPTBENCH_ASV_FULL", "1")
    assert selected_starts() == [str(index) for index in range(100)]
    monkeypatch.delenv("AIRSS_BUILDCELL", raising=False)
    import optbench.runners.airss_go as airss

    monkeypatch.setattr(airss.shutil, "which", lambda _name: None)
    monkeypatch.setattr(airss.Path, "home", lambda: tmp_path)
    with pytest.raises(NotImplementedError):
        AirssLj38().setup("0", "0")


def test_lj75_cap_scores_the_archived_start_count(tmp_path, monkeypatch):
    import json

    import optbench.runners.airss_go as airss
    from optbench.campaign import run

    def fake(job):
        return airss._capped(job, calls=100 + job["index"], energy=-300.0 - job["index"], structures=20)

    monkeypatch.setattr(airss, "search_start", fake)
    monkeypatch.setenv("AIRSS_BUILDCELL", "/bin/true")
    output = tmp_path / "lj75.dat"
    summary = run(
        ROOT,
        "globalopt/lj75",
        "airss/lj75",
        output=output,
        workers=1,
        work=tmp_path / "work",
    )
    text = output.read_text(encoding="utf-8")
    assert "cap 2000000" in text
    assert "n 200" in text
    assert summary["starts"] == 200
    assert summary["nfailed"] == 200
    assert summary["par2_mean"] == 4000000
    rows = [
        json.loads(line)
        for line in output.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in rows[:2]] == ["coords.1", "coords.2"]
    assert rows[-1]["start"] == "coords.200"
    assert rows[0]["energy"] == -300.0
    assert rows[-1]["energy"] == -499.0
    assert all(row["success"] is False and row["failure"] == "cap" and row["counter"] == "harness" for row in rows)


def test_checkpoint_resumes_at_the_next_structure(tmp_path, monkeypatch):
    import optbench.runners.airss_go as airss

    seen = []

    def fake_build(_binary, text, natoms, timeout=60.0):
        seen.append(text)
        return np.zeros((natoms, 3)), ["Al"] * natoms

    def fake_relax(_pos, _box, *, etol, maxsteps, evaluate):
        if len(seen) == 2:
            raise RuntimeError("interrupt")
        return np.zeros((2, 3)), -1.0, 11, True

    monkeypatch.setattr(airss, "build_structure", fake_build)
    monkeypatch.setattr(airss, "relax_positions", fake_relax)
    job = {
        "index": 4,
        "start": "coords.5",
        "buildcell": "/bin/true",
        "reference_energy": -10.0,
        "threshold": -9.0,
        "cap": 25,
        "safety": 4000,
        "code": "AIRSS",
        "code_version": "0.9.4",
        "commit": "abc",
        "tuning": "default",
        "natoms": 2,
        "box": 100.0,
        "posamp": 1.0,
        "minsep": 0.75,
        "symmops": "2-5",
        "nform": 1,
        "etol": 1e-10,
        "maxsteps": 10,
        "potential": "lennard-jones",
        "checkpoint": str(tmp_path / "4.json"),
    }
    with pytest.raises(RuntimeError, match="interrupt"):
        airss.search_start(job)
    assert (tmp_path / "4.json").is_file()
    first_seed = seen[0]
    seen.clear()

    def fake_relax_finish(_pos, _box, *, etol, maxsteps, evaluate):
        return np.zeros((2, 3)), -1.0, 11, True

    monkeypatch.setattr(airss, "relax_positions", fake_relax_finish)
    record = airss.search_start(job)
    assert record["force_calls"] == 33
    assert record["structures"] == 3
    assert record["success"] is False
    assert record["failure"] == "cap"
    assert seen[0] != first_seed
    assert not (tmp_path / "4.json").is_file()


def test_assemble_campaign_scores_shard_files_in_archive_order(tmp_path, monkeypatch):
    import json

    import optbench.runners.airss_go as airss
    from optbench.campaign import load_template, resolve_problem

    monkeypatch.setattr(airss, "_load_start_names", lambda *_args, **_kwargs: ["coords.1", "coords.2"])
    shard = tmp_path / "shard.starts.jsonl"
    shard.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "start": "coords.2",
                    "index": 1,
                    "force_calls": 5,
                    "success": False,
                    "failure": "cap",
                    "energy": -1.0,
                    "structures": 1,
                },
                {
                    "start": "coords.1",
                    "force_calls": 4,
                    "success": True,
                    "failure": None,
                    "energy": -2.0,
                    "structures": 1,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    summary = airss.assemble_campaign(
        ROOT,
        resolve_problem(ROOT, "globalopt/lj98"),
        load_template(ROOT, "airss/lj98"),
        [shard],
        tmp_path / "airss-0.9.4-lj98.dat",
    )
    text = (tmp_path / "airss-0.9.4-lj98.dat").read_text(encoding="utf-8")
    assert summary["starts"] == 2
    assert summary["nfailed"] == 1
    assert "cap 2000000" in text
    written = [
        json.loads(line)
        for line in (tmp_path / "airss-0.9.4-lj98.starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in written] == ["coords.1", "coords.2"]
    assert "index" not in written[0]


def test_compiled_kernel_matches_the_spec_potential():
    coords = load_wales_coordinates()
    energy, gradient = cluster_lennard_jones(coords)
    reference, reference_gradient = lennard_jones(coords)
    assert energy == pytest.approx(reference, abs=1e-8)
    assert gradient == pytest.approx(reference_gradient.reshape(-1), abs=1e-8)
    dimer = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    dimer_energy, dimer_gradient = cluster_lennard_jones(dimer)
    numpy_energy, numpy_gradient = lennard_jones(dimer)
    assert dimer_energy == pytest.approx(numpy_energy, abs=1e-12)
    assert dimer_gradient == pytest.approx(numpy_gradient, abs=1e-12)


def test_lj38_par2_uses_the_board_cap():
    import json
    import tomllib

    from optbench.runners.airss_go import search_cap
    from optbench.score import score_v2

    template = tomllib.loads((ROOT / "templates" / "airss" / "lj38.toml").read_text(encoding="utf-8"))
    problem = tomllib.loads((ROOT / "problems" / "globalopt" / "lj38.toml").read_text(encoding="utf-8"))
    cap = search_cap(problem, template)
    assert cap == 2000000
    starts = ROOT / "results" / "campaign" / "airss-0.9.4-lj38.starts.jsonl"
    records = [json.loads(line) for line in starts.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 100
    assert all(row["success"] for row in records)
    assert max(int(row["force_calls"]) for row in records) < cap
    metrics = score_v2(records, cap)
    text = (ROOT / "results" / "campaign" / "airss-0.9.4-lj38.dat").read_text(encoding="utf-8")
    assert "cap 2000000" in text
    assert f"par2_mean {metrics['par2_mean']:.12g}" in text
    assert "unset" not in text
    assert metrics["par2_mean"] == pytest.approx(metrics["mean"])
    assert metrics["successes"] == 100


def test_buildcell_returns_38_atoms_for_a_fixed_seed():
    try:
        binary = find_buildcell()
    except FileNotFoundError:
        pytest.skip("buildcell is not installed")
    from optbench.runners.airss_go import build_positions

    text = cell_text(
        natoms=38,
        box=100,
        posamp=1.875,
        minsep=0.75,
        symmops="2-5",
        nform=1,
        seed=seed_words(0, 0),
    )
    pos = build_positions(binary, text, 38)
    assert pos.shape == (38, 3)
    assert np.all(pos >= 0.0)
    assert np.all(pos < 100.0)
    again = build_positions(binary, text, 38)
    assert again == pytest.approx(pos)
