"""The aluminum AKMC cost is the saddle plus minimisation sum.

The sum is the ranked counter. A four-atom ext_pot saddle search checks
it against HarnessCounter. The 151-atom catalogues stay on eam_al.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from optbench.board import _comparable
from optbench.dat import parse_lenient
from optbench.eon_amd import COUNT_KEY, _write_result, search_cost
from optbench.score import eon_search_calls

ROOT = Path(__file__).resolve().parents[1]

CATALOGUES = (
    (
        "eon-3.6.0-aladatom-akmc",
        "eon/aladatom-akmc",
        "accelerated-md/aladatom",
        108435,
        855329109.9,
        0.2427275727,
        128,
    ),
    (
        "eon-3.6.0-standard-aladatom-akmc",
        "eon/standard-aladatom-akmc",
        "accelerated-md/standard_aladatom",
        1890241,
        38023490.0,
        0.8257069791,
        2336,
    ),
)


def test_eon_search_calls_sums_saddle_and_minimisation_only():
    assert eon_search_calls(31, 49) == 80
    assert eon_search_calls(0, 0) == 0
    with pytest.raises(ValueError):
        eon_search_calls(-1, 4)
    with pytest.raises(ValueError):
        eon_search_calls(4, -1)
    with pytest.raises(ValueError):
        eon_search_calls(1.0, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        eon_search_calls(True, 1)  # type: ignore[arg-type]
    text = "\n".join(
        [
            "header",
            "rule",
            "1 random 0.2 1.0 10 4 9 good-1",
            "2 random 0.2 1.0 3 1 7 repeat-1",
        ]
    )
    cost = search_cost(text)
    assert cost["force_calls"] == eon_search_calls(10, 4) + eon_search_calls(3, 1)
    assert cost["force_calls"] == 18
    assert cost["repeat_lines"] == 1


def test_campaign_rows_keep_the_measured_catalogues_on_the_adapter():
    from benchmarks.bench_eon_aladatom import EonAladatomAkmc, EonStandardAladatomAkmc

    benches = {
        "eon/aladatom-akmc": EonAladatomAkmc,
        "eon/standard-aladatom-akmc": EonStandardAladatomAkmc,
    }
    for stem, template, problem, calls, rate, catalogue, searches in CATALOGUES:
        dat = ROOT / "results" / "campaign" / f"{stem}.dat"
        fields, _duplicates = parse_lenient(dat.read_text(encoding="utf-8"))
        assert fields["counter"] == "harness"
        assert fields["count_keys"] == COUNT_KEY
        assert fields["template"] == template
        assert fields["problem"] == problem
        assert fields["reducer"] == "amd-catalogue"
        assert float(fields["force_calls"]) == calls
        assert float(fields["true_rate"]) == pytest.approx(rate)
        assert float(fields["rate_accuracy"]) == 0.0
        assert float(fields["catalogue_accuracy"]) == pytest.approx(catalogue)
        assert _comparable(fields, []) is True
        record = json.loads((dat.with_suffix(".starts.jsonl")).read_text(encoding="utf-8"))
        assert record["counter"] == "harness"
        assert record["comparable"] is True
        assert record["count_key"] == COUNT_KEY
        assert record["force_calls"] == calls
        bench = benches[template]
        assert bench.template_name == template
        assert bench.problem_name == problem
        assert bench.potential == "eam_al"
        assert bench.cap == 1000000000
        assert bench.param_names == ["seed", "start"]
        assert bench.params == [["20261018"], ["rep-1"]]
        assert bench.track_force_calls.unit == "force calls"
        assert bench.track_success.unit == "fraction"
        assert bench.track_failure_cap.unit == "count"
        assert bench.track_failure_wrong_stationary.unit == "count"
        assert bench.track_failure_divergence.unit == "count"
        assert bench.track_failure_crash.unit == "count"
        assert bench.track_total_searches.unit == "searches"
        assert bench.track_rate.unit == "per second"
        measured = bench().measure("20261018", "rep-1")
        assert measured["force_calls"] == calls
        assert measured["success"] is True
        assert measured["failure"] is None
        assert measured["total_searches"] == searches
        assert measured["counter"] == "harness"
        assert bench().track_force_calls("20261018", "rep-1") == calls
        assert bench().track_success("20261018", "rep-1") == 1.0
        assert bench().track_failure_divergence("20261018", "rep-1") == 0.0
        assert bench().track_rate("20261018", "rep-1") == pytest.approx(rate)


def test_benchmark_module_imports_without_eonclient():
    completed = subprocess.run(
        [sys.executable, "-c", "import benchmarks.bench_eon_aladatom, sys; assert 'eon' not in sys.modules"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_writer_marks_the_adapter_ranked(tmp_path: Path):
    template = {
        "id": "eon/aladatom-akmc",
        "code": "eOn",
        "code_version": "3.6.0",
        "commit": "1d71c944501832bfafb955ffe13b9aae50c8efd4",
        "algorithm": "AKMC dimer",
    }
    problem = {
        "id": "accelerated-md/aladatom",
        "reference": {"true_rate": 254294100.0},
        "catalogue": {"probabilities": [1.0]},
        "sample": {"products": 5000, "repetitions": 1},
        "thermal": {"temperature_K": 250},
    }
    output = tmp_path / "benchmark.dat"
    _write_result(
        template,
        problem,
        [
            {
                "repetition": 1,
                "force_calls": 80,
                "total_searches": 1,
                "repeats": 20,
                "rate": 1.0e6,
                "confidence": 0.95,
                "reached": True,
                "seed": 20261018,
                "channels": [(0, 1.0e6)],
                "seconds": 1.0,
            }
        ],
        output,
        1,
    )
    fields, _duplicates = parse_lenient(output.read_text(encoding="utf-8"))
    assert fields["counter"] == "harness"
    assert fields["count_keys"] == COUNT_KEY
    assert "partial" not in fields
    record = json.loads(output.with_suffix(".starts.jsonl").read_text(encoding="utf-8"))
    assert record["counter"] == "harness"
    assert record["comparable"] is True
    assert record["count_key"] == COUNT_KEY
    assert record["force_calls"] == 80
    assert record["success"] is True
    assert record["failure"] is None


def _eonclient() -> Path | None:
    env = os.environ.get("EONCLIENT")
    if env and Path(env).is_file():
        return Path(env)
    found = shutil.which("eonclient")
    if found:
        return Path(found)
    installed = Path.home() / ".local" / "eon-main" / "bin" / "eonclient"
    if installed.is_file():
        return installed
    return None


def _four_atom_con() -> str:
    return "\n".join(
        [
            "Generated by optbench",
            "0.0000 TIME",
            "40.0000 40.0000 40.0000",
            "90.0000 90.0000 90.0000",
            "0 0",
            "0 0 0",
            "1",
            "4",
            "1.00000",
            "H",
            "Coordinates of Component 1",
            "  20.000000  20.000000  20.000000  0  0",
            "  21.120000  20.000000  20.000000  0  1",
            "  20.560000  20.970000  20.000000  0  2",
            "  20.560000  20.320000  20.900000  0  3",
            "",
        ]
    )


@pytest.mark.skipif(_eonclient() is None, reason="eonclient is not installed")
def test_saddle_search_matches_harness_counter(tmp_path, monkeypatch):
    """One ext_pot saddle search has both counters.

    The client reports sad-fcs and mins-fcs. HarnessCounter counts every
    force() that reached the server. With a constant prefactor the two
    totals are the same number.
    """
    from benchmarks.bench_eon_develop import lj_cut
    from optbench.eon_akmc import _one_repetition
    from optbench.eon_harness import bridge_binary
    from optbench.potserver import PotServer, remote_count

    client = _eonclient()
    assert client is not None
    geometry = tmp_path / "pos.con"
    geometry.write_text(_four_atom_con(), encoding="utf-8")
    bridge = bridge_binary()
    server = PotServer(lj_cut).start()
    monkeypatch.setenv("OPTBENCH_POTSERVER", server.endpoint())
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    template = {
        "ini": {
            "Main": {
                "job": "akmc",
                "temperature": 300,
                "finite_difference": 0.001,
                "random_seed": 1,
            },
            "Potential": {"potential": "ext_pot", "ext_pot_path": str(bridge)},
            "Optimizer": {
                "opt_method": "cg",
                "converged_force": 0.05,
                "max_iterations": 20,
                "max_move": 0.2,
            },
            "AKMC": {
                "confidence": 1.1,
                "thermally_accessible_window": 20,
                "confidence_scheme": "old",
            },
            "Communicator": {
                "type": "local",
                "num_jobs": 1,
                "jobs_per_bundle": 1,
                "number_of_CPUs": 1,
            },
            "Dimer": {
                "improved": False,
                "opt_method": "sd",
                "rotations_max": 3,
                "rotations_min": 1,
                "converged_angle": 15,
                "finite_angle": 0.01,
            },
            "Prefactor": {"default_value": 1e12},
            "Process Search": {"minimize_first": True},
            "Saddle Search": {
                "method": "min_mode",
                "min_mode_method": "dimer",
                "displace_radius": 4.0,
                "displace_magnitude": 0.2,
                "displace_random_weight": 1.0,
                "max_iterations": 6,
                "converged_force": 0.1,
                "max_energy": 50,
            },
        }
    }
    work = tmp_path / "akmc"
    try:
        progress = _one_repetition(ROOT, template, client, work, 1, 1, passes=1, geometry=geometry)
        results = work / "states" / "0" / "search_results.txt"
        assert results.is_file(), progress.get("error")
        text = results.read_text(encoding="utf-8")
        cost = search_cost(text)
        calls, verify = remote_count(server.host, server.port)
    finally:
        server.stop()
    assert verify == 0
    assert cost["total_searches"] >= 1
    assert cost["force_calls"] == calls
    assert calls == progress["force_calls"]
    saddles = []
    minimisations = []
    for line in text.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 7:
            continue
        saddle = int(float(fields[4]))
        minimisation = int(float(fields[5]))
        prefactor = int(float(fields[6]))
        saddles.append(saddle)
        minimisations.append(minimisation)
        assert prefactor == 0
        assert eon_search_calls(saddle, minimisation) == saddle + minimisation
    assert sum(eon_search_calls(saddle, minimisation) for saddle, minimisation in zip(saddles, minimisations)) == calls
    assert (saddles, minimisations, calls) == ([31], [49], 80)
