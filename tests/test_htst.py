"""The hTST reducer against the archived 2014 tables."""

import gzip
import math
from pathlib import Path

import pytest

from optbench.dat import parse, parse_lenient
from optbench.htst import (
    HTST_CURVATURE_TO_FREQUENCY,
    Process,
    Search,
    format_barrier_force_calls,
    format_jobs,
    format_rate_force_calls,
    harmonic_prefactor,
    metric_lines,
    parse_processtable,
    reduce_htst,
    reduce_tables,
    spec_from_problem,
)
from optbench.problems import load_problem

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "htst"


def _gz(name: str) -> str:
    with gzip.open(DATA / name, "rt", encoding="utf-8") as handle:
        return handle.read()


def _archived(entry: str) -> dict[str, str]:
    text = (ROOT / "archive" / "entries" / "hTST" / entry / "benchmark.dat").read_text(encoding="utf-8")
    fields, duplicates = parse_lenient(text)
    windows = [fields["average"]]
    windows.extend(item["also"] for item in duplicates if item["key"] == "average")
    fields = dict(fields)
    fields["barrier_windows"] = windows[0].split()[-1]
    fields["rate_windows"] = windows[1].split()[-1]
    return fields


def _assert_published(reduction, fields: dict[str, str]) -> None:
    assert format_jobs(reduction.barrier_jobs_mean) == fields["barrier_jobs"].strip()
    assert format_barrier_force_calls(reduction.barrier_force_calls_mean) == fields["barrier_force_calls"].strip()
    assert format_jobs(reduction.rate_jobs_mean) == fields["rate_jobs"].strip()
    assert format_rate_force_calls(reduction.rate_force_calls_mean).strip() == fields["rate_force_calls"].strip()
    assert str(reduction.barrier_windows) == fields["barrier_windows"]
    assert str(reduction.rate_windows) == fields["rate_windows"]
    assert reduction.missed_barriers == []
    assert reduction.unexpected_barriers == []


def test_al_dimer_cg_matches_the_2014_table():
    problem = load_problem(ROOT / "problems" / "hTST" / "Al.toml")
    reduction = reduce_tables(
        _gz("al-eon-dimer-cg-processtable.gz"),
        _gz("al-eon-dimer-cg-search_results.txt.gz"),
        spec_from_problem(problem),
    )
    _assert_published(reduction, _archived("Al/eon-dimer-cg"))
    assert reduction.reference_ids == [0, 1, 2, 3, 4, 5, 6, 7]
    assert reduction.searches == 30997
    assert reduction.last_barrier == 7
    assert reduction.last_rate == 26
    process = reduction.processes[0]
    assert process.id == 0
    assert process.saddle_energy == pytest.approx(-1206.45379)
    assert process.prefactor == pytest.approx(1.2e12)
    assert process.barrier == pytest.approx(0.372)
    assert process.rate == pytest.approx(6.75195e5)
    text = "\n".join(metric_lines(reduction)) + "\n"
    parsed = parse(text)
    assert parsed["barrier_jobs"] == "48"
    assert parsed["barrier_force_calls"] == "2.887e+04"
    assert parsed["barrier_windows"] == "639"
    assert parsed["rate_jobs"] == "1511"
    assert parsed["rate_force_calls"] == "8.979e+05"
    assert parsed["rate_windows"] == "20"


def test_a_partial_reference_set_does_not_close_a_window():
    problem = load_problem(ROOT / "problems" / "hTST" / "Al.toml")
    process = Process(
        id=4,
        saddle_energy=-1206.45,
        prefactor=1.2e12,
        barrier=0.372,
        barrier_raw=0.37202,
        rate=6.75e5,
        repeats=0,
    )
    search = Search(force_calls=10, barrier=0.37202, status="good", process_id=4)
    reduction = reduce_htst([process], [search], spec_from_problem(problem))
    assert reduction.missed_barriers
    assert reduction.barrier_windows == 0
    assert reduction.reference_ids == [4]


def test_al_dimer_cg2_matches_the_2014_table():
    problem = load_problem(ROOT / "problems" / "hTST" / "Al.toml")
    reduction = reduce_tables(
        _gz("al-eon-dimer-cg2-processtable.gz"),
        _gz("al-eon-dimer-cg2-search_results.txt.gz"),
        spec_from_problem(problem),
    )
    _assert_published(reduction, _archived("Al/eon-dimer-cg2"))
    assert reduction.reference_ids == [0, 2, 3, 4, 5, 6, 8, 9]
    assert reduction.searches == 20000


def test_pt_processtable_names_the_fifteen_reference_barriers():
    problem = load_problem(ROOT / "problems" / "hTST" / "Pt.toml")
    processes = parse_processtable(_gz("pt-eon-dimer-cg-processtable.gz"))
    reduction = reduce_htst(processes, [], spec_from_problem(problem))
    assert reduction.reference_ids == [5, 6, 9, 10, 12, 14, 15, 18, 21, 24, 30, 35, 38, 42, 43]
    assert reduction.missed_barriers == []
    process = next(row for row in processes if row.id == 5)
    assert process.saddle_energy == pytest.approx(-1774.80236)
    assert process.prefactor == pytest.approx(1.0e12)
    assert process.barrier == pytest.approx(0.989)
    assert process.rate == pytest.approx(7.60277e4)


def test_pt_stored_search_log_is_shorter_than_the_published_row():
    """The archived search_results.txt has 10000 lines.

    The published benchmark.dat averages 13 barrier windows and 276 rate
    windows. That longer log is not in the archive. This locks the reducer
    on the file that is.
    """
    problem = load_problem(ROOT / "problems" / "hTST" / "Pt.toml")
    reduction = reduce_tables(
        _gz("pt-eon-dimer-cg-processtable.gz"),
        _gz("pt-eon-dimer-cg-search_results.txt.gz"),
        spec_from_problem(problem),
    )
    published = _archived("Pt/eon-dimer-cg")
    assert reduction.barrier_windows == 4
    assert reduction.rate_windows == 48
    assert format_jobs(reduction.barrier_jobs_mean) == "1675"
    assert format_barrier_force_calls(reduction.barrier_force_calls_mean) == "6.360e+05"
    assert format_jobs(reduction.rate_jobs_mean) == "206"
    assert format_rate_force_calls(reduction.rate_force_calls_mean).strip() == "7.853e+04"
    assert format_jobs(reduction.barrier_jobs_mean) != published["barrier_jobs"].strip()
    assert published["barrier_windows"] == "13"
    assert published["rate_windows"] == "276"


def test_rate_window_uses_the_constant_prefactor():
    spec = spec_from_problem(load_problem(ROOT / "problems" / "hTST" / "Al.toml"))
    spec = type(spec)(
        barriers=(0.1,),
        rate_ref=5.0,
        temperature=spec.temperature,
        prefactor=5.0,
        barrier_ceiling=1.0,
        kb=spec.kb,
        rate_accuracy=0.999,
    )
    processes = [
        Process(
            id=0,
            saddle_energy=0.0,
            prefactor=1.0,
            barrier=0.1,
            barrier_raw=0.1,
            rate=0.0,
            repeats=0,
        )
    ]
    searches = [Search(force_calls=1, barrier=0.0, status="good", process_id=0)]
    reduction = reduce_htst(processes, searches, spec)
    assert reduction.rate_windows == 1
    assert reduction.barrier_windows == 1
    assert reduction.barrier_force_calls == [1]


def test_a_search_prefactor_changes_the_rate_window():
    spec = spec_from_problem(load_problem(ROOT / "problems" / "hTST" / "Al.toml"))
    narrow = type(spec)(
        barriers=(1.0,),
        rate_ref=25.0,
        temperature=300.0,
        prefactor=10.0,
        barrier_ceiling=2.0,
        kb=spec.kb,
        rate_accuracy=0.999,
    )
    processes = [
        Process(0, 0.0, 10.0, 1.0, 1.0, 1.0, 0),
    ]
    plain = [
        Search(1, 0.0, "good", 0),
        Search(1, 0.0, "good", 1),
        Search(1, 0.0, "good", 2),
    ]
    with_prefactor = [
        Search(1, 0.0, "good", 0, prefactor=30.0),
        Search(1, 0.0, "good", 1),
        Search(1, 0.0, "good", 2),
    ]
    assert reduce_htst(processes, plain, narrow).rate_jobs == [3]
    assert reduce_htst(processes, with_prefactor, narrow).rate_jobs == [1]


def test_harmonic_prefactor_matches_the_eon_product():
    value = harmonic_prefactor([4.0, 9.0], [-1.0, 4.0])
    expected = 3.0 / (2.0 * math.pi * HTST_CURVATURE_TO_FREQUENCY)
    assert value == pytest.approx(expected)
    assert harmonic_prefactor([-0.1, 1.0], [-1.0, 1.0]) == -1.0
    assert harmonic_prefactor([1.0, 1.0], [1.0, 1.0]) == -1.0
    assert harmonic_prefactor([1.0, 1.0], [-1.0, -0.2]) == -1.0
    assert harmonic_prefactor([1.0], [-1.0, 1.0]) == -1.0


def test_pt_problem_names_the_reference_barriers():
    problem = load_problem(ROOT / "problems" / "hTST" / "Pt.toml")
    spec = spec_from_problem(problem)
    assert spec.temperature == 700
    assert spec.barrier_ceiling == 1.3
    assert len(spec.barriers) == 15
    assert spec.rate_ref == pytest.approx(155055901.244)
