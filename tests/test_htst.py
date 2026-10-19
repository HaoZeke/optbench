"""The hTST reducer against the archived platinum processtable."""

import gzip
from pathlib import Path

import pytest

from optbench.dat import parse_lenient
from optbench.htst import (
    Process,
    Search,
    format_barrier_force_calls,
    format_jobs,
    format_rate_force_calls,
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
    spec = spec_from_problem(load_problem(ROOT / "problems" / "hTST" / "Pt.toml"))
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
