"""Barrier, prefactor, and rate reducer for the 2014 hTST tables.

``archive/entries/hTST/Al/eon-dimer-cg/report.py`` and the platinum
copy score two files from one AKMC state: ``processtable`` and
``search_results.txt``. A barrier window closes when every reference
process id has been seen once, as ``good`` or ``repeat``. A rate window
closes when the running sum of ``prefactor * exp(-barrier / kT)``
reaches ``rate_accuracy`` of ``rate_ref``. The prefactor in that sum is
the search row's harmonic prefactor when the row carries one, and
otherwise the problem constant. Both 2014 scripts use ``1.2e12``. Force
calls are the ``sad-fcs`` column. The processtable prefactor is recorded
on the process and is not substituted into the rate sum.

The means are arithmetic. Jobs are written with truncation toward zero.
Barrier force calls use ``%.3e``. Rate force calls use ``% .3e``. Window
counts are ``barrier_windows`` and ``rate_windows``.

``harmonic_prefactor`` is the eOn ``RATE_HTST`` product. Eigenvalues are
those of the mass-weighted Hessian. A minimum with a negative mode, or
a saddle without exactly one negative mode, returns -1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

HTST_KB = 8.6173324e-5
HTST_PREFACTOR = 1.2e12
HTST_RATE_ACCURACY = 0.999
# eOn curvature-to-frequency factor, seconds. ``1 / 10.18e-15`` is about 9.823e13 Hz.
HTST_CURVATURE_TO_FREQUENCY = 10.18e-15


@dataclass(frozen=True)
class HtstSpec:
    """Constants from one ``[htst]`` problem table."""

    barriers: tuple[float, ...]
    rate_ref: float
    temperature: float
    prefactor: float
    barrier_ceiling: float
    kb: float = HTST_KB
    rate_accuracy: float = HTST_RATE_ACCURACY

    @property
    def threshold(self) -> float:
        return self.rate_ref * self.rate_accuracy


@dataclass(frozen=True)
class Process:
    """One processtable row. ``barrier`` is rounded to three decimals."""

    id: int
    saddle_energy: float
    prefactor: float
    barrier: float
    barrier_raw: float
    rate: float
    repeats: int


@dataclass(frozen=True)
class Search:
    """One search_results data row.

    ``process_id`` is set only for a ``good`` or ``repeat`` result.
    ``prefactor`` is the harmonic prefactor when the caller supplied one.
    """

    force_calls: int
    barrier: float
    status: str
    process_id: int | None
    prefactor: float | None = None
    saddle_energy: float | None = None
    rate: float | None = None


@dataclass
class HtstReduction:
    """Windows and the process rows that named the reference set."""

    processes: list[Process]
    reference_ids: list[int]
    missed_barriers: list[float]
    unexpected_barriers: list[float]
    last_barrier: int
    last_rate: int
    processtable_rate_sum: float
    barrier_jobs: list[int]
    barrier_force_calls: list[int]
    rate_jobs: list[int]
    rate_force_calls: list[int]
    searches: int

    @property
    def barrier_windows(self) -> int:
        return len(self.barrier_jobs)

    @property
    def rate_windows(self) -> int:
        return len(self.rate_jobs)

    def _mean(self, values: list[int]) -> float | None:
        if not values:
            return None
        return float(np.mean(values))

    @property
    def barrier_jobs_mean(self) -> float | None:
        return self._mean(self.barrier_jobs)

    @property
    def barrier_force_calls_mean(self) -> float | None:
        return self._mean(self.barrier_force_calls)

    @property
    def rate_jobs_mean(self) -> float | None:
        return self._mean(self.rate_jobs)

    @property
    def rate_force_calls_mean(self) -> float | None:
        return self._mean(self.rate_force_calls)


def spec_from_problem(problem: dict) -> HtstSpec:
    """Read the ``[htst]`` table. Missing keys raise ``ValueError``."""
    block = problem.get("htst")
    if not isinstance(block, dict):
        raise ValueError(f"{problem.get('id', 'problem')} has no htst table")
    try:
        barriers = tuple(float(value) for value in block["barriers"])
        return HtstSpec(
            barriers=barriers,
            rate_ref=float(block["rate_ref"]),
            temperature=float(block["temperature"]),
            prefactor=float(block["prefactor"]),
            barrier_ceiling=float(block["barrier_ceiling"]),
            kb=float(block.get("kb", HTST_KB)),
            rate_accuracy=float(block.get("rate_accuracy", HTST_RATE_ACCURACY)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{problem.get('id', 'problem')} htst table is incomplete") from exc


def parse_processtable(text: str) -> list[Process]:
    """Parse an eOn processtable. The header row starts with ``proc``."""
    rows: list[Process] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] == "proc":
            continue
        if len(fields) < 9:
            raise ValueError("processtable row has fewer than 9 columns")
        barrier_raw = float(fields[6])
        rows.append(
            Process(
                id=int(fields[0]),
                saddle_energy=float(fields[1]),
                prefactor=float(fields[2]),
                barrier=round(barrier_raw, 3),
                barrier_raw=barrier_raw,
                rate=float(fields[7]),
                repeats=int(fields[8]),
            )
        )
    return rows


def parse_search_results(text: str) -> list[Search]:
    """Parse ``search_results.txt``. The first two lines are the header."""
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("search_results needs two header lines")
    rows: list[Search] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError("search_results row has no sad-fcs column")
        result = fields[-1]
        status = result
        process_id = None
        if "-" in result:
            status, rest = result.split("-", 1)
            if status in ("good", "repeat"):
                process_id = int(rest)
        rows.append(
            Search(
                force_calls=int(fields[4]),
                barrier=float(fields[2]),
                status=status,
                process_id=process_id,
            )
        )
    return rows


def _reference_ids(
    processes: Sequence[Process], spec: HtstSpec
) -> tuple[list[int], list[float], list[float], int, int, float]:
    """Match the reference multiset in processtable order.

    One occurrence of a barrier is consumed per matching process. The
    returned last-process numbers are -1 until the barrier multiset is
    exhausted and until the processtable rate sum reaches the threshold.
    """
    remaining = list(spec.barriers)
    original = list(spec.barriers)
    reference_ids: list[int] = []
    unexpected: list[float] = []
    last_barrier = -1
    last_rate = -1
    rate_sum = 0.0
    for process in processes:
        barrier = process.barrier
        if barrier < spec.barrier_ceiling and barrier not in original:
            unexpected.append(barrier)
        if barrier in remaining:
            reference_ids.append(process.id)
            del remaining[remaining.index(barrier)]
        if not remaining and last_barrier < 0:
            last_barrier = process.id
        rate_sum += process.rate
        if rate_sum >= spec.threshold and last_rate < 0:
            last_rate = process.id
    return reference_ids, remaining, unexpected, last_barrier, last_rate, rate_sum


def reduce_htst(processes: Sequence[Process], searches: Sequence[Search], spec: HtstSpec) -> HtstReduction:
    """Reduce one state the way the 2014 ``report.py`` scripts do."""
    reference_ids, missed, unexpected, last_barrier, last_rate, rate_sum = _reference_ids(processes, spec)
    barrier_ids: list[int] = []
    rate_ids: list[int] = []
    barrier_jobs: list[int] = []
    barrier_calls: list[int] = []
    rate_jobs: list[int] = []
    rate_calls: list[int] = []
    search_bnum = 0
    search_rnum = 0
    fcs_bsum = 0
    fcs_rsum = 0
    rate_sumt = 0.0
    thermal = spec.kb * spec.temperature
    for search in searches:
        search_bnum += 1
        search_rnum += 1
        fcs_bsum += search.force_calls
        fcs_rsum += search.force_calls
        if search.process_id is None or search.status not in ("good", "repeat"):
            continue
        process_id = search.process_id
        # A window is the full reference multiset. Processes still missing
        # from the table must not close a window on the subset found so far.
        if not missed:
            if reference_ids and process_id in reference_ids and process_id not in barrier_ids:
                barrier_ids.append(process_id)
            if reference_ids and len(barrier_ids) == len(reference_ids):
                barrier_jobs.append(search_bnum)
                barrier_calls.append(fcs_bsum)
                barrier_ids = []
                search_bnum = 0
                fcs_bsum = 0
        prefactor = spec.prefactor if search.prefactor is None else search.prefactor
        if rate_sumt < spec.threshold and process_id not in rate_ids:
            rate_ids.append(process_id)
            rate_sumt += prefactor * math.exp(-search.barrier / thermal)
        if rate_sumt >= spec.threshold:
            rate_jobs.append(search_rnum)
            rate_calls.append(fcs_rsum)
            rate_ids = []
            search_rnum = 0
            fcs_rsum = 0
            rate_sumt = 0.0
    return HtstReduction(
        processes=list(processes),
        reference_ids=reference_ids,
        missed_barriers=missed,
        unexpected_barriers=unexpected,
        last_barrier=last_barrier,
        last_rate=last_rate,
        processtable_rate_sum=rate_sum,
        barrier_jobs=barrier_jobs,
        barrier_force_calls=barrier_calls,
        rate_jobs=rate_jobs,
        rate_force_calls=rate_calls,
        searches=len(searches),
    )


def reduce_tables(processtable: str, search_results: str, spec: HtstSpec) -> HtstReduction:
    """Parse both archived texts and reduce them."""
    return reduce_htst(parse_processtable(processtable), parse_search_results(search_results), spec)


def format_jobs(mean: float) -> str:
    """Truncate a positive mean toward zero, as ``%i`` does."""
    return "%i" % mean


def format_barrier_force_calls(mean: float) -> str:
    return "%.3e" % mean


def format_rate_force_calls(mean: float) -> str:
    return "% .3e" % mean


def metric_lines(reduction: HtstReduction) -> list[str]:
    """The four 2014 means plus the two window counts.

    An empty window list omits that pair. The rate force-call value keeps
    the leading space from ``% .3e``.
    """
    lines: list[str] = []
    if reduction.barrier_jobs_mean is not None and reduction.barrier_force_calls_mean is not None:
        lines.append(f"barrier_jobs {format_jobs(reduction.barrier_jobs_mean)}")
        lines.append(
            f"barrier_force_calls {format_barrier_force_calls(reduction.barrier_force_calls_mean)}"
        )
        lines.append(f"barrier_windows {reduction.barrier_windows}")
    if reduction.rate_jobs_mean is not None and reduction.rate_force_calls_mean is not None:
        lines.append(f"rate_jobs {format_jobs(reduction.rate_jobs_mean)}")
        lines.append(f"rate_force_calls {format_rate_force_calls(reduction.rate_force_calls_mean)}")
        lines.append(f"rate_windows {reduction.rate_windows}")
    return lines


def reference_processes(reduction: HtstReduction) -> list[dict]:
    """Saddle energy, harmonic prefactor, barrier, and rate of each reference id."""
    by_id = {process.id: process for process in reduction.processes}
    rows = []
    for process_id in reduction.reference_ids:
        process = by_id[process_id]
        rows.append(
            {
                "id": process.id,
                "saddle_energy": process.saddle_energy,
                "prefactor": process.prefactor,
                "barrier": process.barrier,
                "rate": process.rate,
            }
        )
    return rows


def barrier_start_records(
    reduction: HtstReduction,
    *,
    problem: str,
    template: str,
    code: str,
    code_version: str,
    commit: str,
    tuning: str,
) -> list[dict]:
    """One JSON object per completed barrier window.

    ``force_calls`` is the ``sad-fcs`` sum of that window. The reference
    processes are the same on every row: they are the saddle energies,
    prefactors, and rates the window was required to find.
    """
    processes = reference_processes(reduction)
    rows = []
    for index, (jobs, calls) in enumerate(zip(reduction.barrier_jobs, reduction.barrier_force_calls), start=1):
        rows.append(
            {
                "code": code,
                "code_version": code_version,
                "commit": commit,
                "counter": "htst-reducer",
                "failed": False,
                "failure": None,
                "force_calls": calls,
                "jobs": jobs,
                "problem": problem,
                "processes": processes,
                "reducer": "harness-htst",
                "start": f"barrier-{index}",
                "success": True,
                "template": template,
                "tuning": tuning,
            }
        )
    return rows


def harmonic_prefactor(minimum: Sequence[float], saddle: Sequence[float]) -> float:
    """Reactant-to-saddle hTST prefactor from Hessian eigenvalues.

    The product is every minimum eigenvalue divided by each positive
    saddle eigenvalue. Zero modes stay in the product. A negative
    minimum eigenvalue, or a saddle that does not have exactly one
    negative eigenvalue, returns -1. Unequal lengths return -1.
    """
    minimum_values = np.asarray(minimum, dtype=np.float64).reshape(-1)
    saddle_values = np.asarray(saddle, dtype=np.float64).reshape(-1)
    if minimum_values.shape != saddle_values.shape:
        return -1.0
    if np.any(minimum_values < 0.0):
        return -1.0
    if int(np.sum(saddle_values < 0.0)) != 1:
        return -1.0
    product = 1.0
    for minimum_value, saddle_value in zip(minimum_values, saddle_values):
        product *= float(minimum_value)
        if saddle_value > 0.0:
            product /= float(saddle_value)
    if product < 0.0:
        return -1.0
    return float(math.sqrt(product) / (2.0 * math.pi * HTST_CURVATURE_TO_FREQUENCY))
