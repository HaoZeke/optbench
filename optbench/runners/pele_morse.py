"""pele MYLBFGS on the archived Morse bulk setup.

``archive/entries/minimization/morse-bulk/pele-lbfgs/run.sh`` calls
``minimization_morse.py`` with ``-M 100 --maxstep 2.0``. That script builds
``pele.potentials.Morse`` at rho 1.6047, A 0.7102, r0 2.897, box length
16.387907 and cutoff 9.5, then quenches with ``pele.optimize.mylbfgs``.
The RMS tolerance is ``1e-3/sqrt(3*natoms)``, which is a Euclidean force
norm of 1e-3. A force call is one ``getEnergyGradient`` during the quench.
The post-quench check in ``minimization_tools.Minimizer`` is not counted.
A start fails when the standard deviation of the final gradient is above
that RMS tolerance.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from optbench.score import summarize_capped
from optbench.v2 import write_starts

PELE_COMMIT = "b7081b932a060be11342fbe8287ac35da8ada879"
PELE_VERSION = "0.1.2"
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _quiet() -> None:
    for name in _THREAD_VARS:
        os.environ[name] = "1"


def _method(template: dict) -> str:
    return str((template.get("pele") or {}).get("method") or "")


def _section(template: dict) -> dict:
    section = template.get("pele") or {}
    if _method(template) != "mylbfgs":
        raise ValueError(f"{template['id']} needs pele.method mylbfgs")
    required = ("M", "maxstep", "rho", "A", "r0", "box", "rcut", "natoms")
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(f"{template['id']} is missing pele.{missing}")
    return {
        "M": int(section["M"]),
        "maxstep": float(section["maxstep"]),
        "rho": float(section["rho"]),
        "A": float(section["A"]),
        "r0": float(section["r0"]),
        "box": float(section["box"]),
        "rcut": float(section["rcut"]),
        "natoms": int(section["natoms"]),
        "nsteps": int(section.get("nsteps") or 10000),
    }


class _CountedMorse:
    """pele Morse with a call count on ``getEnergyGradient``."""

    def __init__(self, options: dict):
        import numpy as np
        from pele.potentials import Morse

        self.ncalls = 0
        self._np = np
        box = np.ones(3, dtype=float) * float(options["box"])
        self.inner = Morse(
            rho=float(options["rho"]),
            r0=float(options["r0"]),
            A=float(options["A"]),
            boxvec=box,
            rcut=float(options["rcut"]),
        )

    def getEnergyGradient(self, coords):
        self.ncalls += 1
        return self.inner.getEnergyGradient(coords)


def _quench(job: dict) -> dict:
    _quiet()
    import io
    from contextlib import redirect_stdout

    import numpy as np
    from pele.optimize import mylbfgs

    options = job["options"]
    coords = np.asarray(job["coords"], dtype=np.float64).reshape(-1)
    natoms = int(options["natoms"])
    if coords.size != 3 * natoms:
        raise ValueError(f"{job['start']} has {coords.size // 3} atoms, the script uses {natoms}")
    tol = 1.0e-3 / np.sqrt(3.0 * natoms)
    with redirect_stdout(io.StringIO()):
        potential = _CountedMorse(options)
        result = mylbfgs(
            coords,
            potential,
            M=int(options["M"]),
            maxstep=float(options["maxstep"]),
            tol=float(tol),
            nsteps=int(options["nsteps"]),
            iprint=-1,
        )
    gradient = np.asarray(result.grad, dtype=np.float64).reshape(-1)
    calls = int(potential.ncalls)
    # The 2014 minimizer records pot.ncalls before the tolerance re-evaluation.
    success = bool(np.std(gradient) <= tol)
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "success": success,
        "failure": None if success else "force_norm",
        "counter": "harness",
        "comparable": True,
        "converged": success,
        "energy": float(result.energy),
        "force_norm": float(np.linalg.norm(gradient)),
        "code": "pele",
        "code_version": PELE_VERSION,
        "commit": PELE_COMMIT,
        "tuning": job["tuning"],
    }


def _dat(summary: dict, template: dict, note: str, header: list[str] | None = None) -> str:
    rows = {
        "force_calls": summary.get("force_calls"),
        "force_calls_min": summary.get("force_calls_min"),
        "force_calls_max": summary.get("force_calls_max"),
        "nfailed": summary.get("nfailed"),
        "code": template.get("code"),
        "code_version": PELE_VERSION,
        "commit": PELE_COMMIT,
        "algorithm": template.get("algorithm"),
        "template": template["id"],
        "tuning": template.get("tuning", ""),
        "counter": "harness",
        "problem": template.get("problem"),
    }
    lines = header or [
        "# counter harness. One force call is one getEnergyGradient during mylbfgs.",
        "# The 2014 script records that count before its post-quench gradient check.",
        "# nfailed is a final gradient standard deviation above the RMS stop.",
    ]
    if note:
        lines.append("# " + note.replace("\n", " "))
    for key, value in rows.items():
        if value is None or value == "":
            continue
        lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"


def _installed_version() -> str:
    from importlib.metadata import version

    return version("pele")


class _HarnessMorse:
    """Harness Morse bulk. One force call is one ``getEnergyGradient``."""

    def __init__(self, box):
        import numpy as np

        from optbench.potentials import morse_bulk

        self._np = np
        self._morse = morse_bulk
        self.box = np.asarray(box, dtype=np.float64)
        self.ncalls = 0

    def getEnergy(self, coords):
        energy, _gradient = self._morse(coords, self.box)
        return float(energy)

    def getEnergyGradient(self, coords):
        self.ncalls += 1
        energy, gradient = self._morse(coords, self.box)
        return float(energy), self._np.asarray(gradient, dtype=self._np.float64)


def _quench_stock(job: dict) -> dict:
    _quiet()
    import io
    from contextlib import redirect_stdout

    import numpy as np
    from pele.optimize import lbfgs_cpp

    coords = np.asarray(job["coords"], dtype=np.float64).reshape(-1)
    # Archive stop: Euclidean force norm 1e-3. pele stops on the RMS.
    tol = 1.0e-3 / np.sqrt(coords.size)
    with redirect_stdout(io.StringIO()):
        potential = _HarnessMorse(job["box"])
        result = lbfgs_cpp(coords, potential, tol=float(tol))
    gradient = np.asarray(result.grad, dtype=np.float64).reshape(-1)
    calls = int(potential.ncalls)
    force_norm = float(np.linalg.norm(gradient))
    success = bool(force_norm <= 1.0e-3 and calls < int(job["cap"]))
    return {
        "schema": "optbench.start/v2",
        "start": job["start"],
        "index": job["index"],
        "force_calls": calls,
        "success": success,
        "failure": None if success else "force_norm",
        "counter": "harness",
        "comparable": True,
        "converged": success,
        "energy": float(result.energy),
        "force_norm": force_norm,
        "code": "pele",
        "code_version": PELE_VERSION,
        "commit": PELE_COMMIT,
        "tuning": job["tuning"],
    }


def _run_stock(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int,
) -> dict:
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    if problem["id"] == "minimization/morse-bulk" and geometries is None and not limit and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    cap = int((problem.get("convergence") or {}).get("max_force_calls") or 10000)
    jobs = [
        {
            "index": index,
            "start": name,
            "coords": coords.tolist(),
            "box": box.tolist(),
            "cap": cap,
            "tuning": str(template.get("tuning") or ""),
        }
        for index, (name, coords, box) in enumerate(loaded)
    ]
    if workers <= 1:
        records = [_quench_stock(job) for job in jobs]
    else:
        import multiprocessing as mp

        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            records = list(pool.map(_quench_stock, jobs))
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    calls = [int(record["force_calls"]) for record in records]
    summary = summarize_capped(calls, cap)
    summary["force_calls"] = round(sum(calls) / len(calls)) if calls else None
    summary["force_calls_min"] = min(calls) if calls else None
    summary["force_calls_max"] = max(calls) if calls else None
    summary["nfailed"] = sum(1 for record in records if not record["success"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _dat(
            summary,
            template,
            str(template.get("note") or "").strip(),
            [
                "# counter harness. pele 0.1.2 stock LBFGS_CPP on the harness Morse.",
                "# Memory, maxstep, H0, and maxErise are the release defaults.",
                "# The stop is the archive Euclidean force norm 1e-3.",
            ],
        ),
        encoding="utf-8",
    )
    write_starts(output.with_suffix(".starts.jsonl"), records)
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    return summary


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    geometries: Path | None,
    output: Path,
    workers: int,
    work: Path | None,
    limit: int = 0,
) -> dict:
    version = _installed_version()
    if version != PELE_VERSION:
        raise RuntimeError(f"pele {version} does not match {PELE_VERSION}")
    commit = str(template.get("commit") or "")
    if commit != PELE_COMMIT:
        raise RuntimeError(f"{template['id']} commit {commit} is not pele {PELE_COMMIT}")
    stated = template.get("problem")
    if stated and stated != problem["id"]:
        raise ValueError(f"{template['id']} is for {stated}, not {problem['id']}")
    if _method(template) == "lbfgs_cpp":
        return _run_stock(
            root,
            problem,
            template,
            geometries=geometries,
            output=output,
            workers=workers,
            work=work,
            limit=limit,
        )
    options = _section(template)
    from optbench.runners.rgmin_min import _load_geometries

    loaded = _load_geometries(root, problem, geometries, work, output)
    if limit:
        loaded = loaded[: int(limit)]
    if problem["id"] == "minimization/morse-bulk" and geometries is None and not limit and len(loaded) != 100:
        raise ValueError(f"morse-bulk expected 100 starts, found {len(loaded)}")
    jobs = [
        {
            "index": index,
            "start": name,
            "coords": coords.tolist(),
            "options": options,
            "tuning": str(template.get("tuning") or ""),
        }
        for index, (name, coords, _box) in enumerate(loaded)
    ]
    if workers <= 1:
        records = [_quench(job) for job in jobs]
    else:
        import multiprocessing as mp

        # fork: the quench is launched from this package, and spawn cannot
        # re-import a driver that was started as stdin.
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            records = list(pool.map(_quench, jobs))
    records.sort(key=lambda row: row["index"])
    for record in records:
        record.pop("index", None)
    calls = [int(record["force_calls"]) for record in records]
    cap = int((problem.get("convergence") or {}).get("max_force_calls") or options["nsteps"])
    summary = summarize_capped(calls, cap)
    # The archived getdata mean is round(mean of every start), failures included.
    summary["force_calls"] = round(sum(calls) / len(calls)) if calls else None
    summary["force_calls_min"] = min(calls) if calls else None
    summary["force_calls_max"] = max(calls) if calls else None
    summary["nfailed"] = sum(1 for record in records if not record["success"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(summary, template, str(template.get("note") or "").strip()), encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    summary["starts"] = len(records)
    summary["template"] = template["id"]
    summary["problem"] = problem["id"]
    return summary
