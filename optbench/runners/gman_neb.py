"""Gman 1.0.0.01 climbing-image NEB on the Pt heptamer.

The release tree is ``codes/gman/r235``. Its ``Gman/version`` file is
``1.0.0.01``. QuickMin is ``qm.init()`` with no arguments, so ``maxmove``
is 0.2 and ``dt`` is 0.15. The band is the archive problem: 8 interior
images, spring 5, tangent ``new``, climbing image, ``fmax`` 0.01, and the
problem iteration cap. Endpoints are not counted. ``pot.count`` is the
band total. The archive published ``pot.count / 8``.

The Morse force is the original ``morse.f``, compiled with ``f2py --lower``
so the gfortran ``force_`` symbol matches. ``counter`` is ``harness``.
The starts store the band total, which is the unit the board ranks.
"""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from optbench.campaign import geometry_archive, write_starts


def _source_root(root: Path) -> Path:
    path = (root / "codes" / "gman" / "r235").resolve()
    if not (path / "Gman" / "__init__.py").is_file():
        raise FileNotFoundError(f"{path} is missing the Gman package")
    if not (path / "Gman" / "pot" / "morse" / "code" / "morse.f").is_file():
        raise FileNotFoundError(f"{path} is missing morse.f")
    return path


def _settings(template: dict, problem: dict) -> dict:
    section = template.get("gman") or {}
    optimizer = str(section.get("optimizer") or "")
    required = (
        "optimizer",
        "images",
        "spring",
        "tangent",
        "itrmax",
        "ppd",
        "fmax",
        "fsaddle",
        "method",
        "converge",
        "potential",
    )
    missing = [key for key in required if key not in section]
    if optimizer == "sd" and "step_r" not in section:
        missing.append("step_r")
    if optimizer == "sd" and "maxmove" not in section:
        missing.append("maxmove")
    if missing:
        raise ValueError(f"{template['id']} gman section needs {', '.join(missing)}")
    if optimizer not in {"sd", "qm"}:
        raise ValueError(f"{template['id']} optimizer {optimizer} is not this runner")
    if section["potential"] != "morse":
        raise ValueError(f"{template['id']} potential {section['potential']} is not this runner")
    fmax = float(section["fmax"])
    stop = float((template.get("stop") or {}).get("force_norm", fmax))
    problem_stop = float((problem.get("convergence") or {})["force_norm"])
    if abs(fmax - stop) > 1e-12 or abs(fmax - problem_stop) > 1e-12:
        raise ValueError(f"{template['id']} fmax is not the problem force norm")
    images = int(section["images"])
    stated = int(((problem.get("band") or {}).get("interior_images")))
    if images != stated:
        raise ValueError(f"{template['id']} images {images} are not the problem band")
    pairs = int((problem.get("sample") or {})["pairs"])
    if pairs < 1:
        raise ValueError(f"{problem['id']} has no pairs")
    return {
        "optimizer": optimizer,
        "step_r": float(section["step_r"]) if optimizer == "sd" else None,
        "dt": float(section["dt"]) if optimizer == "qm" and "dt" in section else None,
        "maxmove": float(section["maxmove"]) if "maxmove" in section else None,
        "images": images,
        "spring": float(section["spring"]),
        "tangent": str(section["tangent"]),
        "itrmax": int(section["itrmax"]),
        "ppd": int(section["ppd"]),
        "fmax": fmax,
        "fsaddle": str(section["fsaddle"]),
        "method": str(section["method"]),
        "converge": str(section["converge"]),
        "pairs": pairs,
    }


def _compile_morse(source: Path) -> None:
    dest = (source / "Gman" / "pot" / "morse" / "morse_.so").resolve()
    if dest.is_file():
        return
    fortran = (source / "Gman" / "pot" / "morse" / "code" / "morse.f").resolve()
    import numpy
    import numpy.f2py

    include_numpy = numpy.get_include()
    include_python = sysconfig.get_path("include")
    f2py_src = Path(numpy.f2py.__file__).resolve().parent / "src"
    if not Path(include_python).is_dir():
        raise RuntimeError(f"Python headers are missing at {include_python}")
    build = Path(tempfile.mkdtemp(prefix="gman-morse-"))
    generated = subprocess.run(
        [sys.executable, "-m", "numpy.f2py", "--lower", str(fortran), "-m", "morse_"],
        cwd=build,
        text=True,
        capture_output=True,
        check=False,
    )
    module_c = build / "morse_module.c"
    if generated.returncode != 0 or not module_c.is_file():
        raise RuntimeError(
            "f2py did not write morse_module.c\n"
            f"{generated.stdout}\n{generated.stderr}"
        )
    compiled = subprocess.run(
        [
            "gfortran",
            "-c",
            "-fPIC",
            "-O2",
            "-fno-second-underscore",
            str(fortran),
            "-o",
            str(build / "morse.o"),
        ],
        cwd=build,
        text=True,
        capture_output=True,
        check=False,
    )
    if compiled.returncode != 0:
        raise RuntimeError(f"gfortran failed\n{compiled.stdout}\n{compiled.stderr}")
    libgfortran = subprocess.check_output(
        ["gfortran", "-print-file-name=libgfortran.so"],
        text=True,
    ).strip()
    linked = subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            f"-I{include_numpy}",
            f"-I{include_python}",
            f"-I{f2py_src}",
            str(module_c),
            str(f2py_src / "fortranobject.c"),
            str(build / "morse.o"),
            f"-L{Path(libgfortran).parent}",
            "-lgfortran",
            "-o",
            str(dest),
        ],
        cwd=build,
        text=True,
        capture_output=True,
        check=False,
    )
    if linked.returncode != 0 or not dest.is_file():
        raise RuntimeError(f"link failed\n{linked.stdout}\n{linked.stderr}")


def _prepare_import(source: Path) -> None:
    root = str(source)
    if root not in sys.path:
        sys.path.insert(0, root)


def _optimizer(settings: dict):
    import Gman.opt as Gopt

    if settings["optimizer"] == "sd":
        opt = Gopt.sd()
        opt.init(stepR=settings["step_r"], maxmove=settings["maxmove"])
        return opt
    opt = Gopt.qm()
    # Omitted dt and maxmove are the library defaults (0.15 and 0.2).
    if settings["dt"] is None and settings["maxmove"] is None:
        opt.init()
    else:
        opt.init(dt=settings["dt"], maxmove=settings["maxmove"])
    return opt


def _one_path(minima: Path, name: str, settings: dict) -> dict:
    import Gman.data as data
    import Gman.io as io
    import Gman.neb as Gneb
    from Gman.pot import pot

    first = data.point()
    second = data.point()
    left = minima / name / "a.con"
    right = minima / name / "b.con"
    if not left.is_file() or not right.is_file():
        raise FileNotFoundError(f"{name} is missing a.con or b.con under {minima}")
    io.loadcon(first, str(left))
    io.loadcon(second, str(right))
    if not getattr(first, "NumAtoms", 0) or not getattr(second, "NumAtoms", 0):
        raise RuntimeError(f"{name} did not load")
    potential = pot.set("morse")
    if potential is None:
        raise RuntimeError("Gman failed to load the Morse potential")
    pot.countReset()
    opt = _optimizer(settings)
    band = Gneb.neb()
    band.new(first, second, images=settings["images"])
    band.relax(
        potential,
        opt.step,
        k=settings["spring"],
        tangent=settings["tangent"],
        itrmax=settings["itrmax"],
        ppd=settings["ppd"],
        fmax=settings["fmax"],
        fsaddle=settings["fsaddle"],
        method=settings["method"],
        converge=settings["converge"],
    )
    raw = int(pot.count())
    pot.countReset()
    per_image = raw / float(settings["images"])
    return {
        "geometry": name,
        "force_calls": per_image,
        "images": settings["images"],
        "raw": raw,
        "success": bool(band.converged),
    }


def _benchmark_text(rows: list[dict], template: dict, problem: dict) -> str:
    """Band total in ``force_calls``. The archive unit is the per-image quotient."""
    totals = [int(row["raw"]) for row in rows]
    images = int(rows[0]["images"])
    per_image = [total / images for total in totals]
    kept = [total for total, row in zip(totals, rows) if row["success"]]
    nfailed = len(rows) - len(kept)
    mean_total = sum(totals) / len(totals)
    lines = [
        f"# Gman {template['code_version']} stock QuickMin. force_calls is pot.count, the band total.",
        "# The archive published pot.count/8. Endpoints are not counted.",
        "force_calls %d" % round(mean_total),
        "force_calls_min %d" % min(totals),
        "force_calls_max %d" % max(totals),
        "force_calls_per_image %d" % (sum(per_image) / len(per_image)),
        "force_calls_per_image_min %d" % min(per_image),
        "force_calls_per_image_max %d" % max(per_image),
        "call_unit total",
        "successes %d" % len(kept),
        "nfailed %d" % nfailed,
        "n %d" % len(rows),
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"problem {problem['id']}",
        f"tuning {template.get('tuning') or 'default'}",
        "counter harness",
        f"images {images}",
    ]
    return "\n".join(lines) + "\n"


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    work: Path | None = None,
    limit: int = 0,
) -> dict:
    settings = _settings(template, problem)
    source = _source_root(root)
    _compile_morse(source)
    scratch = Path(work) if work is not None else Path(tempfile.mkdtemp(prefix="gman-neb-"))
    scratch.mkdir(parents=True, exist_ok=True)
    archive = geometry_archive(root, problem)
    with tarfile.open(archive) as packed:
        packed.extractall(scratch, filter="data")
    minima = scratch / "minima"
    if not minima.is_dir():
        raise FileNotFoundError(f"{archive} did not extract minima/")
    pairs = settings["pairs"] if limit <= 0 else min(limit, settings["pairs"])
    names = [f"min{index:02d}" for index in range(1, pairs + 1)]
    log_path = scratch / "neb.log"
    rows: list[dict] = []
    _prepare_import(source)
    with log_path.open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            for name in names:
                rows.append(_one_path(minima, name, settings))
                print(f"{name} {rows[-1]['force_calls']} converged={rows[-1]['success']}", flush=True)
    calls = [float(row["force_calls"]) for row in rows]
    nfailed = sum(1 for row in rows if not row["success"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_benchmark_text(rows, template, problem), encoding="utf-8")
    start_rows = []
    for row in rows:
        record = {
            "algorithm": template["algorithm"],
            "call_unit": "total",
            "code": template["code"],
            "code_version": template["code_version"],
            "commit": template["commit"],
            "counter": "harness",
            "force_calls": row["raw"],
            "geometry": row["geometry"],
            "images": settings["images"],
            "problem": problem["id"],
            "raw": row["raw"],
            "start": row["geometry"],
            "success": row["success"],
            "template": template["id"],
            "tuning": template.get("tuning") or "default",
        }
        if not row["success"]:
            record["failure"] = "cap" if row["force_calls"] >= settings["itrmax"] else "divergence"
        start_rows.append(record)
    write_starts(output.with_suffix(".starts.jsonl"), start_rows)
    summary = {
        "force_calls": int("%d" % (sum(calls) / len(calls))),
        "force_calls_min": int("%d" % min(calls)),
        "force_calls_max": int("%d" % max(calls)),
        "nfailed": nfailed,
        "starts": len(rows),
        "template": template["id"],
        "problem": problem["id"],
        "counts": calls,
    }
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    from optbench.campaign import load_template, resolve_problem

    template = load_template(root, "gman/neb-pt-qm")
    problem = resolve_problem(root, template["problem"])
    output = root / "results" / "campaign" / "gman-1.0.0.01-neb-pt-qm.dat"
    summary = run_template(root, problem, template, output=output)
    print(json.dumps({key: summary[key] for key in ("force_calls", "force_calls_min", "force_calls_max", "nfailed", "starts")}))
    print(" ".join(str(value) for value in summary["counts"]))


if __name__ == "__main__":
    main()
