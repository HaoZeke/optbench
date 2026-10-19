"""Gman r235 climbing-image NEB on the Pt heptamer.

Steepest descent is ``archive/entries/neb/Pt-heptamer/gman-sd/opt.py``:
``Gopt.sd`` with ``stepR=0.02`` and ``maxmove=0.2``, then ``neb.relax``
with spring 5, the improved tangent, ``fmax=0.01``, ``converge="Fmag"``,
``fsaddle="off"``, ``method="ci"``, and ``itrmax=10000``.

L-BFGS with the line minimizer is
``archive/entries/neb/Pt-heptamer/gman-lbfgs-line/opt.py``: ``Gopt.lbfgs``
with ``maxmove=0.2``, ``alpha=0.05``, ``memory=25``, ``min="line"``, and
``itrmax=1000``. The other relax arguments match the steepest-descent row.

Each iteration calls the potential on the 8 movable images. ``pot.count()``
is that band total, including line-search calls. The published integer is
``"%d" % (sum(pot.count()/8) / 13)``, which truncates toward zero.

Gman r235 is the package in the public archive tarball
``website/codes/Gman-r235.tgz``. ``codes/gman/r235`` is that package after
lib2to3, with two edits off the counted path: version-string parsing in
``Gman/__init__.py``, and an extrema sort by key after ``extfind`` in
``Gman/neb/neb.py``. The Morse force is the original ``morse.f``, compiled
with ``f2py --lower`` so the gfortran ``force_`` symbol matches.

The archived script writes ``algorithm lbfgs`` for every Gman optimizer.
This runner records the optimizer the script actually calls. ``pot.count``
is the native counter, the same role as OPTIM's energy+gradient calls, so
each pair is stored with ``counter harness`` and ranked by PAR-2.
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
    required = (
        "optimizer",
        "maxmove",
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
    if missing:
        raise ValueError(f"{template['id']} gman section needs {', '.join(missing)}")
    optimizer = str(section["optimizer"])
    if optimizer not in {"sd", "lbfgs"}:
        raise ValueError(f"{template['id']} optimizer {optimizer} is not this runner")
    if optimizer == "sd" and "step_r" not in section:
        raise ValueError(f"{template['id']} gman section needs step_r")
    if optimizer == "lbfgs":
        for key in ("alpha", "memory", "minimizer"):
            if key not in section:
                raise ValueError(f"{template['id']} gman section needs {key}")
        if str(section["minimizer"]) != "line":
            raise ValueError(f"{template['id']} minimizer {section['minimizer']} is not this runner")
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
    cap = (problem.get("convergence") or {}).get("max_force_calls")
    if cap in (None, ""):
        cap = template.get("failure_cap")
    if cap in (None, ""):
        raise ValueError(f"{template['id']} has no force-call cap")
    settings = {
        "optimizer": optimizer,
        "maxmove": float(section["maxmove"]),
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
        "cap": int(cap),
    }
    if optimizer == "sd":
        settings["step_r"] = float(section["step_r"])
    else:
        settings["alpha"] = float(section["alpha"])
        settings["memory"] = int(section["memory"])
        settings["minimizer"] = str(section["minimizer"])
    return settings


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


def _one_path(minima: Path, name: str, settings: dict) -> dict:
    import Gman.data as data
    import Gman.io as io
    import Gman.neb as Gneb
    import Gman.opt as Gopt
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
    if settings["optimizer"] == "sd":
        opt = Gopt.sd()
        opt.init(stepR=settings["step_r"], maxmove=settings["maxmove"])
    else:
        opt = Gopt.lbfgs()
        opt.init(
            maxmove=settings["maxmove"],
            alpha=settings["alpha"],
            memory=settings["memory"],
            min=settings["minimizer"],
        )
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
    converged = bool(band.converged)
    return {
        "geometry": name,
        "force_calls": per_image,
        "raw": raw,
        "converged": converged,
        "success": converged,
    }


def _write_v2(output: Path, rows: list[dict], template: dict, problem: dict, settings: dict) -> None:
    """PAR-2 of the native band totals, the OPTIM campaign file layout."""
    from optbench.score import score_v2
    from optbench.v2 import collect_provenance

    scored = [
        {
            "force_calls": row["force_calls"],
            "success": row["success"],
            "failure": row.get("failure"),
        }
        for row in rows
    ]
    parameters = {
        "counter": "Gman pot.count",
        "optimizer": settings["optimizer"],
        "images": settings["images"],
        "spring": settings["spring"],
        "tangent": settings["tangent"],
        "fmax": settings["fmax"],
        "method": settings["method"],
        "converge": settings["converge"],
        "potential": "morse",
        "version": template["code_version"],
    }
    if settings["optimizer"] == "lbfgs":
        parameters["minimizer"] = settings["minimizer"]
        parameters["memory"] = settings["memory"]
        parameters["alpha"] = settings["alpha"]
    document = {
        "schema": "optbench.score/v2",
        "problem": problem["id"],
        "template": template["id"],
        "counter": "harness",
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters=parameters,
        ),
        "score": score_v2(scored, settings["cap"]),
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _benchmark_text(rows: list[dict], template: dict, problem: dict) -> str:
    calls = [float(row["force_calls"]) for row in rows]
    nfailed = sum(1 for row in rows if not row["success"])
    lines = [
        "force_calls %d" % (sum(calls) / len(calls)),
        "force_calls_min %d" % min(calls),
        "force_calls_max %d" % max(calls),
        "nfailed %d" % nfailed,
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"problem {problem['id']}",
        "counter harness",
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
                print(f"{name} {rows[-1]['force_calls']}", flush=True)
    calls = [float(row["force_calls"]) for row in rows]
    nfailed = sum(1 for row in rows if not row["success"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_benchmark_text(rows, template, problem), encoding="utf-8")
    start_rows = []
    for row in rows:
        record = {
            "call_unit": "total",
            "code": template["code"],
            "code_version": template["code_version"],
            "commit": template["commit"],
            "converged": row["converged"],
            "counter": "harness",
            "force_calls": row["raw"],
            "geometry": row["geometry"],
            "images": settings["images"],
            "problem": problem["id"],
            "raw": row["raw"],
            "success": row["success"],
            "template": template["id"],
        }
        if not row["success"]:
            record["failure"] = "cap" if row["raw"] >= settings["cap"] else "divergence"
        start_rows.append(record)
    write_starts(output.with_suffix(".starts.jsonl"), start_rows)
    _write_v2(output, start_rows, template, problem, settings)
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

    template_id = sys.argv[1] if len(sys.argv) > 1 else "gman/neb-pt-sd"
    template = load_template(root, template_id)
    problem = resolve_problem(root, template["problem"])
    stem = template_id.split("/", 1)[-1]
    output = root / "results" / "campaign" / f"gman-r235-{stem}.dat"
    summary = run_template(root, problem, template, output=output)
    print(json.dumps({key: summary[key] for key in ("force_calls", "force_calls_min", "force_calls_max", "nfailed", "starts")}))
    print(" ".join(str(value) for value in summary["counts"]))


if __name__ == "__main__":
    main()
