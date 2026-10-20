"""Gman r235 global L-BFGS with the Hessian step, library defaults.

The package is the r235 tree used by the Pt-heptamer steepest-descent cell.
``glbfgs.init`` is called with ``min='hess'`` only. maxmove 0.2, dR 0.0001,
memory 25, and alpha 0.05 stay the library defaults. ``neb.relax`` keeps its
library spring, tangent, climbing-image switch, and Fmag convergence. The
force stop is the archive fmax 0.01. The iteration limit is the archive cap
of 10000 interior-image evaluations: each iteration calls the potential once
on each of the 8 images, and extfind does not increment the counter.

The published integer is the archived opt.py rule: ``pot.count()/8``, then
``%d`` of the mean, the minimum, and the maximum. The starts store
``pot.count()`` so the board ranks the band total.
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

from optbench.campaign import geometry_archive


def _source_root(root: Path) -> Path:
    path = (root / "codes" / "gman" / "r235").resolve()
    if not (path / "Gman" / "__init__.py").is_file():
        raise FileNotFoundError(f"{path} is missing the Gman package")
    if not (path / "Gman" / "pot" / "morse" / "code" / "morse.f").is_file():
        raise FileNotFoundError(f"{path} is missing morse.f")
    return path


def _settings(template: dict, problem: dict) -> dict:
    section = template.get("gman") or {}
    if section.get("optimizer") != "glbfgs" or section.get("minimizer") != "hess":
        raise ValueError(f"{template['id']} is not global L-BFGS with the Hessian step")
    if section.get("potential", "morse") != "morse":
        raise ValueError(f"{template['id']} potential is not morse")
    tuned = sorted(set(section) - {"optimizer", "minimizer", "potential"})
    if tuned:
        raise ValueError(f"{template['id']} sets {', '.join(tuned)} away from the library default")
    convergence = problem.get("convergence") or {}
    fmax = float(convergence["force_norm"])
    stated = (template.get("stop") or {}).get("force_norm")
    if stated is not None and abs(float(stated) - fmax) > 1e-12:
        raise ValueError(f"{template['id']} force stop is not the problem force norm")
    images = int((problem.get("band") or {})["interior_images"])
    cap = int(convergence["max_force_calls"])
    pairs = int((problem.get("sample") or {})["pairs"])
    if pairs < 1 or cap < images or images < 1:
        raise ValueError(f"{problem['id']} is missing a band, cap, or sample")
    return {
        "images": images,
        "fmax": fmax,
        "cap": cap,
        "itrmax": cap // images,
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
        raise RuntimeError(f"f2py did not write morse_module.c\n{generated.stdout}\n{generated.stderr}")
    compiled = subprocess.run(
        ["gfortran", "-c", "-fPIC", "-O2", "-fno-second-underscore", str(fortran), "-o", str(build / "morse.o")],
        cwd=build,
        text=True,
        capture_output=True,
        check=False,
    )
    if compiled.returncode != 0:
        raise RuntimeError(f"gfortran failed\n{compiled.stdout}\n{compiled.stderr}")
    libgfortran = subprocess.check_output(["gfortran", "-print-file-name=libgfortran.so"], text=True).strip()
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
    opt = Gopt.glbfgs()
    opt.init(min="hess")
    band = Gneb.neb()
    band.new(first, second, images=settings["images"])
    band.relax(potential, opt.step, itrmax=settings["itrmax"], fmax=settings["fmax"])
    raw = int(pot.count())
    pot.countReset()
    return {
        "geometry": name,
        "raw": raw,
        "per_image": raw / float(settings["images"]),
        "success": bool(band.converged),
        "force_norm": float(band.fmaxcur),
        "iterations": int(getattr(band, "itr", settings["itrmax"] - 1)) + 1,
    }


def _publish(output: Path, rows: list[dict], template: dict, problem: dict, settings: dict) -> dict:
    from optbench.v2 import collect_provenance, write_starts

    cap = int(settings["cap"])
    images = int(settings["images"])
    records = []
    for row in rows:
        raw = int(row["raw"])
        converged = bool(row["success"])
        success = converged and raw < cap
        failure = None
        if not success:
            failure = "cap" if raw >= cap else "divergence"
        records.append(
            {
                "algorithm": template["algorithm"],
                "code": template["code"],
                "code_version": template["code_version"],
                "commit": template["commit"],
                "converged": converged,
                "counter": "harness",
                "failure": failure,
                "force_calls": raw,
                "force_norm": row["force_norm"],
                "gman_pot_count": raw,
                "iterations": row["iterations"],
                "per_image": row["per_image"],
                "problem": problem["id"],
                "start": row["geometry"],
                "success": success,
                "template": template["id"],
                "tuning": template["tuning"],
            }
        )
    per_image = [float(row["per_image"]) for row in records]
    published = int(sum(per_image) / len(per_image))
    comment = (
        "# force_calls is int(mean(pot.count()/images)), the archived opt.py %d rule\n"
        "# Gman r235 glbfgs.init(min='hess'). maxmove 0.2, dR 1e-4, memory 25, and alpha 0.05 are library defaults.\n"
        "# neb.relax keeps library spring 5, tangent new, climbing image off, and Fmag. fmax 0.01 and the 10000-call cap are the archive criterion.\n"
        "# Starts store pot.count(), the band total. No parameter was set to close the gap to the archive."
    )
    lines = [
        comment,
        f"force_calls {published}",
        f"force_calls_min {int(min(per_image))}",
        f"force_calls_max {int(max(per_image))}",
        f"force_calls_unrounded {sum(per_image) / len(per_image)}",
        f"nfailed {sum(1 for row in records if not row['success'])}",
        f"starts {len(records)}",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template['algorithm']}",
        f"template {template['id']}",
        f"problem {problem['id']}",
        f"tuning {template['tuning']}",
        "counter harness",
        f"images {images}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_starts(output.with_suffix(".starts.jsonl"), records)
    output.with_suffix(".v2.json").write_text(
        json.dumps(
            {
                "counter": "harness",
                "legacy": {"force_calls": published, "n": len(records), "nfailed": sum(1 for row in records if not row["success"])},
                "problem": problem["id"],
                "provenance": collect_provenance(
                    code=template["code"],
                    code_version=template["code_version"],
                    commit=template["commit"],
                    parameters={
                        "counter": "gman pot.count()/8",
                        "fmax": settings["fmax"],
                        "images": images,
                        "itrmax": settings["itrmax"],
                        "minimizer": "hess",
                        "optimizer": "glbfgs",
                        "potential": "morse",
                        "version": template["code_version"],
                    },
                ),
                "schema": "optbench.score/v2",
                "template": template["id"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "problem": problem["id"],
        "template": template["id"],
        "force_calls": published,
        "force_calls_min": int(min(per_image)),
        "force_calls_max": int(max(per_image)),
        "nfailed": sum(1 for row in records if not row["success"]),
        "starts": len(records),
        "counts": [int(row["force_calls"]) for row in records],
        "output": str(output),
    }


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
    scratch = Path(work) if work is not None else Path(tempfile.mkdtemp(prefix="gman-glbfgs-"))
    scratch.mkdir(parents=True, exist_ok=True)
    archive = geometry_archive(root, problem)
    with tarfile.open(archive) as packed:
        packed.extractall(scratch, filter="data")
    minima = scratch / "minima"
    if not minima.is_dir():
        raise FileNotFoundError(f"{archive} did not extract minima/")
    pairs = settings["pairs"] if limit <= 0 else min(limit, settings["pairs"])
    names = [f"min{index:02d}" for index in range(1, pairs + 1)]
    _prepare_import(source)
    rows: list[dict] = []
    log_path = scratch / "neb.log"
    with log_path.open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            for name in names:
                rows.append(_one_path(minima, name, settings))
                print(f"{name} {rows[-1]['raw']} {rows[-1]['force_norm']}", flush=True)
    return _publish(output, rows, template, problem, settings)
