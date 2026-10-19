"""Stock Gman climbing-image NEB on the Pt heptamer.

Gman r235 is the newest public release. Its package version is 1.0.0.01.
``glbfgs.init`` is called with no arguments, so the step uses that release's
defaults. The band is the archive problem: 13 minima pairs, 8 images, spring
5, improved tangent, climbing image, ``fmax`` 0.01, ``converge`` Fmag.

The harness score is ``pot.count``, Gman's own force counter, written by
``write_campaign`` with ``counter`` harness. The 2013 ``benchmark.dat``
integer is the truncated mean of ``pot.count / 8`` and stays in the
unranked diagnostic file.
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
from optbench.engines.common import write_campaign

RANKED_NAME = "gman-r235-glbfgs-line-neb-pt.dat"


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
    if optimizer != "glbfgs":
        raise ValueError(f"{template['id']} optimizer {optimizer or 'missing'} is not stock glbfgs")
    required = (
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
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    return {
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
        "cap": cap,
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
    opt.init()
    stock = {
        "maxmove": float(opt.maxmove),
        "dR": float(opt.dR),
        "memory": int(opt.memory),
        "min": str(opt.min),
        "alpha": float(opt.alpha),
    }
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
    cap = int(settings["cap"])
    converged = bool(band.converged)
    success = converged and raw < cap
    if success:
        failure = None
    elif raw >= cap:
        failure = "cap"
    else:
        failure = "divergence"
    return {
        "start": name,
        "force_calls": raw,
        "success": success,
        "failure": failure,
        "counter": "harness",
        "converged": converged,
        "gman_pot_count": raw,
        "stock": stock,
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
    stock = rows[0]["stock"]
    records = []
    for row in rows:
        records.append(
            {
                "start": row["start"],
                "force_calls": int(row["force_calls"]),
                "success": row["success"],
                "failure": row["failure"],
                "counter": "harness",
                "converged": row["converged"],
                "gman_pot_count": int(row["gman_pot_count"]),
            }
        )
    comment = (
        "# Gman r235, package version 1.0.0.01. Newest public release. "
        "glbfgs.init() with no arguments.\n"
        "# Climbing-image NEB on the 13 Pt-heptamer pairs, fmax 0.01, spring 5, "
        "tangent new, 8 images.\n"
        "# The count is pot.count, Gman's native force counter."
    )
    parameters = {
        "method": "glbfgs-line",
        "init": "stock",
        "counter": "pot.count",
        "maxmove": stock["maxmove"],
        "dR": stock["dR"],
        "memory": stock["memory"],
        "min": stock["min"],
        "alpha": stock["alpha"],
        "fmax": settings["fmax"],
        "images": settings["images"],
        "spring": settings["spring"],
        "tangent": settings["tangent"],
        "method_neb": settings["method"],
        "converge": settings["converge"],
        "itrmax": settings["itrmax"],
    }
    summary = write_campaign(
        output,
        records,
        template,
        problem,
        {"cap": settings["cap"]},
        parameters,
        images=settings["images"],
        comment=comment,
    )
    summary["counts"] = [int(row["force_calls"]) for row in records]
    summary["stock"] = stock
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    from optbench.campaign import load_template, resolve_problem

    template = load_template(root, "gman/neb-pt-glbfgs-line")
    problem = resolve_problem(root, template["problem"])
    output = root / "results" / "campaign" / RANKED_NAME
    summary = run_template(root, problem, template, output=output)
    print(
        json.dumps(
            {
                "force_calls": summary["force_calls"],
                "nfailed": summary["nfailed"],
                "starts": summary["starts"],
                "stock": summary["stock"],
                "counts": summary["counts"],
                "output": summary["output"],
            }
        )
    )


if __name__ == "__main__":
    main()
