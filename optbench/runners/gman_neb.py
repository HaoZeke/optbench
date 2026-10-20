"""Gman 1.0.0.01 steepest descent on the Pt heptamer.

Source commit r235. The archived ``gman-sd`` script calls ``Gopt.sd``
and ``sd.init(stepR=0.02, maxmove=0.2)``. Its ``algorithm lbfgs`` line
is the label written for every Gman optimizer. The band uses that
script: 8 images, spring 5, tangent ``new``, climbing image on,
``fmax=0.01``. The iteration limit is at least 10000. The harness ranks
``pot.count``, and a band total at or above the problem cap, 10000, is
a failure.
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

_MIN_ITERATIONS = 10000


class _BandCap(Exception):
    """The potential call would pass the band-total cap."""


def _source_root(root: Path) -> Path:
    path = (root / "codes" / "gman" / "r235").resolve()
    if not (path / "Gman" / "__init__.py").is_file():
        raise FileNotFoundError(f"{path} is missing the Gman package")
    if not (path / "Gman" / "pot" / "morse" / "code" / "morse.f").is_file():
        raise FileNotFoundError(f"{path} is missing morse.f")
    return path


def _stock_settings(template: dict, problem: dict, section: dict) -> dict:
    """Archive steepest descent. The score cap is the problem's band total."""
    optimizer = section.get("optimizer")
    if optimizer != "sd":
        raise ValueError(f"{template['id']} optimizer {optimizer} is not this runner")
    if section.get("potential", "morse") != "morse":
        raise ValueError(f"{template['id']} potential {section.get('potential')} is not this runner")
    convergence = problem.get("convergence") or {}
    fmax = float(convergence["force_norm"])
    stated = (template.get("stop") or {}).get("force_norm")
    if stated is not None and abs(float(stated) - fmax) > 1e-12:
        raise ValueError(f"{template['id']} force stop is not the problem force norm")
    images = int((problem.get("band") or {})["interior_images"])
    if "images" in section and int(section["images"]) != images:
        raise ValueError(f"{template['id']} images are not the problem band")
    cap = int(convergence["max_force_calls"])
    pairs = int((problem.get("sample") or {})["pairs"])
    if pairs < 1 or cap < 1 or images < 1:
        raise ValueError(f"{problem['id']} is missing a band, cap, or sample")
    step_r = float(section.get("step_r", 0.02))
    maxmove = float(section.get("maxmove", 0.2))
    spring = float(section.get("spring", 5.0))
    if abs(step_r - 0.02) > 1e-12 or abs(maxmove - 0.2) > 1e-12 or abs(spring - 5.0) > 1e-12:
        raise ValueError(f"{template['id']} step is not the archive steepest-descent protocol")
    return {
        "optimizer": optimizer,
        "images": images,
        "fmax": fmax,
        "itrmax": max(_MIN_ITERATIONS, cap),
        "cap": cap,
        "pairs": pairs,
        "step_r": step_r,
        "maxmove": maxmove,
        "spring": spring,
    }


def _settings(template: dict, problem: dict) -> dict:
    return _stock_settings(template, problem, template.get("gman") or {})


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


def _bounded(potential, cap: int):
    """Stop once ``pot.count`` has reached the band-total cap."""
    from Gman.pot import pot

    def wrapped(point, count=True):
        if int(pot.count()) >= cap:
            raise _BandCap()
        return potential(point, count)

    return wrapped


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
    opt = Gopt.sd()
    opt.init(stepR=settings["step_r"], maxmove=settings["maxmove"])
    band = Gneb.neb()
    band.new(first, second, images=settings["images"])
    try:
        band.relax(
            _bounded(potential, int(settings["cap"])),
            opt.step,
            k=settings["spring"],
            tangent="new",
            itrmax=settings["itrmax"],
            ppd=1,
            fmax=settings["fmax"],
            fsaddle="off",
            method="ci",
            converge="Fmag",
        )
        converged = bool(band.converged)
    except _BandCap:
        converged = False
    raw = int(pot.count())
    pot.countReset()
    per_image = raw / float(settings["images"])
    return {
        "geometry": name,
        "force_calls": per_image,
        "raw": raw,
        "success": converged,
    }


def _publish(output: Path, rows: list[dict], template: dict, problem: dict, settings: dict) -> dict:
    """Rank Gman's pot.count through the native-counter campaign writer."""
    from optbench.engines.common import write_campaign

    cap = int(settings["cap"])
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
                "start": row["geometry"],
                "force_calls": raw,
                "gman_pot_count": raw,
                "success": success,
                "failure": failure,
                "counter": "harness",
                "converged": converged,
            }
        )
    comment = (
        "# Gman 1.0.0.01 steepest descent. Source commit r235.\n"
        "# sd.init(stepR=0.02, maxmove=0.2) is the archive protocol. "
        "Spring 5, tangent new, climbing image, fmax 0.01.\n"
        "# The iteration limit is 10000. The harness ranks pot.count. "
        "A band total at or above 10000 is a failure."
    )
    published = write_campaign(
        output,
        records,
        template,
        problem,
        {"cap": cap},
        {
            "method": "ci",
            "optimizer": "sd",
            "counter": "pot.count",
            "potential": "morse",
            "version": template["code_version"],
            "images": settings["images"],
            "fmax": settings["fmax"],
            "itrmax": settings["itrmax"],
            "maxmove": settings["maxmove"],
            "step_r": settings["step_r"],
            "spring": settings["spring"],
            "tangent": "new",
        },
        images=settings["images"],
        comment=comment,
    )
    calls = [int(row["force_calls"]) for row in records]
    published["force_calls_min"] = min(calls) if calls else None
    published["force_calls_max"] = max(calls) if calls else None
    published["counts"] = calls
    return published


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
    return _publish(output, rows, template, problem, settings)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    from optbench.campaign import load_template, resolve_problem

    template = load_template(root, "gman/neb-pt-sd")
    problem = resolve_problem(root, template["problem"])
    output = root / "results" / "campaign" / "gman-r235-neb-pt-sd.dat"
    summary = run_template(root, problem, template, output=output)
    print(json.dumps({key: summary[key] for key in ("force_calls", "force_calls_min", "force_calls_max", "nfailed", "starts")}))
    print(" ".join(str(value) for value in summary["counts"]))


if __name__ == "__main__":
    main()
