"""Gman r235 climbing-image NEB on the Pt heptamer.

Steepest descent follows ``archive/entries/neb/Pt-heptamer/gman-sd/opt.py``:
``Gopt.sd`` with ``stepR=0.02`` and ``maxmove=0.2``, then ``neb.relax`` with
spring 5, the improved tangent, ``fmax=0.01``, ``converge="Fmag"``,
``fsaddle="off"``, ``method="ci"``, and ``itrmax=10000``. Conjugate gradient
follows ``archive/entries/neb/Pt-heptamer/gman-cg/opt.py``: ``Gopt.cg`` with
``maxmove=0.2`` and ``dR=0.0001``, and ``itrmax=1000``. Each iteration calls
the potential on the 8 movable images. ``pot.count()/8`` is that iteration
count, including the converging iteration. The published integer is
``"%d" % (sum(forceCalls) / 13)``, which truncates toward zero.

Gman r235 is the package in the public archive tarball
``website/codes/Gman-r235.tgz``. ``codes/gman/r235`` is that package after
lib2to3, with two edits off the counted path: version-string parsing in
``Gman/__init__.py``, and an extrema sort by key after ``extfind`` in
``Gman/neb/neb.py``. The Morse force is the original ``morse.f``, compiled
with ``f2py --lower`` so the gfortran ``force_`` symbol matches.

The archived script writes ``algorithm lbfgs`` for every Gman optimizer.
This runner records the optimizer the script actually calls. ``pot.count``
is Gman's own counter. The steepest-descent campaign keeps ``counter client``.
Conjugate gradient labels that same native counter ``harness`` and writes the
PAR-2 score, as the OPTIM Pt-island row does for energy+gradient calls.
``benchmark.dat`` keeps the truncated per-image mean. PAR-2 ranks ``pot.count``,
the band total.
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
from optbench.score import score_v2
from optbench.v2 import collect_provenance


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
    if optimizer not in {"sd", "cg"}:
        raise ValueError(f"{template['id']} optimizer {optimizer} is not this runner")
    if optimizer == "sd" and "step_r" not in section:
        raise ValueError(f"{template['id']} gman section needs step_r")
    if optimizer == "cg" and "d_r" not in section:
        raise ValueError(f"{template['id']} gman section needs d_r")
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
    }
    if optimizer == "sd":
        settings["step_r"] = float(section["step_r"])
    else:
        settings["d_r"] = float(section["d_r"])
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
        opt = Gopt.cg()
        opt.init(maxmove=settings["maxmove"], dR=settings["d_r"])
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
        "raw": raw,
        "success": bool(band.converged),
    }


def _benchmark_text(
    rows: list[dict],
    template: dict,
    problem: dict,
    *,
    counter: str = "client",
    header: str = "",
) -> str:
    calls = [float(row["force_calls"]) for row in rows]
    nfailed = sum(1 for row in rows if not row["success"])
    lines = [line for line in header.strip().splitlines() if line] if header else []
    lines.extend(
        [
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
        ]
    )
    if counter == "harness" and template.get("tuning"):
        lines.append(f"tuning {template['tuning']}")
    lines.append(f"counter {counter}")
    return "\n".join(lines) + "\n"


def _paper_count(raw: int, images: int) -> int | float:
    quotient = raw / float(images)
    if quotient.is_integer():
        return int(quotient)
    return quotient


def _write_native(output: Path, rows: list[dict], template: dict, problem: dict, settings: dict) -> None:
    """Per-start pot.count totals and the PAR-2 score of those totals.

    ``per_image`` is the 2014 per-path count, ``pot.count()/8``. ``force_calls``
    is ``pot.count`` itself, the band total the other harness rows rank.
    """
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    images = int(settings["images"])
    stamped = []
    scored = []
    for row in rows:
        raw = int(row["raw"])
        success = bool(row["success"]) and raw < cap
        failure = None if success else ("cap" if raw >= cap else "divergence")
        stamped.append(
            {
                "algorithm": template["algorithm"],
                "call_unit": "total",
                "code": template["code"],
                "code_version": template["code_version"],
                "commit": template["commit"],
                "converged": bool(row["success"]),
                "counter": "harness",
                "failure": failure,
                "force_calls": raw,
                "geometry": row["geometry"],
                "images": images,
                "per_image": _paper_count(raw, images),
                "pot_count": raw,
                "problem": problem["id"],
                "raw": raw,
                "start": row["geometry"],
                "success": success,
                "template": template["id"],
                "tuning": str(template.get("tuning") or "archive"),
            }
        )
        scored.append({"force_calls": raw, "success": success, "failure": failure})
    write_starts(output.with_suffix(".starts.jsonl"), stamped)
    per_image = [float(row["force_calls"]) for row in rows]
    nfailed = sum(1 for row in stamped if not row["success"])
    document = {
        "counter": "harness",
        "legacy": {
            "force_calls": int("%d" % (sum(per_image) / len(per_image))),
            "force_calls_max": int("%d" % max(per_image)),
            "force_calls_min": int("%d" % min(per_image)),
            "n": len(stamped),
            "nfailed": nfailed,
            "successes": len(stamped) - nfailed,
        },
        "problem": problem["id"],
        "provenance": collect_provenance(
            code=template["code"],
            code_version=template["code_version"],
            commit=template["commit"],
            parameters={
                "converge": settings["converge"],
                "counter": "pot.count",
                "d_r": settings["d_r"],
                "fmax": settings["fmax"],
                "fsaddle": settings["fsaddle"],
                "images": images,
                "itrmax": settings["itrmax"],
                "maxmove": settings["maxmove"],
                "method": settings["method"],
                "optimizer": settings["optimizer"],
                "potential": "morse",
                "ppd": settings["ppd"],
                "spring": settings["spring"],
                "tangent": settings["tangent"],
            },
        ),
        "schema": "optbench.score/v2",
        "score": score_v2(scored, cap),
        "template": template["id"],
    }
    output.with_suffix(".v2.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    native = settings["optimizer"] == "cg"
    header = ""
    if native:
        header = (
            "# pot.count is Gman's native counter, one call on every movable image.\n"
            "# force_calls is the truncated mean of pot.count()/8, the 2014 integer.\n"
            "# PAR-2 ranks pot.count(), the band total, with that native counter labelled harness."
        )
    output.write_text(
        _benchmark_text(
            rows,
            template,
            problem,
            counter="harness" if native else "client",
            header=header,
        ),
        encoding="utf-8",
    )
    if native:
        _write_native(output, rows, template, problem, settings)
    else:
        start_rows = []
        for row in rows:
            record = {
                "call_unit": "total",
                "code": template["code"],
                "code_version": template["code_version"],
                "commit": template["commit"],
                "counter": "client",
                "force_calls": row["raw"],
                "geometry": row["geometry"],
                "images": settings["images"],
                "problem": problem["id"],
                "raw": row["raw"],
                "success": row["success"],
                "template": template["id"],
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

    template = load_template(root, "gman/neb-pt-sd")
    problem = resolve_problem(root, template["problem"])
    output = root / "results" / "campaign" / "gman-r235-neb-pt-sd.dat"
    summary = run_template(root, problem, template, output=output)
    print(json.dumps({key: summary[key] for key in ("force_calls", "force_calls_min", "force_calls_max", "nfailed", "starts")}))
    print(" ".join(str(value) for value in summary["counts"]))


if __name__ == "__main__":
    main()
