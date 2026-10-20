"""Gman r235 climbing-image NEB with LBFGS-Hess.

The release is ``Gman-r235.tgz``. Python 3 executes it after translating
print statements and the post-convergence extrema sort. The optimizer,
the band, and the driver arguments are the release. The potential is
``pes.morse_pt`` because that kernel matches the Fortran Morse in the
tarball. The score is ``pot.count()``, once per image per iteration.
The harness stores that band total. The 2014 file divided it by 8.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import types
from pathlib import Path

from .common import prepared, run_jobs, write_campaign

ENGINE = "gman"
TARBALL = "https://optbench.org/codes/Gman-r235.tgz"
TARBALL_SHA256 = "299795f6dc4410b49d3db3006ffea0bbb8613c2ab44126be2937ae7b34842849"
VERSION = "r235"
_EXTRACTED = Path("/tmp/gman-r235/Gman")
_TARBALL = Path("/tmp/Gman-r235.tgz")


def source_root() -> Path | None:
    env = os.environ.get("GMAN_ROOT")
    if env and (Path(env) / "neb" / "neb.py").is_file():
        return Path(env)
    if (_EXTRACTED / "neb" / "neb.py").is_file():
        return _EXTRACTED
    return None


def available() -> str | None:
    if source_root() is not None:
        return None
    return (
        "Gman r235 is not available. "
        f"The pin is the sha256 {TARBALL_SHA256} of {TARBALL}. "
        f"{_EXTRACTED} is not present."
    )


def _py3(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("print ") or stripped.startswith("print\t"):
            rest = stripped.split(None, 1)[1]
            if rest.startswith(">>"):
                body = rest[2:].strip()
                target, _, args = body.partition(",")
                lines.append(f"{indent}print({args.strip()}, file={target.strip()})")
            else:
                lines.append(f"{indent}print({rest})")
            continue
        if "self.Ext.sort(lambda x,y: cmp(x.R,y.R))" in line:
            lines.append(
                line.replace(
                    "self.Ext.sort(lambda x,y: cmp(x.R,y.R))",
                    "self.Ext.sort(key=lambda x: x.R)",
                )
            )
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _load(qualname: str, path: Path) -> types.ModuleType:
    source = _py3(path.read_text(encoding="utf-8", errors="replace"))
    module = types.ModuleType(qualname)
    module.__file__ = str(path)
    sys.modules[qualname] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_gman(root: Path | None = None):
    """The release modules, with the optimizer classes where ``neb`` looks them up."""
    if "Gman.neb.neb" in sys.modules and "Gman.pot.pot" in sys.modules:
        return (
            sys.modules["Gman.pot.pot"],
            sys.modules["Gman.neb.neb"],
            sys.modules["Gman.opt.lbfgs"],
            sys.modules["Gman.data.data"],
            sys.modules["Gman.io.conf"],
        )
    folder = root or source_root()
    if folder is None:
        raise RuntimeError(available())
    pkg = types.ModuleType("Gman")
    pkg.__path__ = [str(folder)]
    pkg.__file__ = str(folder / "__init__.py")
    sys.modules["Gman"] = pkg
    for name, subdir in (
        ("Gman.func", folder / "func"),
        ("Gman.math", folder / "math"),
        ("Gman.data", folder / "data"),
        ("Gman.opt", folder / "opt"),
        ("Gman.io", folder / "io"),
        ("Gman.neb", folder / "neb"),
        ("Gman.pot", folder / "pot"),
    ):
        sub = types.ModuleType(name)
        sub.__path__ = [str(subdir)]
        sub.__file__ = str(subdir / "__init__.py")
        sys.modules[name] = sub
        setattr(pkg, name.split(".")[-1], sub)
    func = _load("Gman.func.func", folder / "func" / "func.py")
    sys.modules["Gman.func"].__dict__.update(func.__dict__)
    math = _load("Gman.math.math", folder / "math" / "math.py")
    sys.modules["Gman.math"].__dict__.update(math.__dict__)
    data = _load("Gman.data.data", folder / "data" / "data.py")
    sys.modules["Gman.data"].__dict__.update(data.__dict__)
    sys.modules["Gman.data"].data = data
    qm = _load("Gman.opt.qm", folder / "opt" / "qm.py")
    sys.modules["Gman.opt"].qm = qm.qm
    lbfgs = _load("Gman.opt.lbfgs", folder / "opt" / "lbfgs.py")
    sys.modules["Gman.opt"].lbfgs = lbfgs.lbfgs
    neb = _load("Gman.neb.neb", folder / "neb" / "neb.py")
    pot = _load("Gman.pot.pot", folder / "pot" / "pot.py")
    conf = _load("Gman.io.conf", folder / "io" / "conf.py")
    return pot, neb, lbfgs, data, conf


def morse_callback(point) -> None:
    import numpy as np

    from optbench.pes import morse_pt

    box = np.array(
        [float(point.Box[0, 0]), float(point.Box[1, 1]), float(point.Box[2, 2])],
        dtype=np.float64,
    )
    energy, grad = morse_pt(np.asarray(point.R, dtype=np.float64), box)
    point.U = float(energy)
    point.F = -np.asarray(grad, dtype=np.float64)
    point.F *= point.FreeD


def _execute(job: dict) -> dict:
    cap = int(job["cap"])
    name = str(job["name"])
    total = None
    converged = False
    error = ""
    try:
        pot, neb_mod, lbfgs_mod, data, conf = load_gman()
        pot.callBack = morse_callback
        pot.countReset()
        left = data.point()
        right = data.point()
        conf.loadcon(left, job["reactant"])
        conf.loadcon(right, job["product"])
        optimizer = lbfgs_mod.lbfgs()
        optimizer.init(maxmove=0.2, alpha=0.05, memory=25, min="hess")
        band = neb_mod.neb()
        band.new(left, right, images=int(job["images"]))
        band.relax(
            pot.force,
            optimizer.step,
            k=5.0,
            tangent="new",
            itrmax=100000,
            ppd=100000,
            fmax=float(job["fmax"]),
            fsaddle="off",
            method="ci",
            converge="Fmag",
        )
        total = int(pot.count())
        converged = bool(band.converged)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    success = bool(converged) and total is not None and total < cap and not error
    record = {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(total) if total is not None else cap,
        "success": success,
        "failure": None if success else ("crash" if total is None or error else "divergence"),
        "counter": "harness",
        "converged": bool(converged),
        "gman_pot_count": None if total is None else int(total),
    }
    if error:
        record["error"] = error
    return record


def _minima_root(archive: Path, dest: Path) -> Path:
    marker = dest / "min01" / "a.con"
    nested = dest / "minima" / "min01" / "a.con"
    if marker.is_file():
        return dest
    if nested.is_file():
        return dest / "minima"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as packed:
        packed.extractall(dest, filter="data")
    if nested.is_file():
        return dest / "minima"
    if marker.is_file():
        return dest
    raise FileNotFoundError(f"{archive} has no min01/a.con")


def annotate_per_image(path: Path, records: list[dict], images: int) -> None:
    """Paper min and max. ``write_campaign`` already stores the per-image mean."""
    per = [int(row["force_calls"]) / images for row in records if row["success"]]
    if not per:
        return
    text = path.read_text(encoding="utf-8")
    extra = []
    if "force_calls_per_image_min" not in text:
        extra.append(f"force_calls_per_image_min {min(per):.10g}")
    if "force_calls_per_image_max" not in text:
        extra.append(f"force_calls_per_image_max {max(per):.10g}")
    if extra:
        path.write_text(text.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")


def publish(output: Path, records: list[dict], template: dict, problem: dict, cap: int, images: int) -> dict:
    published = dict(template)
    published["code"] = "Gman"
    comment = (
        "# Gman r235 LBFGS-Hess. Archive driver: maxmove 0.2, alpha 0.05, memory 25, min hess, "
        "spring 5, tangent new, method ci, converge Fmag, fmax 0.01.\n"
        "# force_calls is pot.count(), the band total. The 2014 benchmark.dat divided that total by 8."
    )
    result = write_campaign(
        output,
        records,
        published,
        problem,
        {"cap": cap},
        {
            "method": "lbfgs-hess",
            "counter": "gman pot.count()",
            "potential": "morse_pt",
            "fmax": 0.01,
            "images": images,
            "spring": 5.0,
            "memory": 25,
            "alpha": 0.05,
            "maxmove": 0.2,
            "min": "hess",
            "version": VERSION,
        },
        images=images,
        comment=comment,
    )
    annotate_per_image(output, records, images)
    return result


def assert_tarball(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TARBALL_SHA256:
        raise RuntimeError(f"{path} sha256 is {digest}. The pin is {TARBALL_SHA256}.")


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    if source_root() is None:
        tarball = Path(os.environ.get("GMAN_TARBALL", str(_TARBALL)))
        if not tarball.is_file():
            raise RuntimeError(available())
        assert_tarball(tarball)
        _EXTRACTED.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as packed:
            packed.extractall(_EXTRACTED.parent, filter="data")
    problem, template, settings = prepared(Path(root), problem_name, template_name, "gman")
    from optbench.campaign import geometry_archive
    from optbench.engines.common import cached_pairs

    archive = geometry_archive(Path(root), problem)
    minima = _minima_root(archive, output.parent / "work" / "gman-minima")
    pairs = cached_pairs(Path(root), problem, settings)
    images = int(settings["n_interior"])
    cap = int(settings["cap"])
    jobs = []
    for pair in pairs.values() if isinstance(pairs, dict) else pairs:
        name = pair.name
        reactant = minima / name / "a.con"
        product = minima / name / "b.con"
        if not reactant.is_file() or not product.is_file():
            raise FileNotFoundError(f"{minima} has no {name}")
        jobs.append(
            {
                "name": name,
                "reactant": str(reactant),
                "product": str(product),
                "cap": cap,
                "images": images,
                "fmax": float(settings["force_norm"]),
            }
        )
    jobs.sort(key=lambda job: job["name"])
    records = run_jobs(jobs, _execute, workers)
    return publish(output, records, template, problem, cap, images)
