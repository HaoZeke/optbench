"""ASE 3.19.1 PreconLBFGS on the archived tsase Morse calculator.

The 17 Jun 2020 morse-bulk row calls ``tsase.calculators.morse`` with
``rc=9.5`` and ``PreconLBFGS(precon=Exp(A=3), maxstep=0.1)``, then
``opt.run(fmax=0.01)``. ``report.sh`` averages ``calc.force_calls``.
ASE 3.19.1 is the release that was current on that date. ASE 3.29 takes
fewer force calls on the same script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ASE_319 = "3.19.1"


def _is_319(python: Path) -> bool:
    if not python.is_file():
        return False
    probe = subprocess.run(
        [str(python), "-c", "import ase; print(ase.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == ASE_319


def ensure_python() -> Path:
    """Return a Python whose ``ase`` package is exactly 3.19.1."""
    override = os.environ.get("OPTBENCH_ASE319")
    if override:
        path = Path(override)
        if not _is_319(path):
            raise RuntimeError(f"OPTBENCH_ASE319={path} is not ASE {ASE_319}")
        return path
    cached = Path.home() / ".cache" / "optbench" / "ase-3.19.1" / "bin" / "python"
    if _is_319(cached):
        return cached
    legacy = Path("/tmp/ase319venv/bin/python")
    if _is_319(legacy):
        return legacy
    root = cached.parent.parent
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "venv", str(root)])
    subprocess.check_call(
        [
            str(cached),
            "-m",
            "pip",
            "install",
            "numpy==1.26.4",
            "scipy==1.11.4",
            "ase==3.19.1",
            "setuptools",
        ]
    )
    site = subprocess.check_output(
        [str(cached), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    Path(site, "sitecustomize.py").write_text("import setuptools\n", encoding="utf-8")
    if not _is_319(cached):
        raise RuntimeError(f"ASE {ASE_319} install at {cached} failed")
    return cached


def run_start(
    geometry: str,
    *,
    rc: float,
    fmax: float,
    maxstep: float,
    amplitude: float,
    max_steps: int,
) -> dict:
    """Minimize one ``.con`` file and return the tsase force-call count."""
    python = ensure_python()
    payload = {
        "geometry": geometry,
        "rc": rc,
        "fmax": fmax,
        "maxstep": maxstep,
        "amplitude": amplitude,
        "max_steps": max_steps,
    }
    completed = subprocess.run(
        [str(python), "-c", _WORKER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    result = _parse_result(completed.stdout)
    if completed.returncode != 0 or result.get("error"):
        detail = result.get("error") or completed.stderr[-2000:]
        raise RuntimeError(detail or f"ASE {ASE_319} worker exited {completed.returncode}")
    return result


def _parse_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    return {}


_WORKER = textwrap.dedent(
    """
    import contextlib
    import io
    import json
    import sys
    import numpy as np
    from ase.io import read
    from ase.optimize.precon import Exp, PreconLBFGS

    request = json.loads(sys.stdin.read())
    D = 0.7102
    ALPHA = 1.6047
    R0 = 2.897

    class Morse:
        def __init__(self, rc):
            self.u0 = D
            self.alpha = ALPHA
            self.r0 = R0
            self.rc = float(rc)
            self.atoms = None
            self.u = None
            self.f = None
            self.force_calls = 0

        def calculate(self):
            pos = np.asarray(self.atoms.positions, dtype=np.float64)
            cell = np.asarray(self.atoms.cell, dtype=np.float64)
            lengths = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
            delta = pos[:, None, :] - pos[None, :, :]
            delta -= lengths * np.round(delta / lengths)
            dist = np.linalg.norm(delta, axis=-1)
            iu, ju = np.triu_indices(len(pos), k=1)
            r = dist[iu, ju]
            mask = r < self.rc
            r = r[mask]
            dvec = delta[iu, ju][mask]
            x = np.exp(self.alpha * (self.r0 - r))
            xc = np.exp(self.alpha * (self.r0 - self.rc))
            shift = xc * xc - 2.0 * xc
            energy = self.u0 * np.sum(x * x - 2.0 * x - shift)
            duu = (x * x - x) / r
            piece = duu[:, None] * dvec
            forces = np.zeros_like(pos)
            np.add.at(forces, iu[mask], piece)
            np.add.at(forces, ju[mask], -piece)
            forces *= 2.0 * self.alpha * self.u0
            self.f = forces
            self.u = float(energy)
            self.force_calls += 1

        def get_potential_energy(self, atoms=None, force_consistent=False):
            if self.calculation_required(atoms, "energy"):
                self.atoms = atoms.copy()
                self.calculate()
            return self.u

        def get_forces(self, atoms):
            if self.calculation_required(atoms, "forces"):
                self.atoms = atoms.copy()
                self.calculate()
            return self.f.copy()

        def calculation_required(self, atoms, quantities):
            if atoms != self.atoms or self.atoms is None:
                return True
            return False

        def set_atoms(self, atoms):
            pass

    try:
        atoms = read(request["geometry"])
        calc = Morse(request["rc"])
        atoms.set_calculator(calc)
        precon = Exp(A=float(request["amplitude"]))
        opt = PreconLBFGS(
            atoms,
            precon=precon,
            maxstep=float(request["maxstep"]),
            logfile=None,
        )
        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet):
            atoms.get_forces()
            opt.run(fmax=float(request["fmax"]), steps=int(request["max_steps"]))
        forces = np.asarray(calc.f, dtype=float)
        fmax = float(np.sqrt((forces * forces).sum(axis=1)).max())
        result = {
            "force_calls": int(calc.force_calls),
            "fmax": fmax,
            "energy": float(calc.u),
            "parameters": {
                "alpha": 70.0,
                "c1": float(opt.c1),
                "c2": float(opt.c2),
                "damping": float(opt.damping),
                "maxstep": float(opt.maxstep),
                "memory": int(opt.memory),
                "precon": "Exp A=%s" % precon.A,
                "use_armijo": bool(opt.use_armijo),
            },
        }
    except Exception as exc:
        result = {"error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write("RESULT " + json.dumps(result) + "\\n")
    """
)
