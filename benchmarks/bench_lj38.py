"""One eonclient subprocess per Lennard-Jones 38 start.

ASV stores the force-call count of every start as its own result.
Repeating a deterministic count does not make a distribution.
The start index is the parameter.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT
    / "inputs"
    / "sha256"
    / "86bf774955b906c486d37282c48e42628e8fe4af3579222f6e07fcf5913157b1"
    / "lj38-clusters.tgz"
)
TEMPLATES = (
    ("Lj38EonCg", "eon-client/cg", 1800),
    ("Lj38EonFire", "eon-client/fire", 1800),
    ("Lj38EonLbfgsM100", "eon-client/lbfgs-m100", 600),
    ("Lj38EonLbfgsM20", "eon-client/lbfgs-m20", 600),
    ("Lj38EonQm", "eon-client/qm", 3600),
    ("Lj38EonSd", "eon-client/sd", 3600),
    ("Lj38EonSdTwopoint", "eon-client/sd-twopoint", 1800),
)


def start_names() -> list[str]:
    with tarfile.open(ARCHIVE) as packed:
        return sorted(name for name in packed.getnames() if name.endswith(".con"))


def client_path() -> str:
    env = os.environ.get("EONCLIENT")
    if env:
        return env
    found = shutil.which("eonclient")
    if not found:
        raise RuntimeError("put eonclient on PATH or set EONCLIENT")
    return found


def _ini(template_name: str) -> str:
    from optbench.campaign import format_ini, load_template

    return format_ini(load_template(ROOT, template_name)["ini"])


def _cap(template_name: str) -> int:
    from optbench.campaign import _check_stop, load_template, resolve_problem

    template = load_template(ROOT, template_name)
    problem_name = template.get("problem", "minimization/lj38")
    return _check_stop(resolve_problem(ROOT, problem_name), template)


STARTS = start_names()


def _bind(template_name: str, limit: int):
    class _Benchmark:
        param_names = ["start"]
        params = STARTS
        timeout = limit
        repeat = 1
        rounds = 1
        number = 1

        def setup(self, start: str) -> None:
            if shutil.which("eonclient") is None and not os.environ.get("EONCLIENT"):
                raise NotImplementedError("eonclient is not on PATH")
            self.work = Path(tempfile.mkdtemp(prefix="optbench-lj38-"))
            self.ini = _ini(template_name)
            self.cap = _cap(template_name)
            self.client = client_path()
            with tarfile.open(ARCHIVE) as packed:
                packed.extract(start, self.work, filter="data")
            self.con = self.work / start

        def teardown(self, start: str) -> None:
            shutil.rmtree(self.work, ignore_errors=True)

        def track_force_calls(self, start: str) -> float:
            from optbench.campaign import run_client

            calls = run_client(self.client, self.ini, str(self.con), str(self.work / "run"))
            if calls is None:
                return float(self.cap)
            return float(calls)

        track_force_calls.unit = "force calls"

    return _Benchmark


for _class_name, _template_name, _timeout in TEMPLATES:
    _cls = _bind(_template_name, _timeout)
    _cls.__name__ = _class_name
    _cls.__qualname__ = _class_name
    globals()[_class_name] = _cls
