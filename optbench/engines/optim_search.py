"""OPTIM registry hook.

The public tree is the Wales group snapshot
``http://www-wales.ch.cam.ac.uk/svn.tar.bz2``. The pin is the sha256 of
that file. The snapshot has no git commit. OPTIM's CMake target sits
inside the rest of the wales tree. This module does not score OPTIM's
own energy-call counter. A harness potential is not wired, so ``run``
refuses before writing a number.
"""

from __future__ import annotations

from pathlib import Path

ENGINE = "optim"
CODE = "optim"
SNAPSHOT = "http://www-wales.ch.cam.ac.uk/svn.tar.bz2"
SHA256 = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"


def available() -> str | None:
    return (
        "OPTIM has no harness potential in this tree. "
        f"The pin is the sha256 {SHA256} of {SNAPSHOT} "
        "(122743078 bytes). The archive is not a git commit. "
        "CMake for OPTIM depends on the surrounding wales snapshot, and no external-call "
        "hook is connected to HarnessCounter. OPTIM's own call count is not a score."
    )


def run(root: Path, problem_name: str, template_name: str, *, output: Path, workers: int = 1) -> dict:
    del root, problem_name, template_name, output, workers
    raise RuntimeError(available())
