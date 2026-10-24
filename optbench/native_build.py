"""Publish one compiled file when several processes build it at once.

The compiler writes a private temporary file. An exclusive lock renames
that file onto the shared path only after the compiler exits. A second
process waits, then reuses the published file.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def exclusive_publish(source: Path, dest: Path, build: Callable[[Path], bool]) -> bool:
    """Return whether ``dest`` is ready.

    ``build`` receives a private path in ``dest``'s directory and returns
    true when that file should replace ``dest``. A published file newer
    than ``source`` is left in place.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.with_name("." + dest.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if _fresh(source, dest):
                return True
            descriptor, name = tempfile.mkstemp(
                prefix="." + dest.name + ".",
                suffix=".tmp",
                dir=dest.parent,
            )
            os.close(descriptor)
            temporary = Path(name)
            try:
                if not build(temporary):
                    return False
                temporary.replace(dest)
                return True
            finally:
                temporary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _fresh(source: Path, dest: Path) -> bool:
    if not dest.is_file() or not source.is_file():
        return False
    return dest.stat().st_mtime >= source.stat().st_mtime
