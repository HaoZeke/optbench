"""OPTIM's printed energy+gradient counter.

The line is ``OPTIM> # of energy+gradient calls=``. The Hessian line
contains ``energy+gradient+Hessian`` and is not this counter.
"""

from __future__ import annotations


def energy_gradient_calls(log: str) -> int | None:
    """Field 6 of the last energy+gradient line, or None when it was not printed."""
    calls = None
    for line in log.splitlines():
        if "energy+gradient calls" not in line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            calls = int(float(parts[5]))
        except ValueError:
            continue
    return calls


def connected_path(log: str) -> bool:
    """True when the log says a connected path was found.

    ``Connected path appears to exist already`` does not match.
    """
    return "Connected path found" in log
