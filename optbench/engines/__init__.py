"""Runner table for saddle-search and kMC codes.

A template ``engine`` string selects a module in this package. The module
exposes ``ENGINE``, ``available``, and ``run``. ``available`` returns
``None`` when a search can start, and a reason when it cannot. ``run``
writes the campaign files or raises that reason. No row is invented.

Adding a code is three pieces: a module here, a pin in
``codes/registry.toml``, and templates under ``templates/``. The list in
this package is the codes named for this lane. It is not a closed survey.
"""

from __future__ import annotations

from . import (
    compass_search,
    dlfind_search,
    lammps_min,
    optim_search,
    partn,
    pykmc_search,
    pysis_search,
    tsase_408,
    tsase_search,
)

_MODULES = (
    partn,
    tsase_search,
    tsase_408,
    optim_search,
    pykmc_search,
    lammps_min,
    dlfind_search,
    compass_search,
    pysis_search,
)
ENGINES = {module.ENGINE: module for module in _MODULES}


def find(engine: str):
    """The module for ``engine``, or ``None`` when this table does not own it."""
    return ENGINES.get(engine)


def reasons() -> dict[str, str | None]:
    """``available`` for every engine in the table."""
    return {name: module.available() for name, module in ENGINES.items()}
