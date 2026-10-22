"""Shared Pt-island start for harness saddle searches.

OPTIM's saddle-pt cell (``cells/optim/saddle-pt`` at 02405b0d) replaced a
kick that moved one of atoms 0-6 by 0.1 Å from a single generator. The
replacement is the displacement below. The parent draws
``search_seed(rep, attempt)`` and each search builds a new generator
from that seed. One epicentre is chosen from the free listed atoms.
Every free atom within the radius, including the epicentre, is shifted
by an independent normal component of standard deviation ``magnitude``.

``draw_search_start`` is the call the runners share. The same reactant,
cell, periodic flags, atom list, radius, magnitude, repetition, and
attempt produce the same coordinates in every caller. A later search
does not continue the previous stream.

eOn's server, with ``random_mode`` left at the ``config.yaml`` default,
draws a second gaussian and uses that as the mode. This generator uses
the one displacement gaussian as the geometry change and as the
min-mode direction, which is the OPTIM cell's protocol.
"""

from __future__ import annotations

import numpy as np

from .surfaces import minimum_image

# The repetition factor inside ``search_seed``. The old shared stream
# used this integer alone as the generator seed.
_SEED_BASE = 20261015
_SEED_STRIDE = 1_000_003


def as_bool(value, default: bool) -> bool:
    """TOML booleans stay booleans. The strings ``true`` and ``false`` do too."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"not a boolean: {value}")
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    raise TypeError(f"not a boolean: {value!r}")


def search_seed(rep: int, attempt: int) -> int:
    """Distinct seed for one search. The parent draws it before launch.

    A worker must not reseed from a shared value. ``attempt`` is the
    1-based index of the search inside the repetition.
    """
    return (_SEED_BASE + int(rep)) * _SEED_STRIDE + int(attempt)


def problem_pbc(problem: dict) -> tuple[bool, bool, bool]:
    """Periodic flags from the problem potential, three booleans."""
    raw = (problem.get("potential") or {}).get("pbc") or (False, False, False)
    flags = tuple(bool(item) for item in raw)
    if len(flags) != 3:
        raise ValueError(f"pbc has length {len(flags)}")
    return flags  # type: ignore[return-value]


def displace_listed(
    positions: np.ndarray,
    fixed: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
    listed: list[int],
    radius: float,
    magnitude: float,
    rng: np.random.Generator,
) -> dict:
    """One listed epicentre and a Gaussian cloud inside the radius.

    ``displace_all_listed`` false picks one epicentre uniformly from the
    listed free atoms. Every free atom at minimum-image distance ``<=
    radius`` of that epicentre, including the epicentre, is shifted by an
    independent normal component of standard deviation ``magnitude`` on
    each Cartesian axis. Frozen atoms stay put. The draw walks atoms in
    index order and skips a frozen atom before drawing. The minimum image
    is ``surfaces.minimum_image`` with the problem's periodic flags.
    """
    fixed = np.asarray(fixed, dtype=bool)
    mobile = [int(index) for index in listed if 0 <= int(index) < len(fixed) and not bool(fixed[int(index)])]
    if not mobile:
        raise RuntimeError("displace atom list has no free atom")
    epicenter = int(mobile[int(rng.integers(0, len(mobile)))])
    start = np.array(positions, dtype=np.float64, copy=True)
    mode = np.zeros_like(start)
    for index in range(len(start)):
        if bool(fixed[index]):
            continue
        if index == epicenter:
            distance = 0.0
        else:
            delta = minimum_image(start[index] - start[epicenter], cell, pbc)
            distance = float(np.linalg.norm(delta))
        if distance <= float(radius):
            mode[index] = rng.normal(0.0, float(magnitude), size=3)
    if float(np.linalg.norm(mode)) == 0.0:
        mode[epicenter, 0] = 1.0
        start[epicenter, 0] = start[epicenter, 0] + float(magnitude)
    else:
        start = start + mode
    free = ~fixed
    kick = mode[free].reshape(-1)
    return {
        "start": start,
        "kick": kick,
        "vector": _full_vector(kick, fixed),
        "epicenter": epicenter,
        "moved": int(np.count_nonzero(np.linalg.norm(mode, axis=1) > 0.0)),
    }


def draw_search_start(
    positions: np.ndarray,
    fixed: np.ndarray,
    cell,
    pbc,
    listed,
    *,
    rep: int,
    attempt: int,
    radius: float = 3.3,
    magnitude: float = 0.1,
) -> dict:
    """Archive displacement for one search, from ``search_seed``.

    The generator is constructed here and discarded. Callers that share
    ``rep`` and ``attempt`` share the coordinates.
    """
    seed = search_seed(rep, attempt)
    flags = problem_pbc({"potential": {"pbc": pbc}})
    drawn = displace_listed(
        positions,
        fixed,
        None if cell is None else np.asarray(cell, dtype=np.float64),
        flags,
        [int(atom) for atom in listed],
        float(radius),
        float(magnitude),
        np.random.default_rng(seed),
    )
    drawn["seed"] = int(seed)
    return drawn


def _full_vector(free_values: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    count = len(fixed)
    full = np.zeros(3 * count, dtype=np.float64)
    full.reshape(count, 3)[~fixed] = np.asarray(free_values, dtype=np.float64).reshape(-1, 3)
    return full
