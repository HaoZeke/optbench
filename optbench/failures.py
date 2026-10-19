"""One primary failure class for a start.

The harness decides the class from the final energy, the geometry, the
true force, and the call count. A runner-reported class string is not
an input.

The order is fixed, so a start lands in one bin:

1. ``divergence`` — the energy is ``+inf``, ``-inf``, or NaN, or the
   geometry blows up. A blow-up is a non-finite coordinate, a
   dissociation, or a maximum atom displacement strictly above a bound
   the caller states. A finite energy and a finite geometry are not a
   blow-up, however large they are, unless that bound is crossed.
2. ``cap`` — the force-call count is at or above the harness cap. A
   process that dies because the harness refused the next call is this
   class. A LAMMPS NEB band that ends on ``max iterations`` or ``max
   force evaluations`` on every replica is this class only when that
   count has reached the harness cap. The method's own N1/N2 budget
   is not the harness cap.
3. ``crash`` — the process failed and the energy and geometry stayed
   finite and under the cap.
4. ``not_converged`` — the process finished, nothing blew up, the cap
   was not hit, and the true force is still above the tolerance. The
   same class covers a method that stopped on its own residual or its
   own step limit. A finished band is this class when the energy and
   the force stayed finite, the call count is under the harness cap,
   and the judged image (the climbing image, or the highest image of
   a plain NEB) is still above the problem stop. A plain NEB has no
   climbing image, so that stop is not a saddle. A LAMMPS NEB band
   whose every replica last stopped on ``max iterations`` or ``max
   force evaluations``, with a finite climbing-image force still
   above the stop, is this class when the call count is still under
   the harness cap. The force comparison is strict: a force equal to
   the tolerance is not above it. A displacement equal to the bound
   is not over it.
5. ``wrong_stationary`` — the force is at the stop and the point is
   rejected. The rejection is a Hessian index other than the one the
   problem asks for, or the energy, the Kabsch RMSD, the link, the
   novelty test, or the barrier. A stopped band whose Hessian index
   is not 1, with the force already at the stop, is this class.

A success has no class.

The 2014 paper (Chill, Stevenson, Ruehle, Shang, Xiao, Farrell, Wales,
and Henkelman, J. Chem. Theory Comput. 2014, 10, 5476–5482) and the
archived OptBench site do not name these tokens. They report
``nfailed``. The site says a run fails when it exceeds the maximum
number of force calls or fails to converge for any other reason. The
NEB section calls the second case a failure to converge.

Mapping onto that wording:

- ``cap`` is "exceed the maximum number of force calls".
- ``not_converged`` is the paper's "failure to converge" when the
  energy and the geometry stayed finite and the harness cap was not
  hit. The paper has no separate class token. This harness name is
  that clause. It is not ``divergence``.
- ``divergence`` is not a paper token. A non-finite energy or a blown
  geometry would be counted inside "fail to converge for any other
  reason". It is a different event from stopping on a residual, so it
  stays its own class.
- ``wrong_stationary`` is not a paper token. The archived transition
  state page treats a run that misses the tolerance, and a run with a
  positive zero eigenvalue, as failures. A point that has met the
  force stop and has the wrong index is this class. A converged saddle
  that fails the link, the novelty test, or the barrier is the same
  class: it is stationary and it is the wrong point.
- ``crash`` is not a paper token. A process that returns no result is
  a failure "for any other reason". It is not a convergence decision,
  so it stays ``crash``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

FAILURES = ("crash", "divergence", "cap", "not_converged", "wrong_stationary")

# Problem-level rejections of a point that already met the force stop.
# These are facts about the point, not class names.
_STATIONARY_REASONS = frozenset({"not_connected", "duplicate", "barrier", "hessian_index", "energy", "rmsd"})
_UNMET_REASONS = frozenset({"mode_overlap", "solver", "force_norm"})
# LAMMPS minimization stats. Both strings are the N1/N2 step budget.
_STEP_LIMITS = frozenset({"max iterations", "max force evaluations"})


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _coordinates(geometry) -> list[float] | None:
    if geometry is None:
        return None
    if isinstance(geometry, (str, bytes)):
        return None
    values: list[float] = []
    try:
        for item in geometry:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                values.extend(float(component) for component in item)
            else:
                values.append(float(item))
    except (TypeError, ValueError):
        return None
    return values


def geometry_nonfinite(geometry) -> bool:
    """True when a stored coordinate is present and not finite."""
    values = _coordinates(geometry)
    if values is None:
        return False
    if not values:
        return True
    return any(not math.isfinite(value) for value in values)


def max_atom_displacement(geometry, reference) -> float | None:
    """Largest Euclidean displacement of one atom. Both arguments are Nx3 or flat."""
    left = _coordinates(geometry)
    right = _coordinates(reference)
    if left is None or right is None or not left or len(left) != len(right) or len(left) % 3:
        return None
    if any(not math.isfinite(value) for value in left + right):
        return None
    largest = 0.0
    for index in range(0, len(left), 3):
        shift = math.sqrt(
            (left[index] - right[index]) ** 2
            + (left[index + 1] - right[index + 1]) ** 2
            + (left[index + 2] - right[index + 2]) ** 2
        )
        if shift > largest:
            largest = shift
    return largest


def _displacement_over(displacement: float | None, bound: float | None) -> bool:
    if displacement is None or bound is None:
        return False
    if not _finite(displacement) or not _finite(bound):
        return True
    return float(displacement) > float(bound)


def ended_on_step_limit(replica_blocks) -> bool:
    """True when every replica's last stage hit an N1/N2 limit.

    ``replica_blocks`` is the LAMMPS shape: one list of
    ``(stopping criterion, neval)`` per replica. A flat sequence of
    last-stage criterion strings is the same test. An empty band is
    not that stop. One replica that stopped on a force tolerance is
    not that stop. The two budget strings may differ across replicas.
    """
    if not replica_blocks:
        return False
    for blocks in replica_blocks:
        if isinstance(blocks, str):
            if blocks not in _STEP_LIMITS:
                return False
            continue
        if not blocks:
            return False
        last = blocks[-1]
        criterion = last if isinstance(last, str) else last[0]
        if criterion not in _STEP_LIMITS:
            return False
    return True


def _on_step_limit(step_limit: bool, replica_stops) -> bool:
    if step_limit:
        return True
    if replica_stops is None:
        return False
    return ended_on_step_limit(replica_stops)


def classify_outcome(
    *,
    energy: float | None = None,
    geometry=None,
    force: float | None = None,
    calls: float | None = None,
    cap: int | None = None,
    force_tol: float | None = None,
    displacement: float | None = None,
    displacement_bound: float | None = None,
    reference_geometry=None,
    process_failed: bool = False,
    dissociated: bool = False,
    hessian_index: int | None = None,
    expected_index: int | None = None,
    energy_rejected: bool = False,
    rmsd_rejected: bool = False,
    stationary_rejected: bool = False,
    force_above: bool | None = None,
    unmet: bool = False,
    step_limit: bool = False,
    replica_stops=None,
) -> str | None:
    """Primary class from the final energy, geometry, force, and call count.

    ``None`` is a success. ``force_above`` is the caller's measurement
    that the true force is still above the tolerance, used when the
    force value itself was not stored. ``unmet`` means the method
    stopped on its own criterion without a stored force. ``step_limit``
    and ``replica_stops`` are the LAMMPS NEB fact that every replica's
    last stage was ``max iterations`` or ``max force evaluations``.
    With a finite force still above the stop, that fact is budget
    exhaustion: ``cap`` when the call count met the harness cap, and
    ``not_converged`` when the run ended before it. None of these is a
    class name.
    """
    if energy is not None and not _finite(energy):
        return "divergence"
    if geometry_nonfinite(geometry):
        return "divergence"
    if force is not None and not _finite(force):
        return "divergence"
    if dissociated:
        return "divergence"
    shift = displacement
    if shift is None and geometry is not None and reference_geometry is not None:
        shift = max_atom_displacement(geometry, reference_geometry)
    if _displacement_over(shift, displacement_bound):
        return "divergence"
    if cap is not None and calls is not None and _finite(calls) and float(calls) >= float(cap):
        return "cap"
    if process_failed:
        return "crash"
    above = force_above
    if above is None and force is not None and force_tol is not None and _finite(force) and _finite(force_tol):
        above = float(force) > float(force_tol)
    # Every replica spent its N1/N2 budget and the climbing image is
    # still above the stop. The harness cap was not reached, so this
    # is the method's own step limit.
    if _on_step_limit(step_limit, replica_stops) and above and (force is None or _finite(force)):
        return "not_converged"
    if above:
        return "not_converged"
    rejected = bool(energy_rejected or rmsd_rejected or stationary_rejected)
    if (
        not rejected
        and hessian_index is not None
        and expected_index is not None
        and int(hessian_index) != int(expected_index)
    ):
        rejected = True
    if rejected:
        return "wrong_stationary"
    if unmet:
        return "not_converged"
    return None


def _cap_error(text: str) -> bool:
    return "force-call cap" in text or "ForceCallCap" in text


def _dissociated_error(text: str) -> bool:
    return "Lost atoms" in text or ("exposed" in text and "atoms" in text) or "Too many neighbor bins" in text


def classify_failure(reasons: list[str]) -> str | None:
    """Class for a reason list that already names facts, not a class string.

    Prefer :func:`classify_outcome` at the runner. This wrapper exists
    so a reason list still goes through that function.
    """
    if not reasons:
        return None
    names = {str(reason) for reason in reasons}
    label = classify_outcome(
        energy=float("nan") if "divergence" in names else 0.0,
        force=float("nan") if "divergence" in names else (1.0 if "force_norm" in names else 0.0),
        calls=1.0 if "cap" in names else 0.0,
        cap=1 if "cap" in names else 10,
        force_tol=0.5,
        process_failed="crash" in names,
        dissociated="divergence" in names,
        hessian_index=1 if "hessian_index" in names else None,
        expected_index=0 if names & _STATIONARY_REASONS else None,
        energy_rejected="energy" in names,
        rmsd_rejected="rmsd" in names,
        stationary_rejected=bool(names & {"not_connected", "duplicate", "barrier"}),
        force_above=True if "force_norm" in names else None,
        unmet=bool(names & {"mode_overlap", "solver"}),
    )
    if label is None:
        return "not_converged"
    return label


def _stored_geometry(record: dict):
    for key in ("geometry", "coords", "x", "point", "positions"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def _stored_step_limit(record: dict) -> bool:
    """The stored LAMMPS fact, when the record kept it."""
    if record.get("step_limit") is True:
        return True
    stops = record.get("replica_stops")
    if stops is None:
        stops = record.get("replica_blocks")
    if stops:
        return ended_on_step_limit(stops)
    stop = record.get("lammps_stop")
    return isinstance(stop, str) and stop in _STEP_LIMITS


def _client_unmet(record: dict) -> bool:
    reason = record.get("client_reason")
    if reason in (None, "", "0", 0):
        return False
    return True


def classify_stored(record: dict, cap: int | None, force_tol: float | None) -> str | None:
    """Class of one stored start. The stored ``failure`` string is ignored.

    A stored success stays a success unless the energy or the geometry
    blew up or the call count met the cap. A stored failure stays a
    failure: when the numbers do not name a tighter class it is
    ``not_converged``.
    """
    energy = record.get("energy") if "energy" in record else None
    force = record.get("force_norm") if "force_norm" in record else None
    if force is not None and not _finite(force):
        force = float("nan")
    geometry = _stored_geometry(record)
    calls = record.get("force_calls")
    reasons = {str(item) for item in (record.get("reasons") or [])}
    error = str(record.get("error") or "")
    dissociated = _dissociated_error(error) or "divergence" in reasons and (
        (energy is not None and not _finite(energy))
        or (force is not None and not _finite(force))
        or geometry_nonfinite(geometry)
        or _dissociated_error(error)
    )
    # An explicit divergence fact with a finite dissociated geometry.
    if "divergence" in reasons and (_dissociated_error(error) or geometry_nonfinite(geometry)):
        dissociated = True
    if "divergence" in reasons and not dissociated:
        # The old label. A finite energy is not a blow-up.
        if energy is not None and not _finite(energy):
            dissociated = False
        elif geometry_nonfinite(geometry):
            dissociated = True
    process_failed = bool(error.strip()) and not _cap_error(error) and not _dissociated_error(error)
    if record.get("failure") == "crash" and not process_failed and energy is None and force is None and geometry is None:
        process_failed = True
    above = None
    if "force_norm" in reasons and (force is None or force_tol is None):
        above = True
    unmet = bool(reasons & {"mode_overlap", "solver"}) or _client_unmet(record)
    expected = record.get("expected_index")
    hessian = record.get("hessian_index")
    label = classify_outcome(
        energy=None if energy is None else float(energy) if _finite(energy) else float("nan"),
        geometry=geometry,
        force=None if force is None else float(force),
        calls=None if calls is None else float(calls),
        cap=cap,
        force_tol=force_tol,
        displacement=record.get("max_displacement"),
        displacement_bound=record.get("displacement_bound"),
        process_failed=process_failed,
        dissociated=dissociated or ("divergence" in reasons and _dissociated_error(error)),
        hessian_index=None if hessian is None else int(hessian),
        expected_index=None if expected is None else int(expected),
        energy_rejected="energy" in reasons,
        rmsd_rejected="rmsd" in reasons,
        stationary_rejected=bool(reasons & {"not_connected", "duplicate", "barrier", "hessian_index"}),
        force_above=above,
        unmet=unmet,
        step_limit=_stored_step_limit(record),
    )
    stored_success = bool(record.get("success"))
    calls_at_cap = cap is not None and calls is not None and _finite(calls) and float(calls) >= float(cap)
    blown = label == "divergence"
    # A lowest-eigenvector start stores the Hessian of the unmoved geometry
    # and an overlap. That index is not the score. The stored success is
    # the overlap test.
    mode_score = record.get("overlap") is not None
    if stored_success and not blown and not calls_at_cap and label in (None, "not_converged", "wrong_stationary"):
        # A stored success already met the problem rule. Do not invent a
        # failure from a missing force. A real blow-up or a spent cap
        # still overrides it, and so does a force that is above the
        # tolerance when that force was stored and the start is not a
        # mode-overlap score.
        if (
            not mode_score
            and force is not None
            and force_tol is not None
            and _finite(force)
            and float(force) > float(force_tol)
        ):
            return "not_converged"
        if (
            not mode_score
            and label == "wrong_stationary"
            and hessian is not None
            and expected is not None
            and int(hessian) != int(expected)
        ):
            return "wrong_stationary"
        return None
    if label is None:
        return None if stored_success else "not_converged"
    return label


def failure_of(record: dict, cap: int | None) -> str | None:
    """Class for scoring. A success has none. A failure is rescored from the numbers."""
    if record.get("success"):
        label = record.get("failure")
        if label not in (None, "", "none"):
            raise ValueError("a success cannot carry a failure class")
        return None
    if not _has_evidence(record):
        label = record.get("failure")
        if label in FAILURES:
            return str(label)
        calls = record.get("force_calls")
        if cap is not None and calls is not None and _finite(calls) and float(calls) >= float(cap):
            return "cap"
        return "not_converged"
    tol = record.get("force_tol")
    revised = classify_stored(record, cap, None if tol is None else float(tol))
    if revised is None:
        return "not_converged"
    return revised


def components_for(record: dict, problem_id: str | None = None) -> int | None:
    """Force-component count from a stored geometry, or from the LJ38 problem."""
    values = _coordinates(_stored_geometry(record))
    if values and len(values) % 3 == 0:
        return len(values)
    ident = str(problem_id or record.get("problem") or "")
    if "lj38" in ident:
        return 38 * 3
    return None


def _has_evidence(record: dict) -> bool:
    """True when the start stores a number or a fact the classifier can read."""
    for key in ("energy", "force_norm", "geometry", "coords", "x", "point", "positions", "max_displacement"):
        value = record.get(key)
        if value is None or isinstance(value, str):
            continue
        return True
    if record.get("reasons"):
        return True
    if str(record.get("error") or "").strip():
        return True
    reason = record.get("client_reason")
    if reason not in (None, "", "0", 0):
        return True
    if record.get("hessian_index") is not None and record.get("expected_index") is not None:
        return True
    return False


def apply_taxonomy(
    record: dict,
    cap: int | None,
    force_tol: float | None,
    components: int | None = None,
) -> dict:
    """Copy ``record`` with ``success`` and ``failure`` set by :func:`classify_stored`.

    A positive ``sigma`` uses the noise-aware force stop when ``components``
    is known. The clean problem tolerance stays in force at sigma 0.
    """
    from optbench.noise import stop_threshold

    out = dict(record)
    calls = out.get("force_calls")
    at_cap = cap is not None and calls is not None and _finite(calls) and float(calls) >= float(cap)
    if not _has_evidence(out):
        if out.get("success") and not at_cap:
            out["success"] = True
            out["failure"] = None
            return out
        if at_cap:
            out["success"] = False
            out["failure"] = "cap"
            return out
        label = out.get("failure")
        if label in FAILURES:
            out["success"] = False
            out["failure"] = label
            return out
        out["success"] = False
        out["failure"] = "not_converged"
        return out
    tol = force_tol
    sigma = out.get("sigma")
    if (
        tol is not None
        and sigma not in (None, "")
        and _finite(sigma)
        and float(sigma) > 0.0
        and components
    ):
        tol = stop_threshold(float(tol), float(sigma), int(components))
    if tol is not None:
        out["force_tol"] = float(tol)
    label = classify_stored(out, cap, None if tol is None else float(tol))
    if label is None:
        out["success"] = True
        out["failure"] = None
    else:
        out["success"] = False
        out["failure"] = label
    return out


def failure_tally(records: list[dict], cap: int | None) -> dict[str, int]:
    """Count each class. The counts plus nothing else equal ``nfailed``."""
    counts = {name: 0 for name in FAILURES}
    for record in records:
        label = failure_of(record, cap)
        if label is None:
            continue
        counts[label] += 1
    return counts
