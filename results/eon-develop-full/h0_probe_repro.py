"""Why the matched r2549 ini still loses LJ38 dimer and lj38_2ts.

The harness ini for ``eon/develop-ts-lj38-dimer`` and ``eon/develop-de-2ts``
sets every L-BFGS, dimer, and climbing-image key that SVN r2549 reads.
``resolved-config.json`` is the 7095b23 ``Parameters::to_json`` dump of
those inis, plus the fields ``to_json`` leaves out, beside the r2549
loader applied to the same ini. The shared keys match. The losses are
in the first L-BFGS step, which is not an ini key.

r2549 ``LBFGS::getStep`` at iteration 0 and with auto scale on displaces
the objective by ``finite_difference`` along the force and sets
``H0 = 1/C``. Develop commit ``3af36c235625`` (2026-09-30, parent
``825fb17d9808``) asks ``knownCurvature()`` first. ``LBFGS::step`` calls
``getGradient`` before ``getStep``, and ``MinModeObjectiveFunction::getGradient``
increments its iteration before returning. ``knownCurvature`` then
returns ``-eigenvalue`` instead of null, and the probe is skipped.
``H0`` is ``1/|eigenvalue|``.

Commit ``8c4198ef3444`` (2026-09-28, parent ``fd25c4b54230``) makes
``NEBObjectiveFunction::supportsFiniteDifferenceCurvature`` return false,
so a band never probes either. The first step keeps inverse curvature
0.01. Later steps still use ``sy/yy`` on both binaries
(``lbfgs.h0 = sy_yy``).

The repro is those two predicates, on 7095b23, with nothing else edited.
``MinModeObjectiveFunction::knownCurvature`` returns nullopt, and
``NEBObjectiveFunction::supportsFiniteDifferenceCurvature`` returns true.
Rebuild ``libeonclib.so``. The client binary's RPATH loads it. This
script does not patch eOn and does not rerun the client. The numbers
below were measured on that library and are stored in
``h0-probe-repro.json``.

Start 0132 is the short dimer case. Climb iteration 0 matches. Iteration
1 has the same eigenvalue, torque, angle, and 20 rotations. The stock
step is 0.263 with delta_e 20.82, which is past ``max_energy`` 20
(reason 3, 24 calls). The probe step is 0.00586 with delta_e -0.00375
and the search finishes in 412 calls. r2549 on the same start is 396.
Start 0108 finishes in 244 calls on both the probe restore and r2549.
Fourteen of the fifteen develop-only failures flip the same way.
Start 0048 does not: its first eigenvalue is positive, so
``knownCurvature`` was already null, and the two climb logs are the
same 50 lines. It still dies at iteration 48 with delta_e 30.17.
r2549 succeeds in 488 calls. That one start is not this H0 change.

On lj38_2ts the probe restore makes pair_04 1267 calls against r2549
1272, and pair_10 and pair_35 succeed instead of capping. pair_07 and
pair_26 become caps, which is what r2549 does. pair_41 stays a cap
until the climbing image is also re-picked every iteration, the r2549
rule. Develop commit ``ed3ad27a9b00`` (2026-10-04) keeps the previous
local maximum. With both restores pair_41 is 5977 calls and r2549 is
5932. pair_11, pair_12, pair_16, pair_30, and pair_46 stay at the cap.
``climbing_image_band_slack = 0`` does not move any of these pairs.
"""

from __future__ import annotations

import json
from pathlib import Path


def load() -> dict:
    return json.loads((Path(__file__).resolve().parent / "h0-probe-repro.json").read_text())


def main() -> None:
    report = load()
    print(f"develop {report['eon_commit']}  svn {report['svn']}")
    print(report["dimer_commit"]["sha"], report["dimer_commit"]["summary"])
    print()
    print(f"{'start':6} {'stock':>10} {'probe':>10} {'svn':>8}  class")
    for row in report["dimer_develop_only_failures"]:
        stock = row["stock_rerun"]
        probe = row["probe_rerun"]
        print(
            f"{row['start']:6} {stock['calls']:10.0f} {probe['calls']:10.0f} "
            f"{row['svn_calls']:8.0f}  {row['develop_failure']} reason {row['develop_reason']}"
            f" -> {probe['failure'] or 'ok'}"
        )
    print()
    print("0132 stock climb:")
    print("\n".join(report["climb"]["0132-stock"]))
    print("0132 probe climb:")
    print("\n".join(report["climb"]["0132-probe"][:3]))
    print(f"0048 climb logs identical: {report['climb_0048_identical']}")
    print()
    print(f"{'pair':8} {'stock':>10} {'probe':>10} {'repick':>10} {'svn':>8}")
    for row in report["neb_focus"]:
        stock = row["stock_rerun"]["calls"]
        probe = row["probe_rerun"]["calls"]
        repick = row["probe_and_repick"]
        repick_s = "" if repick is None else f"{repick['calls']:10.0f}"
        print(f"{row['start']:8} {stock:10.0f} {probe:10.0f} {repick_s:>10} {row['svn_calls']:8.0f}")


if __name__ == "__main__":
    main()
