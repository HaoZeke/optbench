"""ASE basin hopping on globalopt/lj75.

The default parameter is one numbered start. ``OPTBENCH_ASV_FULL=1`` selects
the 200 archived starts. The class calls the campaign runner, so the force
calls are harness calculator entries and the success rule is the problem file.
"""

from __future__ import annotations

import os

from optbench.asv_tracks import TrackedSearch


def _starts() -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(1, 201)]
    return ["1"]


class AseLj75BasinHopping(TrackedSearch):
    template_id = "ase-globalopt/lj75-basinhopping"
    problem_id = "globalopt/lj75"
    cap = 1000000
    timeout = 7200
    param_names = ["seed", "start"]
    params = [["314159"], _starts()]

    def measure(self, seed, start):
        from pathlib import Path

        from optbench.campaign import geometry_archive
        from optbench.problems import load_problem
        from optbench.runners.ase_basinhopping import load_starts, search_one

        if str(seed) != "314159":
            raise ValueError("the lj75 basin-hopping seed base is 314159")
        root = Path(__file__).resolve().parents[1]
        problem = load_problem(root / "problems" / "globalopt" / "lj75.toml")
        index = int(start)
        starts = {item[0]: item[1] for item in load_starts(geometry_archive(root, problem))}
        return search_one(starts[index], index=index, root=root)


AseLj75BasinHopping.track_force_calls.unit = "force calls"
