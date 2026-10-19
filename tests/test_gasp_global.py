"""GASP log parsing and the force-call cap formula."""

from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.runners.gasp_global import (
    _energy_limit,
    _harness_cap,
    _load_starts,
    _start_name,
    lammps_final_energy,
    lammps_force_evaluations,
    search_force_cap,
)

ROOT = Path(__file__).resolve().parents[1]

_LOG = """\
Minimization stats:
  Stopping criterion = force tolerance
  Energy initial, next-to-last, final =
     -13.0  -138.5  -138.530000000001
  Iterations, force evaluations = 1000 1975
Loop time of 0.001 on 1 procs for 0 steps with 38 atoms
"""


def test_lammps_log_counts_minimize_and_run0():
    assert lammps_force_evaluations(_LOG) == 1976
    assert lammps_final_energy(_LOG) == -138.530000000001


def test_lammps_log_ignores_a_minimize_without_run0():
    text = "Iterations, force evaluations = 38 76\n"
    assert lammps_force_evaluations(text) == 76
    assert lammps_final_energy(text) is None


def test_two_minimizes_add_their_force_evaluations():
    text = (
        "Iterations, force evaluations = 1 2\n"
        "Energy initial, next-to-last, final =\n"
        "   -1.0  -2.0  -3.5\n"
        "Iterations, force evaluations = 10 20\n"
        "Energy initial, next-to-last, final =\n"
        "   -3.5  -4.0  -4.25\n"
        "Loop time of 1 on 1 procs for 0 steps with 38 atoms\n"
    )
    assert lammps_force_evaluations(text) == 23
    assert lammps_final_energy(text) == -4.25


def test_force_cap_sits_one_above_the_budget():
    assert search_force_cap(800, 20000) == 800 * 20001 + 1


def test_gasp_templates_match_the_cap_formula():
    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "gasp":
            continue
        found.append(template["id"])
        problem = resolve_problem(ROOT, template["problem"])
        cap = _check_stop(problem, template)
        section = template["gasp"]
        method = search_force_cap(int(section["num_energy_calcs"]), int(section["min_eval"]))
        stated = (problem.get("convergence") or {}).get("max_force_calls")
        if stated is None:
            assert cap == method
        else:
            assert cap == int(stated)
            assert method > cap
        assert _harness_cap(problem, template) == cap
        assert int(template["failure_cap"]) == cap
        assert template["commit"] == "0c8d993c82e0e1c69a05b3c34bbb2fcbbdbb7f07"
        assert template["code"] == "gasp"
    assert found == ["gasp/lj38-default", "gasp/lj75-default"]


def test_start_name_keeps_con_stems_and_coords_files():
    assert _start_name(Path("12.con")) == "12"
    assert _start_name(Path("coords.12")) == "coords.12"
    assert _start_name(Path("POSCAR")) == "POSCAR"


def test_lj38_energy_stop_follows_the_problem_reference():
    template = next(row for row in list_templates(ROOT) if row["id"] == "gasp/lj38-default")
    problem = resolve_problem(ROOT, "globalopt/lj38")
    assert _energy_limit(problem, template) == float(problem["reference"]["energy"]) + 0.001


def test_fast_merge_matches_the_gasp_scan():
    import sys

    try:
        import numpy as np
    except ImportError:
        import pytest

        pytest.skip("numpy is not installed")
    gasp_root = Path("/tmp/GASP-python")
    if not (gasp_root / "gasp" / "variations.py").is_file():
        import pytest

        pytest.skip("GASP source is not on this machine")
    sys.path.insert(0, str(gasp_root))
    try:
        import pymatgen  # noqa: F401
        from gasp.development import Constraints
        from gasp.general import Cell, CompositionSpace
        from gasp.geometry import Cluster
        from gasp.variations import Mating
    except ImportError:
        import pytest

        pytest.skip("pymatgen is not installed")
    from optbench.runners import gasp_global

    original = Mating.merge_sites
    held = gasp_global._install_merge_sites
    gasp_global._install_merge_sites = lambda: None
    try:
        gasp_global.install_shims()
    except Exception:
        gasp_global._install_merge_sites = held
        raise
    mating = Mating({"fraction": 0.5})
    geometry = Cluster({"shape": "cluster"})
    constraints = Constraints(None, CompositionSpace(["C", "Al"]))

    def cell_for(seed: int, count: int, species: list[str]) -> Cell:
        rng = np.random.default_rng(seed)
        coords = rng.random((count, 3)) * 16.0
        for _ in range(max(2, count // 6)):
            coords[int(rng.integers(0, count))] = coords[int(rng.integers(0, count))] + rng.normal(scale=0.2, size=3)
        coords[-1] = coords[0]
        return Cell([[60, 0, 0], [0, 55, 0], [0, 0, 50]], species, coords, coords_are_cartesian=True)

    def same(left, right) -> bool:
        return (
            left.num_sites == right.num_sites
            and np.array_equal(left.lattice.matrix, right.lattice.matrix)
            and np.array_equal(np.asarray(left.frac_coords), np.asarray(right.frac_coords))
            and [str(site.specie) for site in left] == [str(site.specie) for site in right]
        )

    cases = [cell_for(seed, 36, ["C"] * 36) for seed in range(4)]
    mixed = ["C" if index % 3 else "Al" for index in range(30)]
    cases.append(cell_for(9, 30, mixed))
    try:
        lefts = [original(mating, source.copy(), geometry, constraints) for source in cases]
    finally:
        gasp_global._install_merge_sites = held
    gasp_global._install_merge_sites()
    for source, left in zip(cases, lefts, strict=True):
        right = Mating.merge_sites(mating, source.copy(), geometry, constraints)
        assert same(left, right)


def test_lj75_uses_the_problem_rule(tmp_path: Path):
    template = next(row for row in list_templates(ROOT) if row["id"] == "gasp/lj75-default")
    problem = resolve_problem(ROOT, "globalopt/lj75")
    assert _energy_limit(problem, template) == -397.492331 + 0.001
    assert int(template["failure_cap"]) == 1000000
    assert "energy_minimum" not in template["gasp"]
    loaded = _load_starts(ROOT, problem, template, tmp_path)
    assert [name for name, _coords in loaded] == [f"coords.{index}" for index in range(1, 201)]
    assert {len(coords) for _name, coords in loaded} == {75}
