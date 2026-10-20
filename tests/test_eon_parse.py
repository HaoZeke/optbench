from pathlib import Path

import numpy as np

from optbench.eon_akmc import progress_from_state
from optbench.eon_parse import (
    abs_overlap,
    archive_connected,
    endpoint_link,
    first_success,
    flat_floats,
    max_atom_distance,
    saddle_progress,
    same_minimum,
    within_energy_window,
    xyz_to_con,
)


def test_xyz_to_con_centers_a_free_cluster():
    text = "2\nenergy -1\nA 0 0 0\nA 2 0 0\n"
    con = xyz_to_con(text)
    assert "\n2\n" in con
    assert con.strip().endswith("1")
    lines = [line for line in con.splitlines() if line.strip()[:1].isdigit() or line.startswith("  ")]
    coords = [line for line in con.splitlines() if "  0     " in line]
    assert len(coords) == 2
    xs = [float(line.split()[0]) for line in coords]
    assert abs((xs[0] + xs[1]) / 2 - 50) < 1e-6


def test_abs_overlap_does_not_renormalize():
    assert abs_overlap([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs_overlap([0.0, 2.0], [0.0, -0.5]) == 1.0


def test_flat_floats_skips_comments():
    values = flat_floats("#eigenvalue -1\n1 2\n\n3\n")
    assert values == [1.0, 2.0, 3.0]


def test_first_success_returns_the_smallest_k():
    calls = []

    def evaluate(k):
        calls.append(k)
        return k >= 5, {"k": k}

    found, payload = first_success(evaluate, limit=16)
    assert found == 5
    assert payload == {"k": 5}
    assert 5 in calls
    assert 4 in calls or 1 in calls


def test_energy_window_keeps_the_stop_and_the_archive_ceiling():
    assert within_energy_window(1.49)
    assert within_energy_window(0.0)
    assert within_energy_window(1.5) is False
    assert within_energy_window(5.0) is False
    assert within_energy_window(-0.01) is False


def test_saddle_progress_matches_done_py_including_not_connected():
    header = "wuid type barrier max-dist sad-fcs mins-fcs pref-fcs result\n"
    dashes = "-" * 20 + "\n"
    body = [
        "1 min 0.10000 0.1 10 1 0 good-1",
        "2 min 2.00000 0.1 10 1 0 good-2",
        "3 min 0.20000 0.1 4 1 0 Not Connected",
        "4 min 0.30000 0.1 5 2 9 good-4",
    ]
    # Target 2 so the third good saddle below 1.5 stops the sum.
    # good-2 is above the barrier, Not Connected is not good.
    text = header + dashes + "\n".join(body) + "\n"
    progress = saddle_progress(text, target=2, barrier=1.5)
    assert progress["reached"] is True
    assert progress["qualifying"] == 2
    assert progress["unique_saddles"] == 3
    assert progress["total_searches"] == 4
    assert progress["force_calls"] == (10 + 1) + (10 + 1) + (4 + 1) + (5 + 2)
    assert progress["fraction_connected"] == (4 - 1) / 4


def test_same_minimum_uses_minimum_image_and_permutation():
    cell = np.array([10.0, 10.0, 10.0])
    origin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert same_minimum(origin, origin.copy(), cell)
    wrapped = origin.copy()
    wrapped[0, 0] += 10.0
    assert max_atom_distance(origin, wrapped, cell) < 1e-12
    assert same_minimum(origin, wrapped, cell)
    shifted = origin.copy()
    shifted[1, 0] += 0.2
    assert abs(max_atom_distance(origin, shifted, cell) - 0.2) < 1e-12
    assert same_minimum(origin, shifted, cell, permute=False) is False
    swapped = origin.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert same_minimum(origin, swapped, cell, permute=False) is False
    assert same_minimum(origin, swapped, cell, permute=True) is True
    # The vacuum axis is not folded. A full cell shift in z is a different place.
    vacuum = (True, True, False)
    lifted = origin.copy()
    lifted[0, 2] += 10.0
    assert same_minimum(origin, lifted, cell, pbc=vacuum) is False
    link = endpoint_link(origin, swapped, origin, cell, permute=False)
    assert link["connected"] is True
    assert link["reactant_same"] is True
    assert link["product_same"] is False


def test_saddle_progress_stops_before_a_short_file():
    text = "h\n-\n1 min 0.1 0 3 4 0 good-1\n"
    progress = saddle_progress(text, target=2, barrier=1.5)
    assert progress["reached"] is False
    assert progress["force_calls"] == 7
    assert progress["qualifying"] == 1
    assert progress["rejected"] == 0


def test_saddle_progress_skips_a_good_the_archive_check_rejects():
    header = "wuid type barrier max-dist sad-fcs mins-fcs pref-fcs result\n"
    dashes = "-" * 8 + "\n"
    body = "\n".join(
        [
            "1 min 0.2 0 10 1 0 good-1",
            "2 min 0.3 0 4 1 0 good-2",
            "3 min 0.4 0 5 2 0 good-3",
        ]
    )
    text = header + dashes + body + "\n"

    def accept(proc_id, height):
        return proc_id != "1" and height < 1.5

    progress = saddle_progress(text, target=2, barrier=1.5, accept=accept)
    assert progress["reached"] is True
    assert progress["qualifying"] == 2
    assert progress["rejected"] == 1
    assert progress["unique_saddles"] == 3
    assert progress["force_calls"] == (10 + 1) + (4 + 1) + (5 + 2)


def _con(positions: np.ndarray) -> str:
    rows = [
        "comment",
        "0 Time",
        "10.0 10.0 10.0",
        "90.0 90.0 90.0",
        "0 0",
        "0 0",
        "1",
        str(len(positions)),
        "1.0",
        "Pt",
        "Coordinates",
    ]
    for index, (x, y, z) in enumerate(positions):
        rows.append(f"{x:.8f} {y:.8f} {z:.8f} 0 {index}")
    return "\n".join(rows) + "\n"


def test_progress_from_state_stops_on_the_archive_link(tmp_path: Path):
    origin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    moved = origin.copy()
    moved[1, 0] += 0.4
    state = tmp_path / "states" / "0"
    proc = state / "procdata"
    proc.mkdir(parents=True)
    (state / "reactant.con").write_text(_con(origin), encoding="utf-8")
    # good-1 is the reactant at both ends, so the bijection rejects it.
    (proc / "reactant_1.con").write_text(_con(origin), encoding="utf-8")
    (proc / "product_1.con").write_text(_con(origin), encoding="utf-8")
    (proc / "reactant_2.con").write_text(_con(origin), encoding="utf-8")
    (proc / "product_2.con").write_text(_con(moved), encoding="utf-8")
    text = "\n".join(
        [
            "wuid type barrier max-dist sad-fcs mins-fcs pref-fcs result",
            "--------",
            "1 min 0.20000 0 10 1 0 good-1",
            "2 min 0.30000 0 4 2 0 good-2",
            "3 min 0.10000 0 9 9 0 good-3",
        ]
    )
    (state / "search_results.txt").write_text(text + "\n", encoding="utf-8")
    progress = progress_from_state(state, target=1, barrier=1.5)
    assert progress["reached"] is True
    assert progress["qualifying"] == 1
    assert progress["rejected"] == 1
    assert progress["unique_saddles"] == 2
    assert progress["force_calls"] == (10 + 1) + (4 + 2)
    assert progress["total_searches"] == 2


def test_archive_connected_requires_one_endpoint_and_the_stop():
    cell = np.array([10.0, 10.0, 10.0])
    origin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    product = origin.copy()
    product[1, 0] += 0.4
    assert archive_connected(origin, product, origin, cell, 1.2) is True
    assert archive_connected(origin, origin, origin, cell, 1.2) is False
    swapped = origin.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert archive_connected(origin, swapped, origin, cell, 1.2) is False
    assert archive_connected(origin, product, origin, cell, 1.5) is False
    assert archive_connected(origin, product, origin, cell, 5.0) is False
