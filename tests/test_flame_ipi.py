"""Socket framing and Lennard-Jones unit conversion for FLAME msock."""

from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.flame_ipi import (
    BOHR_ANG,
    HA_EV,
    MSGLEN,
    SearchStopped,
    hartree_from_lj,
    ha_per_bohr_from_lj_gradient,
    lowest_relaxed_lj,
    serve,
)
from optbench.potentials import lennard_jones
from optbench.runners.flame_go import LJ38_REFERENCE, reference_energy, search_target, success_ceiling


def _pack_doubles(values) -> bytes:
    array = np.asarray(values, dtype="<f8").reshape(-1)
    return array.tobytes()


def _read_exact(conn: socket.socket, nbytes: int) -> bytes:
    chunks = []
    got = 0
    while got < nbytes:
        piece = conn.recv(nbytes - got)
        assert piece
        chunks.append(piece)
        got += len(piece)
    return b"".join(chunks)


def _exchange(master: socket.socket, positions_bohr: np.ndarray) -> tuple[float, np.ndarray]:
    nat = positions_bohr.size // 3
    master.sendall(b"STATUS      ")
    assert _read_exact(master, MSGLEN) == b"READY       "
    master.sendall(b"POSDATA     ")
    master.sendall(_pack_doubles(np.eye(3).reshape(-1)))
    master.sendall(_pack_doubles(np.eye(3).reshape(-1)))
    master.sendall(struct.pack("<i", nat))
    master.sendall(_pack_doubles(positions_bohr.reshape(-1)))
    master.sendall(b"STATUS      ")
    assert _read_exact(master, MSGLEN) == b"HAVEDATA    "
    master.sendall(b"GETFORCE    ")
    assert _read_exact(master, MSGLEN) == b"FORCEREADY  "
    energy = struct.unpack("<d", _read_exact(master, 8))[0]
    got = struct.unpack("<i", _read_exact(master, 4))[0]
    assert got == nat
    forces = np.frombuffer(_read_exact(master, 8 * 3 * nat), dtype="<f8").copy()
    stress = np.frombuffer(_read_exact(master, 8 * 9), dtype="<f8").copy()
    nmsg = struct.unpack("<i", _read_exact(master, 4))[0]
    assert nmsg == 0
    assert np.all(stress == 0.0)
    return energy, forces


def test_pair_minimum_converts_with_flame_constants():
    distance = 2.0 ** (1.0 / 6.0)
    positions = np.array([0.0, 0.0, 0.0, distance, 0.0, 0.0])
    energy, gradient = lennard_jones(positions)
    assert abs(energy + 1.0) < 1e-10
    assert np.linalg.norm(gradient) < 1e-8
    assert abs(hartree_from_lj(energy) * HA_EV - energy) < 1e-12
    forces = ha_per_bohr_from_lj_gradient(gradient)
    assert np.linalg.norm(forces) < 1e-8
    assert abs(BOHR_ANG - 0.52917720859) < 1e-15
    assert abs(HA_EV - 27.21138386) < 1e-12


def test_slave_returns_the_harness_pair_force():
    distance = 1.5
    angstrom = np.array([0.0, 0.0, 0.0, distance, 0.0, 0.0])
    left, right = socket.socketpair()
    counter = HarnessCounter(cache=False, cap=4)
    error = []

    def evaluate(flat):
        key = tuple(float(value) for value in np.asarray(flat).reshape(-1))
        counter.charge(key)
        return lennard_jones(flat)

    def run():
        try:
            serve(right, evaluate)
        except Exception as exc:  # pragma: no cover - the assertion is on the main thread
            error.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    energy, forces = _exchange(left, angstrom / BOHR_ANG)
    left.sendall(b"STOP        ")
    thread.join(timeout=5)
    assert not error
    expected, gradient = lennard_jones(angstrom)
    assert abs(energy - expected / HA_EV) < 1e-12
    assert np.allclose(forces, ha_per_bohr_from_lj_gradient(gradient))
    assert counter.calls == 1
    left.close()


def test_cap_is_refused_before_the_potential_runs():
    left, right = socket.socketpair()
    counter = HarnessCounter(cache=False, cap=1)
    ran = {"n": 0}

    def evaluate(flat):
        counter.charge(tuple(float(value) for value in np.asarray(flat).reshape(-1)))
        ran["n"] += 1
        return lennard_jones(flat)

    holder = {}

    def run():
        try:
            serve(right, evaluate)
        except ForceCallCap as exc:
            holder["exc"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    angstrom = np.array([0.0, 0.0, 0.0, 1.2, 0.0, 0.0])
    _exchange(left, angstrom / BOHR_ANG)
    left.sendall(b"STATUS      ")
    # The second STATUS is answered READY, then POSDATA charges call 2 and the slave closes.
    assert _read_exact(left, MSGLEN) == b"READY       "
    left.sendall(b"POSDATA     ")
    left.sendall(_pack_doubles(np.eye(3).reshape(-1)))
    left.sendall(_pack_doubles(np.eye(3).reshape(-1)))
    left.sendall(struct.pack("<i", 2))
    left.sendall(_pack_doubles(angstrom / BOHR_ANG))
    thread.join(timeout=5)
    assert isinstance(holder.get("exc"), ForceCallCap)
    assert counter.calls == 1
    assert ran["n"] == 1
    left.close()


def test_stop_before_the_next_hop_does_not_charge(tmp_path: Path):
    mon = tmp_path / "global.mon"
    mon.write_text("         0        0. -6.3912345678901E+00\n", encoding="utf-8")
    seen = lowest_relaxed_lj(mon)
    assert seen is not None
    assert abs(seen - (-6.3912345678901 * HA_EV)) < 1e-9
    left, right = socket.socketpair()
    counter = HarnessCounter(cache=False, cap=10)

    def evaluate(flat):
        counter.charge(tuple(float(value) for value in np.asarray(flat).reshape(-1)))
        return lennard_jones(flat)

    def before():
        if counter.calls >= 1 and lowest_relaxed_lj(mon) is not None:
            raise SearchStopped("success")

    holder = {}

    def run():
        try:
            serve(right, evaluate, before_eval=before)
        except SearchStopped as exc:
            holder["exc"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    angstrom = np.array([0.0, 0.0, 0.0, 1.2, 0.0, 0.0])
    _exchange(left, angstrom / BOHR_ANG)
    left.sendall(b"STATUS      ")
    thread.join(timeout=5)
    assert getattr(holder.get("exc"), "kind", None) == "success"
    assert counter.calls == 1
    left.close()


def test_poscur_uses_the_projective_cube(tmp_path):
    import numpy as np

    from optbench.runners.flame_go import CELL_ANG, _write_case

    positions = np.arange(12, dtype=np.float64).reshape(4, 3)
    _write_case(tmp_path, positions, 9, 3, 4)
    lines = (tmp_path / "poscur.ascii").read_text(encoding="utf-8").splitlines()
    side = f"{CELL_ANG:.10f}"
    assert lines[1] == f"{side} 0.0 {side}"
    assert lines[2] == f"0.0 0.0 {side}"
    text = (tmp_path / "params_new.in").read_text(encoding="utf-8")
    assert "VERBOSE 0" in text
    assert "GEOFIREDTINIT" not in text
    _write_case(tmp_path, positions, 9, 3, 4, {"init": "1.d-9", "min": "1.d-12"})
    lowered = (tmp_path / "params_new.in").read_text(encoding="utf-8")
    assert "GEOFIREDTINIT 1.d-9" in lowered
    assert "GEOFIREDTMIN 1.d-12" in lowered


def test_unruled_reference_bound_keeps_the_raw_call_count():
    from optbench.score import score_v2

    metrics = score_v2(
        [
            {"force_calls": 40, "success": False, "failure": None, "reference_bound": True},
            {"force_calls": 60, "success": False, "failure": "cap", "reference_bound": False},
        ],
        100,
    )
    assert metrics["successes"] == 0
    assert metrics["nfailed"] == 1
    assert metrics["n_cap"] == 1
    assert metrics["n_divergence"] == 0
    assert abs(metrics["par2_mean"] - (40 + 200) / 2) < 1e-9


def test_lj38_ceiling_matches_the_2014_stop():
    from optbench.campaign import load_template, resolve_problem

    root = Path(__file__).resolve().parents[1]
    problem = resolve_problem(root, "globalopt/lj38")
    template = load_template(root, "flame/lj38-minhocao")
    assert reference_energy(problem, template) == LJ38_REFERENCE
    assert abs(success_ceiling(problem, template) - (LJ38_REFERENCE + 0.001)) < 1e-12
    assert abs((LJ38_REFERENCE + 0.001) - (-173.927427)) < 1e-9
    from optbench.runners.flame_go import cap_of, fire_dt_of

    bare = resolve_problem(root, "globalopt/lj75")
    other = load_template(root, "flame/lj75-minhocao")
    assert abs(success_ceiling(bare, other) - (-397.492331 + 0.001)) < 1e-9
    assert abs(search_target(bare, other) - (-397.492331 + 0.001)) < 1e-9
    assert cap_of(bare, other) == 2_000_000
    lj98 = resolve_problem(root, "globalopt/lj98")
    template98 = load_template(root, "flame/lj98-minhocao")
    assert abs(success_ceiling(lj98, template98) - (-543.665361 + 0.001)) < 1e-9
    assert cap_of(lj98, template98) == 2_000_000
    lj150 = resolve_problem(root, "globalopt/lj150")
    template150 = load_template(root, "flame/lj150-minhocao")
    assert abs(success_ceiling(lj150, template150) - (-893.310258 + 0.001)) < 1e-9
    assert cap_of(lj150, template150) == 2_000_000
    assert fire_dt_of(template) == {"init": "1.d-9", "min": "1.d-12"}
    assert fire_dt_of(other) == {"init": "1.d-9", "min": "1.d-12"}


def test_one_cap_is_shared_and_written(tmp_path: Path):
    import pytest

    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.flame_go import (
        HA_EV,
        _dat,
        _write_case,
        cap_of,
        earr_text,
        ioput_text,
        md_lines_of,
        port_base,
    )

    root = Path(__file__).resolve().parents[1]
    lj38 = resolve_problem(root, "globalopt/lj38")
    flame38 = load_template(root, "flame/lj38-minhocao")
    lj75 = resolve_problem(root, "globalopt/lj75")
    flame75 = load_template(root, "flame/lj75-minhocao")
    lj98 = resolve_problem(root, "globalopt/lj98")
    flame98 = load_template(root, "flame/lj98-minhocao")
    lj150 = resolve_problem(root, "globalopt/lj150")
    flame150 = load_template(root, "flame/lj150-minhocao")

    assert cap_of(lj38, flame38) == 2_000_000
    assert cap_of(lj75, flame75) == 2_000_000
    assert cap_of(lj98, flame98) == 2_000_000
    assert cap_of(lj150, flame150) == 2_000_000
    with pytest.raises(ValueError, match="disagrees"):
        cap_of(
            {"id": "globalopt/lj75", "convergence": {"max_force_calls": 1_000_000}},
            {"id": "flame/lj75-minhocao", "failure_cap": 2_000_000},
        )

    published = [
        "MDDTINIT 2",
        "SOFTNIT 20",
        "MDMININIT 3",
        "GEOTOLMXF 1.d-3",
    ]
    assert md_lines_of(flame38) == published
    assert md_lines_of(flame75) == published
    assert md_lines_of(flame98) == published
    assert md_lines_of(flame150) == published
    # minhopp ekin 0.09 is kinetic energy per remaining degree of freedom.
    # 1.5 N kT matches that total when T = 0.18 (N-1) / N / kB.
    for flame, count in ((flame38, 38), (flame75, 75), (flame98, 98), (flame150, 150)):
        ediff, temperature, tmax = (
            float(part.replace("d", "e").replace("D", "E")) for part in ioput_text(flame).split()
        )
        kelvin = 0.18 * ((count - 1) / count) / (3.1668154197285284e-6 * HA_EV)
        assert abs(ediff - (1.1 / HA_EV)) < 5e-13
        assert abs(temperature - kelvin) < 5e-4
        assert tmax == 1.0e5
        enthalpy, fingerprint = (float(part) for part in earr_text(flame).split())
        assert abs(enthalpy - (1.0e-3 / HA_EV)) < 5e-16
        assert fingerprint == 0.1

    positions = np.zeros((4, 3))
    _write_case(tmp_path, positions, 9, 3, 4, {"init": "1.d-9", "min": "1.d-12"}, published, ioput_text(flame75), earr_text(flame75))
    params = (tmp_path / "params_new.in").read_text(encoding="utf-8")
    for line in published:
        assert line in params
    assert "GEOFIREDTINIT 1.d-9" in params
    assert (tmp_path / "ioput").read_text(encoding="utf-8") == ioput_text(flame75)

    text = _dat(
        {"force_calls": 10, "nfailed": 1, "median": 10},
        flame75,
        {
            "par2_mean": 2_000_000,
            "nfailed": 1,
            "successes": 0,
            "n": 1,
            "n_cap": 1,
            "n_divergence": 0,
            "n_crash": 0,
            "n_wrong_stationary": 0,
        },
        "abc",
        "1.0",
        partial=False,
        ceiling=-397.491331,
        cap=1_000_000,
    )
    assert "cap 1000000\n" in text
    assert "n_divergence 0\n" in text
    assert "PAR-2 charges 2*cap" in text
    assert port_base("globalopt/lj75") + 199 + 63 < port_base("globalopt/lj98")
    assert port_base("globalopt/lj98") + 99 + 63 < port_base("globalopt/lj150")
