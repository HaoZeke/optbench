"""eOn SVN templates stay on the r2549 pin and the problem grids."""

from pathlib import Path

import tomllib

from benchmarks.bench_eon_develop import lj_cut
from benchmarks.bench_eon_svn import MORSE_A, MORSE_CUTOFF, MORSE_D, MORSE_RE, SPECS, _morse_fast
from optbench.campaign import _check_stop, load_template, resolve_problem
from optbench.potentials import lennard_jones

ROOT = Path(__file__).resolve().parents[1]


def _eon_morse(positions, box=None):
    """Python copy of r2549 Morse::force, for the C kernel check."""
    import math

    atoms = len(positions) // 3
    bx = by = bz = 0.0
    if box is not None:
        bx, by, bz = box[0], box[4], box[8]
    gap_c = 1.0 - math.exp(-MORSE_A * (MORSE_CUTOFF - MORSE_RE))
    ecut = MORSE_D * (gap_c * gap_c - 1.0)
    energy = 0.0
    forces = [0.0] * len(positions)
    for i in range(atoms):
        ix = 3 * i
        for j in range(i + 1, atoms):
            jx = 3 * j
            dx = positions[ix] - positions[jx]
            dy = positions[ix + 1] - positions[jx + 1]
            dz = positions[ix + 2] - positions[jx + 2]
            if bx > 0.0:
                dx -= bx * math.floor(dx / bx + 0.5)
            if by > 0.0:
                dy -= by * math.floor(dy / by + 0.5)
            if bz > 0.0:
                dz -= bz * math.floor(dz / bz + 0.5)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                continue
            r = math.sqrt(r2)
            if r >= MORSE_CUTOFF:
                continue
            gap = 1.0 - math.exp(-MORSE_A * (r - MORSE_RE))
            energy += MORSE_D * (gap * gap - 1.0) - ecut
            bond = 2.0 * MORSE_D * gap * (gap - 1.0) * MORSE_A
            scale = bond / r
            fx, fy, fz = scale * dx, scale * dy, scale * dz
            forces[ix] += fx
            forces[ix + 1] += fy
            forces[ix + 2] += fz
            forces[jx] -= fx
            forces[jx + 1] -= fy
            forces[jx + 2] -= fz
    return energy, tuple(forces)


def test_c_morse_matches_r2549_morse():
    positions = []
    for i in range(12):
        positions.extend((0.4 * i, 0.15 * (i % 3), 0.2 * (i % 5)))
    box = [8.0, 0.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0, 12.0]
    energy, forces = _eon_morse(positions, box)
    other, fast = _morse_fast(positions, box=box)
    assert abs(energy - other) < 1e-9
    assert len(fast) == len(forces)
    for left, right in zip(fast, forces, strict=True):
        assert abs(left - right) < 1e-8
    bare, _bare_forces = _morse_fast(positions)
    assert abs(bare - _eon_morse(positions)[0]) < 1e-9


def test_lj_cut_matches_the_noise_kernel():
    positions = [0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.2, 1.3, 0.4]
    energy, forces = lj_cut(positions, cutoff=40.0)
    other, gradient = lennard_jones(positions)
    assert abs(energy - other) < 1e-9
    assert len(forces) == len(gradient)
    for left, right in zip(forces, gradient, strict=True):
        assert abs(left + float(right)) < 1e-8


def test_svn_templates_match_the_problem_grids():
    by_template = {spec["template"]: spec for spec in SPECS}
    assert len(by_template) == 21
    expected = {
        "eon-svn/ts-lj38-dimer": 200,
        "eon-svn/ts-lj38-lanczos": 200,
        "eon-svn/ts-pt-dimer": 49,
        "eon-svn/ts-pt-lanczos": 49,
        "eon-svn/de-lj38-ci": 50,
        "eon-svn/de-lj38-neb": 50,
        "eon-svn/de-pt-ci": 59,
        "eon-svn/de-pt-neb": 59,
        "eon-svn/de-2ts-ci": 51,
        "eon-svn/de-2ts-neb": 51,
        "eon-svn/neb-pt-ci": 13,
        "eon-svn/neb-pt-neb": 13,
        "eon-svn/low-lj38-dimer": 200,
        "eon-svn/low-lj38-lanczos": 200,
        "eon-svn/lj38-noise-0": 1000,
        "eon-svn/lj38-noise-1e-4": 1000,
        "eon-svn/lj38-noise-1e-3": 1000,
        "eon-svn/lj38-noise-1e-2": 1000,
        "eon-svn/saddle-pt-dimer": 10,
        "eon-svn/saddle-pt-dimer-improved": 10,
        "eon-svn/saddle-pt-lanczos": 2,
    }
    assert set(by_template) == set(expected)
    for template_id, count in expected.items():
        spec = by_template[template_id]
        template = load_template(ROOT, template_id)
        assert template["commit"] == "r2549"
        assert template["code"] == "Eon svn"
        assert template["engine"] == "eon-client"
        assert template["problem"]
        _check_stop(resolve_problem(ROOT, template["problem"]), template)
        if spec["kind"] == "akmc":
            assert len(spec["campaign_seeds"]) == count
            assert spec["campaign_starts"] == ["pt-island"]
        else:
            assert spec["campaign_seeds"]
            assert len(spec["campaign_starts"]) == count
        optimizer = template["ini"].get("Optimizer") or {}
        if optimizer.get("opt_method") == "lbfgs":
            assert "LBFGS" not in template["ini"]
            assert optimizer["lbfgs_memory"] == 20
            assert "lbfgs_max_inverse_curvature" in optimizer
        assert "ci_mmf" not in template["ini"].get("Nudged Elastic Band", {})
        assert "energy_weighted" not in template["ini"].get("Nudged Elastic Band", {})


def test_registry_pins_r2549():
    data = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in data["code"] if item["id"] == "eon-svn")
    assert row["commit"] == "r2549"
    assert row["name"] == "Eon svn"


def _two_step_endpoints() -> list[tuple[str, str, list[float]]]:
    """Centered free-cluster con text and coordinates for both ends of all 51 pairs."""
    import tarfile

    from optbench.eon_jobs import _stored
    from optbench.eon_parse import xyz_to_con
    from optbench.lj import read_component_coordinates

    archive = _stored(ROOT, "website/double-ended-saddle-search/lj38_2ts_separation/coords.tgz")
    found = []
    with tarfile.open(archive) as packed:
        for index in range(51):
            for side in ("start", "end"):
                member = packed.extractfile(f"coords/{side}_{index}.xyz")
                assert member is not None
                text = xyz_to_con(member.read().decode("utf-8"))
                found.append((f"{side}_{index}", text, read_component_coordinates(text)))
    return found


def _max_pair(positions: list[float]) -> float:
    atoms = len(positions) // 3
    best = 0.0
    for i in range(atoms):
        ix = 3 * i
        for j in range(i + 1, atoms):
            jx = 3 * j
            dx = positions[ix] - positions[jx]
            dy = positions[ix + 1] - positions[jx + 1]
            dz = positions[ix + 2] - positions[jx + 2]
            best = max(best, (dx * dx + dy * dy + dz * dz) ** 0.5)
    return best


def _max_atom_force(forces) -> float:
    best = 0.0
    for index in range(0, len(forces), 3):
        norm = (forces[index] ** 2 + forces[index + 1] ** 2 + forces[index + 2] ** 2) ** 0.5
        best = max(best, norm)
    return best


def test_lj38_2ts_endpoints_match_unshifted_cutoff_40():
    """The harness kernel for these pairs is unshifted Lennard-Jones, cutoff 40.

    Every pair sits inside 15, so a cutoff of 15 and a cutoff of 40 see the
    same pairs. The shifted kernel still differs by the cutoff constant.
    """
    from optbench.eon_harness import lj40
    from optbench.lj import lj_cluster
    from optbench.potentials import lennard_jones

    endpoints = _two_step_endpoints()
    assert len(endpoints) == 102
    for name, _text, positions in endpoints:
        energy, forces = lj_cut(positions, cutoff=40.0)
        other, gradient = lennard_jones(positions)
        surface, surface_forces = lj40(positions)
        shifted, _shifted_forces = lj_cluster(positions)
        assert _max_pair(positions) < 15.0, name
        assert abs(energy - other) < 1e-8, name
        assert abs(energy - surface) < 1e-8, name
        assert abs(energy - shifted) > 1e-4, name
        assert len(forces) == len(gradient) == len(surface_forces)
        for left, right in zip(forces, gradient, strict=True):
            assert abs(left + float(right)) < 1e-8
        for left, right in zip(forces, surface_forces, strict=True):
            assert abs(left - right) < 1e-8


def _svn_client() -> Path | None:
    import os

    env = os.environ.get("EON_SVN_CLIENT")
    if env and Path(env).is_file():
        return Path(env)
    default = Path.home() / "opt" / "eon-svn" / "bin" / "eonclient"
    if default.is_file():
        return default
    return None


def _single_point(client: Path, con: Path, potential: str, work: Path) -> tuple[float, float]:
    import subprocess

    work.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(client), "-s", "-p", potential, str(con)],
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
    )
    energy = force = None
    for line in completed.stdout.splitlines():
        if line.startswith("Energy:"):
            energy = float(line.split()[1])
        elif line.startswith("Max atom force:"):
            force = float(line.split()[-1])
    assert completed.returncode == 0, completed.stderr
    assert energy is not None and force is not None, completed.stdout
    return energy, force


def test_r2549_lj_cluster_and_ext_pot_match_the_harness_on_lj38_2ts(tmp_path: Path):
    """r2549 ``lj_cluster`` is all-pairs and unshifted. These clusters match cutoff 40.

    The campaign does not call that built-in. It serves the same kernel through
    ``ext_pot`` so the harness counter sees every force evaluation.
    """
    import os
    import shutil
    import subprocess

    import pytest

    from optbench.eon_harness import bridge_binary
    from optbench.potserver import PotServer

    client = _svn_client()
    if client is None:
        pytest.skip("r2549 eonclient is not installed")
    version = subprocess.run([str(client), "-v"], check=False, capture_output=True, text=True)
    assert "r2549" in (version.stdout + version.stderr)
    endpoints = _two_step_endpoints()
    builtin = tmp_path / "builtin"
    external = tmp_path / "ext"
    external.mkdir()
    shutil.copy(bridge_binary(), external / "ext_pot")
    os.chmod(external / "ext_pot", 0o755)
    server = PotServer(lambda positions, numbers, box: lj_cut(positions, numbers, box)).start()
    os.environ["OPTBENCH_POTSERVER"] = server.endpoint()
    try:
        for name, text, positions in endpoints:
            reference, forces = lj_cut(positions)
            con = tmp_path / f"{name}.con"
            con.write_text(text, encoding="utf-8")
            energy, peak = _single_point(client, con, "lj_cluster", builtin)
            served, served_peak = _single_point(client, con, "ext_pot", external)
            assert abs(energy - reference) < 1e-8, name
            assert abs(served - reference) < 1e-8, name
            assert abs(peak - _max_atom_force(forces)) < 1e-6, name
            assert abs(served_peak - _max_atom_force(forces)) < 1e-6, name
    finally:
        server.stop()
        os.environ.pop("OPTBENCH_POTSERVER", None)
