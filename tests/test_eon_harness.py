"""Bridge and grid checks for the r2549 harness."""

import json
import os
from pathlib import Path

from optbench.extpot import format_extpot
from optbench.potserver import PotServer

ROOT = Path(__file__).resolve().parents[1]


def test_c_bridge_matches_the_python_reply(tmp_path, monkeypatch):
    from optbench.eon_harness import bridge_binary

    def evaluate(positions, numbers, box):
        del numbers, box
        energy = float(sum(positions))
        forces = tuple(-float(value) for value in positions)
        return energy, forces

    server = PotServer(evaluate).start()
    monkeypatch.setenv("OPTBENCH_POTSERVER", server.endpoint())
    monkeypatch.chdir(tmp_path)
    request = "\n".join(
        [
            "10 0 0",
            "0 10 0",
            "0 0 10",
            "1 0.0 0.0 0.0",
            "1 1.5 0.0 0.0",
        ]
    )
    (tmp_path / "from_eon_to_extpot").write_text(request + "\n", encoding="utf-8")
    import subprocess

    proc = subprocess.run([str(bridge_binary())], cwd=tmp_path, check=False, text=True, capture_output=True)
    server.stop()
    assert proc.returncode == 0, proc.stderr
    reply = (tmp_path / "from_extpot_to_eon").read_text(encoding="utf-8")
    expected = format_extpot(1.5, (0.0, 0.0, 0.0, -1.5, 0.0, 0.0))
    got = [[float(part) for part in line.split()] for line in reply.splitlines() if line.strip()]
    want = [[float(part) for part in line.split()] for line in expected.splitlines() if line.strip()]
    assert got == want


def test_svn_default_grid_is_one_start():
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    from benchmarks.bench_eon_lane import EonSvnTsLj38DimerLjcut

    assert EonSvnTsLj38DimerLjcut.params[0] == [1]
    assert EonSvnTsLj38DimerLjcut.params[1] == ["0000"]
    assert EonSvnTsLj38DimerLjcut.template_name == "eon-svn/ts-lj38-dimer-ljcut"


def test_client_default_grid_is_one_start():
    assert os.environ.get("OPTBENCH_ASV_FULL") != "1"
    from benchmarks.bench_eon_client import EonClientNebPtLbfgs

    assert len(EonClientNebPtLbfgs.params[1]) == 1
    assert EonClientNebPtLbfgs.template_name == "eon-client/neb-pt-lbfgs"


def test_coordinate_range_names_the_archive_index():
    from optbench.eon_harness import template_starts

    assert template_starts({"kind": "basin", "start_first": 1, "start_count": 3}) == ["1", "2", "3"]


def test_lj75_coordinate_file_is_75_atoms(tmp_path):
    from optbench.eon_harness import basin_con
    from optbench.surfaces import read_con

    path = basin_con(
        {
            "archive": "website/globalopt/lj75/lj75_clusters.tgz",
            "member": "coords.{start}",
        },
        "1",
        tmp_path,
    )
    positions, _fixed, _cell = read_con(path.read_text(encoding="utf-8"))
    assert len(positions) == 75
    ten = basin_con(
        {
            "archive": "website/globalopt/lj75/lj75_clusters.tgz",
            "member": "coords.{start}",
        },
        "10",
        tmp_path,
    )
    positions_ten, _fixed, _cell = read_con(ten.read_text(encoding="utf-8"))
    assert len(positions_ten) == 75
    assert positions_ten.tolist() != positions.tolist()


def test_develop_lj75_keeps_the_lj38_ini():
    from optbench.campaign import load_template
    from optbench.eon_harness import _family_of, template_starts

    loaded = load_template(ROOT, "eon-develop/bh-lj75")
    assert _family_of(loaded) == "develop"
    assert loaded["commit"] == "80e94f88b54e24815287860f32a0558683e9c02b"
    assert int(loaded["failure_cap"]) == 2000000
    assert loaded["ini"]["Main"]["temperature"] == 8000
    assert loaded["ini"]["Basin Hopping"]["displacement"] == 0.5
    assert loaded["ini"]["Basin Hopping"]["stop_energy"] == "-397.491331"
    assert loaded["ini"]["Optimizer"]["lbfgs_memory"] == 20
    starts = template_starts(loaded)
    assert starts[0] == "1"
    assert starts[-1] == "200"
    assert len(starts) == 200
    for name in ("bh-lj38", "bh-lj98", "bh-lj150"):
        other = load_template(ROOT, f"eon-develop/{name}")
        assert int(other["failure_cap"]) == 2000000
        assert other["ini"]["Main"]["temperature"] == 8000
        assert other["ini"]["Basin Hopping"]["displacement"] == 0.5


def test_wall_clock_stop_stays_in_the_start_file(tmp_path):
    from optbench.eon_harness import write_campaign

    template = {
        "id": "eon-svn/bh-lj38",
        "problem": "globalopt/lj38",
        "code": "Eon svn",
        "code_version": "r2549",
        "commit": "r2549",
        "algorithm": "Basin hopping",
        "reducer": "harness",
    }
    row = {
        "force_calls": 10.0,
        "success": False,
        "failure": "crash",
        "timed_out": True,
        "counter": "harness",
    }
    output = tmp_path / "campaign.dat"
    summary = write_campaign(template, [(314159, "0", row)], output)
    assert summary is not None
    assert summary["starts"] == 1
    assert summary["nfailed"] == 1
    saved = json.loads(output.with_suffix(".starts").read_text(encoding="utf-8"))
    assert saved["success"] is False
    assert saved["failure"] == "crash"
    assert saved["force_calls"] == 10
