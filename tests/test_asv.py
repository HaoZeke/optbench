import json
import stat
from pathlib import Path

from benchmarks.bench_lj38 import Lj38EonLbfgsM100, start_names
from optbench.asv_uv import Uv
from optbench.campaign import run_client
from optbench.samples import sample_stats

ROOT = Path(__file__).resolve().parents[1]


def test_asv_uses_the_uv_environment():
    conf = json.loads((ROOT / "asv.conf.json").read_text(encoding="utf-8"))
    assert conf["environment_type"] == "uv-pip"
    assert conf["plugins"] == [".optbench.asv_uv"]
    assert conf["pythons"] == ["3.12"]
    assert "uv pip install" in conf["install_command"][0]
    assert Uv.tool_name == "uv-pip"
    assert Uv.matches("3.12") is True
    assert Uv.matches("not-a-version") is False


def test_each_lj38_start_is_one_asv_parameter():
    names = start_names()
    assert len(names) == 1000
    assert Lj38EonLbfgsM100.params == names
    assert Lj38EonLbfgsM100.repeat == 1
    assert Lj38EonLbfgsM100.param_names == ["start"]
    assert Lj38EonLbfgsM100.track_force_calls.unit == "force calls"


def test_one_client_subprocess_returns_its_force_calls(tmp_path: Path):
    con = tmp_path / "start.con"
    con.write_text("pos\n", encoding="utf-8")
    client = tmp_path / "eonclient"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "assert Path('pos.con').is_file()\n"
        "assert 'lbfgs_memory' in Path('config.ini').read_text()\n"
        "Path('results.dat').write_text('0 termination_reason\\n181 total_force_calls\\n')\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    calls = run_client(str(client), "lbfgs_memory = 100\n", str(con), str(tmp_path / "work"))
    assert calls == 181


def test_sample_stats_are_the_distribution_not_one_rounded_mean():
    stats = sample_stats([10, 20, 30, 40])
    assert stats["n"] == 4
    assert stats["mean"] == 25
    assert stats["min"] == 10
    assert stats["max"] == 40
    assert stats["q25"] == 17.5
    assert stats["q75"] == 32.5
    assert stats["stdev"] > 0
