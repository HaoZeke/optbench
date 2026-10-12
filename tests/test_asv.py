import json
from pathlib import Path

from benchmarks.bench_lj38 import Lj38EonLbfgsM100

ROOT = Path(__file__).resolve().parents[1]


def test_asv_does_not_build_or_round_the_client():
    conf = json.loads((ROOT / "asv.conf.json").read_text(encoding="utf-8"))
    assert conf["environment_type"] == "existing"
    assert conf["build_command"] == []
    assert conf["install_command"] == []
    assert conf["benchmark_dir"] == "benchmarks"


def test_lj38_track_keeps_the_unrounded_mean():
    bench = Lj38EonLbfgsM100()
    assert bench.repeat == 1
    assert bench.problem == "minimization/lj38"
    assert bench.template == "eon-client/lbfgs-m100"
    assert bench.track_force_calls({"mean_unrounded": 181.4}) == 181.4
    assert Lj38EonLbfgsM100.track_force_calls.unit == "force calls"
