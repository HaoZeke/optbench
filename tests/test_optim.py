import math
import tomllib
from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.noise_catalog import specs
from optbench.runners.optim_min import fortran_d, gmax, write_odata

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
VERSION = "02aaf45245af"


def test_gmax_is_the_force_norm_over_sqrt_dof():
    assert gmax(0.01, 38) == 0.01 / math.sqrt(114)
    # The 2014 odata stored 9.36585D-04, that quotient to six figures.
    assert abs(gmax(0.01, 38) - 9.36585e-4) < 1e-9
    assert gmax(1.0e-3, 256) == 1.0e-3 / math.sqrt(768)


def test_odata_puts_userpot_last_and_writes_d_exponents(tmp_path: Path):
    path = tmp_path / "odata"
    write_odata(path, [0.1, -0.2, 0.3], gmax_value=gmax(0.01, 1), updates=4, maxbfgs=0.2, steps=10000)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("BFGSMIN ")
    assert "D" in lines[0]
    assert lines[1] == "BFGSSTEPS 10000"
    assert lines[2] == "UPDATES 4"
    assert lines[3].startswith("MAXBFGS ")
    assert lines[4] == "USERPOT"
    assert lines[5].startswith("X ")
    assert "e" not in lines[5]
    assert fortran_d(0.1).endswith("D-01") or "D" in fortran_d(0.1)


def test_optim_templates_match_the_tarball_pin():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "optim")
    assert pinned["commit"] == COMMIT
    assert pinned["committed"] == "2026-10-18"
    wanted = {
        "optim/lj38-lbfgs",
        "optim/lj38-lbfgs-tuned",
        "optim/morse-lbfgs",
        "optim/morse-lbfgs-tuned",
        "optim/noise-lbfgs-s0",
        "optim/noise-lbfgs-s1em4",
        "optim/noise-lbfgs-s1em3",
        "optim/noise-lbfgs-s1em2",
    }
    found = [row for row in list_templates(ROOT) if row["id"] in wanted]
    assert len(found) == len(wanted)
    catalog = {row["id"] for row in specs()}
    seen = set()
    for template in found:
        assert template["engine"] == "optim"
        assert template["commit"] == COMMIT
        assert template["code_version"] == VERSION
        assert template["code"] == "optim"
        assert template["algorithm"] == "LBFGS"
        assert template["id"] not in catalog
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        seen.add(template["id"])
    assert seen == {
        "optim/lj38-lbfgs",
        "optim/lj38-lbfgs-tuned",
        "optim/morse-lbfgs",
        "optim/morse-lbfgs-tuned",
        "optim/noise-lbfgs-s0",
        "optim/noise-lbfgs-s1em4",
        "optim/noise-lbfgs-s1em3",
        "optim/noise-lbfgs-s1em2",
    }
    by_id = {row["id"]: row for row in found}
    assert by_id["optim/lj38-lbfgs"]["optim"] == {"updates": 4, "maxbfgs": 0.2}
    assert by_id["optim/lj38-lbfgs-tuned"]["optim"]["updates"] == 1000
    assert by_id["optim/morse-lbfgs-tuned"]["optim"] == {"updates": 100, "maxbfgs": 2.0}
    assert by_id["optim/noise-lbfgs-s1em2"]["noise"]["sigma"] == 0.01
    assert by_id["optim/lj38-lbfgs"]["tuning"] == "default"
    assert by_id["optim/lj38-lbfgs-tuned"]["tuning"] == "tuned"
