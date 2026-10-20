"""Stock OPTIM BFGSTS keeps the source defaults and the problem stop."""

from pathlib import Path

import numpy as np

from optbench.engines.optim_hef import SVNREV, TARBALL_SHA256, odata_text

ROOT = Path(__file__).resolve().parents[1]


def test_stock_odata_does_not_copy_the_2014_keyword_values():
    frozen = np.zeros(16, dtype=bool)
    frozen[:3] = True
    text = odata_text("/usr/bin/python", Path("/tmp/hook"), True, 1.0e-5, 10000, frozen)
    assert "\nBFGSTS\n" in text
    assert "bfgsts 30" not in text.lower()
    assert "usediag" not in text.lower()
    assert "NOHESS" in text
    assert "BFGSCONV 1.0000000000000001D-05" in text
    assert "STEPS 10000" in text
    assert "READVEC" in text
    assert "FREEZE 1 2 3" in text
    assert "PARAMS" not in text
    assert SVNREV.startswith("df8da599")
    assert len(TARBALL_SHA256) == 64
    template = (ROOT / "templates/optim/hef-ts-pt.toml").read_text(encoding="utf-8")
    assert 'method = "hef"' in template
    assert "df8da599" in template
    assert TARBALL_SHA256 in template
