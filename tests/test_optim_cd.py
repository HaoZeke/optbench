"""The Pt-island CD odata uses the 2014 unit reduction."""

from pathlib import Path

import numpy as np

from optbench.engines.optim_cd import R0, odata_text, paper_calls, read_con

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT
    / "inputs/sha256/1862710eb43954eb11688d06449bfa301be1a669a3717e7461f088f00d4bcd92/pt-island-con.tgz"
)


def _initial(tmp_path: Path):
    import tarfile

    with tarfile.open(ARCHIVE) as packed:
        initial = packed.extractfile("pt-island-con/initial_0.con").read().decode("utf-8")
    path = tmp_path / "initial_0.con"
    path.write_text(initial, encoding="utf-8")
    return read_con(path)


def test_scaled_odata_matches_the_2014_preamble(tmp_path: Path):
    coords, frozen, box = _initial(tmp_path)
    text = odata_text(coords, frozen, box)
    assert "usediag 2" in text
    assert "bfgsts 30 11 30 0.063 100" in text
    assert "bfgsconv 4.3895D-5" in text
    assert (
        "PARAMS %.20f %f %f %f %.20f"
        % (1.6047 * R0, *(box / R0), 9.5 / R0)
    ) in text
    assert "FREEZE 8 9 10 11 12 13 20 21" in text
    scaled = coords[0] / R0
    assert f"M {scaled[0]:.16f} {scaled[1]:.16f} {scaled[2]:.16f}" in text
    assert np.count_nonzero(frozen) == 168


def test_paper_call_parser_uses_field_six():
    log = (
        "hessian energy+gradient calls are not this line\n"
        "OPTIM> # of energy+gradient calls=                52 time=           0.06 %= 93.4\n"
        " geopt>                          **** CONVERGED ****\n"
    )
    assert paper_calls(log) == (52, True)
    assert paper_calls("no calls") == (None, False)
