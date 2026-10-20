"""Stock OPTIM double-ended parsing. The binary is not required."""

from optbench.engines.optim_dneb import finish_text, odata_text, paper_calls
import numpy as np


def test_stock_odata_keeps_the_archived_deck_and_the_archive_symbol():
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    text = odata_text(coords, cap=250000)
    assert text.startswith("debug\nNEWCONNECT 20 1 2.3 13 7 5 0.024\n")
    assert "comment NEWNEB 50 500 0.0001\n" in text
    assert "BFGSSTEPS" not in text
    assert "STEPS 200\n" in text
    assert "REOPTIMISEENDPOINTS\n" in text
    assert "BFGSMIN 0.000001\n" in text
    assert "EDIFFTOL 1D-6\n" in text
    assert "DIJKSTRA 2\n" in text
    assert "250000" not in text
    assert text.rstrip().endswith("AX 4.0000000000000000 5.0000000000000000 6.0000000000000000")
    finish = finish_text(coords)
    assert finish.startswith("1.0000000000000000 2.0000000000000000 3.0000000000000000\n")


def test_paper_calls_reads_the_last_energy_gradient_line():
    log = "\n".join(
        [
            "OPTIM> # of energy+gradient calls=              10 time=           0.01 %=  1.3",
            "OPTIM> # of energy+gradient calls=              12 time=           0.02 %=  1.3",
            "Connected path found",
        ]
    )
    calls, connected = paper_calls(log)
    assert calls == 12
    assert connected is True
    hessian = "OPTIM> # of energy+gradient+Hessian calls=              99 time= 0.1\n"
    calls, connected = paper_calls(hessian)
    assert calls is None
    assert connected is False
