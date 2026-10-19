"""The OPTIM second-order counter is the paper's energy+gradient field."""

from optbench.optim_loweig import PREAMBLE, energy_gradient_calls, last_overlap

LOG = """\
 xmylbfgs> overlap between current and target vector  0.99313251024526428
 xmylbfgs> overlap converged
 geopt>                          **** CONVERGED ****
OPTIM> # of energy+gradient calls=                48 time=           0.00 %=  0.0
OPTIM> # of energy+gradient+Hessian calls=         0 time=           0.00 %=  0.0
"""

PAPER_LOG = """\
 xmylbfgs> overlap between current and target vector  0.99255993823375965
 xmylbfgs> overlap converged
 geopt>                          **** CONVERGED ****
 OPTIM> # of energy+gradient calls=                50 time=           0.00 %=  0.0
 OPTIM> # of energy+gradient+Hessian calls=         0 time=           0.00 %=  0.0
"""


def test_the_paper_counter_skips_the_hessian_line():
    assert energy_gradient_calls(LOG) == 48
    assert energy_gradient_calls(PAPER_LOG) == 50
    assert last_overlap(LOG) > 0.99
    assert last_overlap(PAPER_LOG) > 0.99


def test_the_preamble_is_the_website_bytes():
    assert "BFGSTSPC" not in PREAMBLE.upper()
    assert PREAMBLE.splitlines()[-3:] == ["updates 4 100 ", "usediag 2", "READVEC"]
