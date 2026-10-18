"""eOn 3.5.0 source-build harness benchmarks.

The binary is commit 4d8b34baf1ba8d2d980df29838bac1f969d1662c. It prints
``3.5.0 (4d8b34ba)``. The conda-forge binary that prints ``3.5.0 (7ec1840)``
is a different build and is not used here. The default grid is one start
unless ``OPTBENCH_ASV_FULL=1`` at import.
"""

from __future__ import annotations

from optbench.eon_harness import install_benchmarks

install_benchmarks("client", globals())
