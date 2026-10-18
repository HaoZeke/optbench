"""eOn SVN r2549 harness benchmarks.

One class is one template. Parameters are the seed and the start.
``measure`` returns harness ``force_calls``, ``success``, and one
``failure``. The default grid is one start unless ``OPTBENCH_ASV_FULL=1``
at import, and a template with ``asv_full`` false stays on that one
start. A warm cache makes ``time_search`` a cache read. The search wall
time is the ``seconds`` field stored with the campaign start.
"""

from __future__ import annotations

from optbench.eon_harness import install_benchmarks

install_benchmarks("svn", globals())
