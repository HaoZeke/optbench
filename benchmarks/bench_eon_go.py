"""eOn develop Cambridge basin hopping.

One class is one template under templates/eon-develop that sets ``kind``.
The SVN classes for the same problems are installed by
``benchmarks.bench_eon_lane``. Parameters are the seed and the start.
``asv_full`` false keeps the default grid on one start. ``measure``
returns harness ``force_calls``, ``success``, and one ``failure``.
"""

from __future__ import annotations

from optbench.eon_harness import install_benchmarks

install_benchmarks("develop", globals())
