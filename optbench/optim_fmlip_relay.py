"""File OPTIM launches as ``pysocket/fmlip_relay.py``.

The campaign copies this file into the relay root. The interpreter and
``PYTHONPATH`` come from the parent process.
"""

from __future__ import annotations

import os
import sys

_root = os.environ.get("OPTBENCH_ROOT")
if _root and _root not in sys.path:
    sys.path.insert(0, _root)

from optbench.optim_relay import main

if __name__ == "__main__":
    main()
