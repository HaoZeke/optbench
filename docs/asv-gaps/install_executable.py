"""Show that asv looks up an install command inside the environment bin only.

Run: uv run python docs/asv-gaps/install_executable.py
"""

import tempfile
from pathlib import Path

from asv import util

empty = Path(tempfile.mkdtemp()) / "bin"
empty.mkdir()
try:
    util.which("uv", [str(empty)])
except OSError as exc:
    print(exc)
else:
    raise SystemExit("uv was found in an empty bin; this check is stale")
