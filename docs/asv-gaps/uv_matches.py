"""Show that asv.plugins.uv.Uv.matches reads an instance attribute from the class.

Run: uv run python docs/asv-gaps/uv_matches.py
"""

import inspect

from asv.plugins.uv import Uv

print(f"matches is classmethod: {isinstance(Uv.__dict__['matches'], classmethod)}")
setup_doc = inspect.getdoc(Uv._setup)
print("setup doc:")
print(setup_doc)
print("run_pip:")
print(inspect.getsource(Uv._run_pip).strip())
try:
    accepted = Uv.matches("3.12")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
else:
    print(f"matches returned {accepted!r}")
    raise SystemExit("Uv.matches('3.12') did not raise; this check is stale")
