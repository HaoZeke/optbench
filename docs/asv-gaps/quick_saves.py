"""Show that asv 0.6.6 saves a --quick run even though the help says it does not.

Run: uv run python docs/asv-gaps/quick_saves.py
"""

import inspect

import asv.commands.run as runmod

text = inspect.getsource(runmod)
help_at = text.find("are not saved.")
if help_at < 0:
    raise SystemExit("help text moved; this reproducer is stale")
start = text.rfind('help="""', 0, help_at)
end = text.find('"""', help_at)
help_line = text[start + len('help="""') : end].strip()
print("help:")
print(help_line)
skip_at = text.find("skip_save = ")
if skip_at < 0:
    raise SystemExit("skip_save assignment moved; this reproducer is stale")
skip_end = text.find("skip_list", skip_at)
skip_expr = text[skip_at:skip_end].strip()
print("skip_save:")
print(skip_expr)
if "not saved" not in help_line:
    raise SystemExit("help no longer says results are not saved")
if "quick" in skip_expr:
    raise SystemExit("skip_save now depends on quick; this reproducer is stale")
