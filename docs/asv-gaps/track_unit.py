"""Show that asv.util.human_value drops a unit other than seconds or bytes.

Run: uv run python docs/asv-gaps/track_unit.py
"""

from asv.util import human_value

force = human_value(139, "force calls")
seconds = human_value(1.5, "seconds")
print(f"force calls: {force!r}")
print(f"seconds: {seconds!r}")
if "force" in str(force):
    raise SystemExit("unit was displayed; this reproducer is stale")
