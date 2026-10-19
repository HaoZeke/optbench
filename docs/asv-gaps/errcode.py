"""Show that Results.save drops errcode and stderr.

Run: uv run python docs/asv-gaps/errcode.py
"""

import tempfile
from pathlib import Path

from asv.results import Results
from asv.runner import BenchmarkResult

commit = "a" * 40
with tempfile.TemporaryDirectory() as directory:
    saved = Results({"machine": "box"}, {}, commit, 0, "3.12", "env", {})
    saved.add_result(
        {"name": "bench.track_force_calls", "version": "v", "params": []},
        BenchmarkResult(
            result=[1.0],
            samples=[None],
            number=[None],
            errcode=3,
            stderr="boom",
            profile=None,
        ),
    )
    saved.save(directory)
    path = Path(directory) / "box" / f"{commit[:8]}-env.json"
    text = path.read_text(encoding="utf-8")
    loaded = Results.load(str(path))
    print(f"saved errcode: {saved.errcode}")
    print(f"loaded errcode: {loaded.errcode}")
    print(f"loaded stderr: {loaded.stderr}")
    print(f"file mentions errcode: {'errcode' in text}")
    print(f"file mentions boom: {'boom' in text}")
    if loaded.errcode or "errcode" in text or "boom" in text:
        raise SystemExit("errcode survived; this check is stale")
