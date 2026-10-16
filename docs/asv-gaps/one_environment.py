"""Show that a benchmark has no environment attribute and the config has one type.

Run: uv run python docs/asv-gaps/one_environment.py
"""

import inspect
import json
from pathlib import Path

from asv_runner.benchmarks._base import Benchmark

root = Path(__file__).resolve().parents[2]
conf = json.loads((root / "asv.conf.json").read_text(encoding="utf-8"))
source = inspect.getsource(Benchmark.__init__)
print(f"environment_type: {conf['environment_type']!r}")
print(f"type of environment_type: {type(conf['environment_type']).__name__}")
print(f"Benchmark.__init__ mentions environment: {'environment' in source}")
names = [name for name in dir(Benchmark) if "env" in name.lower()]
print(f"Benchmark attributes containing env: {names or 'none'}")
