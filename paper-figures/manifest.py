"""Join the four figure manifests into one file the site can read."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "paper-figures" / "parts"
OUT = ROOT / "paper-figures" / "manifest.json"


def main() -> None:
    problems: dict[str, list] = {}
    codes: dict[str, list] = {}
    order = ("success", "family", "distribution", "history")
    loaded = []
    for name in order:
        path = PARTS / f"{name}.json"
        if not path.is_file():
            continue
        loaded.append(json.loads(path.read_text(encoding="utf-8")))
    for part in loaded:
        for problem_id, items in (part.get("problems") or {}).items():
            problems.setdefault(problem_id, []).extend(items)
        for code, items in (part.get("codes") or {}).items():
            codes.setdefault(code, []).extend(items)
    OUT.write_text(json.dumps({"problems": problems, "codes": codes}, indent=2) + "\n", encoding="utf-8")
    count = sum(len(items) for items in problems.values())
    print(f"manifest figures: {count}")


if __name__ == "__main__":
    main()
