"""Success rate of each ranked row that recorded one.

A row with no success rate is left off the chart. The caption counts
those rows. A missing rate is not drawn as zero.
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path


def _common():
    path = Path(__file__).resolve().parent / "common.py"
    spec = importlib.util.spec_from_file_location("paper_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _common()


def main() -> None:
    common.apply_style()
    board = common.load_board()
    problems = common.load_problems()
    by_problem: dict[str, list] = defaultdict(list)
    for problem_id, problem in board["problems"].items():
        title = common.display_title(problem_id, problems, str(problem.get("title") or problem_id))
        ranked = list(problem.get("ranked") or [])
        recorded = [row for row in ranked if row.get("success_rate") is not None]
        if not recorded:
            continue
        recorded.sort(key=lambda row: (row.get("rank_cost") is None, float(row.get("rank_cost") or 0)))
        labels = [str(row.get("label") or row.get("method") or "row") for row in recorded]
        rates = [100.0 * float(row["success_rate"]) for row in recorded]
        colors = [common.FAMILY_COLOR.get(str(row.get("family") or ""), common.OKABE[0]) for row in recorded]
        fig, ax, labels = common.categorical_figure(len(recorded), labels)
        positions = list(range(len(recorded)))
        ax.barh(positions, rates, color=colors, height=0.62, edgecolor="none")
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Success rate (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        missing = len(ranked) - len(recorded)
        caption = (
            f"Success rate on {title}, in rank order. "
            "The rate is the share of runs that finished. "
            f"{missing} ranked rows did not record a success rate and are not drawn."
        )
        stem = f"success-{common.problem_slug(problem_id)}"
        saved = common.save_pair(fig, stem, caption)
        by_problem[problem_id].append(
            {
                "kind": "success",
                "problem": problem_id,
                "svg": saved["svg"],
                "pdf": saved["pdf"],
                "caption": caption,
            }
        )
    common.write_part("success", by_problem)
    print(f"success charts: {len(by_problem)}")


if __name__ == "__main__":
    main()
