"""Force calls of every recorded start, one strip for each method.

Failures keep the force-call count the record stored and are marked
with a cross. The box is the quartiles of those same counts.
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
    by_stem = common.start_records()
    by_problem: dict[str, list] = defaultdict(list)
    for problem_id, problem in board["problems"].items():
        title = common.display_title(problem_id, problems, str(problem.get("title") or problem_id))
        methods: list[tuple[str, list[float], list[float]]] = []
        seen: set[str] = set()
        for row in problem.get("ranked") or []:
            records = common.records_for_row(row, by_stem)
            if not records:
                continue
            label = str(row.get("label") or row.get("method") or "method")
            if label in seen:
                label = f"{label} ({row.get('code_version') or row.get('commit') or len(seen)})"
            seen.add(label)
            success: list[float] = []
            failed: list[float] = []
            for record in records:
                calls = record.get("force_calls")
                if calls is None:
                    continue
                value = float(calls)
                if value != value:  # NaN
                    continue
                if record.get("success"):
                    success.append(value)
                else:
                    failed.append(value)
            if success or failed:
                methods.append((label, success, failed))
        if not methods:
            continue
        fig, ax, tick_labels = common.categorical_figure(
            len(methods),
            [label for label, _success, _failed in methods],
            legend_rows=1,
        )
        box_data = []
        positions = []
        all_values: list[float] = []
        success_x: list[float] = []
        success_y: list[float] = []
        failed_x: list[float] = []
        failed_y: list[float] = []
        for index, (_label, success, failed) in enumerate(methods):
            calls = success + failed
            if calls:
                box_data.append(calls)
                positions.append(index)
                all_values.extend(calls)
            for point_index, value in enumerate(success):
                offset = (((point_index * 37) % 100) / 100.0 - 0.5) * 0.28
                success_x.append(value)
                success_y.append(index + offset)
            for point_index, value in enumerate(failed):
                offset = (((point_index * 53) % 100) / 100.0 - 0.5) * 0.28
                failed_x.append(value)
                failed_y.append(index + offset)
        if success_x:
            ax.plot(
                success_x,
                success_y,
                marker="o",
                color=common.SUCCESS,
                markersize=2.2,
                linestyle="none",
                alpha=0.55,
                label="Success",
            )
        else:
            ax.plot([], [], marker="o", color=common.SUCCESS, linestyle="none", label="Success")
        if failed_x:
            ax.plot(
                failed_x,
                failed_y,
                marker="x",
                color=common.FAILURE,
                markersize=3.2,
                linestyle="none",
                label="Failure",
            )
        else:
            ax.plot([], [], marker="x", color=common.FAILURE, linestyle="none", label="Failure")
        if box_data:
            ax.boxplot(
                box_data,
                positions=positions,
                orientation="horizontal",
                widths=0.45,
                showfliers=False,
                manage_ticks=False,
                boxprops={"color": common.INK, "linewidth": 0.6},
                whiskerprops={"color": common.INK, "linewidth": 0.6},
                capprops={"color": common.INK, "linewidth": 0.6},
                medianprops={"color": common.INK, "linewidth": 0.8},
            )
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Force calls")
        ax.set_title(title)
        if all_values and common.log_scale(all_values):
            ax.set_xscale("log")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles, legend_labels = ax.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False, fontsize=6)
        n_fail = sum(len(failed) for _label, _success, failed in methods)
        caption = (
            f"Force calls on each recorded start for {title}. "
            "The box is the quartiles of those counts. "
            f"A blue point is a success and a vermillion cross is a recorded failure ({n_fail}). "
            "The horizontal position is the recorded count. "
            "The vertical offset only keeps marks from covering one another."
        )
        stem = f"distribution-{common.problem_slug(problem_id)}"
        saved = common.save_pair(fig, stem, caption)
        by_problem[problem_id].append(
            {
                "kind": "distribution",
                "problem": problem_id,
                "svg": saved["svg"],
                "pdf": saved["pdf"],
                "caption": caption,
            }
        )
    common.write_part("distribution", by_problem)
    print(f"distributions: {len(by_problem)}")


if __name__ == "__main__":
    main()
