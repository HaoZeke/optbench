"""Cost against the recorded code version, one figure per code and problem.

The points are the board's PAR-2 and mean force calls. A missing PAR-2
is left off that panel. Nothing is interpolated across a missing point.
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


def _series(rows: list[dict]) -> tuple[list[str], dict[str, dict[str, tuple[float | None, float | None]]]]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = common.version_label(row)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    labels.sort(key=common.version_key)
    series: dict[str, dict[str, tuple[float | None, float | None]]] = defaultdict(dict)
    for row in rows:
        label = common.version_label(row)
        if not label:
            continue
        method = str(row.get("method") or row.get("label") or "method")
        par2 = row.get("par2_mean")
        mean = row.get("mean_calls")
        series[method][label] = (
            None if par2 is None else float(par2),
            None if mean is None else float(mean),
        )
    return labels, series


def _runs(labels: list[str], points: dict, slot: int) -> list[tuple[list[int], list[float]]]:
    """Contiguous versions only. A missing value breaks the line."""
    runs: list[tuple[list[int], list[float]]] = []
    xs: list[int] = []
    ys: list[float] = []
    previous: int | None = None
    for position, label in enumerate(labels):
        value = points.get(label, (None, None))[slot]
        if value is None or previous is not None and position != previous + 1:
            if ys:
                runs.append((xs, ys))
            xs, ys = [], []
            if value is None:
                previous = None
                continue
        xs.append(position)
        ys.append(value)
        previous = position
    if ys:
        runs.append((xs, ys))
    return runs


def main() -> None:
    common.apply_style()
    import matplotlib.pyplot as plt

    board = common.load_board()
    problems = common.load_problems()
    by_problem: dict[str, list] = defaultdict(list)
    by_code: dict[str, list] = defaultdict(list)
    for problem_id, problem in board["problems"].items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in problem.get("ranked") or []:
            code = str(row.get("code") or "").strip()
            if code:
                grouped[code].append(row)
        title = common.display_title(problem_id, problems, str(problem.get("title") or problem_id))
        for code, rows in grouped.items():
            labels, series = _series(rows)
            if len(labels) < 2 or not series:
                continue
            stem = f"history-{common.problem_slug(problem_id)}-{common.code_token(code)}"
            caption = (
                f"{code} on {title}. "
                "Each point is one recorded version of one method. "
                "The axis is the recorded version in numeric order. "
                "PAR-2 is omitted where that row has none. "
                "No line is drawn across a missing value."
            )
            n_methods = len(series)
            ncol = 3 if n_methods > 6 else 2 if n_methods > 3 else 1
            legend_rows = (n_methods + ncol - 1) // ncol
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(common.COLUMN_IN, 4.5 + 0.24 * legend_rows),
                sharex=True,
                layout="constrained",
            )
            fig.suptitle(f"{code} on {title}", fontsize=8, color=common.INK)
            drawn = [False, False]
            values: list[list[float]] = [[], []]
            for index, (method, points) in enumerate(sorted(series.items())):
                color = common.OKABE[index % len(common.OKABE)]
                for slot, axis in enumerate(axes):
                    runs = _runs(labels, points, slot)
                    if not runs:
                        continue
                    drawn[slot] = True
                    for run_index, (xs, ys) in enumerate(runs):
                        values[slot].extend(ys)
                        axis.plot(
                            xs,
                            ys,
                            linestyle="-" if len(ys) >= 2 else "none",
                            marker="o",
                            color=color,
                            linewidth=1.1,
                            markersize=3.5,
                            label=method if run_index == 0 else None,
                        )
            names = ("PAR-2", "Mean force calls")
            for slot, axis in enumerate(axes):
                axis.set_ylabel(names[slot])
                axis.set_xticks(range(len(labels)))
                axis.set_xticklabels(labels, rotation=40, ha="right")
                if values[slot] and common.log_scale(values[slot]):
                    axis.set_yscale("log")
                if not drawn[slot]:
                    axis.text(
                        0.5,
                        0.5,
                        f"No {names[slot]} on these rows.",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        color=common.MUTED,
                    )
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
            if not any(drawn):
                plt.close(fig)
                continue
            legend_axis = axes[0] if drawn[0] else axes[1]
            handles, legend_labels = legend_axis.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    legend_labels,
                    loc="outside lower center",
                    ncol=ncol,
                    frameon=False,
                    fontsize=5.5,
                )
            saved = common.save_pair(fig, stem, caption)
            item = {
                "kind": "history",
                "problem": problem_id,
                "code": code,
                "svg": saved["svg"],
                "pdf": saved["pdf"],
                "caption": caption,
            }
            by_problem[problem_id].append(item)
            by_code[code].append(item)
    common.write_part("history", by_problem, by_code)
    print(f"history figures: {sum(len(items) for items in by_problem.values())}")


if __name__ == "__main__":
    main()
