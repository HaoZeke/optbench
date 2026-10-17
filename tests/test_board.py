"""Behaviour of the v2 board built from committed result files."""

import json
from pathlib import Path

from optbench.board import build_board, write_board
from optbench.score import wilson_interval

PIN = "a" * 40
OTHER = "b" * 40


def _problem(root: Path, starts: int = 2, cap: int = 10) -> None:
    path = root / "problems" / "minimization" / "toy.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'id = "minimization/toy"',
                'title = "Toy"',
                'suite = "minimization"',
                'problem = "toy"',
                'contract = "stated"',
                "",
                "[sample]",
                f"starts = {starts}",
                "",
                "[convergence]",
                "force_norm = 0.01",
                f"max_force_calls = {cap}",
                "",
                "[metrics]",
                'required = ["force_calls"]',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _template(root: Path, template_id: str = "toy/fast", *, images: int | None = None) -> None:
    path = root / "templates" / f"{template_id}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'id = "{template_id}"',
        'engine = "toy"',
        'problem = "minimization/toy"',
        'code = "toy"',
        'code_version = "1.0"',
        f'commit = "{PIN}"',
        'tuning = "default"',
    ]
    if images is not None:
        lines.append('reducer = "per_image"')
        lines.append("")
        lines.append("[neb]")
        lines.append(f"moving_images = {images}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _registry(root: Path, commit: str = PIN) -> None:
    path = root / "codes" / "registry.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "[[code]]",
                'id = "toy"',
                'name = "toy"',
                f'commit = "{commit}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _dat(root: Path, stem: str, commit: str, *, counter: str = "harness", template: str = "toy/fast") -> Path:
    path = root / "results" / "campaign" / f"{stem}.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "force_calls 1",
                "nfailed 0",
                "code toy",
                "code_version 1.0",
                f"commit {commit}",
                f"template {template}",
                "problem minimization/toy",
                f"counter {counter}",
                "tuning default",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _starts(root: Path, stem: str, rows: list[dict]) -> None:
    path = root / "results" / "campaign" / f"{stem}.starts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(calls: float, success: bool, start: str, failure: str | None = None) -> dict:
    record = {"force_calls": calls, "success": success, "start": start, "counter": "harness"}
    if not success:
        record["failure"] = failure or "cap"
    return record


def test_full_harness_row_is_ranked_by_par2(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN)
    _starts(
        tmp_path,
        "toy-fast",
        [_row(3, True, "0"), _row(10, False, "1", "cap")],
    )

    document = build_board(tmp_path)
    problem = document["problems"]["minimization/toy"]
    assert problem["partial"] == []
    assert problem["not_comparable"] == []
    assert len(problem["ranked"]) == 1
    row = problem["ranked"][0]
    assert row["template"] == "toy/fast"
    assert row["par2_mean"] == (3 + 20) / 2
    assert row["mean_calls"] == (3 + 10) / 2
    assert row["median_calls"] == (3 + 10) / 2
    assert row["min_calls"] == 3
    assert row["max_calls"] == 10
    assert row["n"] == 2
    assert row["nfailed"] == 1
    assert row["failures"]["cap"] == 1
    assert row["success_rate"] == 0.5
    low, high = wilson_interval(1, 2)
    assert row["wilson_low"] == low
    assert row["wilson_high"] == high
    assert row["current"] is True
    assert row["registry_match"] is True
    assert row["partial"] is False


def test_short_grid_stays_off_the_ranked_table(tmp_path: Path):
    _problem(tmp_path, starts=4, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN)
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(4, True, "1")])

    document = build_board(tmp_path)
    problem = document["problems"]["minimization/toy"]
    assert problem["ranked"] == []
    assert len(problem["partial"]) == 1
    partial = problem["partial"][0]
    assert partial["n"] == 2
    assert partial["partial_reason"] == "scored 2 of 4 starts"
    assert document["partial"][0]["template"] == "toy/fast"


def test_duplicate_files_are_kept_and_the_registry_commit_is_current(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path, PIN)
    kept = _dat(tmp_path, "toy-fast-old", OTHER)
    current = _dat(tmp_path, "toy-fast", PIN)
    _starts(tmp_path, "toy-fast-old", [_row(4, True, "0"), _row(4, True, "1")])
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(3, True, "1")])

    document = build_board(tmp_path)
    ranked = document["problems"]["minimization/toy"]["ranked"]
    assert len(ranked) == 2
    by_commit = {row["commit"]: row for row in ranked}
    assert set(by_commit) == {PIN, OTHER}
    assert by_commit[PIN]["current"] is True
    assert by_commit[OTHER]["current"] is False
    assert kept.is_file()
    assert current.is_file()
    assert ranked[0]["commit"] == PIN

    tied = tmp_path / "results" / "campaign" / "toy-fast-tied.dat"
    tied.write_text(current.read_text(encoding="utf-8").replace(PIN, OTHER), encoding="utf-8")
    _starts(tmp_path, "toy-fast-tied", [_row(3, True, "0"), _row(3, True, "1")])
    # The copy above keeps code_version 1.0 and the other commit. Same PAR-2.
    # Give the tied file its own stem's starts; the current file stays PIN.
    tied_board = build_board(tmp_path)["problems"]["minimization/toy"]["ranked"]
    assert [row["commit"] for row in tied_board if row["par2_mean"] == 3][:2] == [PIN, OTHER]


def test_client_counter_is_not_on_the_ranked_board(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN, counter="client")
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(4, True, "1")])

    document = build_board(tmp_path)
    problem = document["problems"]["minimization/toy"]
    assert problem["ranked"] == []
    assert len(problem["not_comparable"]) == 1
    assert problem["not_comparable"][0]["counter"] == "client"
    assert document["partial"] == []


def test_a_dat_with_no_value_is_listed_and_left_in_place(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    path = tmp_path / "results" / "campaign" / "broken.dat"
    path.parent.mkdir(parents=True)
    path.write_text("force_calls\n", encoding="utf-8")

    document = build_board(tmp_path)
    assert document["ranked_count"] == 0
    assert len(document["unparsed"]) == 1
    assert document["unparsed"][0]["path"] == "results/campaign/broken.dat"
    assert "no value" in document["unparsed"][0]["note"]
    assert path.read_text(encoding="utf-8") == "force_calls\n"


def test_a_problem_without_a_sample_count_does_not_invent_one(tmp_path: Path):
    path = tmp_path / "problems" / "saddle-search" / "island.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                'id = "saddle-search/island"',
                'title = "Island"',
                'suite = "saddle-search"',
                'problem = "island"',
                'contract = "stated"',
                "",
                "[metrics]",
                'required = ["force_calls"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    template = tmp_path / "templates" / "toy" / "fast.toml"
    template.parent.mkdir(parents=True)
    template.write_text(
        "\n".join(
            [
                'id = "toy/fast"',
                'engine = "toy"',
                'problem = "saddle-search/island"',
                'code = "toy"',
                'code_version = "1.0"',
                f'commit = "{PIN}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _registry(tmp_path)
    short = _dat(tmp_path, "short", PIN)
    short.write_text(
        short.read_text(encoding="utf-8").replace("problem minimization/toy", "problem saddle-search/island"),
        encoding="utf-8",
    )
    wide = _dat(tmp_path, "wide", PIN)
    wide.write_text(
        wide.read_text(encoding="utf-8").replace("problem minimization/toy", "problem saddle-search/island"),
        encoding="utf-8",
    )
    _starts(tmp_path, "short", [_row(3, True, "0"), _row(4, True, "1")])
    _starts(tmp_path, "wide", [_row(3, True, str(index)) for index in range(4)])

    problem = build_board(tmp_path)["problems"]["saddle-search/island"]
    assert problem["full_n"] is None
    assert problem["partial"] == []
    assert {row["n"] for row in problem["ranked"]} == {2, 4}


def test_ranked_false_stays_off_the_table(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    path = _dat(tmp_path, "toy-fast", PIN)
    path.write_text(path.read_text(encoding="utf-8") + "ranked false\n", encoding="utf-8")
    _starts(tmp_path, "toy-fast", [_row(3, True, "0"), _row(4, True, "1")])

    problem = build_board(tmp_path)["problems"]["minimization/toy"]
    assert problem["ranked"] == []
    assert problem["partial"][0]["partial_reason"] == "result file sets ranked false"


def test_per_image_band_uses_the_image_cost(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=100)
    _template(tmp_path, images=2)
    _registry(tmp_path)
    dat = _dat(tmp_path, "toy-fast", PIN)
    dat.write_text(dat.read_text(encoding="utf-8") + "reducer per_image\n", encoding="utf-8")
    _starts(
        tmp_path,
        "toy-fast",
        [
            {"force_calls": 10, "success": True, "start": "0", "counter": "harness"},
            {"force_calls": 20, "success": True, "start": "1", "counter": "harness"},
        ],
    )

    row = build_board(tmp_path)["problems"]["minimization/toy"]["ranked"][0]
    assert row["par2_mean"] == (5 + 10) / 2
    assert row["mean_calls"] == (5 + 10) / 2


def _asv_one_start(root: Path) -> None:
    machine = root / "results" / "asv" / "testbox"
    machine.mkdir(parents=True)
    (machine / "machine.json").write_text(
        json.dumps({"machine": "testbox", "version": 1}) + "\n",
        encoding="utf-8",
    )
    params = [["'0'"], ["'0'"]]
    columns = ["result", "params", "version", "started_at", "duration"]
    prefix = "bench_lj38.Lj38EonLbfgsM100"
    tracks = {
        "track_force_calls": [4.0],
        "track_success": [1.0],
        "track_failure_cap": [0.0],
        "track_failure_wrong_stationary": [0.0],
        "track_failure_divergence": [0.0],
        "track_failure_crash": [0.0],
    }
    results = {f"{prefix}.{name}": [value, params, "v", 0, 0.1] for name, value in tracks.items()}
    payload = {
        "commit_hash": "c" * 40,
        "env_name": "py",
        "date": 1,
        "params": {"machine": "testbox"},
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }
    (machine / "cccc-py.json").write_text(json.dumps(payload), encoding="utf-8")


def test_one_start_asv_grid_is_partial(tmp_path: Path):
    _problem(tmp_path, starts=1000, cap=10000)
    problem = tmp_path / "problems" / "minimization" / "toy.toml"
    text = problem.read_text(encoding="utf-8").replace('id = "minimization/toy"', 'id = "minimization/lj38"')
    text = text.replace('problem = "toy"', 'problem = "lj38"')
    text = text.replace('title = "Toy"', 'title = "Lennard-Jones 38 clusters"')
    problem.write_text(text, encoding="utf-8")
    template = tmp_path / "templates" / "eon-client" / "lbfgs-m100.toml"
    template.parent.mkdir(parents=True)
    template.write_text(
        "\n".join(
            [
                'id = "eon-client/lbfgs-m100"',
                'engine = "eon"',
                'problem = "minimization/lj38"',
                'code = "eOn"',
                'code_version = "3.5.0"',
                f'commit = "{PIN}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _asv_one_start(tmp_path)

    document = build_board(tmp_path)
    partial = [row for row in document["partial"] if row["entry"].endswith("Lj38EonLbfgsM100")]
    assert len(partial) == 1
    assert partial[0]["n"] == 1
    assert partial[0]["partial"] is True
    assert "1 of 1000" in partial[0]["partial_reason"]
    ranked = [
        row
        for problem_rows in document["problems"].values()
        for row in problem_rows["ranked"]
        if str(row["entry"]).endswith("Lj38EonLbfgsM100")
    ]
    assert ranked == []


def test_profile_lines_up_the_same_starts(tmp_path: Path):
    _problem(tmp_path, starts=2, cap=100)
    _template(tmp_path, "toy/cheap")
    _template(tmp_path, "toy/dear")
    _registry(tmp_path)
    _dat(tmp_path, "cheap", PIN, template="toy/cheap")
    _dat(tmp_path, "dear", PIN, template="toy/dear")
    _starts(tmp_path, "cheap", [_row(2, True, "cluster_0000.con"), _row(2, True, "cluster_0001.con")])
    _starts(tmp_path, "dear", [_row(4, True, "0"), _row(4, True, "1")])

    written = write_board(tmp_path, tmp_path / "board")
    problem = written["problems"]["minimization/toy"]
    rho = problem["profile"]["rho"]
    cheap = next(label for label in rho if label.startswith("toy/cheap"))
    dear = next(label for label in rho if label.startswith("toy/dear"))
    assert rho[cheap][0] == 1.0
    assert rho[dear][0] == 0.0
    assert rho[cheap][1] == 1.0
    assert rho[dear][1] == 1.0
    assert (tmp_path / "board" / "plots" / "minimization-toy-profile.svg").stat().st_size > 0
    assert (tmp_path / "board" / "plots" / "minimization-toy-profile.png").stat().st_size > 0
    assert (tmp_path / "board" / "plots" / "minimization-toy-cost.svg").stat().st_size > 0
    assert (tmp_path / "board" / "plots" / "minimization-toy-cost.png").stat().st_size > 0
    include = (tmp_path / "docs" / "include" / "v2-board.org").read_text(encoding="utf-8")
    assert "toy/cheap" in include
    assert "PAR-2" in include
    page = json.loads((tmp_path / "board" / "index.json").read_text(encoding="utf-8"))
    assert page["ranked_count"] == 2


def test_a_partial_problem_drops_a_plot_left_from_an_earlier_run(tmp_path: Path):
    _problem(tmp_path, starts=4, cap=10)
    _template(tmp_path)
    _registry(tmp_path)
    _dat(tmp_path, "toy-fast", PIN)
    _starts(tmp_path, "toy-fast", [_row(1, True, "0"), _row(1, True, "1")])
    plots = tmp_path / "board" / "plots"
    plots.mkdir(parents=True)
    stale = plots / "minimization-toy-profile.svg"
    stale.write_text("stale", encoding="utf-8")
    site = tmp_path / "docs" / "source" / "_static" / "board"
    site.mkdir(parents=True)
    site_stale = site / "minimization-toy-cost.png"
    site_stale.write_text("stale", encoding="utf-8")

    written = write_board(tmp_path, tmp_path / "board")

    assert written["problems"]["minimization/toy"]["ranked"] == []
    assert not stale.exists()
    assert not site_stale.exists()
    assert list(plots.iterdir()) == []
    assert list(site.iterdir()) == []
