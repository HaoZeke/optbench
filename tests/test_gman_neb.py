"""Stock Gman 1.0.0.01 CI-NEB records the native counter and stays off the engine table."""

import json
from pathlib import Path

from optbench.campaign import _check_stop, load_template, resolve_problem, run
from optbench.dat import parse
from optbench.engines import ENGINES
from optbench.runners.gman_neb import _settings

ROOT = Path(__file__).resolve().parents[1]


def test_gman_is_not_an_engine_table_entry():
    assert "gman" not in ENGINES


def test_gman_template_is_the_stock_release():
    template = load_template(ROOT, "gman/neb-pt-lbfgs-line")
    assert template["id"] == "gman/neb-pt-lbfgs-line"
    assert template["engine"] == "gman"
    assert template["problem"] == "neb/Pt-heptamer"
    assert template["code"] == "Gman"
    assert template["code_version"] == "1.0.0.01"
    assert template["commit"] == "r235"
    assert template["algorithm"] == "lbfgs"
    assert template["tuning"] == "stock"
    assert template["paper_baseline"] is True
    section = template["gman"]
    assert section["optimizer"] == "lbfgs"
    assert section["images"] == 8
    assert section["fmax"] == 0.01
    assert section["method"] == "ci"
    assert section["potential"] == "morse"
    assert "memory" not in section
    assert "alpha" not in section
    assert "itrmax" not in section
    problem = resolve_problem(ROOT, "neb/Pt-heptamer")
    assert _check_stop(problem, template) == 10000
    assert _settings(template, problem)["optimizer"] == "lbfgs"


def test_gman_dispatch_stays_out_of_the_engine_table(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, *, output, work, limit):
        captured["problem"] = problem["id"]
        captured["template"] = template["id"]
        captured["limit"] = limit
        output.write_text("force_calls 1\n", encoding="utf-8")
        return {
            "force_calls": 1,
            "nfailed": 0,
            "starts": 1,
            "template": template["id"],
            "problem": problem["id"],
        }

    monkeypatch.setattr("optbench.runners.gman_neb.run_template", fake)
    summary = run(
        ROOT,
        "neb/Pt-heptamer",
        "gman/neb-pt-lbfgs-line",
        output=tmp_path / "benchmark.dat",
        workers=1,
        limit=2,
    )
    assert captured == {
        "problem": "neb/Pt-heptamer",
        "template": "gman/neb-pt-lbfgs-line",
        "limit": 2,
    }
    assert summary["force_calls"] == 1


def test_recorded_stock_gman_is_ranked_and_the_paper_script_is_not():
    diagnostic = ROOT / "results" / "campaign" / "gman-r235-neb-pt-lbfgs-line.dat"
    assert diagnostic.is_file()
    assert not (ROOT / "results" / "campaign" / "gman-r235-neb-pt-lbfgs-line.starts.jsonl").exists()
    assert not (ROOT / "results" / "campaign" / "gman-r235-neb-pt-lbfgs-line.v2.json").exists()
    paper = parse(diagnostic.read_text(encoding="utf-8"))
    assert paper["force_calls"] == "108"
    assert paper["code_version"] == "r235"
    assert "counter" not in paper

    dat = ROOT / "results" / "campaign" / "gman-1.0.0.01-neb-pt-lbfgs-line.dat"
    fields = parse(dat.read_text(encoding="utf-8"))
    assert fields["code_version"] == "1.0.0.01"
    assert fields["commit"] == "r235"
    assert fields["counter"] == "harness"
    assert fields["tuning"] == "stock"
    assert fields["algorithm"] == "lbfgs"
    assert fields["template"] == "gman/neb-pt-lbfgs-line"
    starts = [
        json.loads(line)
        for line in (ROOT / "results" / "campaign" / "gman-1.0.0.01-neb-pt-lbfgs-line.starts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["geometry"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    assert all(row["force_calls"] == row["raw"] for row in starts)
    assert all(row["counter"] == "harness" and row["call_unit"] == "total" for row in starts)
    assert all("converged" in row and "success" in row for row in starts)
    score = json.loads(
        (ROOT / "results" / "campaign" / "gman-1.0.0.01-neb-pt-lbfgs-line.v2.json").read_text(encoding="utf-8")
    )
    assert score["schema"] == "optbench.score/v2"
    assert score["counter"] == "harness"
    assert score["provenance"]["code_version"] == "1.0.0.01"
    assert score["score"]["n"] == 13
    assert score["score"]["par2_mean"] == sum(row["raw"] for row in starts) / len(starts)
