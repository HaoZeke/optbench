"""Gman r235 CI-NEB records the native counter and stays off the engine table."""

import json
from pathlib import Path

from optbench.campaign import _check_stop, load_template, resolve_problem, run
from optbench.dat import parse
from optbench.engines import ENGINES
from optbench.runners.gman_neb import _settings

ROOT = Path(__file__).resolve().parents[1]


def test_gman_is_not_an_engine_table_entry():
    assert "gman" not in ENGINES


def test_gman_template_matches_the_archived_lbfgs_line_setup():
    template = load_template(ROOT, "gman/neb-pt-lbfgs-line")
    assert template["id"] == "gman/neb-pt-lbfgs-line"
    assert template["engine"] == "gman"
    assert template["problem"] == "neb/Pt-heptamer"
    assert template["code"] == "Gman"
    assert template["code_version"] == "r235"
    assert template["algorithm"] == "lbfgs"
    assert template["paper_baseline"] is True
    section = template["gman"]
    assert section["optimizer"] == "lbfgs"
    assert section["maxmove"] == 0.2
    assert section["alpha"] == 0.05
    assert section["memory"] == 25
    assert section["minimizer"] == "line"
    assert section["images"] == 8
    assert section["spring"] == 5.0
    assert section["tangent"] == "new"
    assert section["itrmax"] == 1000
    assert section["ppd"] == 1
    assert section["fmax"] == 0.01
    assert section["fsaddle"] == "off"
    assert section["method"] == "ci"
    assert section["converge"] == "Fmag"
    assert section["potential"] == "morse"
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


def test_recorded_gman_lbfgs_line_matches_the_archive_integers():
    dat = ROOT / "results" / "campaign" / "gman-r235-neb-pt-lbfgs-line.dat"
    fields = parse(dat.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "108"
    assert fields["force_calls_max"] == "155"
    assert fields["force_calls_min"] == "63"
    assert fields["nfailed"] == "0"
    assert fields["counter"] == "harness"
    assert fields["algorithm"] == "lbfgs"
    assert fields["template"] == "gman/neb-pt-lbfgs-line"
    starts = [
        json.loads(line)
        for line in (ROOT / "results" / "campaign" / "gman-r235-neb-pt-lbfgs-line.starts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["geometry"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    per_image = [float(row["raw"]) / row["images"] for row in starts]
    assert int("%d" % (sum(per_image) / len(per_image))) == 108
    assert int("%d" % max(per_image)) == 155
    assert int("%d" % min(per_image)) == 63
    assert all(row["force_calls"] == row["raw"] for row in starts)
    assert all(row["counter"] == "harness" and row["call_unit"] == "total" for row in starts)
    assert all(row["converged"] is True and row["success"] is True for row in starts)
    score = json.loads(
        (ROOT / "results" / "campaign" / "gman-r235-neb-pt-lbfgs-line.v2.json").read_text(encoding="utf-8")
    )
    assert score["schema"] == "optbench.score/v2"
    assert score["counter"] == "harness"
    assert score["score"]["n"] == 13
    assert score["score"]["nfailed"] == 0
    assert score["score"]["par2_mean"] == sum(row["raw"] for row in starts) / len(starts)
