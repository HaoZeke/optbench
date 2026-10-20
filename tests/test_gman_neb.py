"""Gman 1.0.0.01 steepest descent and quick-min on the Pt heptamer."""

import json
import statistics
from pathlib import Path

from optbench.board import _cap_of
from optbench.campaign import _check_stop, load_template, resolve_problem, run
from optbench.dat import parse
from optbench.engines import ENGINES
from optbench.runners.gman_neb import _settings

ROOT = Path(__file__).resolve().parents[1]


def test_gman_is_not_an_engine_table_entry():
    assert "gman" not in ENGINES


def test_gman_templates_name_the_archive_optimizers():
    steep = load_template(ROOT, "gman/neb-pt-sd")
    quick = load_template(ROOT, "gman/neb-pt-qm")
    assert steep["code"] == "Gman" and steep["code_version"] == "1.0.0.01"
    assert quick["code"] == "Gman" and quick["code_version"] == "1.0.0.01"
    assert steep["commit"] == "r235" and quick["commit"] == "r235"
    assert steep["algorithm"] == "sd"
    assert quick["algorithm"] == "quick-min"
    assert steep["gman"]["optimizer"] == "sd"
    assert quick["gman"]["optimizer"] == "qm"
    assert steep.get("paper_baseline") is not True
    problem = resolve_problem(ROOT, "neb/Pt-heptamer")
    assert _check_stop(problem, steep) == 10000
    assert _check_stop(problem, quick) == 10000
    assert _settings(steep, problem)["itrmax"] == 10000
    assert _settings(steep, problem)["cap"] == 80000
    assert _settings(quick, problem)["cap"] == 80000
    assert _cap_of(problem, steep, {}) == 80000
    assert _cap_of(problem, quick, {}) == 80000
    assert _cap_of(problem, load_template(ROOT, "rgsaddle/neb"), {}) == 10000


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
        "gman/neb-pt-sd",
        output=tmp_path / "benchmark.dat",
        workers=1,
        limit=2,
    )
    assert captured == {"problem": "neb/Pt-heptamer", "template": "gman/neb-pt-sd", "limit": 2}
    assert summary["force_calls"] == 1


def _starts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_stock_steepest_descent_is_ranked_on_its_own_counter():
    dat_path = ROOT / "results" / "campaign" / "gman-r235-neb-pt-sd.dat"
    fields = parse(dat_path.read_text(encoding="utf-8"))
    assert fields["code"] == "Gman"
    assert fields["code_version"] == "1.0.0.01"
    assert fields["commit"] == "r235"
    assert fields["algorithm"] == "sd"
    assert fields["counter"] == "harness"
    assert fields["n"] == "13"
    assert fields["nfailed"] == "13"
    assert "ranked" not in fields
    assert "partial" not in fields
    starts = _starts(dat_path.with_suffix(".starts.jsonl"))
    assert [row["start"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    assert all(row["counter"] == "harness" and row["gman_pot_count"] == 80000 and not row["success"] for row in starts)
    document = json.loads(dat_path.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["provenance"]["code"] == "Gman"
    assert document["provenance"]["code_version"] == "1.0.0.01"
    assert document["provenance"]["commit"] == "r235"
    assert document["provenance"]["parameters"]["counter"] == "pot.count"
    assert document["provenance"]["parameters"]["optimizer"] == "sd"
    assert document["score"]["n"] == 13
    assert document["score"]["nfailed"] == 13
    assert document["score"]["cap"] == 80000
    assert document["score"]["par2_mean"] == 160000


def test_stock_quick_min_matches_the_archived_quick_min_integers():
    dat_path = ROOT / "results" / "campaign" / "gman-r235-neb-pt-qm.dat"
    fields = parse(dat_path.read_text(encoding="utf-8"))
    starts = _starts(dat_path.with_suffix(".starts.jsonl"))
    per_image = [row["gman_pot_count"] / 8 for row in starts]
    assert int(sum(per_image) / len(per_image)) == 190
    assert int(max(per_image)) == 316
    assert int(min(per_image)) == 103
    assert fields["nfailed"] == "0"
    assert fields["code_version"] == "1.0.0.01"
    assert fields["algorithm"] == "quick-min"
    calls = [row["force_calls"] for row in starts]
    document = json.loads(dat_path.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["score"]["nfailed"] == 0
    assert document["score"]["cap"] == 80000
    assert document["score"]["par2_mean"] == statistics.fmean(calls)
    assert document["score"]["par2_mean"] == document["score"]["mean"]
    assert document["provenance"]["parameters"]["counter"] == "pot.count"
    assert document["provenance"]["parameters"]["optimizer"] == "qm"
