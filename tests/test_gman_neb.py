"""Gman r235 CI-NEB stays off the engine table."""

import json
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, load_template, resolve_problem, run
from optbench.dat import parse
from optbench.engines import ENGINES
from optbench.score import normalize_start, score_v2

ROOT = Path(__file__).resolve().parents[1]


def test_gman_is_not_an_engine_table_entry():
    assert "gman" not in ENGINES


def test_gman_template_matches_the_archived_sd_setup():
    template = load_template(ROOT, "gman/neb-pt-sd")
    assert template["id"] == "gman/neb-pt-sd"
    assert template["engine"] == "gman"
    assert template["problem"] == "neb/Pt-heptamer"
    assert template["code"] == "Gman"
    assert template["code_version"] == "r235"
    assert template["algorithm"] == "sd"
    assert template["paper_baseline"] is False
    section = template["gman"]
    assert section["optimizer"] == "sd"
    assert section["step_r"] == 0.02
    assert section["maxmove"] == 0.2
    assert section["images"] == 8
    assert section["spring"] == 5.0
    assert section["tangent"] == "new"
    assert section["itrmax"] == 10000
    assert section["ppd"] == 1
    assert section["fmax"] == 0.01
    assert section["fsaddle"] == "off"
    assert section["method"] == "ci"
    assert section["converge"] == "Fmag"
    assert section["potential"] == "morse"
    problem = resolve_problem(ROOT, "neb/Pt-heptamer")
    assert _check_stop(problem, template) == 10000


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


def test_gman_template_matches_the_archived_cg_setup():
    template = load_template(ROOT, "gman/neb-pt-cg")
    assert template["id"] == "gman/neb-pt-cg"
    assert template["engine"] == "gman"
    assert template["problem"] == "neb/Pt-heptamer"
    assert template["code"] == "Gman"
    assert template["code_version"] == "r235"
    assert template["algorithm"] == "cg"
    assert template["paper_baseline"] is False
    section = template["gman"]
    assert section["optimizer"] == "cg"
    assert section["d_r"] == 0.0001
    assert section["maxmove"] == 0.2
    assert "step_r" not in section
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


def test_gman_cg_dispatch_uses_the_shared_runner(monkeypatch, tmp_path: Path):
    captured = {}

    def fake(root, problem, template, *, output, work, limit):
        captured["template"] = template["id"]
        captured["limit"] = limit
        output.write_text("force_calls 1\n", encoding="utf-8")
        return {"force_calls": 1, "nfailed": 0, "starts": 1, "template": template["id"], "problem": problem["id"]}

    monkeypatch.setattr("optbench.runners.gman_neb.run_template", fake)
    run(
        ROOT,
        "neb/Pt-heptamer",
        "gman/neb-pt-cg",
        output=tmp_path / "benchmark.dat",
        workers=1,
        limit=1,
    )
    assert captured == {"template": "gman/neb-pt-cg", "limit": 1}


def test_recorded_gman_heptamer_matches_the_archive_integers():
    dat = ROOT / "results" / "campaign" / "gman-r235-neb-pt-sd.dat"
    if not dat.is_file():
        pytest.skip("the steepest-descent campaign is recorded on its own branch")
    fields = parse(dat.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "412"
    assert fields["force_calls_max"] == "697"
    assert fields["force_calls_min"] == "214"
    assert fields["nfailed"] == "0"
    assert fields["counter"] == "client"
    assert fields["algorithm"] == "sd"
    assert fields["template"] == "gman/neb-pt-sd"
    starts = [
        json.loads(line)
        for line in (ROOT / "results" / "campaign" / "gman-r235-neb-pt-sd.starts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["geometry"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    per_image = [float(row["raw"]) / row["images"] for row in starts]
    assert int("%d" % (sum(per_image) / len(per_image))) == 412
    assert int("%d" % max(per_image)) == 697
    assert int("%d" % min(per_image)) == 214
    assert all(row["force_calls"] == row["raw"] for row in starts)
    assert all(row["counter"] == "client" and row["call_unit"] == "total" and row["success"] for row in starts)


def test_archived_gman_cg_rerun_stays_an_unranked_diagnostic():
    dat_path = ROOT / "results" / "campaign" / "gman-r235-neb-pt-cg.dat"
    fields = parse(dat_path.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "120"
    assert fields["force_calls_max"] == "321"
    assert fields["force_calls_min"] == "57"
    assert fields["nfailed"] == "0"
    assert "counter" not in fields
    assert not dat_path.with_suffix(".starts.jsonl").is_file()
    assert not dat_path.with_suffix(".v2.json").is_file()


_STOCK_PER_PATH = [45, 41, 95, 95, 105, 200, 163, 177, 175, 95, 93, 109, 123]


def test_stock_gman_cg_ranks_the_native_counter():
    template = load_template(ROOT, "gman/neb-pt-cg-stock")
    assert template["tuning"] == "stock"
    assert template["code_version"] == "r235"
    assert template["gman"]["method"] == "off"
    assert template["gman"]["itrmax"] == 100
    assert template["gman"]["fmax"] == 0.01
    dat_path = ROOT / "results" / "campaign" / "gman-r235-neb-pt-cg-stock.dat"
    fields = parse(dat_path.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "116"
    assert fields["force_calls_max"] == "200"
    assert fields["force_calls_min"] == "41"
    assert fields["nfailed"] == "1"
    assert fields["counter"] == "harness"
    assert fields["algorithm"] == "cg"
    assert fields["code_version"] == "r235"
    assert fields["template"] == "gman/neb-pt-cg-stock"
    assert fields["tuning"] == "stock"
    starts = [
        json.loads(line)
        for line in dat_path.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["per_image"] for row in starts] == _STOCK_PER_PATH
    assert [row["force_calls"] for row in starts] == [count * 8 for count in _STOCK_PER_PATH]
    assert sum(1 for row in starts if row["success"]) == 12
    assert starts[5]["success"] is False and starts[5]["failure"] == "divergence"
    assert all(row["counter"] == "harness" and row["call_unit"] == "total" and row["pot_count"] == row["raw"] for row in starts)
    normalized = [normalize_start(row, 8) for row in starts]
    cap = int((resolve_problem(ROOT, "neb/Pt-heptamer").get("convergence") or {})["max_force_calls"])
    score = score_v2(
        [{"force_calls": row["force_calls"], "success": raw["success"], "failure": raw["failure"]} for raw, row in zip(starts, normalized)],
        cap,
    )
    document = json.loads(dat_path.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["schema"] == "optbench.score/v2"
    assert document["counter"] == "harness"
    assert document["legacy"]["force_calls"] == 116
    assert document["score"]["par2_mean"] == score["par2_mean"]
    assert document["score"]["n"] == 13
    assert document["score"]["nfailed"] == 1
    assert document["provenance"]["parameters"]["counter"] == "pot.count"
    assert document["provenance"]["code_version"] == "r235"
