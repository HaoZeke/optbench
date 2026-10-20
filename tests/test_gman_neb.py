"""Pt-heptamer QuickMin: Gman 1.0.0.01 is the archive row, TSASE is a separate code."""

import json
from pathlib import Path

from optbench.board import build_board
from optbench.campaign import _check_stop, load_template, resolve_problem, run
from optbench.dat import parse
from optbench.engines import ENGINES

ROOT = Path(__file__).resolve().parents[1]


def test_gman_is_not_an_engine_table_entry():
    assert "gman" not in ENGINES


def test_gman_template_is_stock_quickmin():
    template = load_template(ROOT, "gman/neb-pt-qm")
    assert template["id"] == "gman/neb-pt-qm"
    assert template["engine"] == "gman"
    assert template["problem"] == "neb/Pt-heptamer"
    assert template["code"] == "Gman"
    assert template["code_version"] == "1.0.0.01"
    assert template["commit"] == "r235"
    assert template["algorithm"] == "qm"
    assert template["tuning"] == "default"
    assert not template.get("paper_baseline")
    section = template["gman"]
    assert section["optimizer"] == "qm"
    assert "dt" not in section
    assert "maxmove" not in section
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
        "gman/neb-pt-qm",
        output=tmp_path / "benchmark.dat",
        workers=1,
        limit=2,
    )
    assert captured == {"problem": "neb/Pt-heptamer", "template": "gman/neb-pt-qm", "limit": 2}
    assert summary["force_calls"] == 1


def test_recorded_gman_quickmin_is_the_band_total():
    dat = ROOT / "results" / "campaign" / "gman-1.0.0.01-neb-pt-qm.dat"
    fields = parse(dat.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "1527"
    assert fields["force_calls_max"] == "2528"
    assert fields["force_calls_min"] == "824"
    assert fields["force_calls_per_image"] == "190"
    assert fields["force_calls_per_image_max"] == "316"
    assert fields["force_calls_per_image_min"] == "103"
    assert fields["call_unit"] == "total"
    assert fields["nfailed"] == "0"
    assert fields["n"] == "13"
    assert fields["counter"] == "harness"
    assert fields["code_version"] == "1.0.0.01"
    assert fields["commit"] == "r235"
    assert fields["template"] == "gman/neb-pt-qm"
    assert "ranked" not in fields
    assert "partial" not in fields
    assert not (ROOT / "results" / "campaign" / "gman-r235-neb-pt-qm.dat").exists()
    assert not (ROOT / "results" / "campaign" / "gman-r235-neb-pt-qm.starts.jsonl").exists()
    starts = [
        json.loads(line)
        for line in (ROOT / "results" / "campaign" / "gman-1.0.0.01-neb-pt-qm.starts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["start"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    assert all(row["counter"] == "harness" and row["success"] and row["call_unit"] == "total" for row in starts)
    assert all(row["force_calls"] == row["raw"] and row["raw"] % 8 == 0 for row in starts)


def test_tsase_template_is_stock_quickmin():
    template = load_template(ROOT, "tsase/neb-pt-qm")
    assert template["code"] == "tsase"
    assert template["code_version"] == "1.0"
    assert template["commit"] == "98aac60b5d9905e81ee4e6b8c385f169258d0218"
    assert template["algorithm"] == "qm"
    assert template["tuning"] == "default"
    assert not template.get("paper_baseline")
    section = template["tsase"]
    assert section["method"] == "qm"
    assert section["band"] == "ci"
    assert section["spring_k"] == 5.0
    assert section["ss"] is False
    assert "dt" not in section
    assert "maxmove" not in section


def test_recorded_tsase_quickmin_uses_the_band_total():
    dat = ROOT / "results" / "campaign" / "tsase-1.0-neb-pt-qm.dat"
    fields = parse(dat.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "3651"
    assert fields["force_calls_max"] == "5618"
    assert fields["force_calls_min"] == "1690"
    assert fields["force_calls_per_image"] == "456"
    assert fields["call_unit"] == "total"
    assert fields["nfailed"] == "0"
    assert fields["n"] == "13"
    assert fields["counter"] == "harness"
    assert fields["code_version"] == "1.0"
    assert "ranked" not in fields
    assert "partial" not in fields
    starts = [
        json.loads(line)
        for line in (ROOT / "results" / "campaign" / "tsase-1.0-neb-pt-qm.starts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["start"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    assert all(row["call_unit"] == "total" and row["counter"] == "harness" and row["success"] for row in starts)
    assert all(row["force_calls"] == row["raw"] for row in starts)
    assert "count_key" not in starts[0]


def test_quickmin_rows_share_one_ranked_unit():
    document = build_board(ROOT)
    neb = document["problems"]["neb/Pt-heptamer"]
    assert not any("gman-r235-neb-pt-qm" in str(item.get("path") or "") for item in document["unparsed"])
    ranked = {row["template"]: row for row in neb["ranked"]}
    gman = ranked["gman/neb-pt-qm"]
    tsase = ranked["tsase/neb-pt-qm"]
    for row, per_image in ((gman, 190.92307692307693), (tsase, 456.3269230769231)):
        assert row["n"] == 13
        assert row["nfailed"] == 0
        assert row["counter"] == "harness"
        assert row["partial"] is False
        assert row["par2_mean"] == row["mean_calls"]
        assert row["mean_per_image"] == per_image
        assert abs(row["par2_mean"] / 8 - row["mean_per_image"]) < 1e-12
    assert gman["path"] == "results/campaign/gman-1.0.0.01-neb-pt-qm.dat"
    assert tsase["path"] == "results/campaign/tsase-1.0-neb-pt-qm.dat"
    assert not any(item.get("template") in {"gman/neb-pt-qm", "tsase/neb-pt-qm"} for item in neb["partial"])
