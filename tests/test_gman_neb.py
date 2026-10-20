"""Gman r235 quick-min is the ranked Pt-heptamer row."""

import json
from pathlib import Path

from optbench.campaign import _check_stop, load_template, resolve_problem, run
from optbench.dat import parse
from optbench.engines import ENGINES

ROOT = Path(__file__).resolve().parents[1]


def test_gman_is_not_an_engine_table_entry():
    assert "gman" not in ENGINES


def test_gman_template_is_the_stock_release():
    template = load_template(ROOT, "gman/neb-pt-qm")
    assert template["id"] == "gman/neb-pt-qm"
    assert template["engine"] == "gman"
    assert template["problem"] == "neb/Pt-heptamer"
    assert template["code"] == "Gman"
    assert template["code_version"] == "r235"
    assert template["algorithm"] == "quick-min"
    assert template.get("paper_baseline") is not True
    assert template["tuning"] == "default"
    section = template["gman"]
    assert section["optimizer"] == "qm"
    assert section["potential"] == "morse"
    problem = resolve_problem(ROOT, "neb/Pt-heptamer")
    assert _check_stop(problem, template) == 10000
    assert not (ROOT / "templates" / "gman" / "neb-pt-sd.toml").is_file()


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


def test_latest_gman_ranks_its_own_counter():
    import statistics

    dat_path = ROOT / "results" / "campaign" / "gman-r235-neb-pt-qm.dat"
    fields = parse(dat_path.read_text(encoding="utf-8"))
    assert fields["counter"] == "harness"
    assert fields["code_version"] == "r235"
    assert fields["algorithm"] == "quick-min"
    assert fields["n"] == "13"
    assert fields["nfailed"] == "0"
    assert fields["tuning"] == "default"
    assert "ranked" not in fields
    assert "partial" not in fields
    starts = [
        json.loads(line)
        for line in dat_path.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(starts) == 13
    assert [row["start"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    calls = [row["force_calls"] for row in starts]
    assert all(row["counter"] == "harness" and row["gman_pot_count"] == row["force_calls"] and row["success"] for row in starts)
    assert fields["force_calls"] == str(int(round(statistics.fmean(calls))))
    document = json.loads(dat_path.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["counter"] == "harness"
    assert document["provenance"]["parameters"]["counter"] == "gman pot.count"
    assert document["provenance"]["code_version"] == "r235"
    assert document["score"]["n"] == 13
    assert document["score"]["nfailed"] == 0
    assert document["score"]["par2_mean"] == statistics.fmean(calls)
