"""Gman r235 steepest-descent CI-NEB stays off the engine table."""

import json
from pathlib import Path

from optbench.campaign import _check_stop, load_template, resolve_problem, run
from optbench.dat import parse
from optbench.engines import ENGINES

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
    assert template["paper_baseline"] is True
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


def test_recorded_gman_heptamer_matches_the_archive_integers():
    import statistics

    from optbench.score import normalize_start

    dat_path = ROOT / "results" / "campaign" / "gman-r235-neb-pt-sd.dat"
    text = dat_path.read_text(encoding="utf-8")
    fields = parse(text)
    assert "Gman's pot.count()/8" in text
    assert fields["force_calls"] == "412"
    assert fields["force_calls_max"] == "697"
    assert fields["force_calls_min"] == "214"
    assert fields["nfailed"] == "0"
    assert fields["counter"] == "harness"
    assert fields["algorithm"] == "sd"
    assert fields["template"] == "gman/neb-pt-sd"
    assert fields["tuning"] == "archive"
    starts = [
        json.loads(line)
        for line in dat_path.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["start"] for row in starts] == [f"min{index:02d}" for index in range(1, 14)]
    per_image = [row["force_calls"] for row in starts]
    assert int("%d" % (sum(per_image) / len(per_image))) == 412
    assert max(per_image) == 697
    assert min(per_image) == 214
    assert all(
        row["counter"] == "harness"
        and row["count_key"] == "force_calls_neb"
        and row["gman_pot_count"] == row["raw"]
        and row["force_calls"] == row["raw"] // 8
        and row["success"]
        and row["failure"] is None
        for row in starts
    )
    ranked = [normalize_start(row, 8) for row in starts]
    assert [row["force_calls"] for row in ranked] == [row["raw"] for row in starts]
    assert statistics.fmean(row["per_image"] for row in ranked) == statistics.fmean(per_image)
    document = json.loads(dat_path.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["counter"] == "harness"
    assert document["provenance"]["parameters"]["counter"] == "gman pot.count()/8"
    assert document["legacy"]["force_calls"] == 412
    assert document["score"]["par2_mean"] == statistics.fmean(per_image)
    assert document["score"]["n"] == 13
    assert document["score"]["nfailed"] == 0
