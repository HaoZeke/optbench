"""Gman r235 global L-BFGS with the Hessian step, library defaults."""

import json
from pathlib import Path

from optbench.campaign import _check_stop, load_template, resolve_problem
from optbench.engines import ENGINES
from optbench.reproduction import within_mean

ROOT = Path(__file__).resolve().parents[1]
DAT = ROOT / "results" / "campaign" / "gman-r235-neb-pt-glbfgs-hess.dat"


def test_gman_is_not_an_engine_table_entry():
    assert "gman" not in ENGINES


def test_template_uses_the_hessian_step_and_library_defaults():
    template = load_template(ROOT, "gman/neb-pt-glbfgs-hess")
    assert template["code"] == "Gman"
    assert template["code_version"] == "r235"
    assert template["commit"] == "r235"
    assert template["algorithm"] == "glbfgs-hess"
    assert template["tuning"] == "default"
    assert "paper_baseline" not in template
    section = template["gman"]
    assert section == {"optimizer": "glbfgs", "minimizer": "hess", "potential": "morse"}
    problem = resolve_problem(ROOT, "neb/Pt-heptamer")
    assert _check_stop(problem, template) == 10000


def test_recorded_integer_is_the_archive_truncation():
    text = DAT.read_text(encoding="utf-8")
    assert "force_calls 45\n" in text
    assert "force_calls_min 24\n" in text
    assert "force_calls_max 62\n" in text
    assert "nfailed 0\n" in text
    assert "code Gman\n" in text
    assert "code_version r235\n" in text
    assert "counter harness\n" in text
    assert "ranked false" not in text
    assert "partial true" not in text
    starts = [
        json.loads(line)
        for line in DAT.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(starts) == 13
    per_image = [row["gman_pot_count"] / 8 for row in starts]
    assert int(sum(per_image) / len(per_image)) == 45
    assert int(min(per_image)) == 24
    assert int(max(per_image)) == 62
    assert all(
        row["success"]
        and row["counter"] == "harness"
        and row["code"] == "Gman"
        and row["force_calls"] == row["gman_pot_count"]
        and row["failure"] is None
        for row in starts
    )
    assert within_mean(48.0, 45.0, None)
