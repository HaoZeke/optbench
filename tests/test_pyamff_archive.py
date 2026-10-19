"""The 8037eb98 paper rows use the archived log parser and configs."""

import tomllib
from pathlib import Path

from optbench.campaign import list_templates, resolve_problem
from optbench.runners.pyamff_archive import (
    COMMIT,
    archive_aggregates,
    archive_tail,
    parse_training_line,
    with_unique_port,
)

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("mlopt/emt-pt38", "Rprop", 0.0, "mlopt/emt-pt38/pyamff-rprop10-7-fcoeff0"),
    ("mlopt/emt-pt38", "Adam", 0.0, "mlopt/emt-pt38/pyamff-adam10-7-fcoeff0"),
    ("mlopt/emt-pt38", "LBFGS", 0.0, "mlopt/emt-pt38/pyamff-lbfgs10-7-fcoeff0"),
    ("mlopt/emt-pt38", "Rprop", 0.1, "mlopt/emt-pt38/pyamff-rprop10-7-fcoeff0_1"),
    ("mlopt/emt-pt38", "Adam", 0.1, "mlopt/emt-pt38/pyamff-adam10-7-fcoeff0_1"),
    ("mlopt/emt-pt38", "LBFGS", 0.1, "mlopt/emt-pt38/pyamff-lbfgs10-7-fcoeff0_1"),
    ("mlopt/C2H4", "Rprop", 0.0, "mlopt/C2H4/pyamff-rprop10-7-fcoeff0"),
    ("mlopt/C2H4", "Adam", 0.0, "mlopt/C2H4/pyamff-adam10-7-fcoeff0"),
    ("mlopt/C2H4", "LBFGS", 0.0, "mlopt/C2H4/pyamff-lbfgs10-7-fcoeff0"),
    ("mlopt/C2H4", "Rprop", 0.1, "mlopt/C2H4/pyamff-rprop10-7-fcoeff0_1"),
    ("mlopt/C2H4", "Adam", 0.1, "mlopt/C2H4/pyamff-adam10-7-fcoeff0_1"),
    ("mlopt/C2H4", "LBFGS", 0.1, "mlopt/C2H4/pyamff-lbfgs10-7-fcoeff0_1"),
)


def test_tail_matches_the_archive_script():
    text = "header\n    12  1.0  2.0  3.0  0.25  0.5\nMinimizaton converged\n"
    epoch_line, status = archive_tail(text)
    parsed = parse_training_line(epoch_line, status)
    assert parsed["logged_epoch"] == 12
    assert parsed["energy_rmse"] == 0.25
    assert parsed["logged_force_rmse"] == 0.5
    assert parsed["hit_max_epoch"] is False
    failed = "    10000  1  1  1  0.2  0.3\nMax Epoch Reached\n"
    epoch_line, status = archive_tail(failed)
    parsed = parse_training_line(epoch_line, status)
    assert parsed["logged_epoch"] == 10000
    assert parsed["hit_max_epoch"] is True


def test_port_rewrite_leaves_the_archived_lines():
    original = Path(ROOT / "archive/entries" / PAIRS[3][3] / "config.ini").read_text(encoding="utf-8")
    rewritten = with_unique_port(original, 23456)
    assert "master_port = 23456" in rewritten
    assert "optimizer_type = Rprop" in rewritten
    assert "loss_grad_tol = 0.1" in rewritten
    assert "force_coefficient  = 0.1" in rewritten
    assert "epochs_max = 30000" in rewritten
    assert "learning_rate" not in rewritten
    assert "fp_engine" not in rewritten


def test_archive_mean_keeps_a_capped_start():
    rows = [
        {"logged_epoch": 10, "energy_rmse": 0.2, "logged_force_rmse": 0.3, "status": "Minimizaton converged"},
        {"logged_epoch": 30, "energy_rmse": 0.4, "logged_force_rmse": 0.5, "status": "Max Epoch Reached"},
    ]
    summary = archive_aggregates(rows)
    assert summary["archive_mean_epoch"] == 20
    assert abs(summary["archive_mean_energy_rmse"] - 0.3) < 1e-12
    assert summary["archive_nfailed"] == 1


def test_paper_templates_match_the_twelve_archive_entries():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    pinned = next(row for row in registry["code"] if row["id"] == "pyamff-8037eb98")
    assert pinned["commit"] == COMMIT
    found = [row for row in list_templates(ROOT) if row.get("engine") == "pyamff-archive"]
    assert len(found) == 12
    seen = set()
    for template in found:
        assert template["commit"] == COMMIT
        assert template["code_version"] == "8037eb98"
        assert template["tuning"] == "paper"
        assert template["ranked"] is True
        assert template["stop"]["norm"] == "max_abs"
        problem = resolve_problem(ROOT, template["problem"])
        assert problem["sample"]["initializations"] == 50
        config = (ROOT / "archive" / "entries" / template["archive"] / "config.ini").read_text(encoding="utf-8")
        assert f"optimizer_type = {template['pyamff']['optimizer']}" in config
        assert f"epochs_max = {template['stop']['max_steps']}" in config
        seen.add((template["problem"], template["algorithm"], template["pyamff"]["force_coefficient"], template["archive"]))
    assert seen == set(PAIRS)
