"""Stock TSASE BFGS dimer stays ranked beside the archived LJ38 row."""

import json
from pathlib import Path

from optbench.reproduction import holdback, load_ledger, row_id

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "campaign" / "tsase-98aac60b-lowest-eigenvector-lj38--tsase-mode-lj38-bfgs.dat"


def _fields(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        fields[key] = value
    return fields


def test_stock_bfgs_dimer_stays_ranked_beside_the_archive():
    ledger = {row_id(row): row for row in load_ledger(ROOT / "oracle" / "reproduction.jsonl")}
    row = ledger["lowest-eigenvector/lj38/tsase-dimer-bfgs"]
    assert row["status"] == "not_rerun"
    assert row["rerun"] is None
    assert row["checks"] == []
    assert row["in_paper"] is True
    assert row["archived_setup"]["code_version"] == "408"
    note = row["note"]
    assert "TSASE 1.0 (98aac60)" in note
    assert "TSASE 98aac60b" not in note
    assert "98aac60b5d9905e81ee4e6b8c385f169258d0218" in note
    assert "498dfc6b" not in note
    assert "tsase/mode-lj38-bfgs" in note
    assert "rotationOpt bfgs" in note
    assert "rotationMax is the 10000-call cap" in note
    assert "one get_forces runs the rotation" in note
    assert "SSDimer.search" not in note
    assert "19" in note
    assert "median 19.5" in note
    assert "successes 60" in note
    assert "nfailed 140" in note
    assert "14005.805" in note
    assert "nfailed 73" not in note
    fields = _fields(CAMPAIGN)
    assert fields["code"] == "TSASE"
    assert fields["code_version"] == "1.0 (98aac60)"
    assert fields["commit"] == "98aac60b5d9905e81ee4e6b8c385f169258d0218"
    assert fields["template"] == "tsase/mode-lj38-bfgs"
    assert fields["algorithm"] == "SSDimer BFGS"
    assert fields["problem"] == "lowest-eigenvector/lj38"
    assert fields["tuning"] == "default"
    assert fields["counter"] == "harness"
    assert fields["force_calls"] == "19"
    assert fields["force_calls_min"] == "11"
    assert fields["force_calls_max"] == "30"
    assert fields["force_calls_median"] == "19.5"
    assert fields["successes"] == "60"
    assert fields["nfailed"] == "140"
    assert fields["n"] == "200"
    assert fields.get("ranked", "").lower() != "false"
    assert "rotationOpt is bfgs" in CAMPAIGN.read_text(encoding="utf-8")
    assert "rotationMax is the 10000-call cap" in CAMPAIGN.read_text(encoding="utf-8")
    assert "One get_forces runs the rotation" in CAMPAIGN.read_text(encoding="utf-8")
    assert "SSDimer.search" not in CAMPAIGN.read_text(encoding="utf-8")
    document = json.loads(CAMPAIGN.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["provenance"]["code"] == "TSASE"
    assert document["provenance"]["code_version"] == "1.0 (98aac60)"
    assert document["provenance"]["parameters"]["rotation_opt"] == "bfgs"
    assert document["provenance"]["parameters"]["max_step"] == 0.2
    assert document["provenance"]["parameters"]["dr"] == 0.001
    assert document["provenance"]["parameters"]["phi_tol"] == 5.0
    assert document["provenance"]["parameters"]["rotation_max"] == 10000
    assert document["score"]["cap"] == 10000
    assert document["score"]["n_cap"] == 0
    assert document["score"]["n_divergence"] == 140
    assert document["score"]["n_crash"] == 0
    assert document["score"]["n_wrong_stationary"] == 0
    assert document["score"]["par2_mean"] == 14005.805
    assert document["score"]["successes"] == 60
    starts = [
        json.loads(line)
        for line in CAMPAIGN.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(starts) == 200
    failed = [item for item in starts if not item["success"]]
    assert len(failed) == 140
    assert {item["failure"] for item in failed} == {"divergence"}
    assert all(item["reasons"] == ["mode_overlap"] for item in failed)
    assert all(item["force_calls"] < 10000 for item in failed)
    assert all(item["stop_log"] for item in failed)
    assert all(sum(line.startswith("alpha=") for line in item["stop_log"]) == 1 for item in failed)
    assert all(any(line.startswith("curvature:") for line in item["stop_log"]) for item in failed)
    assert not (ROOT / "results" / "diagnostic" / "tsase-408-lj38-dimer-bfgs.dat").exists()
    assert not (ROOT / "results" / "campaign" / "tsase-408-lj38-dimer-bfgs.dat").exists()
    assert not (ROOT / "results" / "campaign" / "tsase-1.0-lowest-eigenvector-lj38--tsase-mode-lj38-bfgs.dat").exists()
    held = holdback(ROOT)
    assert ("lowest-eigenvector/lj38", "tsase/mode-lj38-bfgs") not in held.templates
    assert CAMPAIGN.relative_to(ROOT).as_posix() not in held.paths
