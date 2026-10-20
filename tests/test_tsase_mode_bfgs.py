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
    assert "dimer_rotate.py" in note
    assert "phi_tol 0" in note
    assert "314159" in note
    assert "2fa92dc" in note
    assert "e43971c" in note
    assert "SSDimer.search" not in note
    assert "28" in note
    assert "median 26" in note
    assert "successes 200" in note
    assert "nfailed 0" in note
    assert "27.895" in note
    fields = _fields(CAMPAIGN)
    assert fields["code"] == "TSASE"
    assert fields["code_version"] == "1.0 (98aac60)"
    assert fields["commit"] == "98aac60b5d9905e81ee4e6b8c385f169258d0218"
    assert fields["template"] == "tsase/mode-lj38-bfgs"
    assert fields["algorithm"] == "SSDimer BFGS"
    assert fields["problem"] == "lowest-eigenvector/lj38"
    assert fields["tuning"] == "default"
    assert fields["counter"] == "harness"
    assert fields["force_calls"] == "28"
    assert fields["force_calls_min"] == "15"
    assert fields["force_calls_max"] == "61"
    assert fields["force_calls_median"] == "26"
    assert fields["successes"] == "200"
    assert fields["nfailed"] == "0"
    assert fields["n"] == "200"
    assert fields.get("ranked", "").lower() != "false"
    assert "Archive dimer_rotate.py" in CAMPAIGN.read_text(encoding="utf-8")
    assert "phi_tol is 0" in CAMPAIGN.read_text(encoding="utf-8")
    assert "seed 314159" in CAMPAIGN.read_text(encoding="utf-8")
    assert "SSDimer.search" not in CAMPAIGN.read_text(encoding="utf-8")
    document = json.loads(CAMPAIGN.with_suffix(".v2.json").read_text(encoding="utf-8"))
    assert document["provenance"]["code"] == "TSASE"
    assert document["provenance"]["code_version"] == "1.0 (98aac60)"
    assert document["provenance"]["parameters"]["driver"] == "dimer_rotate"
    assert document["provenance"]["parameters"]["rotation_opt"] == "bfgs"
    assert document["provenance"]["parameters"]["max_step"] == 0.2
    assert document["provenance"]["parameters"]["dr"] == 0.001
    assert document["provenance"]["parameters"]["phi_tol"] == 0.0
    assert document["provenance"]["parameters"]["rotation_limit"] == 1000
    assert document["provenance"]["parameters"]["mode_seed"] == 314159
    assert document["provenance"]["parameters"]["seed"] == "314159"
    assert "rotation_max" not in document["provenance"]["parameters"]
    assert document["score"]["cap"] == 10000
    assert document["score"]["n_cap"] == 0
    assert document["score"]["n_divergence"] == 0
    assert document["score"]["n_crash"] == 0
    assert document["score"]["n_wrong_stationary"] == 0
    assert document["score"]["par2_mean"] == 27.895
    assert document["score"]["successes"] == 200
    starts = [
        json.loads(line)
        for line in CAMPAIGN.with_suffix(".starts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(starts) == 200
    assert all(item["success"] for item in starts)
    assert all(item["failure"] is None for item in starts)
    assert all(item["seed"] == "314159" for item in starts)
    assert min(item["force_calls"] for item in starts) == 15
    assert max(item["force_calls"] for item in starts) == 61
    assert all(item["overlap"] >= 0.99 for item in starts)
    assert not (ROOT / "results" / "diagnostic" / "tsase-408-lj38-dimer-bfgs.dat").exists()
    assert not (ROOT / "results" / "campaign" / "tsase-408-lj38-dimer-bfgs.dat").exists()
    assert not (ROOT / "results" / "campaign" / "tsase-1.0-lowest-eigenvector-lj38--tsase-mode-lj38-bfgs.dat").exists()
    held = holdback(ROOT)
    assert ("lowest-eigenvector/lj38", "tsase/mode-lj38-bfgs") not in held.templates
    assert CAMPAIGN.relative_to(ROOT).as_posix() not in held.paths
