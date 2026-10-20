"""Stock TSASE 1.0 stays ranked beside the archived LJ38 dimer row."""

from pathlib import Path

from optbench.reproduction import holdback, load_ledger, row_id

ROOT = Path(__file__).resolve().parents[1]
STOCK = ROOT / "results" / "campaign" / "tsase-1.0-lowest-eigenvector-lj38--tsase-mode-lj38.dat"
DIAGNOSTIC = ROOT / "results" / "diagnostic" / "tsase-408-lj38-dimer-bfgs.dat"


def _fields(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, value = line.split(None, 1)
        fields[key] = value
    return fields


def test_stock_tsase_stays_ranked_beside_the_archive():
    ledger = {row_id(row): row for row in load_ledger(ROOT / "oracle" / "reproduction.jsonl")}
    row = ledger["lowest-eigenvector/lj38/tsase-dimer-bfgs"]
    assert row["status"] == "not_rerun"
    assert row["rerun"] is None
    assert row["checks"] == []
    assert row["in_paper"] is True
    assert row["archived_setup"]["code_version"] == "408"
    note = row["note"]
    assert "98aac60b5d9905e81ee4e6b8c385f169258d0218" in note
    assert "248" in note
    assert "nfailed 22" in note
    stock = _fields(STOCK)
    assert stock["code"] == "tsase"
    assert stock["code_version"] == "1.0"
    assert stock["commit"] == "98aac60b5d9905e81ee4e6b8c385f169258d0218"
    assert stock["template"] == "tsase/mode-lj38"
    assert stock["problem"] == "lowest-eigenvector/lj38"
    assert stock["tuning"] == "default"
    assert stock["counter"] == "harness"
    assert stock["force_calls"] == "248"
    assert stock["force_calls_min"] == "25"
    assert stock["force_calls_max"] == "2014"
    assert stock["nfailed"] == "22"
    assert stock["n"] == "200"
    assert not (ROOT / "results" / "campaign" / "tsase-408-lj38-dimer-bfgs.dat").exists()
    diagnostic = _fields(DIAGNOSTIC)
    assert diagnostic["code_version"] == "408"
    assert diagnostic.get("ranked", "").lower() != "false"
    held = holdback(ROOT)
    assert ("lowest-eigenvector/lj38", "tsase/mode-lj38") not in held.templates
    assert STOCK.relative_to(ROOT).as_posix() not in held.paths
