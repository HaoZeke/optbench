"""Archive numbers stay the archived rows. A later file is a finding, not a gate."""

import json
import subprocess
from pathlib import Path

import pytest

from optbench.check import banned_word_errors, held_back_errors, load_oracle
from optbench.findings import (
    campaign_mismatch,
    citation_errors,
    citation_reachability_errors,
    findings_errors,
    load_ledger,
    protected_refs,
    read_campaign,
    require_commit_sha,
    row_id,
)
from optbench.problems import load_problem
from optbench.site import problem_html

ROOT = Path(__file__).resolve().parents[1]


def test_findings_follow_the_oracle():
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    ledger = load_ledger(ROOT / "oracle" / "findings.jsonl")
    assert len(ledger) == len(oracle) == 151
    assert [row_id(row) for row in ledger] == [row_id(row) for row in oracle]
    assert findings_errors(ROOT) == []
    assert not (ROOT / "oracle" / "reproduction.jsonl").exists()
    assert not (ROOT / "docs" / "reproduction.md").exists()
    beside = [row for row in ledger if row.get("beside")]
    assert len(beside) == 86
    text = (ROOT / "docs" / "findings.md").read_text(encoding="utf-8")
    assert "There is no pass/fail window." in text


def test_pele_archive_stop_stays_beside_the_harness_stop():
    ledger = {row_id(row): row for row in load_ledger(ROOT / "oracle" / "findings.jsonl")}
    forward = ledger["tsopt/LJ38/pele_ffd"]
    central = ledger["tsopt/LJ38/pele_cd"]
    island = ledger["tsopt/pt-island/pele"]
    assert forward["beside"]["stop2014_calls"] == 207.51
    assert forward["beside"]["harness_stop_mean"] == 254.48
    assert central["beside"]["stop2014_calls"] == 263.52
    assert central["beside"]["harness_stop_mean"] == 335.06
    assert island["beside"]["stop2014_calls"] == 88.551
    assert "starting set" in forward["finding"]
    assert held_back_errors(ROOT) == []


def test_a_list_that_keeps_templates_off_the_table_is_rejected(tmp_path: Path):
    optbench = tmp_path / "optbench"
    optbench.mkdir()
    (optbench / "board.py").write_text("def holdback(root):\n    return None\n", encoding="utf-8")
    (optbench / "findings.py").write_text('ROWS = [{"hold": ["eon/neb-pt-lbfgs"]}]\n', encoding="utf-8")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "findings.jsonl").write_text(
        json.dumps({"beside": {"omitted": ["eon-client/neb-pt-lbfgs"]}}) + "\n",
        encoding="utf-8",
    )
    errors = held_back_errors(tmp_path)
    assert any(line.startswith("optbench/board.py") for line in errors)
    assert any(line.startswith("optbench/findings.py") for line in errors)
    assert any("oracle/findings.jsonl" in line for line in errors)


SADDLE_SHA = "02405b0d55809cb81921e75f9ef29dc85a13f4bf"
SADDLE_PATHS = {
    "results/campaign/optim-df8da599-rr-loweig.dat",
    "results/campaign/optim-df8da599-hef-ffd-ts-lj38.dat",
    "results/campaign/optim-df8da599-hef-ffd-ts-pt.dat",
    "results/campaign/optim-svn-snapshot-double-ended-saddle-search-lj38--optim-de-lj38.dat",
    "results/campaign/optim-svn-snapshot-double-ended-saddle-search-lj38_2ts_separation--optim-de-2ts.dat",
    "results/campaign/optim-svn-snapshot-double-ended-saddle-search-pt-island--optim-de-pt.dat",
}


def test_a_branch_only_citation_is_rejected():
    with pytest.raises(ValueError, match="commit sha"):
        require_commit_sha("cells/optim/saddle-pt")
    row = {
        "suite": "tsopt",
        "problem": "LJ38",
        "entry": "optim_ffd",
        "beside": {
            "branch": "cells/optim/saddle-pt",
            "path": "results/campaign/optim-df8da599-hef-ffd-ts-lj38.dat",
        },
    }
    errors = citation_errors([row])
    assert errors
    assert "branch name" in errors[0]


def test_saddle_files_stay_on_their_commit():
    ledger = load_ledger(ROOT / "oracle" / "findings.jsonl")
    seen = set()
    for row in ledger:
        beside = row.get("beside") or {}
        if not beside:
            continue
        assert "branch" not in beside
        assert require_commit_sha(beside["sha"]) == beside["sha"]
        if beside["path"] in SADDLE_PATHS:
            assert beside["sha"] == SADDLE_SHA
            seen.add(beside["path"])
    assert seen == SADDLE_PATHS


SOFTSADDLE_SHA = "bd1a5c9fc2685f5876266d53715accb31a049e73"
SOFTSADDLE_HARNESS = {
    "adaptive_HSphere-lbfgs-davidson-sm50": {
        "force_calls": 5298701.7,
        "unique_saddles": 0.0,
        "fraction_connected": 1.0,
    },
    "adaptive_HSphere-lbfgs-lanczos-sm0": {
        "force_calls": 1926653.9,
        "unique_saddles": 0.5,
        "fraction_connected": 0.9682000000000001,
    },
    "adaptive_HSphere-lbfgs-lanczos-sm50": {
        "force_calls": 1582635.3,
        "unique_saddles": 0.7,
        "fraction_connected": 0.93584,
    },
}


def test_softsaddle_rows_cite_the_ten_rep_file():
    ledger = load_ledger(ROOT / "oracle" / "findings.jsonl")
    seen = set()
    for row in ledger:
        if row.get("suite") != "saddle-search" or row.get("entry") not in SOFTSADDLE_HARNESS:
            continue
        beside = row["beside"]
        assert beside["sha"] == SOFTSADDLE_SHA
        assert "branch" not in beside
        expected = SOFTSADDLE_HARNESS[row["entry"]]
        harness = {item["metric"]: item["harness"] for item in row["values"]}
        assert harness == expected
        if row["entry"] == "adaptive_HSphere-lbfgs-davidson-sm50":
            assert "The scored Davidson sm50 row is bd1a5c9f" in row["finding"]
            assert "247b70de is not a scored row" in row["finding"]
            assert "1054 to 1060" in row["finding"]
            assert "The Davidson library requests these gradients itself" in row["finding"]
            assert "not an extra" not in row["finding"]
            assert "1 + 1050 + 8 + 1 = 1060" in row["finding"]
            assert "stays 0 converged" in row["finding"]
        seen.add(row["entry"])
    assert seen == set(SOFTSADDLE_HARNESS)


def test_a_cited_sha_outside_main_names_the_row(monkeypatch):
    from optbench import findings

    main_sha = "62c0d93f650b2bcf17480297c33c6a1fcc981c45"
    rows = [
        {
            "suite": "saddle-search",
            "problem": "pt-island",
            "entry": "adaptive_HSphere-lbfgs-davidson-sm50",
            "beside": {"sha": SOFTSADDLE_SHA},
        },
        {
            "suite": "globalopt",
            "problem": "lj38",
            "entry": "gmin-nosym",
            "beside": {"sha": main_sha},
        },
    ]
    monkeypatch.setattr(findings, "load_ledger", lambda path: rows)
    monkeypatch.setattr(findings, "_main_ref", lambda root: "main")
    monkeypatch.setattr(findings, "protected_refs", lambda root: [])

    def ancestor(root, sha, ref):
        return sha == main_sha and ref == "main"

    monkeypatch.setattr(findings, "_is_ancestor", ancestor)
    errors = citation_reachability_errors(ROOT)
    row = "saddle-search/pt-island/adaptive_HSphere-lbfgs-davidson-sm50"
    assert errors == [
        f"{row} cites {SOFTSADDLE_SHA} outside main and the protected refs"
    ]


def test_an_allowlisted_ref_keeps_the_citation(monkeypatch):
    from optbench import findings

    tag = "cite/bd1a5c9fc268"
    rows = [
        {
            "suite": "saddle-search",
            "problem": "pt-island",
            "entry": "adaptive_HSphere-lbfgs-davidson-sm50",
            "beside": {"sha": SOFTSADDLE_SHA, "tag": tag},
        }
    ]
    monkeypatch.setattr(findings, "load_ledger", lambda path: rows)
    monkeypatch.setattr(findings, "_main_ref", lambda root: "origin/main")
    monkeypatch.setattr(findings, "protected_refs", lambda root: [tag])
    monkeypatch.setattr(findings, "_is_ancestor", lambda root, sha, ref: False)
    monkeypatch.setattr(findings, "_peel_commit", lambda root, ref: SOFTSADDLE_SHA)
    assert citation_reachability_errors(ROOT) == []


def test_a_parent_of_a_protected_tag_is_rejected(monkeypatch):
    from optbench import findings

    tag = "cite/bd1a5c9fc268"
    parent = "a" * 40
    rows = [
        {
            "suite": "saddle-search",
            "problem": "pt-island",
            "entry": "adaptive_HSphere-lbfgs-davidson-sm50",
            "beside": {"sha": parent, "tag": tag},
        }
    ]
    monkeypatch.setattr(findings, "load_ledger", lambda path: rows)
    monkeypatch.setattr(findings, "_main_ref", lambda root: "origin/main")
    monkeypatch.setattr(findings, "protected_refs", lambda root: [tag])
    monkeypatch.setattr(findings, "_is_ancestor", lambda root, sha, ref: False)
    monkeypatch.setattr(findings, "_peel_commit", lambda root, ref: SOFTSADDLE_SHA)
    errors = citation_reachability_errors(ROOT)
    assert errors == [
        f"{rows[0]['suite']}/{rows[0]['problem']}/{rows[0]['entry']} "
        f"cites tag {tag}, which points at {SOFTSADDLE_SHA} rather than {parent}"
    ]


def test_a_missing_protected_tag_is_an_error(monkeypatch):
    from optbench import findings

    tag = "cite/bd1a5c9fc268"
    monkeypatch.setattr(findings, "load_ledger", lambda path: [])
    monkeypatch.setattr(findings, "protected_refs", lambda root: [tag])
    monkeypatch.setattr(findings, "_peel_commit", lambda root, ref: "")
    assert citation_reachability_errors(ROOT) == [f"protected ref {tag} is missing"]


def test_main_reaches_the_gmin_commit_and_not_the_softsaddle_commit():
    from optbench.findings import _is_ancestor, _main_ref

    ref = _main_ref(ROOT)
    assert ref
    assert _is_ancestor(ROOT, "62c0d93f650b2bcf17480297c33c6a1fcc981c45", ref)
    assert not _is_ancestor(ROOT, SOFTSADDLE_SHA, ref)


CITE_TAGS = {
    "cite/bd1a5c9fc268": "bd1a5c9fc2685f5876266d53715accb31a049e73",
    "cite/02405b0d5580": "02405b0d55809cb81921e75f9ef29dc85a13f4bf",
    "cite/0528a37b8ff6": "0528a37b8ff6e54805b9fa75345ee3c40b756a5d",
    "cite/715bc521c827": "715bc521c827f1f18102c1c027d99d0d8624b0fb",
    "cite/158d27812f76": "158d27812f76160469f6134a14eda16e4ab151e1",
    "cite/f3cf41d74f73": "f3cf41d74f73fb2ffb3b5199291c90a4743a1000",
    "cite/54ac7e7a1aa6": "54ac7e7a1aa6581c261c17fc2c52a20e77121911",
}


def test_the_cite_tags_name_the_off_main_commits():
    from optbench.findings import _peel_commit

    listed = [f"refs/tags/{name}" for name in CITE_TAGS]
    assert protected_refs(ROOT) == listed
    for ref, sha in CITE_TAGS.items():
        assert _peel_commit(ROOT, ref) == sha
        assert _peel_commit(ROOT, f"refs/tags/{ref}") == sha
    by_sha = {sha: ref for ref, sha in CITE_TAGS.items()}
    tagged = 0
    for row in load_ledger(ROOT / "oracle" / "findings.jsonl"):
        beside = row.get("beside") or {}
        sha = beside.get("sha")
        if sha not in by_sha:
            assert "tag" not in beside
            continue
        assert beside.get("tag") == by_sha[sha]
        tagged += 1
    assert tagged == 19
    assert citation_reachability_errors(ROOT) == []


def test_protected_refs_skip_comments(tmp_path: Path):
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "protected-refs.txt").write_text(
        "# comment\n\nrefs/tags/keep-me\n",
        encoding="utf-8",
    )
    assert protected_refs(tmp_path) == ["refs/tags/keep-me"]


def test_the_check_command_names_an_unprotected_sha(monkeypatch, capsys):
    from optbench.__main__ import check as cli_check

    monkeypatch.setattr("optbench.check.check", lambda problems, oracle: 0)
    row = "saddle-search/pt-island/adaptive_HSphere-lbfgs-davidson-sm50"
    message = f"{row} cites {SOFTSADDLE_SHA} outside main and the protected refs"
    monkeypatch.setattr(
        "optbench.findings.citation_reachability_errors",
        lambda root: [message],
    )
    with pytest.raises(SystemExit) as caught:
        cli_check(problems=ROOT / "problems", oracle=ROOT / "oracle" / "entries.jsonl")
    assert caught.value.code == 1
    assert message in capsys.readouterr().out


def test_the_check_command_passes_when_citations_are_reachable(monkeypatch):
    from optbench.__main__ import check as cli_check

    monkeypatch.setattr("optbench.check.check", lambda problems, oracle: 0)
    monkeypatch.setattr(
        "optbench.findings.citation_reachability_errors",
        lambda root: [],
    )
    with pytest.raises(SystemExit) as caught:
        cli_check(problems=ROOT / "problems", oracle=ROOT / "oracle" / "entries.jsonl")
    assert caught.value.code == 0


def test_a_missing_campaign_commit_names_the_row_and_the_sha(monkeypatch):
    from optbench import findings

    real = findings.read_campaign

    def read_local(root, citation, path):
        if str(path).endswith("saddle-davidson-sm50.dat"):
            raise findings.MissingCampaign(citation)
        return real(root, citation, path)

    monkeypatch.setattr(findings, "read_campaign", read_local)
    errors = findings.findings_errors(ROOT)
    row = "saddle-search/pt-island/adaptive_HSphere-lbfgs-davidson-sm50"
    assert errors == [f"{row} cites {SOFTSADDLE_SHA}, which is missing"]


def test_a_cited_file_that_disagrees_is_an_error(monkeypatch):
    from optbench import findings

    real = findings.read_campaign

    def read_wrong(root, citation, path):
        if str(path).endswith("saddle-davidson-sm50.dat"):
            return (
                "force_calls 1\n"
                "unique_saddles 1\n"
                "fraction_connected 1\n"
                "benchmark_unique_saddles 1\n"
                "benchmark_fraction_connected 1\n"
                "nfailed 0\n"
            )
        return real(root, citation, path)

    monkeypatch.setattr(findings, "read_campaign", read_wrong)
    errors = findings.findings_errors(ROOT)
    assert any("does not match" in line for line in errors)


def test_a_missing_pin_is_fetched(monkeypatch):
    calls = []

    def fake_check_output(args, **kwargs):
        calls.append(list(args))
        if args[1] == "cat-file":
            raise subprocess.CalledProcessError(1, args)
        if args[1] == "fetch":
            return b""
        if args[1] == "show":
            return "force_calls 1\n"
        raise AssertionError(args)

    monkeypatch.setattr("optbench.findings.subprocess.check_output", fake_check_output)
    text = read_campaign(ROOT, SADDLE_SHA, "results/campaign/x.dat")
    assert text == "force_calls 1\n"
    assert ["git", "fetch", "origin", SADDLE_SHA] in calls


def test_a_campaign_file_that_disagrees_is_reported(tmp_path: Path):
    row = {
        "suite": "minimization",
        "problem": "lj38",
        "entry": "eon-lbfgs",
        "beside": {
            "branch": "main",
            "path": "results/campaign/bad.dat",
            "kind": "dat",
        },
        "values": [{"metric": "force_calls", "archive": 181, "harness": 181}],
    }
    campaign = tmp_path / "results" / "campaign"
    campaign.mkdir(parents=True)
    (campaign / "bad.dat").write_text("force_calls 1000\nnfailed 0\n", encoding="utf-8")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "findings.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    errors = campaign_mismatch(tmp_path)
    assert errors
    assert any("force_calls" in line and "1000" in line for line in errors)


def test_banned_words_are_rejected(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("This is a smoke run.\n", encoding="utf-8")
    errors = banned_word_errors(tmp_path)
    assert any("smoke" in line for line in errors)
    (docs / "note.md").write_text("The archived rows stay in the table.\n", encoding="utf-8")
    assert banned_word_errors(tmp_path) == []
    (tmp_path / "repro-notes").mkdir()
    assert any("repro" in line for line in banned_word_errors(tmp_path))


def test_archive_rows_stay_on_the_archive_table():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    html = problem_html(problem, oracle)
    assert ">eon-cg<" in html
    assert "not reproduced" not in html
    assert "2014 archive" in html
