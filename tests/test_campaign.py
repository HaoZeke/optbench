import stat
from pathlib import Path

import pytest

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.dat import parse

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_templates_use_the_lj38_stop():
    ids = []
    for template in list_templates(ROOT):
        ids.append(template["id"])
        problem_name = template.get("problem", "minimization/lj38")
        _check_stop(resolve_problem(ROOT, problem_name), template)
    assert "eon-client/lbfgs-m100" in ids
    assert "eon-svn/lbfgs-m100" in ids
    assert "eon/lj38-lbfgs" in ids
    assert "eon/lj38-lbfgs-m100" in ids
    assert "eon/neb-pt-lbfgs" in ids
    assert "eon/de-lj38-lbfgs-m100" in ids
    assert "eon/ts-pt-dimer" in ids
    assert "eon/develop-ts-lj38-lanczos-improved" in ids
    assert "eon/develop-de-lj38-lanczos-improved" in ids
    assert "eon/develop-low-lj38-lanczos" in ids
    assert "eon/develop-saddle-pt-lanczos" in ids
    assert "eon/develop-de-2ts-lanczos-improved" in ids
    assert "eon/develop-de-pt-ci-single" in ids
    assert "eon/develop-neb-pt-cg" in ids


def test_develop_caps_match_the_problem_cap():
    from optbench.campaign import load_template

    names = (
        "eon/develop-de-pt-ci-single",
        "eon/develop-de-pt-ci",
        "eon/develop-de-pt-lanczos-improved",
        "eon/develop-de-lj38-lanczos-improved",
        "eon/develop-de-2ts-lanczos-improved",
        "eon/develop-ts-lj38-lanczos-improved",
        "eon/develop-ts-pt-lanczos-improved",
        "eon/develop-neb-pt-lbfgs",
        "eon/develop-neb-pt-cg",
        "eon/develop-low-lj38-lanczos",
        "eon/develop-low-lj38-dimer",
    )
    for name in names:
        template = load_template(ROOT, name)
        problem = resolve_problem(ROOT, template["problem"])
        cap = int(problem["convergence"]["max_force_calls"])
        assert template["failure_cap"] == cap
        assert template["code"] == "eOn"
        assert template["code_version"] == "eOn (develop 500350d9)"
        assert "tuning" not in template
        assert "lbfgs_accept" not in template["ini"].get("LBFGS", {})
        assert "lbfgs_accept" not in template["ini"].get("Optimizer", {})
        assert "LBFGS" not in template["ini"]


def test_develop_rows_use_the_paper_method_and_kernel():
    from optbench.campaign import load_template

    lanczos = (
        "eon/develop-de-pt-lanczos-improved",
        "eon/develop-de-lj38-lanczos-improved",
        "eon/develop-de-2ts-lanczos-improved",
        "eon/develop-ts-lj38-lanczos-improved",
        "eon/develop-ts-pt-lanczos-improved",
    )
    for name in lanczos:
        template = load_template(ROOT, name)
        saddle = template["ini"]["Saddle Search"]
        assert template["ini"]["Main"]["job"] == "saddle_search"
        assert saddle["min_mode_method"] == "lanczos"
        assert saddle["client_displace_type"] == "load"
        assert template["ini"]["Optimizer"]["opt_method"] == "lbfgs"
        assert template["algorithm"] == "Lanczos improved L-BFGS"
    assert load_template(ROOT, "eon/develop-de-pt-ci-single")["ini"]["Nudged Elastic Band"]["images"] == 1
    assert load_template(ROOT, "eon/develop-de-pt-ci")["ini"]["Nudged Elastic Band"]["images"] == 3
    assert load_template(ROOT, "eon/develop-neb-pt-lbfgs")["ini"]["Nudged Elastic Band"]["images"] == 8
    assert "opt_method" not in load_template(ROOT, "eon/develop-neb-pt-lbfgs")["ini"]["Nudged Elastic Band"]
    cg = load_template(ROOT, "eon/develop-neb-pt-cg")
    assert cg["ini"]["Nudged Elastic Band"]["opt_method"] == "cg"
    assert cg["ini"]["Nudged Elastic Band"]["images"] == 8
    assert load_template(ROOT, "eon/develop-de-pt-ci-single")["kernel"] == "morse_shift"
    assert load_template(ROOT, "eon/develop-de-pt-lanczos-improved")["kernel"] == "morse_shift"
    assert load_template(ROOT, "eon/develop-neb-pt-lbfgs")["kernel"] == "morse_shift"
    assert load_template(ROOT, "eon/develop-ts-pt-lanczos-improved")["kernel"] == "morse"
    assert load_template(ROOT, "eon/develop-ts-lj38-lanczos-improved")["kernel"] == "lj40"
    assert load_template(ROOT, "eon/develop-de-lj38-lanczos-improved")["kernel"] == "lj40"
    assert load_template(ROOT, "eon/develop-de-2ts-lanczos-improved")["kernel"] == "lj40"
    assert load_template(ROOT, "eon/develop-low-lj38-lanczos")["kernel"] == "lj40"
    assert load_template(ROOT, "eon/develop-low-lj38-dimer")["kernel"] == "lj40"


def test_run_writes_one_benchmark_dat(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    for name in ("cluster_0000.con", "cluster_0001.con"):
        (geometries / name).write_text("pos\n", encoding="utf-8")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "text = Path('config.ini').read_text()\n"
        "calls = 181 if 'lbfgs_memory = 100' in text else 449\n"
        "Path('results.dat').write_text("
        "'0 termination_reason\\n%f total_force_calls\\n' % calls)\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon-svn/lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
    )
    text = output.read_text(encoding="utf-8")
    assert text == (
        "force_calls 181\n"
        "force_calls_min 181\n"
        "force_calls_max 181\n"
        "nfailed 0\n"
        "code Eon svn\n"
        "code_version r2549\n"
        "algorithm LBFGS\n"
        "template eon-svn/lbfgs-m100\n"
    )
    fields = parse(text)
    assert fields["force_calls"] == "181"
    assert fields["nfailed"] == "0"
    assert fields["template"] == "eon-svn/lbfgs-m100"
    assert summary["starts"] == 2
    import json

    starts = [
        json.loads(line)
        for line in (tmp_path / "benchmark.starts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["start"] for row in starts] == ["cluster_0000", "cluster_0001"]
    assert {row["force_calls"] for row in starts} == {181}
    assert {row["counter"] for row in starts} == {"client"}
    assert {row["comparable"] for row in starts} == {False}
    assert "potential = lj_cluster" in (tmp_path / "work" / "cluster_0000" / "config.ini").read_text()


def test_modern_eon_template_records_commit_and_each_start(tmp_path: Path):
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    (geometries / "cluster_0000.con").write_text("pos\n", encoding="utf-8")
    (geometries / "cluster_0001.con").write_text("pos\n", encoding="utf-8")
    client = tmp_path / "client"
    client.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "text = Path('config.ini').read_text()\n"
        "assert '[LBFGS]' in text\n"
        "assert 'lbfgs_accept = energy' in text\n"
        "calls = 181 if 'lbfgs_memory = 100' in text else 201\n"
        "Path('results.dat').write_text("
        "'0 termination_reason\\n%f total_force_calls\\n' % calls)\n",
        encoding="utf-8",
    )
    client.chmod(client.stat().st_mode | stat.S_IEXEC)
    output = tmp_path / "benchmark.dat"
    summary = run(
        ROOT,
        "minimization/lj38",
        "eon/lj38-lbfgs-m100",
        client=client,
        geometries=geometries,
        output=output,
        work=tmp_path / "work",
    )
    fields = parse(output.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "181"
    assert fields["commit"] == "1d71c944501832bfafb955ffe13b9aae50c8efd4"
    assert fields["problem"] == "minimization/lj38"
    assert fields["template"] == "eon/lj38-lbfgs-m100"
    assert summary["starts"] == 2
    starts = (tmp_path / "benchmark.starts").read_text(encoding="utf-8").splitlines()
    assert len(starts) == 2
    assert '"failed": false' in starts[0]
    assert "cluster_0000.con" in starts[0]


def test_a_short_iteration_cap_is_refused():
    problem = resolve_problem(ROOT, "minimization/lj38")
    template = {"ini": {"Optimizer": {"max_iterations": 10, "converged_force": 0.01}}}
    with pytest.raises(ValueError):
        _check_stop(problem, template)
