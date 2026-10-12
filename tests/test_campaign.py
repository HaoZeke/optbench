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
    fields = parse(output.read_text(encoding="utf-8"))
    assert fields["force_calls"] == "181"
    assert fields["nfailed"] == "0"
    assert fields["template"] == "eon-svn/lbfgs-m100"
    assert summary["starts"] == 2
    assert "potential = lj_cluster" in (tmp_path / "work" / "cluster_0000" / "config.ini").read_text()


def test_a_short_iteration_cap_is_refused():
    problem = resolve_problem(ROOT, "minimization/lj38")
    template = {"ini": {"Optimizer": {"max_iterations": 10, "converged_force": 0.01}}}
    with pytest.raises(ValueError):
        _check_stop(problem, template)
