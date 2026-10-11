import stat
import unittest
from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem, run
from optbench.dat import parse

ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def test_shipped_templates_use_the_lj38_stop(self):
        problem = resolve_problem(ROOT, "minimization/lj38")
        ids = []
        for template in list_templates(ROOT):
            ids.append(template["id"])
            _check_stop(problem, template)
        self.assertIn("eon-client/lbfgs-m100", ids)
        self.assertIn("eon-svn/lbfgs-m100", ids)

    def test_run_writes_one_benchmark_dat(self):
        scratch = ROOT / "build" / "campaign-test"
        if scratch.exists():
            import shutil

            shutil.rmtree(scratch)
        geometries = scratch / "geometries"
        geometries.mkdir(parents=True)
        for name in ("cluster_0000.con", "cluster_0001.con"):
            (geometries / name).write_text("pos\n", encoding="utf-8")
        client = scratch / "client"
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
        output = scratch / "benchmark.dat"
        summary = run(
            ROOT,
            "minimization/lj38",
            "eon-svn/lbfgs-m100",
            client=client,
            geometries=geometries,
            output=output,
            work=scratch / "work",
        )
        fields = parse(output.read_text(encoding="utf-8"))
        self.assertEqual(fields["force_calls"], "181")
        self.assertEqual(fields["nfailed"], "0")
        self.assertEqual(fields["template"], "eon-svn/lbfgs-m100")
        self.assertEqual(summary["starts"], 2)
        self.assertIn("potential = lj_cluster", (scratch / "work" / "cluster_0000" / "config.ini").read_text())

    def test_a_short_iteration_cap_is_refused(self):
        problem = resolve_problem(ROOT, "minimization/lj38")
        template = {
            "ini": {"Optimizer": {"max_iterations": 10, "converged_force": 0.01}}
        }
        with self.assertRaises(ValueError):
            _check_stop(problem, template)


if __name__ == "__main__":
    unittest.main()
