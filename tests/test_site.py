import unittest
from pathlib import Path

from optbench.check import load_oracle
from optbench.problems import load_problem
from optbench.site import problem_html, render

ROOT = Path(__file__).resolve().parents[1]


class SiteTests(unittest.TestCase):
    def test_lj38_table_ranks_the_archived_scores(self):
        problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
        oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
        html = problem_html(problem, oracle)
        self.assertIn("eon-cg", html)
        self.assertIn(">453<", html)
        self.assertIn("held out of the original table", html)
        self.assertIn("0.01", html)
        self.assertIn("bench-plot-data", html)
        self.assertIn('"scale": "log"', html)

    def test_render_writes_the_front_page_table(self):
        render(ROOT)
        featured = (ROOT / "docs" / "include" / "featured.org").read_text(encoding="utf-8")
        self.assertIn("Lennard-Jones 38 clusters", featured)
        self.assertIn("eon-cg", featured)
        self.assertNotIn("rgmin-lbfgs-0.3.0", featured)
        sota = (ROOT / "docs" / "include" / "sota.org").read_text(encoding="utf-8")
        self.assertIn("rgmin-lbfgs-0.3.0", sota)
        self.assertIn("ase-fire-3.29.0", sota)
        self.assertNotIn("eon-cg-3.5.0", sota)
        terra = (ROOT / "docs" / "include" / "terra-pass.org").read_text(encoding="utf-8")
        self.assertIn("rgmin-lbfgs-0.3.0", terra)
        self.assertIn(">216<", terra)
        self.assertIn(">454<", terra)
        self.assertIn(">706<", terra)
        codes = (ROOT / "docs" / "include" / "codes.org").read_text(encoding="utf-8")
        self.assertIn("Rohit Goswami", codes)
        self.assertIn("97c9992f65cc", codes)
        self.assertIn("e089a136910f", codes)
        self.assertIn("8bea1002b06a", codes)
        self.assertIn("9b753517c6c0", codes)
        self.assertIn("3.29.0", codes)


if __name__ == "__main__":
    unittest.main()
