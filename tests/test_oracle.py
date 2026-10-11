import json
import unittest
from pathlib import Path

from optbench.check import check

ROOT = Path(__file__).resolve().parents[1]


class OracleTests(unittest.TestCase):
    def test_archived_entries_match_the_contracts(self):
        self.assertEqual(check(ROOT / "problems", ROOT / "oracle" / "entries.jsonl"), 0)

    def test_main_oracle_has_every_archived_score(self):
        rows = [
            json.loads(line)
            for line in (ROOT / "oracle" / "entries.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 151)
        self.assertTrue(all(row["commit"] == "0f26b796" for row in rows))

    def test_stated_inputs_are_in_the_geometry_store(self):
        manifest = {
            json.loads(line)["path"]
            for line in (ROOT / "inputs" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        }
        self.assertIn("website/minimization/lj38/lj38-clusters.tgz", manifest)
        self.assertIn("website/saddle-search/pt-island/pt-island.con", manifest)
        self.assertIn("website/saddle-search/pt-island/POSCAR_pt_island", manifest)


if __name__ == "__main__":
    unittest.main()
