import subprocess
import unittest

from optbench.score import half_even, summarize_capped


class ScoreTests(unittest.TestCase):
    def test_cap_splits_failures_from_the_mean(self):
        summary = summarize_capped([207, 1153, 10000], cap=10000)
        self.assertEqual(summary["force_calls"], 680)
        self.assertEqual(summary["force_calls_min"], 207)
        self.assertEqual(summary["force_calls_max"], 1153)
        self.assertEqual(summary["nfailed"], 1)

    def test_half_even_matches_awk_printf(self):
        for value in (0.5, 1.5, 2.5, -1.5):
            awk = subprocess.check_output(
                ["awk", "-v", f"x={value}", 'BEGIN{printf "%.0f", x}'],
                text=True,
            )
            self.assertEqual(str(half_even(value)), awk.strip())


if __name__ == "__main__":
    unittest.main()
