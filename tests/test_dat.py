import unittest

from optbench.dat import DatError, normalize, parse


SAMPLE = """\
force_calls 453
force_calls_max 1153
force_calls_min 207
nfailed 0
algorithm Conjugate Gradients
code Eon
contributor Sam Chill
"""


class ParseTests(unittest.TestCase):
    def test_sample_keeps_internal_spaces(self):
        fields = parse(SAMPLE)
        self.assertEqual(fields["force_calls"], "453")
        self.assertEqual(fields["algorithm"], "Conjugate Gradients")
        self.assertEqual(fields["contributor"], "Sam Chill")

    def test_duplicate_key_is_an_error(self):
        with self.assertRaises(DatError):
            parse("force_calls 1\nforce_calls 2\n")

    def test_spelling_aliases(self):
        fields, renamed = normalize({"contributer": "Sam Chill", "uncertanity": "1"})
        self.assertEqual(fields["contributor"], "Sam Chill")
        self.assertEqual(fields["uncertainty"], "1")
        self.assertEqual(renamed, [("contributer", "contributor"), ("uncertanity", "uncertainty")])


if __name__ == "__main__":
    unittest.main()
