import pytest

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


def test_sample_keeps_internal_spaces():
    fields = parse(SAMPLE)
    assert fields["force_calls"] == "453"
    assert fields["algorithm"] == "Conjugate Gradients"
    assert fields["contributor"] == "Sam Chill"


def test_duplicate_key_is_an_error():
    with pytest.raises(DatError):
        parse("force_calls 1\nforce_calls 2\n")


def test_spelling_aliases():
    fields, renamed = normalize({"contributer": "Sam Chill", "uncertanity": "1"})
    assert fields["contributor"] == "Sam Chill"
    assert fields["uncertainty"] == "1"
    assert renamed == [("contributer", "contributor"), ("uncertanity", "uncertainty")]
