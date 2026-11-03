"""Layout constants shared by the site charts and the paper figures."""

import importlib.util
import math
import re
import textwrap
from pathlib import Path

import pytest

from optbench.site_charts import PALETTE_SENTENCE, PALETTE_SIZE, PROFILE_MARGIN, PROFILE_TOP

ROOT = Path(__file__).resolve().parents[1]
NUMBER_WORDS = "zero one two three four five six seven eight nine".split()
LONG_LABEL = "a recorded method label that is much wider than the room beside one journal column"


@pytest.fixture(autouse=True)
def _keep_matplotlib_state(monkeypatch):
    # apply_style sets global rcParams and replaces the SVG marker writer, and
    # later suites render the board plots into the tree.
    import matplotlib
    from matplotlib.backends.backend_svg import RendererSVG

    monkeypatch.setattr(RendererSVG, "draw_markers", RendererSVG.draw_markers)
    monkeypatch.setattr(RendererSVG, "_optbench_compact", False, raising=False)
    with matplotlib.rc_context():
        yield


def _paper_common():
    path = ROOT / "paper-figures" / "common.py"
    spec = importlib.util.spec_from_file_location("paper_common_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_style()
    return module


def test_the_profile_margin_is_the_matplotlib_default():
    import matplotlib

    assert PROFILE_MARGIN == matplotlib.rcParamsDefault["axes.ymargin"]
    assert PROFILE_TOP == 1.0 + PROFILE_MARGIN


def test_the_palette_matches_the_stylesheet_and_the_caption():
    css = (ROOT / "docs" / "source" / "_static" / "optbench.css").read_text(encoding="utf-8")
    for selector in (r"\.plate-chart \.s", r"\.swatch\.s"):
        found = {int(number) for number in re.findall(selector + r"(\d+)\b", css)}
        assert found == set(range(PALETTE_SIZE))
    assert NUMBER_WORDS[PALETTE_SIZE] in PALETTE_SENTENCE.split()


def test_a_long_label_wraps_at_the_widest_width_that_fits():
    common = _paper_common()
    wrapped, inches, _line_in = common.fit_labels([LONG_LABEL, "short"])
    assert "\n" in wrapped[0]
    assert wrapped[1] == "short"
    assert inches <= common.LABEL_ROOM_IN
    longest = max(len(line) for line in wrapped[0].splitlines())
    for width in range(longest + 1, len(LONG_LABEL)):
        wider = textwrap.fill(LONG_LABEL, width=width, break_long_words=False, break_on_hyphens=False)
        if wider != wrapped[0]:
            assert common.fit_labels([wider], max_in=math.inf)[1] > common.LABEL_ROOM_IN
            break


def test_a_row_holds_its_label_lines_and_one_empty_line():
    import matplotlib.pyplot as plt

    common = _paper_common()
    labels = [f"method {index}" for index in range(12)]
    _wrapped, _inches, line_in = common.fit_labels(labels)
    assert line_in > 0
    fig, _ax, _labels = common.categorical_figure(len(labels), labels)
    assert fig.get_size_inches()[1] == pytest.approx(max(2.2, 0.62 + 2 * len(labels) * line_in))
    plt.close(fig)
    fig, _ax, _labels = common.categorical_figure(len(labels), labels, min_row_in=4 * line_in)
    assert fig.get_size_inches()[1] == pytest.approx(0.62 + 4 * len(labels) * line_in)
    plt.close(fig)


def test_caption_lines_break_only_where_the_next_word_would_not_fit():
    pymupdf = pytest.importorskip("pymupdf")
    common = _paper_common()
    caption = " ".join([LONG_LABEL] * 6) + " Dolan, E. D. and Moré, J. J., Math. Program. 91:201-213, 2002."
    width = 72 * common.COLUMN_IN - 2

    def length(text):
        return pymupdf.get_text_length(text + " ", fontname=common.CAPTION_FONT, fontsize=common.CAPTION_PT)

    lines = common.caption_lines(caption, width)
    assert " ".join(lines) == caption
    assert all(length(line) <= width for line in lines)
    assert all(length(f"{line} {after.split()[0]}") > width for line, after in zip(lines, lines[1:]))
    page = pymupdf.open().new_page(width=width + 2, height=200)
    box = pymupdf.Rect(1, 0, width + 1, common.CAPTION_PT * 2 * len(lines))
    assert page.insert_textbox(box, "\n".join(lines), fontsize=common.CAPTION_PT, fontname=common.CAPTION_FONT) >= 0
