"""Budgets for a built OptBench page."""

from optbench.site_weight import (
    DOM_BUDGET,
    EAGER_IMAGE_BUDGET,
    HTML_BUDGET,
    check_site_weight,
    measure_pages,
)


def test_lazy_thumbnails_are_not_eager_and_webp_is(tmp_path):
    (tmp_path / "heavy.svg").write_bytes(b"v" * 400_000)
    (tmp_path / "thumb.webp").write_bytes(b"RIFF" + b"w" * 120)
    (tmp_path / "fallback.png").write_bytes(b"p" * 80)
    (tmp_path / "icon.svg").write_bytes(b"<svg/>")
    (tmp_path / "a.css").write_text(
        "body{background:url(icon.svg)} @font-face{src:url(font.woff2)}",
        encoding="utf-8",
    )
    html = """<!DOCTYPE html>
<html><head>
<link rel="icon" href="icon.svg">
<link rel="stylesheet" href="a.css">
</head><body>
<img src="heavy.svg" alt="" loading="lazy">
<picture>
  <source srcset="thumb.webp" type="image/webp">
  <img src="fallback.png" alt="plot">
</picture>
<p>ok</p>
</body></html>
"""
    (tmp_path / "index.html").write_text(html, encoding="utf-8")
    row = measure_pages(tmp_path)[0]
    icon = (tmp_path / "icon.svg").stat().st_size
    thumb = (tmp_path / "thumb.webp").stat().st_size
    assert row["eager"] == icon + thumb
    assert row["missing"] == []
    assert row["dom"] < DOM_BUDGET
    assert row["html"] < HTML_BUDGET
    assert row["eager"] < EAGER_IMAGE_BUDGET
    assert check_site_weight(tmp_path) == 0


def test_a_page_over_the_dom_budget_fails(tmp_path):
    rows = "".join("<tr><td>1</td></tr>" for _ in range(DOM_BUDGET))
    (tmp_path / "big.html").write_text(f"<html><body><table>{rows}</table></body></html>", encoding="utf-8")
    assert check_site_weight(tmp_path) == 1


def test_a_missing_build_fails(tmp_path):
    assert check_site_weight(tmp_path / "missing") == 2
