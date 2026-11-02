"""Sphinx configuration for the OptBench site."""

from pathlib import Path

project = "OptBench"
author = "OptBench"
copyright = "2026, Rohit Goswami"
html_title = "OptBench"
# Sphinx version and release are the project version.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-version
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-release
# Generated from pyproject.toml by cog (https://nedbatchelder.com/code/cog/).
# [[[cog
# from optbench.project_version import cog_out
# cog_out(cog, 'version = "{version}"')
# cog_out(cog, 'release = "{version}"')
# ]]]
version = "0.0.0"
release = "0.0.0"
# [[[end]]]

extensions = []
exclude_patterns = ["_build"]

html_theme = "shibuya"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_css_files = ["optbench.css"]
html_js_files = ["bench.js"]
html_copy_source = False
html_permalinks = False
html_theme_options = {
    "accent_color": "bronze",
    "github_url": "https://github.com/lode-org/optbench",
    "dark_code": True,
    "globaltoc_expand_depth": 0,
    "page_layout": "default",
    # The copy-page menu adds ChatGPT and Claude links, and the theme
    # default mounts a Read the Docs ad slot. Neither is part of the rank.
    "show_ai_links": False,
    "open_in_chatgpt": False,
    "open_in_claude": False,
    "ethical_ads_publisher": "",
    "nav_links": [
        {
            "title": "Suites",
            "url": "index",
            "children": [
                {"title": "Minimization", "url": "generated/suites/minimization"},
                {"title": "Saddle search", "url": "generated/suites/saddle-search"},
                {"title": "Double-ended saddle search", "url": "generated/suites/double-ended-saddle-search"},
                {"title": "Global optimization", "url": "generated/suites/globalopt"},
                {"title": "Lowest eigenvector", "url": "generated/suites/lowest-eigenvector"},
                {"title": "Transition state optimization", "url": "generated/suites/tsopt"},
                {"title": "Machine learning optimization", "url": "generated/suites/mlopt"},
                {"title": "Nudged elastic band", "url": "generated/suites/neb"},
                {"title": "Harmonic transition state theory", "url": "generated/suites/hTST"},
                {"title": "Accelerated molecular dynamics", "url": "generated/suites/accelerated-md"},
            ],
        },
        {"title": "Board", "url": "board"},
        {"title": "Methodology", "url": "methodology"},
        {"title": "Codes", "url": "codes"},
    ],
}
html_baseurl = "https://lode-org.github.io/optbench/"


def _drop_remote_assets(app, exception):
    """The theme ships a flag font and an ad loader that point at CDNs."""
    if exception is not None:
        return
    out = Path(app.outdir)
    css = out / "_static" / "shibuya.css"
    if css.is_file():
        text = css.read_text(encoding="utf-8")
        start = text.find("@font-face{font-family:Twemoji Country Flags;")
        end = text.find("}", start)
        if start != -1 and end != -1 and "jsdelivr" in text[start:end]:
            text = text[:start] + text[end + 1 :]
            css.write_text(text, encoding="utf-8")
    js = out / "_static" / "shibuya.js"
    if js.is_file():
        text = js.read_text(encoding="utf-8")
        text = text.replace("function Tn(t,e){", "function Tn(t,e){return;")
        text = text.replace("//cdn.carbonads.com/carbon.js?serve=${t}&placement=${e}", "")
        js.write_text(text, encoding="utf-8")


def setup(app):
    app.connect("build-finished", _drop_remote_assets)
