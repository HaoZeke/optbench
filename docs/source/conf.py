"""Sphinx configuration for the OptBench site."""

project = "OptBench"
author = "OptBench"
copyright = "2014--present, OptBench contributors"
html_title = "OptBench"

extensions = []
exclude_patterns = ["_build"]

html_theme = "shibuya"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_css_files = ["optbench.css"]
html_js_files = ["bench.js"]
html_theme_options = {
    "accent_color": "blue",
    "github_url": "https://github.com/lode-org/optbench",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
}
html_baseurl = "https://lode-org.github.io/optbench/"
