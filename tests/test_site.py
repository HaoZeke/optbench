from pathlib import Path

from optbench.check import load_oracle
from optbench.problems import load_problem
from optbench.site import problem_html, render

ROOT = Path(__file__).resolve().parents[1]


def test_lj38_table_ranks_the_archived_scores():
    problem = load_problem(ROOT / "problems" / "minimization" / "lj38.toml")
    oracle = load_oracle(ROOT / "oracle" / "entries.jsonl")
    html = problem_html(problem, oracle)
    assert "eon-cg" in html
    assert ">453<" in html
    assert "held out of the original table" in html
    assert "0.01" in html
    assert "bench-plot-data" in html
    assert '"scale": "log"' in html


def test_render_writes_the_front_page_table():
    render(ROOT)
    featured = (ROOT / "docs" / "include" / "featured.org").read_text(encoding="utf-8")
    assert "Lennard-Jones 38 clusters" in featured
    assert "eon-cg" in featured
    assert "rgmin-lbfgs-0.3.0" not in featured
    sota = (ROOT / "docs" / "include" / "sota.org").read_text(encoding="utf-8")
    assert "rgmin-lbfgs-0.3.0" in sota
    assert "ase-fire-3.29.0" in sota
    assert "eon-3.5.0-fire" in sota
    assert "bench-plot-mount" in sota
    assert "bench-plot-data" in sota
    assert ">731<" in sota
    assert "eon-3.5.0-lbfgs-m100" in sota
    assert "eon-svn-r2549-lbfgs-m100" in sota
    assert ">181<" in sota
    assert ">4901<" in sota
    assert "eon-cg-3.5.0" not in sota
    recorded = (ROOT / "docs" / "include" / "terra-pass.org").read_text(encoding="utf-8")
    assert "rgmin-lbfgs-0.3.0" in recorded
    assert ">216<" in recorded
    assert ">454<" in recorded
    assert ">706<" in recorded
    codes = (ROOT / "docs" / "include" / "codes.org").read_text(encoding="utf-8")
    assert "Rohit Goswami" in codes
    assert "4d8b34baf1ba" in codes
    assert "e089a136910f" in codes
    assert "8bea1002b06a" in codes
