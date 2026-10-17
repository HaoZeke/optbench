from pathlib import Path

from optbench.check import load_oracle
from optbench.problems import load_problem
from optbench.site import problem_html, render

ROOT = Path(__file__).resolve().parents[1]


def test_nav_pages_have_titles_and_the_suites_fold_under_archive():
    org = ROOT / "docs" / "orgmode"
    for name in ("index.org", "run.org", "archive.org", "codes.org"):
        text = (org / name).read_text(encoding="utf-8")
        assert "title:nil" not in text
        assert text.startswith("#+TITLE:")
    index = (org / "index.org").read_text(encoding="utf-8")
    assert ":hidden:" in index
    assert "generated/suites/minimization" in index
    assert "generated/suites/neb" in index
    assert "generated/suites/accelerated-md" in index
    archive = (org / "archive.org").read_text(encoding="utf-8")
    assert "generated/suites/minimization" in archive
    assert "generated/suites/accelerated-md" in archive
    conf = (ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8")
    assert '"globaltoc_expand_depth": 0' in conf


def test_collection_shows_present_runs_beyond_the_front_page():
    from optbench.site import campaign_problem, render

    assert campaign_problem("eon-client-morse-lbfgs") == "minimization/morse-bulk"
    assert campaign_problem("eon-3.5.0-neb-pt-lbfgs") == "neb/Pt-heptamer"
    assert campaign_problem("sella-2.6.0-ts-lj38") == "tsopt/LJ38"
    assert campaign_problem("eon-3.5.0-de-lj38-ci-neb") == "double-ended-saddle-search/lj38"
    assert campaign_problem("eon-3.5.0-saddle-pt-dimer") == "saddle-search/pt-island"
    assert campaign_problem("eon-3.5.0-saddle-pt-classic") == "saddle-search/pt-island"
    render(ROOT)
    collection = (ROOT / "docs" / "include" / "collection.org").read_text(encoding="utf-8")
    assert "Morse bulk" in collection
    assert "Pt-heptamer" in collection
    assert ">52<" in collection
    assert "Sella" in collection
    assert "No present run is recorded." in collection
    neb = (ROOT / "docs" / "orgmode" / "generated" / "suites" / "neb.org").read_text(encoding="utf-8")
    assert "Present codes. Fewer calls is to the left." in neb
    assert "eon-3.5.0-neb-pt-lbfgs" in neb
    assert "Legacy 2014 scores. Fewer calls is to the left." in neb
    assert "2014 scores. Fewer calls is to the left." in neb
    featured = (ROOT / "docs" / "include" / "featured.org").read_text(encoding="utf-8")
    assert "rgmin-lbfgs-0.3.0" not in featured


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
    assert "1d71c9445018" in codes
    assert "e089a136910f" in codes
    assert "c0db2b8d333c" in codes


def test_legacy_rows_stay_and_v2_tables_use_recorded_counts():
    render(ROOT)
    featured = (ROOT / "docs" / "include" / "featured.org").read_text(encoding="utf-8")
    assert "Legacy 2014 archive" in featured
    assert "eon-cg" in featured
    assert ">453<" in featured
    assert "par2-wilson-bootstrap" in featured
    sota = (ROOT / "docs" / "include" / "sota.org").read_text(encoding="utf-8")
    assert "rg.terra" in sota
    assert "v2 success rate" in sota
    assert "0.996173" in sota
    assert "0.963355" in sota
    assert "0.884165" in sota
    assert ">181<" in sota
    assert ">4901<" in sota
    assert "default" in sota
    assert "tuned" in sota
    saddle = (ROOT / "docs" / "orgmode" / "generated" / "suites" / "saddle-search.org").read_text(encoding="utf-8")
    assert "Per-search cost" in saddle
    assert "263.284133" in saddle
    assert "469.759056" in saddle
    assert "Legacy 2014 scores. Fewer calls is to the left." in saddle
    assert "v2 smoke" in sota
    assert "139, 139" in sota
    assert "bowl-reported-as-saddle" in sota
    assert ">10000<" in sota
    assert ">139<" in sota
    assert ">720<" in sota
    assert ">400<" in sota
    assert ">2<" in sota
