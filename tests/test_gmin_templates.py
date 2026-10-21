"""GMIN globalopt templates use the registry pin and the harness cap."""

from __future__ import annotations

import tomllib
from pathlib import Path

from optbench.campaign import _check_stop, list_templates, resolve_problem
from optbench.runners.gmin_go import CODE_VERSION, PIN_REF, TARBALL_SHA256, _coords_text, _data_text

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = (
    "globalopt/lj38",
    "globalopt/lj75",
    "globalopt/lj98",
    "globalopt/lj150",
    "globalopt/lj100-binary",
)


def test_registry_pins_the_tarball():
    registry = tomllib.loads((ROOT / "codes" / "registry.toml").read_text(encoding="utf-8"))
    row = next(item for item in registry["code"] if item["id"] == "gmin-bh")
    assert row["name"] == "GMIN"
    assert row["commit"] == TARBALL_SHA256
    assert row["ref"] == PIN_REF
    assert CODE_VERSION == "df8da599"
    assert TARBALL_SHA256.startswith(row["ref"])
    assert "df8da599267534fbe43a250e578f7396a7946311" in row["note"]
    quench = next(item for item in registry["code"] if item["id"] == "gmin")
    assert "gmin-bh" in quench["note"]


def test_bh_and_bhpt_templates_on_every_problem():
    found = [item for item in list_templates(ROOT) if item.get("engine") == "gmin-bh"]
    ids = sorted(item["id"] for item in found)
    expected = [
        f"gmin/{problem.split('/')[1]}-{method}"
        for problem in PROBLEMS
        for method in ("bh", "bhpt")
    ]
    assert ids == sorted(expected)
    for template in found:
        assert template["code"] == "GMIN"
        assert template["code_version"] == CODE_VERSION
        assert template["commit"] == TARBALL_SHA256
        assert template["problem"] in PROBLEMS
        section = template["gmin"]
        assert section["method"] in {"bh", "bhpt"}
        if section["method"] == "bhpt":
            assert int(section["replicas"]) >= 2
        else:
            assert int(section["replicas"]) == 1
        problem = resolve_problem(ROOT, template["problem"])
        _check_stop(problem, template)
        convergence = problem.get("convergence") or {}
        stated = convergence.get("energy_above_minimum")
        if stated is not None:
            assert abs(float(stated) - float(section["ediff"])) < 1e-12
        stated_cap = convergence.get("max_force_calls")
        if stated_cap is not None:
            assert int(template["failure_cap"]) == int(stated_cap)
        else:
            assert int(template["failure_cap"]) == 2_000_000


def test_cluster_starts_are_the_numbered_website_coords(tmp_path):
    from optbench.runners.gmin_go import load_geometries

    expected = {
        "globalopt/lj75": ("website/globalopt/lj75/lj75_clusters.tgz", 200),
        "globalopt/lj98": ("website/globalopt/lj98/lj98_clusters.tgz", 100),
        "globalopt/lj150": ("website/globalopt/lj150/lj150_clusters.tgz", 100),
    }
    for problem_id, (archive, count) in expected.items():
        problem = resolve_problem(ROOT, problem_id)
        assert (problem.get("input") or {}).get("archive_path") == archive
        loaded = load_geometries(ROOT, problem, tmp_path / problem_id.replace("/", "-"))
        assert len(loaded) == count
        assert loaded[0][0] == "coords.1"
        assert loaded[-1][0] == f"coords.{count}"
        assert "coords" not in {name for name, _pos in loaded}


def test_bhpt_keyword_order_and_repeated_coords():
    job = {
        "replicas": 4,
        "pttmin": 0.1,
        "pttmax": 1.0,
        "exchprob": 0.1,
        "sloppyconv": 0.005,
        "tightconv": 0.001,
        "ediff": 0.001,
        "maxit": 600,
        "maxit2": 500,
        "radius": 50.0,
        "temperature": 0.1,
        "step": 0.4,
        "astep": 0.51,
        "steps": 1000000,
        "seed": 20261018,
        "reference": -173.928426511789,
        "cap": 2000000,
    }
    text = _data_text(job)
    assert text.startswith("MPI\nBHPT 0.10000000 1.00000000 0.10000000\n")
    assert "TARGET -173.928426511789" in text
    serial = dict(job)
    serial["replicas"] = 1
    serial_text = _data_text(serial)
    assert serial_text.startswith("MPI\n")
    assert "BHPT" not in serial_text
    from optbench.runners.gmin_go import _command

    command = _command("/tmp/gmin-build/GMIN", 4)
    assert command[command.index("-np") + 1] == "4"
    assert "mpi_yield_when_idle" in command
    coords = _coords_text([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 4)
    assert len(coords.splitlines()) == 8


def test_every_template_runs_the_stock_deck_at_the_problem_cap():
    found = [item for item in list_templates(ROOT) if item.get("engine") == "gmin-bh"]
    decks = {}
    for template in found:
        problem = resolve_problem(ROOT, template["problem"])
        assert int(template["failure_cap"]) == int(problem["convergence"]["max_force_calls"]) == 2_000_000
        section = dict(template["gmin"])
        method = section.pop("method")
        decks.setdefault(method, []).append(section)
        assert "symmetrise" not in section and "symmetrisecsm" not in section
    assert sorted(decks) == ["bh", "bhpt"]
    for sections in decks.values():
        assert all(section == sections[0] for section in sections)
        assert len(sections) == len(PROBLEMS)


def test_asv_classes_import_with_one_start():
    import benchmarks.bench_gmin_bh as bench

    classes = [
        value
        for value in vars(bench).values()
        if isinstance(value, type) and getattr(value, "param_names", None) == ["seed", "start"]
    ]
    assert len(classes) == 2 * len(PROBLEMS)
    caps = {
        item["id"]: int(item["failure_cap"])
        for item in list_templates(ROOT)
        if item.get("engine") == "gmin-bh"
    }
    for benchmark in classes:
        assert benchmark.params == [["0"], ["0"]]
        assert benchmark.cap == caps[benchmark.spec["template_id"]]
