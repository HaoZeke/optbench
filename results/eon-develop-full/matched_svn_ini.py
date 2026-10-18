"""One develop-binary start on the SVN r2549 ini.

The client is the develop build named by ``EON_COMMIT``. The ini sections
are the SVN template's, so L-BFGS keys sit on ``[Optimizer]`` and the
develop ``[LBFGS]`` catalog is absent. The harness path is the develop
``run_case`` path: ``Main.parallel`` false, potential swapped to
``ext_pot``, ``HarnessCounter`` through the pot server. The probe cap is
high enough that a long trace is not cut off. Nothing is written into
the asv cache.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SVN_REF = "origin/entries/eon-svn-fortran"
PROBE_CAP = 100000

CASES = (
    {
        "name": "ts-lj38-dimer",
        "develop": "eon/develop-ts-lj38-dimer",
        "svn": "templates/eon-svn/ts-lj38-dimer.toml",
        "kind": "ts-lj38",
        "seed": 1,
        "start": "0000",
    },
    {
        "name": "ts-pt-dimer",
        "develop": "eon/develop-ts-pt-dimer",
        "svn": "templates/eon-svn/ts-pt-dimer.toml",
        "kind": "ts-pt",
        "seed": 1,
        "start": "initial_0",
    },
    {
        "name": "de-lj38-ci",
        "develop": "eon/develop-de-lj38-cap",
        "svn": "templates/eon-svn/de-lj38-ci.toml",
        "kind": "de-lj38",
        "seed": 1,
        "start": "pair_00",
    },
    {
        "name": "de-2ts-ci",
        "develop": "eon/develop-de-2ts",
        "svn": "templates/eon-svn/de-2ts-ci.toml",
        "kind": "de-2ts",
        "seed": 1,
        "start": "pair_00",
    },
)


def _svn_template(path: str) -> dict:
    text = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{SVN_REF}:{path}"],
        text=True,
    )
    return tomllib.loads(text)


CLASSIC_LBFGS = (
    "lbfgs_memory",
    "lbfgs_inverse_curvature",
    "lbfgs_max_inverse_curvature",
    "lbfgs_auto_scale",
    "lbfgs_angle_reset",
    "lbfgs_distance_reset",
)


def _copy_ini(sections: dict) -> dict:
    return json.loads(json.dumps(sections))


def run_ini(case: dict, ini: dict, label: str) -> dict:
    from benchmarks.bench_eon_develop import (
        EON_COMMIT,
        _Bridge,
        _client,
        _client_files,
        _ini,
    )
    from optbench.eon_jobs import _execute
    from optbench.potserver import remote_count

    template = {"id": label, "ini": ini, "failure_cap": PROBE_CAP}
    potential = template["ini"]["Potential"]["potential"]
    started = time.monotonic()
    import tempfile
    import shutil

    work = Path(tempfile.mkdtemp(prefix="optbench-eon-matched-"))
    try:
        with _Bridge(potential, PROBE_CAP) as (server, wrapper):
            copies, texts = _client_files(case["kind"], case["start"], work)
            count_key = "total_force_calls"
            ini = _ini(template, int(case["seed"]), wrapper)
            found = _execute(
                (
                    _client(),
                    ini,
                    str(work / "run"),
                    copies,
                    texts,
                    count_key,
                )
            )
            calls, verify = remote_count(server.host, server.port)
        return {
            "name": case["name"],
            "label": label,
            "develop_template": case["develop"],
            "kind": case["kind"],
            "seed": int(case["seed"]),
            "start": case["start"],
            "eon_commit": EON_COMMIT,
            "probe_cap": PROBE_CAP,
            "potential": potential,
            "counter": "harness",
            "force_calls": float(calls),
            "verify_calls": int(verify),
            "client_calls": found.get("raw"),
            "client_reason": found.get("reason"),
            "extrema": found.get("extrema"),
            "seconds": time.monotonic() - started,
            "ini": ini,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _develop_ini(case: dict) -> dict:
    from optbench.campaign import load_template

    loaded = load_template(ROOT, case["develop"])
    return _copy_ini(loaded["ini"])


def _svn_ini(case: dict) -> tuple[dict, str]:
    svn = _svn_template(case["svn"])
    return _copy_ini(svn["ini"]), str(svn["id"])


def run_one(case: dict) -> dict:
    ini, svn_id = _svn_ini(case)
    row = run_ini(case, ini, "svn-ini")
    row["svn_template"] = svn_id
    return row


def _with_lbfgs_section(base: dict, keys: dict) -> dict:
    ini = _copy_ini(base)
    optimizer = ini.setdefault("Optimizer", {})
    for key in CLASSIC_LBFGS:
        optimizer.pop(key, None)
    ini["LBFGS"] = dict(keys)
    return ini


def ablate(case: dict) -> list[dict]:
    """Which ini keys move this start off the SVN count.

    The develop control is the published template. ``svn-ini`` is the r2549
    layout. The other rows add one develop-only piece at a time.
    """
    develop = _develop_ini(case)
    svn, _svn_id = _svn_ini(case)
    catalog = _copy_ini(develop.get("LBFGS") or {})
    classic = {key: catalog[key] for key in CLASSIC_LBFGS if key in catalog}
    extra = {key: value for key, value in catalog.items() if key not in CLASSIC_LBFGS}
    neb_only = (
        "climbing_image_band_slack",
        "opt_method",
        "setup_mmf_peaks",
        "energy_weighted",
        "ci_mmf",
    )
    rows = [
        ("develop-template", develop),
        ("svn-ini", svn),
        ("svn-plus-full-lbfgs-section", _with_lbfgs_section(svn, catalog)),
        ("svn-plus-classic-lbfgs-section", _with_lbfgs_section(svn, classic)),
        ("svn-plus-extra-lbfgs-keys", _with_lbfgs_section(svn, {**classic, **extra})),
    ]
    # One extra catalog key at a time, on top of the classic six.
    for key, value in extra.items():
        rows.append((f"svn-plus-{key}", _with_lbfgs_section(svn, {**classic, key: value})))
    band = develop.get("Nudged Elastic Band")
    svn_band = svn.get("Nudged Elastic Band")
    if band and svn_band:
        added = _copy_ini(svn)
        for key in neb_only:
            if key in band:
                added["Nudged Elastic Band"][key] = band[key]
        rows.append(("svn-plus-neb-keys", added))
        both = _with_lbfgs_section(added, catalog)
        rows.append(("svn-plus-neb-and-lbfgs", both))
    out = []
    for label, ini in rows:
        # ``svn-plus-extra-lbfgs-keys`` is the full catalog; skip the duplicate.
        if label == "svn-plus-extra-lbfgs-keys":
            continue
        row = run_ini(case, ini, label)
        out.append(row)
        print(
            f"{case['name']} {case['start']} {label} harness={row['force_calls']} "
            f"client={row['client_calls']} reason={row['client_reason']} "
            f"seconds={row['seconds']:.3f}",
            flush=True,
        )
    return out


def _accept_ini(case: dict) -> dict:
    develop = _develop_ini(case)
    svn, _svn_id = _svn_ini(case)
    catalog = develop.get("LBFGS") or {}
    classic = {key: catalog[key] for key in CLASSIC_LBFGS if key in catalog}
    classic["lbfgs_accept"] = catalog.get("lbfgs_accept", "energy")
    return _with_lbfgs_section(svn, classic)


CONFIRM = (
    ("ts-lj38-dimer", "0001", "svn-ini"),
    ("ts-lj38-dimer", "0001", "svn-plus-accept"),
    ("ts-pt-dimer", "initial_0", "svn-plus-accept"),
    ("ts-pt-dimer", "initial_1", "svn-ini"),
    ("ts-pt-dimer", "initial_1", "svn-plus-accept"),
    ("de-lj38-ci", "pair_00", "svn-plus-accept"),
    ("de-lj38-ci", "pair_01", "svn-ini"),
    ("de-lj38-ci", "pair_01", "svn-plus-accept"),
    ("de-2ts-ci", "pair_00", "svn-plus-accept"),
    ("de-2ts-ci", "pair_49", "svn-ini"),
    ("de-2ts-ci", "pair_49", "svn-plus-accept"),
)


def confirm() -> list[dict]:
    by_name = {case["name"]: case for case in CASES}
    rows = []
    for name, start, label in CONFIRM:
        case = dict(by_name[name])
        case["start"] = start
        if label == "svn-ini":
            ini, _svn_id = _svn_ini(case)
        elif label == "svn-plus-accept":
            ini = _accept_ini(case)
        else:
            raise ValueError(label)
        row = run_ini(case, ini, label)
        rows.append(row)
        print(
            f"{name} {start} {label} harness={row['force_calls']} "
            f"client={row['client_calls']} reason={row['client_reason']} "
            f"seconds={row['seconds']:.3f}",
            flush=True,
        )
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "matched-svn-ini"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "confirm":
        rows = confirm()
        dest = out_dir / "confirm.json"
        dest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "ablate":
        wanted = set(sys.argv[2:])
        cases = [case for case in CASES if not wanted or case["name"] in wanted]
        for case in cases:
            rows = ablate(case)
            dest = out_dir / f"{case['name']}-ablation.json"
            dest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        return
    wanted = set(sys.argv[1:])
    cases = [case for case in CASES if not wanted or case["name"] in wanted]
    if not cases:
        raise SystemExit(f"no cases in {sys.argv[1:]}")
    for case in cases:
        row = run_one(case)
        dest = out_dir / f"{case['name']}.json"
        dest.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(
            f"{row['name']} harness={row['force_calls']} client={row['client_calls']} "
            f"reason={row['client_reason']} seconds={row['seconds']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
