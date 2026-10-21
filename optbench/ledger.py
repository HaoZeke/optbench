"""Campaigns that sit beside the 2014 archive.

The ledger is ``oracle/ledger.jsonl``. A campaign is recorded with the
numbers it measured. Nothing in this module drops a row for missing an
archive value, and the ranked table does not read a pass or fail mark.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Numbered tables in Chill et al., JCTC 2014, 10, 5476. SoftSaddle is the
# April 2019 website, already in the oracle, and is not one of those tables.
PAPER = {
    "minimization/lj38/optim",
    "minimization/lj38/pele-lbfgs-M100-maxstep0.2",
    "minimization/lj38/eon-lbfgs",
    "minimization/lj38/ase-lbfgs",
    "minimization/lj38/eon-cg",
    "minimization/lj38/eon-fire",
    "minimization/lj38/eon-qm",
    "minimization/lj38/eon-sd",
    "minimization/morse-bulk/optim",
    "minimization/morse-bulk/pele-lbfgs",
    "minimization/morse-bulk/eon-lbfgs",
    "minimization/morse-bulk/ase-lbfgs",
    "minimization/morse-bulk/eon-cg",
    "minimization/morse-bulk/eon-fire",
    "minimization/morse-bulk/eon-sd",
    "globalopt/lj38/gmin-csm",
    "globalopt/lj38/gmin-symmetrise",
    "globalopt/lj38/gmin-nosym",
    "globalopt/lj38/eon-basinhopping",
    "globalopt/lj38/pele-basinhopping",
    "globalopt/lj100-binary/gmin",
    "globalopt/lj100-binary/eon-basinhopping",
    "globalopt/tip4p10/gmin",
    "globalopt/tip4p10/gmin-tbp",
    "globalopt/tip4p15/gmin",
    "globalopt/tip4p15/gmin-tbp",
    "globalopt/tip4p20/gmin",
    "globalopt/tip4p20/gmin-tbp",
    "lowest-eigenvector/lj38/optim-first-order",
    "lowest-eigenvector/lj38/pele-first-order",
    "lowest-eigenvector/lj38/tsase-lanczos",
    "lowest-eigenvector/lj38/tsase-dimer-bfgs",
    "lowest-eigenvector/lj38/eon-dimer-lbfgs",
    "lowest-eigenvector/lj38/eon-lanczos",
    "lowest-eigenvector/lj38/tsase-dimer-cg",
    "lowest-eigenvector/lj38/eon-dimer-cg",
    "tsopt/LJ38/optim_ffd",
    "tsopt/LJ38/pele_ffd",
    "tsopt/LJ38/eon-lanczos-improved",
    "tsopt/LJ38/eon-dimer",
    "tsopt/pt-island/optim_ffd",
    "tsopt/pt-island/pele",
    "tsopt/pt-island/eon-lanczos-improved",
    "tsopt/pt-island/eon-dimer",
    "double-ended-saddle-search/lj38/optim",
    "double-ended-saddle-search/lj38/eon-lanczos-improved",
    "double-ended-saddle-search/lj38/pele",
    "double-ended-saddle-search/lj38/eon-dimer",
    "double-ended-saddle-search/lj38/eon-ci-neb",
    "double-ended-saddle-search/lj38/eon-ci-swdneb",
    "double-ended-saddle-search/pt-island/optim",
    "double-ended-saddle-search/pt-island/eon-ci-neb-single",
    "double-ended-saddle-search/pt-island/eon-lanczos",
    "double-ended-saddle-search/pt-island/eon-dimer",
    "double-ended-saddle-search/pt-island/eon-ci-neb",
    "double-ended-saddle-search/pt-island/pele",
    "double-ended-saddle-search/pt-island/eon-ci-neb-5",
}

PAPER_METRIC = {
    "minimization/lj38/eon-fire": ("force_calls", 645.0),
    "double-ended-saddle-search/pt-island/eon-lanczos": ("force_calls", 165.0),
    "double-ended-saddle-search/pt-island/eon-ci-neb": ("nfailed", 3.0),
}

SAMPLE_N = {
    "minimization/lj38": 1000,
    "minimization/morse-bulk": 100,
    "tsopt/LJ38": 200,
    "tsopt/pt-island": 49,
    "double-ended-saddle-search/lj38": 50,
    "double-ended-saddle-search/lj38_2ts_separation": 51,
    "double-ended-saddle-search/pt-island": 59,
    "lowest-eigenvector/lj38": 200,
    "globalopt/lj38": 100,
    "neb/Pt-heptamer": 13,
}

SHA = {
    "main": "883b4153baa013b52de36c4512f0859bea15dc35",
    "cells/optim/min": "f3cf41d74f73fb2ffb3b5199291c90a4743a1000",
    "cells/optim/saddle-pt": "02405b0d",
    "cells/pele/min": "5bb7c3801f52b697c48ada5811c5f766dd81a23d",
    "cells/pele/go": "0528a37b8ff6e54805b9fa75345ee3c40b756a5d",
    "cells/pele/ts-repro": "54ac7e7a1aa6581c261c17fc2c52a20e77121911",
    "cells/softsaddle/ts-reps": "bd1a5c9fc2685f5876266d53715accb31a049e73",
    "entries/eon-svn-htst": "715bc521c827f1f18102c1c027d99d0d8624b0fb",
    "entries/eon-svn-lane-a": "c140d970f659ba27d6048130f0e67fa02944bd7a",
    "infra/aladatom-repro": "d3a28c98d4f544aa3f79876028f95d7e352fd17b",
}

# Exact-setup campaigns. kind is dat, screen, or htst.
# value_from renames a file field onto the archived metric name.
MATCHES: list[dict] = [
    # minimization / lj38
    {"key": "minimization/lj38/eon-lbfgs", "template": "eon-client/lbfgs-m100", "branch": "main", "path": "results/campaign/eon-client-lbfgs-m100.dat"},
    {"key": "minimization/lj38/eon-cg", "template": "eon-client/cg", "branch": "main", "path": "results/campaign/eon-client-cg.dat"},
    {"key": "minimization/lj38/eon-fire", "template": "eon-client/fire", "branch": "main", "path": "results/campaign/eon-client-fire.dat"},
    {"key": "minimization/lj38/eon-qm", "template": "eon-client/qm", "branch": "main", "path": "results/campaign/eon-client-qm.dat"},
    {"key": "minimization/lj38/eon-sd", "template": "eon-client/sd", "branch": "main", "path": "results/campaign/eon-client-sd.dat"},
    {"key": "minimization/lj38/eon-sd-twopoint", "template": "eon-client/sd-twopoint", "branch": "main", "path": "results/campaign/eon-client-sd-twopoint.dat"},
    {"key": "minimization/lj38/ase-lbfgs", "template": "ase/lj38-lbfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-lbfgs-tuned.dat"},
    {"key": "minimization/lj38/ase-BFGS", "template": "ase/lj38-bfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-bfgs-tuned.dat"},
    {"key": "minimization/lj38/ase-bfgslinesearch", "template": "ase/lj38-bfgslinesearch-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-bfgslinesearch-tuned.dat"},
    {"key": "minimization/lj38/ase-fire", "template": "ase/lj38-fire-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-lj38-fire-tuned.dat"},
    {"key": "minimization/lj38/optim", "template": "optim/lj38-lbfgs-tuned", "branch": "cells/optim/min", "path": "results/campaign/optim-02aaf45245af-lj38-lbfgs-tuned.dat", "extra": {"unrounded_mean": 175.537}},
    {"key": "minimization/lj38/pele-lbfgs-M1-maxstep0.1", "template": "pele-lbfgs-M1-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M1-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M10-maxstep0.1", "template": "pele-lbfgs-M10-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M10-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M100-maxstep0.2", "template": "pele-lbfgs-M100-maxstep0.2", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M100-maxstep0.2.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M20-maxstep0.1", "template": "pele-lbfgs-M20-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M20-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M4-maxstep0.05", "template": "pele-lbfgs-M4-maxstep0.05", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.05.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M4-maxstep0.1", "template": "pele-lbfgs-M4-maxstep0.1", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.1.dat"},
    {"key": "minimization/lj38/pele-lbfgs-M4-maxstep0.2", "template": "pele-lbfgs-M4-maxstep0.2", "branch": "main", "path": "results/campaign/pele-0.1.2-lbfgs-M4-maxstep0.2.dat"},
    {"key": "minimization/lj38/pele-fire", "template": "pele/lj38-fire", "branch": "cells/pele/min", "path": "results/campaign/pele-b7081b93-lj38-fire.dat", "cause": "pele_fire"},
    # minimization / morse
    {"key": "minimization/morse-bulk/eon-lbfgs", "template": "eon-client/morse-lbfgs", "branch": "main", "path": "results/campaign/eon-client-morse-lbfgs.dat"},
    {"key": "minimization/morse-bulk/eon-cg", "template": "eon-client/morse-cg", "branch": "main", "path": "results/campaign/eon-client-morse-cg.dat"},
    {"key": "minimization/morse-bulk/eon-fire", "template": "eon-client/morse-fire", "branch": "main", "path": "results/campaign/eon-client-morse-fire.dat"},
    {"key": "minimization/morse-bulk/eon-sd", "template": "eon-client/morse-sd", "branch": "main", "path": "results/campaign/eon-client-morse-sd.dat", "cause": "morse_sd"},
    {"key": "minimization/morse-bulk/optim", "template": "optim/morse-lbfgs-tuned", "branch": "cells/optim/min", "path": "results/campaign/optim-02aaf45245af-morse-lbfgs-tuned.dat", "cause": "optim_morse_memory"},
    {"key": "minimization/morse-bulk/pele-lbfgs", "template": "pele/morse-lbfgs", "branch": "cells/pele/min", "path": "results/campaign/pele-b7081b93-morse-lbfgs.dat", "cause": "pele_morse"},
    {"key": "minimization/morse-bulk/ase-lbfgs", "template": "ase/morse-lbfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-morse-lbfgs-tuned.dat", "cause": "ase_morse_script"},
    {"key": "minimization/morse-bulk/ase-fire", "template": "ase/morse-fire-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-morse-fire-tuned.dat", "cause": "ase_morse_script"},
    {"key": "minimization/morse-bulk/ase-preconLBFGS", "template": "ase/morse-preconlbfgs-tuned", "branch": "main", "path": "results/campaign/ase-3.29.0-morse-preconlbfgs-tuned.dat", "cause": "ase_morse_script"},
    # lowest eigenvector
    {"key": "lowest-eigenvector/lj38/eon-dimer-cg", "template": "eon-client/low-lj38-dimer-cg", "branch": "main", "path": "results/campaign/eon-client-low-lj38-dimer-cg.dat"},
    {"key": "lowest-eigenvector/lj38/eon-dimer-lbfgs", "template": "eon-client/low-lj38-dimer-lbfgs", "branch": "main", "path": "results/campaign/eon-client-low-lj38-dimer-lbfgs.dat"},
    {"key": "lowest-eigenvector/lj38/eon-lanczos", "template": "eon-svn/low-lj38-lanczos-ljcut", "branch": "main", "path": "results/campaign/eon-svn-low-lj38-lanczos-ljcut.dat"},
    {"key": "lowest-eigenvector/lj38/pele-first-order", "template": "pele/lj38-loweig-forward", "branch": "cells/pele/ts", "path": "results/campaign/pele-b7081b93-lj38-loweig-forward.dat"},
    {"key": "lowest-eigenvector/lj38/pele-second-order", "template": "pele/lj38-loweig-central", "branch": "cells/pele/ts", "path": "results/campaign/pele-b7081b93-lj38-loweig-central.dat"},
    {"key": "lowest-eigenvector/lj38/optim-first-order", "template": "optim/rr-loweig", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-df8da599-rr-loweig.dat", "cause": "optim_rr_defaults"},
    # tsopt
    {"key": "tsopt/LJ38/eon-dimer", "template": "eon-svn/ts-lj38-dimer-ljcut", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-dimer-ljcut.dat"},
    {"key": "tsopt/LJ38/eon-lanczos", "template": "eon-svn/ts-lj38-lanczos-ljcut", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-lanczos-ljcut.dat"},
    {"key": "tsopt/LJ38/eon-lanczos-improved", "template": "eon-svn/ts-lj38-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-lanczos-improved.dat", "cause": "improved_unset"},
    {"key": "tsopt/LJ38/sella", "template": "sella/lj38-ts-default", "branch": "main", "path": "results/campaign/sella-2.6.0-lj38-ts-default.dat"},
    {"key": "tsopt/LJ38/pele_ffd", "template": "pele/lj38-ts-ffd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-lj38-ts-ffd.screen.jsonl", "kind": "screen", "cause": "lj38_ts_starts"},
    {"key": "tsopt/LJ38/pele_cd", "template": "pele/lj38-ts-cd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-lj38-ts-cd.screen.jsonl", "kind": "screen", "cause": "lj38_ts_starts_cd"},
    {"key": "tsopt/LJ38/optim_ffd", "template": "optim/hef-ffd-ts-lj38", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-df8da599-hef-ffd-ts-lj38.dat", "cause": "optim_lj38_ts"},
    {"key": "tsopt/pt-island/eon-dimer", "template": "eon-svn/ts-pt-dimer-smooth", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-dimer-smooth.dat"},
    {"key": "tsopt/pt-island/eon-lanczos", "template": "eon-svn/ts-pt-lanczos-smooth", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-lanczos-smooth.dat"},
    {"key": "tsopt/pt-island/eon-lanczos-improved", "template": "eon-svn/ts-pt-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-lanczos-improved.dat"},
    {"key": "tsopt/pt-island/sella", "template": "sella/pt-island-ts-default", "branch": "main", "path": "results/campaign/sella-2.6.0-pt-island-ts-default.dat"},
    {"key": "tsopt/pt-island/pele", "template": "pele/pt-island-ts-ffd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-pt-island-ts-ffd.screen.jsonl", "kind": "screen"},
    {"key": "tsopt/pt-island/optim_ffd", "template": "optim/hef-ffd-ts-pt", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-df8da599-hef-ffd-ts-pt.dat", "cause": "optim_pt_units"},
    # double-ended
    {"key": "double-ended-saddle-search/lj38/eon-ci-neb", "template": "eon-svn/de-lj38-ci-neb", "branch": "main", "path": "results/campaign/eon-svn-de-lj38-ci-neb.dat"},
    {"key": "double-ended-saddle-search/lj38/eon-ci-swdneb", "template": "eon-3.5.0/de-lj38-ci-swdneb", "branch": "main", "path": "results/campaign/eon-3.5.0-de-lj38-ci-swdneb.dat", "cause": "de_exclusion", "hold": ["eon-client/de-lj38-ci-swdneb"]},
    {"key": "double-ended-saddle-search/lj38/eon-dimer", "template": "eon-svn/de-lj38-dimer", "branch": "main", "path": "results/campaign/eon-svn-de-lj38-dimer.dat", "cause": "improved_unset"},
    {"key": "double-ended-saddle-search/lj38/eon-lanczos", "template": "eon-client/de-lj38-lanczos", "branch": "main", "path": "results/campaign/eon-client-de-lj38-lanczos.dat"},
    {"key": "double-ended-saddle-search/lj38/eon-lanczos-improved", "template": "eon-client/de-lj38-lanczos-improved", "branch": "main", "path": "results/campaign/eon-client-de-lj38-lanczos-improved.dat"},
    {"key": "double-ended-saddle-search/lj38/optim", "template": "optim/de-lj38", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-svn-snapshot-double-ended-saddle-search-lj38--optim-de-lj38.dat", "cause": "optim_pes_counter"},
    {"key": "double-ended-saddle-search/lj38_2ts_separation/optim", "template": "optim/de-2ts", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-svn-snapshot-double-ended-saddle-search-lj38_2ts_separation--optim-de-2ts.dat", "cause": "optim_pes_counter"},
    {"key": "double-ended-saddle-search/pt-island/eon-ci-neb", "template": "eon-svn/de-pt-ci-neb", "branch": "main", "path": "results/campaign/eon-svn-de-pt-ci-neb.dat"},
    {"key": "double-ended-saddle-search/pt-island/eon-ci-swdneb", "template": "eon-svn/de-pt-ci-swdneb", "branch": "main", "path": "results/campaign/eon-svn-de-pt-ci-swdneb.dat"},
    {"key": "double-ended-saddle-search/pt-island/eon-dimer", "template": "eon-svn/de-pt-dimer", "branch": "main", "path": "results/campaign/eon-svn-de-pt-dimer.dat"},
    {"key": "double-ended-saddle-search/pt-island/eon-lanczos", "template": "eon-svn/de-pt-lanczos", "branch": "main", "path": "results/campaign/eon-svn-de-pt-lanczos.dat", "cause": "pt_lanczos"},
    {"key": "double-ended-saddle-search/pt-island/eon-lanczos-improved", "template": "eon-svn/de-pt-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-de-pt-lanczos-improved.dat", "cause": "pt_lanczos"},
    {"key": "double-ended-saddle-search/pt-island/optim", "template": "optim/de-pt", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-svn-snapshot-double-ended-saddle-search-pt-island--optim-de-pt.dat", "cause": "optim_pt_units"},
    # neb per-image means already stored in the 3.5.0 files
    {"key": "neb/Pt-heptamer/eon-lbfgs", "template": "eon-lbfgs", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-lbfgs.dat", "hold": ["eon-client/neb-pt-lbfgs", "eon/neb-pt-lbfgs"]},
    {"key": "neb/Pt-heptamer/eon-cg", "template": "eon-cg", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-cg.dat", "cause": "heptamer_cg", "hold": ["eon-client/neb-pt-cg"]},
    {"key": "neb/Pt-heptamer/eon-fire", "template": "eon-fire", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-fire.dat", "hold": ["eon-client/neb-pt-fire"]},
    {"key": "neb/Pt-heptamer/eon-qm", "template": "eon-qm", "branch": "main", "path": "results/campaign/eon-3.5.0-neb-pt-qm.dat", "hold": ["eon-client/neb-pt-qm"]},
    # saddle search
    {
        "key": "saddle-search/pt-island/eon-dimer",
        "template": "eon-svn/saddle-pt-dimer",
        "branch": "main",
        "path": "results/campaign/eon-svn-r2549-saddle-pt-dimer.dat",
        "metrics": ["force_calls", "unique_saddles", "fraction_connected"],
        "label": "EON SVN r2549",
        "also": [
            {
                "template": "eon/develop-saddle-pt-dimer",
                "branch": "main",
                "path": "results/campaign/eon-develop-500350d9-saddle-pt-dimer.dat",
                "label": "eOn (develop 500350d9)",
            }
        ],
    },
    # globalopt and htst and aladatom rates
    {"key": "globalopt/lj38/eon-basinhopping", "template": "eon-client/bh-lj38", "branch": "entries/eon-svn-lane-a", "path": "results/campaign/eon-client-bh-lj38.dat"},
    {"key": "globalopt/lj38/pele-basinhopping", "template": "pele/lj38-basinhopping", "branch": "cells/pele/go", "path": "results/campaign/pele-b7081b93-lj38-basinhopping.dat", "cause": "pele_bh_defaults"},
    {"key": "hTST/Al/eon-dimer-cg", "template": "eon-svn/htst-al-dimer", "branch": "entries/eon-svn-htst", "path": "results/campaign/eon-svn-htst-al-dimer.dat", "kind": "htst", "cause": "htst_al"},
    {"key": "hTST/Pt/eon-dimer-cg", "template": "eon-svn/htst-pt-dimer", "branch": "entries/eon-svn-htst", "path": "results/campaign/eon-svn-htst-pt-dimer.dat", "kind": "htst"},
]


def row_id(row: dict) -> str:
    return f"{row['suite']}/{row['problem']}/{row['entry']}"



def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").split()[0])
    except ValueError:
        return None


def parse_dat(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) == 2 and parts[0] not in fields:
            fields[parts[0]] = parts[1].strip()
    return fields


def metrics_for(fields: dict[str, str], kind: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    if kind == "htst":
        return ["barrier_force_calls", "barrier_jobs", "rate_force_calls", "rate_jobs"]
    if "true_rate" in fields:
        return ["true_rate"]
    if "avg_energy" in fields:
        return ["avg_energy", "min_energy", "max_energy"]
    if "mean_epoch" in fields:
        return ["mean_epoch"]
    if "barrier_force_calls" in fields:
        return ["barrier_force_calls", "barrier_jobs", "rate_force_calls", "rate_jobs"]
    return ["force_calls"]



def screen_fields(text: str) -> dict[str, str]:
    calls = []
    harness = []
    failed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        calls.append(float(row["stop2014_calls"]))
        harness.append(float(row["force_calls"]))
        if not row.get("stop2014_success", False):
            failed += 1
    count = len(calls)
    mean = sum(calls) / count
    harness_mean = sum(harness) / count
    return {
        "force_calls": f"{mean:.6g}",
        "nfailed": str(failed),
        "n": str(count),
        "harness_stop_mean": f"{harness_mean:.6g}",
        "baseline_metric": "stop2014_calls",
    }


def read_campaign(root: Path, branch: str, path: str) -> str:
    if branch == "main":
        return (root / path).read_text(encoding="utf-8")
    import subprocess

    return subprocess.check_output(
        ["git", "show", f"origin/{branch}:{path}"],
        cwd=root,
        text=True,
    )


def _sha(branch: str) -> str:
    if branch == "cells/optim/saddle-pt" and len(SHA.get(branch, "")) < 12:
        import subprocess

        SHA[branch] = subprocess.check_output(
            ["git", "rev-parse", "origin/cells/optim/saddle-pt"],
            text=True,
        ).strip()
    if branch == "cells/pele/ts" and "cells/pele/ts" not in SHA:
        import subprocess

        SHA[branch] = subprocess.check_output(
            ["git", "rev-parse", "origin/cells/pele/ts"],
            text=True,
        ).strip()
    return SHA[branch]


def load_ledger(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows



def _git_blob_size(root: Path, path: Path) -> int | None:
    rel = path.relative_to(root).as_posix()
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "cat-file", "-s", f":{rel}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return int(out.strip())


def _lfs_tracked(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "check-attr", "filter", "--", rel],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return out.strip().endswith("filter: lfs")


def large_endpoint_files(root: Path, limit: int = 1_000_000) -> list[str]:
    """Endpoint archives stored as git blobs instead of an LFS pointer.

    The committed size is the blob in the index. A path matched by
    git-LFS is allowed while it is still untracked, because ``git add``
    stores the pointer. A loose blob over ``limit`` fails.
    """
    errors = []
    campaign = root / "results" / "campaign"
    if not campaign.is_dir():
        return errors
    for path in sorted(campaign.rglob("*.npz")):
        if "endpoint" not in path.name:
            continue
        committed = _git_blob_size(root, path)
        if committed is not None:
            size = committed
        elif _lfs_tracked(root, path):
            continue
        else:
            size = path.stat().st_size
        if size > limit:
            errors.append(f"{path.relative_to(root)} is {size} bytes")
    return errors



LEDGER_PATH = "oracle/ledger.jsonl"
LEDGER_DOC = "docs/archive-runs.md"


def _spec_list(match: dict) -> list[dict]:
    specs = [match]
    for extra in match.get("also") or []:
        merged = {
            "branch": extra.get("branch", "main"),
            "kind": extra.get("kind", match.get("kind", "dat")),
            "metrics": extra.get("metrics", match.get("metrics")),
            "value_from": extra.get("value_from", match.get("value_from")),
            "template": extra.get("template", match.get("template")),
            "path": extra["path"],
        }
        merged.update(extra)
        specs.append(merged)
    return specs


def _read_fields(root: Path, spec: dict) -> dict[str, str] | None:
    branch = spec.get("branch", "main")
    kind = spec.get("kind", "dat")
    try:
        text = read_campaign(root, branch, spec["path"])
    except (OSError, subprocess.CalledProcessError):
        return None
    if kind == "screen":
        return screen_fields(text)
    return parse_dat(text)


def _metric_rows(
    archived: dict[str, str],
    fields: dict[str, str],
    names: list[str],
    value_from: dict[str, str] | None,
) -> list[dict]:
    renaming = value_from or {}
    rows = []
    for name in names:
        archive = _float(archived.get(name))
        source = renaming.get(name, name)
        latest_text = fields.get(source, "")
        latest = _float(latest_text)
        if archive is None and latest is None:
            continue
        rows.append(
            {
                "metric": name,
                "archive": archive,
                "latest": latest,
                "latest_text": latest_text,
            }
        )
    return rows


def _latest_item(root: Path, spec: dict, archived: dict[str, str], names: list[str]) -> dict | None:
    fields = _read_fields(root, spec)
    if fields is None:
        return None
    code = str(spec.get("code") or fields.get("code") or "")
    version = str(spec.get("code_version") or fields.get("code_version") or "")
    label = str(spec.get("label") or f"{code} {version}".strip() or spec.get("template") or "")
    return {
        "template": spec.get("template", ""),
        "branch": spec.get("branch", "main"),
        "sha": _sha(spec.get("branch", "main")),
        "path": spec["path"],
        "kind": spec.get("kind", "dat"),
        "hold": list(spec.get("hold") or []),
        "code": code,
        "code_version": version,
        "label": label,
        "metrics": _metric_rows(archived, fields, names, spec.get("value_from")),
    }


class Hold:
    """Alternate templates that use a different counter from the ranked row."""

    def __init__(self) -> None:
        self.templates: set[tuple[str, str]] = set()
        self.paths: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self.templates or self.paths)


def holdback(root: Path) -> Hold:
    """Templates named by a campaign as a different counter.

    A campaign is never held because its mean sits away from the archive.
    """
    found = Hold()
    for row in load_ledger(root / LEDGER_PATH):
        problem = f"{row['suite']}/{row['problem']}"
        for item in row.get("latest") or []:
            for extra in item.get("hold") or []:
                found.templates.add((problem, str(extra)))
    return found


def ledger_errors(root: Path) -> list[str]:
    ledger_path = root / LEDGER_PATH
    if not ledger_path.is_file():
        return [f"{LEDGER_PATH} is missing"]
    from .check import load_oracle

    oracle = load_oracle(root / "oracle" / "entries.jsonl")
    ledger = load_ledger(ledger_path)
    errors = []
    if len(ledger) != len(oracle):
        errors.append(f"ledger has {len(ledger)} rows and the oracle has {len(oracle)}")
    if [row_id(row) for row in oracle] != [row_id(row) for row in ledger]:
        errors.append("ledger is not in oracle order")
    errors.extend(large_endpoint_files(root))
    return errors


def build_rows(root: Path) -> list[dict]:
    from .check import load_oracle

    oracle = {row_id(row): row for row in load_oracle(root / "oracle" / "entries.jsonl")}
    matches = {item["key"]: item for item in MATCHES}
    built = []
    for ident, entry in oracle.items():
        fields = entry["fields"]
        record = {
            "suite": entry["suite"],
            "problem": entry["problem"],
            "entry": entry["entry"],
            "archive_path": entry["path"],
            "commit": entry["commit"],
            "archived_setup": {
                "code": fields.get("code", ""),
                "algorithm": fields.get("algorithm", ""),
                "code_version": fields.get("code_version", ""),
            },
            "latest": [],
        }
        match = matches.get(ident)
        if match is None:
            built.append(record)
            continue
        names = metrics_for(fields, match.get("kind", "dat"), match.get("metrics"))
        for spec in _spec_list(match):
            item = _latest_item(root, spec, fields, names)
            if item is not None:
                record["latest"].append(item)
        built.append(record)
    return built


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.6g}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6g}"


def render_markdown(rows: list[dict]) -> str:
    paired = sum(1 for row in rows if row.get("latest"))
    lines = [
        "# Archive and latest runs",
        "",
        "Archive commit `0f26b796`. Each measured campaign is listed with the published archive value. "
        "The ranked table uses the measured numbers.",
        "",
        f"{paired} campaigns are recorded with {len(rows)} archive rows.",
        "",
        "Pt-island pele records the 2014 stop mean. The harness-stop means on the ranked table are a separate count.",
        "",
    ]
    section = ""
    for row in rows:
        suite = row["suite"]
        if suite != section:
            if section:
                lines.append("")
            section = suite
            lines.append(f"## {suite}")
            lines.append("")
            lines.append("| Row | Archive | Latest |")
            lines.append("| --- | --- | --- |")
        ident = f"{row['problem']}/{row['entry']}"
        latest_rows = row.get("latest") or []
        if latest_rows and latest_rows[0].get("metrics"):
            archive = ", ".join(
                f"{item['metric']} {_fmt(item['archive'])}"
                for item in latest_rows[0]["metrics"]
                if item.get("archive") is not None
            )
            bits = []
            for item in latest_rows:
                values = ", ".join(
                    _fmt(metric["latest"]) for metric in item["metrics"] if metric.get("latest") is not None
                )
                bits.append(f"{item['label']}: {values}")
            latest = "; ".join(bits)
        else:
            archive = "—"
            latest = "—"
        lines.append(f"| {ident} | {archive or '—'} | {latest or '—'} |")
    lines.append("")
    return "\n".join(lines)


def write_ledger(root: Path) -> list[dict]:
    rows = build_rows(root)
    (root / LEDGER_PATH).write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (root / LEDGER_DOC).write_text(render_markdown(rows), encoding="utf-8")
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = write_ledger(root)
    print(f"rows {len(rows)} with_latest {sum(1 for row in rows if row.get('latest'))}")


_LATEST_CACHE: dict[str, dict[str, list[dict]]] = {}


def latest_for(root: Path) -> dict[str, list[dict]]:
    """Campaigns to draw beside each archive row, keyed by suite/problem/entry."""
    key = str(Path(root).resolve())
    cached = _LATEST_CACHE.get(key)
    if cached is not None:
        return cached
    found: dict[str, list[dict]] = {}
    for row in build_rows(root):
        if row.get("latest"):
            found[row_id(row)] = row["latest"]
    _LATEST_CACHE[key] = found
    return found


if __name__ == "__main__":
    main()
