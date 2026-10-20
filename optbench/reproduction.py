"""Reproduction ledger for the 2014 archive and the JCTC tables.

A row is reproduced when every archived metric is within one archived
standard error or 10 percent, and the failure count sits in the Wilson
interval of the archived failure count. The ledger is
``oracle/reproduction.jsonl``. ``paper_baseline`` marks the rerun whose
score is the published comparison. The 2014 pele transition-state stop
is the first RMS crossing plus one eigenvector search, with the
transverse-screen calls left out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RELATIVE = 0.10
Z = 1.96

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
    "globalopt/tip4p10": 10000,
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
    "cells/fill-not-rerun": "dba52b93e12e43df95a96bb24bf612f0c4522f9a",
    "repro/globalopt-tip4p10-gmin-tbp": "154da32bbdfa0883c211eed48bb3a567ca557935",
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
    {"key": "minimization/lj38/ase-lbfgslinesearch", "template": "ase/lj38-lbfgslinesearch-tuned", "branch": "cells/fill-not-rerun", "path": "results/campaign/ase-3.29.0-lj38-lbfgslinesearch-tuned.dat", "cause": "linesearch_exceptions"},
    {"key": "minimization/lj38/ase-preconlbfgs", "template": "ase/lj38-preconlbfgs-tuned", "branch": "cells/fill-not-rerun", "path": "results/campaign/ase-3.29.0-lj38-preconlbfgs-tuned.dat", "cause": "precon_hessian_index"},
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
    {"key": "lowest-eigenvector/lj38/sella", "template": "sella-0.1.1/lj38-loweig", "branch": "cells/fill-not-rerun", "path": "results/campaign/sella-0.1.1-lj38-loweig.dat"},
    {"key": "lowest-eigenvector/lj38/pele-first-order", "template": "pele/lj38-loweig-forward", "branch": "cells/pele/ts", "path": "results/campaign/pele-b7081b93-lj38-loweig-forward.dat"},
    {"key": "lowest-eigenvector/lj38/pele-second-order", "template": "pele/lj38-loweig-central", "branch": "cells/pele/ts", "path": "results/campaign/pele-b7081b93-lj38-loweig-central.dat"},
    {"key": "lowest-eigenvector/lj38/optim-first-order", "template": "optim/rr-loweig", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-df8da599-rr-loweig.dat", "cause": "optim_rr_defaults"},
    # tsopt
    {"key": "tsopt/LJ38/eon-dimer", "template": "eon-svn/ts-lj38-dimer-ljcut", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-dimer-ljcut.dat"},
    {"key": "tsopt/LJ38/eon-lanczos", "template": "eon-svn/ts-lj38-lanczos-ljcut", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-lanczos-ljcut.dat"},
    {"key": "tsopt/LJ38/eon-lanczos-improved", "template": "eon-svn/ts-lj38-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-ts-lj38-lanczos-improved.dat", "cause": "improved_unset"},
    {"key": "tsopt/LJ38/sella", "template": "sella/lj38-ts-tuned", "branch": "main", "path": "results/campaign/sella-2.6.0-lj38-ts-tuned.dat", "cause": "sella_trust"},
    {"key": "tsopt/LJ38/pele_ffd", "template": "pele/lj38-ts-ffd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-lj38-ts-ffd.screen.jsonl", "kind": "screen", "cause": "lj38_ts_starts"},
    {"key": "tsopt/LJ38/pele_cd", "template": "pele/lj38-ts-cd", "branch": "cells/pele/ts-repro", "path": "results/campaign/pele-b7081b93-lj38-ts-cd.screen.jsonl", "kind": "screen", "cause": "lj38_ts_starts_cd"},
    {"key": "tsopt/LJ38/optim_ffd", "template": "optim/hef-ffd-ts-lj38", "branch": "cells/optim/saddle-pt", "path": "results/campaign/optim-df8da599-hef-ffd-ts-lj38.dat", "cause": "optim_lj38_ts"},
    {"key": "tsopt/pt-island/eon-dimer", "template": "eon-svn/ts-pt-dimer-smooth", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-dimer-smooth.dat"},
    {"key": "tsopt/pt-island/eon-lanczos", "template": "eon-svn/ts-pt-lanczos-smooth", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-lanczos-smooth.dat"},
    {"key": "tsopt/pt-island/eon-lanczos-improved", "template": "eon-svn/ts-pt-lanczos-improved", "branch": "main", "path": "results/campaign/eon-svn-ts-pt-lanczos-improved.dat"},
    {"key": "tsopt/pt-island/sella", "template": "sella/pt-island-ts-tuned", "branch": "main", "path": "results/campaign/sella-2.6.0-pt-island-ts-tuned.dat", "cause": "sella_trust"},
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
    {"key": "saddle-search/pt-island/eon-dimer", "template": "eon-svn/saddle-pt-dimer", "branch": "main", "path": "results/campaign/eon-3.5.0-saddle-pt-classic.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "cause": "eon_saddle_count"},
    {"key": "saddle-search/pt-island/adaptive_HSphere-lbfgs-davidson-sm50", "template": "softsaddle/saddle-davidson-sm50", "branch": "cells/softsaddle/ts-reps", "path": "results/campaign/softsaddle-2019.04-saddle-davidson-sm50.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "value_from": {"unique_saddles": "benchmark_unique_saddles", "fraction_connected": "benchmark_fraction_connected"}, "cause": "saddle_novelty"},
    {"key": "saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm50", "template": "softsaddle/saddle-lanczos-sm50", "branch": "cells/softsaddle/ts-reps", "path": "results/campaign/softsaddle-2019.04-saddle-lanczos-sm50.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "value_from": {"unique_saddles": "benchmark_unique_saddles", "fraction_connected": "benchmark_fraction_connected"}, "cause": "saddle_novelty"},
    {"key": "saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm0", "template": "softsaddle/saddle-lanczos-sm0", "branch": "cells/softsaddle/ts-reps", "path": "results/campaign/softsaddle-2019.04-saddle-lanczos-sm0.dat", "metrics": ["force_calls", "unique_saddles", "fraction_connected"], "value_from": {"unique_saddles": "benchmark_unique_saddles", "fraction_connected": "benchmark_fraction_connected"}, "cause": "saddle_novelty"},
    # globalopt and htst and aladatom rates
    {"key": "globalopt/lj38/eon-basinhopping", "template": "eon-client/bh-lj38", "branch": "entries/eon-svn-lane-a", "path": "results/campaign/eon-client-bh-lj38.dat"},
    {"key": "globalopt/tip4p10/gmin-tbp", "template": "gmin/tip4p10-stock-score", "branch": "repro/globalopt-tip4p10-gmin-tbp", "path": "results/reproduction/gmin-df8da599-tip4p10-stock.dat", "metrics": ["force_calls", "force_calls_median", "force_calls_min", "force_calls_max"], "cause": "gmin_stock"},
    {"key": "globalopt/lj38/pele-basinhopping", "template": "pele/lj38-basinhopping", "branch": "cells/pele/go", "path": "results/campaign/pele-b7081b93-lj38-basinhopping.dat", "cause": "pele_bh_defaults"},
    {"key": "hTST/Al/eon-dimer-cg", "template": "eon-svn/htst-al-dimer", "branch": "entries/eon-svn-htst", "path": "results/campaign/eon-svn-htst-al-dimer.dat", "kind": "htst", "cause": "htst_al"},
    {"key": "hTST/Pt/eon-dimer-cg", "template": "eon-svn/htst-pt-dimer", "branch": "entries/eon-svn-htst", "path": "results/campaign/eon-svn-htst-pt-dimer.dat", "kind": "htst"},
]


CAUSE_TEXT = {
    "gmin_stock": (
        "Latest GMIN df8da599267534fbe43a250e578f7396a7946311, unmodified, stock basin-hopping "
        "(temperature 0.3, step 0.3, temperature basin-paving off). "
        "The problem is TIP4P n=10 with the archive convergence, TARGET -391.0233 and EDIFF 1e-3, "
        "on the archived 10000 angle-axis starts. "
        "The binary is the project's gfortran Release build of svn.tar.bz2 "
        "sha256 738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31. "
        "Successful NPCALL sits far above the 2014 linear temperature basin-paving mean, "
        "and the failure count is outside the Wilson interval. "
        "The ranked harness row is gmin/tip4p10-stock. The MC ceiling that was run is 20000 steps."
    ),
    "lj38_ts_starts": (
        "Inside the 10% band (207.51 vs 192). "
        "Archive run.sh unpacks coords-xyz.tar.gz (200 xyz, max pair about 4.4). "
        "The harness problem points at coords-con.tar.gz (max pair about 37). "
        "Kabsch RMS is about 6.33, so the sets are not the same geometry. "
        "Neither set has a pair past cutoff 40, and both drivers use RMS tolerance 1e-3/sqrt(114). "
        "The 8% residual is the starting set. The paper baseline is this 2014-stop mean, not the harness-stop mean 254.48."
    ),
    "lj38_ts_starts_cd": (
        "Inside the 10% band (263.52 vs 254). The same xyz-versus-con starting set as ts-ffd. "
        "findTS_local.py for the CD row does not pass orthogZeroEigs; the harness template sets orthog translations-rotations. "
        "The paper baseline is this 2014-stop mean, not the harness-stop mean 335.06."
    ),
    "optim_lj38_ts": (
        "Archive run.sh unpacks coords-xyz.tar.gz. The harness uses the problem coords-con set (Kabsch RMS about 6.33). "
        "The odata preamble is not in the archive snapshot. The template uses updates 4. The mean is 489 against 145."
    ),
    "pt_lanczos": (
        "The svn template uses the archived Lanczos keys (10 iterations, tolerance 0.05) and shifted Morse::force. "
        "Dimer and climbing-image on that kernel reproduce. Lanczos does not, and the failure count is 0 against 5 or 8."
    ),
    "ase_morse_script": (
        "ASE 3.29 with the archived Morse script (L-BFGS memory 20, maxstep 0.2, alpha 25; FIRE maxstep 0.2; PreconLBFGS Exp A 3) "
        "scores 32, 113, and 29 against 53.8, 147.47, and 34. "
        "Constructor-default L-BFGS scores 53, next to 53.8, but that is not the archived alpha and memory. "
        "eOn Morse L-BFGS and FIRE on the same bulk file reproduce, so the geometry set is not the split."
    ),
    "optim_pt_units": (
        "archive run.sh divides coordinates by r0 = 2.897 and scales rho and rcut into those units. "
        "The harness keeps the problem Morse in eV/Angstrom on the unscaled con file."
    ),
    "optim_pes_counter": (
        "The template note records the 2014 mean as OPTIM's own energy-plus-gradient calls. "
        "The harness score is pes.evaluate on the harness potential."
    ),
    "optim_morse_memory": (
        "comments.txt sets L-BFGS memory 100 and maximum move 2.0. That rerun scores 27. "
        "OPTIM defaults (updates 4) score 46, which is the archived integer, but that is not the comment's setup. "
        "The odata preamble was not archived."
    ),
    "optim_rr_defaults": (
        "The Rayleigh-Ritz template uses updates 4. The archived first-order mean is 25 and the second-order mean is 46. "
        "This default scores 61, so it is not either archived order."
    ),
    "pele_fire": "pele FIRE on LJ38 scores 1264 with 7 failures. The archive is 709 with 0 failures. The M100 L-BFGS row on the same build reproduces, so this is the FIRE step, not the LJ kernel.",
    "pele_morse": "pele Morse L-BFGS scores 92. The archive is 51. eOn Morse L-BFGS on the same bulk file reproduces at 52, so the gap is the pele memory and step, not the shared Morse parameters.",
    "morse_sd": "eOn Morse steepest descent scores 7038 with 19 failures. The archive is 196 with 0. Morse L-BFGS, CG, and FIRE on the same client reproduce, so the kernel is not the split.",
    "improved_unset": "The r2549 template turns improved on. The 2014 ini left that key unset. The mean lands outside 10 percent.",
    "sella_trust": "Sella 2.6.0 with the archived trust-region numbers (eta, gamma, delta0, rho, sigma) is the parameter_source=archive row. It misses the 2014 mean. The constructor-default row is a different setup.",
    "heptamer_cg": "The 3.5.0 file already stores the per-image mean, 142 against the archive 120. L-BFGS, FIRE, and QM on the same 8-image spring-5 climb reproduce, so the image count is not the split.",
    "de_exclusion": "The 3.5.0 doubly-nudged file is the per-band mean under the 2500-call exclusion (708, 5 failures). The archive mean is 861. The client total is a different counter and is also outside 10 percent.",
    "eon_saddle_count": (
        "The classic dimer campaign (improved false, the 2014 key) has fraction_connected 0.692 against 0.746, inside 10 percent, "
        "but force calls 6.97e6 against 3.78e6 and unique saddles 91.7 against 109.2. "
        "The archive standard error on the force calls is 5.27e5. "
        "This eOn AKMC counter is not the harness RMSD link test."
    ),
    "saddle_novelty": (
        "The harness link and novelty test uses Cartesian RMSD 0.5 over the full island. "
        "Distinct saddles collapse: benchmark_unique_saddles is 0, 0.7, and 0.5 against the archive 115, 143, and 130. "
        "benchmark_fraction_connected stays near 1 while the v2 connected fraction is about 0. "
        "The per-search cap of 5064 then puts every repetition on the PAR-2 cap."
    ),
    "pele_bh_defaults": (
        "The archived driver is the dangling path bin/js850/globalopt.py. "
        "The harness uses pele LJCluster defaults (temperature 1, step 0.6, LBFGS memory 4) on unshifted cutoff 40. "
        "The mean is 688720 against 521595."
    ),
    "htst_al": "The r2549 Al dimer with the archived rotation string (CG, max 50, tol 1) misses barrier_force_calls and rate_force_calls by about 30 percent. Pt dimer CG with max 10 reproduces all four hTST counts.",
    "precon_hessian_index": (
        "ASE 3.29 PreconLBFGS with the archived opt.py setup (Exp A 3, maxstep 0.1, ASE fmax 0.01) "
        "scores 2605 against 2686, inside 10 percent. "
        "The script uses a tsase LJ cutoff of 35. This row uses the harness cutoff 40. "
        "report.sh counts a failure only when a written total_force_calls is at least 10000. "
        "The harness records 10 failures, each a Hessian index of 1 after the maximum force is under 0.01. "
        "The Wilson interval on 0 failures out of 1000 is about 0 to 3.83."
    ),
    "linesearch_exceptions": (
        "ASE 3.29 LBFGSLineSearch with the archived opt.py arguments (maxstep 0.2, alpha 1/0.001, memory 100) "
        "scores 410 against 417, inside 10 percent. "
        "report.sh counts a start as failed only when a written total_force_calls is at least 10000, and the benchmark.dat label says M=5 while opt.py sets memory 100. "
        "The harness records 7 failures: 5 RuntimeError LineSearch failed (ASE 3.8.2 already raises that; ASE 3.6.0 is not on PyPI) "
        "and 2 Hessian index 1 after the Euclidean force norm is under 0.01. "
        "The Wilson interval on 0 failures out of 1000 is about 0 to 3.83."
    ),
    "akmc_vs_langevin": (
        "The archive rows are Langevin parallel replica. The harness rows are AKMC. "
        "On infra/aladatom-repro, eon/aladatom-akmc true_rate 3.597591452e9 is 14.1 times 2.542941e8, "
        "and eon/standard-aladatom-akmc true_rate 3.80613e7 is 45.5 times 8.359e5."
    ),
}


def row_id(row: dict) -> str:
    return f"{row['suite']}/{row['problem']}/{row['entry']}"


def within_mean(archive: float, rerun: float, se: float | None) -> bool:
    """True when the rerun is inside one standard error or 10 percent."""
    if archive == 0:
        close = rerun == 0
    else:
        close = abs(rerun - archive) / abs(archive) <= RELATIVE + 1e-12
    if se is None:
        return close
    return close or abs(rerun - archive) <= float(se) + 1e-9


def wilson_count_bounds(failures: float, sample: int, z: float = Z) -> tuple[float, float]:
    """Wilson interval on the archived failure count, in counts."""
    if sample <= 0:
        return 0.0, 0.0
    proportion = failures / sample
    z2 = z * z
    denominator = 1.0 + z2 / sample
    center = (proportion + z2 / (2.0 * sample)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / sample + z2 / (4.0 * sample * sample)) / denominator
    low = max(0.0, center - margin) * sample
    high = min(1.0, center + margin) * sample
    return low, high


def failure_inside(archive_failures: float, rerun_failures: float, sample: int) -> bool:
    low, high = wilson_count_bounds(archive_failures, sample)
    return low - 1e-9 <= rerun_failures <= high + 1e-9


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


def standard_error(fields: dict[str, str], metric: str, archive: float) -> float | None:
    if metric == "force_calls":
        found = _float(fields.get("force_calls_mean_error"))
        if found is not None:
            return found
    if metric == "true_rate":
        found = _float(fields.get("uncertainty") or fields.get("uncertanity"))
        if found is not None:
            return found
        percent = _float(fields.get("percent_uncertainty"))
        if percent is not None:
            return abs(archive) * percent / 100.0
    return None


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


def judge_checks(checks: list[dict], failure: dict | None) -> str:
    if not checks:
        return "not_rerun"
    ok = all(item["ok"] for item in checks)
    if failure is not None and not failure["ok"]:
        ok = False
    return "reproduced" if ok else "not_reproduced"


def build_checks(
    archived_fields: dict[str, str],
    rerun_fields: dict[str, str],
    names: list[str],
    value_from: dict[str, str] | None,
) -> list[dict]:
    checks = []
    renaming = value_from or {}
    for name in names:
        archive = _float(archived_fields.get(name))
        source = renaming.get(name, name)
        rerun = _float(rerun_fields.get(source))
        if archive is None or rerun is None:
            continue
        error = standard_error(archived_fields, name, archive)
        checks.append(
            {
                "metric": name,
                "archive": archive,
                "rerun": rerun,
                "se": error,
                "ok": within_mean(archive, rerun, error),
            }
        )
    return checks


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
    local = root / path
    if local.is_file():
        return local.read_text(encoding="utf-8")
    if branch == "main":
        return local.read_text(encoding="utf-8")
    import subprocess

    return subprocess.check_output(
        ["git", "show", f"origin/{branch}:{path}"],
        cwd=root,
        text=True,
    )


def _sha(branch: str) -> str:
    if branch == "cells/fill-not-rerun" and len(SHA.get(branch, "")) < 12:
        import subprocess

        SHA[branch] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
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


class Hold:
    """Exact-setup reruns that must stay off the ranked table."""

    def __init__(self) -> None:
        self.templates: set[tuple[str, str]] = set()
        self.paths: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self.templates or self.paths)


def holdback(root: Path) -> Hold:
    """Templates and campaign paths whose archive rerun missed the tolerance.

    ``hold`` entries are included for every ledger row. A reproduced
    per-image file can still keep a raw-total template off the table.
    """
    found = Hold()
    for row in load_ledger(root / "oracle" / "reproduction.jsonl"):
        rerun = row.get("rerun") or {}
        problem = f"{row['suite']}/{row['problem']}"
        template = str(rerun.get("template") or "")
        if row.get("status") == "not_reproduced":
            if template:
                found.templates.add((problem, template))
            rel = str(rerun.get("path") or "")
            if rel:
                found.paths.add(rel)
        for extra in rerun.get("hold") or []:
            found.templates.add((problem, str(extra)))
    return found


def _local_fields(root: Path, rerun: dict) -> dict[str, str] | None:
    path = root / str(rerun.get("path") or "")
    if path.is_file() and rerun.get("kind", "dat") == "dat":
        return parse_dat(path.read_text(encoding="utf-8"))
    return None


def baseline_errors(root: Path) -> list[str]:
    """Paper-baseline rows whose live campaign misses the archive tolerance."""
    errors = []
    ledger = load_ledger(root / "oracle" / "reproduction.jsonl")
    marked = []
    for row in ledger:
        rerun = row.get("rerun") or {}
        if not rerun.get("paper_baseline"):
            continue
        ident = row_id(row)
        marked.append(rerun.get("template"))
        if row.get("status") != "reproduced":
            errors.append(f"{ident} is paper_baseline and {row.get('status')}")
            continue
        fields = _local_fields(root, rerun)
        if fields is None:
            continue
        names = [item["metric"] for item in row.get("checks") or []]
        fresh = build_checks(
            {item["metric"]: str(item["archive"]) for item in row["checks"]},
            fields,
            names,
            None,
        )
        # The archived side of a stored check has no mean-error key. Carry se.
        by_metric = {item["metric"]: item for item in row["checks"]}
        for item in fresh:
            stored = by_metric.get(item["metric"])
            if stored is None:
                continue
            item["se"] = stored.get("se")
            item["ok"] = within_mean(item["archive"], item["rerun"], item["se"])
            if not item["ok"]:
                errors.append(
                    f"{ident} {item['metric']} is {item['rerun']} outside the archive {item['archive']}"
                )
        failure = row.get("failure_check")
        if failure and "nfailed" in fields and failure.get("sample"):
            rerun_failures = _float(fields.get("nfailed"))
            if rerun_failures is not None and not failure_inside(
                float(failure["archive"]), rerun_failures, int(failure["sample"])
            ):
                errors.append(f"{ident} nfailed {rerun_failures} is outside the archived interval")
    templates = root / "templates"
    if templates.is_dir():
        import tomllib

        known = {item for item in marked if item}
        for path in templates.rglob("*.toml"):
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            if data.get("paper_baseline") and data.get("id") not in known:
                errors.append(f"{data.get('id')} sets paper_baseline and is not a reproduced ledger row")
    return errors


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


def reproduction_errors(root: Path) -> list[str]:
    ledger_path = root / "oracle" / "reproduction.jsonl"
    if not ledger_path.is_file():
        return ["oracle/reproduction.jsonl is missing"]
    from .check import load_oracle

    oracle = load_oracle(root / "oracle" / "entries.jsonl")
    ledger = load_ledger(ledger_path)
    errors = []
    if len(ledger) != len(oracle):
        errors.append(f"reproduction ledger has {len(ledger)} rows and the oracle has {len(oracle)}")
    oracle_ids = [row_id(row) for row in oracle]
    ledger_ids = [row_id(row) for row in ledger]
    if oracle_ids != ledger_ids:
        errors.append("reproduction ledger is not in oracle order")
    for row in ledger:
        status = judge_checks(row.get("checks") or [], row.get("failure_check") if row.get("rerun") else None)
        if row.get("rerun") is None:
            status = "not_rerun"
        if status != row.get("status"):
            errors.append(f"{row_id(row)} stored {row.get('status')} and the checks say {status}")
        if row.get("rerun", {}) and row["rerun"].get("paper_baseline") and row.get("status") != "reproduced":
            errors.append(f"{row_id(row)} paper_baseline is not reproduced")
    errors.extend(baseline_errors(root))
    errors.extend(large_endpoint_files(root))
    return errors


def _gap(checks: list[dict], failure: dict | None) -> str:
    parts = []
    for item in checks:
        if item["ok"]:
            continue
        archive = item["archive"]
        rerun = item["rerun"]
        if archive:
            relative = abs(rerun - archive) / abs(archive)
            parts.append(f"{item['metric']} {rerun:g} vs {archive:g} ({relative:.1%})")
        else:
            parts.append(f"{item['metric']} {rerun:g} vs {archive:g}")
    if failure and not failure["ok"]:
        parts.append(
            f"nfailed {failure['rerun']:g} vs {failure['archive']:g} "
            f"(interval {failure['low']:.2f} to {failure['high']:.2f})"
        )
    return "; ".join(parts)


def build_rows(root: Path) -> list[dict]:
    from .check import load_oracle

    oracle = {row_id(row): row for row in load_oracle(root / "oracle" / "entries.jsonl")}
    matches = {item["key"]: item for item in MATCHES}
    built = []
    for ident, entry in oracle.items():
        fields = entry["fields"]
        match = matches.get(ident)
        sample = SAMPLE_N.get(f"{entry['suite']}/{entry['problem']}")
        record = {
            "suite": entry["suite"],
            "problem": entry["problem"],
            "entry": entry["entry"],
            "archive_path": entry["path"],
            "commit": entry["commit"],
            "in_paper": ident in PAPER,
            "source": "softsaddle-2019" if entry["entry"].startswith("adaptive_HSphere") else ("paper" if ident in PAPER else "archive"),
            "archived_setup": {
                "code": fields.get("code", ""),
                "algorithm": fields.get("algorithm", ""),
                "code_version": fields.get("code_version", ""),
            },
            "rerun": None,
            "checks": [],
            "failure_check": None,
            "status": "not_rerun",
            "gap": "",
            "cause": "",
            "cause_detail": "",
        }
        paper = PAPER_METRIC.get(ident)
        if paper:
            record["paper_metric"] = {"name": paper[0], "value": paper[1]}
        if match is None:
            if ident.startswith("accelerated-md/"):
                record["cause"] = "akmc_vs_langevin"
                record["cause_detail"] = CAUSE_TEXT["akmc_vs_langevin"]
            built.append(record)
            continue
        kind = match.get("kind", "dat")
        text = read_campaign(root, match["branch"], match["path"])
        rerun_fields = screen_fields(text) if kind == "screen" else parse_dat(text)
        names = metrics_for(fields, kind, match.get("metrics"))
        checks = build_checks(fields, rerun_fields, names, match.get("value_from"))
        failure = None
        archive_failures = _float(fields.get("nfailed"))
        rerun_failures = _float(rerun_fields.get("nfailed"))
        if archive_failures is not None and rerun_failures is not None and sample:
            low, high = wilson_count_bounds(archive_failures, sample)
            failure = {
                "archive": archive_failures,
                "rerun": rerun_failures,
                "sample": sample,
                "low": low,
                "high": high,
                "ok": failure_inside(archive_failures, rerun_failures, sample),
            }
        status = judge_checks(checks, failure)
        named = match.get("cause", "")
        cause = named if status == "not_reproduced" else ""
        record["checks"] = checks
        record["failure_check"] = failure
        record["status"] = status
        record["gap"] = _gap(checks, failure) if status == "not_reproduced" else ""
        record["cause"] = cause
        record["cause_detail"] = CAUSE_TEXT.get(cause, "") if cause else ""
        record["note"] = CAUSE_TEXT.get(named, "") if status == "reproduced" and named else ""
        baseline = status == "reproduced"
        # The ranked harness-stop mean is not the 2014 pele score.
        if kind == "screen":
            baseline = status == "reproduced"
        rerun = {
            "template": match["template"],
            "branch": match["branch"],
            "sha": _sha(match["branch"]),
            "path": match["path"],
            "kind": kind,
            "paper_baseline": baseline,
            "baseline_metric": rerun_fields.get("baseline_metric", names[0] if names else "force_calls"),
            "hold": match.get("hold") or [],
        }
        if kind == "screen":
            rerun["stop2014_calls"] = _float(rerun_fields.get("force_calls"))
            rerun["harness_stop_mean"] = _float(rerun_fields.get("harness_stop_mean"))
            rerun["baseline_metric"] = "stop2014_calls"
        if match.get("extra"):
            rerun.update(match["extra"])
        record["rerun"] = rerun
        built.append(record)
    return built


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.6g}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6g}"


def render_markdown(rows: list[dict]) -> str:
    counts = {"reproduced": 0, "not_reproduced": 0, "not_rerun": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = [
        "# Reproduction",
        "",
        "Archive commit `0f26b796`. A rerun matches when each archived metric is within one archived standard error or 10%, and `nfailed` is inside the Wilson interval of the archived failure count.",
        "",
        f"{counts['reproduced']} reproduced, {counts['not_reproduced']} not reproduced, {counts['not_rerun']} not rerun. {len(rows)} archive rows.",
        "",
        "The pele transition-state paper score is the 2014 stop: break at the first RMS below tolerance, then one eigenvector search, transverse screen calls subtracted. Pt-island ts-ffd is 88.55 against 88 and is the paper baseline. The ranked harness-stop means are 111.27, 254.48, and 335.06 and are not that score.",
        "",
        "OPTIM LJ38 L-BFGS tuned: the archived integer is 176 and the unrounded harness mean is 175.537.",
        "",
        "EON FIRE in JCTC Table 1 is 645. The archive file and the rerun are both 731.",
        "",
        "Table 10 prints EON Lanczos 165. The archive file is 325. CI-NEB(3) prints 3 failures; the archive file has 0.",
        "",
        "Endpoint minima are free-atom float32 deltas against the reactant, compressed. A file over 1 MB is a git-LFS pointer, not a loose blob.",
        "",
    ]
    current = ""
    for row in rows:
        suite = row["suite"]
        if suite != current:
            if current:
                lines.append("")
            current = suite
            lines.append(f"## {suite}")
            lines.append("")
            lines.append("| Row | Archive | Rerun | Status |")
            lines.append("| --- | --- | --- | --- |")
        ident = f"{row['problem']}/{row['entry']}"
        if row["checks"]:
            archive = ", ".join(f"{item['metric']} {_fmt(item['archive'])}" for item in row["checks"])
            rerun = ", ".join(f"{_fmt(item['rerun'])}" for item in row["checks"])
        else:
            archive = "—"
            rerun = "—"
        status = row["status"].replace("_", " ")
        if row.get("cause"):
            status = f"{status}; {row['cause']}"
        if row.get("note"):
            status = f"{status}; residual"
        lines.append(f"| {ident} | {archive} | {rerun} | {status} |")
    lines.append("")
    lines.append("## Residuals inside the band")
    lines.append("")
    for row in rows:
        if row.get("note"):
            lines.append(f"- {row['suite']}/{row['problem']}/{row['entry']}: {row['note']}")
    lines.append("")
    lines.append("## Causes")
    lines.append("")
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if not row.get("cause"):
            continue
        grouped.setdefault(row["cause"], []).append(f"{row['suite']}/{row['problem']}/{row['entry']}")
    for cause, idents in grouped.items():
        lines.append(f"### {cause}")
        lines.append("")
        lines.append(CAUSE_TEXT.get(cause, ""))
        lines.append("")
        for ident in idents:
            lines.append(f"- {ident}")
        lines.append("")
    lines.append("Rows with no exact-setup campaign stay not rerun and are not given a cause. AKMC on aladatom is a different method from Langevin parallel replica; the ratios above are that comparison, and the archive rows stay not rerun.")
    lines.append("")
    return "\n".join(lines)


def write_ledger(root: Path) -> list[dict]:
    rows = build_rows(root)
    path = root / "oracle" / "reproduction.jsonl"
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")
    (root / "docs" / "reproduction.md").write_text(render_markdown(rows), encoding="utf-8")
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = write_ledger(root)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(counts)
    for row in rows:
        if row["status"] == "not_reproduced":
            print(f"{row_id(row)} {row['cause']} {row['gap']}")


if __name__ == "__main__":
    main()
