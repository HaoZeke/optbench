# Reproduction

Archive commit `0f26b796`. Each row is the archive. A latest-version rerun is listed beside it. A gap is the relative difference, as a percentage.

46 rows on the archived metrics, 30 rows with a gap, 75 rows with no rerun. 151 archive rows.

The pele transition-state score is the 2014 stop: break at the first RMS below tolerance, then one eigenvector search, transverse screen calls subtracted. Pt-island ts-ffd is 88.55 against 88. The ranked harness-stop means are 111.27, 254.48, and 335.06.

OPTIM LJ38 L-BFGS tuned: the archived integer is 176 and the unrounded harness mean is 175.537.

EON FIRE in JCTC Table 1 is 645. The archive file and the rerun are both 731.

Table 10 prints EON Lanczos 165. The archive file is 325. CI-NEB(3) prints 3 failures; the archive file has 0.

Endpoint minima are free-atom float32 deltas against the reactant, compressed. A file over 1 MB is a git-LFS pointer, not a loose blob.

## accelerated-md

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| aladatom/true_standard/eon-pr-langevin | — | — | akmc_vs_langevin |
| standard_aladatom/eon-md-langevin | — | — | akmc_vs_langevin |

## double-ended-saddle-search

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj38/eon-ci-neb | force_calls 827 | 762 | force_calls 762 vs 827 (7.9%) |
| lj38/eon-ci-swdneb | force_calls 861 | 708 | force_calls 708 vs 861 (17.8%); de_exclusion |
| lj38/eon-dimer | force_calls 337 | 377 | force_calls 377 vs 337 (11.9%); improved_unset |
| lj38/eon-lanczos | force_calls 364 | 342 | force_calls 342 vs 364 (6.0%) |
| lj38/eon-lanczos-improved | force_calls 287 | 259 | force_calls 259 vs 287 (9.8%) |
| lj38/optim | force_calls 131 | 147 | force_calls 147 vs 131 (12.2%); optim_pes_counter |
| lj38/optim-cl-string | — | — | — |
| lj38/pele | — | — | — |
| lj38_2ts_separation/optim | force_calls 2885 | 3454 | force_calls 3454 vs 2885 (19.7%); optim_pes_counter |
| lj38_2ts_separation/pele | — | — | — |
| pt-island/eon-ci-neb | force_calls 248 | 226 | force_calls 226 vs 248 (8.9%) |
| pt-island/eon-ci-neb-5 | — | — | — |
| pt-island/eon-ci-neb-single | — | — | — |
| pt-island/eon-ci-swdneb | force_calls 256 | 246 | force_calls 246 vs 256 (3.9%) |
| pt-island/eon-dimer | force_calls 182 | 192 | force_calls 192 vs 182 (5.5%) |
| pt-island/eon-lanczos | force_calls 325 | 222 | force_calls 222 vs 325 (31.7%); pt_lanczos |
| pt-island/eon-lanczos-improved | force_calls 222 | 181 | force_calls 181 vs 222 (18.5%); pt_lanczos |
| pt-island/optim | force_calls 103 | 138 | force_calls 138 vs 103 (34.0%); optim_pt_units |
| pt-island/optim-6 | — | — | — |
| pt-island/optim-cl-string | — | — | — |
| pt-island/pele | — | — | — |

## globalopt

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj100-binary/eon-basinhopping | — | — | — |
| lj100-binary/gmin | — | — | — |
| lj150/gmin-nosym | — | — | — |
| lj150/gmin-symmetrise | — | — | — |
| lj38/eon-basinhopping | force_calls 508011 | 503146 | force_calls 503146 vs 508011 (1.0%) |
| lj38/gmin-csm | — | — | — |
| lj38/gmin-nosym | — | — | — |
| lj38/gmin-symmetrise | — | — | — |
| lj38/pele-basinhopping | force_calls 521595 | 688720 | force_calls 688720 vs 521595 (32.0%); pele_bh_defaults |
| lj75/gmin | — | — | — |
| lj98/gmin-nosym | — | — | — |
| lj98/gmin-symmetrise | — | — | — |
| tip4p10/gmin | — | — | — |
| tip4p10/gmin-tbp | — | — | — |
| tip4p15/gmin | — | — | — |
| tip4p15/gmin-tbp | — | — | — |
| tip4p20/gmin | — | — | — |
| tip4p20/gmin-tbp | — | — | — |

## hTST

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| Al/eon-dimer-cg | barrier_force_calls 28870, barrier_jobs 48, rate_force_calls 897900, rate_jobs 1511 | 19660, 50, 620600, 1600 | barrier_force_calls 19660 vs 28870 (31.9%); barrier_jobs 50 vs 48 (4.2%); rate_force_calls 620600 vs 897900 (30.9%); rate_jobs 1600 vs 1511 (5.9%); htst_al |
| Al/eon-dimer-cg2 | — | — | — |
| Pt/alpha_dimer-qm | — | — | — |
| Pt/eon-dimer-cg | barrier_force_calls 496000, barrier_jobs 1050, rate_force_calls 25630, rate_jobs 54 | 506000, 1062, 27160, 57 | barrier_force_calls 506000 vs 496000 (2.0%); barrier_jobs 1062 vs 1050 (1.1%); rate_force_calls 27160 vs 25630 (6.0%); rate_jobs 57 vs 54 (5.6%) |
| Pt/eon-dimer-cg2 | — | — | — |
| Pt/eon-dimer-lbfgs | — | — | — |
| Pt/eon-dimer-lbfgs2 | — | — | — |
| Pt/eon-lanczos-cg | barrier_force_calls 673700, barrier_jobs 1419, rate_force_calls 25860, rate_jobs 54 | 1.695e+06, 1916, 50440, 57 | barrier_force_calls 1.695e+06 vs 673700 (151.6%); barrier_jobs 1916 vs 1419 (35.0%); rate_force_calls 50440 vs 25860 (95.1%); rate_jobs 57 vs 54 (5.6%); htst_pt_lanczos_stock |
| Pt/eon-lanczos-cg2 | — | — | — |
| Pt/eon-lanczos-lbfgs | — | — | — |
| Pt/eon-lanczos-lbfgs2 | — | — | — |
| Pt/quench_dimer-qm | — | — | — |
| Pt/tsase-dimer-qm | — | — | — |

## lowest-eigenvector

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj38/eon-dimer-cg | force_calls 30 | 30 | force_calls 30 vs 30 (0.0%) |
| lj38/eon-dimer-lbfgs | force_calls 28 | 28 | force_calls 28 vs 28 (0.0%) |
| lj38/eon-lanczos | force_calls 28 | 26 | force_calls 26 vs 28 (7.1%) |
| lj38/optim-first-order | force_calls 25 | 61 | force_calls 61 vs 25 (144.0%); optim_rr_defaults |
| lj38/optim-second-order | — | — | — |
| lj38/pele-first-order | force_calls 25 | 25 | force_calls 25 vs 25 (0.0%) |
| lj38/pele-second-order | force_calls 47 | 47 | force_calls 47 vs 47 (0.0%) |
| lj38/sella | force_calls 17.8 | 17.8 | force_calls 17.8 vs 17.8 (0.0%) |
| lj38/tsase-dimer-bfgs | — | — | — |
| lj38/tsase-dimer-cg | — | — | — |
| lj38/tsase-lanczos | — | — | — |

## minimization

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj38/ase-BFGS | force_calls 463 | 461 | force_calls 461 vs 463 (0.4%) |
| lj38/ase-bfgslinesearch | force_calls 500 | 506 | force_calls 506 vs 500 (1.2%) |
| lj38/ase-fire | force_calls 656 | 720 | force_calls 720 vs 656 (9.8%) |
| lj38/ase-lbfgs | force_calls 355 | 356 | force_calls 356 vs 355 (0.3%) |
| lj38/ase-lbfgslinesearch | force_calls 417 | 410 | force_calls 410 vs 417 (1.7%); linesearch_exceptions |
| lj38/ase-preconlbfgs | force_calls 2686 | 2605 | force_calls 2605 vs 2686 (3.0%); precon_hessian_index |
| lj38/eon-cg | force_calls 453 | 454 | force_calls 454 vs 453 (0.2%) |
| lj38/eon-fire | force_calls 731 | 731 | force_calls 731 vs 731 (0.0%) |
| lj38/eon-lbfgs | force_calls 181 | 181 | force_calls 181 vs 181 (0.0%) |
| lj38/eon-qm | force_calls 3523 | 3514 | force_calls 3514 vs 3523 (0.3%) |
| lj38/eon-sd | force_calls 4901 | 4901 | force_calls 4901 vs 4901 (0.0%) |
| lj38/eon-sd-twopoint | force_calls 539 | 532 | force_calls 532 vs 539 (1.3%) |
| lj38/newton-method | — | — | — |
| lj38/optim | force_calls 176 | 176 | force_calls 176 vs 176 (0.0%) |
| lj38/pele-fire | force_calls 709 | 1264 | force_calls 1264 vs 709 (78.3%); pele_fire |
| lj38/pele-lbfgs-M1-maxstep0.1 | force_calls 287 | 286 | force_calls 286 vs 287 (0.3%) |
| lj38/pele-lbfgs-M10-maxstep0.1 | force_calls 235 | 235 | force_calls 235 vs 235 (0.0%) |
| lj38/pele-lbfgs-M100-maxstep0.2 | force_calls 179 | 178 | force_calls 178 vs 179 (0.6%) |
| lj38/pele-lbfgs-M20-maxstep0.1 | force_calls 222 | 223 | force_calls 223 vs 222 (0.5%) |
| lj38/pele-lbfgs-M4-maxstep0.05 | force_calls 379 | 381 | force_calls 381 vs 379 (0.5%) |
| lj38/pele-lbfgs-M4-maxstep0.1 | force_calls 249 | 248 | force_calls 248 vs 249 (0.4%) |
| lj38/pele-lbfgs-M4-maxstep0.2 | force_calls 230 | 228 | force_calls 228 vs 230 (0.9%) |
| lj38/scipy-lbfgs | — | — | — |
| lj38/tsase-sdlbfgs | — | — | — |
| morse-bulk/ase-fire | force_calls 147.47 | 113 | force_calls 113 vs 147.47 (23.4%); ase_morse_script |
| morse-bulk/ase-lbfgs | force_calls 53.8 | 32 | force_calls 32 vs 53.8 (40.5%); ase_morse_script |
| morse-bulk/ase-preconLBFGS | force_calls 34 | 29 | force_calls 29 vs 34 (14.7%); ase_morse_script |
| morse-bulk/eon-cg | force_calls 106 | 106 | force_calls 106 vs 106 (0.0%) |
| morse-bulk/eon-fire | force_calls 156 | 156 | force_calls 156 vs 156 (0.0%) |
| morse-bulk/eon-lbfgs | force_calls 52 | 52 | force_calls 52 vs 52 (0.0%) |
| morse-bulk/eon-sd | force_calls 196 | 7038 | force_calls 7038 vs 196 (3490.8%); morse_sd |
| morse-bulk/eon-sd-twopoint | — | — | — |
| morse-bulk/optim | force_calls 46 | 27 | force_calls 27 vs 46 (41.3%); optim_morse_memory |
| morse-bulk/pele-lbfgs | force_calls 51 | 92 | force_calls 92 vs 51 (80.4%); pele_morse |
| morse-bulk/tsase-sdlbfgs | — | — | — |

## mlopt

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| C2H4/pyamff-adam10-7-fcoeff0 | — | — | — |
| C2H4/pyamff-adam10-7-fcoeff0_1 | — | — | — |
| C2H4/pyamff-lbfgs10-7-fcoeff0 | — | — | — |
| C2H4/pyamff-lbfgs10-7-fcoeff0_1 | — | — | — |
| C2H4/pyamff-rprop10-7-fcoeff0 | — | — | — |
| C2H4/pyamff-rprop10-7-fcoeff0_1 | — | — | — |
| emt-pt38/pyamff-adam10-7-fcoeff0 | — | — | — |
| emt-pt38/pyamff-adam10-7-fcoeff0_1 | — | — | — |
| emt-pt38/pyamff-lbfgs10-7-fcoeff0 | — | — | — |
| emt-pt38/pyamff-lbfgs10-7-fcoeff0_1 | — | — | — |
| emt-pt38/pyamff-rprop10-7-fcoeff0 | — | — | — |
| emt-pt38/pyamff-rprop10-7-fcoeff0_1 | — | — | — |

## neb

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| Pt-heptamer/eon-cg | force_calls 120 | 142 | force_calls 142 vs 120 (18.3%); heptamer_cg |
| Pt-heptamer/eon-fire | force_calls 86 | 85 | force_calls 85 vs 86 (1.2%) |
| Pt-heptamer/eon-lbfgs | force_calls 42 | 40 | force_calls 40 vs 42 (4.8%) |
| Pt-heptamer/eon-qm | force_calls 164 | 165 | force_calls 165 vs 164 (0.6%) |
| Pt-heptamer/gman-cg | — | — | — |
| Pt-heptamer/gman-fire | — | — | — |
| Pt-heptamer/gman-glbfgs-hess | — | — | — |
| Pt-heptamer/gman-glbfgs-line | — | — | — |
| Pt-heptamer/gman-lbfgs-hess | — | — | — |
| Pt-heptamer/gman-lbfgs-line | — | — | — |
| Pt-heptamer/gman-qm | — | — | — |
| Pt-heptamer/gman-sd | — | — | — |

## saddle-search

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| pt-island/adaptive_HSphere-lbfgs-davidson-sm50 | force_calls 423491, unique_saddles 114.9, fraction_connected 0.8 | 5.2987e+06, 0, 1 | force_calls 5.2987e+06 vs 423491 (1151.2%); unique_saddles 0 vs 114.9 (100.0%); fraction_connected 1 vs 0.8 (25.0%); saddle_novelty |
| pt-island/adaptive_HSphere-lbfgs-lanczos-sm0 | force_calls 1.00343e+06, unique_saddles 130.4, fraction_connected 0.9 | 1.92665e+06, 0.5, 0.9682 | force_calls 1.92665e+06 vs 1.00343e+06 (92.0%); unique_saddles 0.5 vs 130.4 (99.6%); fraction_connected 0.9682 vs 0.9 (7.6%); saddle_novelty |
| pt-island/adaptive_HSphere-lbfgs-lanczos-sm50 | force_calls 814519, unique_saddles 142.6, fraction_connected 0.9 | 1.58264e+06, 0.7, 0.93584 | force_calls 1.58264e+06 vs 814519 (94.3%); unique_saddles 0.7 vs 142.6 (99.5%); fraction_connected 0.93584 vs 0.9 (4.0%); saddle_novelty |
| pt-island/eon-dimer | force_calls 3.77888e+06, unique_saddles 109.222, fraction_connected 0.746017 | 6.97488e+06, 91.7, 0.691952 | force_calls 6.97488e+06 vs 3.77888e+06 (84.6%); unique_saddles 91.7 vs 109.222 (16.0%); fraction_connected 0.691952 vs 0.746017 (7.2%); eon_saddle_count |
| pt-island/eon-dimer-cg-20-rotations | — | — | — |
| pt-island/eon-dimer-cg-20-rotations-20-bowl_breakout | — | — | — |
| pt-island/eon-dimer-cg-5-rotations-20-bowl_breakout | — | — | — |
| pt-island/eon-dimer-lbfgs-20-rotations | — | — | — |
| pt-island/eon-dimer-lbfgs-20-rotations-20-bowl_breakout | — | — | — |
| pt-island/eon-dimer-lbfgs-5-rotations | — | — | — |
| pt-island/eon-dimer-lbfgs-5-rotations-20-bowl_breakout | — | — | — |
| pt-island/eon-md | — | — | — |

## tsopt

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| LJ38/eon-dimer | force_calls 528 | 523 | force_calls 523 vs 528 (0.9%) |
| LJ38/eon-lanczos | force_calls 321 | 293 | force_calls 293 vs 321 (8.7%) |
| LJ38/eon-lanczos-improved | force_calls 237 | 271 | force_calls 271 vs 237 (14.3%); improved_unset |
| LJ38/optim_cd | — | — | — |
| LJ38/optim_ffd | force_calls 145 | 489 | force_calls 489 vs 145 (237.2%); optim_lj38_ts |
| LJ38/pele_cd | force_calls 254 | 263.52 | force_calls 263.52 vs 254 (3.7%) |
| LJ38/pele_ffd | force_calls 192 | 207.51 | force_calls 207.51 vs 192 (8.1%) |
| LJ38/sella | force_calls 69.55 | 93 | force_calls 93 vs 69.55 (33.7%); sella_trust |
| pt-island/eon-dimer | force_calls 116 | 115 | force_calls 115 vs 116 (0.9%) |
| pt-island/eon-lanczos | force_calls 143 | 146 | force_calls 146 vs 143 (2.1%) |
| pt-island/eon-lanczos-improved | force_calls 106 | 106 | force_calls 106 vs 106 (0.0%) |
| pt-island/optim_cd | — | — | — |
| pt-island/optim_ffd | force_calls 71 | 170 | force_calls 170 vs 71 (139.4%); optim_pt_units |
| pt-island/pele | force_calls 88 | 88.551 | force_calls 88.551 vs 88 (0.6%) |
| pt-island/sella | force_calls 53 | 88 | force_calls 88 vs 53 (66.0%); sella_trust |

## Notes

- tsopt/LJ38/pele_cd: 263.52 vs 254 (3.7%). The same xyz-versus-con starting set as ts-ffd. findTS_local.py for the CD row does not pass orthogZeroEigs; the harness template sets orthog translations-rotations. This 2014-stop mean is the archived score. The harness-stop mean is 335.06.
- tsopt/LJ38/pele_ffd: 207.51 vs 192 (8.1%). Archive run.sh unpacks coords-xyz.tar.gz (200 xyz, max pair about 4.4). The harness problem points at coords-con.tar.gz (max pair about 37). Kabsch RMS is about 6.33, so the sets are not the same geometry. Neither set has a pair past cutoff 40, and both drivers use RMS tolerance 1e-3/sqrt(114). The 8.1% gap is the starting set. This 2014-stop mean is the archived score. The harness-stop mean is 254.48.

## Causes

### akmc_vs_langevin

The archive rows are Langevin parallel replica. The harness rows are AKMC. On infra/aladatom-repro, eon/aladatom-akmc true_rate 3.597591452e9 is 14.1 times 2.542941e8, and eon/standard-aladatom-akmc true_rate 3.80613e7 is 45.5 times 8.359e5.

- accelerated-md/aladatom/true_standard/eon-pr-langevin
- accelerated-md/standard_aladatom/eon-md-langevin

### de_exclusion

The 3.5.0 doubly-nudged file is the per-band mean under the 2500-call exclusion (708, 5 failures). The archive mean is 861 (17.8%). The client total is a different counter.

- double-ended-saddle-search/lj38/eon-ci-swdneb

### improved_unset

The r2549 template turns improved on. The 2014 ini left that key unset. The mean is 377 against 337 (11.9%).

- double-ended-saddle-search/lj38/eon-dimer
- tsopt/LJ38/eon-lanczos-improved

### optim_pes_counter

The template note records the 2014 mean as OPTIM's own energy-plus-gradient calls. The harness score is pes.evaluate on the harness potential.

- double-ended-saddle-search/lj38/optim
- double-ended-saddle-search/lj38_2ts_separation/optim

### pt_lanczos

The svn template uses the archived Lanczos keys (10 iterations, tolerance 0.05) and shifted Morse::force. Dimer and climbing-image on that kernel reproduce. Lanczos does not, and the failure count is 0 against 5 or 8.

- double-ended-saddle-search/pt-island/eon-lanczos
- double-ended-saddle-search/pt-island/eon-lanczos-improved

### optim_pt_units

archive run.sh divides coordinates by r0 = 2.897 and scales rho and rcut into those units. The harness keeps the problem Morse in eV/Angstrom on the unscaled con file.

- double-ended-saddle-search/pt-island/optim
- tsopt/pt-island/optim_ffd

### pele_bh_defaults

The archived driver is the dangling path bin/js850/globalopt.py. The harness uses pele LJCluster defaults (temperature 1, step 0.6, LBFGS memory 4) on unshifted cutoff 40. The mean is 688720 against 521595.

- globalopt/lj38/pele-basinhopping

### htst_al

The r2549 Al dimer with the archived rotation string (CG, max 50, tol 1) misses barrier_force_calls and rate_force_calls by about 30 percent. Pt dimer CG with max 10 reproduces all four hTST counts.

- hTST/Al/eon-dimer-cg

### htst_pt_lanczos_stock

Stock EON SVN r2549, Fortran client, no physics patches. The ini is the archived AKMC setup and the stop is max_atom at 0.001. finite_difference, quit_early, and thermally_accessible_window stay at the r2549 defaults. barrier_jobs 1916 vs 1419 (35.0%), barrier_force_calls 1695000 vs 673700 (151.6%), rate_jobs 57 vs 54 (5.6%), rate_force_calls 50440 vs 25860 (95.1%). eOn 500350d9 is a different code and is ranked on the same problem: barrier_jobs 1720 vs 1419 (21.2%), barrier_force_calls 1468000 vs 673700 (117.9%), rate_jobs 54 vs 54 (0.0%), rate_force_calls 46850 vs 25860 (81.2%).

- hTST/Pt/eon-lanczos-cg

### optim_rr_defaults

The Rayleigh-Ritz template uses updates 4. The archived first-order mean is 25 and the second-order mean is 46. This default scores 61, so it is not either archived order.

- lowest-eigenvector/lj38/optim-first-order

### linesearch_exceptions

ASE 3.29 LBFGSLineSearch with the archived opt.py arguments (maxstep 0.2, alpha 1/0.001, memory 100) scores 410 against 417 (1.7%). report.sh counts a start as failed only when a written total_force_calls is at least 10000, and the benchmark.dat label says M=5 while opt.py sets memory 100. The harness records 7 failures: 5 RuntimeError LineSearch failed (ASE 3.8.2 already raises that; ASE 3.6.0 is not on PyPI) and 2 Hessian index 1 after the Euclidean force norm is under 0.01. The bound on 0 failures out of 1000 is about 0 to 3.83.

- minimization/lj38/ase-lbfgslinesearch

### precon_hessian_index

ASE 3.29 PreconLBFGS with the archived opt.py setup (Exp A 3, maxstep 0.1, ASE fmax 0.01) scores 2605 against 2686 (3.0%). The script uses a tsase LJ cutoff of 35. This row uses the harness cutoff 40. report.sh counts a failure only when a written total_force_calls is at least 10000. The harness records 10 failures, each a Hessian index of 1 after the maximum force is under 0.01. The bound on 0 failures out of 1000 is about 0 to 3.83.

- minimization/lj38/ase-preconlbfgs

### pele_fire

pele FIRE on LJ38 scores 1264 with 7 failures. The archive is 709 with 0 failures. The M100 L-BFGS row on the same build reproduces, so this is the FIRE step, not the LJ kernel.

- minimization/lj38/pele-fire

### ase_morse_script

ASE 3.29 with the archived Morse script (L-BFGS memory 20, maxstep 0.2, alpha 25; FIRE maxstep 0.2; PreconLBFGS Exp A 3) scores 32, 113, and 29 against 53.8, 147.47, and 34. Constructor-default L-BFGS scores 53, next to 53.8, but that is not the archived alpha and memory. eOn Morse L-BFGS and FIRE on the same bulk file reproduce, so the geometry set is not the split.

- minimization/morse-bulk/ase-fire
- minimization/morse-bulk/ase-lbfgs
- minimization/morse-bulk/ase-preconLBFGS

### morse_sd

eOn Morse steepest descent scores 7038 with 19 failures. The archive is 196 with 0. Morse L-BFGS, CG, and FIRE on the same client reproduce, so the kernel is not the split.

- minimization/morse-bulk/eon-sd

### optim_morse_memory

comments.txt sets L-BFGS memory 100 and maximum move 2.0. That rerun scores 27. OPTIM defaults (updates 4) score 46, which is the archived integer, but that is not the comment's setup. The odata preamble was not archived.

- minimization/morse-bulk/optim

### pele_morse

pele Morse L-BFGS scores 92. The archive is 51. eOn Morse L-BFGS on the same bulk file reproduces at 52, so the gap is the pele memory and step, not the shared Morse parameters.

- minimization/morse-bulk/pele-lbfgs

### heptamer_cg

The 3.5.0 file already stores the per-image mean, 142 against the archive 120. L-BFGS, FIRE, and QM on the same 8-image spring-5 climb reproduce, so the image count is not the split.

- neb/Pt-heptamer/eon-cg

### saddle_novelty

The harness link and novelty test uses Cartesian RMSD 0.5 over the full island. Distinct saddles collapse: benchmark_unique_saddles is 0, 0.7, and 0.5 against the archive 115, 143, and 130. benchmark_fraction_connected stays near 1 while the v2 connected fraction is about 0. The per-search cap of 5064 then puts every repetition on the PAR-2 cap.

- saddle-search/pt-island/adaptive_HSphere-lbfgs-davidson-sm50
- saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm0
- saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm50

### eon_saddle_count

The classic dimer campaign (improved false, the 2014 key) has fraction_connected 0.692 against 0.746 (7.2%), but force calls 6.97e6 against 3.78e6 and unique saddles 91.7 against 109.2. The archive standard error on the force calls is 5.27e5. This eOn AKMC counter is not the harness RMSD link test.

- saddle-search/pt-island/eon-dimer

### optim_lj38_ts

Archive run.sh unpacks coords-xyz.tar.gz. The harness uses the problem coords-con set (Kabsch RMS about 6.33). The odata preamble is not in the archive snapshot. The template uses updates 4. The mean is 489 against 145.

- tsopt/LJ38/optim_ffd

### sella_trust

Sella 2.6.0 with the archived trust-region numbers (eta, gamma, delta0, rho, sigma) is the parameter_source=archive row. It misses the 2014 mean. The constructor-default row is a different setup.

- tsopt/LJ38/sella
- tsopt/pt-island/sella

Rows with no exact-setup campaign have no rerun and are not given a cause. AKMC on aladatom is a different method from Langevin parallel replica; the ratios above are that comparison, and those archive rows have no rerun.
