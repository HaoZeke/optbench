# Reproduction

Archive commit `0f26b796`. A rerun matches when each archived metric is within one archived standard error or 10%, and `nfailed` is inside the Wilson interval of the archived failure count.

47 reproduced, 28 not reproduced, 76 not rerun. 151 archive rows.

The archived Pt-island adaptive hypersphere row is the paper row. EON (SVN r2549) and eOn (develop a7aa57472fb7d823953fd36b78cd755fa4842d30) are different codes, ranked beside that row on the same start and the 0.001 force stop. Their harness gaps stay as findings.

The pele transition-state paper score is the 2014 stop: break at the first RMS below tolerance, then one eigenvector search, transverse screen calls subtracted. Pt-island ts-ffd is 88.55 against 88 and is the paper baseline. The ranked harness-stop means are 111.27, 254.48, and 335.06 and are not that score.

OPTIM LJ38 L-BFGS tuned: the archived integer is 176 and the unrounded harness mean is 175.537.

EON FIRE in JCTC Table 1 is 645. The archive file and the rerun are both 731.

Table 10 prints EON Lanczos 165. The archive file is 325. CI-NEB(3) prints 3 failures; the archive file has 0.

Endpoint minima are free-atom float32 deltas against the reactant, compressed. A file over 1 MB is a git-LFS pointer, not a loose blob.

## accelerated-md

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| aladatom/true_standard/eon-pr-langevin | — | — | not rerun; akmc_vs_langevin |
| standard_aladatom/eon-md-langevin | — | — | not rerun; akmc_vs_langevin |

## double-ended-saddle-search

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj38/eon-ci-neb | force_calls 827 | 762 | reproduced |
| lj38/eon-ci-swdneb | force_calls 861 | 708 | not reproduced; de_exclusion |
| lj38/eon-dimer | force_calls 337 | 377 | not reproduced; improved_unset |
| lj38/eon-lanczos | force_calls 364 | 342 | reproduced |
| lj38/eon-lanczos-improved | force_calls 287 | 259 | reproduced |
| lj38/optim | force_calls 131 | 147 | not reproduced; optim_pes_counter |
| lj38/optim-cl-string | — | — | not rerun |
| lj38/pele | — | — | not rerun |
| lj38_2ts_separation/optim | force_calls 2885 | 3454 | not reproduced; optim_pes_counter |
| lj38_2ts_separation/pele | — | — | not rerun |
| pt-island/eon-ci-neb | force_calls 248 | 226 | reproduced |
| pt-island/eon-ci-neb-5 | — | — | not rerun |
| pt-island/eon-ci-neb-single | — | — | not rerun |
| pt-island/eon-ci-swdneb | force_calls 256 | 246 | reproduced |
| pt-island/eon-dimer | force_calls 182 | 192 | reproduced |
| pt-island/eon-lanczos | force_calls 325 | 222 | not reproduced; pt_lanczos |
| pt-island/eon-lanczos-improved | force_calls 222 | 181 | not reproduced; pt_lanczos |
| pt-island/optim | force_calls 103 | 138 | not reproduced; optim_pt_units |
| pt-island/optim-6 | — | — | not rerun |
| pt-island/optim-cl-string | — | — | not rerun |
| pt-island/pele | — | — | not rerun |

## globalopt

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj100-binary/eon-basinhopping | — | — | not rerun |
| lj100-binary/gmin | — | — | not rerun |
| lj150/gmin-nosym | — | — | not rerun |
| lj150/gmin-symmetrise | — | — | not rerun |
| lj38/eon-basinhopping | force_calls 508011 | 503146 | reproduced |
| lj38/gmin-csm | — | — | not rerun |
| lj38/gmin-nosym | — | — | not rerun |
| lj38/gmin-symmetrise | — | — | not rerun |
| lj38/pele-basinhopping | force_calls 521595 | 688720 | not reproduced; pele_bh_defaults |
| lj75/gmin | — | — | not rerun |
| lj98/gmin-nosym | — | — | not rerun |
| lj98/gmin-symmetrise | — | — | not rerun |
| tip4p10/gmin | — | — | not rerun |
| tip4p10/gmin-tbp | — | — | not rerun |
| tip4p15/gmin | — | — | not rerun |
| tip4p15/gmin-tbp | — | — | not rerun |
| tip4p20/gmin | — | — | not rerun |
| tip4p20/gmin-tbp | — | — | not rerun |

## hTST

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| Al/eon-dimer-cg | barrier_force_calls 28870, barrier_jobs 48, rate_force_calls 897900, rate_jobs 1511 | 19660, 50, 620600, 1600 | not reproduced; htst_al |
| Al/eon-dimer-cg2 | — | — | not rerun |
| Pt/alpha_dimer-qm | — | — | not rerun |
| Pt/eon-dimer-cg | barrier_force_calls 496000, barrier_jobs 1050, rate_force_calls 25630, rate_jobs 54 | 506000, 1062, 27160, 57 | reproduced |
| Pt/eon-dimer-cg2 | — | — | not rerun |
| Pt/eon-dimer-lbfgs | — | — | not rerun |
| Pt/eon-dimer-lbfgs2 | — | — | not rerun |
| Pt/eon-lanczos-cg | — | — | not rerun |
| Pt/eon-lanczos-cg2 | — | — | not rerun |
| Pt/eon-lanczos-lbfgs | — | — | not rerun |
| Pt/eon-lanczos-lbfgs2 | — | — | not rerun |
| Pt/quench_dimer-qm | — | — | not rerun |
| Pt/tsase-dimer-qm | — | — | not rerun |

## lowest-eigenvector

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj38/eon-dimer-cg | force_calls 30 | 30 | reproduced |
| lj38/eon-dimer-lbfgs | force_calls 28 | 28 | reproduced |
| lj38/eon-lanczos | force_calls 28 | 26 | reproduced |
| lj38/optim-first-order | force_calls 25 | 61 | not reproduced; optim_rr_defaults |
| lj38/optim-second-order | — | — | not rerun |
| lj38/pele-first-order | force_calls 25 | 25 | reproduced |
| lj38/pele-second-order | force_calls 47 | 47 | reproduced |
| lj38/sella | force_calls 17.8 | 17.8 | reproduced |
| lj38/tsase-dimer-bfgs | — | — | not rerun |
| lj38/tsase-dimer-cg | — | — | not rerun |
| lj38/tsase-lanczos | — | — | not rerun |

## minimization

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| lj38/ase-BFGS | force_calls 463 | 461 | reproduced |
| lj38/ase-bfgslinesearch | force_calls 500 | 506 | reproduced |
| lj38/ase-fire | force_calls 656 | 720 | reproduced |
| lj38/ase-lbfgs | force_calls 355 | 356 | reproduced |
| lj38/ase-lbfgslinesearch | force_calls 417 | 410 | not reproduced; linesearch_exceptions |
| lj38/ase-preconlbfgs | force_calls 2686 | 2605 | not reproduced; precon_hessian_index |
| lj38/eon-cg | force_calls 453 | 454 | reproduced |
| lj38/eon-fire | force_calls 731 | 731 | reproduced |
| lj38/eon-lbfgs | force_calls 181 | 181 | reproduced |
| lj38/eon-qm | force_calls 3523 | 3514 | reproduced |
| lj38/eon-sd | force_calls 4901 | 4901 | reproduced |
| lj38/eon-sd-twopoint | force_calls 539 | 532 | reproduced |
| lj38/newton-method | — | — | not rerun |
| lj38/optim | force_calls 176 | 176 | reproduced |
| lj38/pele-fire | force_calls 709 | 1264 | not reproduced; pele_fire |
| lj38/pele-lbfgs-M1-maxstep0.1 | force_calls 287 | 286 | reproduced |
| lj38/pele-lbfgs-M10-maxstep0.1 | force_calls 235 | 235 | reproduced |
| lj38/pele-lbfgs-M100-maxstep0.2 | force_calls 179 | 178 | reproduced |
| lj38/pele-lbfgs-M20-maxstep0.1 | force_calls 222 | 223 | reproduced |
| lj38/pele-lbfgs-M4-maxstep0.05 | force_calls 379 | 381 | reproduced |
| lj38/pele-lbfgs-M4-maxstep0.1 | force_calls 249 | 248 | reproduced |
| lj38/pele-lbfgs-M4-maxstep0.2 | force_calls 230 | 228 | reproduced |
| lj38/scipy-lbfgs | — | — | not rerun |
| lj38/tsase-sdlbfgs | — | — | not rerun |
| morse-bulk/ase-fire | force_calls 147.47 | 113 | not reproduced; ase_morse_script |
| morse-bulk/ase-lbfgs | force_calls 53.8 | 32 | not reproduced; ase_morse_script |
| morse-bulk/ase-preconLBFGS | force_calls 34 | 29 | not reproduced; ase_morse_script |
| morse-bulk/eon-cg | force_calls 106 | 106 | reproduced |
| morse-bulk/eon-fire | force_calls 156 | 156 | reproduced |
| morse-bulk/eon-lbfgs | force_calls 52 | 52 | reproduced |
| morse-bulk/eon-sd | force_calls 196 | 7038 | not reproduced; morse_sd |
| morse-bulk/eon-sd-twopoint | — | — | not rerun |
| morse-bulk/optim | force_calls 46 | 27 | not reproduced; optim_morse_memory |
| morse-bulk/pele-lbfgs | force_calls 51 | 92 | not reproduced; pele_morse |
| morse-bulk/tsase-sdlbfgs | — | — | not rerun |

## mlopt

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| C2H4/pyamff-adam10-7-fcoeff0 | — | — | not rerun |
| C2H4/pyamff-adam10-7-fcoeff0_1 | — | — | not rerun |
| C2H4/pyamff-lbfgs10-7-fcoeff0 | — | — | not rerun |
| C2H4/pyamff-lbfgs10-7-fcoeff0_1 | — | — | not rerun |
| C2H4/pyamff-rprop10-7-fcoeff0 | — | — | not rerun |
| C2H4/pyamff-rprop10-7-fcoeff0_1 | — | — | not rerun |
| emt-pt38/pyamff-adam10-7-fcoeff0 | — | — | not rerun |
| emt-pt38/pyamff-adam10-7-fcoeff0_1 | — | — | not rerun |
| emt-pt38/pyamff-lbfgs10-7-fcoeff0 | — | — | not rerun |
| emt-pt38/pyamff-lbfgs10-7-fcoeff0_1 | — | — | not rerun |
| emt-pt38/pyamff-rprop10-7-fcoeff0 | — | — | not rerun |
| emt-pt38/pyamff-rprop10-7-fcoeff0_1 | — | — | not rerun |

## neb

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| Pt-heptamer/eon-cg | force_calls 120 | 142 | not reproduced; heptamer_cg |
| Pt-heptamer/eon-fire | force_calls 86 | 85 | reproduced |
| Pt-heptamer/eon-lbfgs | force_calls 42 | 40 | reproduced |
| Pt-heptamer/eon-qm | force_calls 164 | 165 | reproduced |
| Pt-heptamer/gman-cg | — | — | not rerun |
| Pt-heptamer/gman-fire | — | — | not rerun |
| Pt-heptamer/gman-glbfgs-hess | — | — | not rerun |
| Pt-heptamer/gman-glbfgs-line | — | — | not rerun |
| Pt-heptamer/gman-lbfgs-hess | — | — | not rerun |
| Pt-heptamer/gman-lbfgs-line | — | — | not rerun |
| Pt-heptamer/gman-qm | — | — | not rerun |
| Pt-heptamer/gman-sd | — | — | not rerun |

## saddle-search

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| pt-island/adaptive_HSphere-lbfgs-davidson-sm50 | force_calls 423491, unique_saddles 114.9, fraction_connected 0.8 | 498996, 123.5, 0.866061 | reproduced |
| pt-island/adaptive_HSphere-lbfgs-lanczos-sm0 | force_calls 1.00343e+06, unique_saddles 130.4, fraction_connected 0.9 | 1.92665e+06, 0.5, 0.9682 | not reproduced; saddle_novelty |
| pt-island/adaptive_HSphere-lbfgs-lanczos-sm50 | force_calls 814519, unique_saddles 142.6, fraction_connected 0.9 | 1.58264e+06, 0.7, 0.93584 | not reproduced; saddle_novelty |
| pt-island/eon-dimer | force_calls 3.77888e+06, unique_saddles 109.222, fraction_connected 0.746017 | 6.97488e+06, 91.7, 0.691952 | not reproduced; eon_saddle_count |
| pt-island/eon-dimer-cg-20-rotations | — | — | not rerun |
| pt-island/eon-dimer-cg-20-rotations-20-bowl_breakout | — | — | not rerun |
| pt-island/eon-dimer-cg-5-rotations-20-bowl_breakout | — | — | not rerun |
| pt-island/eon-dimer-lbfgs-20-rotations | — | — | not rerun |
| pt-island/eon-dimer-lbfgs-20-rotations-20-bowl_breakout | — | — | not rerun |
| pt-island/eon-dimer-lbfgs-5-rotations | — | — | not rerun |
| pt-island/eon-dimer-lbfgs-5-rotations-20-bowl_breakout | — | — | not rerun |
| pt-island/eon-md | — | — | not rerun |

## tsopt

| Row | Archive | Rerun | Status |
| --- | --- | --- | --- |
| LJ38/eon-dimer | force_calls 528 | 523 | reproduced |
| LJ38/eon-lanczos | force_calls 321 | 293 | reproduced |
| LJ38/eon-lanczos-improved | force_calls 237 | 271 | not reproduced; improved_unset |
| LJ38/optim_cd | — | — | not rerun |
| LJ38/optim_ffd | force_calls 145 | 489 | not reproduced; optim_lj38_ts |
| LJ38/pele_cd | force_calls 254 | 263.52 | reproduced; residual |
| LJ38/pele_ffd | force_calls 192 | 207.51 | reproduced; residual |
| LJ38/sella | force_calls 69.55 | 93 | not reproduced; sella_trust |
| pt-island/eon-dimer | force_calls 116 | 115 | reproduced |
| pt-island/eon-lanczos | force_calls 143 | 146 | reproduced |
| pt-island/eon-lanczos-improved | force_calls 106 | 106 | reproduced |
| pt-island/optim_cd | — | — | not rerun |
| pt-island/optim_ffd | force_calls 71 | 170 | not reproduced; optim_pt_units |
| pt-island/pele | force_calls 88 | 88.551 | reproduced |
| pt-island/sella | force_calls 53 | 88 | not reproduced; sella_trust |

## Residuals inside the band

- tsopt/LJ38/pele_cd: Inside the 10% band (263.52 vs 254). The same xyz-versus-con starting set as ts-ffd. findTS_local.py for the CD row does not pass orthogZeroEigs; the harness template sets orthog translations-rotations. The paper baseline is this 2014-stop mean, not the harness-stop mean 335.06.
- tsopt/LJ38/pele_ffd: Inside the 10% band (207.51 vs 192). Archive run.sh unpacks coords-xyz.tar.gz (200 xyz, max pair about 4.4). The harness problem points at coords-con.tar.gz (max pair about 37). Kabsch RMS is about 6.33, so the sets are not the same geometry. Neither set has a pair past cutoff 40, and both drivers use RMS tolerance 1e-3/sqrt(114). The 8% residual is the starting set. The paper baseline is this 2014-stop mean, not the harness-stop mean 254.48.

## Causes

### akmc_vs_langevin

The archive rows are Langevin parallel replica. The harness rows are AKMC. On infra/aladatom-repro, eon/aladatom-akmc true_rate 3.597591452e9 is 14.1 times 2.542941e8, and eon/standard-aladatom-akmc true_rate 3.80613e7 is 45.5 times 8.359e5.

- accelerated-md/aladatom/true_standard/eon-pr-langevin
- accelerated-md/standard_aladatom/eon-md-langevin

### de_exclusion

The 3.5.0 doubly-nudged file is the per-band mean under the 2500-call exclusion (708, 5 failures). The archive mean is 861. The client total is a different counter and is also outside 10 percent.

- double-ended-saddle-search/lj38/eon-ci-swdneb

### improved_unset

The r2549 template turns improved on. The 2014 ini left that key unset. The mean lands outside 10 percent.

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

### optim_rr_defaults

The Rayleigh-Ritz template uses updates 4. The archived first-order mean is 25 and the second-order mean is 46. This default scores 61, so it is not either archived order.

- lowest-eigenvector/lj38/optim-first-order

### linesearch_exceptions

ASE 3.29 LBFGSLineSearch with the archived opt.py arguments (maxstep 0.2, alpha 1/0.001, memory 100) scores 410 against 417, inside 10 percent. report.sh counts a start as failed only when a written total_force_calls is at least 10000, and the benchmark.dat label says M=5 while opt.py sets memory 100. The harness records 7 failures: 5 RuntimeError LineSearch failed (ASE 3.8.2 already raises that; ASE 3.6.0 is not on PyPI) and 2 Hessian index 1 after the Euclidean force norm is under 0.01. The Wilson interval on 0 failures out of 1000 is about 0 to 3.83.

- minimization/lj38/ase-lbfgslinesearch

### precon_hessian_index

ASE 3.29 PreconLBFGS with the archived opt.py setup (Exp A 3, maxstep 0.1, ASE fmax 0.01) scores 2605 against 2686, inside 10 percent. The script uses a tsase LJ cutoff of 35. This row uses the harness cutoff 40. report.sh counts a failure only when a written total_force_calls is at least 10000. The harness records 10 failures, each a Hessian index of 1 after the maximum force is under 0.01. The Wilson interval on 0 failures out of 1000 is about 0 to 3.83.

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

- saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm0
- saddle-search/pt-island/adaptive_HSphere-lbfgs-lanczos-sm50

### eon_saddle_count

The classic dimer campaign (improved false, the 2014 key) has fraction_connected 0.692 against 0.746, inside 10 percent, but force calls 6.97e6 against 3.78e6 and unique saddles 91.7 against 109.2. The archive standard error on the force calls is 5.27e5. This eOn AKMC counter is not the harness RMSD link test.

- saddle-search/pt-island/eon-dimer

### optim_lj38_ts

Archive run.sh unpacks coords-xyz.tar.gz. The harness uses the problem coords-con set (Kabsch RMS about 6.33). The odata preamble is not in the archive snapshot. The template uses updates 4. The mean is 489 against 145.

- tsopt/LJ38/optim_ffd

### sella_trust

Sella 2.6.0 with the archived trust-region numbers (eta, gamma, delta0, rho, sigma) is the parameter_source=archive row. It misses the 2014 mean. The constructor-default row is a different setup.

- tsopt/LJ38/sella
- tsopt/pt-island/sella

Rows with no exact-setup campaign stay not rerun and are not given a cause. AKMC on aladatom is a different method from Langevin parallel replica; the ratios above are that comparison, and the archive rows stay not rerun.
