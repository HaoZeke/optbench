# Saddle and kMC codes in this lane

Lane list of 2026-10-18. A code is a module in `optbench/engines`, a pin in
`codes/registry.toml`, and a template under `templates/`. `campaign.run`
calls `optbench.engines.find`. `available` returns the reason a code cannot
start. `run` writes harness campaign files or raises that reason. The score
is the harness force count. A client's own tally is not a score.

Pt-island geometries use the smooth-linear Morse potential. Aluminum
hTST and the accelerated-md reactant use the r2549 EAM force routine.
hTST and accelerated-md problem files state rates and barriers. pyKMC
runs a KMC campaign on those reactants and writes the archive columns.
The saddle catalogue stores both endpoint minima and scores the eOn r2205 link.

## Rows on disk

Failure classes are the v2 counts. They sum to `nfailed`. The median is the
success-only median. A row with no success has no median.

| problem | template | successes | failures | median calls |
| --- | --- | --- | --- | --- |
| lowest-eigenvector/lj38 | partn/mode-lj38 | 174/200 | 26 divergence | 22 |
| tsopt/pt-island | partn/ts-pt | 49/49 | 0 | 325 |
| tsopt/LJ38 | partn/ts-lj38 | 178/200 | 21 cap, 1 crash | 3302 |
| saddle-search/pt-island | partn/saddle-pt | 1/200 | 186 divergence, 13 crash | 925 |
| lowest-eigenvector/lj38 | tsase/mode-lj38 | 178/200 | 22 divergence | 221.5 |
| tsopt/pt-island | tsase/dimer-pt | 49/49 | 0 | 892 |
| tsopt/LJ38 | tsase/dimer-lj38 | 0/200 | 200 cap | — |
| minimization/lj38 | tsase/lbfgs-lj38 | 996/1000 | 4 wrong stationary | 169 |
| minimization/lj38 | tsase/mdmin-lj38 | 0/1000 | 1000 cap | — |
| minimization/morse-bulk | tsase/lbfgs-morse | 12/100 | 88 wrong stationary | 53.5 |
| neb/Pt-heptamer | tsase/ci-neb | 13/13 | 0 | 1427 |
| neb/Pt-heptamer | tsase/neb | 0/13 | 13 divergence | — |
| double-ended-saddle-search/lj38 | tsase/ci-neb | 26/50 | 17 cap, 7 divergence | 1233 |
| double-ended-saddle-search/lj38 | tsase/neb | 0/50 | 36 divergence, 14 cap | — |
| double-ended-saddle-search/lj38_2ts_separation | tsase/ci-neb | 5/51 | 43 cap, 2 wrong stationary, 1 divergence | 7693 |
| double-ended-saddle-search/lj38_2ts_separation | tsase/neb | 0/51 | 44 cap, 4 wrong stationary, 3 divergence | — |
| double-ended-saddle-search/pt-island | tsase/ci-neb | 38/59 | 20 divergence, 1 cap | 426 |
| double-ended-saddle-search/pt-island | tsase/neb | 0/59 | 56 divergence, 3 cap | — |
| minimization/lj38 | lammps/cg-lj38 | 848/1000 | 148 cap, 3 wrong stationary, 1 divergence | 1410 |
| minimization/lj38 | lammps/sd-lj38 | 383/1000 | 617 cap | 5252 |
| minimization/lj38 | lammps/hftn-lj38 | 997/1000 | 3 wrong stationary | 974 |
| minimization/lj38 | lammps/quickmin-lj38 | 991/1000 | 5 divergence, 4 wrong stationary | 2543 |
| minimization/lj38 | lammps/fire-lj38 | 727/1000 | 224 divergence, 49 wrong stationary | 472 |
| minimization/morse-bulk | lammps/cg-morse | 5/100 | 95 wrong stationary | 95 |
| minimization/morse-bulk | lammps/sd-morse | 4/100 | 96 wrong stationary | 255.5 |
| minimization/morse-bulk | lammps/hftn-morse | 3/100 | 97 wrong stationary | 511 |
| minimization/morse-bulk | lammps/quickmin-morse | 5/100 | 95 wrong stationary | 2820 |
| minimization/morse-bulk | lammps/fire-morse | 12/100 | 88 wrong stationary | 447 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s0 | 848/1000 | 148 cap, 3 wrong stationary, 1 divergence | 1410 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s1em4 | 896/1000 | 100 cap, 3 divergence, 1 wrong stationary | 2480 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s1em3 | 997/1000 | 3 divergence | 1776 |
| minimization/lj38-noise | lammps/cg-lj38-noise-s1em2 | 874/1000 | 88 divergence, 38 wrong stationary | 1310 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s0 | 383/1000 | 617 cap | 5252 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s1em4 | 12/1000 | 988 cap | 6037 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s1em3 | 684/1000 | 313 cap, 3 wrong stationary | 5905.5 |
| minimization/lj38-noise | lammps/sd-lj38-noise-s1em2 | 698/1000 | 227 cap, 75 wrong stationary | 5627.5 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s0 | 997/1000 | 3 wrong stationary | 974 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s1em4 | 224/1000 | 776 divergence | 6945 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s1em3 | 283/1000 | 713 divergence, 4 wrong stationary | 6882 |
| minimization/lj38-noise | lammps/hftn-lj38-noise-s1em2 | 587/1000 | 243 divergence, 170 wrong stationary | 5438 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s0 | 991/1000 | 5 divergence, 4 wrong stationary | 2543 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s1em4 | 983/1000 | 13 divergence, 4 wrong stationary | 2450 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s1em3 | 741/1000 | 9 cap, 231 divergence, 19 wrong stationary | 5454 |
| minimization/lj38-noise | lammps/quickmin-lj38-noise-s1em2 | 576/1000 | 10 cap, 248 divergence, 166 wrong stationary | 4455 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s0 | 727/1000 | 224 divergence, 49 wrong stationary | 472 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s1em4 | 729/1000 | 224 divergence, 47 wrong stationary | 462 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s1em3 | 706/1000 | 224 divergence, 70 wrong stationary | 435 |
| minimization/lj38-noise | lammps/fire-lj38-noise-s1em2 | 564/1000 | 224 divergence, 212 wrong stationary | 335 |
| minimization/lj38 | dlfind/lbfgs-lj38 | 998/1000 | 1 cap, 1 wrong stationary | 184 |
| minimization/lj38 | dlfind/sd-lj38 | 990/1000 | 10 cap | 2563.5 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s0 | 990/1000 | 10 cap | 2563.5 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s1em4 | 991/1000 | 8 cap, 1 wrong stationary | 2564 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s1em3 | 996/1000 | 3 cap, 1 wrong stationary | 2371 |
| minimization/lj38-noise | dlfind/sd-lj38-noise-s1em2 | 955/1000 | 45 wrong stationary | 1778 |
| minimization/lj38 | dlfind/cg-lj38 | 991/1000 | 9 cap | 2541 |
| minimization/morse-bulk | dlfind/cg-morse | 18/100 | 82 wrong stationary | 250 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s0 | 991/1000 | 9 cap | 2541 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s1em4 | 993/1000 | 6 cap, 1 wrong stationary | 2511 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s1em3 | 995/1000 | 3 cap, 2 wrong stationary | 2329 |
| minimization/lj38-noise | dlfind/cg-lj38-noise-s1em2 | 956/1000 | 44 wrong stationary | 1718 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s0 | 998/1000 | 1 cap, 1 wrong stationary | 184 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em4 | 998/1000 | 2 wrong stationary | 184 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em3 | 999/1000 | 1 wrong stationary | 209 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em2 | 974/1000 | 26 wrong stationary | 199 |
| minimization/morse-bulk | dlfind/lbfgs-morse | 32/100 | 68 wrong stationary | 77 |
| minimization/morse-bulk | dlfind/sd-morse | 13/100 | 87 wrong stationary | 253 |
| minimization/morse-bulk | optim/lbfgs-morse | 13/100 | 87 wrong stationary | 72 |
| minimization/lj38 | optim/lbfgs-lj38 | 998/1000 | 2 wrong stationary | 217 |
| double-ended-saddle-search/lj38 | optim/de-lj38 | 50/50 | 0 | 129 |
| double-ended-saddle-search/lj38 | pathsample/de-lj38 | 50/50 | 0 | 131 |
| double-ended-saddle-search/lj38_2ts_separation | optim/de-2ts | 44/51 | 7 cap | 2832.5 |
| double-ended-saddle-search/lj38_2ts_separation | pathsample/de-2ts | 43/51 | 8 cap | 2972 |
| double-ended-saddle-search/pt-island | optim/de-pt | 59/59 | 0 | 133 |
| double-ended-saddle-search/pt-island | pathsample/de-pt | 59/59 | 0 | 137 |
| double-ended-saddle-search/lj38 | compass/sqnm | 47/50 | 3 cap | 1221 |
| double-ended-saddle-search/lj38_2ts_separation | compass/sqnm | 41/51 | 9 cap, 1 wrong stationary | 2265 |
| neb/Pt-heptamer | compass/sqnm | 13/13 | 0 | 548 |
| double-ended-saddle-search/pt-island | compass/sqnm | 58/59 | 1 cap | 1000.5 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s0 | 996/1000 | 4 wrong stationary | 206 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s0-tuned | 996/1000 | 4 wrong stationary | 206 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em4 | 994/1000 | 6 wrong stationary | 205 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em4-tuned | 994/1000 | 6 wrong stationary | 205 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em3 | 989/1000 | 11 wrong stationary | 195 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em3-tuned | 989/1000 | 11 wrong stationary | 195 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em2 | 884/1000 | 116 wrong stationary | 161 |
| minimization/lj38-noise | rgmin/lj38-noise-lbfgs-s1em2-tuned | 884/1000 | 116 wrong stationary | 161 |
| lowest-eigenvector/lj38 | pysisyphus/lanczos-lj38 | 160/200 | 40 divergence | 22 |
| minimization/lj38 | pysisyphus/lbfgs-lj38 | 0/1000 | 1000 cap | — |
| minimization/lj38 | pysisyphus/cg-lj38 | 254/1000 | 745 cap, 1 wrong stationary | 693 |
| minimization/lj38 | pysisyphus/sd-lj38 | 205/1000 | 795 cap | 808 |
| tsopt/pt-island | pykmc/ts-pt | 46/49 | 1 cap, 2 divergence | 746.5 |
| hTST/Pt | pykmc/htst-pt | 13/13 | 0 | 3367604 |
| hTST/Al | pykmc/htst-al | 639/639 | 0 | 56329 |
| accelerated-md/aladatom | pykmc/aladatom | 3/3 | 0 | 166853 |
| accelerated-md/standard_aladatom | pykmc/standard-aladatom | 3/3 | 0 | 119079 |
| saddle-search/pt-island | pykmc/saddle-pt | 512/792 | 210 wrong stationary, 69 crash, 1 not converged | 1260.5 |

The rgsaddle band campaigns on `neb/Pt-heptamer` and the three double-ended
problems are on disk (12 templates each). The eight rgmin `lj38-noise` L-BFGS rows are on disk.
Both tunings set `maxstep` 0.2, and each pair matches. Every start met the
force stop. The v2 failures are Hessian index not 0. Zero noise is 996 of
1000. `1e-4` is 994 of 1000. `1e-3` is 989 of 1000. `1e-2` is 884 of 1000,
116 wrong stationary, success median 161.

pARTn on `tsopt/LJ38` with `fire_dt_init` 0.05 grew a FIRE step to length
10.2. `block_perprelax.f90` then aborted every start with `BOX EXPLOSION`
once an atomic force norm passed `1e4`. The cluster and the cell were
already the harness geometry. `LAMMPS.LJ.SaddleRefine` uses timestep 0.015
and `dmax` 0.1. `dmax` is a LAMMPS `fix artn` input, and `engine="other"`
has no such cap. The commented `step_max` in `m_artn_fire.f90` is not a
user input. A run at `fire_dt_init` 0.015 and `fire_dt_max_f` 10 finished
165 of 200, with 26 `BOX EXPLOSION` crashes and 9 `EIGENVALUE LOST`
crashes, success median 407. The row on disk uses `fire_dt_init` 0.005
and `fire_dt_max_f` 1, so the timestep cannot grow. That row is 178 of
200, 21 cap, 1 `EIGENVALUE LOST` on start `0023` at 486 calls, success
median 3302. The largest step in that row is 0.05. `nevalf` is not the
count.

TSASE MDMin on LJ38 hits the cap of 10000 on every start. The step limit
keeps the force above the stop. The cap record has no final force norm,
because the cap is raised before that geometry is stored. The class is cap.

## OPTIM and PATHSAMPLE

Pin `optim`. The snapshot is `http://www-wales.ch.cam.ac.uk/svn.tar.bz2`,
122743078 bytes, sha256
`02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f`.
There is no git commit. `OPTIM/source/py.f90` is the Paramonov–Yaliraki
ellipsoid potential, not a Python hook. The external hook is the ASE socket
in `INTERFACES/pysocket/ase_interface.f90`. The tarball does not contain
`fmlip_relay.py`. The harness server speaks that protocol: each `COMPUTE`
is one `pes.evaluate`. `EV` and `ANG` scales are 1. The OPTIM target was
built with gfortran, `WITH_ALIGN` off and `WITH_OPENMP` off. Minimization
L-BFGS sends `BFGSMIN` with an RMS stop equal to the Euclidean force norm
divided by `sqrt(3N)`, `UPDATES` 4, `MAXSTEP` 0.2, and `STEPS` equal to
the cap. The relay refuses the call that would pass the cap and does not
count it. A one-start check on LJ38 `cluster_0000` made 228 harness calls.
OPTIM reported 228 energy-and-gradient calls. The stored point re-evaluates
to energy `-163.2168156980899` and Euclidean force `0.00817050460275337`.
That check is not itself a campaign row. The 1000-start LJ38 L-BFGS row is
998 of 1000, 2 wrong stationary, success median 217. Morse bulk L-BFGS is 13 of 100. The
other 87 meet the force stop with a Hessian index other than 0, so they
are wrong stationary. The success median is 72. Start `pos_0000` took 55
harness calls and stopped at force `0.000842553062284986`, energy
`-1421.4513232624358`, vibrational index 6. `optim/unhooked` writes nothing.
Noise has no harness row: the relay has no noisy-force window. NEB, tsopt, and
hTST have no harness row. The double-ended rows use the 2014 `odata` preambles.
The scored geometry is the socket frame whose energy matches the line
printed before `Converged to TS`. `lbfgs.xyz` is written at the start of
that `mylbfgs` call. PATHSAMPLE launches those OPTIM jobs. The score sums
`pes.evaluate` over every child, including the two endpoint quenches
(`DUMPDATA`, `NOFRQS`). Wall time is the parent process.

The 2014 archive rows stay on the board. Their binary is `wales-27058.tar.bz2`.
This pin is the svn snapshot above. LJ38 DNEB in the archive is mean 131,
median 120, min 66, max 260, 0 failed, 50 pairs. `optim/de-lj38` is 50 of
50, mean 147, median 129, min 74, max 308. `pathsample/de-lj38` is 50 of
50, mean 149, median 131, min 76, max 310. On checked pairs the harness
count matched OPTIM's energy-and-gradient count. PATHSAMPLE adds the two
endpoint quenches, which is 2 calls on LJ38. Pt-island archive OPTIM is
mean 103, median 98, min 59, max 194, and `optim-6` is mean 117, median
115, min 60, max 201. `optim/de-pt` is 59 of 59, mean 138, median 133, min
95, max 327. `pathsample/de-pt` is 59 of 59, mean 146, median 137, min 99,
max 459. Morse coordinates are written to OPTIM in units of `r0` and the
energy in units of `D`. The harness potential stays eV and angstrom. The
scaled endpoint quench takes more LBFGS steps than the LJ quench, so
PATHSAMPLE sits about 8 calls above direct OPTIM. `lj38_2ts_separation`
in the archive is mean 2885, median 2565, min 665, max 7138, 0 failed, and
the path carries 5 transition states. `optim/de-2ts` is 44 of 51, 7 cap,
success mean 3454, median 2832.5, min 695, max 9680. `pathsample/de-2ts`
is 43 of 51, 8 cap, success mean 3507, median 2972, min 698, max 9658.
The cap is the problem limit of 10000. The seven OPTIM pairs that hit it
still have a stored force under `1e-5`, because a transition state was
printed and the connection search kept calling the potential. Pt-heptamer
has no OPTIM archive row.

## pARTn

Pin `artn`, 1.1.0, DEVEL `9b753517`. `libartn.so` is the engine plugin and
the standalone `next_displ` loop. Morse uses `lammps/metal`. Lennard-Jones
uses `lammps/lj`. Each returned displacement is applied until `has_sad`.
The displacement on that call is the push off the saddle, and the loop
stops there. The scored point is `tau_sad`, re-evaluated in verification
because the search already paid for that geometry. The 2014 archive has
no ARTn row. The eOn and pele saddle rows stay.

`tsopt/pt-island` with the Pt111 knobs (`tuning` `pt111`, `lpush_final`
false, `fire_dt_init` 4.134, `fire_dt_max_f` 1) is 49 of 49, success
median 325. The earlier `tuning` `default` row was 45 of 49, 4
eigenvalue-lost crashes, median 247.

`saddle-search/pt-island` uses the OptBench saddle script and the Pt111
FIRE timestep. `search_budget` is 2000, the authors' `minimize` maxeval.
`fire_dt_max_f` is 2. The authors also set LAMMPS `dmax` 8, which this
loop cannot set. Two hundred searches from the script's in-plane push
(1 to 7 of atoms 0-6, length 0.05 to 0.5) stored 187 index-1 saddles.
Thirteen searches crashed with `EIGENVALUE LOST`. The stored row used
Cartesian RMSD 0.5 over the 175 free atoms. One search meets that test:
barrier 0.996 eV, 925 calls, endpoint RMSD 0.318 and 0.519. The
repetition total is 154419, with 3 unique saddles and fraction connected
0.005. The success median is 925. The other 186 are divergence. Ten
searches have an endpoint RMSD above 0.5. The other nine of those repeat
a stored saddle or have a barrier of at least 1.5 eV. The common products
sit near RMSD 0.22 and 0.32, so both endpoints fall inside 0.5. A pure
2.71 Å hop of one atom is RMSD 2.71/sqrt(175), about 0.205, and the same
test calls it the reactant. The starts file stores those RMSD scalars and
not the endpoint coordinates. The ranked row
is one record per search, with cap 6000 (the pARTn budget plus the two
endpoint budgets of 2000). The PAR-2 mean is 11944.625. The OptBench
catalogue calls a saddle connected when the nearer minimum's `delr` is
under 0.5. That is a different test from either harness link.

## Pt-island endpoint link

The harness link is now the 2014 eOn test that produced the archived
fraction connected. `archive/entries/saddle-search/pt-island/eon-dimer`
is Eon r2205, dated 15 Jul 2014, `fraction_connected` 0.746017.
`done.py` increments `not_connected` when the result token contains
`Connected`, which matches the label `Not Connected`. That label is
termination reason 6. The public tree on that date is `8dc9bf61`.
`ProcessSearchJob::doProcessSearch` calls `initial->compare(min1)` and
`initial->compare(min2)` with one argument. `Matter::compare` defaults
`indistinguishable` to false (`Matter.h`), and `checkRotation` defaults
to false, so the branch is

```
parameters->distanceDifference > perAtomNorm(*matter)
```

`Parameters.cpp` sets `distanceDifference` to 0.1. The archived
`config.ini` has no `[Structure Comparison]` section, so 0.1 Å applies.
`perAtomNorm` is the maximum per-atom distance after `Matter::pbc`, which
folds every positive cell axis with `fmod(fmod(fraction, 1) + 1.5, 1) - 0.5`.
Two minima are the same when that maximum is strictly below 0.1 Å. A
search is connected when exactly one endpoint matches. `rmsd_connect` 0.5
stays the saddle-novelty test and is not this link.

The rows below do not store `minimum`, `endpoint_plus`, and
`endpoint_minus`. RMSD scalars do not determine the per-atom maximum.
Their connected counts and PAR-2 stay the stored values. The MATLAB smoke in
`softsaddle/entry.toml` records two searches and does not store endpoint
coordinates.

| row | geometries | old connected | new connected | old PAR-2 | new PAR-2 |
| --- | --- | --- | --- | --- | --- |
| sella/pt-island-search-default, asv n=16 | absent (success track only) | 12/16 | 12/16 | 5201.625 | 5201.625 |
| sella/pt-island-search-tuned, asv n=16 | absent (success track only) | 11/16 | 11/16 | 6420.125 | 6420.125 |
| rgsaddle/saddle-dimer-rotskip-tuned | absent; RMSD scalars on 17719 of 20000 | 2/20000 | 2/20000 | 10127.05255 | 10127.05255 |
| rgsaddle/saddle-dimer-fire | absent; no RMSD scalars | 0/20000 | 0/20000 | 10128 | 10128 |
| rgsaddle/saddle-dimer-fire-tuned | absent; RMSD scalars on 9 of 20000 | 0/20000 | 0/20000 | 10128 | 10128 |
| rgsaddle/saddle-dimer-lbfgs | absent; no RMSD scalars | 0/20000 | 0/20000 | 10128 | 10128 |
| rgsaddle/saddle-dimer-lbfgs-tuned | absent; RMSD scalars on 19957 of 20000 | 0/20000 | 0/20000 | 10128 | 10128 |
| rgsaddle/saddle-dimer-rotskip | absent; RMSD scalars on 19718 of 20000 | 0/20000 | 0/20000 | 10128 | 10128 |
| rgsaddle/saddle-lanczos | absent; RMSD scalars on 4068 of 20000 | 0/20000 | 0/20000 | 10128 | 10128 |
| rgsaddle/saddle-lanczos-tuned | absent; RMSD scalars on 13743 of 20000 | 0/20000 | 0/20000 | 10128 | 10128 |
| partn/saddle-pt | absent; `geometry` is `tau_sad`; RMSD on 187 of 200 | 1/200 | 1/200 | 11944.625 | 11944.625 |
| ase/dimer-pt-island-default, campaign n=16 | absent | 16/16 | 16/16 | 1118.0625 | 1118.0625 |
| ase/dimer-pt-island-tuned, campaign n=16 | absent | 16/16 | 16/16 | 915.4375 | 915.4375 |
| ase/dimer-pt-island-default, asv n=1 | absent | 1/1 | 1/1 | 699 | 699 |
| ase/dimer-pt-island-tuned, asv n=1 | absent | 1/1 | 1/1 | 564 | 564 |
| sella/pt-island-search-default, asv n=1 | absent | 0/1 | 0/1 | 20000 | 20000 |
| sella/pt-island-search-tuned, asv n=1 | absent | 0/1 | 0/1 | 20000 | 20000 |
| eon-client/saddle-pt-dimer | absent; `geometry` is the string `pt-island` | fraction 0.8125 | fraction 0.8125 | 40000000 | 40000000 |
| eon-client/saddle-pt-lanczos | absent; same string | fraction 1 | fraction 1 | 40000000 | 40000000 |
| eon-client/saddle-pt-md | absent; same string | fraction 0 | fraction 0 | 40000000 | 40000000 |
| eon-svn/saddle-pt-dimer-bounded | absent; same string | fraction 0.75 | fraction 0.75 | 40000000 | 40000000 |
| eon-svn/saddle-pt-lanczos-bounded | absent; same string | fraction 1 | fraction 1 | 40000000 | 40000000 |
| eon-svn/saddle-pt-md | absent; same string | fraction 0 | fraction 0 | 40000000 | 40000000 |
| eon/develop-saddle-pt-dimer, asv n=1 | absent | 0/1 | 0/1 | 200000000 | 200000000 |
| eon/develop-saddle-pt-lanczos, asv n=1 | absent | 0/1 | 0/1 | 200000000 | 200000000 |
| eon-3.5.0-saddle-pt-classic | absent; repetition means | fraction 0.6919518057 | fraction 0.6919518057 | not applied | not applied |
| eon-3.5.0-saddle-pt-dimer | absent; repetition means | fraction 0.7517199457 | fraction 0.7517199457 | not applied | not applied |
| archive eon-dimer r2205 | absent | fraction 0.746017 | fraction 0.746017 | not applied | not applied |
| archive eon-md r2205 | absent | fraction 0.886966 | fraction 0.886966 | not applied | not applied |
| archive eon-dimer-cg-20-rotations r2213 | absent | fraction 0.634205777778 | fraction 0.634205777778 | not applied | not applied |
| archive eon-dimer-cg-20-rotations-20-bowl_breakout r2213 | absent | fraction 0.746989444444 | fraction 0.746989444444 | not applied | not applied |
| archive eon-dimer-cg-5-rotations-20-bowl_breakout r2213 | absent | fraction 0.703835111111 | fraction 0.703835111111 | not applied | not applied |
| archive eon-dimer-lbfgs-20-rotations r2213 | absent | fraction 0.745814111111 | fraction 0.745814111111 | not applied | not applied |
| archive eon-dimer-lbfgs-20-rotations-20-bowl_breakout r2213 | absent | fraction 0.796864222222 | fraction 0.796864222222 | not applied | not applied |
| archive eon-dimer-lbfgs-5-rotations r2213 | absent | fraction 0.747722 | fraction 0.747722 | not applied | not applied |
| archive eon-dimer-lbfgs-5-rotations-20-bowl_breakout r2213 | absent | fraction 0.853230777778 | fraction 0.853230777778 | not applied | not applied |
| archive adaptive_HSphere-lbfgs-davidson-sm50 | absent | fraction 0.8 | fraction 0.8 | not applied | not applied |
| archive adaptive_HSphere-lbfgs-lanczos-sm0 | absent | fraction 0.9 | fraction 0.9 | not applied | not applied |
| archive adaptive_HSphere-lbfgs-lanczos-sm50 | absent | fraction 0.9 | fraction 0.9 | not applied | not applied |

Sella and ASE successes are order-1 convergence, not this endpoint link.
The eOn client, eOn SVN, eOn 3.5.0, and archive fractions are those codes'
own `Not Connected` labels. PAR-2 stays off a campaign ratio and off
the 2014 archive.

## TSASE

Pin `tsase`, commit `98aac60` (2025-09-26). `setup.py` says 1.0. The
checkout has no LICENSE file. Fortran calculators use `numpy.distutils` and
stay unloaded. SSNEB and the solid-state dimer are loaded without
`tsase/__init__.py`. `ss` is false. A periodic all-free Hessian drops the
three translations before the index is counted.

The LJ38 dimer hits the cap on every start. The Pt-island dimer and the
climbing-image NEB on Pt-heptamer are the saddle rows. The plain NEB on
that heptamer diverges on all 13 pairs. Morse bulk L-BFGS meets the force
stop and the Hessian index is not 0 for 88 of 100 starts. The success
median is 53.5.

Still open, and not given an invented budget: basin hopping and minima
hopping on `globalopt/*`. Those problems state `energy_above_minimum` and
do not state `max_force_calls`. AMD-SVM on `accelerated-md/aladatom` has
no harness row. The problem has no force-call protocol and the harness has no EAM
potential. hTST prefactors have no harness row for the same reason.

## LAMMPS

Pin `lammps`, `stable_30Sep2026`, commit `8de817dd`, version string
`30 Sep 2026`. Minimizers `cg`, `sd`, `hftn`, `quickmin`, and `fire` call
the harness through `fix external pf/callback` and `pair_style zero`. A
native pair style is not the score. FIRE and quickmin use timestep 0.002.
The velocity kick is not limited by `dmax`, and a larger timestep loses
atoms on the first LJ38 step.

On Morse bulk the force stop is met and the Hessian index is not 0 for 95
of 100 conjugate-gradient starts, 96 of 100 steepest-descent starts, 97 of
100 Hessian-free Newton starts, 95 of 100 quickmin starts, and 88 of 100
FIRE starts. FIRE is 12 of 100, success median 447. A fresh Hessian at the
stored CG point `pos_0000` (67 calls, force `0.0006534685520009086`, energy
`-1421.4513146864156`) has vibrational eigenvalues `-7.023206723482451` and
`-6.994643641886293` after the three translations are removed. `pos_0002`
(75 calls, force `0.0008961440060698902`) has seven eigenvalues from
`-8.207944407072567` to `-5.991636654081658`. `pos_0013` meets the stop
with index 0. The curvature tolerance at the `1e-3` force stop is `0.001`
(the Hessian Frobenius norms are `419.652` and `418.714`). Both that
tolerance and the fixed `1e-6` cutoff count the same negative modes. These
are index-2 and index-7 stationary points, not a cutoff artifact. Zero-noise
LJ38 conjugate gradient is 848 of 1000, the same failure split as the clean
LJ38 row, success median 1410. At sigma `1e-4` it is 896 of 1000, 100 cap,
3 divergence, 1 wrong stationary, success median 2480. At sigma `1e-3` it
is 997 of 1000, 3 divergence, success median 1776. At sigma `1e-2` it is
874 of 1000, 88 divergence, 38 wrong stationary, success median 1310.
Zero-noise steepest descent is 383 of 1000, 617 cap, success median 5252,
the same split as the clean LJ38 row. At sigma `1e-4` steepest descent is
12 of 1000, 988 cap, success median 6037. At sigma `1e-3` it is 684 of
1000, 313 cap, 3 wrong stationary, success median 5905.5. At sigma `1e-2`
it is 698 of 1000, 227 cap, 75 wrong stationary, success median 5627.5.
Zero-noise Hessian-free Newton is 997 of 1000, 3 wrong stationary, success
median 974, the same split as the clean LJ38 row. At sigma `1e-4` it is
224 of 1000, 776 divergence, success median 6945. At sigma `1e-3` it is
283 of 1000, 713 divergence, 4 wrong stationary, success median 6882. At
sigma `1e-2` it is 587 of 1000, 243 divergence, 170 wrong stationary,
success median 5438. Zero-noise quickmin is 991 of 1000, 5 divergence, 4
wrong stationary, success median 2543, the same split as the clean LJ38
row. At sigma `1e-4` it is 983 of 1000, 13 divergence, 4 wrong stationary,
success median 2450. At sigma `1e-3` it is 741 of 1000, 9 cap, 231
divergence, 19 wrong stationary, success median 5454. At sigma `1e-2` it
is 576 of 1000, 10 cap, 248 divergence, 166 wrong stationary, success
median 4455. Zero-noise FIRE is 727 of 1000, 224 divergence, 49 wrong
stationary, success median 472, the same split as the clean LJ38 row. At
sigma `1e-4` it is 729 of 1000, 224 divergence, 47 wrong stationary,
success median 462. At sigma `1e-3` it is 706 of 1000, 224 divergence, 70
wrong stationary, success median 435. At sigma `1e-2` it is 564 of 1000,
224 divergence, 212 wrong stationary, success median 335. `fix neb` needs a multi-partition MPI
run and has no harness row. `prd`, `tad`,
`hyper`, and `dynamical_matrix` have no harness row. The Al, Pt, and aladatom
problems are EAM, and the harness has no EAM evaluator.

## DL-FIND

Pin `dlfind`, libdlfind 0.0.3, commit `4167998d`. `iopt` 3 is L-BFGS, 0 is
steepest descent, and 1 is conjugate gradient. The gradient callback is the
score. Coordinates stay in the harness unit. The component tolerance is the
Euclidean stop divided by `sqrt` of the free components. On Morse bulk L-BFGS the force stop is met and the Hessian index is not 0
for 68 of 100 starts. Steepest descent on LJ38 is 990 of 1000, 10 cap,
success median 2563.5. Steepest descent on Morse bulk is 13 of 100, 87
wrong stationary, success median 253. Steepest descent on zero-noise LJ38
is 990 of 1000, 10 cap, success median 2563.5. At sigma `1e-4` it is 991
of 1000, 8 cap, 1 wrong stationary, success median 2564. At sigma `1e-3`
it is 996 of 1000, 3 cap, 1 wrong stationary, success median 2371. At sigma
`1e-2` it is 955 of 1000, 45 wrong stationary, success median 1778.
Conjugate gradient on LJ38 is 991 of 1000, 9 cap, success median 2541.
On Morse bulk it is 18 of 100, 82 wrong stationary, success median 250.
Zero-noise conjugate gradient matches the clean LJ38 row. At sigma `1e-4`
it is 993 of 1000, 6 cap, 1 wrong stationary, success median 2511. At
sigma `1e-3` it is 995 of 1000, 3 cap, 2 wrong stationary, success median
2329. At sigma `1e-2` it is 956 of 1000, 44 wrong stationary, success
median 1718. NEB (`icoord` 190)
and the dimer (`icoord` near 210) are not in these rows.

## COMPASS

Pin `compass`, 1.0.0, commit `22da7bd` (2024-08-08),
`https://gitlab.com/goedeckergroup/compass`. The ASE calculator is the
score. `forcetol`, `minimize_fmax`, and both target distances are the
problem force stop. The images stop that far apart. On LJ38 a target of
0.2 left a harness force near 1. A target equal to the force stop left a
harness force under the stop and Hessian index 1. ASE 3.29 moved
`ExpCellFilter` to `ase.filters`. The runner aliases it back. The library
treats any periodic boundary as a variable cell. These problems keep the
given cell, so `is_varshape` is false. Fixed atoms are restored on every
`set_positions`. Pt-island is 58 of 59, one cap, success median 1000.5.

## pysisyphus

Pin `pysisyphus`, commit `a4ce10dd` (2025-10-29). The clone has no tag, so
the installed package reports `0.1.dev1`. GPL-3.0. Lanczos on
`lowest-eigenvector/lj38` overlaps the reference on 160 of 200 starts.
The other 40 are divergence. The success median is 22. `dx` is 5e-3 and
`dl` is 1e-2. Six rigid modes are removed. Cartesian L-BFGS, conjugate
gradient, and steepest descent keep the library step limits in Bohr and
call the potential in the harness unit. On start `cluster_0000` each one
used the cap of 10000 with the force still above 0.01. The 1000-start
L-BFGS row does the same on every start. Conjugate gradient is 254 of 1000,
745 cap, 1 wrong stationary, success median 693. Steepest descent is 205 of
1000, 795 cap, success median 808. The dimer drops translation and not rotation. The NEB
springs stay at `k_min` 0.1 and `k_max` 0.3. Those two drivers are not
connected, so there is no pysisyphus row for `tsopt/*`,
`saddle-search/pt-island`, `double-ended-saddle-search/*`, or
`neb/Pt-heptamer`.

## k-ART

`https://gitlab.com/groupe_mousseau/kart` still answers `could not read
Username`. The public docs at `kart-doc.readthedocs.io` point at that same
repository and say access is granted on request. No public copy was found.
No commit is pinned.

## pyKMC

Pin `pykmc`, version string 0.0.1, commit `9fd1f6e9085ef92642142406f3dd74e574209898`
(main, 2026-10-05, merge of pull request 130). The repository is
`https://github.com/hugomoison/pyKMC`, Hugo Moison and Laurent Karim
Béland. This is not SourceForge lkmc 1.3 and not `hockyg/PyKMC`.
`develop` at `551a822` is an older divergent tip, so the pin stays on main.
LAMMPS is `stable_22Jul2025_update3`, commit `9f06a79b`. pARTn is
`9b753517`. IRA is `4bb13190`.

Chill et al., J. Chem. Theory Comput. 2014, doi:10.1021/ct5008718, and the
archived `benchmark.dat` files have no pyKMC row. `oracle/entries.jsonl`
has none either. There is no 2014 pyKMC configuration to copy.

A search that starts at a minimum uses the published k-ART push.
Béland, Mousseau, and coauthors, Comput. Mater. Sci. (2015),
doi:10.1016/j.commatsci.2014.11.047, deform one atom and its surroundings
by 0.2–0.5 Å and then take one to three perpendicular steps. Lanczos there
costs about 12–15 force evaluations. The pinned artn-plugin Pt(111)
heptamer, `examples/LAMMPS.Pt111.d/artn.in`, uses `ninit` 1, `nsmooth` 2,
`nnewchance` 2, `eigen_step_size` 0.25, `eigval_thr` -0.05,
`lanczos_disp` 1e-4, `lanczos_eval_conv_thr` 1e-2, and `push_over` 1.0.
`lammps.in` sets `dmax` 5.0 and `pair_style morse/smooth/linear`. That
example pushes along a file, one known hop, so a catalogue stays on
`push_mode rad` with `push_dist_thr` 3.0, which reaches the Pt neighbor
at r0 2.897. pyKMC's own default of 1.0 does not. The example `forc_thr`
is 0.01. The harness stop stays 0.001. The profile name in the result is
`beland-2015-pt111`. hTST keeps the archive prefactor 1.2e12. The aluminum
catalogues keep the eOn AKMC `default_value` 1e12. The paper's own runs
often use 1e13, and that value is not used here.

`pykmc/ts-pt` has `ninit` 0 and `push_mode` `list`. The
other pARTn values are the PartnConfig defaults that scored row used:
`eigval_thr` -0.01, `nsmooth` 3, `nnewchance` 0, `lanczos_disp` 0.0005,
`lanczos_eval_conv_thr` 0.001, `eigen_step_size` 0.2, Lanczos size 10 to
20, `push_over` 1, `dmax` 6. They are written in the template so the
minimum-search profile stays the one in the template. The archive has no
2014 pyKMC row.

`partn_search` does not forward a force norm. The worker sets
`converge_property` to `norm` and `forc_thr` to 0.001, because the problem
stop is the Euclidean norm and pARTn's own default is the maximum
component. `delr_thr` is 1e9, so pyKMC's 0.5 displacement filter does not
drop a saddle before the harness judges it. Fixed atoms are left out of
the LAMMPS system. `pair_style zero` contributes nothing. Forces come from
`fix external pf/callback` and `HarnessCounter` counts them. The baseline
saddle and hTST Pt templates use the archive shifted Morse, `kernel morse_shift`.
Aluminum uses the r2549 EAM routine. The 2014 Pt entries used that
shifted Morse.

The saddle link is the eOn r2205 test in `optbench.connection`: exactly
one endpoint has `perAtomNorm` strictly below 0.1 Å. `rmsd_connect` 0.5
on the free atoms within 1.5 Å of the highest free atom is the
saddle-novelty test. On the stored Pt saddle a hop moves one atom 2.71 Å.
The mean over all 175 free atoms is 0.290, and the same hop on the seven
island atoms is 1.449. The slab mean is stored as `rmsd_plus_free` and
`rmsd_minus_free`. It is not the accept test.

| problem | template | successes | failures | median calls | PAR-2 |
| --- | --- | --- | --- | --- | --- |
| tsopt/pt-island | pykmc/ts-pt | 46/49 | 1 cap, 2 crash | 746.5 | 1956.45 |
| hTST/Pt | pykmc/htst-pt | 13/13 | 0 | 3367604 | — |
| hTST/Al | pykmc/htst-al | 639/639 | 0 | 56329 | — |
| accelerated-md/aladatom | pykmc/aladatom | 3/3 | 0 | 166853 | 178325.67 |
| accelerated-md/standard_aladatom | pykmc/standard-aladatom | 3/3 | 0 | 119079 | 126131.67 |
| saddle-search/pt-island | pykmc/saddle-pt | 512/792 | 210 wrong stationary, 69 crash, 1 not converged | 1260.5 | 14983.76 |

`pykmc/ts-pt` is one search from each of the 49 guesses. `initial_13`
stopped at 325 calls and `initial_44` at 652, both `EVENT_NOT_FOUND`.
`initial_30` hit the cap of 10000. The success median is 746.5. PAR-2 is
1956.45. Each of the three failures costs 20000. The archive has no 2014 pyKMC row.

`pykmc/saddle-pt` is the shifted-Morse baseline. The reactant is already a
minimum of that potential, so minimization took 1 call.
`displace_magnitude` is 0. The central atom is one of atoms 0–6. The
catalogue stopped 160 searches after the last new unique saddle under
1.5 eV. That saddle is search 632, barrier 1.483 eV, so the run stops at
792 searches. The goal of 27 unique saddles under 1.5 eV was not met:
that count is 16. `unique_saddles` is 212. The campaign spent 1095478
harness calls. `nfailed` on the repetition is 1.

Successes are 512 of 792. `fraction_connected` is 512/792 =
0.6464646465. `fraction_eon_connected` is 513/792 = 0.6477272727.
Search 88 is linked and at the force stop with Hessian index 2, so
`apply_taxonomy` names it `wrong_stationary` and it is the extra link.
The other classes are 209 more `wrong_stationary`, 69 `crash`, and 1
`not_converged`. The 69 crashes store `ErrorType.EVENT_NOT_FOUND: No
event found`. The 69 `pykmc/saddle-pt` crashes and the two `pykmc/ts-pt`
crashes (`initial_13` at 325 calls, `initial_44` at 652) are pyKMC
`partn_search` returning `ErrorType.EVENT_NOT_FOUND: No event found`
when pARTn `get_error()` is ierr -10, `EIGENVALUE LOST, try to increase
nnewchance or nsmooth`. The `not_converged` search stopped on the method
budget at 10000 calls, under the harness cap of 20000. None of the 792 is
divergence and none spent the harness cap. The success median is 1260.5
calls. The median of the 792 call counts is 1315.5. Mean calls on the
successes are 1303.009765625. The v2 median of the penalised costs is
1513, bootstrap 1443 to 1600. PAR-2 is 14983.7638888889. Wilson 95% is
0.6125359640 to 0.6789793842.

The archived `eon-dimer` row is `fraction_connected` 0.746017,
`unique_saddles` 109.222, `force_calls` 3.77888e+06, `total_searches`
11366.8. Eon r2205, Sam Chill, 15 Jul 2014, shifted Morse. This pyKMC
row uses that potential and the r2205 link, and it stops on the harness
160-search stall.

Every search has a line in
`pykmc-0.0.1-saddle-search-pt-island--pykmc-saddle-pt.endpoints.jsonl`,
schema `optbench.endpoint-minima/v1`. The file has 792 rows. 722 rows
store both endpoint minima. 70 store neither. No row stores one side
only.

`pykmc/htst-pt` is a KMC campaign at 700 K with prefactor 1.2e12 on the
shifted Morse potential. The cycle rules are
`archive/entries/hTST/Pt/eon-dimer-lbfgs/report.py`. A barrier matches
only when `round(barrier, 3)` is still in the reference list. The
reference list has 15 barriers and the barrier-cycle target is 13. The
rate target is 0.999 of 155055901.244. The run stopped when the 13th
barrier window closed, on the search that found the last missing
reference process. It used 36096 searches and 374 processes, and matched
15 of 15 reference barriers. `problems/hTST/Pt.toml` has no
`convergence.max_force_calls`. `score_v2` therefore leaves `par_applied`
false and `par2_mean` unset. The eOn hTST Pt templates set `failure_cap`
2000000, and the board reads that template number because the problem
file states none. That number is not in the problem file, so this row
does not copy it. The 13 window costs are the scored starts, all successes.
`barrier_jobs` is 2776.076923, `barrier_force_calls` is 3833527,
`barrier_average_over` is 13. The median is 3367604. The bootstrap
interval is 2033873 to 5315178. The window costs run from 1424997 to
7695865. With `par2_mean` unset, the board orders the row by that median.
Rate windows are reported and do not decide the order: 386
closed, `rate_jobs` 93.31088083, `rate_force_calls` 128857.3912. The
archived `eon-dimer-lbfgs` row is barrier force calls 7.788e+05 averaged
over 13 and rate force calls 4.705e+04 averaged over 224. Dimer-LBFGS,
Eon r2006, Penghao Xiao, 28 Jan 2013. This mean barrier cost is 4.922
times that archive mean. The template ceiling is 250000 searches. The
run stopped at 13 windows, short of that ceiling.

`pykmc/htst-al` uses the same reducer at 300 K and prefactor 1.2e12, from
`archive/entries/hTST/Al/eon-dimer-cg/report.py`. The kernel is the r2549
aluminum EAM. The reference list is four 0.233 eV barriers and four
0.372 eV barriers. The barrier-cycle target is 639. The run resumed the
stored EAM checkpoint and stopped when the 639th barrier window closed,
after 46274 searches and 35 processes. It matched 8 of 8 reference
barriers. `problems/hTST/Al.toml` has no `convergence.max_force_calls`.
`score_v2` therefore leaves `par_applied` false and `par2_mean` unset.
The eOn hTST Al templates set `failure_cap` 2000000. That number is not
in the problem file, so this row does not copy it.
`barrier_jobs` is 72.41158059, `barrier_force_calls` is 64635.96401,
`barrier_average_over` is 639. The median is 56329. The bootstrap
interval is 53919 to 60108. The window costs run from 7710 to 268072.
With `par2_mean` unset, the board orders the row by that median.
No rate window closed. The HTST sum of the 35 catalogue
processes is 591004711.377. The cycle threshold is 0.999 × 592041917 =
591449875.083. The gap is 445163.706. A repeat of a process already in
the open window does not add rate. The archived `eon-dimer-cg` row is
barrier force calls 2.887e+04 averaged over 639 and rate force calls
8.979e+05 averaged over 20. Dimer-CG, Eon r2006, Penghao Xiao, 28 Jan
2013. This mean barrier cost is 2.239 times that archive mean. The
template ceiling is 400000 searches. The run stopped at 639 windows,
short of that ceiling.

`failure_cap` 2000000 was added to the six eOn hTST templates in commit
`1eb16a77` (2026-10-18). The hTST template notes cite no paper cap. The
same commit's basin-hopping templates, `templates/eon-svn/bh-lj38.toml`
and `templates/eon-client/bh-lj38.toml`, say the 2014 file stated no
force cap and that 2000000 is above the 2014 maximum of 1795901. The
hTST archive `config.ini` files set optimizer `max_iterations` 5000 and
state no force-call cap. `archive/entries/hTST/Pt/eon-dimer-lbfgs/report.py`
and the Aluminum twin sum saddle force calls until the barrier list and
the rate threshold are met, and they state no cap. Published archive
barrier means include Pt `alpha_dimer-qm` 1.908e7 and Pt `tsase-dimer-qm`
3.239e6, both above 2000000. `problems/globalopt/lj100-binary.toml` sets
`max_evaluations` 2000000 on that LJ problem. `problems/hTST/Pt.toml` and
`problems/hTST/Al.toml` stay without `convergence.max_force_calls`.
`optbench.campaign._check_stop` describes a template `failure_cap` as the
cap the archived report used. The hTST archive reports have no such
number. These rows stay scored with `par2_mean` unset.

`pykmc/aladatom` and `pykmc/standard-aladatom` stay on the minimized
reactant and stop when the old-scheme confidence `max(0, 1 - 1/repeats)`
reaches 0.95 and the thermal-window rate is positive. The prefactor is
the eOn AKMC `default_value` 1e12. The window, catalogue RMSD, and
product count are the problem values. The columns match the eOn AKMC
reducer. Each seed is one start. Three seeds finished at both
temperatures, so n is 3 and `nfailed` is 0. `infra/aladatom-repro` is
not on origin. The problem `sample.repetitions` stays 1, so the eOn n=1
rows stay ranked.

`rate_accuracy` is `max(0, 1 - abs(rate - reference) / reference)`. It is
0 when the catalogue rate is at least twice the reference. The catalogue
rate is the HTST sum over the AKMC processes. The reference is a
parallel-replica Langevin escape rate. Those are different estimators.
The eOn AKMC rows also score 0 against the same references.

At 250 K, `pykmc/aladatom` mean `true_rate` is 1831767358, standard error
59545363.9, `rate_accuracy` 0, mean `catalogue_accuracy` 0.4965483678,
mean force calls 178325.6667, PAR-2 178325.6667, median 166853.
`force_calls_per_simulated_time` is 6.533022707e+10. The mean rate is
7.20334 times the reference. Seed 1: 165 searches, 20 repeats, rate
1890960743, catalogue accuracy 0.5059949401, 209928 calls. Seed 2: 128
searches, 22 repeats, rate 1712677323, catalogue accuracy 0.4776552234,
166853 calls. Seed 3: 128 searches, 21 repeats, rate 1891664010,
catalogue accuracy 0.5059949401, 158196 calls. The eOn AKMC row on the
same problem, counter `client`, is `true_rate` 855329109.9,
`rate_accuracy` 0, `catalogue_accuracy` 0.2427275727, 108435 force
calls, 128 searches, 21 repeats. The archive `eon-pr-langevin` reference
is `true_rate` 2.542941e+08, file spelling `uncertanity` 1.785425e+06.
Langevin parallel replica, Juliana Duncan, 18 Oct 2012.

At 150 K, `pykmc/standard-aladatom` mean `true_rate` is 20288397.82,
standard error 2416030.378, `rate_accuracy` 0, mean `catalogue_accuracy`
0.4572934971, mean force calls 126131.6667, PAR-2 126131.6667, median
119079. `force_calls_per_simulated_time` is 511801886.3. The mean rate
is 24.2713 times the reference. Seed 1: 84 searches, 20 repeats, rate
22704080.98, catalogue accuracy 0.4471177715, 93065 calls. Seed 2: 136
searches, 22 repeats, rate 15456337.08, catalogue accuracy
0.4776552234, 166251 calls. Seed 3: 104 searches, 21 repeats, rate
22704775.4, catalogue accuracy 0.4471074965, 119079 calls. The eOn AKMC
row is `true_rate` 38023490, `rate_accuracy` 0, `catalogue_accuracy`
0.8257069791, 1890241 force calls, 2336 searches, 20 repeats. The
archive `eon-md-langevin` reference is `true_rate` 8.359e+05 and
uncertainty 7.211e+04. Juliana Duncan, 26 Oct 2012.

On the unminimized `pt-island.con` reactant, 343 atoms, 175 free, box
19.2088 × 19.0118 × 30.0, the campaign smooth-linear evaluator gives
energy −1771.358308 eV and a free-atom force norm of 0.030113. r2549
shifted Morse, `morse_shift` on the 9-vector cell, gives −1775.791150 eV
and a free-atom force norm of 9.43108e-7, so this geometry is a minimum
of the shifted potential. `pes.morse_pt` is the smooth-linear kernel, not
the shifted one. The smooth-linear energy is 4.432842 eV higher. The
largest free-atom force difference is 0.003255 eV/Å. The smooth-linear
campaign then minimizes, 14 calls, to −1771.358522 eV and a free-atom
force norm of 9.604e-4. The saddle template and the hTST Pt template use
shifted Morse.

The same smooth-linear kernel, `morse`, is what `eon-client` uses for
hTST Pt, the saddle-pt dimer, Lanczos, and high-temperature rows, and the
tsopt Pt rows, and what `eon-svn` uses for `saddle-pt-dimer-bounded`,
`saddle-pt-lanczos-bounded`, `saddle-pt-md`, `ts-pt-dimer-smooth`,
`ts-pt-lanczos-smooth`, and `ts-pt-lanczos-improved`. `eon-svn` hTST Pt,
`htst-pt-dimer-lbfgs` and `htst-pt-lanczos-lbfgs`, sets `kernel =
"morse_shift"`. `eon-svn/saddle-pt-dimer`, `saddle-pt-dimer-improved`, and
`saddle-pt-lanczos` name `morse_pt` and describe r2549 `Morse::force`, the
shifted potential. The develop saddle and tsopt Pt rows set `potential =
"morse_pt"`, and `benchmarks.bench_eon_develop.morse_cluster` is Morse
with no cutoff and the box ignored. On this reactant that energy is
40168.955936 eV and the force norm is 13.432784. The 2014 archive
`eon-dimer` saddle row is r2205 on shifted Morse. The pyKMC baseline
uses that shifted Morse. The no-cutoff develop potential is a third surface.

hTST Al is the r2549 aluminum EAM in pyKMC, `eon-svn`, and `eon-client`.
The eOn aladatom AKMC rows call the client's built-in `eam_al` and count
client force calls. pyKMC calls the harness r2549 EAM and counts
`HarnessCounter`. That is one EAM family. Bit identity of the two EAM
calls is open.

## Codes outside the harness rows

**softsaddle.** The tree under `softsaddle/` is MATLAB. `softsaddle/entry.toml`
is the April 2019 archive (mean force calls 423491.1 over 10 repetitions).
The GitHub workflow builds the Morse, Davidson, and Lanczos libraries and
runs a two-search MATLAB smoke. MATLAB is not installed here. That smoke is
not a harness campaign.

**CatLearn.** Commit `6a307f32` (2023-02-07). `setup.py` says 0.6.2 and
`catlearn/__init__.py` says 0.7.0. ASE 3.29 has no `ase.neb` (it is
`ase.mep.neb`). NumPy 2 rejects `np.argwhere` as a scalar index and rejects
a length-1 GP prediction stored in one gradient slot. After those two
edits, MLMin on LJ38 start 0 took 7 harness calls and 82 seconds and was
still at force 144678. Each new training point retrains the GP. A
1000-start minimization is absent.

**SEAKMC_py.** Commit `3d94a64`. `LammpsRunner` writes `pair_style` and
shells out to a LAMMPS binary. The requirements include pymatgen. There is
no potential callback the harness can wrap.

**OpenFLY.** Commit `991baf0` (2023-06-01). The potential in the tree is
EAM. There is no Morse class. The accelerated-md problems need EAM, which
the harness does not evaluate.

**SaddleSearch.jl.** Commit `47b4f35`. `Project.toml` requires Julia 1.9 or
1.10. Julia is not installed. The dimer entry point is `run!(method, E, dE,
x0, v0)`.

**CP2K.** v2026.2 has no build here. A FIST classical potential would still need
a harness callback, and the build is a separate stack from the running
campaigns.

**GULP.** The academic licence is registration-only. No tarball was
downloaded, and a registered binary is not redistributed here.

**i-PI.** Commit `a6b3503` (v3.3.1, 2026-10-17). Geometry optimisation is
`GeopMotion`, bound to beads, a cell, and a forcefield inside the socket
server. NEB and the string are in that same motion stack. The socket was
not wired to `pes.evaluate`.

**GPR_calculator.** Commit `c4d6327`. `setup.py` compiles C++ kernels and
requires cffi, mpi4py, and pyxtal. That extension has no build here.

**geodesic-interpolate.** Commit `476aa04` (v1.0.0). It builds an initial
path from a molecular bond list (`min_neighbors` default 4). It is not a
saddle search. It has no NEB row.

**pyGSM.** Commit `1b4d5a5`. `examples/ase_api_de_gsm.py` grows a string in
TRIC delocalized internals from a molecular topology. The dependency list
includes xtb. That path has no LJ38 row.

**phonopy.** hTST asks for barrier and rate force calls. phonopy is the
finite-displacement half of a prefactor. There is no EAM evaluator and no
force-call protocol on `hTST/Al` or `hTST/Pt`.

**ParSplice.** `https://gitlab.com/exaalt/parsplice`, commit `4a48ef8` on
master (2019-05-16). It schedules external molecular dynamics. The build is
C++ and MPI. The accelerated-md problems have no force-call protocol and no
EAM evaluator.

**TAMMBER.** Commit `05afc05`. Same class of build and the same missing EAM
evaluator. The published metrics are a rate and an uncertainty, not a
force-call sample.
