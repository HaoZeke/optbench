# Saddle and kMC codes in this lane

Lane list of 2026-10-18. A code is a module in `optbench/engines`, a pin in
`codes/registry.toml`, and a template under `templates/`. `campaign.run`
calls `optbench.engines.find`. `available` returns the reason a code cannot
start. `run` writes harness campaign files or raises that reason. The score
is the harness force count. A client's own tally is not a score.

Pt-island geometries use the Morse potential in `optbench.pes` (`morse_pt`).
A code runs those problems when it can take that potential or an ASE or
LAMMPS calculator that calls it. The harness has no EAM evaluator. hTST and
accelerated-md problems state rates and barriers, not a force-call cap, so
they are not given a force-call row.

## Rows on disk

Failure classes are the v2 counts. They sum to `nfailed`. The median is the
success-only median. A row with no success has no median.

| problem | template | successes | failures | median calls |
| --- | --- | --- | --- | --- |
| lowest-eigenvector/lj38 | partn/mode-lj38 | 174/200 | 26 divergence | 22 |
| tsopt/pt-island | partn/ts-pt | 45/49 | 4 crash | 247 |
| tsopt/LJ38 | partn/ts-lj38 | 0/200 | 200 crash | — |
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
| minimization/lj38 | dlfind/lbfgs-lj38 | 998/1000 | 1 cap, 1 wrong stationary | 184 |
| minimization/lj38 | dlfind/sd-lj38 | 990/1000 | 10 cap | 2563.5 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s0 | 998/1000 | 1 cap, 1 wrong stationary | 184 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em4 | 998/1000 | 2 wrong stationary | 184 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em3 | 999/1000 | 1 wrong stationary | 209 |
| minimization/lj38-noise | dlfind/lbfgs-lj38-noise-s1em2 | 974/1000 | 26 wrong stationary | 199 |
| minimization/morse-bulk | dlfind/lbfgs-morse | 32/100 | 68 wrong stationary | 77 |
| minimization/morse-bulk | dlfind/sd-morse | 13/100 | 87 wrong stationary | 253 |
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

The rgsaddle band campaigns on `neb/Pt-heptamer` and the three double-ended
problems are on disk (12 templates each). The saddle-search catalog on
`saddle-search/pt-island` is still running and has not written a file. A
sample of that process was on search 387 of the first repetition, with 1
connected saddle. The eight rgmin `lj38-noise` L-BFGS rows are on disk.
Both tunings set `maxstep` 0.2, and each pair matches. Every start met the
force stop. The v2 failures are Hessian index not 0. Zero noise is 996 of
1000. `1e-4` is 994 of 1000. `1e-3` is 989 of 1000. `1e-2` is 884 of 1000,
116 wrong stationary, success median 161.

pARTn on `tsopt/LJ38` stops with `pARTn error -9: BOX EXPLOSION` on every
start. The Lennard-Jones push leaves the cluster. The same build's mode
search and the Morse tsopt row are the other two pARTn point scores.
`nevalf` is not the count.

TSASE MDMin on LJ38 hits the cap of 10000 on every start. The step limit
keeps the force above the stop. The cap record has no final force norm,
because the cap is raised before that geometry is stored. The class is cap.

## OPTIM and PATHSAMPLE

Pin `optim`. The snapshot is `http://www-wales.ch.cam.ac.uk/svn.tar.bz2`,
122743078 bytes, sha256
`02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f`.
There is no git commit. `OPTIM/source/py.f90` is the Paramonov–Yaliraki
ellipsoid potential, not a Python hook. The external hook in the Wales tree
is `INTERFACES/pysocket/ase_interface.f90`. That socket was not built, and
the extracted tree has no `optim` binary. `optim/unhooked` writes no number.
OPTIM's own counter is not a score. PATHSAMPLE is in the same snapshot and
has the same blocker.

## pARTn

Pin `artn`, 1.1.0, DEVEL `9b753517`. `libartn.so` is the engine plugin and
the standalone `next_displ` loop. Morse uses `lammps/metal`. Lennard-Jones
uses `lammps/lj`. The saddle-search catalog on `saddle-search/pt-island`
is running. It has not written a file.

## TSASE

Pin `tsase`, commit `98aac60` (2025-09-26). `setup.py` says 1.0. The
checkout has no LICENSE file. Fortran calculators use `numpy.distutils` and
are not built. SSNEB and the solid-state dimer are loaded without
`tsase/__init__.py`. `ss` is false. A periodic all-free Hessian drops the
three translations before the index is counted.

The LJ38 dimer hits the cap on every start. The Pt-island dimer and the
climbing-image NEB on Pt-heptamer are the saddle rows. The plain NEB on
that heptamer diverges on all 13 pairs. Morse bulk L-BFGS meets the force
stop and the Hessian index is not 0 for 88 of 100 starts. The success
median is 53.5.

Still open, and not given an invented budget: basin hopping and minima
hopping on `globalopt/*`. Those problems state `energy_above_minimum` and
do not state `max_force_calls`. AMD-SVM on `accelerated-md/aladatom` is not
scored. The problem has no force-call protocol and the harness has no EAM
potential. hTST prefactors are not scored for the same reason.

## LAMMPS

Pin `lammps`, `stable_30Sep2026`, commit `8de817dd`, version string
`30 Sep 2026`. Minimizers `cg`, `sd`, `hftn`, `quickmin`, and `fire` call
the harness through `fix external pf/callback` and `pair_style zero`. A
native pair style is not the score. FIRE and quickmin use timestep 0.002.
The velocity kick is not limited by `dmax`, and a larger timestep loses
atoms on the first LJ38 step.

On Morse bulk the force stop is met and the Hessian index is not 0 for 95
of 100 conjugate-gradient starts, 96 of 100 steepest-descent starts, 97 of
100 Hessian-free Newton starts, and 95 of 100 quickmin starts. FIRE and the
LJ38 noise rows are not in these rows. `fix neb` needs a multi-partition MPI
run and is not scored. `prd`, `tad`,
`hyper`, and `dynamical_matrix` are not scored. The Al, Pt, and aladatom
problems are EAM, and the harness has no EAM evaluator.

## DL-FIND

Pin `dlfind`, libdlfind 0.0.3, commit `4167998d`. `iopt` 3 is L-BFGS, 0 is
steepest descent, and 1 is conjugate gradient. The gradient callback is the
score. Coordinates stay in the harness unit. The component tolerance is the
Euclidean stop divided by `sqrt` of the free components. On Morse bulk L-BFGS the force stop is met and the Hessian index is not 0
for 68 of 100 starts. Steepest descent on LJ38 is 990 of 1000, 10 cap,
success median 2563.5. Steepest descent on Morse bulk is 13 of 100, 87
wrong stationary, success median 253. Conjugate gradient and the
steepest-descent noise rows are not in these rows. NEB (`icoord` 190) and
the dimer (`icoord` near 210) are not in these rows.

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
L-BFGS row does the same on every start. Conjugate gradient and steepest
descent were not run. The dimer drops translation and not rotation. The NEB
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

The lane name is SourceForge lkmc release 1.3, not `hockyg/PyKMC`. The
archive is Perl from February 2010. `TransSearchSSH.pl` shells out to
`LBOMD.exe`, which is not in the zip. There is no licence file and no
potential callback. `pykmc/lattice` writes nothing.

## Codes that were not scored

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
1000-start minimization was not launched.

**SEAKMC_py.** Commit `3d94a64`. `LammpsRunner` writes `pair_style` and
shells out to a LAMMPS binary. The requirements include pymatgen. There is
no potential callback the harness can wrap.

**OpenFLY.** Commit `991baf0` (2023-06-01). The potential in the tree is
EAM. There is no Morse class. The accelerated-md problems need EAM, which
the harness does not evaluate.

**SaddleSearch.jl.** Commit `47b4f35`. `Project.toml` requires Julia 1.9 or
1.10. Julia is not installed. The dimer entry point is `run!(method, E, dE,
x0, v0)`.

**CP2K.** v2026.2 was not built. A FIST classical potential would still need
a harness callback, and the build is a separate stack from the running
campaigns.

**GULP.** The academic licence is registration-only. No tarball was
downloaded, and a registered binary is not redistributed here.

**i-PI.** Commit `a6b3503` (v3.3.1, 2026-10-17). Geometry optimisation is
`GeopMotion`, bound to beads, a cell, and a forcefield inside the socket
server. NEB and the string are in that same motion stack. The socket was
not wired to `pes.evaluate`.

**GPR_calculator.** Commit `c4d6327`. `setup.py` compiles C++ kernels and
requires cffi, mpi4py, and pyxtal. That extension was not built.

**geodesic-interpolate.** Commit `476aa04` (v1.0.0). It builds an initial
path from a molecular bond list (`min_neighbors` default 4). It is not a
saddle search. It was not paired with a NEB.

**pyGSM.** Commit `1b4d5a5`. `examples/ase_api_de_gsm.py` grows a string in
TRIC delocalized internals from a molecular topology. The dependency list
includes xtb. That path was not run on LJ38.

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
