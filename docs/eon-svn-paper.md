# eOn SVN paper revisions

The 2014 archive stamps eOn at r2025 (LJ38 minimization), r2030 (Morse), r2090 (Pt-heptamer NEB), r2156 (double-ended search, tsopt, lowest eigenvector, basin hopping), and r2205 (Pt-island saddle search). `docs/reproduction.md` is not in this repository. The comparison below uses each row's `report.sh`: the same integer fields, the same force-call cut, and the same failure rule. hTST, accelerated MD, and the r2213 bowl-breakout saddle rows are outside this set.

`svn export -r REV` fails. It follows copy-from history into paths the server rejects. `svn list -R -r REV` and `svn cat -r REV` of the pegged URL (`URL/path@REV`) export the tree. The r2549 pin stays the harness lane. These revisions are separate pins. Their templates are the archived `config.ini` with the last duplicate key kept, and nothing added.

## Build

Every client below is a Fortran build. `make` is invoked with `CXXFLAGS=-std=gnu++98` in the environment. Putting those flags on the `make` command line replaces the makefile's `-O2 -DNDEBUG -DLINUX` and, when requested, `-DLAMMPS_POT`. The potentials string in the version banner is written by the makefile, so a banner can say `+LAMMPS` while `Potential.cpp` was compiled without it.

QSC is the svn:external `QSC http://theory.cm.utexas.edu/svn/qsc/c` at r20 (2012-09-21). It is copied in. `client/Rules.mk` and `client/Potential.cpp` are not edited.

`client/version.sh` is not executable after a `cat` export. It is marked executable and left as it is. A `svnversion` stub on `PATH` prints the bare revision. The client adds the `r`. A stub that prints `r2030` produces the banner `rr2030`.

LAMMPS is the 2013-02-04 snapshot, commit `f727814b69` (SVN r9396), with the OPT package. `g++` 13 stops in `src/compute_ti.cpp` at the `ivar2 < 0` test because `ivar2` is `int*`. That file is not edited. The serial library is compiled with gcc 7.5.0 and linked into the g++ 13 client. gcc 7 cannot link the client itself: the conda `libm` uses RELR relocations. LJ38 L-BFGS, CG, FIRE, steepest descent, and basin hopping match the archive on this library, so the forces are the paper's `lj/cut/opt`.

| File | What changed | Why |
| --- | --- | --- |
| `client/Prefactor.cpp` at r2090 | Nothing. The compile line adds `-DprefactorFilterMode=prefactorFilterScheme`. | The source still names the old field. |
| `client/INIFile.cpp` `GetValueF`, on the r2204 and r2205 trees | A missing key returns `defValue`. A key the file sets is still parsed with `atof`. | `sprintf("%.15e", -DBL_MAX)` rounds above `DBL_MAX`. `atof` then raises `FE_OVERFLOW`. `ClientEON.cpp` calls `feenableexcept` for overflow when `LINUX` is defined, and it does that before the ini is read. r2159 defaults `basinHoppingStopEnergy` to `-DBL_MAX`. The default value is unchanged. |
| `eon/atoms.py` on the r2205 server | `if ibox == None` becomes `if ibox is None`. | NumPy 1.16 raises when `ibox` is the inverse box. The test is whether the caller omitted the inverse. |

No optimizer, dimer, Lanczos, NEB, or potential source was rewritten. Where a later revision matches, the rerun uses that revision's file:

- r2091 `FIRE.cpp` and `Parameters.cpp` on the r2090 tree. r2090's own next commit says it broke FIRE.
- r2158 `NudgedElasticBand.cpp` on r2156. One function: drop the net translation when every atom is free.
- r2159 `BasinHoppingJob.cpp`, `Parameters.cpp`, and `Parameters.h` on r2156. This is the `stop_energy` key.
- r2169 `Lanczos.cpp` on the r2204 tree. The only `Lanczos.cpp` difference from r2169 to r2204 is the loop bound. r2169 uses `i >= lanczosMaxIterations`. The basis is allocated to `lanczosMaxIterations`, so the last iteration reads one past the end. r2186 uses `lanczosMaxIterations - 1`. r2168, which adds `quit_early`, does not compile (`positions` is not declared).
- r2222 `ImprovedDimer.cpp` on the r2204 tree. The commit message is "rotation debug".

Min-mode displacement and direction for the double-ended rows come from the archived `interpolate.py` and `make_mode.py` under Python 2.7 and NumPy 1.16. `write_con` prints coordinates at `%.6f`.

## Archived against rerun

`mean / max / min`, then `median` and `nfailed` when the archive writes them. A cell marked exact matches every integer the report writes.

### LJ38 minimization

The report stamps r2025 and drops `total_force_calls >= 10000`. r2025 converges on the maximum atomic force and has no steepest descent. The unset metric becomes the norm at r2092. The matching client is r2156. Potential `lammps` is `lj/cut/opt` cutoff 40. QuickMin sets `potential = lj`.

| Row | Archive | r2025 | r2156 |
| --- | --- | --- | --- |
| L-BFGS | 181 / 405 / 90, nfailed 0 | 277 / 1333 / 143, 0 | exact |
| CG | 453 / 1153 / 207, 0 | 431 / 1143 / 197, 0 | exact |
| FIRE | 731 / 2918 / 227, 0 | 1032 / 9912 / 243, 2 | exact |
| steepest descent | 4901 / 9982 / 1355, 96 | no steepest descent | exact |
| two-point steepest descent | 539 / 2502 / 182, 0 | no steepest descent | exact |
| QuickMin | 3523 / 9929 / 667, 23 | 1921 / 7831 / 397, 0 | 3520 / 9715 / 574, 25 |

QuickMin stays a near-miss on the same r2156 binary that matches the five `lammps` rows. The potential is the built-in `lj` cutoff of 15. The trajectories are long. `LJ.cpp` was not changed.

### Morse bulk

The report stamps r2030 and uses the same cut of 10000. The potential is built-in `morse_pt`. r2030 does not match. r2204 matches four methods. L-BFGS matches the mean and the minimum. The maximum is one call higher, and only `pos_0049` moves (82 against 81).

| Row | Archive | r2030 | r2204 |
| --- | --- | --- | --- |
| L-BFGS | 52 / 81 / 35, 0 | 59 / 78 / 43, 0 | 52 / 82 / 35, 0 |
| CG | 106 / 183 / 67, 0 | 52 / 83 / 31, 0 | exact |
| FIRE | 156 / 212 / 107, 0 | 90 / 196 / 64, 0 | exact |
| steepest descent | 196 / 360 / 95, 0 | 82 / 101 / 50, 0 | exact |
| two-point steepest descent | 64 / 138 / 28, 0 | 28 / 36 / 20, 0 | exact |

Two r2030 starts were repeated on that binary: `pos_0001` is 43 and `pos_0096` is 78, the same calls as the stored r2030 set.

### Pt-heptamer NEB

The mean is `force_calls_neb / 8` for totals under 10000. `nfailed` is commented out. The potential is `morse_pt`. Eight images.

| Row | Archive | Rerun | Client |
| --- | --- | --- | --- |
| L-BFGS | 42 / 55 / 29 | exact | r2090 |
| CG | 120 / 211 / 53 | exact | r2090 |
| QuickMin | 164 / 227 / 89 | exact | r2090 |
| FIRE | 86 / 102 / 66 | 1000 / 1000 / 1000 on r2090, every band 8000 raw calls. Exact on r2091 | r2091 |

### Double-ended LJ38

The climbing-image reports cut the mean at 2500 and count `nfailed` as `termination_reason != 0`. The dimer and Lanczos reports cut the mean at 3000. The problem file's cap is 2500. Potential `lammps`, `lj/cut/opt` cutoff 40.

| Row | Archive | Rerun | Client |
| --- | --- | --- | --- |
| CI-NEB | 827 / 2347 / 407, median 788.5, nfailed 7 | exact | r2158 |
| doubly nudged CI | 861 / 2187 / 407, 793.5, 5 | 875 / 2142 / 407, 793.5, 5 | r2158 |
| dimer | 337 / 2079 / 89, 203.5, 0 | 327 / 2490 / 91, 208, 0 | r2156 and r2204, same integers |
| Lanczos | 364 / 2794 / 110, 241, 0 | 331 / 2794 / 110, 241, 0 | r2169 bound |
| Lanczos quit_early | 287 / 1623 / 85, 201, 0 | 282 / 1528 / 85, 201, 0 | r2169 bound |

r2156, before the translation removal, gives CI-NEB 956 / 2072 / 402, median 1668.5, nfailed 21. The r2204 Lanczos bound (the r2186 fix) keeps the minimum and the median and shortens the tail: plain Lanczos 284 / 939 / 110, 241, 0.

The dimer source does not change from r2156 through r2226. `LJ.cpp` is the same file from r2025 through r2204. The Pt dimer on the same interpolate script is exact, and LJ38 L-BFGS on this binary is exact. Writing the midpoint at `%.6f` changes 34 of the 50 searches relative to a higher-precision midpoint. r2156 and r2204 with the archived writer produce the same 50 integers. One search, run 32, is 2490 calls. The archive maximum under the 3000 cut is 2079. Start 13 is 3190 and falls outside the cut. The archive does not keep the per-start files.

On the r2169 Lanczos bound the minimum, the maximum, and the median of the plain row are exact. The mean is the searches that run into the one-past-the-end iteration. Those integers are deterministic on this g++ 13 build and are not the archived tail.

### Double-ended Pt island

The report cuts the mean at 3000. `nfailed` is `termination_reason != 0`. Potential `morse_pt`. The substrate is frozen, so the r2158 translation removal does not apply. Displacement for the dimer and Lanczos rows uses the archived scripts.

| Row | Archive | Rerun | Client |
| --- | --- | --- | --- |
| CI-NEB, 3 images | 248 / 761 / 86, 211, 0 | 260 / 1331 / 86, 211, 0 | r2156 |
| CI-NEB, 5 images | 391 / 797 / 162, 395, 0 | 391 / 777 / 162, 395, 0 | r2156 |
| CI-NEB, 1 image | 110 / 594 / 29, 62, 0 | 112 / 812 / 29, 62, 0 | r2156 |
| doubly nudged CI | 256 / 956 / 92, 219, 0 | 253 / 818 / 92, 219, 0 | r2156 |
| dimer | 182 / 322 / 74, 182, 0 | exact | r2204 |
| Lanczos | 325 / 2660 / 95, 207.5, 5 | 264 / 1580 / 95, 207.5, 5 | r2169 bound |
| Lanczos quit_early | 222 / 2039 / 72, 157, 8 | 278 / 2407 / 72, 157, 3 | r2169 bound |

The r2158 translation removal runs only when every atom is free. The Pt substrate is frozen, so that branch does not run, and these bands stay on r2156. Two climbing-image searches are longer than the archived maximum: run 44 is 1331 and run 53 is 821. The five-image band has no search above the archived maximum. The single-image band has one, run 44 at 812 against 594. The doubly nudged band has none. The r2204 Lanczos bound keeps the minimum and the median and loses the failures: plain 213 / 362 / 95, 207.5, nfailed 0; quit_early 166 / 308 / 72, 157, 0. Pt start 6 on the r2169 bound was repeated three times and returned termination 5 and 10101 calls each time. The archive maximum under 3000 is 2660. The cap-hitting path is the out-of-range basis read. It was not "fixed" to chase the tail.

### tsopt

The report cuts the mean at 10000 and counts `nfailed` as `termination_reason != 0`.

Pt island, potential `morse_pt`. The mode is the normalized displacement from the reactant, with no periodic wrap, from the archived script.

| Row | Archive | Rerun | Client |
| --- | --- | --- | --- |
| dimer | 116 / 160 / 83, median 114, 0 | 120 / 269 / 82, 115.5 on r2156 and r2204. Exact on r2222 | r2222 |
| Lanczos | 143 / 292 / 106, 139, 0 | exact | r2204 |
| Lanczos quit_early | 106 / 163 / 71, 100, 0 | exact on r2204. 259 / 1986 / 71, 112, 3 on the r2169 bound | r2204 |

The r2222 commit is 18 Jun 2014. The archive dates the dimer row 20 Jun 2014 and stamps r2156. The rotation debug is the change that matches.

LJ38 tsopt writes `direction.dat` with unseeded `np.random.random((38, 3))`. The index is ignored. The archive does not store the modes. A fresh draw is not the paper's sample, so those three rows were not scored as integers. The configs are `templates/eon-svn-r2156/ts-lj38-dimer.toml`, `templates/eon-svn-r2204/ts-lj38-lanczos.toml`, and `templates/eon-svn-r2204/ts-lj38-lanczos-improved.toml`. The dimer row is dated the day of r2156. The Lanczos rows set `quit_early`, which r2156 does not read.

### Lowest eigenvector

The report cuts the mean at 10000 and writes `nfailed` 0. The mode is `RandomState(314159)`, reseeded on every attempt, and the reference modes are in the geometry archive. Potential `lammps`, `lj/cut/opt` cutoff 40. Calls are steps plus 3.

| Row | Archive | r2156 | r2204 | r2222 |
| --- | --- | --- | --- | --- |
| Lanczos | 28 / 73 / 13, 0 | exact |  |  |
| dimer CG | 30 / 93 / 14, 0 | 51 / 274 / 16 | 51 / 274 / 16 | exact |
| dimer L-BFGS | 28 / 70 / 14, 0 | 50 / 335 / 16 | 39 / 92 / 21 | exact |

The archive dates the dimer rows 17 Jun 2014 and the r2222 commit is 18 Jun 2014. The integers match r2222.

### Basin hopping

LJ38, r2159 (r2156 plus the `stop_energy` files). The report averages every run and writes no `nfailed`. The archive is 508011 / 1795901 / 4579. The rerun is exact. The config sets `stop_energy = -173.927427`, `random_seed = 314159`, and `lj/cut/opt` cutoff 40. r2156 ignores `stop_energy` and would run to `1e7` steps.

Binary LJ100, stamped r2156 and dated 21 May 2014, records energies only: minimum -596.13, maximum -574.14, average -584.579. There are 100 starts and 60000 steps, and the config sets no `random_seed` and no `stop_energy`. With `random_seed` omitted, `Parameters.cpp` keeps the default -1 and then sets the seed from `time(NULL)`. The archive stores neither the seeds nor `bh.dat`. A rerun is a different sample. A 3-step probe on r2156 with the archived `in.lammps` (`lj/cut` 20 and the binary coefficients) returned 0, energy -540.80, and 1213 force calls. A 30-step probe took 0.925 s and 11376 force calls. At that rate one archived trajectory is about half an hour. The energies were not rerun.

### Pt-island saddle search

r2205, potential `morse_pt`. `config.yaml` defaults `random_seed` to -1, and the archived ini does not set it. The server then leaves NumPy unseeded. The client, given a negative seed, uses `time(NULL)`. `done.py` stops at 27 good saddles below 1.5 eV. The archive has no force-call cap. The dimer mean is 3.77888e6 with mean error 527337, 109.222 unique saddles, connected fraction 0.746017, and 11366.8 searches. The dynamics mean is 1.478168e6 with mean error 118736. The replicates are independent of the archived sample.

Six dimer replicates finished. The rows are 1933552 force calls, 97 unique saddles, connected fraction 0.741324, 5590 searches; 4542407, 121, 0.744922, 13145; 3348750, 109, 0.746985, 9703; 3446880, 112, 0.735450, 9983; 9654589, 138, 0.744721, 27848; and 6988583, 138, 0.745085, 20195. Their averages are 4.986e6 force calls, 119.2 unique saddles, connected fraction 0.743, and 14411 searches. The archived connected fraction is 0.746017. The force-call mean error on the archive is 5.27e5, and the sample standard deviation that implies is about 1.67e6. Two of the six replicates are longer than that band. The dynamics replicates follow the ten dimer replicates in the same driver and had not been started with these six. The server is the r2205 Python tree, with `ibox is None` in `atoms.py`, and the client is the r2205 Fortran binary.
