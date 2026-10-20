# Latest stock rows beside the 2014 archive

The archived `benchmark.dat` values are the paper rows. These runs do not
rebuild a paper-era client, tarball, pin, or compiler, and they do not
change an optimizer parameter to move a number toward the archive. Each
code is its latest release, stock. The problems, starts, and convergence
tests are the current harness.

pele is 0.1.2 at `b7081b932a060be11342fbe8287ac35da8ada879`. ASE is 3.29.0
at `f27c0005ae6a67ea419f996e728668865bfc1f86`. EON is SVN r2549, display
`EON (SVN r2549)`, built with the Fortran potentials. eOn is develop
`a7aa57472fb7d823953fd36b78cd755fa4842d30`, display
`eOn (develop a7aa57472fb7)`. eOn is a separate code from EON. One force
call is one harness evaluation.

The ranked cost is the v2 PAR-2 mean. A band row also has the per-image
mean, the harness total divided by 8. An hTST row whose reference window
did not close ranks the finished searches. `window_count` is 0 on that
row, and `force_calls` there is the mean search cost, not a window mean.

| Row | Archive | Latest stock | What the gap is |
| --- | --- | --- | --- |
| globalopt/lj38 pele basin hopping | 521595 calls, 0 failures, max 2533900 | PAR-2 5375223.79. 14 of 100 starts reach the Wales energy. Their mean is 1537313 (min 313041, max 2547405). 86 stop at the 3000000 cap | Temperature 1.0 and `RandomDisplacement` stepsize 1.0 are the constructors. The quench is `mylbfgs` at its constructor. The cap sits above the archived maximum |
| minimization/lj38 pele FIRE | 709 calls, 0 failures | PAR-2 2141.38 over 1000 starts. The mean of starts under the cap is 1264. v2 failures 48 (7 cap, 41 wrong stationary) | FIRE `dt` 0.1, `maxstep` 0.5, `dtmax` 1, the constructor |
| minimization/morse-bulk ASE FIRE | 147.47 | PAR-2 113.36, 0 failures, 100 starts | ASE 3.29.0 FIRE constructor. `maxstep` 0.2, `dt` 0.1, `dtmax` 1 |
| minimization/morse-bulk ASE LBFGS | 53.8 | PAR-2 53.36, 0 failures, 100 starts | Constructor memory 100, `alpha` 70, `maxstep` 0.2 |
| minimization/morse-bulk ASE PreconLBFGS | 34, 0 failures | PAR-2 59.83, 0 failures, 100 starts | Constructor memory 100, `maxstep` 0.04, Exp preconditioner `A=3` |
| minimization/morse-bulk EON SD | 196 calls, 0 failures, min 95, max 360 | PAR-2 20000. 100 of 100 starts use 10000 calls and stop on the cap. The force norm stays between 10.7 and 245 | r2549 constructor `sd_alpha` 0.1 and `max_move` 0.2. The step sits on `max_move` and the energy oscillates |
| minimization/morse-bulk eOn SD | 196 calls, 0 failures | PAR-2 20000. 100 of 100 starts use 10000 calls and stop on the cap. Same force-norm range | develop constructor step, same omission of `sd_alpha`, `sd_twopoint`, and `max_move` |
| minimization/morse-bulk pele LBFGS | 51, 0 failures | PAR-2 92.06, 0 failures, 100 starts | Constructor memory 4, `maxstep` 0.1, `maxErise` 1e-4, `H0` 0.1 |
| neb/Pt-heptamer EON CG | 120 per image, min 53, max 211 | Total PAR-2 929.385, per-image mean 116.173, 0 failures, 13 pairs | Climbing-image CG, `spring` 5, 8 images, force norm 0.01, `max_move` 0.2. The harness kernel is r2549 `Morse::force` |
| neb/Pt-heptamer eOn CG | 120 per image | Total PAR-2 1141.08, per-image mean 142.635, 0 failures, 13 pairs | Same band stop on develop `a7aa57472fb7`. The kernel is the same Fortran Morse force |
| hTST/Al EON dimer CG | barrier window 28870 calls and 48 jobs over 639 windows; rate window 897900 calls and 1511 jobs over 20 windows | 0 barrier windows and 0 rate windows in 64 passes. 64 searches, PAR-2 4802.58, total 307365. One good process matches 0.233 eV. Seven reference barriers are absent. 49 searches are not connected | Dimer CG, 50 rotations, angle 1. Confidence, displacement, and the prefactor stay at the r2549 constructor. Stock confidence does not end a reactant that has not registered the reference processes, so 64 passes is the observation |
| hTST/Al eOn dimer CG | same archive windows | 0 barrier windows and 0 rate windows in 64 passes. 64 searches, PAR-2 4721.53, total 302178. The process table has no reference barrier. 55 searches are not connected | Same dimer block on develop. The kernel is the same Fortran Aluminum EAM |

ASE 3.29.0 and pele 0.1.2 minimization rows above are the constructor
campaigns already stored for those pins. Basin hopping, both steepest-descent
campaigns, both heptamer bands, and both aluminum reactant searches were
run for this table. Nothing in those runs was adjusted after the first result.
