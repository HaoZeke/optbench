# Changelog

## 2026-10-25

Findings citations are commit shas. The six campaign files that `cells/optim/saddle-pt` no longer carries are cited at `02405b0d`. SoftSaddle scores are cited at `247b70de`, the commit that holds them. A branch name is rejected.

The three SoftSaddle adaptive HSphere rows cite `bd1a5c9f`. Their
harness columns are the benchmark fields in that file. Davidson
sm50 force_calls is 5298701.7, unique_saddles is 0, and
fraction_connected is 1. Lanczos sm0 is 1926653.9, 0.5, and
0.9682. Lanczos sm50 is 1582635.3, 0.7, and 0.93584. unique_saddles
is benchmark_unique_saddles, the mean of connected. Davidson's
50000 starts all hit the cap and record no saddle, so 0 stays.
A missing commit keeps the stored finding. A file that is present
and disagrees with the stored row is still an error.

The scored Davidson sm50 row is the fully counted bd1a5c9f file.
The Davidson library requests these gradients itself: Hessianominimo
asks for a base gradient and one displaced gradient on each of the
525 free coordinates, minimumMode asks once at the start, finiteH
asks once per product (up to 8), and the first LBFGS step asks for
one curvature gradient. 247b70de wraps Hessianominimo in
SuppressCharge, leaving that column build off the harness tally.
The template budget stays max_iterations 1000, and the row stays
0 converged.

## 2026-10-21

`integrate/batch-3` ranks the stock rows from the finished board branches.
Each ranked row is one code, one version, and one stock method or preset.
A finite-difference Hessian counts as one force call per coordinate.
PAR-2 charges each failure twice the cap. hTST Al states no cap, so those
rows leave `par2_mean` unset.

eOn develop `500350d9` on `neb/Pt-heptamer`, 13 starts, 0 failures:

- CG PAR-2 1141.076923076923
- FIRE PAR-2 678.9230769230769
- LBFGS PAR-2 319.53846153846155
- QuickMin PAR-2 1322.0

EON SVN r2549 on the same problem, 13 starts, 0 failures: CG 929.3846153846154, FIRE 678.9230769230769, LBFGS 336.7692307692308, QuickMin 1322.0.
eOn 3.6.0 LBFGS stays ranked at PAR-2 3517.6923076923076.

eOn develop `500350d9` on `double-ended-saddle-search/lj38`, 50 starts: doubly nudged CI-NEB PAR-2 1228.96, 6 failures.

eOn develop `500350d9` on `double-ended-saddle-search/lj38_2ts_separation`, 51 starts:

- Lanczos improved PAR-2 536.6666666666666, 0 failures
- Lanczos PAR-2 573.7450980392157, 0 failures
- dimer PAR-2 3236.3529411764707, 0 failures
- energy-weighted CI-NEB PAR-2 15260.333333333334, 0 failures
- CI-NEB lj/cut PAR-2 38080.529411764706, 0 failures
- CI-NEB ljcluster PAR-2 50000.450980392154, 2 failures
- doubly nudged CI-NEB PAR-2 72306.94117647059, 4 failures
- OCI-NEB PAR-2 95094.3137254902, 6 failures

n2p2 2.3.0 Kalman energy: `mlopt/C2H4` PAR-2 17397.12 (40/50 failures), `mlopt/emt-pt38` PAR-2 25.02 (0/50).
n2p2 2.3.0 Kalman force: `mlopt/C2H4` and `mlopt/emt-pt38` are 0/50 at PAR-2 60000.0.

hTST Al dimer, 639 windows, 0 failures. PAR-2 is unset. The mean of the recorded window costs is 45626.85758998435 for EON SVN r2549 and 48128.758998435056 for eOn develop `500350d9`.

The asv-perch job was failing every pull request after a clean comparison table. asv-spyglass exits 1 at its default factor of 1.1, and the action does not ignore that status. The workflow now passes `--factor 10`, the same ratio as `regression-threshold`, so only a slowdown past 10x fails the job.
