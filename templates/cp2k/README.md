# CP2K

GEO_OPT transition-state search in CP2K 2026.2 is the dimer with conjugate gradients. The transition-state line search implements 2PNT. GOLD and FIT abort.

BFGS and L-BFGS are unsupported for `TRANSITION_STATE` in commit `702cc3d81422b92864ecdd7f544c21c56c04adca`. The optimizers abort before a step:

- `src/motion/bfgs_optimizer.F` line 187 calls `CPABORT("BFGS method not yet working with DIMER")` when the geometry task is a transition state.
- `src/motion/cp_lbfgs_geo.F` line 116 calls `CPABORT("L-BFGS method not yet working with DIMER")` on the same task.

Those combinations are not ranked.

A default CG row keeps the stock iteration limit, `MAX_ITER` 200, and the stock force thresholds, `MAX_FORCE` 4.5e-4 Ha/bohr and `RMS_FORCE` 3.0e-4 Ha/bohr (`src/start/input_cp2k_motion.F`). The socket multiplies the force by 0.05, and the written thresholds are multiplied by the same factor. CP2K tests the dimer-modified gradient. The harness tests the raw Euclidean norm of the free-atom force.

That norm stays above 0.001 eV/Å on every stock start. The energy and the geometry stay finite, and the call count stays under the cap. The runner records `force_norm` together with the raw final energy, geometry, and force. It does not relabel the row. The central classifier scores that record as not_converged: a finished search still above the force stop, not a blow-up. The library convergence banner is present on 122 of 200 LJ38 starts and 41 of 49 Pt starts. A Pt start with the banner has the dimer-modified gradient inside the stock force thresholds while the raw norm is still above 0.001. An LJ38 start without the banner reached `MAX_ITER` with the raw norm still above 0.001. `DR` is the stock 0.01 angstrom. `ANGLE_TOLERANCE` in these rows is 1 degree; the stock default is 5 degrees. The two-point `MAX_ALLOWED_STEP` in these rows is 0.05 bohr; the stock default is 0.25. The tuned rows use those same three settings and meet the harness stop, so the stock failure is not the dimer length, the angle tolerance, or the two-point step. It is the stock force stop and iteration limit, applied to the dimer-modified gradient.
