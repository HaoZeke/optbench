# CP2K

GEO_OPT transition-state search in CP2K 2026.2 is the dimer with conjugate gradients. The transition-state line search implements 2PNT. GOLD and FIT abort.

BFGS and L-BFGS are unsupported for `TRANSITION_STATE` in commit `702cc3d81422b92864ecdd7f544c21c56c04adca`. The optimizers abort before a step:

- `src/motion/bfgs_optimizer.F` line 187 calls `CPABORT("BFGS method not yet working with DIMER")` when the geometry task is a transition state.
- `src/motion/cp_lbfgs_geo.F` line 116 calls `CPABORT("L-BFGS method not yet working with DIMER")` on the same task.

Those combinations are not ranked.

A default CG row keeps the stock iteration limit, `MAX_ITER` 200, and the stock force thresholds, `MAX_FORCE` 4.5e-4 Ha/bohr and `RMS_FORCE` 3.0e-4 Ha/bohr (`src/start/input_cp2k_motion.F`). The socket multiplies the force by 0.05, and the written thresholds are multiplied by the same factor. CP2K tests the dimer-modified gradient. The harness tests the raw Euclidean norm. That norm stays above 0.001 eV/Å, under the call cap, so the class is divergence. `DR` is the stock 0.01 angstrom. `ANGLE_TOLERANCE` in these rows is 1 degree; the stock default is 5 degrees. The two-point `MAX_ALLOWED_STEP` in these rows is 0.05 bohr; the stock default is 0.25.
