# pyGSM climb step clamps only a positive tangent projection

Upstream: https://github.com/ZimmermanGroup/pyGSM

Pin: `master` `1b4d5a5557a61adc6a3df7405d6b7255b1fa4865` (2026-08-13)

## Summary

`base_optimizer.get_constraint_steps` limits a CLIMB step to `0.05 / SCALE_CLIMB` only when the tangent projection of the gradient is positive. A negative projection is used as the displacement. On a 38-atom Lennard-Jones cluster that step walks atoms out of the cluster during `optimize_string`.

## Where

`pyGSM/optimizers/base_optimizer.py`, `get_constraint_steps`, `opt_type == "CLIMB"`:

```python
gts = np.dot(g.T, molecule.constraints[:, 0])
max_step = 0.05/self.SCALE_CLIMB
if gts > np.abs(max_step):
    gts = np.sign(gts)*max_step
constraint_steps = gts*molecule.constraints[:, 0]
```

`DMAX` caps the perpendicular eigenvector-follow step. It does not cap this constraint step. `abs_max_step` (0.4) is not applied here. `DQMAG_MAX` is the single-ended `add_node` branch (`nodeP` absent) and is not this double-ended climb.

The same one-sided comparison is repeated for `opt_type == "TS-SEAM"` a few lines below. The double-ended growing string hits the `CLIMB` branch.

## What happens

The gradient stored for the harness Lennard-Jones kernel is O(1) in the displacement units. `gts` is that projection onto the tangent. When `gts` is negative and larger in magnitude than 0.05, the code skips the clamp and sets the constraint displacement to `gts * tangent`.

Measured on LJ38 pair 00, Cartesian primitives, `nnodes` 11, `DMAX` 0.1, stock climb step, one BLAS thread:

- Growth frames stayed near a nearest-neighbour distance of 1.11.
- At evaluation 855, during `optimize_string` / `NoLineSearch`, the largest atomic step was 4.2437, the step RMS was 0.9916, `dq` was 6.1128, the energy was −116.702, and the largest nearest-neighbour distance was 2.7310.
- The search then returned a dissociated cluster: 1271 force calls, 85.4 s, energy −8.801e−7, force 1.464e−7, largest nearest-neighbour distance 904.84.

A dissociated Lennard-Jones cluster has no pairs inside the cutoff, so the energy and the force are both about zero. That point is not a saddle of the intact cluster.

That measurement stored the reduced Lennard-Jones energy as a Hartree. `ASELoT` divides an eV energy by the Hartree before the same storage (`level_of_theories/ase.py`), and the harness follows that conversion. A 1 ε/Å projection is then 0.037 in the displacement units, under the 0.05 cap. The one-sided comparison is unchanged.

## Fix

Clamp on the absolute projection. `np.sign` already preserves the direction:

```python
if np.abs(gts) > np.abs(max_step):
    gts = np.sign(gts)*max_step
```

The same edit belongs on the `TS-SEAM` climb. The diff below is the `CLIMB` branch, which is the double-ended path:

```diff
--- a/pyGSM/optimizers/base_optimizer.py
+++ b/pyGSM/optimizers/base_optimizer.py
@@ -368,7 +368,7 @@
             gts = np.dot(g.T, molecule.constraints[:, 0])
             # stepsize=np.linalg.norm(constraint_steps)
             max_step = 0.05/self.SCALE_CLIMB
-            if gts > np.abs(max_step):
+            if np.abs(gts) > np.abs(max_step):
                 gts = np.sign(gts)*max_step
                 # constraint_steps = constraint_steps*max_step/stepsize
             print(" gts %1.4f" % gts)
```

With that clamp, the same pair 00 stayed one cluster through the 2500-call cap (energy −170.218, force 12.231, nearest neighbour 1.111). Pair 04 did the same (energy −170.068, force 8.613, nearest neighbour 1.109). The clamp is a local change on top of commit `1b4d5a55`. It is not a stock pyGSM setting.
