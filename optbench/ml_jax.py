"""JAX evaluation of the mlopt loss.

The values match ``optbench.ml_landscape.loss_and_grad``. This module is
imported only when a campaign runs. The public test suite does not import it.
"""

from __future__ import annotations

import numpy as np

from optbench.ml_landscape import Landscape

_COMPILED: dict[tuple, object] = {}


def _compile(landscape: Landscape, energy_coefficient: float, force_coefficient: float):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    species = []
    for block in landscape.species:
        specs = [spec for spec in landscape.layout.specs if spec.species == block.symbol]
        species.append(
            (
                jnp.asarray(block.fingerprints),
                None if block.d_fingerprints is None else jnp.asarray(block.d_fingerprints),
                tuple((spec.offset, spec.size, spec.shape) for spec in specs),
            )
        )
    energies = jnp.asarray(landscape.energies)
    forces = jnp.asarray(landscape.forces)
    n_atoms = jnp.asarray(landscape.n_atoms)
    n_images = landscape.n_images
    norm_slices = tuple((spec.offset, spec.offset + spec.size) for spec in landscape.layout.specs)
    use_force = force_coefficient != 0.0

    def tensors(flat, stored):
        found = []
        for offset, size, shape in stored:
            found.append(flat[offset : offset + size].reshape(shape))
        return found

    def one(flat):
        predicted = jnp.zeros((n_images,))
        force = jnp.zeros_like(forces)
        for fingerprints, d_fingerprints, stored in species:
            weight0, bias0, weight1, bias1, weight2, bias2 = tensors(flat, stored)
            hidden0 = jnp.tanh(jnp.einsum("hf,iaf->iah", weight0, fingerprints) + bias0)
            hidden1 = jnp.tanh(jnp.einsum("qh,iah->iaq", weight1, hidden0) + bias1)
            atomic = jnp.einsum("q,iaq->ia", weight2[0], hidden1) + bias2[0]
            predicted = predicted + atomic.sum(axis=1)
            if use_force:
                sensitivity1 = (1.0 - hidden1**2) * weight2[0]
                pre = jnp.einsum("qh,iaq->iah", weight1, sensitivity1)
                sensitivity0 = (1.0 - hidden0**2) * pre
                dedg = jnp.einsum("hf,iah->iaf", weight0, sensitivity0)
                n_local, width = fingerprints.shape[1], fingerprints.shape[2]
                left = dedg.reshape(n_images, n_local * width)
                right = d_fingerprints.reshape(n_images, n_local * width, -1)
                mapped = -jnp.matmul(left[:, None, :], right).reshape(forces.shape)
                force = force + mapped
        residual = (predicted - energies) / n_atoms
        energy_loss = jnp.sum(residual * residual)
        if use_force:
            diff = force - forces
            force_loss = jnp.sum(jnp.sum(diff * diff, axis=-1) / (3.0 * n_atoms[:, None]))
            force_rmse = jnp.sqrt(force_loss / n_images)
        else:
            force_loss = jnp.zeros(())
            force_rmse = jnp.array(-1.0)
        loss = energy_coefficient * energy_loss + force_coefficient * force_loss
        energy_rmse = jnp.sqrt(energy_loss / n_images)
        return loss, (energy_rmse, force_rmse)

    def gnorm(gradient):
        return jnp.max(jnp.stack([jnp.linalg.norm(gradient[start:stop]) for start, stop in norm_slices]))

    def batch(parameters):
        (loss, aux), gradient = jax.vmap(jax.value_and_grad(one, has_aux=True))(parameters)
        energy_rmse, force_rmse = aux
        return loss, gradient, jax.vmap(gnorm)(gradient), energy_rmse, force_rmse

    return jax.jit(batch)


def loss_and_grad(
    parameters: np.ndarray,
    landscape: Landscape,
    *,
    energy_coefficient: float = 1.0,
    force_coefficient: float = 0.0,
) -> dict[str, np.ndarray]:
    """Same record as the NumPy loss, evaluated with JAX."""
    import os

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax.numpy as jnp

    flat = np.asarray(parameters, dtype=np.float64)
    if flat.ndim == 1:
        flat = flat[None, :]
    key = (id(landscape), float(energy_coefficient), float(force_coefficient), flat.shape[1])
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = _compile(landscape, float(energy_coefficient), float(force_coefficient))
        _COMPILED[key] = compiled
    loss, gradient, norms, energy_rmse, force_rmse = compiled(jnp.asarray(flat))
    return {
        "loss": np.asarray(loss, dtype=np.float64),
        "gradient": np.asarray(gradient, dtype=np.float64),
        "grad_norm": np.asarray(norms, dtype=np.float64),
        "energy_rmse": np.asarray(energy_rmse, dtype=np.float64),
        "force_rmse": np.asarray(force_rmse, dtype=np.float64),
    }
