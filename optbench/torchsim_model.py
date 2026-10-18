"""TorchSim model that evaluates an OptBench potential.

The optimizer sees forces. The harness potential returns the gradient,
so the force is the negative of that gradient. Each evaluator is one
system. One call of an evaluator is one force evaluation, and the
evaluator owns the harness counter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from torch_sim.models.interface import ModelInterface
from torch_sim.state import SimState

Evaluate = Callable[[np.ndarray], tuple[float, np.ndarray]]


class SpecPotentialModel(ModelInterface):
    """Batch of harness potentials addressed by system index.

    ``evaluators[i]`` receives the Cartesian rows of system ``i`` and
    returns ``(energy, gradient)``. The gradient is ``dE/dx`` with one
    component per coordinate. The model returns ``forces = -gradient``.
    """

    def __init__(
        self,
        evaluators: Sequence[Evaluate],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if not evaluators:
            raise ValueError("SpecPotentialModel needs at least one evaluator")
        self.evaluators = list(evaluators)
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = True
        self._memory_scales_with = "n_atoms"

    def forward(self, state: SimState, **kwargs) -> dict[str, torch.Tensor]:
        if int(state.n_systems) != len(self.evaluators):
            raise ValueError(
                f"state has {state.n_systems} systems and the model has {len(self.evaluators)} evaluators"
            )
        positions = state.positions.detach().to(dtype=torch.float64).cpu().numpy()
        system_idx = state.system_idx.detach().cpu().numpy()
        energies: list[float] = []
        force_blocks: list[np.ndarray] = []
        for system_id, evaluate in enumerate(self.evaluators):
            rows = positions[system_idx == system_id]
            if rows.size == 0:
                raise ValueError(f"system {system_id} has no atoms")
            energy, gradient = evaluate(rows)
            grad = np.asarray(gradient, dtype=np.float64).reshape(rows.shape)
            if grad.shape != rows.shape:
                raise ValueError("gradient shape does not match the positions")
            energies.append(float(energy))
            force_blocks.append(-grad)
        energy_t = torch.tensor(energies, dtype=self.dtype, device=self.device)
        forces_t = torch.tensor(np.concatenate(force_blocks, axis=0), dtype=self.dtype, device=self.device)
        # Position-only FIRE and gradient descent still store a stress tensor.
        # The cell is held fixed, so this block stays zero.
        stress_t = torch.zeros((state.n_systems, 3, 3), dtype=self.dtype, device=self.device)
        return {"energy": energy_t, "forces": forces_t, "stress": stress_t}
