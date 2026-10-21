"""Train the PyAMFF network with Optax.

The network is one 10-7 tanh model per element. A linear layer is
``x @ W.T + b``, the same map as ``torch.nn.Linear``. Initial weights
and biases are uniform on [-1, 1). The loss is the PyAMFF squared error:
the energy term sums ``((E - Eref) / n_atoms)^2`` over images, and the
force term sums ``||F - Fref||^2 / n_atoms / 3`` over atoms. Gradients
are of that sum. When the force coefficient is zero the parameter
gradient does not include forces. The force RMSE is still computed,
because the joint RMSE stop uses it.

One harness call is one evaluation of that loss and its parameter
gradient. Adam and Rprop take one evaluation per epoch, at the weights
before the first step and after every later step. L-BFGS evaluates the
initial point, then each zoom line-search trial. The RMSE read at an
accepted L-BFGS point repeats the last trial and is a verification call.
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optbench.counter import HarnessCounter
from optbench.failures import classify_failure

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainSettings:
    method: str
    learning_rate: float | None
    hidden: tuple[int, ...]
    seed: int
    energy_coefficient: float
    force_coefficient: float
    grad_tol: float
    energy_rmse_tol: float
    force_rmse_tol: float
    cap: int
    memory_size: int = 10
    max_linesearch_steps: int = 20


def _enable_x64():
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    return previous


def _restore_x64(previous: bool) -> None:
    import jax

    jax.config.update("jax_enable_x64", previous)


def initial_parameters(prepared: dict, settings: TrainSettings, indices: list[int]):
    """One parameter tree per start, stacked on a leading axis.

    The key for start ``i`` is ``seed + i``. Each key is split across
    weight then bias, element by element, layer by layer.
    """
    import jax
    import jax.numpy as jnp

    elements = list(prepared["elements"])
    hidden = tuple(int(size) for size in settings.hidden)
    if not hidden:
        raise ValueError("the network needs a hidden layer")

    def one(index: int):
        key = jax.random.key(int(settings.seed) + int(index))
        sizes = {element: (int(prepared["n_fp"][element]), *hidden, 1) for element in elements}
        count = sum(2 * (len(sizes[element]) - 1) for element in elements)
        keys = jax.random.split(key, count)
        cursor = 0
        tree = {}
        for element in elements:
            layers = []
            width = sizes[element]
            for din, dout in zip(width[:-1], width[1:], strict=True):
                w_key = keys[cursor]
                b_key = keys[cursor + 1]
                cursor += 2
                layers.append(
                    {
                        "w": jax.random.uniform(
                            w_key, (dout, din), minval=-1.0, maxval=1.0, dtype=jnp.float64
                        ),
                        "b": jax.random.uniform(
                            b_key, (dout,), minval=-1.0, maxval=1.0, dtype=jnp.float64
                        ),
                    }
                )
            tree[element] = tuple(layers)
        return tree

    trees = [one(index) for index in indices]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


def _check_prepared(prepared: dict) -> tuple[list[str], int, int]:
    elements = list(prepared["elements"])
    n_images = int(prepared["n_images"])
    natoms = int(prepared["natoms"])
    if prepared["energies"].shape != (n_images,):
        raise ValueError("energies must be one value per image")
    if prepared["forces"].shape != (n_images, natoms, 3):
        raise ValueError("forces must be (images, atoms, 3)")
    for element in elements:
        fps = np.asarray(prepared["fps"][element])
        deriv = np.asarray(prepared["dgdr"][element])
        n_fp = int(prepared["n_fp"][element])
        if fps.ndim != 3 or fps.shape[0] != n_images or fps.shape[-1] != n_fp:
            raise ValueError(f"{element} fingerprints have shape {fps.shape}")
        if deriv.shape != (n_images, fps.shape[1], n_fp, natoms, 3):
            raise ValueError(f"{element} dgdr has shape {deriv.shape}")
    return elements, n_images, natoms


def _forward_factory(prepared: dict, settings: TrainSettings):
    """Build the unbatched loss. The parameter tree has no start axis."""
    import jax
    import jax.numpy as jnp

    elements, n_images, natoms = _check_prepared(prepared)
    use_force = float(settings.force_coefficient) != 0.0
    energy_coefficient = float(settings.energy_coefficient)
    force_coefficient = float(settings.force_coefficient)
    fps = {element: jnp.asarray(prepared["fps"][element], dtype=jnp.float64) for element in elements}
    dgdr = {element: jnp.asarray(prepared["dgdr"][element], dtype=jnp.float64) for element in elements}
    target_energy = jnp.asarray(prepared["energies"], dtype=jnp.float64)
    target_force = jnp.asarray(prepared["forces"], dtype=jnp.float64)

    def loss_metrics(params):
        energy = jnp.zeros((n_images,), dtype=jnp.float64)
        force = jnp.zeros((n_images, natoms, 3), dtype=jnp.float64)
        for element in elements:
            rows = fps[element]
            layers = params[element]

            def atom_energy(row, layers=layers):
                hidden = row
                last = len(layers) - 1
                for index, layer in enumerate(layers):
                    hidden = hidden @ layer["w"].T + layer["b"]
                    if index != last:
                        hidden = jnp.tanh(hidden)
                return hidden[0]

            atom = jax.vmap(jax.vmap(atom_energy))(rows)
            de_dg = jax.vmap(jax.vmap(jax.grad(atom_energy)))(rows)
            energy = energy + jnp.sum(atom, axis=-1)
            force = force + jnp.einsum("icf,icfaj->iaj", de_dg, dgdr[element])
        force = -force
        energy_se = jnp.sum(((energy - target_energy) / natoms) ** 2)
        force_se = jnp.sum(jnp.sum((force - target_force) ** 2, axis=-1) / natoms) / 3.0
        loss = energy_coefficient * energy_se
        if use_force:
            loss = loss + force_coefficient * force_se
        energy_rmse = jnp.sqrt(energy_se / n_images)
        force_rmse = jnp.sqrt(force_se / n_images)
        return loss, (energy_rmse, force_rmse)

    def evaluate(params):
        (loss, (energy_rmse, force_rmse)), grad = jax.value_and_grad(loss_metrics, has_aux=True)(params)
        leaves = jax.tree.leaves(grad)
        grad_max = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
        finite = jnp.isfinite(loss) & jnp.isfinite(energy_rmse) & jnp.isfinite(force_rmse) & jnp.isfinite(grad_max)
        return loss, grad, energy_rmse, force_rmse, grad_max, finite

    batched = jax.jit(jax.vmap(evaluate))
    metrics = jax.jit(jax.vmap(lambda params: loss_metrics(params)[1]))
    return elements, loss_metrics, batched, metrics


def _solver(settings: TrainSettings, max_linesearch_steps: int | None = None):
    import optax

    method = settings.method
    if method == "adam":
        if settings.learning_rate is None:
            raise ValueError("adam needs a learning rate")
        return optax.adam(
            float(settings.learning_rate),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            eps_root=0.0,
            nesterov=False,
        )
    if method == "rprop":
        if settings.learning_rate is None:
            raise ValueError("rprop needs a learning rate")
        return optax.rprop(float(settings.learning_rate))
    if method == "lbfgs":
        steps = settings.max_linesearch_steps if max_linesearch_steps is None else int(max_linesearch_steps)
        if steps < 1:
            raise ValueError("the line search needs at least one evaluation")
        return optax.lbfgs(
            learning_rate=None,
            memory_size=int(settings.memory_size),
            scale_init_precond=True,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=steps,
                initial_guess_strategy="one",
            ),
        )
    raise ValueError(f"unknown method {method}")


def _bucket(count: int, full: int) -> int:
    size = 1
    while size < count:
        size *= 2
    if size > full:
        return full
    return size


def _index_tree(tree, index):
    import jax.numpy as jnp

    chosen = jnp.asarray(list(index), dtype=jnp.int32)
    return jax_tree_index(tree, chosen)


def jax_tree_index(tree, chosen):
    import jax

    return jax.tree.map(lambda leaf: leaf[chosen], tree)


def _scatter_tree(dest, index, source, count: int):
    import jax
    import jax.numpy as jnp

    chosen = jnp.asarray(list(index)[:count], dtype=jnp.int32)

    def place(old, new):
        return old.at[chosen].set(new[:count])

    return jax.tree.map(place, dest, source)


def _moved(before, after) -> bool:
    import jax

    for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            return True
    return False


def _charge(counter: HarnessCounter, times: int) -> None:
    for _ in range(int(times)):
        counter.charge((float(counter.calls),))


def _verify(counter: HarnessCounter) -> None:
    with counter.verification():
        counter.charge((float(counter.verify_calls),))


def _blank(index: int, seed: int) -> dict:
    return {
        "index": int(index),
        "seed": int(seed) + int(index),
        "calls": 0,
        "verify": 0,
        "status": "running",
        "converged": False,
        "epoch": 0,
        "loss": None,
        "energy_rmse": None,
        "force_rmse": None,
        "grad_max": None,
        "error": "",
    }


def _host(value) -> np.ndarray:
    import jax

    return np.asarray(jax.device_get(value))


def _save_checkpoint(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    temporary.replace(path)


def _load_checkpoint(path: Path | None, indices: list[int], settings: TrainSettings):
    if path is None or not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if int(payload.get("version") or 0) != CHECKPOINT_VERSION:
        raise RuntimeError(f"{path} uses checkpoint version {payload.get('version')}")
    if list(payload.get("indices") or []) != list(indices):
        raise RuntimeError(f"{path} was written for a different start set")
    if payload.get("method") != settings.method or int(payload.get("cap")) != int(settings.cap):
        raise RuntimeError(f"{path} was written for a different method or cap")
    return payload


def _records(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        reasons: list[str] = []
        if row["status"] == "crash":
            reasons.append("crash")
        elif row["status"] == "divergence":
            reasons.append("divergence")
        elif row["status"] == "cap":
            reasons.append("cap")
        elif row["status"] != "success":
            reasons.append("divergence")
        failure = classify_failure(reasons)
        record = {
            "schema": "optbench.start/v2",
            "start": f"init_{int(row['index']):04d}",
            "seed": int(row["seed"]),
            "force_calls": int(row["calls"]),
            "verify_calls": int(row["verify"]),
            "counter": "harness",
            "comparable": True,
            "success": failure is None,
            "failure": failure,
            "reasons": reasons,
            "converged": bool(row["converged"]),
            "epoch": int(row["epoch"]),
            "energy_rmse": row["energy_rmse"],
            "force_rmse": row["force_rmse"],
            "loss": row["loss"],
            "grad_max": row["grad_max"],
        }
        if row["error"]:
            record["error"] = row["error"]
        records.append(record)
    return records


def _judge(rows, counters, settings: TrainSettings, observed, which: list[int]) -> None:
    loss, energy_rmse, force_rmse, grad_max, finite = observed
    for index in which:
        row = rows[index]
        if row["status"] != "running":
            continue
        row["loss"] = float(loss[index])
        row["energy_rmse"] = float(energy_rmse[index])
        row["force_rmse"] = float(force_rmse[index])
        row["grad_max"] = float(grad_max[index])
        calls = int(counters[index].calls)
        row["calls"] = calls
        row["verify"] = int(counters[index].verify_calls)
        if not bool(finite[index]):
            row["status"] = "divergence"
            continue
        stopped = row["grad_max"] < float(settings.grad_tol) or (
            row["energy_rmse"] < float(settings.energy_rmse_tol)
            and row["force_rmse"] < float(settings.force_rmse_tol)
        )
        if stopped and calls < int(settings.cap):
            row["status"] = "success"
            row["converged"] = True
        elif stopped or calls >= int(settings.cap):
            row["status"] = "cap"
            row["converged"] = bool(stopped)
        else:
            row["status"] = "running"


def train(
    prepared: dict,
    settings: TrainSettings,
    indices: list[int],
    *,
    checkpoint: Path | None = None,
    resume: bool = False,
    probe=None,
    log=None,
) -> list[dict]:
    """Train ``indices`` and return one start record each, in that order."""
    if not indices:
        raise ValueError("indices is empty")
    if int(settings.cap) < 1:
        raise ValueError("cap must be positive")
    previous = _enable_x64()
    try:
        return _train(prepared, settings, [int(index) for index in indices], checkpoint, resume, probe, log)
    finally:
        _restore_x64(previous)


def _train(prepared, settings: TrainSettings, indices, checkpoint, resume, probe, log) -> list[dict]:
    import jax

    elements, loss_metrics, evaluate, metrics = _forward_factory(prepared, settings)
    del elements
    rows = [_blank(index, settings.seed) for index in indices]
    counters = [HarnessCounter(cache=False, cap=int(settings.cap)) for _ in indices]
    params = initial_parameters(prepared, settings, indices)
    solver = _solver(settings)
    state = jax.vmap(solver.init)(params)
    updates = _compiled_updates(settings, solver, loss_metrics, probe)
    restored = _load_checkpoint(checkpoint, indices, settings) if resume else None
    if restored is not None:
        params = _as_jax(restored["params"])
        state = _as_jax(restored["state"])
        updates["grad"] = _as_jax(restored["grad"])
        updates["value"] = _as_jax(restored["value"])
        for row, saved, counter in zip(rows, restored["rows"], counters, strict=True):
            row.update(saved)
            counter.calls = int(row["calls"])
            counter.verify_calls = int(row["verify"])
            counter.cap = int(settings.cap)
    else:
        charged = _charge_running(rows, counters)
        _note_probe(probe, len(charged))
        result = evaluate(params)
        updates["grad"] = result[1]
        updates["value"] = result[0]
        for row in rows:
            row["epoch"] = 1
        _judge(rows, counters, settings, _scalars(result), list(range(len(indices))))
        _report(log, rows, 0)
        _snapshot(checkpoint, indices, settings, params, state, updates, rows)

    step = 0
    while any(row["status"] == "running" for row in rows):
        step += 1
        if settings.method == "lbfgs":
            params, state = _lbfgs_outer(
                rows,
                counters,
                settings,
                params,
                state,
                updates,
                metrics,
                probe,
            )
        else:
            params, state = _first_order_outer(rows, counters, settings, params, state, evaluate, updates, probe)
        if step == 1 or step % 100 == 0:
            _report(log, rows, step)
        if step % 200 == 0:
            _snapshot(checkpoint, indices, settings, params, state, updates, rows)
    _snapshot(checkpoint, indices, settings, params, state, updates, rows)
    return _records(rows)


def _charge_running(rows, counters) -> list[int]:
    charged = []
    for index, (row, counter) in enumerate(zip(rows, counters, strict=True)):
        if row["status"] != "running":
            continue
        if counter.calls >= counter.cap:
            row["status"] = "cap"
            row["calls"] = int(counter.calls)
            continue
        _charge(counter, 1)
        row["calls"] = int(counter.calls)
        charged.append(index)
    return charged


def _scalars(result):
    loss, _grad, energy_rmse, force_rmse, grad_max, finite = result
    return (
        _host(loss),
        _host(energy_rmse),
        _host(force_rmse),
        _host(grad_max),
        _host(finite),
    )


def _note_probe(probe, times: int) -> None:
    if probe is None:
        return
    for _ in range(int(times)):
        probe()


def _compiled_updates(settings: TrainSettings, solver, loss_metrics, probe):
    import jax
    import jax.numpy as jnp
    import optax

    cache: dict[tuple, object] = {}

    def _ping(_marker):
        probe()

    def scalar_loss(params):
        loss, _aux = loss_metrics(params)
        if probe is not None:
            jax.debug.callback(_ping, jnp.zeros((), dtype=jnp.float64))
        return loss

    def first_order(grad, state, params, active):
        new_updates, new_state = solver.update(grad, state, params)

        def blend(new, old):
            return jnp.where(active, new, old)

        kept_state = jax.tree.map(blend, new_state, state)
        masked = jax.tree.map(lambda leaf: jnp.where(active, leaf, jnp.zeros_like(leaf)), new_updates)
        return optax.apply_updates(params, masked), kept_state

    def lbfgs_one(max_steps: int):
        local = _solver(settings, max_steps)

        def one(params, state, value, grad):
            new_updates, new_state = local.update(
                grad,
                state,
                params,
                value=value,
                grad=grad,
                value_fn=scalar_loss,
            )
            new_params = optax.apply_updates(params, new_updates)
            nls = jnp.asarray(optax.tree.get(new_state, "num_linesearch_steps"), dtype=jnp.int32)
            new_value = optax.tree.get(new_state, "value")
            new_grad = optax.tree.get(new_state, "grad")
            return new_params, new_state, new_value, new_grad, nls

        return jax.jit(jax.vmap(one))

    cache["first"] = jax.jit(jax.vmap(first_order))
    cache["lbfgs"] = lbfgs_one
    cache["built"] = {}
    return cache


def _first_order_outer(rows, counters, settings, params, state, evaluate, updates, probe):
    import jax.numpy as jnp

    active = jnp.asarray([row["status"] == "running" for row in rows])
    if not bool(np.any(np.asarray(active))):
        return params, state
    # The gradient in ``state`` is not stored. Re-read it by evaluating only
    # through the update that was given the last gradient. The caller passes
    # the gradient by stashing it on the function attribute after each eval.
    grad = updates["grad"]
    params, state = updates["first"](grad, state, params, active)
    charged = _charge_running(rows, counters)
    if not charged:
        return params, state
    _note_probe(probe, len(charged))
    result = evaluate(params)
    updates["grad"] = result[1]
    updates["value"] = result[0]
    for index in charged:
        rows[index]["epoch"] = int(rows[index]["epoch"]) + 1
    _judge(rows, counters, settings, _scalars(result), charged)
    return params, state


def _lbfgs_outer(rows, counters, settings, params, state, updates, metrics, probe):
    import jax.numpy as jnp

    del probe
    full = len(rows)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["status"] != "running":
            continue
        room = int(settings.cap) - int(counters[index].calls)
        if room < 1:
            row["status"] = "cap"
            row["calls"] = int(counters[index].calls)
            continue
        groups[min(int(settings.max_linesearch_steps), room)].append(index)
    value = updates.get("value")
    grad = updates.get("grad")
    if value is None or grad is None:
        raise RuntimeError("L-BFGS has no stored value")
    touched: list[int] = []
    for max_steps, group in groups.items():
        bucket = _bucket(len(group), full)
        padded = list(group) + [group[-1]] * (bucket - len(group))
        key = (max_steps, bucket)
        if key not in updates["built"]:
            updates["built"][key] = updates["lbfgs"](max_steps)
        sliced_params = _index_tree(params, padded)
        sliced_state = _index_tree(state, padded)
        sliced_value = value[jnp.asarray(padded, dtype=jnp.int32)]
        sliced_grad = _index_tree(grad, padded)
        new_params, new_state, new_value, new_grad, nls = updates["built"][key](
            sliced_params, sliced_state, sliced_value, sliced_grad
        )
        nls_host = _host(nls)
        new_value_host = _host(new_value)
        moved_flags = []
        for local, index in enumerate(group):
            taken = int(nls_host[local])
            if taken > int(settings.cap) - int(counters[index].calls):
                raise RuntimeError(
                    f"start {rows[index]['index']} line search took {taken} "
                    f"with {int(settings.cap) - int(counters[index].calls)} left"
                )
            before = _index_tree(params, [index])
            after = _index_tree(new_params, [local])
            moved = _moved(before, after)
            moved_flags.append(moved)
            if taken == 0 and not moved:
                rows[index]["status"] = "divergence"
                rows[index]["calls"] = int(counters[index].calls)
                continue
            # Charge the kept line-search iterations. vmap also evaluates the
            # other branch of the solver's internal cond; that branch is discarded.
            _charge(counters[index], taken)
            rows[index]["calls"] = int(counters[index].calls)
            rows[index]["epoch"] = int(rows[index]["epoch"]) + 1
            if not np.isfinite(new_value_host[local]):
                rows[index]["status"] = "divergence"
                continue
            touched.append(index)
        params = _scatter_tree(params, group, new_params, len(group))
        state = _scatter_tree(state, group, new_state, len(group))
        value = value.at[jnp.asarray(group, dtype=jnp.int32)].set(new_value[: len(group)])
        grad = _scatter_tree(grad, group, new_grad, len(group))
        del moved_flags
    updates["value"] = value
    updates["grad"] = grad
    if not touched:
        return params, state
    energy_rmse, force_rmse = metrics(params)
    energy_host = _host(energy_rmse)
    force_host = _host(force_rmse)
    grad_max = _grad_max_batch(grad)
    finite = np.isfinite(energy_host) & np.isfinite(force_host) & np.isfinite(grad_max)
    # The accepted value was counted inside the line search. This RMSE is the
    # same point read again.
    for index in touched:
        if rows[index]["status"] == "running":
            _verify(counters[index])
            rows[index]["verify"] = int(counters[index].verify_calls)
    observed = (
        _host(value),
        energy_host,
        force_host,
        grad_max,
        finite,
    )
    _judge(rows, counters, settings, observed, touched)
    return params, state


def _grad_max_batch(grad) -> np.ndarray:
    import jax

    leaves = [np.asarray(jax.device_get(leaf)) for leaf in jax.tree.leaves(grad)]
    reduced = [np.max(np.abs(leaf), axis=tuple(range(1, leaf.ndim))) for leaf in leaves]
    return np.max(np.stack(reduced, axis=0), axis=0)


def _report(log, rows, step: int) -> None:
    if log is None:
        return
    active = [row for row in rows if row["status"] == "running"]
    grads = [row["grad_max"] for row in active if row["grad_max"] is not None]
    errors = [row["energy_rmse"] for row in active if row["energy_rmse"] is not None]
    calls = [int(row["calls"]) for row in rows]
    text = (
        f"{time.strftime('%H:%M:%S')} step {step} running {len(active)} done {len(rows) - len(active)} "
        f"calls {min(calls)} {float(np.median(calls)):.0f} {max(calls)} "
        f"max_grad {max(grads) if grads else float('nan'):.6g} "
        f"mean_energy_rmse {float(np.mean(errors)) if errors else float('nan'):.6g}"
    )
    log(text)


def _snapshot(path, indices, settings: TrainSettings, params, state, updates, rows) -> None:
    import jax

    if path is None:
        return
    _save_checkpoint(
        path,
        {
            "version": CHECKPOINT_VERSION,
            "indices": list(indices),
            "method": settings.method,
            "cap": int(settings.cap),
            "params": jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), params),
            "state": jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), state),
            "grad": jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), updates["grad"]),
            "value": np.asarray(jax.device_get(updates["value"])),
            "rows": [dict(row) for row in rows],
        },
    )


def _as_jax(tree):
    import jax
    import jax.numpy as jnp

    return jax.tree.map(lambda leaf: jnp.asarray(leaf), tree)
