"""PANNA Adam steps on the mlopt squared-error loss.

The TensorFlow trainer is the Adam object constructed in
``src/panna/neuralnet/config_parser_train.py``. The JAX trainer is the
``optax.adam`` object constructed in ``src/panna/train_jax.py`` when
``l1_coef`` is zero. Neither trainer supplies the loss. One harness call
is one full-batch evaluation of that loss and its parameter gradient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.ml_landscape import (
    FP_SHA256,
    HIDDEN,
    INIT_SEED,
    Landscape,
    SpeciesBlock,
    build_landscape,
    initial_parameters,
    layout_for,
    loss_and_grad,
    parse_fingerprints,
)

PANNA_COMMIT = "132a2df662598ecbbb2c9b8d37b992bb6ada6348"
PANNA_VERSION = "2.1.0"
TF_ADAM_LINE = "epsilon=training_params.getfloat('adam_eps', 1e-7)"
JAX_ADAM_LINE = "tx = optax.adam(schedule)"


def assert_panna_pin() -> None:
    """Fail closed when the import is not the registry source pin."""
    from importlib.metadata import version

    import panna

    found = version("panna")
    if found != PANNA_VERSION:
        raise RuntimeError(f"PANNA version is {found}, expected {PANNA_VERSION}")
    package = Path(panna.__file__).resolve().parent
    jax_source = (package / "train_jax.py").read_text(encoding="utf-8")
    tf_source = (package / "neuralnet" / "config_parser_train.py").read_text(encoding="utf-8")
    if JAX_ADAM_LINE not in jax_source:
        raise RuntimeError("PANNA JAX trainer does not construct optax.adam")
    if TF_ADAM_LINE not in tf_source:
        raise RuntimeError("PANNA TensorFlow Adam does not default epsilon to 1e-7")
    repo = package.parents[1]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != PANNA_COMMIT:
        raise RuntimeError(f"PANNA checkout is {commit}, expected {PANNA_COMMIT}")


class TrainingPaused(RuntimeError):
    """The checkpoint is current and this process is stopping on purpose."""


class _TensorflowAdam:
    """The Adam object PANNA constructs for its TensorFlow trainer."""

    def __init__(self, parameters: np.ndarray, learning_rate: float):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam

        self._tf = tf
        self.variable = tf.Variable(np.array(parameters, dtype=np.float64, copy=True), dtype=tf.float64)
        self.optimizer = Adam(
            learning_rate=float(learning_rate),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
        )

    def __call__(self, values: np.ndarray, gradient: np.ndarray, active: np.ndarray) -> np.ndarray:
        masked = np.array(gradient, dtype=np.float64, copy=True)
        masked[~active] = 0.0
        self.optimizer.apply_gradients([(self._tf.constant(masked, dtype=self._tf.float64), self.variable)])
        updated = np.array(self.variable.numpy(), dtype=np.float64, copy=True)
        if not bool(np.all(active)):
            updated[~active] = values[~active]
            self.variable.assign(updated)
        return updated

    def state_dict(self) -> dict:
        return {
            "values": np.array(self.variable.numpy(), dtype=np.float64, copy=True),
            "slots": [np.array(variable.numpy(), copy=True) for variable in self.optimizer.variables],
        }

    def load_state_dict(self, state: dict) -> None:
        slots = list(state["slots"])
        if len(self.optimizer.variables) != len(slots):
            zeros = self._tf.zeros_like(self.variable)
            self.optimizer.apply_gradients([(zeros, self.variable)])
        if len(self.optimizer.variables) != len(slots):
            raise RuntimeError("TensorFlow Adam checkpoint does not match this optimizer")
        for variable, saved in zip(self.optimizer.variables, slots, strict=True):
            variable.assign(np.asarray(saved))
        self.variable.assign(np.asarray(state["values"], dtype=np.float64))


class _JaxAdam:
    """The optax.adam object PANNA constructs when l1_coef is zero."""

    def __init__(self, parameters: np.ndarray, learning_rate: float):
        os.environ["JAX_ENABLE_X64"] = "1"
        import jax
        import optax

        jax.config.update("jax_enable_x64", True)
        signature_eps = __import__("inspect").signature(optax.adam).parameters["eps"].default
        if float(signature_eps) != 1e-8:
            raise RuntimeError(f"optax.adam epsilon default is {signature_eps}, expected 1e-8")
        self._jax = jax
        self._optax = optax
        self.optimizer = optax.adam(float(learning_rate))
        self.state = self.optimizer.init(jax.numpy.asarray(parameters))

    def __call__(self, values: np.ndarray, gradient: np.ndarray, active: np.ndarray) -> np.ndarray:
        current = self._jax.numpy.asarray(values)
        masked = np.array(gradient, dtype=np.float64, copy=True)
        masked[~active] = 0.0
        updates, self.state = self.optimizer.update(self._jax.numpy.asarray(masked), self.state, current)
        updated = np.array(self._optax.apply_updates(current, updates), dtype=np.float64, copy=True)
        if not bool(np.all(active)):
            updated[~active] = values[~active]
        return updated

    def state_dict(self) -> dict:
        return {"tree": self._jax.tree.map(lambda leaf: np.array(leaf, copy=True), self.state)}

    def load_state_dict(self, state: dict) -> None:
        self.state = self._jax.tree.map(self._jax.numpy.asarray, state["tree"])


def _tensorflow_adam(parameters: np.ndarray, learning_rate: float) -> _TensorflowAdam:
    return _TensorflowAdam(parameters, learning_rate)


def _jax_adam(parameters: np.ndarray, learning_rate: float) -> _JaxAdam:
    return _JaxAdam(parameters, learning_rate)


def _record(
    *,
    index: int,
    calls: int,
    success: bool,
    failure: str | None,
    grad_norm: float | None,
    energy_rmse: float | None,
    force_rmse: float | None,
    seconds: float,
    template: dict,
) -> dict:
    return {
        "schema": "optbench.start/v2",
        "counter": "harness",
        "comparable": True,
        "template": template["id"],
        "problem": template["problem"],
        "code": template["code"],
        "code_version": template["code_version"],
        "commit": template["commit"],
        "tuning": template.get("tuning") or template.get("profile") or "default",
        "start": str(index),
        "force_calls": int(calls),
        "success": bool(success),
        "failure": failure,
        "grad_norm": None if grad_norm is None else float(grad_norm),
        "energy_rmse": None if energy_rmse is None else float(energy_rmse),
        "force_rmse": None if force_rmse is None else float(force_rmse),
        "seconds": float(seconds),
    }


def _parameter_key(row: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in row)


def _batch_width(active_count: int, full: int) -> int:
    """Largest power-of-two bucket that still covers the active starts.

    The compiled loss stays on that width until the active set shrinks into
    the next bucket. Finished starts are left out of the smaller batch.
    """
    width = full
    for size in (32, 16, 8, 4, 2, 1):
        if active_count <= size:
            width = size
    return min(width, full)


def _evaluate_active(evaluate, parameters, chosen, landscape, *, energy_coefficient, force_coefficient):
    width = _batch_width(int(chosen.size), int(parameters.shape[0]))
    if width == parameters.shape[0]:
        return evaluate(
            parameters,
            landscape,
            energy_coefficient=energy_coefficient,
            force_coefficient=force_coefficient,
        )
    if chosen.size < width:
        pad = np.full(width - int(chosen.size), int(chosen[0]), dtype=int)
        taken = np.concatenate([chosen, pad])
    else:
        taken = chosen
    partial = evaluate(
        parameters[taken],
        landscape,
        energy_coefficient=energy_coefficient,
        force_coefficient=force_coefficient,
    )
    result = {
        "loss": np.zeros(parameters.shape[0], dtype=np.float64),
        "grad_norm": np.zeros(parameters.shape[0], dtype=np.float64),
        "energy_rmse": np.zeros(parameters.shape[0], dtype=np.float64),
        "force_rmse": np.zeros(parameters.shape[0], dtype=np.float64),
        "gradient": np.zeros_like(parameters),
    }
    real = int(chosen.size)
    result["loss"][chosen] = partial["loss"][:real]
    result["grad_norm"][chosen] = partial["grad_norm"][:real]
    result["energy_rmse"][chosen] = partial["energy_rmse"][:real]
    result["force_rmse"][chosen] = partial["force_rmse"][:real]
    result["gradient"][chosen] = partial["gradient"][:real]
    return result


def _save_checkpoint(
    path: Path,
    parameters: np.ndarray,
    active: np.ndarray,
    counters: list[HarnessCounter],
    opened: list[float],
    finished: list[dict | None],
    stepper,
) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.perf_counter()
    payload = {
        "parameters": np.asarray(parameters, dtype=np.float64),
        "active": np.asarray(active, dtype=bool),
        "calls": np.asarray([counter.calls for counter in counters], dtype=np.int64),
        "elapsed": np.asarray([now - stamp for stamp in opened], dtype=np.float64),
        "finished": np.asarray(["" if row is None else json.dumps(row) for row in finished]),
    }
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    temporary.replace(path)
    if hasattr(stepper, "state_dict"):
        stepper_path = path.with_suffix(".stepper")
        stepper_tmp = stepper_path.with_suffix(".tmp")
        with stepper_tmp.open("wb") as handle:
            pickle.dump(stepper.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        stepper_tmp.replace(stepper_path)


def _load_checkpoint(
    path: Path,
    counters: list[HarnessCounter],
    stepper,
) -> tuple[np.ndarray, np.ndarray, list[dict | None], np.ndarray]:
    import pickle

    saved = np.load(path, allow_pickle=False)
    parameters = np.array(saved["parameters"], dtype=np.float64)
    active = np.array(saved["active"], dtype=bool)
    calls = np.array(saved["calls"], dtype=np.int64)
    elapsed = np.array(saved["elapsed"], dtype=np.float64)
    if calls.shape != (len(counters),) or active.shape != (len(counters),):
        raise RuntimeError(f"{path} does not match this start list")
    finished: list[dict | None] = []
    for raw in saved["finished"]:
        text = str(raw)
        finished.append(None if text == "" else json.loads(text))
    for counter, count in zip(counters, calls, strict=True):
        counter.calls = int(count)
    stepper_path = path.with_suffix(".stepper")
    if stepper_path.is_file():
        if not hasattr(stepper, "load_state_dict"):
            raise RuntimeError(f"{stepper_path} needs a stepper that can load its state")
        with stepper_path.open("rb") as handle:
            stepper.load_state_dict(pickle.load(handle))
    return parameters, active, finished, elapsed


def train_starts(
    landscape: Landscape,
    template: dict,
    *,
    cap: int,
    tolerance: float,
    energy_coefficient: float,
    force_coefficient: float,
    stepper,
    starts: list[int],
    seed: int,
    progress: bool = True,
    evaluate=loss_and_grad,
    checkpoint: Path | None = None,
    checkpoint_every: int = 25,
    pause_after: int | None = None,
) -> list[dict]:
    """Train each start until the parameter-set stop or the harness cap.

    ``evaluate`` is called on the whole batch so a compiled loss keeps one
    shape. A repeated parameter vector is a harness cache hit and is recorded
    as divergence. ``checkpoint`` stores the batch after an Adam step.
    """
    if cap < 1:
        raise ValueError("cap must be positive")
    parameters = np.stack([initial_parameters(landscape.layout, start, seed) for start in starts]).astype(np.float64)
    active = np.ones(len(starts), dtype=bool)
    counters = [HarnessCounter(cache=True, cap=cap) for _start in starts]
    finished: list[dict | None] = [None] * len(starts)
    elapsed = np.zeros(len(starts), dtype=np.float64)
    if checkpoint is not None and Path(checkpoint).is_file():
        parameters, active, finished, elapsed = _load_checkpoint(Path(checkpoint), counters, stepper)
        if progress:
            done = sum(row is not None for row in finished)
            print(f"resumed {Path(checkpoint)} with {done} finished starts", flush=True)
    if progress:
        print("compiling the loss on the full batch", flush=True)
    started = time.perf_counter()
    evaluate(
        parameters,
        landscape,
        energy_coefficient=energy_coefficient,
        force_coefficient=force_coefficient,
    )
    if progress:
        print(f"loss compiled in {time.perf_counter() - started:.1f}s", flush=True)
    opened = [time.perf_counter() - float(elapsed[index]) for index in range(len(starts))]
    guard = int(max(counter.calls for counter in counters))
    seen = 0
    while bool(np.any(active)):
        guard += 1
        seen += 1
        if guard > cap + 1:
            raise RuntimeError("the harness cap did not stop the training loop")
        for index in np.flatnonzero(active):
            before = counters[index].calls
            key = _parameter_key(parameters[index])
            try:
                fresh, _cached = counters[index].charge(key)
            except ForceCallCap:
                finished[index] = _record(
                    index=starts[index],
                    calls=counters[index].calls,
                    success=False,
                    failure="cap",
                    grad_norm=None,
                    energy_rmse=None,
                    force_rmse=None,
                    seconds=time.perf_counter() - opened[index],
                    template=template,
                )
                active[index] = False
                continue
            if not fresh or counters[index].calls == before:
                finished[index] = _record(
                    index=starts[index],
                    calls=counters[index].calls,
                    success=False,
                    failure="divergence",
                    grad_norm=None,
                    energy_rmse=None,
                    force_rmse=None,
                    seconds=time.perf_counter() - opened[index],
                    template=template,
                )
                active[index] = False
                continue
            counters[index].remember(key, (0.0, ()))
        if not bool(np.any(active)):
            break
        chosen = np.flatnonzero(active)
        try:
            result = _evaluate_active(
                evaluate,
                parameters,
                chosen,
                landscape,
                energy_coefficient=energy_coefficient,
                force_coefficient=force_coefficient,
            )
        except Exception as exc:
            for index in chosen:
                finished[index] = _record(
                    index=starts[index],
                    calls=counters[index].calls,
                    success=False,
                    failure="crash",
                    grad_norm=None,
                    energy_rmse=None,
                    force_rmse=None,
                    seconds=time.perf_counter() - opened[index],
                    template=template,
                )
                active[index] = False
            if progress:
                print(f"loss evaluation crashed: {exc}", flush=True)
            continue
        gradient = np.zeros_like(parameters)
        for index in chosen:
            calls = counters[index].calls
            grad_norm = float(result["grad_norm"][index])
            loss_value = float(result["loss"][index])
            row_grad = result["gradient"][index]
            finite = np.isfinite(loss_value) and np.isfinite(grad_norm) and bool(np.isfinite(row_grad).all())
            energy_rmse = float(result["energy_rmse"][index])
            force_rmse = float(result["force_rmse"][index])
            if not finite:
                kind = "divergence"
                success = False
                keep = False
            elif grad_norm <= tolerance and calls < cap:
                kind = None
                success = True
                keep = False
            elif calls >= cap:
                kind = "cap"
                success = False
                keep = False
            else:
                kind = None
                success = False
                keep = True
            if keep:
                gradient[index] = row_grad
                continue
            finished[index] = _record(
                index=starts[index],
                calls=calls,
                success=success,
                failure=kind,
                grad_norm=grad_norm,
                energy_rmse=energy_rmse,
                force_rmse=force_rmse,
                seconds=time.perf_counter() - opened[index],
                template=template,
            )
            active[index] = False
        if bool(np.any(active)):
            try:
                parameters = np.asarray(stepper(parameters, gradient, active), dtype=np.float64)
            except Exception as exc:
                for index in np.flatnonzero(active):
                    finished[index] = _record(
                        index=starts[index],
                        calls=counters[index].calls,
                        success=False,
                        failure="crash",
                        grad_norm=None,
                        energy_rmse=None,
                        force_rmse=None,
                        seconds=time.perf_counter() - opened[index],
                        template=template,
                    )
                    active[index] = False
                if progress:
                    print(f"Adam step crashed: {exc}", flush=True)
        if checkpoint is not None and checkpoint_every > 0 and (seen % checkpoint_every == 0 or not bool(np.any(active))):
            _save_checkpoint(Path(checkpoint), parameters, active, counters, opened, finished, stepper)
        if progress and (guard == 1 or seen == 1 or guard % 25 == 0 or not bool(np.any(active))):
            running = np.flatnonzero(active)
            if running.size:
                norms = np.asarray(result["grad_norm"], dtype=np.float64)[running]
                detail = f" grad max {float(np.max(norms)):.4g} median {float(np.median(norms)):.4g}"
            else:
                detail = ""
            print(f"epoch {guard} active {int(running.size)} of {len(starts)}{detail}", flush=True)
        if pause_after is not None and seen >= pause_after and bool(np.any(active)):
            if checkpoint is None:
                raise ValueError("pause_after requires a checkpoint")
            _save_checkpoint(Path(checkpoint), parameters, active, counters, opened, finished, stepper)
            raise TrainingPaused(str(checkpoint))
    if any(row is None for row in finished):
        raise RuntimeError("a start finished without a record")
    return [row for row in finished if row is not None]


def _clear_checkpoint(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)
    path.with_suffix(".stepper").unlink(missing_ok=True)


def _cache_path(cache: Path, positions: np.ndarray, fingerprint_text: str, with_forces: bool) -> Path:
    digest = hashlib.sha256()
    digest.update(positions.tobytes())
    digest.update(fingerprint_text.encode("utf-8"))
    digest.update(b"forces" if with_forces else b"energy")
    digest.update(b"v1")
    return cache / f"{digest.hexdigest()}.npz"


def load_landscape(root: Path, template: dict, problem: dict, *, cache: Path | None = None) -> Landscape:
    from optbench.campaign import geometry_archive

    fingerprint_path = root / str(template["panna"]["fingerprints"])
    text = fingerprint_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(fingerprint_path.read_bytes()).hexdigest()
    problem_name = str(problem["id"]).rsplit("/", 1)[-1]
    expected = FP_SHA256.get(problem_name)
    if expected is not None and digest != expected:
        raise RuntimeError(f"{fingerprint_path} sha256 is {digest}, expected {expected}")
    parameters = parse_fingerprints(text)
    hidden = tuple(int(value) for value in template["panna"]["hidden"])
    if tuple(hidden) != HIDDEN:
        raise ValueError("the mlopt network is hidden widths 10 and 7")
    frame_path = geometry_archive(root, problem)
    positions, symbols, cell, pbc, energies, forces = _read_trajectory(frame_path)
    force_coefficient = float(template["panna"]["force_coefficient"])
    store = cache or Path(os.environ.get("OPTBENCH_PANNA_CACHE", "/tmp/optbench-panna-fp"))
    store.mkdir(parents=True, exist_ok=True)
    path = _cache_path(store, positions, text, force_coefficient != 0.0)
    if path.is_file():
        saved = np.load(path)
        fingerprints = saved["fingerprints"]
        derivatives = saved["derivatives"] if "derivatives" in saved.files else None
        blocks = []
        symbol_array = np.asarray(symbols)
        counts = parameters.counts()
        cursor = 0
        for element in parameters.elements:
            width = counts[element]
            index = np.flatnonzero(symbol_array == element).astype(np.int32)
            block_g = fingerprints[:, cursor : cursor + len(index), :width]
            block_dg = None if derivatives is None else derivatives[:, cursor : cursor + len(index), :width]
            blocks.append(
                SpeciesBlock(
                    symbol=element,
                    atom_index=index,
                    fingerprints=np.ascontiguousarray(block_g),
                    d_fingerprints=None if block_dg is None else np.ascontiguousarray(block_dg),
                )
            )
            cursor += len(index)
        return Landscape(
            species=tuple(blocks),
            energies=energies,
            forces=forces,
            n_atoms=np.full(len(energies), positions.shape[1], dtype=np.float64),
            layout=layout_for(counts, parameters.elements, HIDDEN),
            symbols=tuple(symbols),
        )
    landscape = build_landscape(
        positions,
        symbols,
        cell,
        pbc,
        energies,
        forces,
        parameters,
        with_forces=force_coefficient != 0.0,
        hidden=HIDDEN,
    )
    packed_g = np.concatenate([block.fingerprints for block in landscape.species], axis=1)
    payload = {"fingerprints": packed_g}
    if force_coefficient != 0.0:
        payload["derivatives"] = np.concatenate([block.d_fingerprints for block in landscape.species], axis=1)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    temporary.replace(path)
    return landscape


def _read_trajectory(path: Path):
    from ase.io import read

    frames = read(str(path), index=":")
    if not frames:
        raise ValueError(f"{path} has no frames")
    symbols = tuple(frames[0].get_chemical_symbols())
    cell = np.asarray(frames[0].cell.array, dtype=np.float64)
    pbc = np.asarray(frames[0].pbc, dtype=bool)
    positions = []
    energies = []
    forces = []
    for frame in frames:
        if tuple(frame.get_chemical_symbols()) != symbols:
            raise ValueError(f"{path} changes the atom order")
        if np.max(np.abs(np.asarray(frame.cell.array) - cell)) > 1e-8:
            raise ValueError(f"{path} changes the cell")
        positions.append(np.asarray(frame.get_positions(), dtype=np.float64))
        energies.append(float(frame.get_potential_energy()))
        forces.append(np.asarray(frame.get_forces(), dtype=np.float64))
    return (
        np.stack(positions),
        symbols,
        cell,
        pbc,
        np.asarray(energies, dtype=np.float64),
        np.stack(forces),
    )


def _require_problem_budget(template: dict, problem: dict) -> tuple[int, float]:
    panna = template["panna"]
    convergence = problem["convergence"]
    force_coefficient = float(panna["force_coefficient"])
    energy_coefficient = float(panna["energy_coefficient"])
    if abs(energy_coefficient - 1.0) > 1e-15:
        raise ValueError("the energy coefficient is 1")
    optimizer = template["optimizer"]
    if abs(float(optimizer["learning_rate"]) - 0.01) > 1e-15:
        raise ValueError("the learning rate is 0.01")
    if abs(float(optimizer["beta_1"]) - 0.9) > 1e-15 or abs(float(optimizer["beta_2"]) - 0.999) > 1e-15:
        raise ValueError("Adam uses beta1 0.9 and beta2 0.999")
    trainer = str(panna["trainer"])
    expected_eps = 1e-7 if trainer == "tensorflow" else 1e-8
    if trainer not in {"tensorflow", "jax"}:
        raise ValueError(f"unknown PANNA trainer {trainer}")
    if abs(float(optimizer["epsilon"]) - expected_eps) > 1e-18:
        raise ValueError(f"{trainer} Adam epsilon is {expected_eps}")
    if trainer == "tensorflow" and optimizer.get("amsgrad") is not False:
        raise ValueError("TensorFlow Adam does not use AMSGrad")
    if int(panna["initialization_seed"]) != INIT_SEED:
        raise ValueError("the initialization seed is 20261015")
    if force_coefficient == 0.0:
        cap = int(convergence["max_epochs"])
        tolerance = float(convergence["energy_only_grad_norm"])
    elif abs(force_coefficient - 0.1) < 1e-15:
        cap = int(convergence["force_max_epochs"])
        tolerance = float(convergence["force_training_grad_norm"])
    else:
        raise ValueError("the force coefficient is 0 or 0.1")
    stop = template["stop"]
    if int(stop["max_force_calls"]) != cap:
        raise ValueError("template max_force_calls is not the problem epoch cap")
    if abs(float(stop["grad_norm"]) - tolerance) > 1e-15:
        raise ValueError("template grad_norm is not the problem gradient stop")
    full = int((problem.get("sample") or {})["initializations"])
    if full != 50:
        raise ValueError("the mlopt start count is 50")
    return cap, tolerance


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def _success_stats(records: list[dict], key: str) -> tuple[float, float] | None:
    values = [float(row[key]) for row in records if row["success"] and row.get(key) is not None]
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    return sum(values) / len(values), median


def write_campaign(directory: Path, stem: str, template: dict, records: list[dict], cap: int, *, partial: bool) -> Path:
    from optbench.score import score_v2

    directory.mkdir(parents=True, exist_ok=True)
    summary = score_v2(records, cap)
    dat = directory / f"{stem}.dat"
    lines = [
        "# counter harness. One force call is one full-batch loss and gradient.",
        "# The potential is the mlopt squared error on the archived Behler-Parrinello fingerprints.",
        "# PANNA Adam supplies the step. A call that reaches the cap is a cap failure.",
        "# Force coefficient 0 records force RMSE -1.",
    ]
    fields: list[tuple[str, object]] = [
        ("template", template["id"]),
        ("problem", template["problem"]),
        ("code", template["code"]),
        ("code_version", template["code_version"]),
        ("commit", template["commit"]),
        ("algorithm", template.get("algorithm") or "Adam"),
        ("tuning", template.get("tuning") or "default"),
        ("counter", "harness"),
        ("cap", cap),
        ("n", summary["n"]),
        ("nfailed", summary["nfailed"]),
        ("par2_mean", summary["par2_mean"]),
        ("median", summary["median"]),
        ("median_low", summary["median_low"]),
        ("median_high", summary["median_high"]),
        ("wilson_low", summary["wilson_low"]),
        ("wilson_high", summary["wilson_high"]),
        ("contributor", "Rohit Goswami"),
    ]
    if partial:
        fields.append(("partial", True))
    for key, label in (
        ("force_calls", "mean_epoch"),
        ("energy_rmse", "mean_energy_rmse"),
        ("force_rmse", "mean_force_rmse"),
    ):
        stats = _success_stats(records, key)
        if stats is None:
            continue
        mean, median = stats
        fields.append((label, mean))
        fields.append((label.replace("mean_", "median_"), median))
    for key, value in fields:
        lines.append(f"{key} {_format(value)}")
    dat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    starts = directory / f"{stem}.starts.jsonl"
    body = [json.dumps(row, separators=(",", ":")) for row in records]
    starts.write_text("\n".join(body) + "\n", encoding="utf-8")
    return dat


def campaign_stem(template: dict) -> str:
    slug = str(template["id"]).split("/", 1)[1]
    return f"panna-{template['code_version']}-{slug}"


def run_template(
    root: Path,
    template_path: Path,
    *,
    output: Path | None = None,
    limit: int | None = None,
    max_calls: int | None = None,
    progress: bool = True,
) -> Path:
    from optbench.campaign import load_template, resolve_problem

    root = Path(root)
    template = load_template(root, str(template_path))
    if template.get("engine") != "panna":
        raise ValueError(f"{template_path} is not a PANNA template")
    if template.get("code") != "PANNA" or template.get("commit") != PANNA_COMMIT:
        raise ValueError("the template pin is not PANNA 2.1.0 at the registry commit")
    if template.get("code_version") != PANNA_VERSION:
        raise ValueError("the template version is not 2.1.0")
    problem = resolve_problem(root, template["problem"])
    cap, tolerance = _require_problem_budget(template, problem)
    full = int(problem["sample"]["initializations"])
    count = full if limit is None else int(limit)
    if count < 1 or count > full:
        raise ValueError(f"limit must be from 1 to {full}")
    partial = count < full or max_calls is not None
    if max_calls is not None:
        cap = int(max_calls)
    trainer = str(template["panna"]["trainer"])
    assert_panna_pin()
    if trainer == "tensorflow":
        # The stepper is built after the parameters exist. The pin check imports PANNA.
        pass
    elif trainer == "jax":
        pass
    else:
        raise ValueError(f"unknown PANNA trainer {trainer}")
    if progress:
        print(f"building fingerprints for {template['id']}", flush=True)
    landscape = load_landscape(root, template, problem)
    starts = list(range(count))
    seed = int(template["panna"]["initialization_seed"])
    sample = np.stack([initial_parameters(landscape.layout, start, seed) for start in starts])
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from optbench.ml_jax import loss_and_grad as evaluate_loss

    if trainer == "tensorflow":
        stepper = _tensorflow_adam(sample, float(template["optimizer"]["learning_rate"]))
    else:
        stepper = _jax_adam(sample, float(template["optimizer"]["learning_rate"]))
    checkpoint = Path(os.environ.get("OPTBENCH_PANNA_CKPT", "/tmp/optbench-panna-ckpt"))
    checkpoint.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint / f"{campaign_stem(template)}.npz"
    try:
        records = train_starts(
            landscape,
            template,
            cap=cap,
            tolerance=tolerance,
            energy_coefficient=float(template["panna"]["energy_coefficient"]),
            force_coefficient=float(template["panna"]["force_coefficient"]),
            stepper=stepper,
            starts=starts,
            seed=seed,
            progress=progress,
            evaluate=evaluate_loss,
            checkpoint=checkpoint,
        )
    except TrainingPaused as paused:
        print(f"paused {template['id']} at {paused}", flush=True)
        return Path(str(paused))
    directory = output or (root / "results" / "campaign")
    written = write_campaign(directory, campaign_stem(template), template, records, cap, partial=partial)
    _clear_checkpoint(checkpoint)
    return written


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Panna" + "".join(part.capitalize() for part in slug.split("-"))


def write_asv(root: Path, commit: str) -> Path:
    """Write asv tracks from the finished PANNA campaign files."""
    from optbench.campaign import list_templates

    root = Path(root)
    templates = [item for item in list_templates(root) if item.get("engine") == "panna"]
    if len(templates) != 8:
        raise ValueError(f"expected 8 PANNA templates, found {len(templates)}")
    campaign = root / "results" / "campaign"
    grouped: dict[str, list[dict]] = {}
    for template in templates:
        stem = campaign_stem(template)
        path = campaign / f"{stem}.starts.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 50:
            raise ValueError(f"{path} has {len(rows)} starts")
        rows.sort(key=lambda row: int(row["start"]))
        grouped[template["id"]] = rows
    machine = json.loads((root / "results" / "asv" / "cursor" / "machine.json").read_text(encoding="utf-8"))
    params = {
        "arch": machine["arch"],
        "cpu": machine["cpu"],
        "machine": machine["machine"],
        "num_cpu": machine["num_cpu"],
        "os": machine["os"],
        "ram": machine["ram"],
        "python": "3.12",
    }
    now = int(time.time() * 1000)
    seeds = [repr("20261015")]
    starts = [repr(str(index)) for index in range(50)]
    columns = ["result", "params", "version", "started_at", "duration"]
    results = {}
    tracks = (
        "time_search",
        "track_force_calls",
        "track_success",
        "track_failure_cap",
        "track_failure_wrong_stationary",
        "track_failure_divergence",
        "track_failure_crash",
    )
    for template_id, rows in grouped.items():
        class_name = _class_name(template_id)
        calls = [float(row["force_calls"]) for row in rows]
        success = [1.0 if row["success"] else 0.0 for row in rows]
        failure = {name: [] for name in ("cap", "wrong_stationary", "divergence", "crash")}
        seconds = []
        for row in rows:
            label = row.get("failure")
            for name in failure:
                failure[name].append(1.0 if label == name else 0.0)
            seconds.append(float(row["seconds"]))
        payload = {
            "time_search": seconds,
            "track_force_calls": calls,
            "track_success": success,
            "track_failure_cap": failure["cap"],
            "track_failure_wrong_stationary": failure["wrong_stationary"],
            "track_failure_divergence": failure["divergence"],
            "track_failure_crash": failure["crash"],
        }
        duration = sum(seconds)
        for method in tracks:
            key = f"bench_panna.{class_name}.{method}"
            version = hashlib.sha256(key.encode("utf-8")).hexdigest()
            results[key] = [payload[method], [seeds, starts], version, now, duration]
    document = {
        "commit_hash": commit,
        "env_name": "uv-pip-py3.12",
        "date": now,
        "params": params,
        "python": "3.12",
        "requirements": {},
        "env_vars": {},
        "result_columns": columns,
        "results": results,
        "durations": {},
        "version": 2,
    }
    dest = root / "results" / "asv" / "cursor" / f"{commit[:8]}-uv-pip-py3.12.json"
    dest.write_text(json.dumps(document, separators=(", ", ": ")), encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PANNA Adam on one mlopt template.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("template", type=Path)
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--max-calls", type=int, default=None)
    asv = sub.add_parser("asv")
    asv.add_argument("--commit", required=True)
    asv.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.command == "asv":
        path = write_asv(args.root, args.commit)
        print(path)
        return
    path = run_template(
        args.root,
        args.template,
        output=args.output,
        limit=args.limit,
        max_calls=args.max_calls,
    )
    print(path)


if __name__ == "__main__":
    main()
