"""Harness callback for one GeometryOptimization.jl minimization.

The Julia solver asks this session for the spec potential. One force
call is one ``RgpotCounter.calculate`` entry on a ``HarnessCounter``.
A repeated coordinate is the cached entry. Noise, when the job asks
for it, is added after that call.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from optbench.counter import ForceCallCap, HarnessCounter
from optbench.failures import classify_failure
from optbench.noise import KAPPA, WINDOW, converged_under_noise, stop_threshold
from optbench.potentials import lennard_jones, morse_bulk
from optbench.potserver import RgpotCounter
from optbench.runners.noise_min import _Observed, _index
from optbench.runners.rgmin_min import _index_at


class Session:
    """One start. Julia owns the loop and calls into this object."""

    def __init__(self, payload: dict):
        self.kind = str(payload["kind"])
        self.method = str(payload["method"])
        self.start = str(payload.get("start") or "")
        self.index = int(payload.get("index") or 0)
        self._coords = [float(value) for value in payload["coords"]]
        box = payload.get("box")
        self.box = None if box is None else [float(value) for value in box]
        self.cap = int(payload["cap"])
        self.clean_tol = float(payload["gtol"])
        self.sigma = float(payload.get("sigma") or 0.0)
        self.noise = bool(payload.get("noise"))
        self.seed = int(payload.get("seed") or 0)
        self.energy_tol = float(payload.get("energy_tol") or 0.0)
        self.components = len(self._coords)
        if self.kind == "lj":

            def evaluate(positions, _numbers=None, _box=None):
                return lennard_jones(positions)

        elif self.kind == "morse":

            def evaluate(positions, _numbers=None, _box=None):
                return morse_bulk(positions, self.box)

        else:
            raise ValueError(f"unknown potential kind {self.kind}")
        self.counter = HarnessCounter(cache=True, cap=self.cap)
        self.potential = RgpotCounter(evaluate, self.counter)
        self.observed = None
        if self.noise:
            self.observed = _Observed(
                self.potential,
                self.sigma,
                random.Random(self.seed + self.index),
            )
        self.reason = ""
        self.error = ""

    @staticmethod
    def from_json(text: str) -> Session:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("job must be an object")
        return Session(payload)

    def job_kind(self) -> str:
        return self.kind

    def job_method(self) -> str:
        return self.method

    def job_coords(self) -> list[float]:
        return list(self._coords)

    def job_box(self) -> list[float]:
        if self.box is None:
            return []
        return list(self.box)

    def job_cap(self) -> int:
        return self.cap

    def calls(self) -> int:
        return int(self.counter.calls)

    def call(self, coords) -> dict[str, Any]:
        flat = [float(value) for value in coords]
        try:
            if self.observed is None:
                energy, gradient = self.potential.calculate(flat)
            else:
                energy, gradient = self.observed(flat)
        except ForceCallCap as exc:
            self.reason = "cap"
            return {"cap": True, "energy": 0.0, "gradient": [], "error": f"ForceCallCap: {exc}"}
        except Exception as exc:
            self.reason = "crash"
            self.error = f"{type(exc).__name__}: {exc}"
            return {"cap": False, "energy": 0.0, "gradient": [], "error": self.error}
        return {
            "cap": False,
            "energy": float(energy),
            "gradient": [float(value) for value in gradient],
            "error": "",
        }

    def met(self, force_l2: float) -> bool:
        """Whether the accepted point may stop the search."""
        if self.observed is None:
            return float(force_l2) <= self.clean_tol and math.isfinite(float(force_l2))
        return converged_under_noise(
            force_norm=float(force_l2),
            energies=self.observed.energies,
            clean_tol=self.clean_tol,
            sigma=self.sigma,
            components=self.components,
            energy_tol=self.energy_tol,
            kappa=KAPPA,
            window=WINDOW,
        )

    def _threshold(self) -> float:
        if not self.noise:
            return self.clean_tol
        return stop_threshold(self.clean_tol, self.sigma, self.components, KAPPA)

    def finish(
        self,
        positions,
        reason: str,
        error: str,
        seconds: float,
        library_converged: bool,
    ) -> str:
        point = [float(value) for value in positions] if positions is not None else None
        calls = int(self.counter.calls)
        verify_calls = int(self.counter.verify_calls)
        energy = None
        force_norm = None
        hessian_index = None
        min_eigenvalues: list[float] = []
        stored_error = str(error or self.error or "")
        if reason:
            self.reason = str(reason)
        threshold = self._threshold()
        if point is not None and len(point) == self.components:
            try:
                with self.counter.verification():
                    energy, gradient = self.potential.calculate(point)
                force_norm = float(math.sqrt(sum(float(value) * float(value) for value in gradient)))
                energy = float(energy)
                verify_calls = int(self.counter.verify_calls)
            except Exception as exc:
                if not stored_error:
                    stored_error = f"{type(exc).__name__}: {exc}"
            if stored_error == "" or "force-call cap" in stored_error or "ForceCallCap" in stored_error:
                try:
                    if self.noise:
                        hessian_index, min_eigenvalues = _index(point, threshold)
                    else:
                        hessian_index, min_eigenvalues = _index_at(self.kind, point, self.box, self.clean_tol)
                except Exception as exc:
                    if not stored_error:
                        stored_error = f"{type(exc).__name__}: {exc}"
        finite = (
            point is not None
            and len(point) == self.components
            and all(math.isfinite(value) for value in point)
            and energy is not None
            and math.isfinite(energy)
            and force_norm is not None
            and math.isfinite(force_norm)
        )
        cap_hit = self.reason == "cap" or "force-call cap" in stored_error or "ForceCallCap" in stored_error
        stopped = self.reason == "tolerance"
        if self.noise:
            at_force = finite and stopped and force_norm is not None and force_norm <= threshold and calls < self.cap and not stored_error
        else:
            at_force = finite and force_norm is not None and force_norm <= self.clean_tol and calls < self.cap and not stored_error
        reasons: list[str] = []
        if stored_error and not cap_hit:
            reasons.append("crash")
        elif not finite:
            reasons.append("divergence")
        elif calls >= self.cap or cap_hit:
            reasons.append("cap")
        elif not at_force:
            reasons.append("force_norm")
        elif hessian_index != 0:
            reasons.append("hessian_index")
        failure = classify_failure(reasons)
        record: dict[str, Any] = {
            "schema": "optbench.start/v2",
            "start": self.start,
            "index": self.index,
            "force_calls": calls,
            "verify_calls": verify_calls,
            "counter": "harness",
            "comparable": True,
            "success": failure is None,
            "failure": failure,
            "reasons": reasons,
            "energy": energy,
            "force_norm": force_norm,
            "hessian_index": hessian_index,
            "expected_index": 0,
            "converged": at_force,
            "hessian_calls": 0,
            "min_eigenvalues": min_eigenvalues,
            "seconds": float(seconds),
            "geomopt_reason": self.reason,
            "library_converged": bool(library_converged),
        }
        if self.noise:
            record["sigma"] = self.sigma
            record["seed"] = self.seed + self.index
        if stored_error:
            record["error"] = stored_error[:500]
        return json.dumps(record)
