"""GL-BFGS Hessian climbing-image NEB for the Pt-heptamer archive row.

Anonymous SVN does not export Gman r235. The step transcribed here is
VTST ``lbfgs.F`` file revision 128, exported at peg r147 (2011-10-10),
Hessian branch: ``LLINEOPT`` false and ``LGLOBAL`` true. The archived
``opt.py`` sets memory 25, inverse curvature 0.05, and maxmove 0.2.
The global step is capped at ``maxmove * sqrt(images)``.

The band is the Henkelman improved tangent, a climbing image, and a
spring of 5. Convergence is the maximum Euclidean NEB force, Fmag.
A Morse evaluation is one force call. The published integer is the
``%d`` truncation of the mean of ``pot.count() / images``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .campaign import geometry_archive, resolve_problem
from .pes import Pair, load_pairs, morse_pt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "campaign" / "gman-vtst-r128-neb-pt-glbfgs-hess.dat"

MEMORY = 25
INVCURV = 0.05
MAXMOVE = 0.2
SPRING = 5.0
FMAX = 0.01
IMAGES = 8
FAILURE_CAP = 10000


def _mic(delta: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return delta
    return delta - box * np.round(delta / box)


class _Counter:
    def __init__(self) -> None:
        self.n = 0

    def eval(self, coords: np.ndarray, box: np.ndarray | None) -> tuple[float, np.ndarray]:
        self.n += 1
        energy, grad = morse_pt(coords, box)
        return energy, -grad


def neb_forces(
    positions: list[np.ndarray],
    energies: list[float],
    true_force: list[np.ndarray | None],
    spring: float,
    box: np.ndarray | None,
) -> list[np.ndarray]:
    """Improved-tangent CI-NEB forces. Endpoints are returned as zeros.

    ``positions[i]`` is ``(n_free, 3)``. ``dR`` is the neighbour minus
    this image, minimum image, matching VTST ``neb.F``.
    """
    n = len(positions)
    out = [np.zeros_like(positions[i]) for i in range(n)]
    tangents = [np.zeros_like(positions[i]) for i in range(n)]
    dist_prev = np.zeros(n)
    dist_next = np.zeros(n)
    for i in range(1, n - 1):
        d_prev = _mic(positions[i - 1] - positions[i], box)
        d_next = _mic(positions[i + 1] - positions[i], box)
        dist_prev[i] = float(np.linalg.norm(d_prev))
        dist_next[i] = float(np.linalg.norm(d_next))
        prev_higher = energies[i - 1] > energies[i]
        next_higher = energies[i + 1] > energies[i]
        if prev_higher != next_higher:
            tangent = -d_prev if prev_higher else d_next
        else:
            diff_prev = energies[i - 1] - energies[i]
            diff_next = energies[i + 1] - energies[i]
            lo = min(abs(diff_prev), abs(diff_next))
            hi = max(abs(diff_prev), abs(diff_next))
            if diff_prev > diff_next:
                tangent = d_next * lo - d_prev * hi
            else:
                tangent = d_next * hi - d_prev * lo
        norm = float(np.linalg.norm(tangent))
        tangents[i] = tangent / norm if norm > 0.0 else tangent
    highest = int(np.argmax(energies))
    climb = highest if 0 < highest < n - 1 else -1
    for i in range(1, n - 1):
        force = true_force[i]
        if force is None:
            raise ValueError(f"image {i} has no force")
        tangent = tangents[i]
        parallel = tangent * float(np.sum(force * tangent))
        if i == climb:
            out[i] = force - 2.0 * parallel
        else:
            spring_par = spring * (dist_next[i] - dist_prev[i]) * tangent
            out[i] = spring_par + (force - parallel)
    return out


def lbfgs_step(
    positions: np.ndarray,
    force: np.ndarray,
    state: dict,
    *,
    memory: int,
    invcurv: float,
    maxmove: float,
    n_images: int,
) -> np.ndarray:
    """One Hessian L-BFGS step on the flat global band vector.

    The first step, and any step with an empty force history, follows
    the force. Later steps are the two-loop recursion with ``H0 =
    invcurv * I``. The displacement is capped at ``maxmove *
    sqrt(n_images)``, which is ``SQRT(maxmove**2 * images)`` in
    ``lbfgs.F``.
    """
    fold = state["fold"]
    reset = float(np.dot(fold, fold)) == 0.0
    if not reset:
        y = fold - force
        step = positions - state["rold"]
        ys = float(np.dot(y, step))
        state["s"].append(step)
        state["y"].append(y)
        state["rho"].append(1.0 / ys)
        if len(state["s"]) > memory:
            state["s"].pop(0)
            state["y"].pop(0)
            state["rho"].pop(0)
    else:
        state["s"] = []
        state["y"] = []
        state["rho"] = []
    state["rold"] = positions.copy()
    state["fold"] = force.copy()
    direction = -force.copy()
    bound = len(state["s"])
    if bound:
        alpha = np.zeros(bound)
        for index in range(bound - 1, -1, -1):
            alpha[index] = state["rho"][index] * float(np.dot(state["s"][index], direction))
            direction -= alpha[index] * state["y"][index]
        direction *= invcurv
        for index in range(bound):
            beta = state["rho"][index] * float(np.dot(state["y"][index], direction))
            direction += state["s"][index] * (alpha[index] - beta)
    direction *= -1.0
    cap = maxmove * np.sqrt(float(n_images))
    length = float(np.linalg.norm(direction))
    if length > cap and length > 0.0:
        direction *= cap / length
    return positions + direction


def run_pair(
    pair: Pair,
    *,
    images: int = IMAGES,
    spring: float = SPRING,
    fmax: float = FMAX,
    memory: int = MEMORY,
    invcurv: float = INVCURV,
    maxmove: float = MAXMOVE,
    cap: int = FAILURE_CAP,
) -> dict:
    """Relax one band. ``calls`` counts every Morse evaluation."""
    if images < 1:
        raise ValueError("images must be positive")
    box = None if pair.box is None else np.asarray(pair.box, dtype=np.float64)
    free = pair.free
    reactant = np.array(pair.reactant, dtype=np.float64, copy=True)
    product = np.array(pair.product, dtype=np.float64, copy=True)
    delta = _mic(product - reactant, box)
    counter = _Counter()

    def frame(free_xyz: np.ndarray) -> np.ndarray:
        full = reactant.copy()
        full[free] = free_xyz
        return full

    def evaluate(free_xyz: np.ndarray) -> tuple[float, np.ndarray]:
        energy, force = counter.eval(frame(free_xyz), box)
        return energy, force[free]

    energy_reactant, _force_reactant = evaluate(reactant[free])
    energy_product, _force_product = evaluate((reactant + delta)[free])
    total = images + 2
    band = []
    for index in range(total):
        moved = reactant + (index / (total - 1)) * delta
        band.append(moved[free].copy())
    width = int(free.sum()) * 3
    state = {
        "fold": np.zeros(images * width),
        "rold": None,
        "s": [],
        "y": [],
        "rho": [],
    }
    converged = False
    force_norm = None
    iterations = 0
    while counter.n < cap:
        energies = [energy_reactant]
        forces: list[np.ndarray | None] = [None]
        for image in band[1:-1]:
            energy, force = evaluate(image)
            energies.append(energy)
            forces.append(force)
            if counter.n >= cap:
                break
        energies.append(energy_product)
        forces.append(None)
        if any(force is None for force in forces[1:-1]):
            break
        neb = neb_forces(band, energies, forces, spring, box)
        stacked = np.stack([neb[i] for i in range(1, images + 1)])
        norms = np.linalg.norm(stacked.reshape(images, -1), axis=1)
        force_norm = float(norms.max())
        iterations += 1
        if force_norm < fmax:
            converged = True
            break
        if counter.n >= cap:
            break
        updated = lbfgs_step(
            np.stack(band[1:-1]).reshape(-1),
            stacked.reshape(-1),
            state,
            memory=memory,
            invcurv=invcurv,
            maxmove=maxmove,
            n_images=images,
        ).reshape(images, -1, 3)
        for index in range(images):
            band[index + 1] = updated[index]
    failed = (not converged) or counter.n >= cap
    return {
        "start": pair.name,
        "calls": counter.n,
        "per_image": counter.n / float(images),
        "iterations": iterations,
        "converged": converged and not failed,
        "force_norm": force_norm,
        "failed": failed,
    }


def paper_report(totals: list[int], images: int) -> dict[str, int | float]:
    """Archived ``opt.py`` integers: ``%d`` of ``pot.count() / images``."""
    if not totals:
        raise ValueError("no paths")
    if images <= 0:
        raise ValueError("images must be positive")
    per = [total / float(images) for total in totals]
    mean = sum(per) / len(per)
    return {
        "force_calls": int(mean),
        "force_calls_min": int(min(per)),
        "force_calls_max": int(max(per)),
        "unrounded": mean,
    }


def _settings(problem: dict, template: dict) -> dict:
    band = problem.get("band") or {}
    images = int(band.get("interior_images") or IMAGES)
    glbfgs = template.get("glbfgs") or {}
    stated = glbfgs.get("images")
    if stated is not None and int(stated) != images:
        raise ValueError("template image count is not the problem image count")
    stop = template.get("stop") or {}
    convergence = problem.get("convergence") or {}
    fmax = float(stop.get("force_norm", convergence.get("force_norm", FMAX)))
    cap = int(convergence.get("max_force_calls") or template.get("failure_cap") or FAILURE_CAP)
    return {
        "images": images,
        "spring": float(glbfgs.get("spring", SPRING)),
        "fmax": fmax,
        "memory": int(glbfgs.get("memory", MEMORY)),
        "invcurv": float(glbfgs.get("alpha", INVCURV)),
        "maxmove": float(glbfgs.get("maxmove", MAXMOVE)),
        "cap": cap,
    }


def _dat(rows: list[dict], template: dict, images: int) -> str:
    totals = [int(row["calls"]) for row in rows]
    report = paper_report(totals, images)
    failed = sum(1 for row in rows if int(row["calls"]) >= FAILURE_CAP)
    note = (
        "Archived opt.py glbfgs(hess): memory 25, alpha 0.05, maxmove 0.2, "
        "8 images, spring 5, climbing image, improved tangent, Fmag 0.01. "
        "Anonymous SVN does not export Gman r235. The step is VTST lbfgs.F "
        "file revision 128 at peg r147, Hessian branch, global cap "
        "maxmove*sqrt(images). force_calls is the %d truncation of the mean "
        "of Morse evaluations divided by 8. A start fails at 10000 evaluations."
    )
    lines = [
        "# force_calls is int(mean(evaluations / images)), the archived opt.py %d rule",
        "# starts store undivided Morse evaluations, including two endpoint evaluations",
        f"force_calls {report['force_calls']}",
        f"force_calls_min {report['force_calls_min']}",
        f"force_calls_max {report['force_calls_max']}",
        f"force_calls_unrounded {report['unrounded']:.16g}",
        f"nfailed {failed}",
        f"starts {len(rows)}",
        f"code {template['code']}",
        f"code_version {template['code_version']}",
        f"commit {template['commit']}",
        f"algorithm {template.get('algorithm', 'lbfgs')}",
        f"template {template['id']}",
        f"problem {template['problem']}",
        "counter harness",
        "reducer opt-py",
        f"images {images}",
        f"note {note}",
    ]
    return "\n".join(lines) + "\n"


def _starts(rows: list[dict], template: dict) -> str:
    lines = []
    for row in rows:
        lines.append(
            json.dumps(
                {
                    "calls_per_image": row["per_image"],
                    "comparable": True,
                    "converged": row["converged"],
                    "counter": "harness",
                    "failed": row["failed"],
                    "force_calls": row["calls"],
                    "force_norm": row["force_norm"],
                    "iterations": row["iterations"],
                    "problem": template["problem"],
                    "schema": "optbench.start/v2",
                    "seed": "0",
                    "start": row["start"],
                    "success": row["converged"] and not row["failed"],
                    "template": template["id"],
                },
                sort_keys=True,
            )
        )
    return "\n".join(lines) + "\n"


def run_template(
    root: Path,
    problem: dict,
    template: dict,
    *,
    output: Path,
    workers: int = 1,
    limit: int = 0,
) -> dict:
    """Run the 13 Pt-heptamer pairs and write the campaign files."""
    del workers
    if problem["id"] != "neb/Pt-heptamer":
        raise ValueError(f"gman glbfgs hess is the Pt-heptamer band, not {problem['id']}")
    settings = _settings(problem, template)
    archive = geometry_archive(root, problem)
    pairs = load_pairs(archive, problem["geometry"]["kind"], problem["sample"]["pairs"])
    if limit:
        pairs = pairs[:limit]
    rows = [run_pair(pair, **settings) for pair in pairs]
    if not limit and any(not row["converged"] for row in rows):
        missed = [row["start"] for row in rows if not row["converged"]]
        raise RuntimeError(f"bands did not converge: {', '.join(missed)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dat(rows, template, settings["images"]), encoding="utf-8")
    output.with_suffix(".starts.jsonl").write_text(_starts(rows, template), encoding="utf-8")
    report = paper_report([int(row["calls"]) for row in rows], settings["images"])
    return {
        "force_calls": report["force_calls"],
        "force_calls_min": report["force_calls_min"],
        "force_calls_max": report["force_calls_max"],
        "unrounded": report["unrounded"],
        "nfailed": sum(1 for row in rows if int(row["calls"]) >= settings["cap"]),
        "starts": len(rows),
        "template": template["id"],
        "problem": problem["id"],
    }


def main() -> None:
    from .campaign import load_template

    problem = resolve_problem(ROOT, "neb/Pt-heptamer")
    template = load_template(ROOT, "gman/neb-pt-glbfgs-hess")
    summary = run_template(ROOT, problem, template, output=DEFAULT_OUTPUT)
    print(
        f"force_calls {summary['force_calls']} "
        f"min {summary['force_calls_min']} max {summary['force_calls_max']} "
        f"unrounded {summary['unrounded']:.16g} starts {summary['starts']}"
    )


if __name__ == "__main__":
    main()
