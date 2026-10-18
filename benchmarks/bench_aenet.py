"""aenet mlopt training, one class per template.

The default grid is seed ``20261015`` and start ``0``. ``asv check``
stays on that one start. ``OPTBENCH_ASV_FULL=1`` selects all 50
initializations. The flag is read at import.

``setup`` raises ``NotImplementedError`` when the aenet L-BFGS-B archive
or the trajectory is absent, and asv records that parameter as NaN.
Importing this module does not import ASE and does not compile Fortran.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from optbench.asv_tracks import TrackedSearch

ROOT = Path(__file__).resolve().parents[1]
SEED = "20261015"


def _cache_dir() -> Path:
    return Path(os.environ.get("OPTBENCH_ASV_CACHE", "/tmp/optbench-asv-aenet"))


def _ready() -> bool:
    if shutil.which("gfortran") is None:
        return False
    try:
        import ase  # noqa: F401
    except ImportError:
        return False
    from optbench.aenet_ml import lbfgsb_library_path

    if lbfgsb_library_path() is None:
        return False
    traj = ROOT / "inputs/sha256/2ff6a9ab51619942c08dd9239b9a2be57552481808ced9de9e703a03ba2da402/pt_md.traj"
    other = ROOT / "inputs/sha256/0d82bbb8892bcc8871bb805c1a84de5e429aaa90c254a510fad0dad96ff36d4d/C2H4_md.traj"
    return traj.is_file() and other.is_file()


def selected_indices(count: int) -> list[str]:
    if os.environ.get("OPTBENCH_ASV_FULL") == "1":
        return [str(index) for index in range(count)]
    return ["0"]


def _class_name(template_id: str) -> str:
    slug = template_id.split("/", 1)[1]
    return "Aenet" + "".join(part.capitalize() for part in slug.split("-"))


def _specs() -> list[dict]:
    from optbench.campaign import list_templates

    found = []
    for template in list_templates(ROOT):
        if template.get("engine") != "aenet":
            continue
        block = template.get("aenet") or {}
        stop = template.get("stop") or {}
        found.append(
            {
                "class_name": _class_name(str(template["id"])),
                "template_id": str(template["id"]),
                "problem_id": str(template["problem"]),
                "method": str(block["method"]),
                "cap": int(template["failure_cap"]),
                "tolerance": float(stop["gradient_norm"]),
                "n": 50,
            }
        )
    return found


def _cache_path(template_id: str, seed: str, start: str) -> Path:
    slug = template_id.replace("/", "-")
    return _cache_dir() / slug / f"{seed}-{int(start)}.json"


def _compute(spec: dict, seed: str, start: str) -> dict:
    if str(seed) != SEED:
        raise ValueError(f"seed {seed} is not {SEED}")
    from optbench.campaign import load_template, resolve_problem
    from optbench.runners.aenet_ml import _train_index, prepare_model

    import optbench.runners.aenet_ml as runner

    template = load_template(ROOT, spec["template_id"])
    problem = resolve_problem(ROOT, spec["problem_id"])
    runner._MODEL = prepare_model(ROOT, problem, template)
    runner._SPEC = runner._checked(problem, template)
    if runner._SPEC["method"] == "bfgs":
        from optbench.aenet_ml import ensure_lbfgsb_built

        ensure_lbfgsb_built()
    record = _train_index(int(start))
    record["seed"] = SEED
    path = _cache_path(spec["template_id"], SEED, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)
    return record


def _load(spec: dict, seed: str, start: str) -> dict:
    path = _cache_path(spec["template_id"], SEED, start)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return _compute(spec, seed, start)


class _AenetSearch(TrackedSearch):
    def setup(self, seed, start):
        if not _ready():
            raise NotImplementedError("aenet L-BFGS-B or the mlopt trajectory is not available")
        self.record = None

    def time_search(self, seed, start):
        self.record = _compute(self.spec, seed, start)

    def _stored(self, seed, start) -> dict:
        if self.record is None:
            self.record = _load(self.spec, seed, start)
        return self.record

    def track_force_calls(self, seed, start) -> float:
        return float(self._stored(seed, start)["force_calls"])

    def track_success(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["success"] else 0.0

    def track_failure_cap(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "cap" else 0.0

    def track_failure_wrong_stationary(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "wrong_stationary" else 0.0

    def track_failure_divergence(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "divergence" else 0.0

    def track_failure_crash(self, seed, start) -> float:
        return 1.0 if self._stored(seed, start)["failure"] == "crash" else 0.0


_AenetSearch.track_force_calls.unit = "force calls"
_AenetSearch.track_success.unit = "fraction"
_AenetSearch.track_failure_cap.unit = "count"
_AenetSearch.track_failure_wrong_stationary.unit = "count"
_AenetSearch.track_failure_divergence.unit = "count"
_AenetSearch.track_failure_crash.unit = "count"


def _benchmark(spec: dict):
    bound = spec
    starts = selected_indices(spec["n"])

    class _Benchmark(_AenetSearch):
        param_names = ["seed", "start"]
        params = [[SEED], starts]
        timeout = 14400
        cap = bound["cap"]
        template_id = bound["template_id"]
        problem_id = bound["problem_id"]
        spec = bound

    _Benchmark.__name__ = spec["class_name"]
    _Benchmark.__qualname__ = spec["class_name"]
    return _Benchmark


for _spec in _specs():
    globals()[_spec["class_name"]] = _benchmark(_spec)
