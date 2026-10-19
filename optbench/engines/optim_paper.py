"""2014 OPTIM paper rows through the ASE socket.

The odata body is the archived preamble. The score is the last
``energy+gradient calls`` field, the same awk the 2014 ``run_optim.sh``
scripts used. A success is that script's grep, not a harness Hessian.

Pt and Morse bulk coordinates are divided by ``r0``. The energy OPTIM
sees is the harness energy over ``D``. The gradient is that reduced
gradient divided by ``rho``, which is the 2014 ``morse_bulk`` formula.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

from .common import write_campaign
from .geometries import _stored
from .optim_search import _RELAY, binary_path

R0 = 2.897
D_WELL = 0.7102
# 2014 PARAMS rho, alpha * r0, as written in the preambles.
RHO = 4.6488159
# bc -l prints of rho*r0 and rcut/r0. The box lengths use Python %f.
RHO_TEXT = "4.64881590"
RCUT_TEXT = "3.27925440110459095616"

_PAPER = {
    "optim/de-lj38": {
        "kind": "lj",
        "symbol": "AX",
        "units": "lj",
        "success": "Converged to TS",
        "timeout": 3600,
        "geometry": "de-lj38",
    },
    "optim/de-2ts": {
        "kind": "lj",
        "symbol": "AX",
        "units": "lj",
        "success": "Connected path found",
        "timeout": 7200,
        "geometry": "de-2ts",
    },
    "optim/de-pt": {
        "kind": "morse_pt",
        "symbol": "M",
        "units": "morse-2014",
        "success": "Converged to TS",
        "timeout": 7200,
        "geometry": "de-pt",
        "append_params": True,
    },
    "optim/morse-lbfgs-tuned": {
        "kind": "morse_pt",
        "symbol": "M",
        "units": "morse-2014",
        "success": "**** CONVERGED ****",
        "timeout": 1800,
        "geometry": "morse-min",
    },
    "optim/hef-ffd-ts-lj38": {
        "kind": "lj",
        "symbol": "AX",
        "units": "lj",
        "success": "**** CONVERGED ****",
        "timeout": 1800,
        "geometry": "ts-lj38",
    },
    "optim/hef-ffd-ts-pt": {
        "kind": "morse_pt",
        "symbol": "M",
        "units": "morse-2014",
        "success": "**** CONVERGED ****",
        "timeout": 3600,
        "geometry": "ts-pt",
        "append_params": True,
        "mode": "displacement",
    },
    "optim/rr-loweig": {
        "kind": "lj",
        "symbol": "AX",
        "units": "lj",
        "success": "**** CONVERGED ****",
        "timeout": 1800,
        "geometry": "rr-lj38",
        "mode": "archive",
    },
}


def _members(archive: Path) -> dict[str, str]:
    found = {}
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if not member.isfile():
                continue
            extracted = packed.extractfile(member)
            if extracted is None:
                continue
            found[member.name] = extracted.read().decode("utf-8", errors="replace")
    return found


def _xyz_rows(text: str) -> list[list[str]]:
    lines = text.splitlines()
    count = int(lines[0].split()[0])
    rows = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError("xyz coordinate is missing x y z")
        rows.append(parts[1:4])
    if len(rows) != count:
        raise ValueError(f"xyz announced {count} atoms and held {len(rows)}")
    return rows


def _scale_rows(coords: np.ndarray, scale: float) -> list[list[str]]:
    rows = []
    for row in np.asarray(coords, dtype=np.float64):
        rows.append([f"{float(value) * scale:.16f}" for value in row])
    return rows


def _params_line(box: np.ndarray) -> str:
    reduced = np.asarray(box, dtype=np.float64).reshape(-1)[:3] / R0
    lengths = " ".join(f"{float(value):.6f}" for value in reduced)
    return f"PARAMS {RHO_TEXT} {lengths} {RCUT_TEXT}"


def _freeze_lines(free: np.ndarray) -> list[str]:
    indices = [str(index + 1) for index, flag in enumerate(np.asarray(free, dtype=bool)) if not flag]
    lines = []
    for start in range(0, len(indices), 8):
        lines.append("FREEZE " + " ".join(indices[start : start + 8]))
    return lines


def _symbol_lines(rows: list[list[str]], symbol: str) -> list[str]:
    return [f"{symbol}  {' '.join(row)}" for row in rows]


def _bare(rows: list[list[str]]) -> str:
    return "\n".join(" ".join(row) for row in rows) + "\n"


def _write_xyz(path: Path, rows: list[list[str]], box) -> None:
    lines = [str(len(rows))]
    if box is None:
        lines.append("optbench")
    else:
        lengths = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        lattice = f"{lengths[0]:.16g} 0 0 0 {lengths[1]:.16g} 0 0 0 {lengths[2]:.16g}"
        lines.append(f'Lattice="{lattice}" Properties=species:S:1:pos:R:3')
    for row in rows:
        lines.append(f"Ar {' '.join(row)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ase_line(python: str, hook: Path, periodic: bool) -> str:
    pbc = "PBC" if periodic else "NOPBC"
    return f"ASE geom.xyz {python} {hook} dummy dummy cpu FIXCELL {pbc} EV ANG"


def energy_gradient_calls(text: str) -> int | None:
    """Last ``energy+gradient calls`` integer, the 2014 awk field 6."""
    matched = [line for line in text.splitlines() if "energy+gradient calls" in line]
    if not matched:
        return None
    parts = matched[-1].split()
    if len(parts) < 6:
        return None
    try:
        return int(parts[5])
    except ValueError:
        return None


def _limit() -> int | None:
    raw = os.environ.get("OPTBENCH_PAPER_LIMIT", "").strip()
    if not raw:
        return None
    return int(raw)


def _mode_dump(start: np.ndarray, reactant: np.ndarray, free: np.ndarray) -> str:
    delta = np.asarray(start, dtype=np.float64) - np.asarray(reactant, dtype=np.float64)
    delta[~np.asarray(free, dtype=bool)] = 0.0
    flat = delta.reshape(-1)
    scale = float(np.linalg.norm(flat))
    if scale < 1e-14:
        raise ValueError("Pt displacement has no free direction")
    flat = flat / scale
    lines = ["0.0"]
    for index in range(0, flat.size, 3):
        lines.append(" ".join(f"{float(value):.16e}" for value in flat[index : index + 3]))
    return "\n".join(lines) + "\n"


def _jobs(root: Path, problem: dict, template: dict, spec: dict, preamble: str) -> list[dict]:
    from optbench.pes import read_con
    from optbench.surfaces import read_con as read_con_flexible

    archive = _stored(root, (problem.get("input") or {})["archive_path"])
    if spec["geometry"] == "ts-lj38":
        archive = _stored(root, "website/tsopt/LJ38/coords-xyz.tar.gz")
    elif spec["geometry"] == "rr-lj38":
        archive = _stored(root, "website/lowest-eigenvector/lj38/coords.tar.gz")
    texts = _members(archive)
    units = spec["units"]
    scale = 1.0 / R0 if units == "morse-2014" else 1.0
    cap = int((problem.get("convergence") or {})["max_force_calls"])
    rows: list[dict] = []

    def pack(name, start_rows, free, box, finish_rows=None, mode_text=None, target_text=None):
        optim_box = None if box is None else np.asarray(box, dtype=np.float64).reshape(-1)[:3] * scale
        rows.append(
            {
                "name": name,
                "kind": spec["kind"],
                "cap": cap,
                "timeout": int(spec["timeout"]),
                "preamble": preamble,
                "symbol": spec["symbol"],
                "start_rows": start_rows,
                "finish_rows": finish_rows,
                "free": np.asarray(free, dtype=bool),
                "box": None if box is None else np.asarray(box, dtype=np.float64).reshape(-1)[:3],
                "optim_box": optim_box,
                "scale_l": scale,
                "scale_e": (1.0 / D_WELL) if units == "morse-2014" else 1.0,
                "force_div": RHO if units == "morse-2014" else 1.0,
                "append_params": bool(spec.get("append_params")),
                "success_token": spec["success"],
                "periodic": box is not None,
                "vector_dump": mode_text,
                "target_vector": target_text,
            }
        )

    geometry = spec["geometry"]
    if geometry in {"de-lj38", "de-2ts"}:
        from optbench.pes import load_pairs

        pairs = load_pairs(archive, "xyz-pairs", int((problem.get("sample") or {})["pairs"]))
        by_index = {}
        for name, text in texts.items():
            stem = Path(name).name
            if stem.startswith("start_") and stem.endswith(".xyz"):
                by_index[int(stem[len("start_") : -len(".xyz")])] = name
        for pair in pairs:
            index = int(pair.name.split("_")[1])
            start = _xyz_rows(texts[by_index[index]])
            end_name = next(
                name
                for name in texts
                if Path(name).name in {f"end_{index:02d}.xyz", f"end_{index}.xyz"}
            )
            finish = _xyz_rows(texts[end_name])
            free = np.ones(len(start), dtype=bool)
            pack(pair.name, start, free, None, finish)
    elif geometry == "de-pt":
        from optbench.pes import load_pairs

        pairs = load_pairs(archive, "con-reactant-products", int((problem.get("sample") or {})["pairs"]))
        for pair in pairs:
            start = _scale_rows(pair.reactant, scale)
            finish = _scale_rows(pair.product, scale)
            pack(pair.name, start, pair.free, pair.box, finish)
    elif geometry == "morse-min":
        names = sorted(name for name in texts if name.endswith(".con"))
        for name in names:
            coords, free, box = read_con(texts[name])
            pack(Path(name).stem, _scale_rows(coords, scale), free, box)
    elif geometry == "ts-lj38":
        names = sorted(name for name in texts if name.endswith(".xyz"))
        for name in names:
            rows_xyz = _xyz_rows(texts[name])
            free = np.ones(len(rows_xyz), dtype=bool)
            pack(Path(name).stem, rows_xyz, free, None)
    elif geometry == "ts-pt":
        reactant_name = next(name for name in texts if Path(name).name == "reactant.con")
        reactant, fixed, _cell = read_con_flexible(texts[reactant_name])
        free = ~np.asarray(fixed, dtype=bool)
        names = sorted(
            (int(Path(name).stem.split("_")[1]), name)
            for name in texts
            if Path(name).name.startswith("initial_") and name.endswith(".con")
        )
        for index, name in names:
            coords, fixed_i, box = read_con_flexible(texts[name])
            free_i = ~np.asarray(fixed_i, dtype=bool)
            mode = _mode_dump(coords, reactant, free_i)
            pack(f"initial_{index}", _scale_rows(coords, scale), free_i, box, mode_text=mode)
            if not np.array_equal(free, free_i):
                raise ValueError(f"{name} freeze mask does not match the reactant")
    elif geometry == "rr-lj38":
        initial = next(name for name in texts if Path(name).name == "initial_mode")
        dump = "100\n" + texts[initial].strip() + "\n"
        names = sorted(name for name in texts if name.endswith(".xyz"))
        for name in names:
            mode_name = str(Path(name).with_name(Path(name).stem + "_mode"))
            rows_xyz = _xyz_rows(texts[name])
            free = np.ones(len(rows_xyz), dtype=bool)
            pack(Path(name).stem, rows_xyz, free, None, mode_text=dump, target_text=texts[mode_name])
    else:
        raise ValueError(f"no paper geometry {geometry}")
    expected = (problem.get("sample") or {}).get("starts") or (problem.get("sample") or {}).get("pairs")
    if expected is not None and len(rows) != int(expected):
        raise RuntimeError(f"{problem['id']} prepared {len(rows)} starts, the problem states {expected}")
    limit = _limit()
    if limit is not None:
        return rows[:limit]
    return rows


def _record(name: str, calls: int, cap: int, success: bool, reason: str) -> dict:
    if success:
        failure = None
        reasons: list[str] = []
    elif calls >= cap:
        failure = "cap"
        reasons = ["cap"]
    elif reason.startswith("OPTIM exceeded") or "Timeout" in reason:
        failure = "cap"
        reasons = ["cap"]
    elif reason:
        failure = "crash"
        reasons = ["crash"]
    else:
        failure = "divergence"
        reasons = ["divergence"]
    return {
        "schema": "optbench.start/v2",
        "start": name,
        "force_calls": int(calls),
        "verify_calls": 0,
        "counter": "optim-fcall",
        "comparable": True,
        "success": success,
        "failure": failure,
        "reasons": reasons,
        "energy": None,
        "force_norm": None,
        "hessian_index": None,
        "expected_index": None,
        "converged": success,
        "error": reason,
    }


def execute(job: dict) -> dict:
    """One OPTIM start. The stored count is OPTIM's energy+gradient calls."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    name = str(job["name"])
    cap = int(job["cap"])
    calls = cap
    success = False
    reason = ""
    print(f"optim-paper {name} start", flush=True)
    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="optim-paper-") as raw:
            folder = Path(raw)
            hook = folder / "hook"
            relay = hook / "pysocket" / "fmlip_relay.py"
            relay.parent.mkdir(parents=True)
            shutil.copyfile(_RELAY, relay)
            _write_xyz(folder / "geom.xyz", job["start_rows"], job["optim_box"])
            lines = [_ase_line(sys.executable, hook, bool(job["periodic"])), str(job["preamble"]).rstrip()]
            if job["append_params"]:
                if job["box"] is None:
                    raise RuntimeError(f"{name} has no cell")
                lines.append(_params_line(job["box"]))
            lines.extend(_freeze_lines(job["free"]))
            lines.append("points")
            lines.extend(_symbol_lines(job["start_rows"], str(job["symbol"])))
            (folder / "odata").write_text("\n".join(lines) + "\n", encoding="utf-8")
            if job["finish_rows"] is not None:
                (folder / "finish").write_text(_bare(job["finish_rows"]), encoding="utf-8")
            if job["vector_dump"] is not None:
                (folder / "vector.dump").write_text(str(job["vector_dump"]), encoding="utf-8")
            if job["target_vector"] is not None:
                (folder / "target.vector").write_text(str(job["target_vector"]), encoding="utf-8")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["OPTBENCH_KIND"] = str(job["kind"])
            env["OPTBENCH_CAP"] = "1000000000"
            env["OPTBENCH_COUNT_PATH"] = str(folder / "calls.txt")
            env["OPTBENCH_E_SCALE"] = f"{float(job['scale_e']):.16g}"
            env["OPTBENCH_L_SCALE"] = f"{float(job['scale_l']):.16g}"
            env["OPTBENCH_FORCE_DIV"] = f"{float(job['force_div']):.16g}"
            env.pop("OPTBENCH_CAPPED_PATH", None)
            if job["box"] is None:
                env.pop("OPTBENCH_BOX", None)
            else:
                lengths = np.asarray(job["box"], dtype=np.float64).reshape(-1)[:3]
                env["OPTBENCH_BOX"] = " ".join(f"{float(value):.16g}" for value in lengths)
            stdout_path = folder / "stdout.txt"
            stderr_path = folder / "stderr.txt"
            timed_out = False
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(
                    [str(binary_path())],
                    cwd=folder,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    proc.wait(timeout=int(job["timeout"]))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log = stdout_path.read_text(encoding="utf-8", errors="replace")
            err = stderr_path.read_text(encoding="utf-8", errors="replace")
            found = energy_gradient_calls(log)
            if found is None:
                found = energy_gradient_calls(err)
            if found is not None:
                calls = int(found)
            token = str(job["success_token"])
            matched = token in log or token in err
            if timed_out:
                reason = f"OPTIM exceeded {job['timeout']}s"
                success = False
            elif matched and calls < cap:
                success = True
            elif found is None and proc.returncode not in (0, None):
                tail = (err.strip().splitlines() or log.strip().splitlines() or ["no output"])[-1]
                reason = f"OPTIM exit {proc.returncode}: {tail}"
            elif not matched:
                reason = "OPTIM did not print the 2014 success line"
            else:
                reason = f"energy+gradient calls {calls} reached the cap {cap}"
            if os.environ.get("OPTBENCH_PAPER_KEEP"):
                destination = Path(os.environ["OPTBENCH_PAPER_KEEP"]) / name
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(folder, destination)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        success = False
    record = _record(name, calls, cap, success, reason)
    print(
        f"optim-paper {name} calls {record['force_calls']} success {int(success)} {reason}",
        flush=True,
    )
    return record


def run_paper(root: Path, problem: dict, template: dict, *, output: Path, workers: int) -> dict:
    spec = _PAPER.get(template["id"])
    if spec is None:
        raise RuntimeError(f"{template['id']} is not a 2014 OPTIM paper template")
    if problem["id"] != template.get("problem"):
        raise ValueError(f"{template['id']} belongs to {template.get('problem')}")
    preamble_name = str((template.get("optim") or {}).get("preamble", ""))
    preamble_path = root / preamble_name
    if not preamble_path.is_file():
        raise RuntimeError(f"{template['id']} preamble is not at {preamble_path}")
    preamble = preamble_path.read_text(encoding="utf-8")
    jobs = _jobs(root, problem, template, spec, preamble)
    from .common import run_jobs

    records = run_jobs(jobs, execute, workers)
    comment = (
        "# counter OPTIM energy+gradient calls. "
        "run_optim.sh greps that line and takes awk field 6.\n"
        f"# success is the 2014 string {spec['success']!r}. "
        f"odata preamble {preamble_path.name}. "
        f"units {spec['units']}."
    )
    parameters = {
        "method": str((template.get("optim") or {}).get("method", "")),
        "preamble": preamble_name,
        "counter": "optim-fcall",
        "success": spec["success"],
        "units": spec["units"],
        "socket": "ASE",
    }
    settings = {"cap": int((problem.get("convergence") or {})["max_force_calls"])}
    return write_campaign(
        output,
        records,
        template,
        problem,
        settings,
        parameters,
        images=None,
        comment=comment,
    )
