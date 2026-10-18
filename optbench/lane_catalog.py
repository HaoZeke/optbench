"""Templates for the lane-B harness.

One template is one client, one problem, and one profile. Adding a
later code is a registry block plus a spec in this list.
"""

from __future__ import annotations

from pathlib import Path

SCIPY = ("scipy", "1.18.1", "e4e854eaa8f18d807cd3496028e257e36caa93cc")
PELE = ("pele", "0.1.2", "b7081b932a060be11342fbe8287ac35da8ada879")
SQNM = ("sqnm", "1.3", "4241692173cccb42a9ae05974bfd62285ec2cd7b")
GEOMETRIC = ("geometric", "1.1.1", "f657e69bae9c81bf7a23e7375db98502ca5c8c30")
ASE = ("ASE", "3.29.0", "f27c0005ae6a67ea419f996e728668865bfc1f86")
ASE_GA = ("ase-ga", "1.0.3", "f3ecaa80915d5ddbf2298e17702e6df21f5e7b9c")
NLOPT = ("nlopt", "2.11.0", "88c424d4f458412787df96fcc95218acbca224fd")
OPTKING = ("optking", "767e2c0c", "767e2c0c13ff2b851ac82101ccd5aab9a558a4f1")

_MIN = {
    "minimization/lj38": (0.01, 10000, "lj38"),
    "minimization/morse-bulk": (0.001, 10000, "morse"),
}
_GLOBAL = {
    "globalopt/lj38": (1000000, "lj38"),
    "globalopt/lj75": (500000, "lj75"),
    "globalopt/lj98": (1000000, "lj98"),
    "globalopt/lj150": (1000000, "lj150"),
    "globalopt/lj100-binary": (2000000, "binary"),
}
_SCIPY_MIN = (
    ("lbfgsb", "L-BFGS-B"),
    ("bfgs", "BFGS"),
    ("cg", "CG"),
    ("newton-cg", "Newton-CG"),
    ("tnc", "TNC"),
    ("trust-exact", "trust-exact"),
    ("trust-krylov", "trust-krylov"),
    ("trust-ncg", "trust-ncg"),
)
_SIGMA = (0.0, 1e-4, 1e-3, 1e-2)
_FOLDERS = ("scipy", "pele", "sqnm", "geometric", "ase-global", "ase-ga", "nlopt", "optking")


def specs() -> list[dict]:
    found: list[dict] = []
    for problem, (gtol, cap, slug) in _MIN.items():
        for name, method in _SCIPY_MIN:
            found.append(
                _spec(
                    f"scipy/min-{slug}-{name}",
                    SCIPY,
                    method,
                    problem,
                    "minimize",
                    "default",
                    {
                        "family": "minimize",
                        "method": "scipy-minimize",
                        "scipy_method": method,
                    },
                    gtol,
                    cap,
                    (
                        "SciPy TNC. The harness stops at the problem Euclidean force norm. "
                        "maxfun is the force-call cap. maxiter is omitted because SciPy 1.18 rejects it."
                        if method == "TNC"
                        else (
                            f"SciPy {method}. The harness stops at the problem Euclidean force norm. "
                            "The SciPy gradient tolerance is 1e-14, and maxiter is the force-call cap."
                        )
                    ),
                )
            )
        found.extend(_pele_min(problem, gtol, cap, slug))
        found.append(
            _spec(
                f"sqnm/min-{slug}-default",
                SQNM,
                "SQNM",
                problem,
                "minimize",
                "default",
                {"family": "minimize", "method": "sqnm"},
                gtol,
                cap,
                (
                    "vc-SQNM free_sqnm ASE-wrapper defaults: initial_step_size -0.01, nhist_max 10, "
                    "alpha_min 1e-3, eps_subsp 1e-3. On the archived LJ38 starts the force norm is "
                    "about 1e5 to 1e6, so the first displacement dissociates the cluster and the "
                    "force stop is met at a near-zero energy. This row is not ranked."
                ),
                ranked=False,
            )
        )
        found.append(
            _spec(
                f"sqnm/min-{slug}-tuned",
                SQNM,
                "SQNM",
                problem,
                "minimize",
                "tuned",
                {"family": "minimize", "method": "sqnm"},
                gtol,
                cap,
                (
                    "The first displacement has Euclidean length 0.01: initial_step_size is -0.01 "
                    "divided by the force norm at the start. SQNM then estimates the step. "
                    "nhist_max is 10, alpha_min is 1e-3, and eps_subsp is 1e-3."
                ),
            )
        )
        found.append(
            _spec(
                f"geometric/min-{slug}",
                GEOMETRIC,
                "geomeTRIC",
                problem,
                "minimize",
                "default",
                {"family": "minimize", "method": "geometric", "coordsys": "cart"},
                gtol,
                cap,
                (
                    "geomeTRIC 1.1.1 in Cartesian coordinates. Carbon labels on the first LJ38 start "
                    "produce 662 bonds, and both TRIC and the stock Cartesian setup were still "
                    "building primitive internals after 20 seconds, so primitive generation is skipped. "
                    "convergence_grms and convergence_gmax are the Euclidean stop divided by sqrt(N). "
                    "Energy and displacement criteria are loose. maxiter is the force-call cap."
                ),
            )
        )
        lj38 = slug == "lj38"
        found.append(
            _spec(
                f"nlopt/min-{slug}-tnewton",
                NLOPT,
                "LD_TNEWTON",
                problem,
                "minimize",
                "default",
                {"family": "minimize", "method": "nlopt-minimize", "nlopt_algorithm": "LD_TNEWTON"},
                gtol,
                cap,
                (
                    "NLopt 2.11.0 LD_TNEWTON. maxeval is the force-call cap. The objective calls "
                    "force_stop when the Euclidean force norm meets the problem stop. "
                    "LD_LBFGS, LD_VAR1, and LD_SLSQP leave the first LJ38 start in two evaluations "
                    "and are not ranked."
                    + (
                        " On the archived LJ38 starts, hundreds of those force stops are dissociated "
                        "clusters with energy near zero, so this row is not ranked."
                        if lj38
                        else ""
                    )
                ),
                ranked=not lj38,
            )
        )
        if slug == "lj38":
            found.append(
                _spec(
                    "optking/min-lj38",
                    OPTKING,
                    "OptKing",
                    problem,
                    "minimize",
                    "default",
                    {"family": "minimize", "method": "optking", "coordsys": "cart"},
                    gtol,
                    cap,
                    (
                        "OptKing Cartesian coordinates. CustomHelper receives the harness energy and "
                        "gradient. Redundant internals on the first LJ38 start do not finish coordinate "
                        "setup in 20 seconds. The harness stops at the problem Euclidean force norm."
                    ),
                )
            )
    for sigma in _SIGMA:
        label = _sigma_label(sigma)
        for code, method, algorithm, extra, profile, ident in (
            (SCIPY, "scipy-minimize", "L-BFGS-B", {"scipy_method": "L-BFGS-B"}, "default", f"scipy/noise-lj38-lbfgsb-{label}"),
            (PELE, "pele-lbfgs", "LBFGS", {}, "default", f"pele/noise-lj38-lbfgs-{label}"),
            (SQNM, "sqnm", "SQNM", {}, "tuned", f"sqnm/noise-lj38-tuned-{label}"),
            (GEOMETRIC, "geometric", "geomeTRIC", {"coordsys": "cart"}, "default", f"geometric/noise-lj38-{label}"),
            (OPTKING, "optking", "OptKing", {"coordsys": "cart"}, "default", f"optking/noise-lj38-{label}"),
        ):
            search = {"family": "minimize", "method": method, "sigma": sigma}
            search.update(extra)
            found.append(
                _spec(
                    ident,
                    code,
                    algorithm,
                    "minimization/lj38-noise",
                    "minimize",
                    profile if code is SQNM else label,
                    search,
                    0.01,
                    10000,
                    (
                        f"sigma {sigma}. Noise is drawn inside the harness evaluation, so one geometry keeps one sample. "
                        "Positive sigma is compared with an uncounted clean minimization of the same start."
                        + (
                            " SQNM uses the tuned first displacement of length 0.01."
                            if code is SQNM
                            else ""
                        )
                    ),
                )
            )
    for problem, (cap, slug) in _GLOBAL.items():
        for name, method, algorithm, note in (
            ("bh", "scipy-bh", "basinhopping", "SciPy basinhopping defaults T=1 and stepsize=0.5. niter is the force-call cap."),
            ("da", "scipy-da", "dual_annealing", "SciPy dual_annealing library defaults. maxiter and maxfun are the force-call cap."),
            ("de", "scipy-de", "differential_evolution", "SciPy differential_evolution library defaults. maxiter is the force-call cap. Polish is the only quench that can hit."),
            (
                "shgo",
                "scipy-shgo",
                "shgo",
                "SciPy shgo uses sobol sampling with n=100. The default simplicial rule replaces n=100 with 2**dimension+1, which does not return on these clusters. iters is the force-call cap.",
            ),
        ):
            found.append(
                _spec(
                    f"scipy/{name}-{slug}",
                    SCIPY,
                    algorithm,
                    problem,
                    "globalopt",
                    "default",
                    {"family": "globalopt", "method": method},
                    None,
                    cap,
                    note + " A hit is the first successful local minimization inside the energy window.",
                )
            )
        found.append(
            _spec(
                f"pele/bh-{slug}-default",
                PELE,
                "basin hopping",
                problem,
                "globalopt",
                "default",
                {"family": "globalopt", "method": "pele-bh"},
                None,
                cap,
                "pele basin hopping. temperature 1, stepsize 1, mylbfgs quench at the library defaults.",
            )
        )
        found.append(
            _spec(
                f"pele/bh-{slug}-tuned",
                PELE,
                "basin hopping",
                problem,
                "globalopt",
                "tuned",
                {"family": "globalopt", "method": "pele-bh"},
                None,
                cap,
                "pele basin hopping. temperature 1, stepsize 0.5, LBFGS memory 100, maxstep 0.2, quench RMS tolerance 0.001/sqrt(N).",
            )
        )
        found.append(
            _spec(
                f"ase-global/bh-{slug}-default",
                ASE,
                "BasinHopping",
                problem,
                "globalopt",
                "default",
                {"family": "globalopt", "method": "ase-bh"},
                None,
                cap,
                "ASE BasinHopping constructor defaults: temperature 100 kB, FIRE, fmax 0.1, dr 0.1. The seed is numpy.random.seed.",
            )
        )
        found.append(
            _spec(
                f"ase-global/bh-{slug}-tuned",
                ASE,
                "BasinHopping",
                problem,
                "globalopt",
                "tuned",
                {"family": "globalopt", "method": "ase-bh"},
                None,
                cap,
                "ASE BasinHopping with temperature 1, LBFGS, fmax 0.001, dr 0.5. fmax is the maximum per-atom force norm.",
            )
        )
        found.append(
            _spec(
                f"ase-global/mh-{slug}-default",
                ASE,
                "MinimaHopping",
                problem,
                "globalopt",
                "default",
                {"family": "globalopt", "method": "ase-mh"},
                None,
                cap,
                "ASE MinimaHopping constructor defaults, including T0 1000 K, Ediff0 0.5, fmax 0.05.",
            )
        )
        found.append(
            _spec(
                f"ase-global/mh-{slug}-tuned",
                ASE,
                "MinimaHopping",
                problem,
                "globalopt",
                "tuned",
                {"family": "globalopt", "method": "ase-mh"},
                None,
                cap,
                "ASE MinimaHopping with fmax 0.001. T0 and Ediff0 stay at the constructor defaults.",
            )
        )
        found.append(
            _spec(
                f"ase-ga/{slug}",
                ASE_GA,
                "RandomMutation",
                problem,
                "globalopt",
                "default",
                {"family": "globalopt", "method": "ase-ga"},
                None,
                cap,
                "ase-ga RandomMutation length 2, then an ASE LBFGS quench at fmax 0.001. A raw mutant energy is not a hit.",
            )
        )
    found.append(
        _spec(
            "scipy/eig-lj38-eigsh",
            SCIPY,
            "eigsh",
            "lowest-eigenvector/lj38",
            "lowest-eigenvector",
            "default",
            {"family": "eigen", "method": "scipy-eigsh"},
            None,
            10000,
            "eigsh which=SA on a central-difference Hessian-vector product. One matvec is two force calls.",
        )
    )
    found.append(
        _spec(
            "scipy/eig-lj38-lobpcg",
            SCIPY,
            "LOBPCG",
            "lowest-eigenvector/lj38",
            "lowest-eigenvector",
            "default",
            {"family": "eigen", "method": "scipy-lobpcg"},
            None,
            10000,
            "LOBPCG largest=False on the same central-difference product, from a seeded normal start.",
        )
    )
    return found


def _pele_min(problem: str, gtol: float, cap: int, slug: str) -> list[dict]:
    rows = []
    for profile, note in (
        ("default", "pele LBFGS memory 4, maxstep 0.1. The RMS tolerance is the Euclidean stop divided by sqrt(N)."),
        ("tuned", "pele LBFGS memory 100, maxstep 0.2. The RMS tolerance is the Euclidean stop divided by sqrt(N)."),
    ):
        rows.append(
            _spec(
                f"pele/min-{slug}-lbfgs-{profile}",
                PELE,
                "LBFGS",
                problem,
                "minimize",
                profile,
                {"family": "minimize", "method": "pele-lbfgs"},
                gtol,
                cap,
                note,
            )
        )
    for profile, note in (
        ("default", "pele FIRE maxstep 0.5. The stop is the Euclidean force norm."),
        ("tuned", "pele FIRE maxstep 0.2. The stop is the Euclidean force norm."),
    ):
        rows.append(
            _spec(
                f"pele/min-{slug}-fire-{profile}",
                PELE,
                "FIRE",
                problem,
                "minimize",
                profile,
                {"family": "minimize", "method": "pele-fire"},
                gtol,
                cap,
                note,
            )
        )
    return rows


def _sigma_label(sigma: float) -> str:
    if sigma == 0.0:
        return "s0"
    return "s" + f"{sigma:.0e}".replace("+", "").replace("-", "m")


def _spec(ident, pin, algorithm, problem, task, profile, search, gtol, cap, note, *, ranked: bool = True) -> dict:
    code, version, commit = pin
    row = {
        "id": ident,
        "engine": "harness",
        "code": code,
        "code_version": version,
        "commit": commit,
        "algorithm": algorithm,
        "problem": problem,
        "task": task,
        "profile": profile,
        "tuning": profile,
        "reducer": "capped",
        "ranked": ranked,
        "note": note,
        "search": search,
        "stop": {"max_force_calls": cap},
    }
    if gtol is not None:
        row["stop"]["norm"] = "l2"
        row["stop"]["force_norm"] = gtol
    return row


def _toml(spec: dict) -> str:
    lines = []
    for key in (
        "id",
        "engine",
        "code",
        "code_version",
        "commit",
        "algorithm",
        "problem",
        "task",
        "profile",
        "tuning",
        "reducer",
        "ranked",
        "note",
    ):
        lines.append(f"{key} = {_scalar(spec[key])}")
    lines.append("")
    lines.append("[search]")
    for key, value in spec["search"].items():
        lines.append(f"{key} = {_scalar(value)}")
    lines.append("")
    lines.append("[stop]")
    for key, value in spec["stop"].items():
        lines.append(f"{key} = {_scalar(value)}")
    lines.append("")
    return "\n".join(lines)


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_templates(root: Path) -> list[Path]:
    rows = specs()
    written: list[Path] = []
    expected: set[Path] = set()
    for spec in rows:
        path = root / "templates" / f"{spec['id']}.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_toml(spec), encoding="utf-8")
        written.append(path)
        expected.add(path.resolve())
    for folder in _FOLDERS:
        directory = root / "templates" / folder
        if not directory.is_dir():
            continue
        for path in directory.glob("*.toml"):
            if path.resolve() not in expected:
                path.unlink()
    return written
