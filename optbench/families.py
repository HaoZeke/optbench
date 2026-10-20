"""Method families for a ranked row.

A ranked row is keyed by code, version, and method or preset. The family
is derived from that method. ``docs/design/families.md`` is the map.
"""

from __future__ import annotations

FAMILIES: tuple[tuple[str, str], ...] = (
    ("band", "Band / NEB"),
    ("min-mode", "Minimum mode"),
    ("single-ended", "Single-ended saddle"),
    ("connect", "Connect / double-ended"),
    ("global", "Basin hopping / global"),
    ("minimize", "Minimizers"),
)

FAMILY_TITLE = dict(FAMILIES)

# Cartesian coordinates for the finite-difference Hessian equivalent.
# Central differences call the force twice per coordinate.
COORDINATES: dict[str, int] = {
    "minimization/lj38": 114,
    "minimization/lj38-noise": 114,
    "tsopt/LJ38": 114,
    "lowest-eigenvector/lj38": 114,
    "double-ended-saddle-search/lj38": 114,
    "double-ended-saddle-search/lj38_2ts_separation": 114,
    "globalopt/lj38": 114,
    "globalopt/lj75": 225,
    "globalopt/lj98": 294,
    "globalopt/lj100-binary": 300,
    "globalopt/lj150": 450,
    "saddle-search/pt-island": 1029,
    "tsopt/pt-island": 1029,
    "double-ended-saddle-search/pt-island": 1029,
    "neb/Pt-heptamer": 1029,
    "hTST/Pt": 1029,
    "accelerated-md/aladatom": 303,
    "accelerated-md/standard_aladatom": 303,
    "hTST/Al": 303,
}

# Result-file tokens that keep a row off the ranked table. Both the
# previous spellings and the renamed spellings are rejected.
ADJUSTED_TOKENS = frozenset({"tuned", "adjusted"})
SHORT_TOKENS = frozenset({"partial", "short"})


def family_title(family_id: str) -> str:
    return FAMILY_TITLE.get(family_id, family_id)


def coordinate_count(problem: dict | None) -> int | None:
    """Coordinates in one finite-difference Hessian, when the problem states them."""
    if not problem:
        return None
    stated = problem.get("coordinates")
    if isinstance(stated, int) and stated > 0:
        return stated
    mobile = problem.get("mobile_atoms")
    atoms = problem.get("atoms")
    if mobile not in (None, ""):
        return 3 * int(mobile)
    if atoms not in (None, ""):
        return 3 * int(atoms)
    return COORDINATES.get(str(problem.get("id") or ""))


def finite_difference_calls(hessian_calls: float | None, coordinates: int | None) -> float | None:
    """Force calls a central-difference Hessian would spend for this analytic count."""
    if hessian_calls is None:
        return None
    if hessian_calls == 0:
        return 0.0
    if coordinates is None:
        return None
    return float(hessian_calls) * 2.0 * int(coordinates)


def version_text(code_version: str, commit: str = "") -> str:
    text = str(code_version or "").strip()
    if len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        text = text[1:-1].strip()
    if text:
        return text
    commit = str(commit or "").strip()
    if not commit:
        return ""
    if commit.startswith("r") and commit[1:].isdigit():
        return commit
    return commit[:12]


def display_code(code: str, registry: list[dict] | None = None) -> str:
    text = str(code or "").strip()
    if not text or not registry:
        return text
    key = text.lower()
    for pin in registry:
        names = {str(pin.get("id") or "").lower(), str(pin.get("name") or "").lower()}
        names.discard("")
        if key in names and pin.get("name"):
            return str(pin["name"])
    return text


def _plain_name(value: object) -> str:
    """A stock method name. A parameter record is not a method."""
    if isinstance(value, dict):
        return ""
    text = str(value or "").strip()
    if not text or text[0] in "{[":
        return ""
    return text


def method_name(template: dict | None, fields: dict | None = None, algorithm: str = "") -> str:
    """Stock method or preset. An explicit method or preset wins over the optimizer name."""
    template = template or {}
    fields = fields or {}
    for source in (fields, template):
        preset = _plain_name(source.get("preset"))
        if preset:
            return preset
        method = _plain_name(source.get("method"))
        if method:
            return method
    named = _plain_name(fields.get("algorithm") or template.get("algorithm") or algorithm)
    if named:
        return named
    return str(template.get("id") or fields.get("template") or "").strip()


def row_label(code: str, version: str, method: str) -> str:
    """``eOn (develop 500350d9) · Lanczos improved``."""
    code = str(code or "").strip()
    version = str(version or "").strip()
    method = str(method or "").strip()
    if code and version and method:
        return f"{code} ({version}) · {method}"
    if code and method:
        return f"{code} · {method}"
    if code and version:
        return f"{code} ({version})"
    return code or method or version


def family_of(method: str, *, suite: str = "", kind: str = "") -> str:
    """Pick one family from the method string, then the problem suite."""
    text = str(method or "").lower()
    kind_text = str(kind or "").lower()
    suite_text = str(suite or "")
    if any(token in text for token in ("hybrid eigenvector", "dijkstra", "connect")):
        return "connect"
    if any(token in text for token in ("neb", "nudged", "elastic band")):
        return "band"
    if kind_text in {"de-2ts-mode", "min-mode", "mode"} or any(
        token in text for token in ("dimer", "lanczos", "lowest eigenvector", "min-mode", "minimum mode", "ssdimer")
    ):
        return "min-mode"
    if "sella" in text and suite_text == "minimization":
        return "minimize"
    if any(token in text for token in ("p-rfo", "prfo", " rfo", "artn", "eigenvector following", "sella", "softsaddle")):
        return "single-ended"
    if any(
        token in text
        for token in ("basin", "hopping", "paving", "bhpt", "airss", "minima hopping", "symmetris", "monte carlo")
    ):
        return "global"
    if suite_text == "globalopt":
        return "global"
    if suite_text == "lowest-eigenvector":
        return "min-mode"
    if suite_text in {"tsopt", "saddle-search"}:
        return "single-ended"
    if suite_text in {"neb", "double-ended-saddle-search"}:
        return "connect"
    return "minimize"


def is_adjusted(row: dict) -> bool:
    tuning = str(row.get("tuning") or row.get("profile") or "").strip().lower()
    if tuning in ADJUSTED_TOKENS:
        return True
    template = str(row.get("template") or "")
    return template.endswith("-tuned") or template.endswith("-adjusted")


def is_short_token(value: str) -> bool:
    return str(value or "").strip().lower() in SHORT_TOKENS
