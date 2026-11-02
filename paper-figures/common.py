"""Shared loaders for the paper figures.

Each figure script draws one kind of figure. This module only reads the
archived board and the campaign start records. It does not rescore them.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PDF_DIR = ROOT / "paper-figures" / "pdf"
SVG_DIR = ROOT / "paper-figures" / "svg"
SITE_DIR = ROOT / "docs" / "source" / "_static" / "paper"
PARTS_DIR = ROOT / "paper-figures" / "parts"

# Okabe-Ito, yellow last because it is faint on white.
OKABE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
    "#F0E442",
)
SUCCESS = "#0072B2"
FAILURE = "#D55E00"
INK = "#1c140c"
MUTED = "#8a8178"

FAMILY_COLOR = {
    "minimize": "#0072B2",
    "min-mode": "#009E73",
    "single-ended": "#E69F00",
    "connect": "#CC79A7",
    "global": "#56B4E9",
    "band": "#D55E00",
}


def problem_slug(problem_id: str) -> str:
    return problem_id.replace("/", "-")


def code_token(code: str) -> str:
    base = re.sub(r"-+", "-", "".join(ch.lower() if ch.isalnum() else "-" for ch in code)).strip("-")
    digest = hashlib.sha1(code.encode()).hexdigest()[:4]
    return f"{base or 'code'}-{digest}"


def version_label(row: dict) -> str:
    """Recorded version, or the commit when the row has no version.

    A few harness rows stored the method after the version
    (``(develop 500350d9) · LBFGS``). The method is already its own field,
    so the axis keeps the version only.
    """
    version = str(row.get("code_version") or "").strip()
    method = str(row.get("method") or "").strip()
    leaked = f" · {method}" if method else ""
    if version and leaked and version.endswith(leaked):
        version = version[: -len(leaked)].strip()
    if version:
        return version
    commit = str(row.get("commit") or row.get("asv_commit") or "").strip()
    if commit:
        return commit[:7]
    return ""


def version_key(label: str) -> tuple:
    numbers = tuple(int(piece) for piece in re.findall(r"\d+", label))
    if numbers:
        return (0, numbers, label)
    return (1, (), label)


def load_board(root: Path | None = None) -> dict:
    root = root or ROOT
    return json.loads((root / "results" / "board" / "index.json").read_text(encoding="utf-8"))


def load_problems(root: Path | None = None) -> dict:
    from optbench.problems import load_all

    root = root or ROOT
    return load_all(root / "problems")


def _row_stems(row: dict) -> list[str]:
    stems = []
    for part in str(row.get("path") or "").split(";"):
        name = Path(part.strip()).name
        for suffix in (".starts.jsonl", ".v2.json", ".starts", ".dat"):
            if name.endswith(suffix):
                stems.append(name[: -len(suffix)])
                break
    return stems


def start_records(root: Path | None = None) -> dict[str, list[dict]]:
    """Per-start records keyed by campaign file stem.

    The loader is the one the board uses, so a failure is the recorded
    failure and not a new cutoff.
    """
    from optbench.board import (
        _campaign_groups,
        _cap_of,
        _first_raw,
        _force_tol,
        _image_count,
        _load_group_records,
    )
    from optbench.campaign import list_templates
    from optbench.dat import DatError, parse_lenient

    root = root or ROOT
    problems = load_problems(root)
    templates = {item["id"]: item for item in list_templates(root)} if (root / "templates").is_dir() else {}
    found: dict[str, list[dict]] = {}
    for stem, group in _campaign_groups(root / "results" / "campaign").items():
        fields: dict[str, str] = {}
        if "dat" in group:
            try:
                fields, _duplicates = parse_lenient(group["dat"].read_text(encoding="utf-8"))
            except (DatError, OSError):
                fields = {}
        probe = _first_raw(group)
        template_id = fields.get("template") or str(probe.get("template") or "")
        template = templates.get(template_id)
        problem_id = (
            fields.get("problem")
            or str(probe.get("problem") or "")
            or str((template or {}).get("problem") or "")
        )
        problem = problems.get(problem_id)
        cap = _cap_of(problem, template, fields)
        force_tol = _force_tol(problem, template)
        images = _image_count(problem, template, fields)
        try:
            records, _source = _load_group_records(group, cap, force_tol, images)
        except (OSError, ValueError, json.JSONDecodeError):
            records = []
        found[stem] = records
    return found


def records_for_row(row: dict, by_stem: dict[str, list[dict]]) -> list[dict]:
    for stem in _row_stems(row):
        records = by_stem.get(stem) or []
        if records:
            return records
    return []


# Journal single column, and a full text line when the labels need it.
COLUMN_IN = 3.5
TEXT_IN = 7.05


def apply_style() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlepad": 4,
        }
    )
    _install_compact_svg_markers()


def _install_compact_svg_markers(min_points: int = 24) -> None:
    """Write many identical markers as one group of short <use> nodes.

    One plotted point per artist makes a multi-megabyte SVG. The PDF
    backend is left alone, so the print file stays a normal vector figure.
    """
    import matplotlib.backends.backend_svg as svg_backend

    renderer = svg_backend.RendererSVG
    if getattr(renderer, "_optbench_compact", False):
        return
    original = renderer.draw_markers

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        try:
            count = len(path.vertices)
        except (AttributeError, TypeError):
            count = 0
        if count < min_points:
            return original(self, gc, marker_path, marker_trans, path, trans, rgbFace)
        writer = self.writer
        path_data = self._convert_path(
            marker_path,
            marker_trans + svg_backend.Affine2D().scale(1.0, -1.0),
            simplify=False,
        )
        style = self._get_style_dict(gc, rgbFace)
        css = svg_backend._generate_css(style)
        key = (path_data, css)
        oid = self._markers.get(key)
        stroke_css = svg_backend._generate_css(
            {name: value for name, value in style.items() if name.startswith("stroke")}
        )
        if oid is None:
            oid = self._make_id("m", key)
            writer.start("defs")
            writer.element("path", id=oid, d=path_data, style=stroke_css)
            writer.end("defs")
            self._markers[key] = oid
        writer.start("g", style=self._get_style(gc, rgbFace), **self._get_clip_attrs(gc))
        trans_and_flip = self._make_flip_transform(trans)
        href = f"#{oid}"
        for vertices, _code in path.iter_segments(trans_and_flip, simplify=False):
            if len(vertices) == 0:
                continue
            x, y = vertices[-2:]
            writer.element(
                "use",
                attrib={
                    "xlink:href": href,
                    "x": svg_backend._short_float_fmt(x),
                    "y": svg_backend._short_float_fmt(y),
                },
            )
        writer.end("g")

    renderer.draw_markers = draw_markers
    renderer._optbench_compact = True


def fit_labels(labels: list[str], fontsize: float = 7.0, max_in: float = 3.05) -> tuple[list[str], float]:
    """Wrap a label only when it is wider than the space beside a column."""
    import textwrap

    import matplotlib.pyplot as plt

    def measure(items: list[str]) -> float:
        fig = plt.figure(figsize=(1, 1))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        widest = 0.0
        for label in items:
            artist = fig.text(0, 0, label, fontsize=fontsize)
            widest = max(widest, artist.get_window_extent(renderer).width / fig.dpi)
            artist.remove()
        plt.close(fig)
        return widest

    wrapped = list(labels)
    inches = measure(wrapped)
    width_chars = 46
    while inches > max_in and width_chars >= 22:
        wrapped = [
            textwrap.fill(label, width=width_chars, break_long_words=False, break_on_hyphens=False)
            for label in labels
        ]
        inches = measure(wrapped)
        if inches <= max_in:
            break
        width_chars -= 6
    return wrapped, inches


def categorical_figure(n_rows: int, labels: list[str], legend_rows: int = 0):
    """A column-width data region, widened so the recorded labels stay whole."""
    import matplotlib.pyplot as plt

    wrapped, label_in = fit_labels(labels)
    extra_lines = sum(label.count("\n") for label in wrapped)
    units = n_rows + extra_lines
    row = 0.32 if units > 12 else 0.46
    width = label_in + COLUMN_IN + 0.78
    width = min(TEXT_IN, max(width, COLUMN_IN + 1.5))
    height = 0.62 + 0.36 * legend_rows + row * units
    fig, ax = plt.subplots(figsize=(width, max(2.2, height)), layout="constrained")
    return fig, ax, wrapped


def display_title(problem_id: str, problems: dict, fallback: str) -> str:
    """The same heading the site uses for this problem."""
    from optbench.site_pages import human_title

    problem = problems.get(problem_id)
    if not problem:
        return fallback
    by_slug: dict[str, list] = {}
    for item in problems.values():
        by_slug.setdefault(str(item.get("problem") or ""), []).append(item)
    return human_title(problem, by_slug)


def _standalone_svg(path: Path) -> None:
    """Keep the SVG self-contained: no external doctype, metadata, or xlink namespace."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!DOCTYPE[^>]*>\s*", "", text, count=1)
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, count=1, flags=re.DOTALL)
    text = text.replace(' xmlns:xlink="http://www.w3.org/1999/xlink"', "")
    text = text.replace("xlink:href", "href")
    path.write_text(text, encoding="utf-8")


def _stamp_caption(pdf_path: Path, caption: str) -> None:
    """Add the caption under the saved figure.

    A tight bounding box repositions an outside legend onto a caption drawn
    in the same figure, so the caption is appended to the finished page.
    """
    import textwrap

    import pymupdf

    src = pymupdf.open(pdf_path)
    page_src = src[0]
    width = float(page_src.rect.width)
    lines = textwrap.wrap(caption, width=max(36, int(width / 3.35))) or [caption]
    leading = 8
    extra = 8 + leading * len(lines)
    out = pymupdf.open()
    page = out.new_page(width=width, height=float(page_src.rect.height) + extra)
    page.show_pdf_page(pymupdf.Rect(0, 0, width, float(page_src.rect.height)), src, 0)
    rect = pymupdf.Rect(1, float(page_src.rect.height) + 2, width - 1, float(page_src.rect.height) + extra - 1)
    unused = page.insert_textbox(
        rect,
        "\n".join(lines),
        fontsize=6,
        fontname="helv",
        color=(28 / 255, 20 / 255, 12 / 255),
        align=0,
    )
    if unused < 0:
        src.close()
        out.close()
        raise RuntimeError(f"caption does not fit on {pdf_path.name}")
    src.close()
    tmp = pdf_path.with_suffix(".pdf.tmp")
    out.save(tmp, garbage=4, deflate=True)
    out.close()
    tmp.replace(pdf_path)


def save_pair(fig, stem: str, caption: str) -> dict[str, str]:
    """Write the site SVG, then the print PDF with the caption under the figure."""
    import shutil

    import matplotlib.pyplot as plt

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    svg_name = f"{stem}.svg"
    pdf_name = f"{stem}.pdf"
    fig.savefig(SVG_DIR / svg_name, format="svg", bbox_inches="tight", facecolor="white")
    _standalone_svg(SVG_DIR / svg_name)
    shutil.copyfile(SVG_DIR / svg_name, SITE_DIR / svg_name)
    pdf_path = PDF_DIR / pdf_name
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _stamp_caption(pdf_path, caption)
    shutil.copyfile(pdf_path, SITE_DIR / pdf_name)
    return {"svg": svg_name, "pdf": pdf_name}


def write_part(name: str, problems: dict, codes: dict | None = None) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"problems": problems, "codes": codes or {}}
    (PARTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def log_scale(values: list[float]) -> bool:
    """Same display rule as the site cost chart. Not a scoring cutoff."""
    from optbench.site_charts import spans_a_decade

    return spans_a_decade(values)
