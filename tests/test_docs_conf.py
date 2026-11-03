import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONF = runpy.run_path(str(ROOT / "docs" / "source" / "conf.py"))
FLAG_FONT = CONF["_FLAG_FONT"] + "src:url(https://cdn.jsdelivr.net/flags.woff2)}"


def _build(tmp_path: Path, css: str, js: str):
    static = tmp_path / "_static"
    static.mkdir()
    (static / "shibuya.css").write_text(css, encoding="utf-8")
    (static / "shibuya.js").write_text(js, encoding="utf-8")
    app = SimpleNamespace(outdir=str(tmp_path), builder=SimpleNamespace(format="html"))
    CONF["_drop_remote_assets"](app, None)
    return static


def test_the_theme_assets_lose_every_remote_url(tmp_path):
    js = ";".join(old for old, _new in CONF["_JS_EDITS"])
    static = _build(tmp_path, "a{}" + FLAG_FONT + "b{}", js)
    assert (static / "shibuya.css").read_text(encoding="utf-8") == "a{}b{}"
    patched = (static / "shibuya.js").read_text(encoding="utf-8")
    assert "https://" not in patched
    assert "//cdn." not in patched


@pytest.mark.parametrize("old", [old for old, _new in CONF["_JS_EDITS"]])
def test_a_theme_without_a_target_stops_the_build(tmp_path, old):
    js = ";".join(other for other, _new in CONF["_JS_EDITS"] if other != old)
    with pytest.raises(CONF["ThemeAssetError"], match="shibuya.js has no") as caught:
        _build(tmp_path, FLAG_FONT, js)
    assert repr(old) in str(caught.value)


def test_a_theme_without_the_flag_font_stops_the_build(tmp_path):
    js = ";".join(old for old, _new in CONF["_JS_EDITS"])
    with pytest.raises(CONF["ThemeAssetError"], match="shibuya.css has no"):
        _build(tmp_path, "a{}", js)
