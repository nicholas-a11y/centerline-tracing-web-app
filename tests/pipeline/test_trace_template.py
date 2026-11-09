"""Trace template round trip: template -> simulated CAD export -> reference reader (Stage 2c)."""

from __future__ import annotations

import base64
import io
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from tests.pipeline.lesson import LESSONS_DIR
from tests.pipeline.svg_reference import SvgReferenceError, read_reference_svg
from tests.pipeline.trace_template import README_TEXT, build_template_files, fade_pixels, main

LESSON_IMAGE = LESSONS_DIR / "cross-t2-w8-a35" / "image.png"
W = H = 256
# Two traced strokes in bitmap pixel-edge coordinates (x, y), as a user would draw them.
LINE_A = [(0.0, 128.0), (256.0, 128.0)]
LINE_B = [(10.5, 200.25), (240.75, 30.5)]


def _expected_rc(line):
    return np.array([[y - 0.5, x - 0.5] for x, y in line])


@pytest.fixture(scope="module")
def files():
    return build_template_files(LESSON_IMAGE.read_bytes(), "cross")


def _png_b64(svg_text: str) -> str:
    return re.search(r'href="data:image/png;base64,([^"]+)"', svg_text).group(1)


def test_faded_png_size_dpi_and_contrast(files):
    with Image.open(io.BytesIO(files["cross-faded.png"])) as im:
        assert im.size == (W, H)
        assert im.info["dpi"] == pytest.approx((96, 96), abs=0.02)   # PNG stores 3780 px/m = 96.012 dpi
        faded = np.asarray(im.convert("L"), dtype=int)
    with Image.open(LESSON_IMAGE) as im:
        orig = np.asarray(im.convert("L"), dtype=np.uint8)
    np.testing.assert_array_equal(faded, fade_pixels(orig, 0.35))
    assert faded.min() == round(0.35 * 255)          # black strokes become light grey
    assert faded.max() == orig.max()


def test_fade_bounds():
    v = np.array([0, 100, 255], dtype=np.uint8)
    np.testing.assert_array_equal(fade_pixels(v, 0.0), v)
    np.testing.assert_array_equal(fade_pixels(v, 0.5), [128, 178, 255])
    with pytest.raises(ValueError):
        build_template_files(LESSON_IMAGE.read_bytes(), "x", fade=1.0)


def test_template_svg_shape(files):
    text = files["cross-template.svg"].decode("utf-8")
    root = ET.fromstring(text)
    assert root.get("width") == f"{W}px" and root.get("height") == f"{H}px"
    assert root.get("viewBox") == f"0 0 {W} {H}"
    children = list(root)
    assert len(children) == 1 and children[0].tag.endswith("image")
    img = children[0]
    assert [img.get(k) for k in ("x", "y", "width", "height")] == ["0", "0", str(W), str(H)]
    assert base64.b64decode(_png_b64(text)) == files["cross-faded.png"]
    assert "<!DOCTYPE" not in text


def test_template_alone_has_frame_but_no_geometry(files):
    with pytest.raises(SvgReferenceError) as exc:
        read_reference_svg(files["cross-template.svg"], (W, H))
    assert exc.value.code == "svg_no_geometry"


def test_readme_is_three_lines(files):
    assert files["cross-README.txt"].decode("utf-8") == README_TEXT
    assert len(README_TEXT.strip().splitlines()) == 3


def _check(svg: bytes | str):
    ref = read_reference_svg(svg.encode() if isinstance(svg, str) else svg, (W, H))
    assert ref.alignment == "image-frame"
    assert len(ref.paths) == 2
    for got, line in zip(ref.paths, (LINE_A, LINE_B)):
        np.testing.assert_allclose(got[[0, -1]], _expected_rc(line), atol=1e-6)
    return ref


def test_round_trip_px_export(files):
    """Paths drawn straight onto the template in px, the template kept as is."""
    text = files["cross-template.svg"].decode("utf-8")
    paths = "".join(f'<path d="M{a[0]} {a[1]} L{b[0]} {b[1]}" fill="none" stroke="#000"/>'
                    for a, b in (LINE_A, LINE_B))
    _check(text.replace("</svg>", paths + "</svg>"))


@pytest.mark.parametrize("dpi", [96.0, 72.0], ids=["96dpi", "72dpi"])
def test_round_trip_lightburn_style_mm_yflip(files, dpi):
    """mm document, Y axis up, offset origin, image placed with matrix(1,0,0,-1,...) and stored
    upside down, paths with their own Y-flip; the import dpi only changes the frame size."""
    k = 25.4 / dpi                                   # mm per bitmap pixel
    w_mm, h_mm = W * k, H * k
    ox, oy = 110.0122, 265.0814                      # bed position of the bitmap's bottom-left (Y up)
    cx, cy = ox + w_mm / 2, oy + h_mm / 2

    def to_bed(x, y):                                # bitmap pixel edge -> Y-up mm
        return ox + x * k, oy + (H - y) * k

    with Image.open(io.BytesIO(files["cross-faded.png"])) as im:
        buf = io.BytesIO()
        im.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(buf, format="PNG")
    stored = base64.b64encode(buf.getvalue()).decode()
    paths = ""
    for a, b in (LINE_A, LINE_B):
        (x0, y0), (x1, y1) = to_bed(*a), to_bed(*b)
        paths += (f'<path transform="matrix(1,0,0,-1,0,0)" style="fill:none;stroke:#000" '
                  f'd="M{x0:.6f},{y0:.6f}L{x1:.6f},{y1:.6f}"/>')
    svg = (f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
           f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{w_mm:.6f}mm" height="{h_mm:.6f}mm" '
           f'viewBox="{ox:.6f} {-(oy + h_mm):.6f} {w_mm:.6f} {h_mm:.6f}">'
           f'<image transform="matrix(1,0,0,-1,{cx:.6f},{-cy:.6f})" x="{-w_mm / 2:.6f}" y="{-h_mm / 2:.6f}" '
           f'width="{w_mm:.6f}" height="{h_mm:.6f}" xlink:href="data:image/png;base64,{stored}"/>'
           f'{paths}</svg>')
    ref = _check(svg)
    assert ref.frame["flip_y"] is True


def test_export_without_image_is_rejected_with_hint(files):
    """The most likely user mistake: hiding/deleting the image before exporting from a mm document."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="67.733mm" height="67.733mm" '
           'viewBox="110 -332.8 67.733 67.733"><path d="M110 -300 L177 -300"/></svg>')
    with pytest.raises(SvgReferenceError, match="Keep the bitmap in the export") as exc:
        read_reference_svg(svg.encode(), (W, H))
    assert exc.value.code == "svg_no_registration"


def test_cli_writes_three_files(tmp_path, capsys):
    assert main([str(LESSON_IMAGE), "--fade", "0.5", "--out", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["image-README.txt", "image-faded.png", "image-template.svg"]
    with Image.open(tmp_path / "image-faded.png") as im:
        assert np.asarray(im.convert("L")).min() == 128
    assert "image-template.svg" in capsys.readouterr().out


def test_cli_rejects_bad_fade(tmp_path):
    with pytest.raises(SystemExit):
        main([str(LESSON_IMAGE), "--fade", "1.5", "--out", str(tmp_path)])
