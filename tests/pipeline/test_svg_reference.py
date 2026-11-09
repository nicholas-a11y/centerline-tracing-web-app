from __future__ import annotations

import base64
import io
import math

import numpy as np
import pytest
from PIL import Image

from tests.pipeline import svg_reference
from tests.pipeline.svg_reference import SvgReferenceError, read_reference_svg


def _svg(body: str, w=100, h=100, extra="") -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" {extra}>{body}</svg>').encode()


def _one(body, size=(100, 100), **kw):
    ref = read_reference_svg(_svg(body, *size, **kw), size)
    return ref


def _xy(pts):
    """Back to SVG (x, y) for readable assertions."""
    return np.stack([pts[:, 1] + 0.5, pts[:, 0] + 0.5], axis=1)


@pytest.mark.parametrize("payload", [
    b'<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY x "y">]><svg xmlns="http://www.w3.org/2000/svg"/>',
    b'<!ENTITY lol "lol"><svg/>',
    b'<!doctype svg><svg/>',
])
def test_rejects_doctype_and_entity(payload):
    with pytest.raises(SvgReferenceError, match="DOCTYPE or ENTITY"):
        read_reference_svg(payload, (10, 10))


def test_rejects_large_file():
    with pytest.raises(SvgReferenceError, match="limit"):
        read_reference_svg(_svg('<path d="M0 0 L1 1"/>' + " " * 5000), (100, 100), max_bytes=1000)


def test_rejects_too_many_elements():
    body = '<line x1="0" y1="0" x2="1" y2="1"/>' * 6
    with pytest.raises(SvgReferenceError, match="geometry elements"):
        read_reference_svg(_svg(body), (100, 100), max_elements=5)


def test_rejects_too_many_points():
    with pytest.raises(SvgReferenceError, match="too detailed"):
        read_reference_svg(_svg('<path d="M0 0 L99 99"/>'), (100, 100), max_points=50)


def test_rejects_use():
    with pytest.raises(SvgReferenceError, match="use") as exc:
        _one('<path id="a" d="M0 0 L5 5"/><use href="#a"/>')
    assert exc.value.code == "svg_use_unsupported"


def test_rejects_size_mismatch_with_fix_hint():
    with pytest.raises(SvgReferenceError, match="does not match the bitmap size") as exc:
        read_reference_svg(_svg('<path d="M0 0 L5 5"/>', 50, 50), (100, 100))
    assert exc.value.code == "svg_no_registration"
    assert "trace template" in str(exc.value)


# --- transforms -------------------------------------------------------------

def _ends(body, size=(100, 100)):
    return _xy(_one(body, size).paths[0])[[0, -1]]


@pytest.mark.parametrize("body, expected", [
    ('<g transform="translate(10,20)"><path d="M0 0 L10 0"/></g>', [[10, 20], [20, 20]]),
    ('<g transform="translate(10)"><path d="M0 0 L10 0"/></g>', [[10, 0], [20, 0]]),
    ('<path transform="scale(2)" d="M1 1 L5 1"/>', [[2, 2], [10, 2]]),
    ('<path transform="scale(2,3)" d="M1 1 L5 1"/>', [[2, 3], [10, 3]]),
    ('<path transform="rotate(90)" d="M10 0 L20 0"/>', [[0, 10], [0, 20]]),
    ('<path transform="rotate(90 50 50)" d="M60 50 L70 50"/>', [[50, 60], [50, 70]]),
    ('<path transform="skewX(45)" d="M0 10 L0 20"/>', [[10, 10], [20, 20]]),
    ('<path transform="skewY(45)" d="M10 0 L20 0"/>', [[10, 10], [20, 20]]),
    ('<path transform="matrix(1,0,0,-1,0,100)" d="M10 20 L30 20"/>', [[10, 80], [30, 80]]),
    # a list applies right-to-left to the points: scale first, then translate
    ('<path transform="translate(10,0) scale(2)" d="M1 1 L2 1"/>', [[12, 2], [14, 2]]),
    ('<path transform="scale(2) translate(10,0)" d="M1 1 L2 1"/>', [[22, 2], [24, 2]]),
    # nested groups compose outer-to-inner, same as the list form
    ('<g transform="translate(10,0)"><g transform="scale(2)"><path d="M1 1 L2 1"/></g></g>', [[12, 2], [14, 2]]),
    ('<g transform="translate(5 5)"><line transform="translate(1,1)" x1="0" y1="0" x2="4" y2="0"/></g>',
     [[6, 6], [10, 6]]),
])
def test_transforms(body, expected):
    np.testing.assert_allclose(_ends(body), expected, atol=1e-9)


@pytest.mark.parametrize("t", ["translate(1,2,3)", "rotate(1,2)", "perspective(1)", "scale(0)",
                               "translate(1) junk", "matrix(1,0,0,1,0)", "skewX(1,2)", "rotate(a)"])
def test_invalid_transform_rejected(t):
    with pytest.raises(SvgReferenceError, match="transform|argument") as exc:
        _one(f'<path transform="{t}" d="M0 0 L5 5"/>')
    assert exc.value.code == "svg_transform_invalid"


def test_sampling_step_is_in_bitmap_pixels_under_scale():
    xy = _xy(_one('<g transform="scale(10)"><path d="M0 1 Q4.5 9 9 1"/></g>').paths[0])
    assert np.max(np.hypot(*np.diff(xy, axis=0).T)) <= 0.51


def test_non_rendered_and_style_ignored():
    body = ('<style>path{stroke:red}</style><defs><path d="M0 0 L9 9"/></defs>'
            '<clipPath><rect x="0" y="0" width="5" height="5"/></clipPath><text x="1" y="1">hi</text>'
            '<path style="stroke:#000" class="a" d="M10 10 L20 10"/>')
    ref = _one(body)
    assert len(ref.paths) == 1
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[10, 10], [20, 10]])


# --- units ------------------------------------------------------------------

@pytest.mark.parametrize("w", ["100px", "100"])
def test_pixel_size_match(w):
    raw = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}"><path d="M10 20 L30 20"/></svg>'
    ref = read_reference_svg(raw.encode(), (100, 100))
    assert ref.alignment == "size-match"
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[10, 20], [30, 20]], atol=1e-3)


@pytest.mark.parametrize("w", ["26.458333mm", "2.6458333cm", "1.0416667in", "75pt", "6.25pc"])
def test_physical_size_alone_is_not_trusted(w):
    """100 px at 96 px/in in each unit: equal only through an assumed dpi, so not a registration."""
    raw = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}"><path d="M10 20 L30 20"/></svg>'
    with pytest.raises(SvgReferenceError) as exc:
        read_reference_svg(raw.encode(), (100, 100))
    assert exc.value.code == "svg_no_registration"


@pytest.mark.parametrize("w", ["26.458333mm", "1.0416667in", "75pt"])
def test_physical_units_scale_geometry(w):
    """Units are converted (96 px/in) when the frame comes from a border rect."""
    raw = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}">'
           f'<rect id="border" x="0" y="0" width="{w}" height="{w}"/><line x1="0" y1="{w}" x2="{w}" y2="0"/></svg>')
    ref = read_reference_svg(raw.encode(), (100, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[0, 100], [100, 0]], atol=1e-3)


def test_mm_canvas_with_pixel_viewbox():
    raw = (b'<svg xmlns="http://www.w3.org/2000/svg" width="80mm" height="80mm" viewBox="0 0 100 100">'
           b'<path d="M10 20 L30 20"/></svg>')
    ref = read_reference_svg(raw, (100, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[10, 20], [30, 20]])


def test_mm_without_registration_rejected():
    raw = b'<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="100mm"><path d="M0 0 L5 5"/></svg>'
    with pytest.raises(SvgReferenceError) as exc:
        read_reference_svg(raw, (100, 100))
    assert exc.value.code == "svg_no_registration"


def test_percent_size_needs_viewbox():
    with pytest.raises(SvgReferenceError, match="percentage") as exc:
        read_reference_svg(b'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">'
                           b'<path d="M0 0 L5 5"/></svg>', (100, 100))
    assert exc.value.code == "svg_units_unsupported"
    ref = read_reference_svg(b'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" '
                             b'viewBox="0 0 100 100"><path d="M10 20 L30 20"/></svg>', (100, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[10, 20], [30, 20]])


def test_unknown_unit_rejected():
    with pytest.raises(SvgReferenceError, match="unit") as exc:
        read_reference_svg(b'<svg xmlns="http://www.w3.org/2000/svg" width="100em" height="100em">'
                           b'<path d="M0 0 L5 5"/></svg>', (100, 100))
    assert exc.value.code == "svg_units_unsupported"


# --- <image> registration ---------------------------------------------------

_XLINK = 'xmlns:xlink="http://www.w3.org/1999/xlink"'
_PNG_STUB = "data:image/png;base64,!!not-decoded!!"


def _lightburn_like(paths: str, image_attrs: str | None = None, extra_images: str = "") -> bytes:
    """mm canvas, offset viewBox, Y-flipped image and paths, like a LightBurn export (bitmap 100x100)."""
    image_attrs = image_attrs or ('transform="matrix(1,0,0,-1,1025,-475)" x="-25" y="-25" '
                                  'width="50" height="50"')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" {_XLINK} width="50mm" height="50mm" '
            f'viewBox="1000 -500 50 50"><image {image_attrs} xlink:href="{_PNG_STUB}"/>{extra_images}'
            f'{paths}</svg>').encode()


def test_image_frame_registration_with_flip_and_mm():
    raw = _lightburn_like('<path transform="matrix(1,0,0,-1,0,0)" d="M1000 475 L1050 475"/>'
                          '<path transform="matrix(1,0,0,-1,0,0)" d="M1010 500 L1010 450"/>')
    ref = read_reference_svg(raw, (100, 100))
    assert ref.alignment == "image-frame"
    assert ref.frame["flip_y"] is True and ref.frame["flip_x"] is False
    assert ref.frame["href"] == "image/png"
    assert ref.svg_size == pytest.approx((50.0, 50.0))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[0, 50], [100, 50]], atol=1e-9)
    np.testing.assert_allclose(_xy(ref.paths[1])[[0, -1]], [[20, 0], [20, 100]], atol=1e-9)
    np.testing.assert_allclose(ref.paths[0][0], [49.5, -0.5])  # pixel-centre (row, col)


def test_image_frame_overrides_wrong_root_size():
    raw = (f'<svg xmlns="http://www.w3.org/2000/svg" {_XLINK} width="999" height="7">'
           f'<image x="10" y="10" width="20" height="10" xlink:href="{_PNG_STUB}"/>'
           '<path d="M10 15 L30 15"/></svg>').encode()
    ref = read_reference_svg(raw, (200, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[0, 50], [200, 50]])


def test_image_href_plain_attribute_and_scale_frame():
    raw = (b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><g transform="scale(0.1)">'
           b'<image width="100" height="100" href="x.png"/></g><path d="M0 5 L10 5"/></svg>')
    ref = read_reference_svg(raw, (100, 100))
    assert ref.frame["href"] == "external"
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[0, 50], [100, 50]])


def test_image_in_defs_is_not_a_frame():
    raw = (f'<svg xmlns="http://www.w3.org/2000/svg" {_XLINK} width="100" height="100"><defs>'
           f'<image width="5" height="5" xlink:href="{_PNG_STUB}"/></defs><path d="M10 20 L30 20"/></svg>').encode()
    assert read_reference_svg(raw, (100, 100)).alignment == "size-match"


def test_rotated_image_rejected():
    raw = _lightburn_like('<path d="M1000 -475 L1050 -475"/>',
                          'transform="rotate(30 1025 -475)" x="1000" y="-500" width="50" height="50"')
    with pytest.raises(SvgReferenceError, match="rotated") as exc:
        read_reference_svg(raw, (100, 100))
    assert exc.value.code == "svg_image_rotated"


def test_image_aspect_mismatch_rejected():
    raw = _lightburn_like('<path d="M1000 -475 L1050 -475"/>', 'x="1000" y="-500" width="50" height="40"')
    with pytest.raises(SvgReferenceError, match="aspect") as exc:
        read_reference_svg(raw, (100, 100))
    assert exc.value.code == "svg_image_aspect"


def test_multiple_images_rejected():
    raw = _lightburn_like('<path d="M1000 -475 L1050 -475"/>',
                          extra_images=f'<image x="0" y="0" width="5" height="5" xlink:href="{_PNG_STUB}"/>')
    with pytest.raises(SvgReferenceError, match="exactly one") as exc:
        read_reference_svg(raw, (100, 100))
    assert exc.value.code == "svg_multiple_images"


def test_embedded_base64_counts_against_size_cap():
    big = "data:image/png;base64," + "A" * 4000
    raw = (f'<svg xmlns="http://www.w3.org/2000/svg" {_XLINK} width="100" height="100">'
           f'<image width="100" height="100" xlink:href="{big}"/><path d="M0 0 L5 5"/></svg>').encode()
    with pytest.raises(SvgReferenceError, match="embedded images count") as exc:
        read_reference_svg(raw, (100, 100), max_bytes=2000)
    assert exc.value.code == "svg_too_large"
    assert read_reference_svg(raw, (100, 100), max_bytes=5000).alignment == "image-frame"


# --- embedded image header check (stored size; pixel data never decoded) -----

def _png_bytes(w, h) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (w, h), 255).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(w, h, pad=0) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (w, h), 255).save(buf, format="JPEG")
    data = buf.getvalue()
    segments = b""
    while pad > 0:  # COM segments before SOF, like large EXIF/ICC blocks
        n = min(pad, 65533)
        segments += b"\xff\xfe" + (n + 2).to_bytes(2, "big") + b"x" * n
        pad -= n
    return data[:2] + segments + data[2:]


def _framed(data: bytes, mime="image/png", frame=(100, 100)) -> bytes:
    href = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" {_XLINK} width="{frame[0]}" height="{frame[1]}">'
            f'<image width="{frame[0]}" height="{frame[1]}" xlink:href="{href}"/>'
            '<path d="M0 5 L10 5"/></svg>').encode()


def test_stored_png_size_matches_bitmap():
    ref = read_reference_svg(_framed(_png_bytes(100, 100)), (100, 100))
    assert ref.frame["stored_format"] == "png" and ref.frame["stored_size"] == [100, 100]
    assert ref.warnings == []


def test_stored_size_same_aspect_different_pixels_warns():
    ref = read_reference_svg(_framed(_png_bytes(256, 256)), (300, 300))
    assert ref.alignment == "image-frame"
    assert any("256x256 px but the bitmap is 300x300 px" in w for w in ref.warnings)


def test_stored_aspect_mismatch_rejected():
    """The frame matches the bitmap's aspect, but the embedded image does not: a different picture."""
    with pytest.raises(SvgReferenceError, match="different image") as exc:
        read_reference_svg(_framed(_png_bytes(200, 100)), (100, 100))
    assert exc.value.code == "svg_bitmap_mismatch"


def test_stored_aspect_within_one_percent_warns_only():
    ref = read_reference_svg(_framed(_png_bytes(1000, 995)), (100, 100))
    assert ref.frame["stored_size"] == [1000, 995] and len(ref.warnings) == 1


def test_stored_size_read_from_header_only():
    """A valid IHDR followed by garbage is enough: pixel data is never decoded."""
    ihdr = (b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + (64).to_bytes(4, "big")
            + (64).to_bytes(4, "big") + b"\x08\x00\x00\x00\x00" + b"\x00" * 4 + b"IDAT-not-a-zlib-stream" * 10)
    ref = read_reference_svg(_framed(ihdr), (128, 128))
    assert ref.frame["stored_size"] == [64, 64]


def test_stored_dimension_cap():
    ihdr = (b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + (100_000).to_bytes(4, "big")
            + (100_000).to_bytes(4, "big") + b"\x08\x00\x00\x00\x00")
    with pytest.raises(SvgReferenceError) as exc:
        read_reference_svg(_framed(ihdr), (100, 100))
    assert exc.value.code == "svg_image_dimensions"


def test_stored_jpeg_size_after_large_segments():
    ref = read_reference_svg(_framed(_jpeg_bytes(64, 32, pad=100_000), "image/jpeg", frame=(128, 64)), (128, 64))
    assert ref.frame["stored_format"] == "jpeg" and ref.frame["stored_size"] == [64, 32]
    with pytest.raises(SvgReferenceError) as exc:
        read_reference_svg(_framed(_jpeg_bytes(64, 64), "image/jpeg", frame=(128, 64)), (128, 64))
    assert exc.value.code == "svg_bitmap_mismatch"


def test_jpeg_header_beyond_window_warns(monkeypatch):
    monkeypatch.setattr(svg_reference, "EMBEDDED_HEADER_BYTES", 1024)
    ref = read_reference_svg(_framed(_jpeg_bytes(64, 64, pad=4000), "image/jpeg"), (100, 100))
    assert ref.frame["stored_size"] is None
    assert any("readable header" in w for w in ref.warnings)


@pytest.mark.parametrize("href, reason", [
    (_PNG_STUB, "malformed"),
    ("x.png", "linked"),
    ("data:image/png,%89PNG", "not base64"),
    ("data:image/gif;base64,R0lGODlhAQABAAAAACw=", "readable header"),
])
def test_unreadable_stored_size_warns(href, reason):
    raw = (f'<svg xmlns="http://www.w3.org/2000/svg" {_XLINK} width="100" height="100">'
           f'<image width="100" height="100" xlink:href="{href}"/><path d="M0 5 L10 5"/></svg>').encode()
    ref = read_reference_svg(raw, (100, 100))
    assert ref.frame["stored_size"] is None
    assert any("Could not check" in w and reason in w for w in ref.warnings)


def test_border_rect_through_transform():
    body = ('<g transform="translate(50,50) scale(2)"><rect id="border" width="100" height="50"/>'
            '<path d="M0 25 L100 25"/></g>')
    ref = read_reference_svg(_svg(body, 400, 300), (100, 50))
    assert ref.alignment == "border-rect"
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[0, 25], [100, 25]])


def test_px_unit_accepted():
    ref = read_reference_svg(b'<svg xmlns="http://www.w3.org/2000/svg" width="100px" height="100px">'
                             b'<path d="M10 20 L30 20"/></svg>', (100, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[10, 20], [30, 20]])


def test_viewbox_scaling():
    ref = read_reference_svg(_svg('<path d="M1 2 L3 2"/>', 100, 100, extra='viewBox="0 0 10 10"'), (100, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[10, 20], [30, 20]])


def test_viewbox_only_and_offset():
    raw = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="5 5 100 100"><path d="M15 25 L35 25"/></svg>'
    ref = read_reference_svg(raw, (100, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[10, 20], [30, 20]])


def test_non_uniform_viewbox_rejected_unless_none():
    with pytest.raises(SvgReferenceError, match="aspect"):
        read_reference_svg(_svg('<path d="M0 0 L1 1"/>', 100, 100, extra='viewBox="0 0 10 20"'), (100, 100))
    ref = read_reference_svg(_svg('<path d="M0 0 L10 20"/>', 100, 100,
                                  extra='viewBox="0 0 10 20" preserveAspectRatio="none"'), (100, 100))
    np.testing.assert_allclose(_xy(ref.paths[0])[-1], [100, 100])


def test_border_rect_alignment():
    body = '<rect id="border" x="50" y="50" width="200" height="100" fill="none"/><path d="M50 100 L250 100"/>'
    ref = read_reference_svg(_svg(body, 400, 300), (100, 50))
    assert ref.alignment == "border-rect"
    assert len(ref.paths) == 1  # border is not a centerline
    np.testing.assert_allclose(_xy(ref.paths[0])[[0, -1]], [[0, 25], [100, 25]])


def test_relative_commands_h_v_and_implicit_lineto():
    ref = _one('<path d="m10 10 20 0 v10 h-20 z"/>')
    xy = _xy(ref.paths[0])
    assert ref.closed == [True]
    for corner in ([10, 10], [30, 10], [30, 20], [10, 20]):
        assert np.min(np.hypot(*(xy - corner).T)) < 1e-9


def test_exponent_numbers_and_compact_separators():
    ref = _one('<path d="M1e1,2e1L3.0e1-1.5e0.5.5"/>')
    xy = _xy(ref.paths[0])
    np.testing.assert_allclose(xy[0], [10, 20])
    assert np.min(np.hypot(*(xy - [30, -1.5]).T)) < 1e-9
    np.testing.assert_allclose(xy[-1], [0.5, 0.5])  # ".5.5" is two numbers (implicit lineto)


def test_cubic_and_smooth_cubic_sampling():
    ref = _one('<path d="M10 50 C10 10 50 10 50 50 S90 90 90 50"/>')
    xy = _xy(ref.paths[0])
    np.testing.assert_allclose(xy[0], [10, 50])
    np.testing.assert_allclose(xy[-1], [90, 50])
    # cubic midpoint at t=0.5: (30, 20); S reflects control -> second midpoint (70, 80)
    assert np.min(np.hypot(*(xy - [30, 20]).T)) < 0.3
    assert np.min(np.hypot(*(xy - [70, 80]).T)) < 0.3
    assert np.max(np.hypot(*np.diff(xy, axis=0).T)) <= 0.51


def test_quadratic_and_smooth_quadratic_sampling():
    ref = _one('<path d="M10 50 Q30 10 50 50 T90 50"/>')
    xy = _xy(ref.paths[0])
    assert np.min(np.hypot(*(xy - [30, 30]).T)) < 0.3   # Q midpoint
    assert np.min(np.hypot(*(xy - [70, 70]).T)) < 0.3   # T reflects control to (70, 90)


@pytest.mark.parametrize("d", ["M10 50 A40 40 0 0 1 90 50", "M10 50 a40 40 0 0180 0", "M10 50 A40,40,0,0,1,90,50"])
def test_arc_sampling_on_circle(d):
    ref = _one(f'<path d="{d}"/>')
    xy = _xy(ref.paths[0])
    np.testing.assert_allclose(xy[-1], [90, 50], atol=1e-9)
    r = np.hypot(xy[:, 0] - 50, xy[:, 1] - 50)
    np.testing.assert_allclose(r, 40, atol=1e-6)
    # sweep=1 from (10,50) to (90,50) goes through the top (y=10) in SVG's y-down frame
    assert xy[:, 1].min() < 10.5


def test_large_arc_flag():
    small = _xy(_one('<path d="M50 10 A40 40 0 0 1 90 50"/>').paths[0])
    large = _xy(_one('<path d="M50 10 A40 40 0 1 1 90 50"/>').paths[0])
    length = lambda p: float(np.hypot(*np.diff(p, axis=0).T).sum())  # noqa: E731
    assert length(small) == pytest.approx(math.pi * 40 / 2, rel=0.01)
    assert length(large) == pytest.approx(3 * math.pi * 40 / 2, rel=0.01)


def test_basic_shapes():
    ref = _one('<line x1="0" y1="0" x2="10" y2="0"/><polyline points="0,5 10,5 10,15"/>'
               '<polygon points="20,20 30,20 30,30"/><circle cx="50" cy="50" r="10"/>'
               '<ellipse cx="80" cy="80" rx="5" ry="3"/><rect x="60" y="0" width="10" height="5"/>')
    assert len(ref.paths) == 6
    assert ref.closed == [False, False, True, True, True, True]
    circ = _xy(ref.paths[3])
    np.testing.assert_allclose(np.hypot(circ[:, 0] - 50, circ[:, 1] - 50), 10, atol=1e-9)


def test_empty_svg_rejected():
    with pytest.raises(SvgReferenceError, match="no centerline geometry"):
        _one("")


def test_malformed_path_rejected():
    with pytest.raises(SvgReferenceError):
        _one('<path d="M10 10 L20"/>')
    with pytest.raises(SvgReferenceError):
        _one('<path d="L10 10"/>')
    with pytest.raises(SvgReferenceError):
        _one('<path d="M10 10 X20 20"/>')
