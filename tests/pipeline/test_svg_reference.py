from __future__ import annotations

import math

import numpy as np
import pytest

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


def test_rejects_transform():
    with pytest.raises(SvgReferenceError, match="transform"):
        _one('<g transform="translate(1,1)"><path d="M0 0 L5 5"/></g>')


def test_rejects_use():
    with pytest.raises(SvgReferenceError, match="use"):
        _one('<path id="a" d="M0 0 L5 5"/><use href="#a"/>')


@pytest.mark.parametrize("unit", ["mm", "in", "cm", "pt", "%"])
def test_rejects_physical_units(unit):
    with pytest.raises(SvgReferenceError, match="unit"):
        read_reference_svg(f'<svg xmlns="http://www.w3.org/2000/svg" width="100{unit}" height="100{unit}">'
                           '<path d="M0 0 L5 5"/></svg>'.encode(), (100, 100))


def test_rejects_size_mismatch_with_fix_hint():
    with pytest.raises(SvgReferenceError, match="does not match the bitmap size"):
        read_reference_svg(_svg('<path d="M0 0 L5 5"/>', 50, 50), (100, 100))


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
