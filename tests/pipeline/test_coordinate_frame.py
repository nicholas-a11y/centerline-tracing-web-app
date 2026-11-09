"""Pin down the coordinate frame of the live pipeline before any scoring relies on it.

Findings encoded here:
  * /upload does not resize: image_shape equals the bitmap's (height, width).
  * Float paths (session.initial_paths) are (row, col) in original-image pixel
    coordinates with integer = pixel centre, for both supersampled (<600 px)
    and non-supersampled images.
  * The /process_immediate JSON response truncates with int(), i.e. floor()
    for these non-negative coordinates (mean bias about -0.5 px per axis).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from tests.pipeline.runner import run_live
from tests.pipeline.svg_reference import read_reference_svg, write_reference_svg


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _band_image(h, w, rows=None, cols=None):
    img = np.full((h, w), 255, dtype=np.uint8)
    if rows is not None:
        img[rows[0]:rows[1], 10:w - 10] = 0
    if cols is not None:
        img[10:h - 10, cols[0]:cols[1]] = 0
    return img


@pytest.mark.parametrize("h,w", [(64, 64), (120, 700)], ids=["supersampled-64px", "native-700px"])
def test_horizontal_band_centre_row(h, w):
    # 3 px band on rows 30..32 -> centre row 31.0 in pixel-centre convention
    res = run_live(_png(_band_image(h, w, rows=(30, 33))))
    assert res.error is None, res.error
    assert tuple(res.image_shape) == (h, w)
    pts = np.vstack(res.paths)
    interior = pts[(pts[:, 1] > 15) & (pts[:, 1] < w - 15)]
    assert len(interior)
    assert abs(float(np.median(interior[:, 0])) - 31.0) <= 0.25
    assert np.all(np.abs(interior[:, 0] - 31.0) <= 0.5)


@pytest.mark.parametrize("h,w,tol", [(64, 64, 0.25), (700, 120, 0.5)], ids=["supersampled-64px", "native-700px"])
def test_vertical_band_centre_col(h, w, tol):
    # 4 px band on cols 20..23 -> true centre col 21.5. Without supersampling (>= 600 px)
    # the skeleton sits on whole pixels, so an even-width stroke is off by 0.5 px (finding).
    res = run_live(_png(_band_image(h, w, cols=(20, 24))))
    assert res.error is None, res.error
    pts = np.vstack(res.paths)
    interior = pts[(pts[:, 0] > 15) & (pts[:, 0] < h - 15)]
    assert abs(float(np.median(interior[:, 1])) - 21.5) <= tol


def test_response_paths_are_floor_of_float_paths():
    res = run_live(_png(_band_image(64, 64, rows=(30, 34))))  # centre row 31.5
    assert res.error is None, res.error
    assert len(res.paths) == len(res.response_paths)
    for f, r in zip(res.paths, res.response_paths):
        np.testing.assert_array_equal(np.floor(f), r)
    pts = np.vstack(res.paths)
    resp = np.vstack(res.response_paths)
    assert float(np.mean(resp[:, 0] - pts[:, 0])) < -0.25  # truncation bias toward smaller row/col


def test_svg_reference_round_trip(tmp_path):
    paths = [np.array([[31.0, 5.0], [31.0, 58.0]]), np.array([[3.25, 7.75], [40.5, 12.0], [60.0, 60.0]])]
    svg = tmp_path / "ref.svg"
    write_reference_svg(svg, paths, (64, 64))
    back = read_reference_svg(svg, (64, 64))
    assert back.alignment == "size-match"
    for a, b in zip(paths, back.paths):
        np.testing.assert_allclose(b[0], a[0], atol=1e-3)
        np.testing.assert_allclose(b[-1], a[-1], atol=1e-3)
    # SVG x = col + 0.5, y = row + 0.5 (same as create_svg_output)
    assert '5.500 31.500' in svg.read_text()
