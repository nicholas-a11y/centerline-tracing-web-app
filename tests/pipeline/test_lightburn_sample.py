"""The owner's LightBurn export over lesson cross-t2-w8-a35 (Stage 2b).

The SVG uses mm units, an offset viewBox, a Y-flipped <image> placement and
Y-flipped paths. Registration comes from the <image> frame only.
"""

from __future__ import annotations

import base64
import io
import math
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.pipeline.evaluate import PRIMARY_VARIANT, evaluate
from tests.pipeline.lesson import LESSONS_DIR, load_lesson
from tests.pipeline.svg_reference import read_reference_svg

SAMPLE = Path(__file__).parent / "samples" / "lightburn-cross-t2-w8-a35.svg"
FADED_SAMPLE = Path(__file__).parent / "samples" / "lightburn-faded-cross-t2-w8-a35.svg"
GENERATED = LESSONS_DIR / "cross-t2-w8-a35"
USER = LESSONS_DIR / "user-lightburn-cross-t2-w8-a35"
CENTRE = 127.5  # (256 - 1) / 2, pixel-centre coordinates


def _lines(sample=SAMPLE):
    ref = read_reference_svg(sample, (256, 256))
    flat = [p for p in ref.paths if np.ptp(p[:, 0]) < 1.0]
    sloped = [p for p in ref.paths if np.ptp(p[:, 0]) >= 1.0]
    assert len(flat) == 1 and len(sloped) == 1
    return ref, flat[0], sloped[0]


def test_registered_by_image_frame():
    ref, _, _ = _lines()
    assert ref.alignment == "image-frame"
    assert ref.frame["flip_y"] is True and ref.frame["flip_x"] is False
    assert ref.frame["href"] == "image/png"
    assert ref.frame["stored_format"] == "png" and ref.frame["stored_size"] == [256, 256]
    assert ref.warnings == []


@pytest.mark.parametrize("sample", [SAMPLE, FADED_SAMPLE], ids=["original", "faded"])
def test_samples_pass_strict_image_mode(sample):
    """Teach reads SVGs with strict_image=True; both real LightBurn exports must pass it unchanged."""
    strict = read_reference_svg(sample, (256, 256), strict_image=True)
    loose = read_reference_svg(sample, (256, 256))
    assert strict.warnings == [] and strict.frame == loose.frame
    assert all(np.array_equal(a, b) for a, b in zip(strict.paths, loose.paths))


def test_wrong_bitmap_same_aspect_warns():
    """The frame alone fits any square bitmap; the PNG header (256x256) flags a 300x300 bitmap."""
    ref = read_reference_svg(SAMPLE, (300, 300))
    assert ref.alignment == "image-frame"
    assert len(ref.warnings) == 1 and "256x256 px but the bitmap is 300x300 px" in ref.warnings[0]


def test_thick_line_row_matches_hand_check():
    _, thick, _ = _lines()
    assert thick[:, 0].mean() == pytest.approx(CENTRE, abs=0.1)
    assert np.ptp(thick[:, 0]) < 1e-6
    assert thick[:, 1].min() == pytest.approx(-0.5, abs=0.1)   # spans the full bitmap width
    assert thick[:, 1].max() == pytest.approx(255.5, abs=0.1)


def test_thin_line_angle_and_position_match_hand_check():
    _, _, thin = _lines()
    d = thin[-1] - thin[0]
    angle = math.degrees(math.atan2(-d[0], d[1]))   # rows grow downwards
    assert angle == pytest.approx(35.0, abs=0.1)
    a = math.radians(35.0)
    rel = thin - CENTRE
    lateral = rel[:, 0] * math.cos(a) + rel[:, 1] * math.sin(a)   # distance to the true thin centreline
    assert np.abs(lateral).max() <= 0.1


def test_embedded_png_is_stored_vertically_flipped():
    """Test-only decode (the reader never decodes it): the stored PNG is image.png flipped top-to-bottom."""
    text = SAMPLE.read_text(encoding="utf-8")
    b64 = re.search(r'href="data:image/png;base64,([^"]+)"', text).group(1)
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        embedded = np.asarray(im.convert("L"), dtype=int)
    with Image.open(GENERATED / "image.png") as im:
        lesson = np.asarray(im.convert("L"), dtype=int)
    assert embedded.shape == lesson.shape
    assert not np.array_equal(embedded, lesson)
    assert np.array_equal(embedded, np.flipud(lesson))


def test_user_lesson_matches_generated_lesson_baseline():
    gen, user = load_lesson(GENERATED), load_lesson(USER)
    assert user.status == gen.status == "baseline"
    assert user.params == gen.params and user.scoring == gen.scoring
    ev_gen, ev_user = evaluate(gen), evaluate(user)
    assert ev_user.outcome == ev_gen.outcome
    for name in ev_gen.variants:
        g, u = ev_gen.variants[name], ev_user.variants[name]
        assert u.graph == g.graph, name
        assert u.score.passed == g.score.passed, name
        gs = sorted((ln.passed, ln.fragments, round(ln.best_single_coverage, 2)) for ln in g.score.lines)
        us = sorted((ln.passed, ln.fragments, round(ln.best_single_coverage, 2)) for ln in u.score.lines)
        assert us == gs, name
    gp = ev_gen.variants[PRIMARY_VARIANT].score.summary()["max_p95_px"]
    up = ev_user.variants[PRIMARY_VARIANT].score.summary()["max_p95_px"]
    assert up == pytest.approx(gp, abs=0.1)


# --- second export: traced over a faded copy of the image, placed elsewhere on the canvas ---

def _distance_to_line(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (b - a) / np.linalg.norm(b - a)
    rel = pts - a
    return np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])


def test_faded_sample_registered_by_moved_image_frame():
    ref, _, _ = _lines(FADED_SAMPLE)
    assert ref.alignment == "image-frame"
    assert ref.frame["flip_y"] is True and ref.frame["flip_x"] is False
    f = ref.frame
    assert (f["x0"], f["x1"], f["y0"], f["y1"]) == pytest.approx((216.1376, 283.8624, -283.8624, -216.1376), abs=1e-3)
    assert ref.frame["stored_size"] == [256, 256]
    assert ref.warnings == []


def test_faded_sample_matches_owner_inspect_run():
    _, thick, thin = _lines(FADED_SAMPLE)
    assert thick[:, 0].mean() == pytest.approx(127.50, abs=0.1)
    ends = thin[[0, -1]] if thin[0, 1] < thin[-1, 1] else thin[[-1, 0]]
    np.testing.assert_allclose(ends, [[217.14, -0.5], [37.93, 255.5]], atol=0.1)  # edge to edge


def test_faded_sample_recovers_same_lines_as_first_sample():
    """Different trace, different canvas position: same two lines within 0.1 px."""
    _, thick1, thin1 = _lines(SAMPLE)
    _, thick2, thin2 = _lines(FADED_SAMPLE)
    for first, second in ((thick1, thick2), (thin1, thin2)):
        assert _distance_to_line(second, first[0], first[-1]).max() <= 0.1
        assert _distance_to_line(first, second[0], second[-1]).max() <= 0.1
        assert np.sort(second[:, 1])[[0, -1]] == pytest.approx(np.sort(first[:, 1])[[0, -1]], abs=0.1)


def test_faded_sample_embeds_a_lighter_flipped_copy():
    """Test-only decode: the embedded PNG is a lighter copy of image.png, stored vertically flipped."""
    text = FADED_SAMPLE.read_text(encoding="utf-8")
    b64 = re.search(r'href="data:image/png;base64,([^"]+)"', text).group(1)
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        embedded = np.flipud(np.asarray(im.convert("L"), dtype=int))
    with Image.open(GENERATED / "image.png") as im:
        lesson = np.asarray(im.convert("L"), dtype=int)
    assert embedded.shape == lesson.shape
    assert embedded.min() > lesson.min()                         # faded towards white
    assert np.all(embedded >= lesson)
    ink = lesson < 128
    assert embedded[ink].mean() < embedded[~ink].mean() - 50    # same strokes, same place
