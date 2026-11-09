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
GENERATED = LESSONS_DIR / "cross-t2-w8-a35"
USER = LESSONS_DIR / "user-lightburn-cross-t2-w8-a35"
CENTRE = 127.5  # (256 - 1) / 2, pixel-centre coordinates


def _lines():
    ref = read_reference_svg(SAMPLE, (256, 256))
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
