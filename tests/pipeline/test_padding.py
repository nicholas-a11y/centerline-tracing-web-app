from __future__ import annotations

from tests.pipeline.lesson import LESSONS_DIR
from tests.pipeline.padding import engine_supersample_factor, padding_check

IMAGE = LESSONS_DIR / "cross-t2-w8-a35-640px" / "image.png"
INNER = (280, 280, 360, 360)  # row0, col0, row1, col1 around the crossing at (319.5, 319.5)


def test_supersample_factor_matches_engine_rule():
    assert engine_supersample_factor((64, 64)) == 3
    assert engine_supersample_factor((256, 256)) == 3
    assert engine_supersample_factor((300, 250)) == 2
    assert engine_supersample_factor((640, 100)) == 1


def test_padding_covering_whole_image_is_identical():
    rep = padding_check(IMAGE, INNER, pad_px=1000)
    assert rep.consistent, rep
    assert rep.max_diff_px == 0.0


def test_small_crop_is_reported():
    rep = padding_check(IMAGE, INNER, pad_px=20)
    assert rep.crop_supersample == 3 and rep.full_supersample == 1
    assert "supersample factor differs" in rep.note
    assert rep.max_diff_px is None or rep.max_diff_px >= 0.0
