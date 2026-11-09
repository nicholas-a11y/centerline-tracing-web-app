"""Main app's Teach checkbox (Stage 5c item 4): status route, read-only cropped-image download,
and the engine-derived crop equivalence verdict. Runs against a fresh copy of the real
centerline_web_app module with Teach registered before its first request."""

from __future__ import annotations

import contextlib
import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from teach import crop, schema
from teach.validate import effective_params

from .conftest import REPO

LESSON_IMAGE = REPO / "tests" / "lessons" / "cross-t2-w8-a35" / "image.png"
LARGE_IMAGE = REPO / "test_images" / "centerline_gap_bridge_test.png"
REGION = {"top": 0.1, "bottom": 0.83, "left": 0.2, "right": 0.9}


def _params() -> dict:
    p = effective_params(schema.default_settings())
    p["enable_optimization"] = False
    return p


def _upload(client, data: bytes, name: str = "image.png") -> str:
    p = _params()
    with contextlib.redirect_stdout(io.StringIO()):
        r = client.post("/upload", data={"file": (io.BytesIO(data), name),
                                         "normalization_mode": p["normalization_mode"],
                                         "normalization_sensitivity": p["normalization_sensitivity"]},
                        content_type="multipart/form-data")
    body = r.get_json()
    assert "session_id" in body, body
    return body["session_id"]


def _process(mod, client, sid: str, region: dict | None):
    body = {"session_id": sid, "parameters": _params()}
    if region:
        body["crop_region"] = region
    with contextlib.redirect_stdout(io.StringIO()):
        r = client.post("/process_immediate", json=body)
    out = r.get_json()
    assert out.get("success"), out.get("error")
    return [np.asarray(p, dtype=float) for p in dict.get(mod.sessions, sid).initial_paths], out


def _query(region: dict | None) -> str:
    if not region:
        return ""
    return "?" + "&".join(f"{k}={region[k]!r}" for k in ("top", "bottom", "left", "right"))


def _crop_info(client, sid, region):
    r = client.get(f"/teach/api/crop/{sid}{_query(region)}")
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _crop_png(client, sid, region) -> bytes:
    r = client.get(f"/teach/api/crop/{sid}/image.png{_query(region)}")
    assert r.status_code == 200 and r.mimetype == "image/png"
    return r.data


def _engine_box(shape, region, with_applied=False):
    # Independent copy of /process_immediate's rule (centerline_web_app.py, crop_region block).
    h, w = shape[:2]
    y1 = max(0, int(round(float(region.get("top", 0)) * h)))
    y2 = min(h, int(round(float(region.get("bottom", 1)) * h)))
    x1 = max(0, int(round(float(region.get("left", 0)) * w)))
    x2 = min(w, int(round(float(region.get("right", 1)) * w)))
    if y2 > y1 + 10 and x2 > x1 + 10:
        return (y1, y2, x1, x2, True) if with_applied else (y1, y2, x1, x2)
    return (0, h, 0, w, False) if with_applied else (0, h, 0, w)


def _same_paths(a, b) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


# ------------------------------------------------------------------ status and isolation


def test_status_on(main_on):
    r = main_on.app.test_client().get("/teach/api/status")
    assert r.status_code == 200 and r.get_json() == {"teach": "enabled"}
    assert r.headers["Cache-Control"] == "no-store"


def test_status_and_crop_404_when_off(main_off):
    c = main_off.app.test_client()
    assert c.get("/teach/api/status").status_code == 404
    assert c.get("/teach/api/crop/00000000-0000-0000-0000-000000000000").status_code == 404


# ------------------------------------------------------------------ rounding rule


@pytest.mark.parametrize("region", [
    REGION,
    {"top": 0.0, "bottom": 1.0, "left": 0.0, "right": 1.0},
    {"top": 0.123456789, "bottom": 0.5, "left": 0.0019, "right": 0.9981},
    {"top": 0.5, "bottom": 0.52, "left": 0.1, "right": 0.9},   # too thin: engine ignores the crop
    {"top": 0.3, "bottom": 0.2, "left": 0.1, "right": 0.9},    # inverted: ignored
])
@pytest.mark.parametrize("shape", [(256, 256), (1700, 2200), (37, 513)])
def test_crop_box_matches_process_immediate(region, shape):
    assert crop.crop_box(shape, region) == _engine_box(shape, region, with_applied=True)


@pytest.mark.parametrize("args", [
    {"top": "0.1"},
    {"top": "0.1", "bottom": "0.9", "left": "0", "right": "nan"},
    {"top": "0.1", "bottom": "0.9", "left": "0", "right": "inf"},
    {"top": "-0.1", "bottom": "0.9", "left": "0", "right": "1"},
    {"top": "0.1", "bottom": "1.5", "left": "0", "right": "1"},
    {"top": "x", "bottom": "0.9", "left": "0", "right": "1"},
    {"top": "0." + "1" * 40, "bottom": "0.9", "left": "0", "right": "1"},
])
def test_parse_region_rejects(args):
    with pytest.raises(ValueError):
        crop.parse_region(args)


def test_parse_region_absent_is_none():
    assert crop.parse_region({}) is None


def test_bad_requests(main_on):
    c = main_on.app.test_client()
    assert c.get("/teach/api/crop/not-a-session").status_code == 404
    assert c.get("/teach/api/crop/00000000-0000-0000-0000-000000000000").get_json()["errors"][0]["code"] == "not_found"
    sid = _upload(c, LESSON_IMAGE.read_bytes())
    try:
        r = c.get(f"/teach/api/crop/{sid}?top=0.1")
        assert r.status_code == 400 and r.get_json()["errors"][0]["code"] == "invalid_crop_region"
    finally:
        dict.pop(main_on.sessions, sid, None)


# ------------------------------------------------------------------ download = engine region


def test_download_pixels_equal_engine_region(main_on):
    c = main_on.app.test_client()
    sid = _upload(c, LESSON_IMAGE.read_bytes())
    try:
        paths, _body = _process(main_on, c, sid, REGION)
        session = dict.get(main_on.sessions, sid)
        y1, y2, x1, x2 = _engine_box(session.image.shape, REGION)
        pts = np.concatenate(paths)
        assert pts[:, 0].min() >= y1 and pts[:, 0].max() < y2 and pts[:, 1].min() >= x1 and pts[:, 1].max() < x2
        r = c.get(f"/teach/api/crop/{sid}/image.png{_query(REGION)}")
        assert r.headers["Content-Disposition"] == f'attachment; filename="cropped-{x2 - x1}x{y2 - y1}.png"'
        img = Image.open(io.BytesIO(r.data))
        assert img.mode == "L" and img.size == (x2 - x1, y2 - y1)
        got = np.asarray(img)
        engine_region = session.image[y1:y2, x1:x2]
        assert np.array_equal(got, np.round(engine_region * 255.0).astype(np.uint8))
        # ... and those are exactly the source file's pixels in that box.
        source = np.asarray(Image.open(LESSON_IMAGE).convert("L"))
        assert np.array_equal(got, source[y1:y2, x1:x2])
        info = _crop_info(c, sid, REGION)
        assert info["image_sha256"] == hashlib.sha256(r.data).hexdigest()
        assert info["box"] == {"top": y1, "bottom": y2, "left": x1, "right": x2} and info["crop_applied"]
    finally:
        dict.pop(main_on.sessions, sid, None)


def test_no_crop_downloads_the_processed_bitmap_unchanged(main_on):
    c = main_on.app.test_client()
    sid = _upload(c, LESSON_IMAGE.read_bytes())
    try:
        for region in (None, {"top": 0.5, "bottom": 0.52, "left": 0.0, "right": 1.0}):
            got = np.asarray(Image.open(io.BytesIO(_crop_png(c, sid, region))))
            assert np.array_equal(got, np.asarray(Image.open(LESSON_IMAGE).convert("L")))
            info = _crop_info(c, sid, region)
            assert info["crop_applied"] is False and info["equivalent"] is True
    finally:
        dict.pop(main_on.sessions, sid, None)


def test_crop_routes_are_read_only(main_on):
    c = main_on.app.test_client()
    sid = _upload(c, LESSON_IMAGE.read_bytes())
    try:
        session = dict.get(main_on.sessions, sid)
        session.last_accessed = 12345.0
        before_ids = set(dict.keys(main_on.sessions))
        before_params = dict(session.parameters)
        _crop_info(c, sid, REGION)
        _crop_png(c, sid, REGION)
        assert set(dict.keys(main_on.sessions)) == before_ids
        assert session.last_accessed == 12345.0
        assert session.parameters == before_params and session.initial_paths is None
    finally:
        dict.pop(main_on.sessions, sid, None)


# ------------------------------------------------------------------ equivalence, both branches


def test_crop_equivalence_small_image(main_on):
    """The cropped PNG run on its own gives exactly the cropped run's paths (after the offset)."""
    c = main_on.app.test_client()
    sid = _upload(c, LESSON_IMAGE.read_bytes())
    sid2 = None
    try:
        crop_paths, _ = _process(main_on, c, sid, REGION)
        info = _crop_info(c, sid, REGION)
        assert info["equivalent"] is True and info["reasons"] == [] and info["note"] is None
        png = _crop_png(c, sid, REGION)
        sid2 = _upload(c, png)
        alone, _ = _process(main_on, c, sid2, None)
        y1, x1 = info["box"]["top"], info["box"]["left"]
        shifted = [p + np.array([y1, x1], dtype=float) for p in alone]
        assert crop_paths and _same_paths(shifted, crop_paths)
    finally:
        dict.pop(main_on.sessions, sid, None)
        if sid2:
            dict.pop(main_on.sessions, sid2, None)


@pytest.mark.skipif(not LARGE_IMAGE.exists(), reason="large test image not present")
def test_crop_not_equivalent_on_large_image(main_on):
    """Long side 2200 px > 1600: the full image's settings are scaled, the crop's are not."""
    c = main_on.app.test_client()
    sid = _upload(c, LARGE_IMAGE.read_bytes(), LARGE_IMAGE.name)
    sid2 = None
    region = {"top": 0.3, "bottom": 0.6, "left": 0.3, "right": 0.6}
    try:
        info = _crop_info(c, sid, region)
        assert info["equivalent"] is False and info["reasons"] == [crop.REASON_SCALE]
        assert info["note"] == crop.NOTE_LARGE
        assert info["parameter_scale"]["full_image"] > 1.0 == info["parameter_scale"]["crop"]
        # The download is still offered, at the engine's crop size.
        png = _crop_png(c, sid, region)
        assert Image.open(io.BytesIO(png)).size == (info["width"], info["height"])
        # And the guard is needed: the crop on its own really traces differently.
        crop_paths, _ = _process(main_on, c, sid, region)
        sid2 = _upload(c, png)
        alone, _ = _process(main_on, c, sid2, None)
        shifted = [p + np.array([info["box"]["top"], info["box"]["left"]], dtype=float) for p in alone]
        assert not _same_paths(shifted, crop_paths)
    finally:
        dict.pop(main_on.sessions, sid, None)
        if sid2:
            dict.pop(main_on.sessions, sid2, None)


def test_note_text_is_the_approved_wording():
    assert crop.NOTE_LARGE == (
        "This image is large, so the app sizes its settings from the whole picture and the cropped piece may "
        "trace differently on its own. To teach with it: download the cropped image, upload it to the app as a "
        "new image, check that the problem still shows, then tick Teach again.")


def test_crop_not_equivalent_when_pixels_change(main_on):
    """A colour source is converted to grey with non-8-bit values, so the 8-bit download reloads differently."""
    rgb = np.full((96, 96, 3), 255, dtype=np.uint8)
    rgb[40:46, :, :] = (200, 30, 30)
    rgb[:, 50:53, :] = (20, 90, 160)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    c = main_on.app.test_client()
    sid = _upload(c, buf.getvalue())
    region = {"top": 0.2, "bottom": 0.8, "left": 0.2, "right": 0.8}
    try:
        info = _crop_info(c, sid, region)
        assert info["equivalent"] is False and crop.REASON_PIXELS in info["reasons"]
        assert info["note"] == crop.NOTE_PIXELS
    finally:
        dict.pop(main_on.sessions, sid, None)


def test_crop_rate_limited(main_on, monkeypatch):
    from teach import limits

    monkeypatch.setattr(limits, "CROP_PER_IP", (2, limits.HOUR))
    c = main_on.app.test_client()
    url = "/teach/api/crop/00000000-0000-0000-0000-00000000abcd"
    codes = [c.get(url, environ_base={"REMOTE_ADDR": "10.9.9.9"}).status_code for _ in range(3)]
    assert codes == [404, 404, 429]
