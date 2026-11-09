"""One test per error code (design 9, 14.4). The final test checks that every code in
schema.ERROR_CODES is covered here."""

from __future__ import annotations

import io

import pytest

from teach import limits, schema
from teach.validate import Problems, validate_bitmap, validate_reference, validate_submission

from .conftest import IMAGE, META, REFERENCE, SAMPLES, data_uri, multipart, png_bytes, png_header_only, post, submit

NS = 'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
LINE = '<path d="M 0 128 L 256 128"/>'


def svg(body: str = LINE, attrs: str = 'width="256" height="256" viewBox="0 0 256 256"') -> bytes:
    return f"<svg {NS} {attrs}>{body}</svg>".encode()


def image_el(href: str, w=256, h=256, extra="") -> str:
    return f'<image x="0" y="0" width="{w}" height="{h}" xlink:href="{href}" {extra}/>'


FRAME_ONLY = 'width="100" height="100" viewBox="0 0 256 256"'
GOOD_IMAGE = data_uri(png_bytes(256, 256))

SVG_CASES = {
    "svg_too_large": b"<svg>" + b" " * (schema.MAX_SVG_BYTES + 1) + b"</svg>",
    "svg_doctype": b'<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY a "b">]>' + svg(),
    "svg_parse": f"<svg {NS}><path".encode(),
    "svg_use_unsupported": svg(LINE + '<use href="#p"/>'),
    "svg_transform_invalid": svg('<path transform="wobble(2)" d="M 0 128 L 256 128"/>'),
    "svg_units_unsupported": svg(LINE, 'width="10em" height="10em"'),
    "svg_no_registration": svg(LINE, 'width="100" height="100"'),
    "svg_multiple_images": svg(image_el(GOOD_IMAGE) + image_el(GOOD_IMAGE) + LINE, FRAME_ONLY),
    "svg_image_rotated": svg(image_el(GOOD_IMAGE, extra='transform="rotate(30)"') + LINE, FRAME_ONLY),
    "svg_image_aspect": svg(image_el(GOOD_IMAGE, 256, 128) + LINE, FRAME_ONLY),
    "svg_bitmap_mismatch": svg(image_el(data_uri(png_bytes(300, 200))) + LINE, FRAME_ONLY),
    "svg_image_dimensions": svg(image_el(data_uri(png_header_only(20000, 20000))) + LINE, FRAME_ONLY),
    "svg_image_unreadable": svg(image_el("image.png") + LINE, FRAME_ONLY),
    "svg_no_geometry": svg("<text>hello</text>"),
    "svg_too_many_elements": svg("".join(f'<line x1="0" y1="{i}" x2="10" y2="{i}"/>' for i in range(201))),
}


def _apng() -> bytes:
    from PIL import Image

    frames = [Image.new("L", (64, 64), v) for v in (0, 255)]
    buf = io.BytesIO()
    frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:])
    return buf.getvalue()


IMAGE_CASES = {
    "image_too_large": b"\x89PNG\r\n\x1a\n" + b"\0" * schema.MAX_IMAGE_BYTES,
    "image_dimensions": png_bytes(16, 16),
    "image_format": b"GIF89a" + b"\0" * 100,
    "image_multiframe": _apng(),
}

FIELD_CASES = {
    "field_missing": {"title": ""},
    "field_too_long": {"title": "x" * 121},
    "param_out_of_range": {"settings": {"dark_threshold": 0.9}},
    "consent_required": {"consent": False},
    "meta_invalid": {"colour": "blue"},
}

HTTP_CODES = {"duplicate_exact", "rate_limited", "quota_exceeded", "busy", "scoring_timeout", "request_too_large",
              "invalid_state", "not_found", "manage_token_invalid"}


def codes(exc: Problems) -> list[str]:
    return [p.code for p in exc.problems]


@pytest.mark.parametrize("code", sorted(SVG_CASES))
def test_svg_error_codes(code):
    with pytest.raises(Problems) as info:
        validate_reference(SVG_CASES[code], (256, 256))
    assert codes(info.value) == [code]
    problem = info.value.problems[0]
    assert problem.field == "reference" and problem.message


@pytest.mark.parametrize("code", sorted(IMAGE_CASES))
def test_image_error_codes(code):
    with pytest.raises(Problems) as info:
        validate_bitmap(IMAGE_CASES[code])
    assert codes(info.value) == [code]


@pytest.mark.parametrize("code", sorted(FIELD_CASES))
def test_field_error_codes(code):
    meta = dict(META)
    meta.update(FIELD_CASES[code])
    with pytest.raises(Problems) as info:
        validate_submission(IMAGE, REFERENCE, __import__("json").dumps(meta))
    assert codes(info.value) == [code]


def test_validate_reports_every_error_at_once():
    meta = dict(META, title="", consent=False)
    with pytest.raises(Problems) as info:
        validate_submission(IMAGE, SVG_CASES["svg_no_registration"], __import__("json").dumps(meta))
    assert set(codes(info.value)) == {"field_missing", "consent_required", "svg_no_registration"}


def test_lightburn_samples_are_accepted_in_strict_mode():
    from PIL import Image

    for name in ("lightburn-cross-t2-w8-a35.svg", "lightburn-faded-cross-t2-w8-a35.svg"):
        ref = validate_reference((SAMPLES / name).read_bytes(), (256, 256))
        assert ref.registration == "image_frame" and len(ref.paths) == 2
    assert Image.open(io.BytesIO(IMAGE)).size == (256, 256)


def test_bitmap_is_reencoded_as_grey_png_without_metadata():
    from PIL import Image

    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    rgba.putpixel((5, 5), (0, 0, 0, 255))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG", pnginfo=__import__("PIL.PngImagePlugin", fromlist=["PngInfo"]).PngInfo())
    bm = validate_bitmap(buf.getvalue())
    im = Image.open(io.BytesIO(bm.png))
    assert im.mode == "L" and not im.info.get("dpi") and bm.gray[0, 0] == 255 and bm.gray[5, 5] == 0


def test_warning_codes():
    assert [p.code for p in validate_bitmap(png_bytes(64, 64, 255)).warnings] == ["image_no_dark"]
    assert [p.code for p in validate_bitmap(png_bytes(64, 64, 0)).warnings] == ["image_mostly_dark"]
    assert "image_large" in [p.code for p in validate_bitmap(png_bytes(1100, 40, 255)).warnings]


# ---------------------------------------------------------------- HTTP-level codes


def _first_code(resp) -> str:
    return resp.get_json()["errors"][0]["code"]


def test_validate_response_shape(teach):
    r = post(teach.client, "/teach/api/validate")
    body = r.get_json()
    assert r.status_code == 200 and set(body) == {"ok", "errors", "warnings", "result"}
    assert body["ok"] is True and body["result"]["outcome"] == "fail"
    assert body["result"]["overlay_png"].startswith("data:image/png;base64,")
    assert "trace_template_url" not in body["result"]
    bad = post(teach.client, "/teach/api/validate", title="")
    assert bad.status_code == 200 and bad.get_json()["ok"] is False and _first_code(bad) == "field_missing"


def test_duplicate_exact(teach):
    submit(teach)
    r = post(teach.client, "/teach/api/submissions")
    assert r.status_code == 409 and _first_code(r) == "duplicate_exact"


def test_rate_limited(teach, monkeypatch):
    monkeypatch.setattr(limits, "VALIDATE_PER_IP", (2, limits.HOUR))
    for _ in range(2):
        assert post(teach.client, "/teach/api/validate").status_code == 200
    r = post(teach.client, "/teach/api/validate")
    assert r.status_code == 429 and _first_code(r) == "rate_limited" and int(r.headers["Retry-After"]) > 0


def test_quota_exceeded(teach, monkeypatch):
    monkeypatch.setattr(limits, "SUBMIT_PER_IP_DAY", 1)
    submit(teach)
    r = post(teach.client, "/teach/api/submissions", title="Another one")
    assert r.status_code == 429 and _first_code(r) == "quota_exceeded"


def test_busy(teach):
    teach.queue.busy = "ip_busy"
    r = post(teach.client, "/teach/api/validate")
    assert r.status_code == 503 and _first_code(r) == "busy" and r.headers["Retry-After"] == "30"


def test_scoring_timeout_is_a_warning(teach):
    teach.queue.outcome = "timeout"
    r = post(teach.client, "/teach/api/validate")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] is True
    assert [w["code"] for w in body["warnings"]] == ["scoring_timeout"] and body["result"]["outcome"] == "timeout"
    r = post(teach.client, "/teach/api/submissions")
    assert r.status_code == 201 and "scoring_timeout" in [w["code"] for w in r.get_json()["warnings"]]


def test_request_too_large(teach):
    big = b"\x89PNG\r\n\x1a\n" + b"\0" * (schema.MAX_BODY_BYTES + 10)
    r = post(teach.client, "/teach/api/validate", image=big)
    assert r.status_code == 413 and _first_code(r) == "request_too_large"


def test_invalid_state(teach):
    _body, sub_id, token = submit(teach)
    r = teach.client.patch(f"/teach/api/manage/{sub_id}", json={"note": "more info"},
                           headers={"X-Teach-Manage-Token": token})
    assert r.status_code == 409 and _first_code(r) == "invalid_state"


def test_not_found(teach):
    r = teach.client.get("/teach/api/submissions/99")
    assert r.status_code == 404 and _first_code(r) == "not_found"
    r = teach.client.post("/teach/api/submissions/99/me-too", data={"meta": '{"note": "same"}'})
    assert r.status_code == 404 and _first_code(r) == "not_found"


def test_manage_token_invalid(teach):
    _body, sub_id, _token = submit(teach)
    r = teach.client.get(f"/teach/api/manage/{sub_id}", headers={"X-Teach-Manage-Token": "wrong"})
    assert r.status_code == 404 and _first_code(r) == "manage_token_invalid"


def test_every_error_code_is_tested():
    listed = {c for group in schema.ERROR_CODES.values() for c in group}
    tested = set(SVG_CASES) | set(IMAGE_CASES) | set(FIELD_CASES) | HTTP_CODES
    assert listed == tested


def test_multipart_helper_roundtrip():
    data = multipart(title="x")
    assert '"title": "x"' in data["meta"]
