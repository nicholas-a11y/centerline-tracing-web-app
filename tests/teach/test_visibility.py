"""Public visibility (design 11): sanitised title public at once; everything else private until review."""

from __future__ import annotations

import json

import pytest

from teach.sanitize import clean_text, sanitize_title, strip_links
from teach.service import TeachService
from teach.similarity import hamming, jaccard, phash, tokens

from .conftest import IMAGE, post, submit


@pytest.mark.parametrize("raw,expected", [
    ("Crossing bug", "Crossing bug"),
    ("See https://evil.example/x now", "See [link removed] now"),
    ("visit www.example.com", "visit [link removed]"),
    ("mail me@example.com", "mail [link removed]"),
    ("go to evil.com today", "go to [link removed] today"),
    ("drawing.png and part.dxf", "drawing.png and part.dxf"),
    ("bad\u202echars\u0000\u200b here", "badchars here"),
    ("tabs\tand\nnewlines", "tabs and newlines"),
    ("", "Untitled"),
    ("   ", "Untitled"),
    ("\u0007", "Untitled"),
])
def test_sanitize_title(raw, expected):
    assert sanitize_title(raw) == expected


def test_title_max_80():
    out = sanitize_title("x" * 200)
    assert len(out) <= 80 and out.endswith("…")


def test_clean_text_keeps_newlines():
    assert clean_text("a\r\nb\u0000c\u202e") == "a\nbc"
    assert strip_links("javascript:alert(1)") == "[link removed]"


def test_unreviewed_item_is_title_only(teach):
    body, sub_id, _ = submit(teach, title="Kink at https://x.example crossing",
                             what_is_wrong="SECRET-DESCRIPTION", notes="SECRET-NOTES", public_thumbnail=True)
    n = body["number"]
    item = teach.client.get(f"/teach/api/submissions/{n}").get_json()
    assert item["title"] == "Kink at [link removed] crossing" and item["status"] == "pending_review"
    assert item["details"] == "awaiting review" and "thumbnail_url" not in item
    for url in ("/teach/api/queue", f"/teach/api/submissions/{n}", "/teach/queue", f"/teach/q/{n}"):
        text = teach.client.get(url).get_data(as_text=True)
        assert "SECRET" not in text and "https://x.example" not in text, url
    assert teach.client.get(f"/teach/q/{n}/thumb.png").status_code == 404
    public = json.loads((teach.storage.root / "teach/index/public.json").read_text())
    assert "SECRET" not in json.dumps(public)


def _review(t, sub_id, **changes):
    meta = t.svc.load_meta(sub_id)
    meta.update(changes)
    t.svc._save(meta)


def test_reviewed_item_shows_details_and_opted_in_thumbnail(teach):
    body, sub_id, _ = submit(teach, what_is_wrong="Visible after review", public_thumbnail=True)
    _review(teach, sub_id, reviewed=True, status="accepted")
    n = body["number"]
    item = teach.client.get(f"/teach/api/submissions/{n}").get_json()
    assert item["what_is_wrong"] == "Visible after review" and item["thumbnail_url"] == f"/teach/q/{n}/thumb.png"
    r = teach.client.get(f"/teach/q/{n}/thumb.png")
    assert r.status_code == 200 and r.mimetype == "image/png"


def test_reviewed_without_opt_in_has_no_thumbnail(teach):
    body, sub_id, _ = submit(teach)
    _review(teach, sub_id, reviewed=True)
    assert "thumbnail_url" not in teach.client.get(f"/teach/api/submissions/{body['number']}").get_json()
    assert teach.client.get(f"/teach/q/{body['number']}/thumb.png").status_code == 404


def test_hidden_title(teach):
    body, sub_id, _ = submit(teach, title="Hide me please")
    _review(teach, sub_id, title_hidden=True)
    item = teach.client.get(f"/teach/api/submissions/{body['number']}").get_json()
    assert item["title"] is None and item["title_hidden"] is True
    assert "Hide me please" not in teach.client.get("/teach/queue").get_data(as_text=True)
    assert "(title hidden)" in teach.client.get("/teach/queue").get_data(as_text=True)


def test_titles_are_escaped(teach):
    submit(teach, title="<script>alert(1)</script>")
    html = teach.client.get("/teach/queue").get_data(as_text=True)
    assert "<script>alert" not in html and "&lt;script&gt;" in html


def test_queue_search(teach):
    submit(teach, title="Crossing at 35 degrees")
    html = teach.client.get("/teach/queue?q=crossing").get_data(as_text=True)
    assert "Crossing at 35 degrees" in html
    assert "Crossing at 35 degrees" not in teach.client.get("/teach/queue?q=nomatch").get_data(as_text=True)


def test_deleted_items_leave_a_tombstone(teach):
    body, sub_id, token = submit(teach)
    teach.client.delete(f"/teach/api/manage/{sub_id}", headers={"X-Teach-Manage-Token": token})
    item = teach.client.get(f"/teach/api/submissions/{body['number']}").get_json()
    assert item == {"number": body["number"], "status": "deleted"}
    assert teach.client.get("/teach/api/queue").get_json()["items"] == []


def test_similar_exposes_public_fields_only(teach):
    submit(teach, what_is_wrong="PRIVATE-WRONG")
    r = post(teach.client, "/teach/api/similar")
    body = r.get_json()
    assert r.status_code == 200 and body["result"]["exact_duplicate"] is True
    m = body["result"]["matches"][0]
    assert set(m) == {"number", "url", "status", "title", "title_hidden", "details", "exact", "signals"}
    assert "PRIVATE" not in json.dumps(body)


def test_similar_non_exact_does_not_block(teach):
    import io

    from PIL import Image

    first, _, _ = submit(teach)
    img = Image.open(io.BytesIO(IMAGE)).convert("L")
    img.putpixel((0, 0), 0 if img.getpixel((0, 0)) > 127 else 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = post(teach.client, "/teach/api/submissions", image=buf.getvalue(), title="Another crossing problem",
             not_duplicate_reason="Different pixel")
    body = r.get_json()
    assert r.status_code == 201, body
    assert any(w["code"] == "similar" and f"#{first['number']}" in w["message"] for w in body["warnings"])
    meta = teach.svc.load_meta(teach.svc.id_for_number(body["number"]))
    assert meta["similar_at_submit"][0]["number"] == first["number"]


def test_similarity_signals():
    from .conftest import png_bytes
    from teach.validate import validate_bitmap

    a = phash(validate_bitmap(IMAGE).gray)
    b = phash(validate_bitmap(png_bytes(256, 256, 255)).gray)
    assert len(a) == 16 and hamming(a, a) == 0 and hamming(a, b) > 0
    assert jaccard(tokens("thin line crossing thick"), tokens("thick line crossing thin")) == 1.0
    assert "the" not in tokens("the line")


def test_public_entry_is_the_only_builder():
    meta = {"number": 3, "status": "pending_review", "title_public": "T", "reviewed": False,
            "what_is_wrong": "W", "notes": "N", "visibility": {"thumbnail_opt_in": True, "thumbnail_public": True}}
    entry = TeachService.public_entry(meta)
    assert "what_is_wrong" not in entry and "notes" not in entry and "thumbnail_url" not in entry
