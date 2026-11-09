"""Stage 5d C: the text field is optional; the duplicate check copes with empty text."""

from __future__ import annotations

import json

import pytest

from teach import similarity
from teach.similarity import MIN_TEXT_TOKENS, compare, text_similarity, tokens

from .conftest import META, post, submit

FAR_A, FAR_B = "0" * 16, "f" * 16  # pHash Hamming distance 64: never an image match


def sig(title: str, description: str = "", phash: str = FAR_A, **extra) -> dict:
    out = {"image_sha256": f"img-{phash}-{title}", "geometry_sha256": f"geo-{title}-{description}",
           "phash": phash, "width": 10, "height": 10,
           "tokens": tokens(title, description), "title_tokens": tokens(title),
           "has_text": bool(description)}
    out.update(extra)
    return out


def test_min_text_tokens_is_four():
    assert MIN_TEXT_TOKENS == 4


# ------------------------------------------------------------------ submit with and without text


def test_submit_without_text_field_stores_empty_string(teach):
    meta = {k: v for k, v in META.items() if k != "description"}
    r = teach.client.post("/teach/api/submissions", content_type="multipart/form-data",
                          data={**{k: v for k, v in post_data().items() if k != "meta"}, "meta": json.dumps(meta)})
    assert r.status_code == 201, r.get_json()
    sub_id = r.get_json()["manage_url"].split("/teach/manage/")[1].split("#")[0]
    stored = teach.svc.load_meta(sub_id)
    assert stored["description"] == ""
    assert teach.svc.admin_entry(stored)["has_text"] is False


def test_submit_with_blank_text_stores_empty_string(teach):
    _, sub_id, _ = submit(teach, description="   ")
    assert teach.svc.load_meta(sub_id)["description"] == ""


def test_submit_with_text_keeps_it(teach):
    _, sub_id, _ = submit(teach, description="The thin line bends at the crossing.")
    stored = teach.svc.load_meta(sub_id)
    assert stored["description"] == "The thin line bends at the crossing."
    entry = teach.svc.admin_entry(stored)
    assert entry["has_text"] is True and entry["title_tokens"] == tokens(META["title"])


def test_validate_without_text_is_ok(teach):
    r = post(teach.client, "/teach/api/validate", description="")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] is True, body


def test_pending_public_entry_unchanged_without_text(teach):
    body, _, _ = submit(teach, description="")
    entry = teach.client.get(f"/teach/api/submissions/{body['number']}").get_json()
    assert "description" not in entry and entry["details"] == "awaiting review"


def post_data():
    from .conftest import multipart

    return multipart()


# ------------------------------------------------------------------ duplicate check with empty text


def test_short_identical_titles_without_text_do_not_match():
    a, b = sig("Crossing bug"), sig("Crossing bug", phash=FAR_B)
    assert text_similarity(a, b) == 0.0
    assert compare(a, b) is None


def test_three_token_titles_do_not_match_but_four_do():
    three = "thin crossing kink"
    four = "thin crossing kink thick"
    assert len(tokens(three)) == 3 and len(tokens(four)) == 4
    assert compare(sig(three), sig(three, phash=FAR_B)) is None
    match = compare(sig(four), sig(four, phash=FAR_B))
    assert match == {"exact": False, "image": False, "geometry": False, "text": 1.0}


def test_empty_text_compares_titles_only():
    title = "Thin line kinks where it crosses a thick line"
    new = sig(title)                                         # no description
    other = sig(title, "Completely different words about gears and pulleys and belts", phash=FAR_B)
    assert text_similarity(new, other) == 1.0                # description of the other side ignored
    assert compare(new, other)["text"] == 1.0


def test_both_with_text_uses_title_and_description():
    title = "Thin line kinks where it crosses a thick line"
    a = sig(title, "gears pulleys belts chains sprockets")
    b = sig(title, "arcs circles spirals ellipses ovals", phash=FAR_B)
    expected = similarity.jaccard(a["tokens"], b["tokens"])
    assert text_similarity(a, b) == pytest.approx(expected) and expected < similarity.TEXT_ALONE


def test_entries_from_before_5d_fall_back_to_tokens():
    title = "Thin line kinks where it crosses a thick line"
    old = sig(title, "x", phash=FAR_B)
    del old["title_tokens"], old["has_text"]
    old["tokens"] = tokens(title)
    assert text_similarity(sig(title), old) == 1.0


def test_short_title_with_text_still_needs_four_tokens_on_both_sides():
    a = sig("Kink", "short")
    b = sig("Kink", "short", phash=FAR_B)
    assert len(a["tokens"]) < MIN_TEXT_TOKENS and text_similarity(a, b) == 0.0


def test_exact_duplicate_still_blocked_without_text(teach):
    submit(teach, description="")
    r = post(teach.client, "/teach/api/submissions", description="")
    assert r.status_code == 409 and r.get_json()["errors"][0]["code"] == "duplicate_exact"
