"""Manage link (design 13.3): view, delete, answer needs_info; token stored only as a hash."""

from __future__ import annotations

import io
import json

from teach import limits

from .conftest import REFERENCE, submit


def _h(token):
    return {"X-Teach-Manage-Token": token}


def test_only_token_hash_is_stored(teach):
    _, sub_id, token = submit(teach)
    raw = (teach.storage.root / f"teach/submissions/{sub_id}/meta.json").read_text()
    assert token not in raw and json.loads(raw)["manage_token_sha256"]
    for key in teach.storage.list("teach/"):
        assert token.encode() not in (teach.storage.root / key).read_bytes(), key


def test_view(teach):
    body, sub_id, token = submit(teach, what_is_wrong="Private detail")
    r = teach.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token))
    res = r.get_json()["result"]
    assert r.status_code == 200 and res["number"] == body["number"] and res["what_is_wrong"] == "Private detail"
    assert res["overlay_png"].startswith("data:image/png;base64,") and res["can_answer"] is False
    assert "manage_token_sha256" not in res
    assert r.headers["Cache-Control"] == "no-store"


def test_wrong_or_missing_token(teach):
    _, sub_id, token = submit(teach)
    for headers in ({}, _h("x" * 43), _h(token[:-1])):
        r = teach.client.get(f"/teach/api/manage/{sub_id}", headers=headers)
        assert r.status_code == 404 and r.get_json()["errors"][0]["code"] == "manage_token_invalid"
    assert teach.client.get("/teach/api/manage/t_nonexistent0", headers=_h(token)).status_code == 404


def test_manage_page_has_no_token_in_html(teach):
    _, sub_id, token = submit(teach)
    r = teach.client.get(f"/teach/manage/{sub_id}")
    assert r.status_code == 200 and token not in r.get_data(as_text=True)


def test_delete_keeps_only_tombstone(teach):
    body, sub_id, token = submit(teach)
    r = teach.client.delete(f"/teach/api/manage/{sub_id}", headers=_h(token))
    assert r.status_code == 200 and r.get_json()["result"]["status"] == "deleted"
    assert teach.storage.list(f"teach/submissions/{sub_id}/") == [f"teach/submissions/{sub_id}/meta.json"]
    tomb = json.loads((teach.storage.root / f"teach/submissions/{sub_id}/meta.json").read_text())
    assert tomb["status"] == "deleted" and "title" not in tomb and "manage_token_sha256" not in tomb
    assert teach.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token)).status_code == 404


def _needs_info(t, sub_id):
    meta = t.svc.load_meta(sub_id)
    meta["status"] = "needs_info"
    t.svc._save(meta)


def test_answer_only_when_needs_info(teach):
    _, sub_id, token = submit(teach)
    r = teach.client.patch(f"/teach/api/manage/{sub_id}", json={"note": "more info"}, headers=_h(token))
    assert r.status_code == 409 and r.get_json()["errors"][0]["code"] == "invalid_state"


def test_answer_with_note(teach):
    _, sub_id, token = submit(teach)
    _needs_info(teach, sub_id)
    r = teach.client.patch(f"/teach/api/manage/{sub_id}", json={"note": "It is 35 degrees"}, headers=_h(token))
    assert r.status_code == 200 and r.get_json()["result"]["status"] == "pending_review"
    meta = teach.svc.load_meta(sub_id)
    assert meta["answers"][-1]["note"] == "It is 35 degrees" and not meta["answers"][-1]["replaced_reference"]
    assert len(teach.queue.calls) == 1  # not re-scored


def test_answer_empty_is_422(teach):
    _, sub_id, token = submit(teach)
    _needs_info(teach, sub_id)
    r = teach.client.patch(f"/teach/api/manage/{sub_id}", json={"note": "  "}, headers=_h(token))
    assert r.status_code == 422 and r.get_json()["errors"][0]["code"] == "field_missing"


def test_answer_with_replacement_svg_rescores(teach):
    _, sub_id, token = submit(teach)
    _needs_info(teach, sub_id)
    r = teach.client.patch(f"/teach/api/manage/{sub_id}", headers=_h(token), content_type="multipart/form-data",
                           data={"note": "", "reference": (io.BytesIO(REFERENCE), "new.svg")})
    assert r.status_code == 200, r.get_json()
    meta = teach.svc.load_meta(sub_id)
    assert len(meta["results"]) == 2 and meta["answers"][-1]["replaced_reference"]
    assert len(teach.queue.calls) == 2


def test_answer_with_bad_svg_is_422(teach):
    _, sub_id, token = submit(teach)
    _needs_info(teach, sub_id)
    r = teach.client.patch(f"/teach/api/manage/{sub_id}", headers=_h(token), content_type="multipart/form-data",
                           data={"reference": (io.BytesIO(b"<svg"), "bad.svg")})
    assert r.status_code == 422 and r.get_json()["errors"][0]["code"].startswith("svg_")
    assert teach.svc.load_meta(sub_id)["status"] == "needs_info"


def test_manage_rate_limit(make_teach, monkeypatch):
    monkeypatch.setattr(limits, "MANAGE_PER_IP", (2, limits.HOUR))
    t = make_teach()
    _, sub_id, token = submit(t)
    assert t.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token)).status_code == 200
    assert t.client.get(f"/teach/api/manage/{sub_id}", headers=_h("bad")).status_code == 404
    assert t.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token)).status_code == 429
