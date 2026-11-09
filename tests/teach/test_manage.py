"""Manage link: view and delete; the retired answer operation is unavailable."""

from __future__ import annotations

import json
import pytest

from teach import limits

from .conftest import submit


def _h(token):
    return {"X-Teach-Manage-Token": token}


def test_only_token_hash_is_stored(teach):
    _, sub_id, token = submit(teach)
    raw = (teach.storage.root / f"teach/submissions/{sub_id}/meta.json").read_text()
    assert token not in raw and json.loads(raw)["manage_token_sha256"]
    for key in teach.storage.list("teach/"):
        assert token.encode() not in (teach.storage.root / key).read_bytes(), key


def test_view(teach):
    body, sub_id, token = submit(teach, description="Private detail")
    r = teach.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token))
    res = r.get_json()["result"]
    assert r.status_code == 200 and res["number"] == body["number"] and res["description"] == "Private detail"
    assert res["overlay_png"].startswith("data:image/png;base64,") and "can_answer" not in res
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


def test_answer_operation_removed(teach):
    _, sub_id, token = submit(teach)
    r = teach.client.patch(f"/teach/api/manage/{sub_id}", json={"note": "more info"}, headers=_h(token))
    assert r.status_code == 405
    assert "patch" not in teach.client.get("/teach/openapi.json").get_json()["paths"]["/teach/api/manage/{id}"]


@pytest.mark.parametrize("old_status", ["needs_info", "pending_review", "submitted"])
def test_old_records_load_as_pending(teach, old_status):
    _, sub_id, token = submit(teach)
    meta = teach.svc.load_meta(sub_id)
    meta["status"] = old_status
    teach.svc._save(meta)
    assert teach.svc.load_meta(sub_id)["status"] == "pending"
    assert teach.svc.admin_index()["items"][sub_id]["status"] == "pending"
    assert teach.svc.public_index()["items"][str(meta["number"])]["status"] == "pending"
    r = teach.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token))
    assert r.get_json()["result"]["status"] == "pending"


def test_manage_rate_limit(make_teach, monkeypatch):
    monkeypatch.setattr(limits, "MANAGE_PER_IP", (2, limits.HOUR))
    t = make_teach()
    _, sub_id, token = submit(t)
    assert t.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token)).status_code == 200
    assert t.client.get(f"/teach/api/manage/{sub_id}", headers=_h("bad")).status_code == 404
    assert t.client.get(f"/teach/api/manage/{sub_id}", headers=_h(token)).status_code == 429
