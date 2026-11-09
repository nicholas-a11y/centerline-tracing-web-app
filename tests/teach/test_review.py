import pytest

from .conftest import submit
from .test_admin import AUTH


@pytest.mark.parametrize("status", ["pending", "accepted", "rejected"])
def test_review_status_notes_and_privacy(teach, status):
    body, sid, token = submit(teach)
    n = body["number"]
    url = f"/teach/api/admin/cases/{n}"
    r = teach.client.patch(url, headers=AUTH, json={"status": status, "admin_note_public": "Public note",
                                                  "admin_note_private": "Private note"})
    assert r.status_code == 200 and r.get_json()["case"]["status"] == status
    public = teach.client.get(f"/teach/api/submissions/{n}").get_json()
    assert public["admin_note_public"] == "Public note" and "admin_note_private" not in public
    assert "Public note" in teach.client.get("/teach/queue").get_data(as_text=True)
    own = teach.client.get(f"/teach/api/manage/{sid}", headers={"X-Teach-Manage-Token": token}).get_json()["result"]
    assert own["admin_note_public"] == "Public note" and "admin_note_private" not in own
    assert "manage_token_sha256" not in r.get_json()["case"]


@pytest.mark.parametrize("data", [
    {"status": "other"}, {"status": "needs_info"}, {"tolerance_px": 0},
    {"tolerance_px": True}, {"region": {"x0": 0}}, {"unknown": 1},
    {"thumbnail_public": True}, {"bitmap_public": True},
])
def test_invalid_update_is_atomic(teach, data):
    b, sid, _ = submit(teach, public_thumbnail=False, share_bitmap=False)
    before = teach.svc.load_meta(sid)
    r = teach.client.patch(f"/teach/api/admin/cases/{b['number']}", json=data, headers=AUTH)
    assert r.status_code == 422
    assert teach.svc.load_meta(sid) == before


def test_hide_title_is_per_case_and_visibility(teach):
    b, sid, _ = submit(teach, public_thumbnail=True, share_bitmap=True)
    n = b["number"]
    url = f"/teach/api/admin/cases/{n}"
    teach.client.patch(url, headers=AUTH, json={"title_hidden": True})
    assert teach.svc.load_meta(sid)["reviewed"] is False
    assert teach.client.get(f"/teach/api/submissions/{n}").get_json()["title_hidden"]
    r = teach.client.patch(url, headers=AUTH, json={"status": "accepted", "title_hidden": False,
                                                  "thumbnail_public": True, "bitmap_public": True})
    assert r.status_code == 200
    assert teach.client.get(f"/teach/q/{n}/image.png").status_code == 200
    assert teach.client.get(f"/teach/q/{n}/thumb.png").status_code == 200
    assert teach.client.patch(url, headers=AUTH, json={"bitmap_public": False}).status_code == 200
    assert teach.client.get(f"/teach/q/{n}/image.png").status_code == 404


def test_queue_filter_search_and_delete(teach):
    b, _, _ = submit(teach)
    n = b["number"]
    r = teach.client.get(f"/teach/api/admin/cases?q={n}&status=pending", headers=AUTH).get_json()
    assert r["shown"] == 1 and r["counts"]["pending"] == 1
    assert teach.client.get("/teach/api/admin/cases?status=unknown", headers=AUTH).status_code == 422
    assert teach.client.delete(f"/teach/api/admin/cases/{n}", headers=AUTH).status_code == 200
    assert teach.client.get(f"/teach/api/admin/cases/{n}", headers=AUTH).status_code == 404
    assert teach.client.get("/teach/api/admin/summary", headers=AUTH).get_json()["total"] == 0
    assert teach.client.post("/teach/api/admin/rebuild-index", headers=AUTH).get_json()["items"] == 1


def test_raw_svg_download_never_inline(teach):
    b, _, _ = submit(teach)
    url = f"/teach/api/admin/cases/{b['number']}/hand_traced.svg"
    assert teach.client.get(url).status_code == 401
    r = teach.client.get(url, headers=AUTH)
    assert r.content_type.startswith("text/plain")
    assert r.headers["Content-Disposition"].startswith("attachment;")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
