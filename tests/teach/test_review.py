import pytest
import time

from .conftest import submit
from .test_admin import AUTH
from teach.rescore import history_label
from teach.storage import submission_key


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
    {"status": "other"}, {"status": "unknown"}, {"tolerance_px": 0},
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


def test_scoring_stores_private_app_centerline_and_admin_can_rescore(teach):
    body, sid, _ = submit(teach)
    key = submission_key(sid, "app_centerline.svg")
    first = teach.storage.get(key)
    assert first and b"<svg" in first
    public = teach.client.get(f"/teach/api/submissions/{body['number']}").get_json()
    assert "app_centerline" not in str(public)
    assert teach.client.get(
        f"/teach/api/admin/cases/{body['number']}/app_centerline.svg"
    ).status_code == 401

    start = teach.client.post(f"/teach/api/admin/cases/{body['number']}/rescore", headers=AUTH)
    assert start.status_code == 202
    job = start.get_json()["job"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = teach.client.get(f"/teach/api/admin/jobs/{job['id']}", headers=AUTH)
        assert status.status_code == 200
        job = status.get_json()["job"]
        if job["done"]:
            break
        time.sleep(0.01)
    assert job["done"] and job["failed"] == 0
    meta = teach.svc.load_meta(sid)
    assert len(meta["results"]) == 2
    assert teach.storage.get(key)

    download = teach.client.get(
        f"/teach/api/admin/cases/{body['number']}/app_centerline.svg", headers=AUTH
    )
    assert download.status_code == 200
    assert download.data == teach.storage.get(key)
    assert download.headers["Content-Disposition"].startswith("attachment;")


def test_failing_since_and_now_passing_labels():
    assert history_label([{"outcome": "fail", "engine_commit": "a"},
                          {"outcome": "fail", "engine_commit": "b"}]) == "Failing since a"
    assert history_label([{"outcome": "fail", "engine_commit": "a"},
                          {"outcome": "pass", "engine_commit": "b"}]) == "Now passing"
    assert history_label([{"outcome": "pass", "engine_commit": "a"}]) == "Passing"


def test_rescore_all_skips_cases_scored_by_current_engine(teach, monkeypatch):
    body, _, _ = submit(teach)
    calls_before = len(teach.queue.calls)
    monkeypatch.setattr("teach.rescore.engine_commit", lambda: "test")
    response = teach.client.post("/teach/api/admin/rescore-all", headers=AUTH)
    assert response.status_code == 202
    job = response.get_json()["job"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = teach.client.get(f"/teach/api/admin/jobs/{job['id']}", headers=AUTH).get_json()["job"]
        if job["done"]:
            break
        time.sleep(0.01)
    assert job["done"] and job["total"] == 1 and job["skipped"] == 1
    assert len(teach.queue.calls) == calls_before
