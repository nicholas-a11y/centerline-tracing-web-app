import copy
import time
from datetime import datetime, timedelta, timezone

from teach.retention import LAST_SWEEP_KEY, retention_status, run_sweep
from teach.service import new_id
from teach.storage import submission_key

from .conftest import submit
from .test_admin import AUTH


def _set_status_since(teach, sub_id, status, now, days):
    meta = teach.svc.load_meta(sub_id)
    meta["status"] = status
    meta["retention"] = {"status_since": (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    teach.storage.put_json(submission_key(sub_id, "meta.json"), meta)
    index = teach.svc.admin_index()
    index["items"][sub_id]["status"] = status
    teach.storage.put_json("teach/index/admin.json", index)


def _clone_case(teach, source_id):
    source = teach.svc.load_meta(source_id)
    new_case_id = new_id()
    number = teach.svc._next_number()
    old_prefix = f"teach/submissions/{source_id}/"
    new_prefix = f"teach/submissions/{new_case_id}/"
    for key in teach.storage.list(old_prefix):
        teach.storage.put(new_prefix + key[len(old_prefix):], teach.storage.get(key))
    clone = copy.deepcopy(source)
    clone["id"] = new_case_id
    clone["number"] = number
    clone["title"] = f"Retention case {number}"
    clone["title_public"] = clone["title"]
    teach.svc._save(clone)
    return number, new_case_id


def test_retention_ages_status_compatibility_and_tombstones(teach):
    now = datetime.now(timezone.utc)
    body, base_id, _ = submit(teach, title="Retention base")
    cases = {"old-pending": (body["number"], base_id)}
    for label in ("old-rejected", "fresh-pending", "fresh-rejected", "old-accepted", "legacy"):
        cases[label] = _clone_case(teach, base_id)
    _set_status_since(teach, cases["old-pending"][1], "pending", now, 91)
    _set_status_since(teach, cases["old-rejected"][1], "rejected", now, 31)
    _set_status_since(teach, cases["fresh-pending"][1], "pending", now, 89)
    _set_status_since(teach, cases["fresh-rejected"][1], "rejected", now, 29)
    _set_status_since(teach, cases["old-accepted"][1], "accepted", now, 365)
    _set_status_since(teach, cases["legacy"][1], "needs_info", now, 91)

    assert retention_status(teach.svc, now)["items_due"] == 3
    result = run_sweep(teach.svc, now)
    assert result["deleted_count"] == 3
    assert set(result["deleted_cases"]) == {
        cases["old-pending"][0], cases["old-rejected"][0], cases["legacy"][0]
    }
    for label in ("old-pending", "old-rejected", "legacy"):
        number, sid = cases[label]
        meta = teach.svc.load_meta(sid)
        assert meta["status"] == "deleted"
        assert "title" not in meta and "description" not in meta and "manage_token_sha256" not in meta
        assert teach.storage.list(f"teach/submissions/{sid}/") == [submission_key(sid, "meta.json")]
        public = teach.client.get(f"/teach/api/submissions/{number}").get_json()
        assert public == {"number": number, "status": "deleted"}
    for label in ("fresh-pending", "fresh-rejected", "old-accepted"):
        assert teach.svc.load_meta(cases[label][1])["status"] != "deleted"
    assert run_sweep(teach.svc, now)["deleted_count"] == 0
    assert teach.storage.get_json(LAST_SWEEP_KEY)["deleted_count"] == 0


def test_authenticated_admin_visit_schedules_a_missed_sweep(teach):
    body, sid, _ = submit(teach)
    _set_status_since(teach, sid, "pending", datetime.now(timezone.utc), 91)
    response = teach.client.get("/teach/api/admin/cases", headers=AUTH)
    assert response.status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if teach.storage.get_json(LAST_SWEEP_KEY, None):
            break
        time.sleep(0.01)
    assert teach.storage.get_json(LAST_SWEEP_KEY, None)
    assert teach.svc.load_meta(sid)["status"] == "deleted"


def test_admin_can_run_retention_sweep_now(teach):
    response = teach.client.post("/teach/api/admin/retention-sweep", headers=AUTH)
    assert response.status_code == 200
    assert response.get_json()["result"]["deleted_count"] == 0
