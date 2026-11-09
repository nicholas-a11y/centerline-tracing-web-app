"""Rate limits, quotas, proxy hops and the one-job-per-client scoring rule (design 12, 15)."""

from __future__ import annotations

import json
import sys
import threading
import time

import pytest

from teach import limits
from teach.limits import DailyQuota, QuotaExceeded, RateLimited, SlidingWindows, client_ip, quota_secret
from teach.scoring_worker import Busy, ScoringQueue
from teach.storage import LocalStorage

from .conftest import TOKEN, post


@pytest.mark.parametrize("remote,xff,hops,expected", [
    ("10.0.0.1", None, 0, "10.0.0.1"),
    ("10.0.0.1", "1.1.1.1", 0, "10.0.0.1"),                # hops 0: XFF ignored
    ("10.0.0.1", "1.1.1.1", 1, "1.1.1.1"),
    ("10.0.0.1", "6.6.6.6, 1.1.1.1", 1, "1.1.1.1"),        # spoofed left entries ignored
    ("10.0.0.1", "6.6.6.6, 1.1.1.1, 2.2.2.2", 2, "1.1.1.1"),
    ("10.0.0.1", "1.1.1.1", 2, "10.0.0.1"),                # fewer entries than hops
    (None, None, 0, "unknown"),
])
def test_client_ip(remote, xff, hops, expected):
    assert client_ip(remote, xff, hops) == expected


def test_sliding_window():
    now = [0.0]
    w = SlidingWindows(clock=lambda: now[0])
    for _ in range(3):
        w.hit(("k", 3, 60.0))
    with pytest.raises(RateLimited) as info:
        w.hit(("k", 3, 60.0))
    assert 0 < info.value.retry_after_s <= 60
    now[0] = 61.0
    w.hit(("k", 3, 60.0))


def test_global_window_not_consumed_when_ip_limited():
    w = SlidingWindows(clock=lambda: 0.0)
    w.hit(("ip", 1, 60.0), ("global", 5, 60.0))
    with pytest.raises(RateLimited):
        w.hit(("ip", 1, 60.0), ("global", 5, 60.0))
    for _ in range(4):
        w.hit((f"other{_}", 1, 60.0), ("global", 5, 60.0))


def test_daily_quota_persists_and_hashes_ips(tmp_path, monkeypatch):
    store = LocalStorage(tmp_path)
    secret = quota_secret(store)
    q = DailyQuota(store, secret)
    monkeypatch.setattr(limits, "SUBMIT_PER_IP_DAY", 2)
    q.record("203.0.113.9", 100)
    q2 = DailyQuota(store, quota_secret(store))  # restart: same salt, same counts
    q2.record("203.0.113.9", 100)
    with pytest.raises(QuotaExceeded) as info:
        q2.check("203.0.113.9")
    assert info.value.scope == "ip"
    q2.check("198.51.100.1")
    raw = b"".join((tmp_path / k).read_bytes() for k in store.list("teach/quota/"))
    assert b"203.0.113.9" not in raw
    doc = json.loads((tmp_path / store.list("teach/quota/2")[0]).read_text())
    assert doc["submissions"] == 2 and doc["bytes"] == 200


def test_quota_bytes_and_global_caps(tmp_path, monkeypatch):
    store = LocalStorage(tmp_path)
    q = DailyQuota(store, b"k" * 32)
    with pytest.raises(QuotaExceeded):
        q.check("a", limits.BYTES_PER_IP_DAY + 1)
    monkeypatch.setattr(limits, "SUBMIT_GLOBAL_DAY", 1)
    q.record("a", 1)
    with pytest.raises(QuotaExceeded) as info:
        q.check("b")
    assert info.value.scope == "global" and q.global_closed()


def test_quota_salt_not_derived_from_admin_token(tmp_path):
    store = LocalStorage(tmp_path)
    salt = quota_secret(store)
    assert len(salt) == 32 and quota_secret(store) == salt
    assert TOKEN.encode() not in salt and TOKEN not in (tmp_path / "teach/quota/salt.json").read_text()


# ---------------------------------------------------------------- scoring queue

SLEEP = [sys.executable, "-c", "import time; time.sleep(1.5)"]


def _start(q, ip, results):
    def go():
        try:
            results.append(q.run(ip, b"png", {"paths": []}, {}, {}))
        except Busy as exc:
            results.append(exc.reason)
    t = threading.Thread(target=go)
    t.start()
    return t


def _wait_depth(q, depth, timeout=10.0):
    end = time.time() + timeout
    while q.depth < depth and time.time() < end:
        time.sleep(0.02)
    assert q.depth >= depth


def test_one_job_per_ip():
    q = ScoringQueue(timeout_s=10, command=SLEEP)
    results: list = []
    t = _start(q, "1.1.1.1", results)
    _wait_depth(q, 1)
    with pytest.raises(Busy) as info:
        q.run("1.1.1.1", b"png", {"paths": []}, {}, {})
    assert info.value.reason == "ip_busy"
    t.join()
    assert results[0]["outcome"] == "error"  # the sleep command writes no result


def test_queue_full_after_three_waiting():
    q = ScoringQueue(timeout_s=10, max_waiting=3, command=SLEEP)
    results: list = []
    threads = [_start(q, f"10.0.0.{i}", results) for i in range(4)]
    _wait_depth(q, 4)
    with pytest.raises(Busy) as info:
        q.run("10.0.0.99", b"png", {"paths": []}, {}, {})
    assert info.value.reason == "queue_full"
    for t in threads:
        t.join()
    assert len(results) == 4 and q.depth == 0


def test_timeout():
    q = ScoringQueue(timeout_s=0.5, command=[sys.executable, "-c", "import time; time.sleep(5)"])
    t0 = time.time()
    out = q.run("1.1.1.1", b"png", {"paths": []}, {}, {})
    assert out["outcome"] == "timeout" and out["timed_out"] and time.time() - t0 < 4
    assert q.depth == 0


def test_second_request_from_same_ip_gets_503(make_teach):
    t = make_teach()
    t.queue.busy = "ip_busy"
    r = post(t.client, "/teach/api/submissions")
    assert r.status_code == 503 and r.get_json()["errors"][0]["code"] == "busy"
    assert t.storage.list("teach/submissions/") == []


def test_validate_per_ip_limit(make_teach, monkeypatch):
    monkeypatch.setattr(limits, "VALIDATE_PER_IP", (2, limits.HOUR))
    t = make_teach()
    assert post(t.client, "/teach/api/validate").status_code == 200
    assert post(t.client, "/teach/api/validate").status_code == 200
    r = post(t.client, "/teach/api/validate")
    assert r.status_code == 429 and r.get_json()["errors"][0]["code"] == "rate_limited"


def test_submission_limit_per_hour(make_teach, monkeypatch):
    monkeypatch.setattr(limits, "SUBMIT_PER_IP_HOUR", (1, limits.HOUR))
    t = make_teach()
    assert post(t.client, "/teach/api/submissions").status_code == 201
    r = post(t.client, "/teach/api/submissions", title="Second")
    assert r.status_code == 429 and r.get_json()["errors"][0]["code"] == "rate_limited"


def test_trusted_proxy_hops_in_app(make_teach, monkeypatch):
    monkeypatch.setattr(limits, "VALIDATE_PER_IP", (1, limits.HOUR))
    t = make_teach(env_extra={"TEACH_TRUSTED_PROXY_HOPS": "1"})
    h1 = {"X-Forwarded-For": "6.6.6.6, 1.1.1.1"}
    h2 = {"X-Forwarded-For": "7.7.7.7, 1.1.1.1"}  # spoofed prefix: still the same client
    h3 = {"X-Forwarded-For": "2.2.2.2"}
    from .conftest import multipart

    def go(h):
        return t.client.post("/teach/api/validate", data=multipart(), headers=h, content_type="multipart/form-data")

    assert go(h1).status_code == 200
    assert go(h2).status_code == 429
    assert go(h3).status_code == 200
