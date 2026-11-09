"""ntfy alerts (design 15.5): debounced, title/status/link only, never drawings or descriptions."""

from __future__ import annotations

import logging

from teach.notify import MAX_ITEMS, Notifier

URL = "https://ntfy.sh/secret-topic-abc"
BASE = "https://centerline.example"


class Sender:
    def __init__(self, fail: int = 0):
        self.sent: list[tuple[str, bytes, dict]] = []
        self.fail = fail

    def __call__(self, url, body, headers):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("down")
        self.sent.append((url, body, headers))


def make(sender, now):
    return Notifier(URL, BASE, sender=sender, clock=lambda: now[0], background=False)


def test_unconfigured_does_nothing():
    s = Sender()
    n = Notifier(None, BASE, sender=s, background=False)
    n.event(1, "pending", "T")
    n.urgent("global_cap")
    assert not n.configured and s.sent == []


def test_debounce_and_batch():
    s, now = Sender(), [0.0]
    n = make(s, now)
    n.event(1, "pending", "First")
    assert len(s.sent) == 1
    now[0] = 60.0
    n.event(2, "pending", "Second")
    n.event(3, "pending", None)
    assert len(s.sent) == 1  # held back
    now[0] = 31 * 60.0
    n.flush()
    assert len(s.sent) == 2
    url, body, headers = s.sent[1]
    text = body.decode()
    assert url == URL and headers["Title"] == "Teach: 2 updates"
    assert "#2 pending: Second https://centerline.example/teach/admin#q=2" in text
    assert "#3 pending: title hidden" in text


def test_fixed_headers_only():
    s, now = Sender(), [0.0]
    make(s, now).event(1, "pending", "T")
    headers = s.sent[0][2]
    assert set(headers) == {"Content-Type", "Title", "Tags", "Priority", "Click"}
    assert headers["Click"] == f"{BASE}/teach/admin"


def test_more_than_max_items():
    n = Notifier(URL, BASE, sender=Sender(), background=False)
    body = n.body([(i, "pending", "t") for i in range(MAX_ITEMS + 3)]).decode()
    assert body.count("\n") == MAX_ITEMS + 1 and "+3 more" in body


def test_failure_logs_class_name_not_url(caplog):
    s, now = Sender(fail=2), [0.0]
    with caplog.at_level(logging.WARNING, logger="teach"):
        make(s, now).event(1, "pending", "T")
    assert "Teach alert failed: ConnectionError" in caplog.text
    assert "secret-topic" not in caplog.text and "ntfy" not in caplog.text


def test_one_retry():
    s, now = Sender(fail=1), [0.0]
    make(s, now).event(1, "pending", "T")
    assert len(s.sent) == 1


def test_urgent_at_most_once_per_hour():
    s, now = Sender(), [0.0]
    n = make(s, now)
    n.urgent("global_cap")
    n.urgent("global_cap")
    now[0] = 3601.0
    n.urgent("global_cap")
    n.urgent("unknown_kind")
    assert len(s.sent) == 2 and s.sent[0][2]["Priority"] == "high"


def test_submission_sends_title_only(make_teach):
    s = Sender()
    t = make_teach(env_extra={"TEACH_ALERT_WEBHOOK_URL": URL, "TEACH_PUBLIC_BASE_URL": BASE},
                   alert_sender=s, notifier=None)
    from .conftest import submit

    t.svc.notifier._background = False
    submit(t, title="Visible title", description="PRIVATE-WRONG and PRIVATE-NOTES")
    assert len(s.sent) == 1
    body = s.sent[0][1].decode()
    assert "Visible title" in body and "PRIVATE" not in body and "base64" not in body
