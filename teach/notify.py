"""ntfy admin alerts (design 15.5): debounced, title/status/link only, URL treated as a secret."""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("teach")

DEBOUNCE_S = 30 * 60
URGENT_MIN_INTERVAL_S = 60 * 60
MAX_ITEMS = 10
MAX_BODY_BYTES = 3000
SEND_TIMEOUT_S = 5
URGENT_TEXT = {
    "global_cap": "Teach: a global daily limit was reached; submissions are closed until 00:00 UTC.",
    "admin_lockout": "Teach: admin is locked for everyone for 1 hour after repeated failed attempts.",
}


def urllib_sender(url: str, body: bytes, headers: dict) -> None:
    import urllib.request

    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=SEND_TIMEOUT_S) as resp:  # noqa: S310 (configured https URL)
        resp.read(1024)


class Notifier:
    """Collects (number, status, public title) events and sends at most one message per 30 minutes."""

    def __init__(self, url: str | None, base_url: str | None, sender=None, clock=time.monotonic,
                 background: bool = True, debounce_s: float = DEBOUNCE_S):
        self._url = url
        self._base = (base_url or "").rstrip("/")
        self._sender = sender or urllib_sender
        self._clock = clock
        self._background = background
        self._debounce_s = debounce_s
        self._lock = threading.Lock()
        self._pending: list[tuple[int, str, str | None]] = []
        self._last_sent: float | None = None
        self._urgent_last: dict[str, float] = {}
        self._timer: threading.Timer | None = None

    @property
    def configured(self) -> bool:
        return bool(self._url)

    def event(self, number: int, status: str, title_public: str | None) -> None:
        """``title_public`` is the sanitised public title, or None when the admin hid it."""
        if not self._url:
            return
        with self._lock:
            self._pending.append((int(number), str(status), title_public))
            now = self._clock()
            if self._last_sent is None or now - self._last_sent >= self._debounce_s:
                self._flush_locked()
            elif self._timer is None and self._background:
                delay = self._debounce_s - (now - self._last_sent)
                self._timer = threading.Timer(delay, self.flush)
                self._timer.daemon = True
                self._timer.start()

    def flush(self) -> None:
        with self._lock:
            self._timer = None
            if self._pending:
                self._flush_locked()

    def urgent(self, kind: str) -> None:
        if not self._url or kind not in URGENT_TEXT:
            return
        with self._lock:
            now = self._clock()
            last = self._urgent_last.get(kind)
            if last is not None and now - last < URGENT_MIN_INTERVAL_S:
                return
            self._urgent_last[kind] = now
        self._dispatch(self.headers(1, priority="high"), URGENT_TEXT[kind].encode("utf-8"))

    def headers(self, count: int, priority: str = "default") -> dict:
        h = {"Content-Type": "text/plain; charset=utf-8", "Title": f"Teach: {count} update{'s' if count != 1 else ''}",
             "Tags": "inbox_tray", "Priority": priority}
        if self._base:
            h["Click"] = f"{self._base}/teach/admin"
        return h

    def body(self, items: list[tuple[int, str, str | None]]) -> bytes:
        lines = []
        for number, status, title in items[:MAX_ITEMS]:
            shown = title if title is not None else "title hidden"
            lines.append(f"#{number} {status}: {shown} {self._base}/teach/admin#q={number}")
        if len(items) > MAX_ITEMS:
            lines.append(f"+{len(items) - MAX_ITEMS} more")
        out = b""
        for line in lines:
            enc = (line + "\n").encode("utf-8")
            if len(out) + len(enc) > MAX_BODY_BYTES:
                out += b"+more (truncated)\n"
                break
            out += enc
        return out

    def _flush_locked(self) -> None:
        items, self._pending = self._pending, []
        self._last_sent = self._clock()
        self._dispatch(self.headers(len(items)), self.body(items))

    def _dispatch(self, headers: dict, body: bytes) -> None:
        if self._background:
            threading.Thread(target=self._send, args=(headers, body), daemon=True).start()
        else:
            self._send(headers, body)

    def _send(self, headers: dict, body: bytes) -> None:
        for attempt in range(2):  # one retry
            try:
                self._sender(self._url, body, headers)
                return
            except Exception as exc:
                if attempt == 1:
                    log.warning("Teach alert failed: %s", type(exc).__name__)
