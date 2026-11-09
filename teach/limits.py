"""Teach abuse limits (design 12, 15.4): client IP, hourly windows and persisted daily quotas."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timezone

HOUR = 3600.0

# (limit, window seconds); per IP unless the key says global
VALIDATE_PER_IP = (60, HOUR)          # /api/validate
VALIDATE_GLOBAL = (600, HOUR)
SIMILAR_PER_IP = (60, HOUR)
SUBMIT_PER_IP_HOUR = (5, HOUR)        # submissions + me-too
MANAGE_PER_IP = (30, HOUR)

SUBMIT_PER_IP_DAY = 20
SUBMIT_GLOBAL_DAY = 200
BYTES_PER_IP_DAY = 20 * 1024 * 1024
BYTES_GLOBAL_DAY = 500 * 1024 * 1024


def client_ip(remote_addr: str | None, forwarded_for: str | None, hops: int) -> str:
    """The n-th entry from the right of X-Forwarded-For (n = trusted hops); remote_addr when n = 0.

    Left-most entries are client-controlled and never trusted. If the header has fewer
    entries than trusted hops, the request did not come through the expected proxies and
    the socket address is used.
    """
    if hops <= 0:
        return remote_addr or "unknown"
    entries = [e.strip() for e in (forwarded_for or "").split(",") if e.strip()]
    if len(entries) < hops:
        return remote_addr or "unknown"
    return entries[-hops]


class RateLimited(Exception):
    def __init__(self, retry_after_s: int, scope: str = "ip"):
        super().__init__(f"rate limited ({scope})")
        self.retry_after_s = max(1, int(retry_after_s))
        self.scope = scope


class QuotaExceeded(Exception):
    def __init__(self, scope: str, message: str):
        super().__init__(message)
        self.scope = scope
        self.message = message


class SlidingWindows:
    """In-memory sliding-window counters (reset on restart, as designed)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, window: float, now: float) -> deque:
        q = self._hits.setdefault(key, deque())
        while q and q[0] <= now - window:
            q.popleft()
        return q

    def check(self, key: str, limit: int, window: float) -> None:
        with self._lock:
            now = self._clock()
            q = self._prune(key, window, now)
            if len(q) >= limit:
                raise RateLimited(q[0] + window - now, "global" if key.startswith("global:") else "ip")

    def hit(self, *checks: tuple[str, int, float]) -> None:
        """Check every (key, limit, window) and only then record one hit on each."""
        with self._lock:
            now = self._clock()
            for key, limit, window in checks:
                q = self._prune(key, window, now)
                if len(q) >= limit:
                    raise RateLimited(q[0] + window - now, "global" if key.startswith("global:") else "ip")
            for key, _limit, _window in checks:
                self._hits[key].append(now)


QUOTA_SALT_KEY = "teach/quota/salt.json"


def quota_secret(storage) -> bytes:
    """Random HMAC key for per-IP quota entries, created once and kept in storage.

    Stored so counts survive restarts; deliberately not derived from the admin token.
    """
    doc = storage.get_json(QUOTA_SALT_KEY, None)
    if not doc or not isinstance(doc.get("salt"), str) or len(doc["salt"]) < 32:
        doc = {"salt": secrets.token_hex(32)}
        storage.put_json(QUOTA_SALT_KEY, doc)
    return bytes.fromhex(doc["salt"])


class DailyQuota:
    """Submission and byte counts per UTC day, persisted in teach/quota/<date>.json.

    Per-IP entries are keyed by an HMAC of the IP, so no raw IP is ever stored.
    """

    def __init__(self, storage, secret: bytes, utcnow=None):
        self.storage = storage
        self._secret = secret
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()

    def ip_key(self, ip: str) -> str:
        return hmac.new(self._secret, ip.encode("utf-8"), hashlib.sha256).hexdigest()[:20]

    def _key(self) -> str:
        return f"teach/quota/{self._utcnow().strftime('%Y-%m-%d')}.json"

    def _load(self) -> dict:
        data = self.storage.get_json(self._key(), None) or {}
        data.setdefault("submissions", 0)
        data.setdefault("bytes", 0)
        data.setdefault("per_ip", {})
        return data

    def check(self, ip: str, add_bytes: int = 0) -> None:
        with self._lock:
            self._check(self._load(), ip, add_bytes)

    def _check(self, data: dict, ip: str, add_bytes: int) -> None:
        mine = data["per_ip"].get(self.ip_key(ip), {"submissions": 0, "bytes": 0})
        if data["submissions"] >= SUBMIT_GLOBAL_DAY:
            raise QuotaExceeded("global", "Teach submissions are closed for the rest of the UTC day (global limit).")
        if data["bytes"] + add_bytes > BYTES_GLOBAL_DAY:
            raise QuotaExceeded("global", "Teach storage for today is full; submissions reopen tomorrow (UTC).")
        if mine["submissions"] >= SUBMIT_PER_IP_DAY:
            raise QuotaExceeded("ip", f"You have reached {SUBMIT_PER_IP_DAY} submissions today (UTC).")
        if mine["bytes"] + add_bytes > BYTES_PER_IP_DAY:
            raise QuotaExceeded("ip", "You have reached today's upload size limit (20 MB, UTC day).")

    def record(self, ip: str, add_bytes: int) -> dict:
        """Check and record one submission; returns the updated global counts."""
        with self._lock:
            data = self._load()
            self._check(data, ip, add_bytes)
            k = self.ip_key(ip)
            mine = data["per_ip"].setdefault(k, {"submissions": 0, "bytes": 0})
            mine["submissions"] += 1
            mine["bytes"] += int(add_bytes)
            data["submissions"] += 1
            data["bytes"] += int(add_bytes)
            self.storage.put_json(self._key(), data)
            return {"submissions": data["submissions"], "bytes": data["bytes"]}

    def global_closed(self) -> bool:
        data = self._load()
        return data["submissions"] >= SUBMIT_GLOBAL_DAY or data["bytes"] >= BYTES_GLOBAL_DAY


_LOG_SALT = secrets.token_bytes(16)  # per process, so log hashes cannot be brute-forced back to IPs offline


def log_ip_hash(ip: str) -> str:
    """Truncated hash for log lines only (13.4)."""
    return hmac.new(_LOG_SALT, ip.encode("utf-8"), hashlib.sha256).hexdigest()[:8]
