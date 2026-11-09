"""Scheduled retention for rejected and unreviewed Teach cases."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from . import schema

log = logging.getLogger("teach")
SWEEP_INTERVAL = timedelta(hours=24)
REJECTED_AGE = timedelta(days=30)
PENDING_AGE = timedelta(days=90)
LAST_SWEEP_KEY = "teach/sweep/last.json"


def _parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _status_since(meta: dict) -> datetime | None:
    current = schema.normalize_status(meta.get("status", "pending"))
    value = (meta.get("retention") or {}).get("status_since")
    parsed = _parse_time(value)
    if parsed:
        return parsed
    for event in reversed(meta.get("status_history") or []):
        if schema.normalize_status(event.get("status")) == current:
            parsed = _parse_time(event.get("at"))
            if parsed:
                return parsed
    return _parse_time(meta.get("created_at"))


def _due(meta: dict, now: datetime) -> bool:
    status = schema.normalize_status(meta.get("status", "pending"))
    if status == "accepted" or status == "deleted":
        return False
    age = REJECTED_AGE if status == "rejected" else PENDING_AGE
    since = _status_since(meta)
    return since is not None and now - since >= age


def retention_status(service, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    with service._lock:
        last = service.storage.get_json(LAST_SWEEP_KEY, None)
        finished = _parse_time((last or {}).get("finished_at"))
        next_run = finished + SWEEP_INTERVAL if finished else now
        due = 0
        for item in service.admin_index()["items"].values():
            if item.get("status") == "deleted":
                continue
            meta = service.load_meta(item.get("id"))
            if meta and _due(meta, now):
                due += 1
    return {"last": last, "next_run": next_run.strftime("%Y-%m-%dT%H:%M:%SZ"), "items_due": due}


def run_sweep(service, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    started = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    deleted = []
    with service._lock:
        for item in service.admin_index()["items"].values():
            case_id = item.get("id")
            if not case_id:
                continue
            meta = service.load_meta(case_id)
            if meta and meta.get("status") != "deleted" and _due(meta, now):
                number = int(meta["number"])
                service.manage_delete(meta, by="retention")
                deleted.append(number)
        finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = {"started_at": started, "finished_at": finished, "deleted_count": len(deleted),
                  "deleted_cases": sorted(deleted)}
        service.storage.put_json(LAST_SWEEP_KEY, result)
    log.info("Teach retention sweep completed: %d case(s) removed", len(deleted))
    return result


class RetentionManager:
    def __init__(self, service):
        self.service = service
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def maybe_start(self) -> bool:
        with self._lock:
            if self._running:
                return False
            now = datetime.now(timezone.utc)
            last = self.service.storage.get_json(LAST_SWEEP_KEY, None)
            finished = _parse_time((last or {}).get("finished_at"))
            if finished and now - finished < SWEEP_INTERVAL:
                return False
            self._running = True
            self._thread = threading.Thread(target=self._background, daemon=True, name="teach-retention-sweep")
            self._thread.start()
            return True

    def _background(self):
        try:
            run_sweep(self.service)
        except Exception:
            log.exception("Teach retention sweep failed")
        finally:
            with self._lock:
                self._running = False

    def wait(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
        if thread:
            thread.join(timeout)

    def run_now(self) -> dict | None:
        with self._lock:
            if self._running:
                return None
            self._running = True
        try:
            return run_sweep(self.service)
        finally:
            with self._lock:
                self._running = False
