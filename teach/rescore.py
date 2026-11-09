"""Background admin rescoring; expensive scoring never runs in a Flask request thread."""

from __future__ import annotations

import logging
import secrets
import threading

from .scoring_worker import engine_commit

log = logging.getLogger("teach")


def history_label(results):
    results = list(results or [])
    if not results:
        return "Not scored"
    if results[-1].get("outcome") == "pass":
        if any(r.get("outcome") == "fail" for r in results[:-1]):
            return "Now passing"
        return "Passing"
    if results[-1].get("outcome") == "fail":
        first = len(results) - 1
        while first > 0 and results[first - 1].get("outcome") == "fail":
            first -= 1
        return "Failing since " + str(results[first].get("engine_commit") or "unknown")
    return "Not scored"


class RescoreManager:
    def __init__(self, service):
        self.service = service
        self._lock = threading.Lock()
        self._current = None
        self._jobs = {}

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _start(self, mode, ip, numbers, skipped=0, total=None):
        with self._lock:
            if self._current and not self._jobs[self._current]["done"]:
                return None
            job_id = secrets.token_urlsafe(18)
            job = {"id": job_id, "mode": mode, "done": False,
                   "total": len(numbers) + skipped if total is None else total, "processed": 0,
                   "skipped": skipped, "failed": 0, "current_number": None, "message": "Queued"}
            self._jobs[job_id] = job
            self._current = job_id
            while len(self._jobs) > 10:
                oldest = next(iter(self._jobs))
                if oldest == self._current:
                    break
                self._jobs.pop(oldest)
            thread = threading.Thread(target=self._run, args=(job_id, ip, numbers), daemon=True,
                                      name="teach-admin-rescore")
            thread.start()
            return dict(job)

    def start_one(self, number, ip):
        return self._start("one", ip, [int(number)])

    def start_all(self, ip):
        current = engine_commit()
        numbers = []
        skipped = 0
        total = 0
        for item in self.service.admin_index()["items"].values():
            if item.get("status") == "deleted":
                continue
            meta = self.service.load_meta(item["id"])
            if meta and meta.get("status") != "deleted":
                total += 1
                latest = (meta.get("results") or [{}])[-1]
                if latest.get("engine_commit") != current:
                    numbers.append(int(meta["number"]))
                else:
                    skipped += 1
        return self._start("all", ip, numbers, skipped=skipped, total=total)

    def _run(self, job_id, ip, numbers):
        for number in numbers:
            with self._lock:
                job = self._jobs[job_id]
                job["current_number"] = number
                job["message"] = f"Scoring case #{number}"
            try:
                self.service.rescore(ip, number)
            except Exception as exc:
                log.exception("Teach admin re-score failed for case #%d", number)
                with self._lock:
                    self._jobs[job_id]["failed"] += 1
                    self._jobs[job_id]["last_error"] = type(exc).__name__
            finally:
                with self._lock:
                    self._jobs[job_id]["processed"] += 1
        with self._lock:
            job = self._jobs[job_id]
            job["current_number"] = None
            job["done"] = True
            job["message"] = "Complete" if not job["failed"] else "Complete with errors"
