"""Admin-only review routes, nested under the enabled Teach blueprint."""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
import threading
import time
from collections import deque
from pathlib import Path

from flask import abort, jsonify, render_template, request

from . import limits

log = logging.getLogger("teach")
LOCKOUT_S = 15 * 60


class AdminAuth:
    def __init__(self, token, quota, clock=time.monotonic):
        self.token = token
        self.quota = quota
        self.clock = clock
        self.failures = {}
        self.locked = {}
        self.lock = threading.Lock()

    def check(self, ip, header):
        key = self.quota.ip_key(ip)
        with self.lock:
            now = self.clock()
            until = self.locked.get(key, 0)
            if until > now:
                return 429, max(1, int(until - now))
            self.locked.pop(key, None)
            attempts = self.failures.setdefault(key, deque())
            while attempts and attempts[0] <= now - LOCKOUT_S:
                attempts.popleft()
            provided = header[7:] if header and header.startswith("Bearer ") else ""
            valid = secrets.compare_digest(provided.encode("utf-8"), self.token.encode("utf-8"))
            if valid:
                return 200, 0
            attempts.append(now)
            log.warning("Teach admin authentication failed (visitor %s)", key[:8])
            if len(attempts) >= 5:
                self.locked[key] = now + LOCKOUT_S
                attempts.clear()
                return 429, LOCKOUT_S
            return 401, 0


def masked_ip(value):
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return "unavailable"
    if ip.version == 4:
        return ".".join(value.split(".")[:2]) + ".*.*"
    return str(ipaddress.ip_network(f"{ip}/32", strict=False).network_address) + "/32 (masked)"


def memory_usage(pid=None):
    pid = pid or os.getpid()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "rss", "peak_paged", "paged",
                                                    "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.rss
            return None
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError):
        return None
    return None


def register_admin(bp, svc, windows, environment):
    cfg = svc.cfg
    auth = AdminAuth(cfg.admin_token, svc.quota)
    svc.admin_auth = auth

    def client():
        return limits.client_ip(request.remote_addr, request.headers.get("X-Forwarded-For"), cfg.trusted_proxy_hops)

    from .rescore import RescoreManager
    from .retention import RetentionManager, retention_status

    rescores = RescoreManager(svc)
    svc.rescores = rescores
    retention = RetentionManager(svc)
    svc.retention = retention
    from .review import register_review

    register_review(bp, svc, rescores, client)

    @bp.before_request
    def protect_admin():
        if not request.path.startswith(("/teach/admin", "/teach/api/admin/")):
            return None
        forwarded = [p.strip().lower() for p in request.headers.get("X-Forwarded-Proto", "").split(",")]
        proxy_https = cfg.trusted_proxy_hops > 0 and len(forwarded) >= cfg.trusted_proxy_hops \
            and forwarded[-cfg.trusted_proxy_hops] == "https"
        local = cfg.local_mode and request.host.split(":")[0] in ("localhost", "127.0.0.1") \
            and request.remote_addr in ("127.0.0.1", "::1")
        if not (request.is_secure or proxy_https or local):
            abort(403)
        if request.path.startswith("/teach/api/admin/"):
            status, retry = auth.check(client(), request.headers.get("Authorization"))
            if status != 200:
                resp = jsonify({"ok": False, "error": "Admin locked; try again later." if retry else "Sign in required."})
                resp.status_code = status
                if retry:
                    resp.headers["Retry-After"] = str(retry)
                return resp
            if request.path != "/teach/api/admin/retention-sweep":
                retention.maybe_start()
        return None

    @bp.get("/admin")
    @bp.get("/admin/")
    @bp.get("/admin/environment")
    def admin_page():
        return render_template("teach/admin.html", local_mode=cfg.local_mode)

    @bp.get("/api/admin/environment")
    def admin_environment():
        xff = [e.strip() for e in request.headers.get("X-Forwarded-For", "").split(",") if e.strip()]
        proto = [p.strip().lower() for p in request.headers.get("X-Forwarded-Proto", "").split(",")]
        sha = environment.get("RAILWAY_GIT_COMMIT_SHA")
        return jsonify({"ok": True, "scheme": request.scheme,
                        "forwarded_proto": [p if p in ("http", "https") else "invalid" for p in proto if p],
                        "forwarded_entries": len(xff), "client_ip": masked_ip(client()),
                        "trusted_hops": cfg.trusted_proxy_hops,
                        "proxy_hint": "Railway alone = 1; Cloudflare in front = 2. Confirm on this deployment.",
                        "railway_commit_present": bool(sha), "railway_commit": sha or None,
                        "storage": svc.storage.kind, "public_base_url_set": bool(cfg.public_base_url),
                        "alerts_configured": svc.notifier.configured,
                        "admin_token_present": True, "admin_token_length": len(cfg.admin_token),
                        "web_memory_bytes": memory_usage(),
                        "worker_memory_bytes": memory_usage(svc.queue.pid) if getattr(svc.queue, "pid", None) else None,
                        "scoring_queue": getattr(svc.queue, "depth", 0),
                        "quotas": {"hourly": windows.snapshot(), "daily": svc.quota.snapshot()},
                        "retention": retention_status(svc)})

    @bp.post("/api/admin/storage-check")
    def admin_storage_check():
        key = "teach/diagnostics/" + secrets.token_hex(12) + ".txt"
        try:
            svc.storage.put(key, b"Teach storage check", "text/plain")
            if svc.storage.get(key) != b"Teach storage check":
                raise OSError("storage read mismatch")
            svc.storage.delete(key)
            if svc.storage.get(key) is not None:
                raise OSError("storage delete mismatch")
        except Exception as exc:
            log.error("Teach storage check failed (%s)", type(exc).__name__)
            try:
                svc.storage.delete(key)
            except Exception as exc:
                log.error("Teach storage check cleanup failed (%s)", type(exc).__name__)
            return jsonify({"ok": False, "error": "Storage write/read/delete check failed."}), 503
        return jsonify({"ok": True})

    @bp.post("/api/admin/test-alert")
    def admin_test_alert():
        if not svc.notifier.configured:
            return jsonify({"ok": False, "error": "Alerts are not configured."}), 409
        svc.notifier.send_test()
        return jsonify({"ok": True, "message": "Test alert queued; check the topic and server logs for delivery."}), 202

    @bp.post("/api/admin/rescore-all")
    def admin_rescore_all():
        job = rescores.start_all(client())
        if job is None:
            return jsonify({"ok": False, "error": "A re-score job is already running."}), 409
        return jsonify({"ok": True, "job": job}), 202

    @bp.post("/api/admin/retention-sweep")
    def admin_retention_sweep():
        result = retention.run_now()
        if result is None:
            return jsonify({"ok": False, "error": "A retention sweep is already running."}), 409
        return jsonify({"ok": True, "result": result})

    @bp.get("/api/admin/jobs/<job_id>")
    def admin_job(job_id):
        job = rescores.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found."}), 404
        return jsonify({"ok": True, "job": job})
