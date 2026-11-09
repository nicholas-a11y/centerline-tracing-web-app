"""Teach public routes (design 14.3), all under /teach. Admin routes are Stage 6."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, Response, abort, jsonify, render_template, request

from . import limits, schema
from .limits import QuotaExceeded, RateLimited
from .notify import Notifier
from .scoring_worker import Busy, ScoringQueue
from .service import TeachService, data_url, public_result
from .validate import Problem, Problems, validate_submission

log = logging.getLogger("teach")
HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE / "example"
SPEC_PATH = HERE / "spec.md"

CSP = "default-src 'self'; img-src 'self' data:; script-src 'self'; object-src 'none'; base-uri 'none'; " \
      "frame-ancestors 'none'; form-action 'self'"
NOINDEX_ENDPOINTS = {"teach.queue_page", "teach.item_page", "teach.manage_page"}
EXAMPLE_FILES = {"image.png": "image/png", "reference.svg": "image/svg+xml", "request.json": "application/json"}


def _response(ok: bool, errors=(), warnings=(), result=None, status: int = 200, **extra):
    body = {"ok": ok, "errors": [p.as_dict() for p in errors], "warnings": [p.as_dict() for p in warnings],
            "result": result}
    body.update(extra)
    return jsonify(body), status


def _problem(status: int, field: str, code: str, message: str, hint: str = "", retry_after: int | None = None, **extra):
    resp, _ = _response(False, [Problem(field, code, message, hint)], status=status, **extra)
    resp.status_code = status
    if retry_after is not None:
        resp.headers["Retry-After"] = str(int(retry_after))
    return resp


def llms_txt_view():
    base = request.url_root.rstrip("/")
    text = ("# Centerline tracing web app\n\n"
            "Teach collects hard tracing cases (a bitmap plus a hand-traced reference SVG) to improve the engine.\n\n"
            f"- Spec: {base}/teach/spec\n- OpenAPI: {base}/teach/openapi.json\n")
    return Response(text, mimetype="text/plain")


def create_blueprint(cfg, storage, scorer=None, alert_sender=None, notifier=None, queue=None) -> Blueprint:
    bp = Blueprint("teach", __name__, url_prefix="/teach", template_folder="templates",
                   static_folder="static", static_url_path="/static")
    windows = limits.SlidingWindows()
    quota = limits.DailyQuota(storage, limits.quota_secret(storage))
    notifier = notifier or Notifier(cfg.alert_url, cfg.public_base_url, sender=alert_sender)
    queue = queue or scorer or ScoringQueue()
    svc = TeachService(cfg, storage, queue, notifier, quota)
    bp.teach_service = svc  # for tests

    def ip() -> str:
        return limits.client_ip(request.remote_addr, request.headers.get("X-Forwarded-For"), cfg.trusted_proxy_hops)

    def base_url() -> str:
        return cfg.public_base_url or request.url_root.rstrip("/")

    @bp.before_request
    def _body_limit():
        request.max_content_length = schema.MAX_BODY_BYTES

    @bp.after_request
    def _headers(resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.endpoint in NOINDEX_ENDPOINTS:
            resp.headers["X-Robots-Tag"] = "noindex"
        if request.path.startswith("/teach/api/") or request.endpoint == "teach.manage_page":
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @bp.errorhandler(413)
    def _too_large(_exc):
        return _problem(413, "request", "request_too_large",
                        f"Request body is too large (limit {schema.MAX_BODY_BYTES} bytes).",
                        "Crop the bitmap or remove the embedded image's extra resolution.")

    def _limited(exc: RateLimited):
        return _problem(429, "request", "rate_limited", "Too many requests; please wait and try again.",
                        f"Retry after {exc.retry_after_s} s.", retry_after=exc.retry_after_s,
                        retry_after_s=exc.retry_after_s)

    def _busy(exc: Busy):
        msg = ("You already have a scoring job running or waiting." if exc.reason == "ip_busy"
               else "The scoring queue is full.")
        return _problem(503, "request", "busy", msg, "Retry after 30 s.", retry_after=30, retry_after_s=30)

    def _quota(exc: QuotaExceeded):
        if exc.scope == "global":
            notifier.urgent("global_cap")
        return _problem(429, "request", "quota_exceeded", exc.message, "Validation keeps working.")

    def _files():
        img = request.files.get("image")
        ref = request.files.get("reference")
        return (img.read() if img else None), (ref.read() if ref else None)

    # ------------------------------------------------------------------ pages and docs

    @bp.get("")
    @bp.get("/")
    def index():
        return render_template("teach/index.html", rules=schema.RULES, consent=schema.CONSENT_TEXT,
                               advice=schema.ADVICE_TEXT, settings=schema.SETTINGS, setting_ui=schema.SETTING_UI,
                               settings_line=schema.settings_summary(schema.default_settings()),
                               text_fields=schema.TEXT_FIELDS,
                               spec=SPEC_PATH.read_text(encoding="utf-8"), local_mode=cfg.local_mode,
                               limits=schema)

    @bp.get("/spec")
    def spec():
        return Response(SPEC_PATH.read_text(encoding="utf-8"), mimetype="text/markdown", content_type="text/markdown; charset=utf-8")

    @bp.get("/openapi.json")
    def openapi():
        return jsonify(schema.openapi())

    @bp.get("/schema/submission.json")
    def submission_schema():
        return jsonify(schema.submission_meta_schema())

    @bp.get("/example/<name>")
    def example(name):
        if name not in EXAMPLE_FILES:
            abort(404)
        resp = Response((EXAMPLE_DIR / name).read_bytes(), mimetype=EXAMPLE_FILES[name])
        resp.headers["Content-Disposition"] = f'attachment; filename="{name}"'
        return resp

    @bp.get("/queue")
    def queue_page():
        q = (request.args.get("q") or "").strip().lower()[:100]
        items = svc.public_items()
        if q:
            items = [e for e in items if q in (e.get("title") or "").lower() or q == str(e["number"])]
        return render_template("teach/queue.html", items=items, q=q, local_mode=cfg.local_mode)

    @bp.get("/q/<int:number>")
    def item_page(number):
        entry = svc.public_item(number)
        if entry is None:
            abort(404)
        return render_template("teach/item.html", item=entry, local_mode=cfg.local_mode)

    @bp.get("/q/<int:number>/thumb.png")
    def item_thumb(number):
        png = svc.thumbnail(number)
        if png is None:
            abort(404)
        return Response(png, mimetype="image/png")

    @bp.get("/manage/<sub_id>")
    def manage_page(sub_id):
        return render_template("teach/manage.html", sub_id=sub_id, local_mode=cfg.local_mode)

    # ------------------------------------------------------------------ public API

    @bp.get("/api/queue")
    def api_queue():
        return jsonify({"items": svc.public_items()})

    @bp.get("/api/submissions/<int:number>")
    def api_item(number):
        entry = svc.public_item(number)
        if entry is None:
            return _problem(404, "number", "not_found", f"#{number} does not exist.")
        return jsonify(entry)

    @bp.post("/api/validate")
    def api_validate():
        client = ip()
        try:
            windows.hit((f"validate:{client}",) + limits.VALIDATE_PER_IP, ("global:validate",) + limits.VALIDATE_GLOBAL)
        except RateLimited as exc:
            return _limited(exc)
        image, reference = _files()
        try:
            sub = validate_submission(image, reference, request.form.get("meta"))
        except Problems as exc:
            return _response(False, exc.problems)
        try:
            worker = svc.score(client, sub)
        except Busy as exc:
            return _busy(exc)
        overlay = worker.pop("overlay_png", None)
        worker.pop("thumb_png", None)
        post_overlay = worker.pop("post_fit_overlay_png", None)
        warnings = list(sub.warnings)
        if worker.get("outcome") == "timeout":
            warnings.append(Problem("request", "scoring_timeout", "Scoring did not finish within 30 s.",
                                    "Crop to the problem area."))
        return _response(True, warnings=warnings,
                         result=public_result(worker, sub.reference.registration, data_url(overlay),
                                              data_url(post_overlay)))

    @bp.post("/api/similar")
    def api_similar():
        try:
            windows.hit((f"similar:{ip()}",) + limits.SIMILAR_PER_IP)
        except RateLimited as exc:
            return _limited(exc)
        image, reference = _files()
        try:
            sub = validate_submission(image, reference, request.form.get("meta"))
        except Problems as exc:
            return _response(False, exc.problems)
        matches, exact = svc.similar(sub)
        return _response(True, warnings=sub.warnings, result={"matches": matches, "exact_duplicate": exact})

    @bp.post("/api/submissions")
    def api_submit():
        client = ip()
        try:
            windows.check(f"submit:{client}", *limits.SUBMIT_PER_IP_HOUR)
            quota.check(client)
        except RateLimited as exc:
            return _limited(exc)
        except QuotaExceeded as exc:
            return _quota(exc)
        image, reference = _files()
        try:
            sub = validate_submission(image, reference, request.form.get("meta"))
        except Problems as exc:
            resp, _ = _response(False, exc.problems)
            resp.status_code = 422
            return resp
        try:
            windows.hit((f"submit:{client}",) + limits.SUBMIT_PER_IP_HOUR)
            out = svc.submit(client, sub)
        except RateLimited as exc:
            return _limited(exc)
        except QuotaExceeded as exc:
            return _quota(exc)
        except Busy as exc:
            return _busy(exc)
        except Problems as exc:
            code = 409 if exc.problems[0].code == "duplicate_exact" else 422
            resp, _ = _response(False, exc.problems)
            resp.status_code = code
            return resp
        base = base_url()
        warnings = list(sub.warnings)
        if out["result"]["outcome"] == "timeout":
            warnings.append(Problem("request", "scoring_timeout", "Scoring did not finish within 30 s; the case was "
                                    "stored with outcome 'timeout'.", ""))
        for m in out["similar"]:
            warnings.append(Problem("image", "similar", f"Looks similar to #{m['number']}.", m["url"]))
        log.info("Teach submission #%s stored (ip %s)", out["number"], limits.log_ip_hash(client))
        body = {"ok": True, "errors": [], "warnings": [p.as_dict() for p in warnings], "number": out["number"],
                "status": out["status"], "queue_url": f"{base}/teach/q/{out['number']}",
                "manage_url": f"{base}/teach/manage/{out['id']}#k={out['token']}", "result": out["result"]}
        return jsonify(body), 201

    @bp.post("/api/submissions/<int:number>/me-too")
    def api_me_too(number):
        client = ip()
        try:
            windows.check(f"submit:{client}", *limits.SUBMIT_PER_IP_HOUR)
            quota.check(client)
        except RateLimited as exc:
            return _limited(exc)
        except QuotaExceeded as exc:
            return _quota(exc)
        meta_raw = request.form.get("meta")
        try:
            from .validate import parse_meta_json

            meta = parse_meta_json(meta_raw) if meta_raw else {}
            unknown = set(meta) - {"note", "consent"}
            if unknown:
                raise Problems([Problem(sorted(unknown)[0], "meta_invalid", "Allowed fields: note, consent.", "")])
            note = meta.get("note") or ""
            if not isinstance(note, str):
                raise Problems([Problem("note", "meta_invalid", "'note' must be a string.", "")])
            image, reference = _files()
            windows.hit((f"submit:{client}",) + limits.SUBMIT_PER_IP_HOUR)
            out = svc.me_too(client, number, note, image, reference, meta.get("consent") is True)
        except RateLimited as exc:
            return _limited(exc)
        except QuotaExceeded as exc:
            return _quota(exc)
        except Problems as exc:
            code = 404 if exc.problems[0].code == "not_found" else 422
            resp, _ = _response(False, exc.problems)
            resp.status_code = code
            return resp
        return jsonify({"ok": True, "errors": [], "warnings": [], "result": out}), 201

    # ------------------------------------------------------------------ manage link (13.3)

    def _manage_meta(sub_id):
        windows.hit((f"manage:{ip()}",) + limits.MANAGE_PER_IP)
        meta = svc.check_manage(sub_id, request.headers.get("X-Teach-Manage-Token"))
        if meta is None:
            raise Problems([Problem("token", "manage_token_invalid",
                                    "This manage link is not valid, or the item was deleted.", "")])
        return meta

    @bp.route("/api/manage/<sub_id>", methods=["GET", "PATCH", "DELETE"])
    def api_manage(sub_id):
        try:
            meta = _manage_meta(sub_id)
            if request.method == "GET":
                return jsonify({"ok": True, "errors": [], "warnings": [], "result": svc.manage_view(meta)})
            if request.method == "DELETE":
                return jsonify({"ok": True, "errors": [], "warnings": [], "result": svc.manage_delete(meta)})
            note, reference = _patch_inputs()
            return jsonify({"ok": True, "errors": [], "warnings": [],
                            "result": svc.manage_answer(ip(), meta, note, reference)})
        except RateLimited as exc:
            return _limited(exc)
        except Busy as exc:
            return _busy(exc)
        except Problems as exc:
            code = {"manage_token_invalid": 404, "invalid_state": 409}.get(exc.problems[0].code, 422)
            resp, _ = _response(False, exc.problems)
            resp.status_code = code
            return resp

    def _patch_inputs():
        if request.is_json:
            body = request.get_json(silent=True) or {}
            return str(body.get("note") or ""), None
        ref = request.files.get("reference")
        return request.form.get("note") or "", (ref.read() if ref else None)

    return bp
