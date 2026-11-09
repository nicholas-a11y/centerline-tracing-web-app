"""Admin case reads and validated review updates; all writes use the service lock."""

import math

from flask import Response, abort, jsonify, render_template, request

from . import schema
from .sanitize import clean_text
from .service import data_url, utc_now
from .storage import submission_key
from .validate import validate_focus_box


def case_for_number(svc, number):
    sid = svc.id_for_number(number)
    meta = svc.load_meta(sid) if sid else None
    if not meta or meta["status"] == "deleted":
        abort(404)
    return meta


def settings_label(params):
    values = schema.default_settings() | (params or {})
    if values == schema.default_settings():
        return "Defaults"
    return "Post-fit on" if values["enable_post_fit_export"] else "Custom settings"


def case_view(svc, meta):
    out = {k: v for k, v in meta.items() if k not in ("manage_token_sha256", "text_tokens")}
    out["params"] = schema.default_settings() | (meta.get("params") or {})
    out["settings_label"] = settings_label(out["params"])
    out["overlay_png"] = data_url(svc.storage.get(submission_key(meta["id"], "overlay.png")))
    out["post_fit_overlay_png"] = data_url(svc.storage.get(submission_key(meta["id"], "overlay_post_fit.png")))
    out["bitmap_png"] = data_url(svc.storage.get(submission_key(meta["id"], "image.png")))
    out["me_too_count"] = len(meta.get("me_too") or [])
    return out


def review_update(svc, number, data):
    allowed = {"status", "admin_note_public", "admin_note_private", "tolerance_px", "region",
               "title_hidden", "thumbnail_public", "bitmap_public"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Unknown review field.")
    with svc._lock:
        meta = case_for_number(svc, number)
        status = data.get("status", meta["status"])
        if status not in schema.STATUSES:
            raise ValueError("Choose Awaiting review, Accepted or Rejected.")
        for name in ("admin_note_public", "admin_note_private"):
            if name in data:
                if not isinstance(data[name], str) or len(data[name]) > 4000:
                    raise ValueError("Notes must be text of at most 4000 characters.")
                meta[name] = clean_text(data[name])
        vis = meta["visibility"]
        for name in ("title_hidden", "thumbnail_public", "bitmap_public"):
            if name not in data:
                continue
            if not isinstance(data[name], bool):
                raise ValueError("Visibility values must be booleans.")
            if name == "title_hidden":
                meta[name] = data[name]
            else:
                opt = "thumbnail_opt_in" if name == "thumbnail_public" else "bitmap_opt_in"
                if data[name] and not vis.get(opt):
                    raise ValueError("The submitter did not consent to that visibility.")
                vis[name] = data[name]
        scoring = meta["scoring"]
        if "tolerance_px" in data:
            v = data["tolerance_px"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.05 <= v <= 100:
                raise ValueError("Tolerance must be between 0.05 and 100 px.")
            scoring["tolerance_px"] = float(v)
            scoring["set_by"] = "admin"
        if "region" in data:
            region, errors = validate_focus_box(data["region"], meta["image"]["width"], meta["image"]["height"])
            if errors:
                raise ValueError(errors[0].message)
            scoring["region"] = region
            scoring["set_by"] = "admin"
        if status != meta["status"]:
            now = utc_now()
            meta["status_history"].append({"status": status, "at": now, "by": "admin"})
            meta["retention"] = {"status_since": now}
        meta["status"] = status
        # Hiding a title alone does not constitute review.
        if set(data) != {"title_hidden"}:
            meta["reviewed"] = True
        svc._save(meta)
    return meta


def register_review(bp, svc):
    @bp.get("/admin/cases/<int:number>")
    def admin_case_page(number):
        return render_template("teach/admin.html", local_mode=svc.cfg.local_mode)

    @bp.get("/api/admin/summary")
    @bp.get("/api/admin/cases")
    def admin_cases():
        items = []
        counts = {s: 0 for s in schema.STATUSES}
        for entry in svc.admin_index()["items"].values():
            if entry["status"] == "deleted":
                continue
            meta = svc.load_meta(entry["id"])
            if not meta:
                continue
            counts[meta["status"]] += 1
            last = (meta.get("results") or [{}])[-1]
            items.append({"number": meta["number"], "title": meta.get("title", ""),
                          "title_hidden": meta.get("title_hidden", False), "status": meta["status"],
                          "outcome": last.get("outcome", "unknown"), "engine_commit": last.get("engine_commit"),
                          "me_too_count": len(meta.get("me_too") or []), "merged": bool(meta.get("merged")),
                          "settings": settings_label(meta.get("params")), "created_at": meta.get("created_at")})
        status = request.args.get("status", "")
        if status and status not in schema.STATUSES:
            return jsonify({"ok": False, "error": "Unknown status filter."}), 422
        query = request.args.get("q", "").strip().lower()[:120].lstrip("#")
        total = len(items)
        items = [e for e in items if (not status or e["status"] == status)
                 and (not query or query in e["title"].lower() or query == str(e["number"]))]
        return jsonify({"ok": True, "items": sorted(items, key=lambda e: -e["number"]),
                        "counts": counts, "shown": len(items), "total": total,
                        "retention": svc.storage.get_json("teach/sweep/last.json", None)})

    @bp.route("/api/admin/cases/<int:number>", methods=["GET", "PATCH", "DELETE"])
    def admin_case(number):
        if request.method == "DELETE":
            return jsonify({"ok": True, "result": svc.manage_delete(case_for_number(svc, number), by="admin")})
        if request.method == "PATCH":
            try:
                meta = review_update(svc, number, request.get_json())
            except ValueError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 422
        else:
            meta = case_for_number(svc, number)
        return jsonify({"ok": True, "case": case_view(svc, meta)})

    @bp.get("/api/admin/cases/<int:number>/hand_traced.svg")
    def admin_hand_trace(number):
        meta = case_for_number(svc, number)
        raw = svc.storage.get(submission_key(meta["id"], "reference.raw.svg"))
        if raw is None:
            abort(404)
        resp = Response(raw, content_type="text/plain; charset=utf-8")
        resp.headers["Content-Disposition"] = f'attachment; filename="hand_traced-{number}.svg"'
        return resp

    @bp.post("/api/admin/rebuild-index")
    def admin_rebuild():
        with svc._lock:
            admin, public = {"schema_version": 1, "items": {}}, {"schema_version": 1, "items": {}}
            for key in svc.storage.list("teach/submissions/"):
                if key.endswith("/meta.json"):
                    meta = svc.load_meta(key.split("/")[2])
                    if meta:
                        admin["items"][meta["id"]] = svc.admin_entry(meta)
                        public["items"][str(meta["number"])] = svc.public_entry(meta)
            svc.storage.put_json("teach/index/admin.json", admin)
            svc.storage.put_json("teach/index/public.json", public)
        return jsonify({"ok": True, "items": len(admin["items"])})
