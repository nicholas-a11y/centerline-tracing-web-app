"""Teach service: submissions, indexes, visibility and manage actions (design 6, 7, 10, 11, 13.3).

Everything that touches storage goes through here so the blueprint stays thin. All
writers are serialised by one in-process lock (single gunicorn worker; design 7.3).
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import string
import threading
from datetime import datetime, timezone

from . import schema
from .sanitize import clean_text, sanitize_title
from .similarity import compare, phash, tokens
from .storage import PREFIX, submission_key
from .validate import Problem, Problems, validate_bitmap, validate_reference

ID_ALPHABET = string.ascii_lowercase + string.digits
ID_LEN = 12
ADMIN_INDEX = PREFIX + "index/admin.json"
PUBLIC_INDEX = PREFIX + "index/public.json"
COUNTER = PREFIX + "counter.json"
MAX_ME_TOO = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id() -> str:
    return "t_" + "".join(secrets.choice(ID_ALPHABET) for _ in range(ID_LEN))


def valid_id(sub_id: str) -> bool:
    return (isinstance(sub_id, str) and len(sub_id) == ID_LEN + 2 and sub_id.startswith("t_")
            and all(c in ID_ALPHABET for c in sub_id[2:]))


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def data_url(png: bytes | None) -> str | None:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii") if png else None


def public_result(worker: dict, registration: str | None, overlay_url: str | None) -> dict:
    """The ``result`` block of a validate/submit response (design 14.4)."""
    primary = (worker.get("variants") or {}).get(worker.get("variant") or "prune_off") or {}
    others = {k: v.get("summary") for k, v in (worker.get("variants") or {}).items() if k != "prune_off"}
    return {
        "outcome": worker.get("outcome"),
        "passed": worker.get("outcome") == "pass",
        "variant": "prune_off",
        "summary": worker.get("summary"),
        "lines": [{k: ln.get(k) for k in ("index", "best_single_coverage", "any_coverage", "fragments", "passed",
                                          "reason")} | {"p95_px": (ln.get("ref_to_best") or {}).get("p95")}
                  for ln in primary.get("lines", [])],
        "other_variants": others,
        "engine_commit": worker.get("engine_commit"),
        "elapsed_ms": worker.get("elapsed_ms"),
        "timed_out": bool(worker.get("timed_out")),
        "error": worker.get("error"),
        "registration": registration,
        "overlay_png": overlay_url,
    }


def result_entry(worker: dict) -> dict:
    """One ``results[]`` entry in meta.json (design 6.2)."""
    return {"engine_commit": worker.get("engine_commit"), "at": utc_now(), "outcome": worker.get("outcome"),
            "variant": "prune_off", "summary": worker.get("summary"), "metrics": worker.get("variants") or {},
            "elapsed_ms": worker.get("elapsed_ms"), "timed_out": bool(worker.get("timed_out")),
            "error": worker.get("error")}


class TeachService:
    def __init__(self, cfg, storage, queue, notifier, quota):
        self.cfg = cfg
        self.storage = storage
        self.queue = queue
        self.notifier = notifier
        self.quota = quota
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ storage helpers

    def _next_number(self) -> int:
        counter = self.storage.get_json(COUNTER, None) or {"next": 1}
        number = int(counter["next"])
        self.storage.put_json(COUNTER, {"next": number + 1})
        return number

    def admin_index(self) -> dict:
        return self.storage.get_json(ADMIN_INDEX, None) or {"schema_version": 1, "items": {}}

    def public_index(self) -> dict:
        return self.storage.get_json(PUBLIC_INDEX, None) or {"schema_version": 1, "items": {}}

    def load_meta(self, sub_id: str) -> dict | None:
        if not valid_id(sub_id):
            return None
        return self.storage.get_json(submission_key(sub_id, "meta.json"), None)

    def id_for_number(self, number: int) -> str | None:
        for sub_id, entry in self.admin_index()["items"].items():
            if entry.get("number") == number:
                return sub_id
        return None

    def _save(self, meta: dict) -> None:
        """Write meta.json and update both indexes incrementally (design 7.2)."""
        meta["updated_at"] = utc_now()
        self.storage.put_json(submission_key(meta["id"], "meta.json"), meta)
        admin = self.admin_index()
        admin["items"][meta["id"]] = self.admin_entry(meta)
        self.storage.put_json(ADMIN_INDEX, admin)
        public = self.public_index()
        public["items"][str(meta["number"])] = self.public_entry(meta)
        self.storage.put_json(PUBLIC_INDEX, public)

    # ------------------------------------------------------------------ index builders (11)

    @staticmethod
    def admin_entry(meta: dict) -> dict:
        deleted = meta["status"] == "deleted"
        img = meta.get("image") or {}
        return {
            "id": meta["id"], "number": meta["number"], "status": meta["status"], "reviewed": meta.get("reviewed", False),
            "created_at": meta.get("created_at"), "updated_at": meta.get("updated_at"),
            "title_public": meta.get("title_public"), "title_hidden": meta.get("title_hidden", False),
            "me_too_count": len(meta.get("me_too") or []),
            "image_sha256": None if deleted else img.get("sha256"), "phash": None if deleted else img.get("phash"),
            "width": img.get("width"), "height": img.get("height"),
            "geometry_sha256": None if deleted else (meta.get("reference") or {}).get("sha256_geometry"),
            "tokens": [] if deleted else meta.get("text_tokens", []),
            "retention_status_since": (meta.get("retention") or {}).get("status_since"),
        }

    @staticmethod
    def public_entry(meta: dict) -> dict:
        """The only builder of public data: queue, item API/page, duplicate check and alerts use it."""
        number = meta["number"]
        if meta["status"] == "deleted":
            return {"number": number, "status": "deleted"}
        hidden = bool(meta.get("title_hidden"))
        entry = {
            "number": number,
            "status": meta["status"],
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "me_too_count": len(meta.get("me_too") or []),
            "title": None if hidden else meta.get("title_public"),
            "title_hidden": hidden,
            "reviewed": bool(meta.get("reviewed")),
            "url": f"/teach/q/{number}",
        }
        if meta.get("reviewed"):
            vis = meta.get("visibility") or {}
            entry["what_is_wrong"] = meta.get("what_is_wrong", "")
            entry["what_is_expected"] = meta.get("what_is_expected", "")
            entry["notes"] = meta.get("notes", "")
            entry["admin_note_public"] = meta.get("admin_note_public", "")
            if vis.get("thumbnail_opt_in") and vis.get("thumbnail_public"):
                entry["thumbnail_url"] = f"/teach/q/{number}/thumb.png"
        else:
            entry["details"] = "awaiting review"
        return entry

    def public_items(self) -> list[dict]:
        items = [e for e in self.public_index()["items"].values() if e.get("status") != "deleted"]
        return sorted(items, key=lambda e: -e["number"])

    def public_item(self, number: int) -> dict | None:
        return self.public_index()["items"].get(str(number))

    def thumbnail(self, number: int) -> bytes | None:
        entry = self.public_item(number)
        if not entry or "thumbnail_url" not in entry:
            return None
        sub_id = self.id_for_number(number)
        return self.storage.get(submission_key(sub_id, "thumb.png")) if sub_id else None

    # ------------------------------------------------------------------ similarity (10)

    def _signature(self, sub) -> dict:
        return {"image_sha256": sub.bitmap.sha256, "geometry_sha256": sub.reference.sha256_geometry,
                "phash": phash(sub.bitmap.gray), "width": sub.bitmap.width, "height": sub.bitmap.height,
                "tokens": tokens(sub.fields.get("title", ""), sub.fields.get("what_is_wrong", "")),
                "paths": sub.reference.paths}

    def _load_paths(self, entry: dict):
        doc = self.storage.get_json(submission_key(entry["id"], "reference.json"), None)
        return doc["paths"] if doc else None

    def similar(self, sub, signature: dict | None = None) -> tuple[list[dict], bool]:
        """Matches shown to the user, and whether any is an exact duplicate (blocked)."""
        sig = signature or self._signature(sub)
        matches, exact = [], False
        public = self.public_index()["items"]
        for entry in self.admin_index()["items"].values():
            if entry["status"] == "deleted":
                continue
            signals = compare(sig, entry, self._load_paths)
            if not signals:
                continue
            exact = exact or signals["exact"]
            pub = public.get(str(entry["number"])) or {}
            matches.append({"number": entry["number"], "url": f"/teach/q/{entry['number']}",
                            "status": pub.get("status"), "title": pub.get("title"),
                            "title_hidden": pub.get("title_hidden", False),
                            "details": None if pub.get("reviewed") else "pending review, details hidden",
                            "exact": signals["exact"], "signals": {k: signals[k] for k in ("image", "geometry", "text")}})
        matches.sort(key=lambda m: (not m["exact"], -m["number"]))
        return matches[:10], exact

    # ------------------------------------------------------------------ scoring

    def score(self, ip: str, sub) -> dict:
        return self.queue.run(ip, sub.bitmap.png, sub.reference.as_json(), sub.params, sub.scoring)

    # ------------------------------------------------------------------ submit (5, 6.2)

    def submit(self, ip: str, sub) -> dict:
        sig = self._signature(sub)
        matches, exact = self.similar(sub, sig)
        if exact:
            raise Problems([Problem("image", "duplicate_exact",
                                    f"This exact bitmap and drawing are already #{matches[0]['number']}.",
                                    f"Use \"me too\" on /teach/q/{matches[0]['number']} instead.")])
        raw_bytes = len(sub.bitmap.png) + len(sub.reference.raw)
        self.quota.check(ip, raw_bytes)
        worker = self.score(ip, sub)
        overlay, thumb = worker.pop("overlay_png", None), worker.pop("thumb_png", None)
        ref_json = json.dumps(sub.reference.as_json()).encode("utf-8")
        total = raw_bytes + len(ref_json) + len(overlay or b"") + len(thumb or b"")
        token = secrets.token_urlsafe(32)
        now = utc_now()
        f = sub.fields
        with self._lock:
            counts = self.quota.record(ip, total)
            number = self._next_number()
            sub_id = new_id()
            while self.storage.get(submission_key(sub_id, "meta.json")) is not None:
                sub_id = new_id()
            meta = {
                "schema_version": 1, "id": sub_id, "number": number, "created_at": now, "status": "pending_review",
                "status_history": [{"status": "submitted", "at": now, "by": "server"},
                                   {"status": "pending_review", "at": utc_now(), "by": "server"}],
                "reviewed": False,
                "title": f["title"], "what_is_wrong": f["what_is_wrong"], "what_is_expected": f["what_is_expected"],
                "notes": f.get("notes", ""), "not_duplicate_reason": f.get("not_duplicate_reason", ""),
                "similar_at_submit": [{"number": m["number"], "signals": m["signals"]} for m in matches],
                "title_public": sanitize_title(f["title"]), "title_hidden": False,
                "admin_note_public": "", "admin_note_private": "",
                "visibility": {"thumbnail_opt_in": f["public_thumbnail"], "bitmap_opt_in": f["share_bitmap"],
                               "thumbnail_public": f["public_thumbnail"], "bitmap_public": False},
                "consent": {"licence": schema.LICENCE, "text_version": schema.CONSENT_TEXT_VERSION, "accepted_at": now},
                "image": {"width": sub.bitmap.width, "height": sub.bitmap.height, "sha256": sub.bitmap.sha256,
                          "phash": sig["phash"]},
                "reference": {"paths": len(sub.reference.paths), "points": sub.reference.points,
                              "sha256_geometry": sub.reference.sha256_geometry,
                              "registration": sub.reference.registration},
                "params": sub.params,
                "scoring": {"region": sub.region, "tolerance_px": 1.0,
                            "set_by": "submitter_focus_box" if sub.region else "default"},
                "results": [result_entry(worker)],
                "me_too": [],
                "manage_token_sha256": token_hash(token),
                "lesson": {"lesson_id": None, "exported_at": None, "merged_commit": None,
                           "private_copies_deleted_at": None},
                "retention": {"status_since": now},
                "text_tokens": sig["tokens"],
                "quota_bytes": total,
            }
            put = self.storage.put
            put(submission_key(sub_id, "image.png"), sub.bitmap.png, "image/png")
            put(submission_key(sub_id, "reference.raw.svg"), sub.reference.raw, "text/plain")
            put(submission_key(sub_id, "reference.json"), ref_json, "application/json")
            if overlay:
                put(submission_key(sub_id, "overlay.png"), overlay, "image/png")
            if thumb:
                put(submission_key(sub_id, "thumb.png"), thumb, "image/png")
            self._save(meta)
        entry = self.public_entry(meta)
        self.notifier.event(number, meta["status"], entry.get("title"))
        if counts["submissions"] >= _global_cap() or self.quota.global_closed():
            self.notifier.urgent("global_cap")
        return {"number": number, "id": sub_id, "token": token, "status": meta["status"],
                "result": public_result(worker, sub.reference.registration, data_url(overlay)),
                "similar": matches}

    # ------------------------------------------------------------------ me too (10)

    def me_too(self, ip: str, number: int, note: str, image: bytes | None, reference: bytes | None,
               consent: bool) -> dict:
        sub_id = self.id_for_number(number)
        meta = self.load_meta(sub_id) if sub_id else None
        if not meta or meta["status"] == "deleted":
            raise Problems([Problem("number", "not_found", f"#{number} does not exist or was deleted.", "")])
        if len(meta.get("me_too") or []) >= MAX_ME_TOO:
            raise Problems([Problem("number", "quota_exceeded", f"#{number} has enough \"me too\" entries.", "")])
        note = clean_text(note or "")
        if len(note) > schema.ME_TOO_NOTE_MAX:
            raise Problems([Problem("note", "field_too_long", f"'note' is limited to {schema.ME_TOO_NOTE_MAX} characters.", "")])
        variant = None
        if image or reference:
            if not consent:
                raise Problems([Problem("consent", "consent_required",
                                        "Accept the consent and licence text (CC0 1.0) to add a variant.", "")])
            problems = []
            try:
                bm = validate_bitmap(image or b"", fld="image")
            except Problems as exc:
                raise Problems(exc.problems) from None
            try:
                ref = validate_reference(reference or b"", (bm.width, bm.height))
            except Problems as exc:
                problems.extend(exc.problems)
            if problems:
                raise Problems(problems)
            variant = (bm, ref)
        size = len(note.encode("utf-8")) + (len(variant[0].png) + len(variant[1].raw) if variant else 0)
        with self._lock:
            self.quota.record(ip, size)
            meta = self.load_meta(sub_id)
            entry = {"at": utc_now(), "note": note, "variant_key": None}
            if variant:
                key = f"v{len(meta['me_too']) + 1}"
                bm, ref = variant
                self.storage.put(submission_key(sub_id, f"variants/{key}/image.png"), bm.png, "image/png")
                self.storage.put(submission_key(sub_id, f"variants/{key}/reference.raw.svg"), ref.raw, "text/plain")
                self.storage.put_json(submission_key(sub_id, f"variants/{key}/reference.json"), ref.as_json())
                entry["variant_key"] = key
                entry["variant"] = {"width": bm.width, "height": bm.height, "sha256": bm.sha256,
                                    "registration": ref.registration}
            meta["me_too"].append(entry)
            meta["quota_bytes"] = int(meta.get("quota_bytes", 0)) + size
            self._save(meta)
        return {"number": number, "me_too_count": len(meta["me_too"])}

    # ------------------------------------------------------------------ manage link (13.3)

    def check_manage(self, sub_id: str, token: str | None) -> dict | None:
        meta = self.load_meta(sub_id)
        if not meta or meta["status"] == "deleted" or not token:
            return None
        stored = (meta.get("manage_token_sha256") or "").encode("ascii")
        given = token_hash(token).encode("ascii")
        return meta if secrets.compare_digest(stored, given) else None

    def manage_view(self, meta: dict) -> dict:
        overlay = self.storage.get(submission_key(meta["id"], "overlay.png"))
        keys = ("number", "status", "created_at", "updated_at", "title", "title_public", "title_hidden",
                "what_is_wrong", "what_is_expected", "notes", "admin_note_public", "visibility", "image",
                "reference", "params", "scoring", "results", "status_history")
        out = {k: meta.get(k) for k in keys}
        out["me_too_count"] = len(meta.get("me_too") or [])
        out["queue_url"] = f"/teach/q/{meta['number']}"
        out["overlay_png"] = data_url(overlay)
        out["can_answer"] = meta["status"] == "needs_info"
        return out

    def manage_delete(self, meta: dict) -> dict:
        with self._lock:
            sub_id = meta["id"]
            self.storage.delete_prefix(PREFIX + f"submissions/{sub_id}/", keep=(submission_key(sub_id, "meta.json"),))
            now = utc_now()
            tomb = {"schema_version": 1, "id": sub_id, "number": meta["number"], "created_at": meta.get("created_at"),
                    "status": "deleted", "status_history": [{"status": "deleted", "at": now, "by": "submitter"}],
                    "reviewed": meta.get("reviewed", False), "deleted_at": now, "retention": {"status_since": now}}
            self._save(tomb)
        return {"number": meta["number"], "status": "deleted"}

    def manage_answer(self, ip: str, meta: dict, note: str, reference: bytes | None) -> dict:
        if meta["status"] != "needs_info":
            raise Problems([Problem("status", "invalid_state",
                                    "Only items with status needs_info can be answered.", "")])
        note = clean_text(note or "")
        if not note and not reference:
            raise Problems([Problem("note", "field_missing", "Add a note or a replacement reference SVG.", "")])
        if len(note) > schema.MANAGE_ANSWER_MAX:
            raise Problems([Problem("note", "field_too_long",
                                    f"'note' is limited to {schema.MANAGE_ANSWER_MAX} characters.", "")])
        worker = ref = None
        if reference:
            img = meta.get("image") or {}
            ref = validate_reference(reference, (img["width"], img["height"]))
            png = self.storage.get(submission_key(meta["id"], "image.png"))
            worker = self.queue.run(ip, png, ref.as_json(), meta["params"], meta["scoring"])
        with self._lock:
            meta = self.load_meta(meta["id"])
            now = utc_now()
            meta.setdefault("answers", []).append({"at": now, "note": note, "replaced_reference": bool(ref)})
            if ref is not None:
                overlay, thumb = worker.pop("overlay_png", None), worker.pop("thumb_png", None)
                self.storage.put(submission_key(meta["id"], "reference.raw.svg"), ref.raw, "text/plain")
                self.storage.put_json(submission_key(meta["id"], "reference.json"), ref.as_json())
                if overlay:
                    self.storage.put(submission_key(meta["id"], "overlay.png"), overlay, "image/png")
                if thumb:
                    self.storage.put(submission_key(meta["id"], "thumb.png"), thumb, "image/png")
                meta["reference"] = {"paths": len(ref.paths), "points": ref.points,
                                     "sha256_geometry": ref.sha256_geometry, "registration": ref.registration}
                meta["results"].append(result_entry(worker))
            meta["status"] = "pending_review"
            meta["status_history"].append({"status": "pending_review", "at": now, "by": "submitter"})
            meta["retention"] = {"status_since": now}
            self._save(meta)
        self.notifier.event(meta["number"], meta["status"], self.public_entry(meta).get("title"))
        return self.manage_view(meta)


def _global_cap() -> int:
    from .limits import SUBMIT_GLOBAL_DAY

    return SUBMIT_GLOBAL_DAY
