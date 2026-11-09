"""Shared lesson and AI-bundle builders for the admin UI and command-line tools."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from lessonkit.lesson import DEFAULT_SCORING, LESSONS_DIR, load_lesson

from . import schema
from .sanitize import clean_text
from .scoring_worker import engine_commit
from .similarity import compare, phash, tokens
from .storage import submission_key
from .validate import canonical_geometry, validate_focus_box, validate_reference

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_ROLE_TO_STATUS = {"known_problem": "baseline", "must_pass": "active"}


class ExportError(ValueError):
    def __init__(self, message, checks=None):
        super().__init__(message)
        self.checks = checks or {"errors": [message], "warnings": []}


def slugify(value: str) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return (words[:64].strip("-") or "untitled")


def export_defaults(meta: dict) -> dict:
    title = meta.get("title") or ""
    lesson_id = "teach-" + str(meta["number"]) + "-" + slugify(title)
    description = meta.get("description") or ""
    return {
        "lesson_id": lesson_id[:64].strip("-"),
        "description": description,
        "expected_behaviour": description or title,
        "role": "must_pass",
        "tolerance_px": float((meta.get("scoring") or {}).get("tolerance_px", 1.0)),
        "region": (meta.get("scoring") or {}).get("region"),
    }


def _validate_export_fields(meta: dict, values: dict, lessons_dir: Path) -> dict:
    errors = []
    if not isinstance(values, dict):
        raise ExportError("Export fields must be an object.")
    if set(values) - {"lesson_id", "description", "expected_behaviour", "role", "tolerance_px", "region"}:
        errors.append("Unknown export field.")
    lesson_id = values.get("lesson_id", "")
    if not isinstance(lesson_id, str) or not _SAFE_ID.fullmatch(lesson_id):
        errors.append("Lesson id must be 1-64 lowercase letters, numbers or hyphens and start with a letter or number.")
    elif (lessons_dir / lesson_id).exists():
        errors.append(f"Lesson id '{lesson_id}' already exists.")
    description = values.get("description", "")
    expected = values.get("expected_behaviour", "")
    for label, text in (("Description", description), ("Expected behaviour", expected)):
        if not isinstance(text, str) or len(text) > 4000:
            errors.append(f"{label} must be text of at most 4000 characters.")
    role = values.get("role")
    if role not in _ROLE_TO_STATUS:
        errors.append("Choose Known problem or Must pass.")
    tolerance = values.get("tolerance_px")
    tolerance_value = None
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) \
            and math.isfinite(float(tolerance)) and 0.05 <= float(tolerance) <= 100:
        tolerance_value = float(tolerance)
    else:
        errors.append("Tolerance must be between 0.05 and 100 px.")
    region, region_errors = validate_focus_box(values.get("region"), meta["image"]["width"], meta["image"]["height"])
    errors.extend(problem.message for problem in region_errors)
    if errors:
        raise ExportError("Export fields are invalid.", {"errors": errors, "warnings": []})
    return {
        "lesson_id": lesson_id,
        "description": clean_text(description),
        "expected_behaviour": clean_text(expected),
        "role": role,
        "tolerance_px": tolerance_value,
        "region": region,
    }


def _signature(image: bytes, reference: dict, title: str, description: str) -> dict:
    with Image.open(io.BytesIO(image)) as im:
        gray = im.convert("L")
        gray.load()
        gray_arr = np.asarray(gray, dtype=np.uint8)
        width, height = gray.size
    paths = reference["paths"]
    return {
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "geometry_sha256": hashlib.sha256(canonical_geometry(paths)).hexdigest(),
        "phash": phash(gray_arr),
        "width": width,
        "height": height,
        "tokens": tokens(title, description),
        "title_tokens": tokens(title),
        "has_text": bool(description),
        "paths": paths,
    }


def _lesson_duplicates(signature: dict, lessons_dir: Path) -> list[dict]:
    duplicates = []
    for folder in sorted(lessons_dir.iterdir()) if lessons_dir.exists() else ():
        if not (folder / "lesson.json").is_file():
            continue
        lesson = load_lesson(folder)
        image_bytes = lesson.image_path.read_bytes()
        paths = lesson.load_reference()
        with Image.open(lesson.image_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
        other = {
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "geometry_sha256": hashlib.sha256(canonical_geometry(paths)).hexdigest(),
            "phash": phash(gray),
            "width": lesson.size_wh[0],
            "height": lesson.size_wh[1],
            "tokens": tokens(lesson.meta.get("title", ""), lesson.meta.get("description", "")),
            "title_tokens": tokens(lesson.meta.get("title", "")),
            "has_text": bool(lesson.meta.get("description")),
        }
        signals = compare(signature, other, lambda _, p=paths: p)
        if signals:
            duplicates.append({"id": lesson.id, "title": lesson.meta.get("title", ""), "signals": signals})
    return duplicates


def _stable_score(result: dict) -> object:
    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k != "elapsed_ms"}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value

    return stable({k: v for k, v in result.items()
                   if k not in ("overlay_png", "thumb_png", "post_fit_overlay_png", "app_centerline_svg")})


def _required_files(service, meta):
    case_id = meta["id"]
    image = service.storage.get(submission_key(case_id, "image.png"))
    raw_svg = service.storage.get(submission_key(case_id, "reference.raw.svg"))
    reference = service.storage.get_json(submission_key(case_id, "reference.json"), None)
    if image is None or raw_svg is None or reference is None:
        raise ExportError("Case files are unavailable.")
    return image, raw_svg, reference


def check_lesson_export(service, number: int, values: dict, lessons_dir: Path = LESSONS_DIR,
                        ip: str = "admin-export") -> dict:
    meta = service.load_meta(service.id_for_number(number))
    if not meta or meta["status"] == "deleted":
        raise ExportError("Case does not exist.")
    fields = _validate_export_fields(meta, values, lessons_dir)
    image, raw_svg, reference = _required_files(service, meta)
    warnings = []
    errors = []
    registration = None
    try:
        parsed = validate_reference(raw_svg, (meta["image"]["width"], meta["image"]["height"]))
        registration = parsed.registration
        if parsed.registration not in ("image_frame", "border_rect", "size_match"):
            errors.append("Hand traced SVG does not register to the bitmap.")
    except Exception as exc:
        errors.append(f"Hand traced SVG registration check failed ({type(exc).__name__}).")
    if (meta.get("consent") or {}).get("licence") != schema.LICENCE:
        errors.append("The case does not carry the required CC0 licence.")
    try:
        first = service.queue.run(ip, image, reference, meta["params"], {
            **DEFAULT_SCORING, **(meta.get("scoring") or {}),
            "region": fields["region"], "tolerance_px": fields["tolerance_px"],
        })
        second = service.queue.run(ip, image, reference, meta["params"], {
            **DEFAULT_SCORING, **(meta.get("scoring") or {}),
            "region": fields["region"], "tolerance_px": fields["tolerance_px"],
        })
        if first.get("outcome") not in ("pass", "fail") or second.get("outcome") not in ("pass", "fail"):
            errors.append("Scoring did not produce two completed results.")
        if _stable_score(first) != _stable_score(second):
            errors.append("Scoring was not identical on two consecutive runs.")
    except Exception as exc:
        errors.append(f"Could not run the repeatability check ({type(exc).__name__}).")
    signature = _signature(image, reference, meta.get("title", ""), fields["description"])
    duplicates = _lesson_duplicates(signature, lessons_dir)
    if duplicates:
        warnings.append("Possible existing lesson duplicate: " + ", ".join(item["id"] for item in duplicates))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "duplicates": duplicates,
            "registration": registration,
            "scores_identical": not any("identical" in error for error in errors),
            "licence": (meta.get("consent") or {}).get("licence"), "fields": fields}


def _lesson_meta(meta: dict, fields: dict) -> dict:
    return {
        "schema_version": 1,
        "id": fields["lesson_id"],
        "title": meta.get("title") or fields["lesson_id"],
        "category": "user",
        "description": fields["description"],
        "expected_behaviour": fields["expected_behaviour"],
        "role": fields["role"],
        "notes": "",
        "source": {"kind": "teach", "case_number": meta["number"]},
        "licence": schema.LICENCE,
        "image": {"file": "image.png", "width": meta["image"]["width"], "height": meta["image"]["height"]},
        "reference": {"file": "reference.svg"},
        "params": schema.default_settings() | (meta.get("params") or {}),
        "scoring": {
            "region": fields["region"], "tolerance_px": fields["tolerance_px"],
            "min_single_path_coverage": 0.95, "require_single_path": True, "trim_ends_px": 0.0,
        },
        "status": _ROLE_TO_STATUS[fields["role"]],
        "teach": {
            "case_number": meta["number"],
            "submitted_at": meta.get("created_at"),
            "submitter_text": meta.get("description", ""),
            "settings": schema.default_settings() | (meta.get("params") or {}),
            "score_history": meta.get("results", []),
            "engine_commit_at_export": engine_commit(),
            "licence": schema.LICENCE,
            "me_too_count": len(meta.get("me_too") or []),
            "consent_flags": {"consent": meta.get("consent", {}), "visibility": meta.get("visibility", {})},
        },
    }


def build_lesson_archive(service, number: int, values: dict, lessons_dir: Path = LESSONS_DIR,
                         ip: str = "admin-export") -> tuple[bytes, dict]:
    checks = check_lesson_export(service, number, values, lessons_dir, ip)
    if not checks["ok"]:
        raise ExportError("Lesson export checks failed.", checks)
    meta = service.load_meta(service.id_for_number(number))
    image, raw_svg, _ = _required_files(service, meta)
    lesson_meta = _lesson_meta(meta, checks["fields"])
    folder = checks["fields"]["lesson_id"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{folder}/lesson.json", json.dumps(lesson_meta, indent=2) + "\n")
        archive.writestr(f"{folder}/image.png", image)
        archive.writestr(f"{folder}/reference.svg", raw_svg)
    return buffer.getvalue(), checks


def write_lesson_archive(service, number: int, values: dict, destination: Path,
                         lessons_dir: Path | None = None, ip: str = "admin-export") -> Path:
    destination = Path(destination)
    archive, checks = build_lesson_archive(service, number, values, lessons_dir or destination, ip)
    lesson_id = checks["fields"]["lesson_id"]
    folder = destination / lesson_id
    if folder.exists():
        raise ExportError(f"Destination already contains lesson '{lesson_id}'.")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".teach-export-", dir=destination) as temp:
        staged = Path(temp) / lesson_id
        staged.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            for name in (f"{lesson_id}/lesson.json", f"{lesson_id}/image.png", f"{lesson_id}/reference.svg"):
                (staged / Path(name).name).write_bytes(zipped.read(name))
        os.replace(staged, folder)
    record_lesson_export(service, number, checks["fields"])
    return folder


def _ensure_app_centerline(service, meta: dict, ip: str) -> bytes:
    key = submission_key(meta["id"], "app_centerline.svg")
    svg = service.storage.get(key)
    if svg is None:
        service.rescore(ip, int(meta["number"]))
        svg = service.storage.get(key)
    if svg is None:
        raise ExportError("The app centerline could not be generated for this case.")
    return svg


def build_ai_bundle(service, number: int, ip: str = "admin-bundle") -> bytes:
    meta = service.load_meta(service.id_for_number(number))
    if not meta or meta["status"] == "deleted":
        raise ExportError("Case does not exist.")
    image, raw_svg, _ = _required_files(service, meta)
    app_centerline = _ensure_app_centerline(service, meta, ip)
    meta = service.load_meta(service.id_for_number(number))
    overlay = service.storage.get(submission_key(meta["id"], "overlay.png"))
    if overlay is None:
        service.rescore(ip, number)
        overlay = service.storage.get(submission_key(meta["id"], "overlay.png"))
    if overlay is None:
        raise ExportError("The overlay could not be generated for this case.")
    teach_meta = _lesson_meta(meta, export_defaults(meta))["teach"]
    metadata = {"case_number": number, "teach": teach_meta, "scores": meta.get("results", []),
                "settings": meta.get("params", {})}
    expected = ((meta.get("lesson") or {}).get("expected_behaviour")
                or meta.get("description") or meta.get("title") or "")
    readme = (
        f"# Teach case {number}\n\n"
        f"**What is wrong:** {meta.get('description') or '(No submitter text supplied.)'}\n\n"
        f"**What was expected:** {expected}\n\n"
        "## Files\n"
        "- `bitmap.png` is the cropped bitmap submitted to Teach.\n"
        "- `hand_traced.svg` is the submitter's SVG, included as-is and not executed.\n"
        "- `app_centerline.svg` is the app pipeline output in bitmap coordinates.\n"
        "- `overlay.png` shows the stored scoring overlay.\n\n"
        f"Tolerance: {float((meta.get('scoring') or {}).get('tolerance_px', 1.0))} px.\n"
        f"Region: {json.dumps((meta.get('scoring') or {}).get('region'))}.\n\n"
        "Reproduce by using the settings in `metadata.json` with the live pipeline, then compare "
        "the app centerline with the hand trace using the recorded tolerance and region.\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bitmap.png", image)
        archive.writestr("hand_traced.svg", raw_svg)
        archive.writestr("app_centerline.svg", app_centerline)
        archive.writestr("overlay.png", overlay)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2) + "\n")
        archive.writestr("README.md", readme)
    return buffer.getvalue()


def record_lesson_export(service, number: int, fields: dict) -> None:
    meta = service.load_meta(service.id_for_number(number))
    if not meta or meta["status"] == "deleted":
        raise ExportError("Case does not exist.")
    lesson = dict(meta.get("lesson") or {})
    lesson.update({"lesson_id": fields["lesson_id"], "exported_at": datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"),
        "expected_behaviour": fields["expected_behaviour"], "role": fields["role"],
        "tolerance_px": fields["tolerance_px"], "region": fields["region"]})
    meta["lesson"] = lesson
    service._save(meta)


def service_from_environment():
    from .config import load_config
    from .limits import DailyQuota, quota_secret
    from .notify import Notifier
    from .scoring_worker import ScoringQueue
    from .service import TeachService
    from .storage import make_storage

    cfg, error = load_config()
    if error or cfg is None:
        raise ExportError(error or "Teach configuration is unavailable.")
    storage, error = make_storage(cfg)
    if error or storage is None:
        raise ExportError(error or "Teach storage is unavailable.")
    return TeachService(cfg, storage, ScoringQueue(), Notifier(cfg.alert_url, cfg.public_base_url),
                       DailyQuota(storage, quota_secret(storage)))
