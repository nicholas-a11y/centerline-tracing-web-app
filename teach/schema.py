"""One definition of the Teach form fields and limits (design 5, 12.1, 14.2).

The validator, ``/teach/schema/submission.json`` and ``/teach/openapi.json`` are all
generated from the constants below, so they cannot drift apart.
"""

from __future__ import annotations

MAX_BODY_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_SVG_BYTES = 512 * 1024
MIN_SIDE_PX = 32
MAX_SIDE_PX = 2048
MAX_PIXELS = 4_000_000
LARGE_SIDE_PX = 1024
MOSTLY_DARK_SHARE = 0.60
MAX_META_BYTES = 64 * 1024

TEXT_FIELDS = {
    # name: (required, max characters, description)
    "title": (True, 120, "Short title. A sanitised version (<= 80 characters, no links) is public straight away."),
    "what_is_wrong": (True, 2000, "What the traced centerline does wrong."),
    "what_is_expected": (True, 2000, "What it should do instead."),
    "notes": (False, 4000, "Anything else (optional)."),
    "not_duplicate_reason": (False, 200, "One line on why this is not a duplicate of a similar item (optional)."),
}

SETTINGS = {
    # name: JSON-schema fragment; same whitelist and ranges as the main UI
    "dark_threshold": {"type": "number", "minimum": 0.05, "maximum": 0.8, "default": 0.2},
    "min_path_length": {"type": "integer", "minimum": 1, "maximum": 20, "default": 2},
    "enable_pruning": {"type": "boolean", "default": False},
    "normalization_mode": {"type": "string", "enum": ["auto", "on", "off"], "default": "auto"},
    "normalization_sensitivity": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
}

BOOL_FIELDS = {
    "public_thumbnail": "Allow the overlay thumbnail to be public once reviewed (default off).",
    "share_bitmap": "Allow the full bitmap to be public once reviewed (default off).",
}

ME_TOO_NOTE_MAX = 2000
MANAGE_ANSWER_MAX = 2000

CONSENT_TEXT_VERSION = 1
LICENCE = "CC0-1.0"
CONSENT_TEXT = (
    "I made this image and drawing, or have the right to share them. I have cropped them to the "
    "problem area and they contain no personal or confidential information. I license them under "
    "CC0 1.0 (public domain dedication) so they can be used as test data for this project. If the "
    "case is accepted, it may be published in the project's public repository. After that it can "
    "be removed from future versions but not from the repository's history. Until then I can "
    "delete it at any time with my manage link."
)
ADVICE_TEXT = (
    "Crop to the problem area, with some context around it. Remove title blocks, names, addresses "
    "and logos. Never upload confidential drawings."
)

RULES = (
    "One problem per submission.",
    "Search the queue first. If your case is already there, use \"me too\".",
    "The SVG must be registered to the bitmap. Best: trace over the original (cropped) image and keep "
    "the image in the exported SVG (in LightBurn you may disable the image layer's output; the image "
    "stays in the export). Otherwise make the SVG width/height/viewBox equal the bitmap's pixel size, "
    "with coordinates in pixels. Draw one open path per stroke, along the ideal centre.",
    "Crop to the problem area with some context. No confidential or personal content.",
    "You cannot set the scoring tolerance; the maintainer does.",
    "Run /teach/api/validate first and fix every error before submitting.",
)

ERROR_CODES = {
    "image": ["image_too_large", "image_dimensions", "image_format", "image_multiframe"],
    "reference": ["svg_too_large", "svg_doctype", "svg_parse", "svg_use_unsupported", "svg_transform_invalid",
                  "svg_units_unsupported", "svg_no_registration", "svg_multiple_images", "svg_image_rotated",
                  "svg_image_aspect", "svg_bitmap_mismatch", "svg_image_dimensions", "svg_image_unreadable",
                  "svg_no_geometry", "svg_too_many_elements"],
    "fields": ["field_missing", "field_too_long", "param_out_of_range", "consent_required", "meta_invalid"],
    "limits": ["duplicate_exact", "rate_limited", "quota_exceeded", "busy", "scoring_timeout",
               "request_too_large", "invalid_state", "not_found", "manage_token_invalid"],
}


def _focus_box_schema() -> dict:
    num = {"type": "number"}
    return {
        "type": ["object", "null"],
        "description": "Optional box around the problem, in bitmap pixel units (x right, y down, pixel edges). "
                       "Used as the initial scoring region; otherwise the whole image is scored.",
        "properties": {k: num for k in ("x0", "y0", "x1", "y1")},
        "required": ["x0", "y0", "x1", "y1"],
        "additionalProperties": False,
    }


def submission_meta_schema() -> dict:
    props: dict = {}
    for name, (_req, limit, desc) in TEXT_FIELDS.items():
        props[name] = {"type": "string", "maxLength": limit, "description": desc}
    props["focus_box"] = _focus_box_schema()
    props["settings"] = {
        "type": "object", "description": "Pipeline settings (UI defaults otherwise). Tolerance is not settable.",
        "properties": SETTINGS, "additionalProperties": False,
    }
    for name, desc in BOOL_FIELDS.items():
        props[name] = {"type": "boolean", "default": False, "description": desc}
    props["consent"] = {"const": True, "description": CONSENT_TEXT}
    required = [n for n, (req, _l, _d) in TEXT_FIELDS.items() if req] + ["consent"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "/teach/schema/submission.json",
        "title": "Teach submission meta (the 'meta' form field, JSON)",
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def me_too_meta_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "note": {"type": "string", "maxLength": ME_TOO_NOTE_MAX},
            "consent": {"type": "boolean", "description": "Required (true) when a variant image and reference are sent."},
        },
        "additionalProperties": False,
    }


def _response_schema() -> dict:
    problem = {"type": "object", "properties": {k: {"type": "string"} for k in ("field", "code", "message", "hint")}}
    return {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "errors": {"type": "array", "items": problem},
            "warnings": {"type": "array", "items": problem},
            "result": {"type": ["object", "null"]},
        },
        "required": ["ok", "errors", "warnings", "result"],
    }


def openapi() -> dict:
    multipart = {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "format": "binary",
                              "description": f"PNG or JPEG, <= {MAX_IMAGE_BYTES} bytes, {MIN_SIDE_PX}-{MAX_SIDE_PX} px "
                                             f"per side, <= {MAX_PIXELS} pixels"},
                    "reference": {"type": "string", "format": "binary",
                                  "description": f"SVG registered to the bitmap, <= {MAX_SVG_BYTES} bytes"},
                    "meta": {"type": "string", "description": "JSON; see /teach/schema/submission.json"},
                },
                "required": ["image", "reference", "meta"],
            }
        }
    }
    json_resp = {"application/json": {"schema": _response_schema()}}
    std = {"200": {"description": "Validation response", "content": json_resp},
           "413": {"description": "request_too_large"}, "429": {"description": "rate_limited"},
           "503": {"description": "busy (Retry-After)"}}
    number = [{"name": "number", "in": "path", "required": True, "schema": {"type": "integer"}}]
    manage = [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
              {"name": "X-Teach-Manage-Token", "in": "header", "required": True, "schema": {"type": "string"}}]
    return {
        "openapi": "3.1.0",
        "info": {"title": "Centerline Teach API", "version": "1",
                 "description": "Submit hard tracing cases. Human-readable spec: /teach/spec"},
        "paths": {
            "/teach/api/validate": {"post": {"summary": "Dry run: validate and score; stores nothing",
                                             "requestBody": {"content": multipart}, "responses": std}},
            "/teach/api/similar": {"post": {"summary": "Duplicate check only",
                                            "requestBody": {"content": multipart}, "responses": std}},
            "/teach/api/submissions": {"post": {
                "summary": "Submit", "requestBody": {"content": multipart},
                "responses": {"201": {"description": "Stored; returns number, queue_url and manage_url"},
                              "409": {"description": "duplicate_exact"}, "422": {"description": "validation errors"},
                              "413": {"description": "request_too_large"}, "429": {"description": "rate_limited / quota_exceeded"},
                              "503": {"description": "busy"}}}},
            "/teach/api/submissions/{number}": {"get": {"summary": "Public data for one item", "parameters": number,
                                                        "responses": {"200": {"description": "public entry"}}}},
            "/teach/api/submissions/{number}/me-too": {"post": {
                "summary": "Add a note (and optionally a variant image + reference) to an existing item",
                "parameters": number, "requestBody": {"content": {"multipart/form-data": {"schema": {
                    "type": "object", "properties": {
                        "meta": {"type": "string", "description": "JSON: {note, consent}"},
                        "image": {"type": "string", "format": "binary"},
                        "reference": {"type": "string", "format": "binary"}}}}}},
                "responses": {"201": {"description": "added"}}}},
            "/teach/api/queue": {"get": {"summary": "Public queue", "responses": {"200": {"description": "items"}}}},
            "/teach/api/manage/{id}": {
                "get": {"summary": "Your own item (full)", "parameters": manage, "responses": {"200": {"description": "item"}}},
                "patch": {"summary": "Answer a needs_info question (note and/or replacement reference SVG)",
                          "parameters": manage, "responses": {"200": {"description": "updated"}}},
                "delete": {"summary": "Delete your item (leaves a tombstone)", "parameters": manage,
                           "responses": {"200": {"description": "deleted"}}},
            },
        },
        "components": {"schemas": {"SubmissionMeta": submission_meta_schema(), "MeTooMeta": me_too_meta_schema(),
                                   "ValidationResponse": _response_schema()}},
    }
