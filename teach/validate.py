"""Teach validation: bitmap, reference SVG and form fields -> structured problems (design 9, 14.4)."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass, field

from . import schema
from .sanitize import clean_text

_SVG_HINTS = {
    "svg_no_registration": ("keep the image in the export, or make the SVG width/height/viewBox equal the bitmap's "
                            "pixel size, with coordinates in pixels"),
    "svg_bitmap_mismatch": "The SVG was traced over a different picture; trace over this exact bitmap.",
    "svg_image_unreadable": "Embed the bitmap as a base64 PNG or JPEG (not a link).",
    "svg_image_aspect": "Do not stretch or crop the bitmap in the editor.",
    "svg_image_rotated": "Place the bitmap upright (flips are fine) and re-export.",
    "svg_multiple_images": "Keep exactly one image: the bitmap being traced.",
    "svg_use_unsupported": "Convert clones/symbols to plain paths before exporting.",
    "svg_doctype": "Remove the DOCTYPE/ENTITY declaration.",
    "svg_too_large": f"Keep the SVG under {schema.MAX_SVG_BYTES // 1024} KB, including any embedded image.",
    "svg_too_many_elements": "Draw one open path per stroke; remove other shapes.",
    "svg_no_geometry": "Draw one open path per stroke along the ideal centre.",
    "svg_units_unsupported": "Use px, mm, cm, in, pt or pc units.",
    "svg_transform_invalid": "Flatten or simplify transforms before exporting.",
    "svg_image_dimensions": "Re-export the SVG with the bitmap embedded at its normal size.",
    "svg_parse": "Export a plain SVG file from your editor.",
}


@dataclass
class Problem:
    field: str
    code: str
    message: str
    hint: str = ""

    def as_dict(self) -> dict:
        return {"field": self.field, "code": self.code, "message": self.message, "hint": self.hint}


class Problems(Exception):
    """Raised with one or more problems; becomes the ``errors`` list of the response."""

    def __init__(self, problems: list[Problem]):
        super().__init__("; ".join(p.code for p in problems))
        self.problems = problems


@dataclass
class Bitmap:
    png: bytes                 # re-encoded 8-bit greyscale PNG, no metadata
    gray: object               # numpy uint8 (H, W)
    width: int
    height: int
    sha256: str
    warnings: list[Problem] = field(default_factory=list)


@dataclass
class Reference:
    paths: list                # list of (N, 2) float arrays, [row, col] pixel centres
    closed: list[bool]
    registration: str          # image_frame | border_rect | size_match
    points: int
    sha256_geometry: str
    raw: bytes
    warnings: list[Problem] = field(default_factory=list)

    def as_json(self) -> dict:
        return {"schema_version": 1, "coords": "row_col_pixel_centre", "registration": self.registration,
                "closed": self.closed, "paths": [[[float(r), float(c)] for r, c in p] for p in self.paths]}


# ---------------------------------------------------------------- bitmap (9.1)

def _image_problem(code: str, message: str, hint: str = "") -> Problems:
    return Problems([Problem("image", code, message, hint)])


def validate_bitmap(data: bytes, dark_threshold: float = 0.2, fld: str = "image") -> Bitmap:
    import numpy as np
    from PIL import Image

    from lessonkit.svg_reference import _header_dimensions  # header-only PNG IHDR / JPEG SOF reader

    if not data:
        raise Problems([Problem(fld, "field_missing", "No bitmap was uploaded.", "Send a PNG or JPEG as 'image'.")])
    if len(data) > schema.MAX_IMAGE_BYTES:
        raise Problems([Problem(fld, "image_too_large",
                                f"Bitmap is {len(data)} bytes; the limit is {schema.MAX_IMAGE_BYTES} bytes.",
                                "Crop to the problem area.")])
    if not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")):
        raise Problems([Problem(fld, "image_format", "Bitmap must be a PNG or JPEG file.", "Export as PNG.")])
    header = _header_dimensions(data[:256 * 1024])
    if header is None:
        raise Problems([Problem(fld, "image_format", "Could not read the PNG/JPEG header.", "Re-export as PNG.")])
    _kind, w, h = header
    if not (schema.MIN_SIDE_PX <= w <= schema.MAX_SIDE_PX and schema.MIN_SIDE_PX <= h <= schema.MAX_SIDE_PX) \
            or w * h > schema.MAX_PIXELS:
        raise Problems([Problem(fld, "image_dimensions",
                                f"Bitmap is {w}x{h} px; allowed is {schema.MIN_SIDE_PX}-{schema.MAX_SIDE_PX} px per "
                                f"side and at most {schema.MAX_PIXELS} pixels.", "Crop to the problem area.")])
    try:
        im = Image.open(io.BytesIO(data))
        if im.size != (w, h):
            raise Problems([Problem(fld, "image_format", "Bitmap header is inconsistent.", "Re-export as PNG.")])
        if getattr(im, "n_frames", 1) > 1 or getattr(im, "is_animated", False) or im.format == "MPO":
            raise Problems([Problem(fld, "image_multiframe", "Animated or multi-frame images are not accepted.",
                                    "Export a single still PNG.")])
        im.load()
    except Problems:
        raise
    except Exception:
        raise Problems([Problem(fld, "image_format", "Bitmap could not be decoded.", "Re-export as PNG.")]) from None

    gray = _to_gray(im, np)
    buf = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buf, format="PNG")
    png = buf.getvalue()

    warnings: list[Problem] = []
    dark = float(np.mean(gray.astype(np.float32) / 255.0 < dark_threshold))
    if dark == 0.0:
        warnings.append(Problem(fld, "image_no_dark", "No pixels are darker than the threshold; nothing will be traced.",
                                "Check the image or lower dark_threshold."))
    elif dark > schema.MOSTLY_DARK_SHARE:
        warnings.append(Problem(fld, "image_mostly_dark", f"{dark:.0%} of the pixels are dark; the image may be inverted.",
                                "Use dark lines on a light background."))
    if max(w, h) > schema.LARGE_SIDE_PX:
        warnings.append(Problem(fld, "image_large", f"Bitmap is {w}x{h} px.", "Crop to the problem area."))
    return Bitmap(png=png, gray=gray, width=w, height=h, sha256=hashlib.sha256(png).hexdigest(), warnings=warnings)


def _to_gray(im, np):
    """8-bit greyscale; alpha is composited onto white and 16-bit data is scaled."""
    from PIL import Image

    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float64)
        top = 65535.0 if mode.startswith("I;16") or arr.max() > 255 else 255.0
        return np.clip(np.round(arr * 255.0 / top), 0, 255).astype(np.uint8)
    if mode == "P":
        im = im.convert("RGBA")
    elif mode == "PA":
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA", "La", "RGBa"):
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, rgba)
    return np.asarray(im.convert("L"), dtype=np.uint8).copy()


# ---------------------------------------------------------------- reference SVG (9.2)

def canonical_geometry(paths) -> bytes:
    rows = [[[round(float(r), 2), round(float(c), 2)] for r, c in p] for p in paths]
    return json.dumps(rows, separators=(",", ":")).encode("ascii")


def validate_reference(data: bytes, bitmap_size: tuple[int, int], fld: str = "reference") -> Reference:
    from lessonkit.svg_reference import SvgReferenceError, read_reference_svg

    if not data:
        raise Problems([Problem(fld, "field_missing", "No reference SVG was uploaded.",
                                "Send the traced SVG as 'reference', with the traced image kept in the export.")])
    try:
        ref = read_reference_svg(data, bitmap_size, max_bytes=schema.MAX_SVG_BYTES, strict_image=True)
    except SvgReferenceError as exc:
        raise Problems([Problem(fld, exc.code, str(exc), _SVG_HINTS.get(exc.code, ""))]) from None
    except Exception as exc:  # anything unexpected in the reader is a parse failure, never a 500
        raise Problems([Problem(fld, "svg_parse", f"SVG could not be read ({type(exc).__name__}).",
                                _SVG_HINTS["svg_parse"])]) from None
    warnings = [Problem(fld, "svg_warning", w) for w in ref.warnings]
    points = int(sum(len(p) for p in ref.paths))
    return Reference(paths=ref.paths, closed=ref.closed, registration=ref.alignment.replace("-", "_"),
                     points=points, sha256_geometry=hashlib.sha256(canonical_geometry(ref.paths)).hexdigest(),
                     raw=data, warnings=warnings)


# ---------------------------------------------------------------- fields (5, 9.3)

def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "1", "on", "yes"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "0", "off", "no", ""):
        return False
    if value is None:
        return False
    return None


def parse_meta_json(raw: str | bytes | None, max_bytes: int = schema.MAX_META_BYTES) -> dict:
    if raw is None or (isinstance(raw, (str, bytes)) and not raw.strip()):
        raise Problems([Problem("meta", "field_missing", "The 'meta' field (JSON) is missing.",
                                "See /teach/schema/submission.json.")])
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if len(raw) > max_bytes:
        raise Problems([Problem("meta", "field_too_long", f"'meta' is longer than {max_bytes} bytes.", "")])
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Problems([Problem("meta", "meta_invalid", "'meta' is not valid UTF-8 JSON.",
                                "See /teach/schema/submission.json.")]) from None
    if not isinstance(obj, dict):
        raise Problems([Problem("meta", "meta_invalid", "'meta' must be a JSON object.", "")])
    return obj


def validate_fields(meta: dict) -> tuple[dict, list[Problem]]:
    """Text fields, settings, opt-ins and consent. The focus box is checked later against the bitmap."""
    problems: list[Problem] = []
    out: dict = {}
    known = set(schema.TEXT_FIELDS) | set(schema.BOOL_FIELDS) | {"settings", "focus_box", "consent"}
    for key in meta:
        if key not in known:
            problems.append(Problem(key, "meta_invalid", f"Unknown field '{key}'.", "See /teach/schema/submission.json."))
    for name, (required, limit, _desc) in schema.TEXT_FIELDS.items():
        value = meta.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if required:
                problems.append(Problem(name, "field_missing", f"'{name}' is required.", ""))
            out[name] = ""
            continue
        if not isinstance(value, str):
            problems.append(Problem(name, "meta_invalid", f"'{name}' must be a string.", ""))
            continue
        text = clean_text(value)
        if len(text) > limit:
            problems.append(Problem(name, "field_too_long", f"'{name}' is {len(text)} characters; the limit is {limit}.", ""))
        elif required and not text:
            problems.append(Problem(name, "field_missing", f"'{name}' is required.", ""))
        out[name] = text
    for name in schema.BOOL_FIELDS:
        b = _as_bool(meta.get(name))
        if b is None:
            problems.append(Problem(name, "meta_invalid", f"'{name}' must be true or false.", ""))
        out[name] = bool(b)
    if _as_bool(meta.get("consent")) is not True:
        problems.append(Problem("consent", "consent_required", "You must accept the consent and licence text (CC0 1.0).",
                                "Set \"consent\": true after reading the text in the spec."))
    settings, setting_problems = validate_settings(meta.get("settings"))
    problems.extend(setting_problems)
    out["settings"] = settings
    out["focus_box"] = meta.get("focus_box")
    return out, problems


def validate_settings(raw) -> tuple[dict, list[Problem]]:
    problems: list[Problem] = []
    out: dict = {}
    if raw is None:
        return out, problems
    if not isinstance(raw, dict):
        return out, [Problem("settings", "meta_invalid", "'settings' must be an object.", "")]
    for key, value in raw.items():
        spec = schema.SETTINGS.get(key)
        fld = f"settings.{key}"
        if spec is None:
            problems.append(Problem(fld, "param_out_of_range", f"'{key}' is not an allowed setting.",
                                    "Allowed: " + ", ".join(schema.SETTINGS)))
            continue
        kind = spec["type"]
        if kind == "boolean":
            b = _as_bool(value)
            if b is None:
                problems.append(Problem(fld, "param_out_of_range", f"'{key}' must be true or false.", ""))
            else:
                out[key] = b
        elif kind == "string":
            if value not in spec["enum"]:
                problems.append(Problem(fld, "param_out_of_range", f"'{key}' must be one of {spec['enum']}.", ""))
            else:
                out[key] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(Problem(fld, "param_out_of_range", f"'{key}' must be a number.", ""))
                continue
            if kind == "integer" and float(value) != int(value):
                problems.append(Problem(fld, "param_out_of_range", f"'{key}' must be an integer.", ""))
                continue
            if not spec["minimum"] <= value <= spec["maximum"]:
                problems.append(Problem(fld, "param_out_of_range",
                                        f"'{key}' must be between {spec['minimum']} and {spec['maximum']}.", ""))
                continue
            out[key] = int(value) if kind == "integer" else float(value)
    return out, problems


def validate_focus_box(box, width: int, height: int) -> tuple[dict | None, list[Problem]]:
    if box is None:
        return None, []
    keys = ("x0", "y0", "x1", "y1")
    if not isinstance(box, dict) or set(box) != set(keys):
        return None, [Problem("focus_box", "meta_invalid", "focus_box must be {x0, y0, x1, y1} or null.", "")]
    vals = {}
    for k in keys:
        v = box[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None, [Problem("focus_box", "param_out_of_range", f"focus_box.{k} must be a number.", "")]
        vals[k] = float(v)
    if not (0 <= vals["x0"] < vals["x1"] <= width and 0 <= vals["y0"] < vals["y1"] <= height):
        return None, [Problem("focus_box", "param_out_of_range",
                              f"focus_box must satisfy 0 <= x0 < x1 <= {width} and 0 <= y0 < y1 <= {height}.", "")]
    return vals, []


def effective_params(settings: dict) -> dict:
    from lessonkit.lesson import DEFAULT_PARAMS

    params = dict(DEFAULT_PARAMS)
    params.update(settings or {})
    return params


def effective_scoring(region: dict | None) -> dict:
    from lessonkit.lesson import DEFAULT_SCORING

    scoring = dict(DEFAULT_SCORING)
    scoring["region"] = region
    scoring["tolerance_px"] = 1.0
    return scoring


@dataclass
class Submission:
    fields: dict
    bitmap: Bitmap
    reference: Reference
    region: dict | None
    params: dict
    scoring: dict

    @property
    def warnings(self) -> list[Problem]:
        return self.bitmap.warnings + self.reference.warnings


def validate_submission(image: bytes | None, reference: bytes | None, meta_raw) -> Submission:
    """Everything except limits and duplicates. Raises Problems with all errors found."""
    problems: list[Problem] = []
    try:
        meta = parse_meta_json(meta_raw)
    except Problems as exc:
        problems.extend(exc.problems)
        meta = {}
    fields, field_problems = validate_fields(meta) if meta else ({"settings": {}, "focus_box": None}, [])
    problems.extend(field_problems)
    dark = fields.get("settings", {}).get("dark_threshold", 0.2)
    bitmap = ref = None
    try:
        bitmap = validate_bitmap(image or b"", dark)
    except Problems as exc:
        problems.extend(exc.problems)
    region = None
    if bitmap is not None:
        try:
            ref = validate_reference(reference or b"", (bitmap.width, bitmap.height))
        except Problems as exc:
            problems.extend(exc.problems)
        region, box_problems = validate_focus_box(fields.get("focus_box"), bitmap.width, bitmap.height)
        problems.extend(box_problems)
    elif not reference:
        problems.append(Problem("reference", "field_missing", "No reference SVG was uploaded.", ""))
    if problems:
        raise Problems(problems)
    return Submission(fields=fields, bitmap=bitmap, reference=ref, region=region,
                      params=effective_params(fields["settings"]), scoring=effective_scoring(region))
