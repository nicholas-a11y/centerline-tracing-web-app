"""Read-only cropped-image download for the main app's Teach checkbox (Stage 5c item 4).

``/process_immediate`` crops ``session.image`` with fractional ``crop_region`` values
(top, bottom, left, right), rounded to pixels, and applies the crop only when it is more
than 10 px in both directions. This module repeats that rounding rule exactly, encodes the
session's raw (pre-normalisation) pixels in that box as an 8-bit grayscale PNG, and decides
whether the crop on its own is *equivalent* to the cropped run: Teach re-uploads the bitmap,
so the crop is equivalent only if, when the downloaded bytes go through the engine's own
loader with the session's normalisation settings,

* the parameter scale (``centerline_core.resolve_parameter_scale``) is the same for the
  crop's shape as for the full image the run sized its settings from,
* the extraction profile (``centerline_core.resolve_extraction_profile``) is the same, and
* the extraction pixels are exactly the pixels the run processed.

Nothing here creates, modifies or touches a session: the registry is read with
``dict.get`` so its last-access time does not change.
"""

from __future__ import annotations

import hashlib
import math
import re
from io import BytesIO

MIN_CROP_EXCESS_PX = 10  # /process_immediate applies the crop only if y2 > y1 + 10 and x2 > x1 + 10
SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REGION_KEYS = ("top", "bottom", "left", "right")

REASON_SCALE = "parameter_scale"
REASON_PROFILE = "extraction_profile"
REASON_PIXELS = "pixels"

NOTE_LARGE = ("This image is large, so the app sizes its settings from the whole picture and the cropped piece "
              "may trace differently on its own. To teach with it: download the cropped image, upload it to the "
              "app as a new image, check that the problem still shows, then tick Teach again.")
NOTE_PIXELS = ("The app adjusted this image when it loaded it, so the cropped piece may trace differently on its "
               "own. To teach with it: download the cropped image, upload it to the app as a new image, check "
               "that the problem still shows, then tick Teach again.")


def parse_region(args) -> dict | None:
    """Return the crop region from query args, None if absent; ValueError if malformed."""
    present = [k for k in REGION_KEYS if args.get(k) not in (None, "")]
    if not present:
        return None
    if len(present) != len(REGION_KEYS):
        raise ValueError("give all of top, bottom, left and right, or none")
    region = {}
    for key in REGION_KEYS:
        raw = str(args.get(key))
        if len(raw) > 32:
            raise ValueError(f"{key} is too long")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{key} must be a number from 0 to 1")
        region[key] = value
    return region


def crop_box(shape, region: dict | None) -> tuple[int, int, int, int, bool]:
    """(y1, y2, x1, x2, applied) using /process_immediate's rounding rule."""
    h, w = int(shape[0]), int(shape[1])
    if not region:
        return 0, h, 0, w, False
    y1 = max(0, int(round(float(region.get("top", 0)) * h)))
    y2 = min(h, int(round(float(region.get("bottom", 1)) * h)))
    x1 = max(0, int(round(float(region.get("left", 0)) * w)))
    x2 = min(w, int(round(float(region.get("right", 1)) * w)))
    if y2 > y1 + MIN_CROP_EXCESS_PX and x2 > x1 + MIN_CROP_EXCESS_PX:
        return y1, y2, x1, x2, True
    return 0, h, 0, w, False


def find_session(app, session_id: str):
    """The main app's session for ``session_id`` without touching its last-access time."""
    if not SESSION_ID_RE.match(session_id or ""):
        return None
    view = app.view_functions.get("upload_file")
    registry = getattr(view, "__globals__", {}).get("sessions") if view else None
    if not isinstance(registry, dict):
        return None
    return dict.get(registry, session_id)


def _session_arrays(session):
    raw = getattr(session, "display_image", None)
    processed = getattr(session, "image", None)
    if raw is None or processed is None or getattr(raw, "shape", None) != getattr(processed, "shape", None):
        return None, None
    return raw, processed


def encode_png(gray) -> bytes:
    import numpy as np
    from PIL import Image

    pixels = np.clip(np.round(np.asarray(gray, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")  # 2-D uint8 -> mode "L"
    return buf.getvalue()


def cropped_png(session, region: dict | None) -> tuple[bytes, tuple[int, int, int, int, bool]] | None:
    raw, _ = _session_arrays(session)
    if raw is None:
        return None
    box = crop_box(raw.shape, region)
    y1, y2, x1, x2, _applied = box
    return encode_png(raw[y1:y2, x1:x2]), box


def assess(session, region: dict | None) -> dict | None:
    """The download's size, hash and equivalence verdict for ``region`` on ``session``."""
    import numpy as np
    from centerline_core import load_and_process_image, resolve_extraction_profile, resolve_parameter_scale

    raw, processed = _session_arrays(session)
    if raw is None:
        return None
    box = crop_box(raw.shape, region)
    y1, y2, x1, x2, applied = box
    png = encode_png(raw[y1:y2, x1:x2])
    params = dict(getattr(session, "parameters", None) or {})
    mode = str(params.get("normalization_mode", "auto"))
    sensitivity = str(params.get("normalization_sensitivity", "medium"))

    reasons = []
    full_scale = float(resolve_parameter_scale(processed.shape))
    crop_scale = float(resolve_parameter_scale((y2 - y1, x2 - x1)))
    if crop_scale != full_scale:
        reasons.append(REASON_SCALE)
    else:
        _raw2, extraction2, info2 = load_and_process_image(png, normalization_mode=mode,
                                                           normalization_sensitivity=sensitivity,
                                                           source_name="image.png")
        if resolve_extraction_profile(info2) != resolve_extraction_profile(getattr(session, "preprocessing_info", None)):
            reasons.append(REASON_PROFILE)
        if not np.array_equal(np.asarray(extraction2), np.asarray(processed[y1:y2, x1:x2])):
            reasons.append(REASON_PIXELS)

    equivalent = not reasons
    note = None if equivalent else (NOTE_LARGE if REASON_SCALE in reasons else NOTE_PIXELS)
    return {
        "width": int(x2 - x1),
        "height": int(y2 - y1),
        "box": {"top": int(y1), "bottom": int(y2), "left": int(x1), "right": int(x2)},
        "crop_applied": bool(applied),
        "image_sha256": hashlib.sha256(png).hexdigest(),
        "image_bytes": len(png),
        "equivalent": equivalent,
        "reasons": reasons,
        "note": note,
        "parameter_scale": {"full_image": full_scale, "crop": crop_scale},
        "normalization_mode": mode,
        "normalization_sensitivity": sensitivity,
    }
