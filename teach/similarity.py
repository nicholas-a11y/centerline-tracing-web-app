"""Duplicate detection (design 10): exact hashes, a 64-bit DCT pHash, geometry chamfer and text Jaccard."""

from __future__ import annotations

import re

PHASH_MAX_HAMMING = 10
CHAMFER_MAX_PX = 2.0
TEXT_WITH_GEOMETRY = 0.3
TEXT_ALONE = 0.8
# The text signal is only used when both sides have at least this many distinct tokens (after
# stop words), so very short titles cannot reach a threshold by themselves.
MIN_TEXT_TOKENS = 4

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
STOP_WORDS = frozenset("""
a an and are as at be but by for from has have in into is it its of on or that the this to was
were when where which with not no does do did line lines path paths centerline centre center
""".split())


def phash(gray) -> str:
    """64-bit DCT perceptual hash of a uint8 greyscale array, as 16 hex digits."""
    import numpy as np
    from PIL import Image
    from scipy.fft import dctn

    small = Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L").resize((32, 32), Image.BILINEAR)
    coeffs = dctn(np.asarray(small, dtype=np.float64), norm="ortho")[:8, :8].ravel()
    median = np.median(coeffs[1:])
    bits = 0
    for value in coeffs:
        bits = (bits << 1) | int(value > median)
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def tokens(*texts: str) -> list[str]:
    words = set()
    for text in texts:
        for w in _WORD.findall((text or "").lower()):
            if len(w) > 1 and w not in STOP_WORDS:
                words.add(w)
    return sorted(words)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def chamfer(paths_a, paths_b) -> float:
    """Symmetric mean chamfer distance between two sets of polylines (pixel units)."""
    import numpy as np
    from scipy.spatial import cKDTree

    from lessonkit.scoring import densify

    def cloud(paths):
        parts = [densify(np.asarray(p, dtype=float)) for p in paths if len(p)]
        return np.vstack(parts) if parts else np.zeros((0, 2))

    a, b = cloud(paths_a), cloud(paths_b)
    if not len(a) or not len(b):
        return float("inf")
    da = cKDTree(b).query(a)[0].mean()
    db = cKDTree(a).query(b)[0].mean()
    return float((da + db) / 2.0)


def text_similarity(new: dict, other: dict) -> float:
    """Token Jaccard on title + description; on the title only when either side has no description.

    Entries without "title_tokens"/"has_text" (stored before Stage 5d) fall back to "tokens".
    Returns 0 unless both compared token sets have at least MIN_TEXT_TOKENS tokens.
    """
    if new.get("has_text", True) and other.get("has_text", True):
        a, b = new.get("tokens", ()), other.get("tokens", ())
    else:
        a = new.get("title_tokens", new.get("tokens", ()))
        b = other.get("title_tokens", other.get("tokens", ()))
    a, b = set(a), set(b)
    if len(a) < MIN_TEXT_TOKENS or len(b) < MIN_TEXT_TOKENS:
        return 0.0
    return jaccard(a, b)


def compare(new: dict, other: dict, load_paths=None) -> dict | None:
    """Return the match signals if ``other`` should be shown as similar, else None.

    ``new``/``other``: {"image_sha256", "geometry_sha256", "phash", "width", "height", "tokens",
    "title_tokens", "has_text"}; ``new`` may also carry "paths". ``load_paths(other)`` loads the
    other item's hand traced paths, and is only called when the bitmap sizes are equal.
    """
    exact = (new["image_sha256"] == other.get("image_sha256")
             and new["geometry_sha256"] == other.get("geometry_sha256"))
    if exact:
        return {"exact": True, "image": True, "geometry": True, "text": 1.0}
    image = bool(other.get("phash")) and hamming(new["phash"], other["phash"]) <= PHASH_MAX_HAMMING
    text = text_similarity(new, other)
    geometry = False
    if (not image and text >= TEXT_WITH_GEOMETRY and load_paths is not None and new.get("paths") is not None
            and (new["width"], new["height"]) == (other.get("width"), other.get("height"))):
        other_paths = load_paths(other)
        geometry = other_paths is not None and chamfer(new["paths"], other_paths) <= CHAMFER_MAX_PX
    if image or (geometry and text >= TEXT_WITH_GEOMETRY) or text >= TEXT_ALONE:
        return {"exact": False, "image": image, "geometry": geometry, "text": round(text, 3)}
    return None
