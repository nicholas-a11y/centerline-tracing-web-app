"""Padding check: is there enough context around a crop?

Runs the live pipeline on a full image and on a padded crop, then compares the
centerlines inside the inner box. If they differ, the crop did not include
enough surrounding context (or supersampling changed because the crop is
smaller than the engine's 600 px supersample threshold; that is reported too).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .runner import REPO_ROOT, run_live  # noqa: F401 (REPO_ROOT import puts the repo on sys.path)
from .scoring import densify

from centerline_engine_light import (  # noqa: E402  read-only use of engine constants
    SMALL_IMAGE_SUPERSAMPLE_MAX_FACTOR as SUPERSAMPLE_MAX_FACTOR,
    SMALL_IMAGE_SUPERSAMPLE_TARGET_MAX_DIM as SUPERSAMPLE_TARGET_PX,
)


def engine_supersample_factor(shape) -> int:
    m = max(shape[:2])
    if m >= SUPERSAMPLE_TARGET_PX:
        return 1
    return int(min(SUPERSAMPLE_MAX_FACTOR, math.ceil(SUPERSAMPLE_TARGET_PX / max(m, 1))))


@dataclass
class PaddingReport:
    inner_box: tuple[int, int, int, int]   # row0, col0, row1, col1 in full-image pixels (exclusive end)
    pad_px: int
    full_supersample: int
    crop_supersample: int
    max_diff_px: float | None
    p95_diff_px: float | None
    full_paths_in_box: int
    crop_paths_in_box: int
    consistent: bool
    note: str = ""


def _clip_points(paths, box):
    r0, c0, r1, c1 = box
    out = []
    for p in paths:
        d = densify(np.asarray(p, float), 0.25)
        m = (d[:, 0] >= r0) & (d[:, 0] <= r1 - 1) & (d[:, 1] >= c0) & (d[:, 1] <= c1 - 1)
        if m.any():
            out.append(d[m])
    return out


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def padding_check(image_path, inner_box, pad_px: int, params: dict | None = None,
                  tolerance_px: float = 0.5) -> PaddingReport:
    """inner_box = (row0, col0, row1, col1) in the full image (end exclusive)."""
    with Image.open(image_path) as im:
        full = np.asarray(im.convert("RGB"))
    h, w = full.shape[:2]
    r0, c0, r1, c1 = inner_box
    pr0, pc0 = max(0, r0 - pad_px), max(0, c0 - pad_px)
    pr1, pc1 = min(h, r1 + pad_px), min(w, c1 + pad_px)
    crop = full[pr0:pr1, pc0:pc1]

    full_res = run_live(_png_bytes(full), params)
    crop_res = run_live(_png_bytes(crop), params)
    if full_res.error or crop_res.error:
        return PaddingReport(tuple(inner_box), pad_px, engine_supersample_factor(full.shape),
                             engine_supersample_factor(crop.shape), None, None, 0, 0, False,
                             f"pipeline error: {full_res.error or crop_res.error}")
    crop_paths = [np.asarray(p, float) + (pr0, pc0) for p in crop_res.paths]
    a = _clip_points(full_res.paths, inner_box)
    b = _clip_points(crop_paths, inner_box)
    ss_full, ss_crop = engine_supersample_factor(full.shape), engine_supersample_factor(crop.shape)
    note = ""
    if ss_full != ss_crop:
        note = (f"supersample factor differs (full x{ss_full}, crop x{ss_crop}); differences may come from "
                "resolution, not missing context")
    if not a and not b:
        return PaddingReport(tuple(inner_box), pad_px, ss_full, ss_crop, 0.0, 0.0, 0, 0, True, note)
    if not a or not b:
        return PaddingReport(tuple(inner_box), pad_px, ss_full, ss_crop, None, None, len(a), len(b), False,
                             (note + "; " if note else "") + "centerlines present in only one of the runs")
    A, B = np.vstack(a), np.vstack(b)
    d = np.concatenate([cKDTree(B).query(A)[0], cKDTree(A).query(B)[0]])
    mx, p95 = float(d.max()), float(np.percentile(d, 95))
    return PaddingReport(tuple(inner_box), pad_px, ss_full, ss_crop, mx, p95, len(a), len(b),
                         bool(mx <= tolerance_px and len(a) == len(b)), note)
