"""Overlay rendering (numpy + PIL only): engine centerlines vs reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PATH_COLOURS = [(220, 0, 180), (0, 90, 255), (255, 120, 0), (130, 0, 255), (0, 160, 160), (200, 40, 40)]
REFERENCE_COLOUR = (0, 170, 0)
JUNCTION_COLOUR = (255, 0, 0)
ENDPOINT_COLOUR = (255, 160, 0)
REGION_COLOUR = (0, 200, 255)


def _scale_for(shape, target: int = 768, max_scale: int = 12) -> int:
    h, w = shape[:2]
    return int(max(1, min(max_scale, target // max(h, w, 1))))


def render_overlay(gray: np.ndarray, paths: list[np.ndarray], reference: list[np.ndarray] | None = None,
                   graph: dict | None = None, region: dict | None = None, scale: int | None = None,
                   title: str | None = None, zoom_to_region: bool = False, zoom_margin_px: int = 4) -> Image.Image:
    """gray: 0..1 float or uint8. Paths are (row, col) pixel-centre floats.

    zoom_to_region crops to the scoring region (plus margin) so crossings can be inspected closely.
    """
    g = np.asarray(gray)
    if g.dtype != np.uint8:
        g = np.clip(np.round(g * 255.0), 0, 255).astype(np.uint8)
    offset = np.zeros(2)
    if zoom_to_region and region:
        h, w = g.shape[:2]
        r0 = max(0, int(np.floor(region["y0"])) - zoom_margin_px)
        c0 = max(0, int(np.floor(region["x0"])) - zoom_margin_px)
        r1 = min(h, int(np.ceil(region["y1"])) + zoom_margin_px)
        c1 = min(w, int(np.ceil(region["x1"])) + zoom_margin_px)
        g = g[r0:r1, c0:c1]
        offset = np.array([r0, c0], dtype=float)
        region = {"x0": region["x0"] - c0, "x1": region["x1"] - c0, "y0": region["y0"] - r0, "y1": region["y1"] - r0}
    s = scale or _scale_for(g.shape, max_scale=24 if zoom_to_region else 12)
    faded = (155 + (g.astype(np.float32) * 100.0 / 255.0)).astype(np.uint8)
    base = Image.fromarray(faded, mode="L").resize((g.shape[1] * s, g.shape[0] * s), Image.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(base)

    def xy(pts):
        pts = np.asarray(pts, dtype=float) - offset
        return [((c + 0.5) * s, (r + 0.5) * s) for r, c in pts]

    if region:
        draw.rectangle([region["x0"] * s, region["y0"] * s, region["x1"] * s, region["y1"] * s],
                       outline=REGION_COLOUR, width=max(1, s // 3))
    for ref in reference or []:
        if len(ref) >= 2:
            draw.line(xy(ref), fill=REFERENCE_COLOUR, width=max(3, s))
    for i, p in enumerate(paths):
        if len(p) >= 2:
            draw.line(xy(p), fill=PATH_COLOURS[i % len(PATH_COLOURS)], width=max(2, s // 2))
    if graph:
        rad = max(3, s)
        for pt in graph.get("endpoints", []):
            (x, y), = xy([pt])
            draw.rectangle([x - rad, y - rad, x + rad, y + rad], outline=ENDPOINT_COLOUR, width=2)
        for j in graph.get("junctions", []):
            (x, y), = xy([j["point"]])
            draw.ellipse([x - rad - 1, y - rad - 1, x + rad + 1, y + rad + 1], outline=JUNCTION_COLOUR, width=2)
    if title:
        draw.rectangle([0, 0, base.width, 14], fill=(255, 255, 255))
        draw.text((3, 1), title[:160], fill=(0, 0, 0))
    return base


def save_overlay(path: str | Path, *args, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(*args, **kwargs).save(path)
    return path
