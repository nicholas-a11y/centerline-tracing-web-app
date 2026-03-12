#!/usr/bin/env python3
"""Generate synthetic bitmap test patterns for centerline extraction."""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def draw_gapped_line(draw, p1, p2, gap_px, width=6):
    """Draw a line with a centered gap of configurable size."""
    x1, y1 = p1
    x2, y2 = p2
    if gap_px <= 0:
        draw.line((x1, y1, x2, y2), fill=0, width=width)
        return

    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length <= gap_px + 4:
        return

    ux = dx / length
    uy = dy / length
    half_gap = gap_px / 2.0
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0

    g1x = mid_x - ux * half_gap
    g1y = mid_y - uy * half_gap
    g2x = mid_x + ux * half_gap
    g2y = mid_y + uy * half_gap

    draw.line((x1, y1, g1x, g1y), fill=0, width=width)
    draw.line((g2x, g2y, x2, y2), fill=0, width=width)


def draw_test_pattern(draw, stroke=6, variable_widths=False):
    """Draw the full centerline stress-test scene."""
    # Block A: horizontal gap ramp (small to large)
    gaps = [2, 4, 6, 8, 10, 14, 18, 24, 32, 44]
    y0 = 120
    for i, gap in enumerate(gaps):
        y = y0 + i * 55
        width = stroke
        if variable_widths:
            width = max(3, stroke + ((i % 5) - 2))
        draw_gapped_line(draw, (80, y), (980, y), gap, width=width)

    # Block B: diagonal gap ramp
    x0 = 1150
    y0 = 120
    for i, gap in enumerate(gaps):
        y = y0 + i * 55
        width = stroke
        if variable_widths:
            width = max(3, stroke + (2 - (i % 5)))
        draw_gapped_line(draw, (x0, y + 20), (2050, y + 50), gap, width=width)

    # Block C: intersections (cross, skewed, X)
    cy = 760
    draw.line((120, cy, 560, cy), fill=0, width=stroke + (2 if variable_widths else 0))
    draw.line((340, cy - 220, 340, cy + 220), fill=0, width=stroke)

    draw.line((700, cy - 120, 1080, cy + 120), fill=0, width=stroke - 1 if variable_widths else stroke)
    draw.line((700, cy + 120, 1080, cy - 120), fill=0, width=stroke + 1 if variable_widths else stroke)

    draw.line((1210, cy - 180, 1650, cy + 30), fill=0, width=stroke + (1 if variable_widths else 0))
    draw.line((1280, cy + 180, 1600, cy - 160), fill=0, width=stroke - (1 if variable_widths else 0))

    # Block D: T and Y junctions
    ty = 1110
    # T junction
    draw.line((120, ty, 520, ty), fill=0, width=stroke)
    draw.line((320, ty - 170, 320, ty), fill=0, width=stroke + (1 if variable_widths else 0))

    # Inverted T
    draw.line((620, ty, 1020, ty), fill=0, width=stroke - (1 if variable_widths else 0))
    draw.line((820, ty, 820, ty + 180), fill=0, width=stroke + (2 if variable_widths else 0))

    # Y junction
    draw.line((1200, ty - 180, 1380, ty), fill=0, width=stroke)
    draw.line((1560, ty - 180, 1380, ty), fill=0, width=stroke)
    draw.line((1380, ty, 1380, ty + 200), fill=0, width=stroke + (2 if variable_widths else 0))

    # Angled Y junction
    draw.line((1700, ty - 120, 1870, ty + 15), fill=0, width=stroke - (1 if variable_widths else 0))
    draw.line((2030, ty - 130, 1870, ty + 15), fill=0, width=stroke + (1 if variable_widths else 0))
    draw.line((1870, ty + 15, 1810, ty + 230), fill=0, width=stroke + (1 if variable_widths else 0))

    # Block E: open and closed shapes
    by = 1410

    # Open polyline corridor
    draw.line((100, by + 120, 290, by + 20), fill=0, width=stroke)
    draw.line((290, by + 20, 520, by + 120), fill=0, width=stroke)
    draw.line((520, by + 120, 720, by + 40), fill=0, width=stroke + (1 if variable_widths else 0))

    # Open C shape
    draw.arc((780, by - 20, 1040, by + 240), start=35, end=325, fill=0, width=stroke + (1 if variable_widths else 0))

    # Open U shape
    draw.arc((1100, by + 20, 1400, by + 250), start=0, end=180, fill=0, width=stroke)
    draw.line((1100, by + 140, 1100, by + 250), fill=0, width=stroke)
    draw.line((1400, by + 140, 1400, by + 250), fill=0, width=stroke)

    # Closed rectangle
    draw.rectangle((1480, by + 30, 1730, by + 230), outline=0, width=stroke + (2 if variable_widths else 0))

    # Closed ellipse
    draw.ellipse((1780, by + 30, 2100, by + 230), outline=0, width=stroke - (1 if variable_widths else 0))

    # Intentional short disconnected stubs near major features.
    draw.line((560, 760, 620, 760), fill=0, width=stroke)
    draw.line((1650, 790, 1710, 840), fill=0, width=stroke)
    draw.line((1000, 1550, 1040, 1600), fill=0, width=stroke)


def make_degraded_variant(img):
    """Create a realistic camera-like variant with blur, shading, and noise."""
    # Mild blur to emulate lens softness and antialiasing.
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1.1))
    arr = np.array(blurred, dtype=np.float32)

    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    # Uneven background illumination (slow gradient + vignette-like term).
    grad = 8.0 * ((xx / max(w - 1, 1)) - 0.5) + 6.0 * ((yy / max(h - 1, 1)) - 0.5)
    cx = w * 0.52
    cy = h * 0.48
    rr = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
    vignette = 26.0 * rr

    rng = np.random.default_rng(7)
    noise = rng.normal(loc=0.0, scale=5.0, size=(h, w)).astype(np.float32)

    degraded = arr + grad + vignette + noise
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    return Image.fromarray(degraded, mode="L")


def main():
    out_dir = Path("/Users/nicholas/Documents/centerline-tracing-web-app/test_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = 2200, 1700
    gaps = [2, 4, 6, 8, 10, 14, 18, 24, 32, 44]
    # Clean baseline pattern.
    img_clean = Image.new("L", (w, h), 255)
    draw_clean = ImageDraw.Draw(img_clean)
    draw_test_pattern(draw_clean, stroke=6, variable_widths=False)

    # Variant with controlled thickness variation.
    img_varwidth = Image.new("L", (w, h), 255)
    draw_varwidth = ImageDraw.Draw(img_varwidth)
    draw_test_pattern(draw_varwidth, stroke=6, variable_widths=True)

    # Degraded camera-like version derived from variable-width test case.
    img_degraded = make_degraded_variant(img_varwidth)

    png_path = out_dir / "centerline_gap_bridge_test.png"
    bmp_path = out_dir / "centerline_gap_bridge_test.bmp"
    var_png_path = out_dir / "centerline_gap_bridge_test_varwidth.png"
    var_bmp_path = out_dir / "centerline_gap_bridge_test_varwidth.bmp"
    degraded_png_path = out_dir / "centerline_gap_bridge_test_varwidth_noisy.png"
    degraded_bmp_path = out_dir / "centerline_gap_bridge_test_varwidth_noisy.bmp"
    manifest_path = out_dir / "centerline_gap_bridge_manifest.json"

    img_clean.save(png_path)
    img_clean.save(bmp_path)
    img_varwidth.save(var_png_path)
    img_varwidth.save(var_bmp_path)
    img_degraded.save(degraded_png_path)
    img_degraded.save(degraded_bmp_path)

    manifest = {
        "version": 1,
        "image_size": {"width": w, "height": h},
        "files": {
            "clean_png": png_path.name,
            "clean_bmp": bmp_path.name,
            "varwidth_png": var_png_path.name,
            "varwidth_bmp": var_bmp_path.name,
            "noisy_png": degraded_png_path.name,
            "noisy_bmp": degraded_bmp_path.name,
        },
        "gap_test": {
            "horizontal_block": {
                "start_xy": [80, 120],
                "row_spacing_px": 55,
                "line_length_px": 900,
                "gaps_px": gaps,
            },
            "diagonal_block": {
                "start_xy": [1150, 140],
                "row_spacing_px": 55,
                "approx_line_length_px": 900,
                "gaps_px": gaps,
            },
            "interpretation": {
                "merge_gap_6": "typically bridges rows with gaps up to about 6 px",
                "merge_gap_10": "typically bridges rows with gaps up to about 10 px",
                "merge_gap_18": "typically bridges rows with gaps up to about 18 px",
                "merge_gap_24": "typically bridges rows with gaps up to about 24 px",
                "merge_gap_32": "typically bridges rows with gaps up to about 32 px",
                "merge_gap_44": "aggressive; may bridge nearly all designed gap rows",
            },
            "note": "Exact behavior can vary with thresholding, skeletonization, and path filtering settings.",
        },
        "topology_coverage": [
            "cross intersection (+)",
            "x intersection",
            "skewed intersection",
            "t junction",
            "inverted t junction",
            "y junction (vertical stem)",
            "y junction (angled stem)",
            "open polyline",
            "open c shape",
            "open u shape",
            "closed rectangle",
            "closed ellipse",
            "short disconnected stubs"
        ]
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {png_path}")
    print(f"Saved: {bmp_path}")
    print(f"Saved: {var_png_path}")
    print(f"Saved: {var_bmp_path}")
    print(f"Saved: {degraded_png_path}")
    print(f"Saved: {degraded_bmp_path}")
    print(f"Saved: {manifest_path}")
    print(f"Image size: {w}x{h}")


if __name__ == "__main__":
    main()
