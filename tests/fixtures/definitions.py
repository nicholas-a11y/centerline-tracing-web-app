"""
tests/fixtures/definitions.py

Ground-truth fixture bank for curve-fitting tests.

Each FixtureDefinition knows:
  - how to rasterize itself (draw onto a PIL canvas)
  - how to produce an analytically-exact ideal curve sample at any resolution

The ideal curve sample is the reference signal for golden-master tests; it is
*not* stored in the generated PNG/JSON — it is re-computed from the mathematical
definition every time a test runs, so it never drifts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import ImageDraw


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class FixtureDefinition:
    id: str
    description: str
    # "line" | "curve" | "corner"
    category: str

    # draw(draw: ImageDraw.ImageDraw, size: int, stroke_width: int) → None
    draw: Callable[[ImageDraw.ImageDraw, int, int], None]

    # ideal_sample(size: int, n_points: int) → np.ndarray shape (N, 2) [[row, col]]
    # This is the analytically-correct dense spine of the shape.
    ideal_sample: Callable[[int, int], np.ndarray]

    # Max acceptable mean point-to-curve deviation for contract tests (pixels).
    tolerance_px: float = 3.0

    # Segment types we expect the fitter to predominantly produce,
    # used by purity contract tests.
    expected_segment_types: list[str] = field(default_factory=lambda: ["cubic"])


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _sample_arc(
    centre_rc: tuple[float, float],
    radius: float,
    theta0_deg: float,
    theta1_deg: float,
    n: int,
) -> np.ndarray:
    """Dense sample of a circular arc in (row, col) image coords.

    Angles follow PIL/image convention: 0° = right, increases clockwise.
    """
    t = np.linspace(math.radians(theta0_deg), math.radians(theta1_deg), n)
    col = centre_rc[1] + radius * np.cos(t)
    row = centre_rc[0] + radius * np.sin(t)
    return np.stack([row, col], axis=1)


def _sample_arc_from_int_bbox(
    size: int,
    x0_frac: float,
    y0_frac: float,
    x1_frac: float,
    y1_frac: float,
    theta0_deg: float,
    theta1_deg: float,
    n: int,
    inset_px: float = 0.0,
) -> np.ndarray:
    """Sample an elliptical/circular arc using the same int bbox as draw.arc.

    This keeps the analytical reference aligned with raster fixtures that use
    int(...) quantized PIL arc bounds.
    """
    x0 = int(size * x0_frac)
    y0 = int(size * y0_frac)
    x1 = int(size * x1_frac)
    y1 = int(size * y1_frac)

    centre_c = (x0 + x1) / 2.0
    centre_r = (y0 + y1) / 2.0
    radius_c = max(0.1, (x1 - x0) / 2.0 - float(inset_px))
    radius_r = max(0.1, (y1 - y0) / 2.0 - float(inset_px))

    t = np.linspace(math.radians(theta0_deg), math.radians(theta1_deg), n)
    col = centre_c + radius_c * np.cos(t)
    row = centre_r + radius_r * np.sin(t)
    return np.stack([row, col], axis=1)


def _sample_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n: int,
) -> np.ndarray:
    """Dense sample of a cubic Bézier; all pts in (row, col) coords."""
    t = np.linspace(0.0, 1.0, n)
    omt = 1.0 - t
    p0, p1, p2, p3 = (np.array(p) for p in (p0, p1, p2, p3))
    pts = (
        (omt**3)[:, None] * p0
        + (3 * omt**2 * t)[:, None] * p1
        + (3 * omt * t**2)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    return pts  # (N, 2) [[row, col]]


def _sample_line(
    r0: float, c0: float, r1: float, c1: float, n: int
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    row = r0 + t * (r1 - r0)
    col = c0 + t * (c1 - c0)
    return np.stack([row, col], axis=1)


def _offset_points(points: np.ndarray, dr: float = 0.0, dc: float = 0.0) -> np.ndarray:
    """Apply a constant (row, col) offset to a sampled path."""
    if points.size == 0:
        return points
    return points + np.array([dr, dc], dtype=float)


def _rotate_points(points: np.ndarray, angle_deg: float, center_rc: tuple[float, float]) -> np.ndarray:
    """Rotate (row, col) points clockwise in image coordinates."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr

    center = np.array(center_rc, dtype=float)
    rel = arr - center
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    row = center[0] + rel[:, 0] * cos_a - rel[:, 1] * sin_a
    col = center[1] + rel[:, 0] * sin_a + rel[:, 1] * cos_a
    return np.stack([row, col], axis=1)


def _draw_cubic_pil(
    draw: ImageDraw.ImageDraw,
    p0_xy: tuple[float, float],
    p1_xy: tuple[float, float],
    p2_xy: tuple[float, float],
    p3_xy: tuple[float, float],
    fill: int,
    width: int,
    steps: int = 60,
) -> None:
    """Rasterize a cubic Bézier as a polyline on a PIL ImageDraw.
    All points in PIL (col, row) = (x, y) order.
    """
    t = np.linspace(0, 1, steps)
    omt = 1 - t
    p0, p1, p2, p3 = (np.array(p) for p in (p0_xy, p1_xy, p2_xy, p3_xy))
    pts = (
        (omt**3)[:, None] * p0
        + (3 * omt**2 * t)[:, None] * p1
        + (3 * omt * t**2)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    xy = [(float(x), float(y)) for x, y in pts]
    draw.line(xy, fill=fill, width=width)


def _concat_path_segments(*segments: np.ndarray) -> np.ndarray:
    """Concatenate sampled path segments, dropping duplicate junction points."""
    parts: list[np.ndarray] = []
    for seg in segments:
        arr = np.asarray(seg, dtype=float)
        if arr.size == 0:
            continue
        if parts and np.allclose(parts[-1][-1], arr[0]):
            arr = arr[1:]
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(parts)


def _draw_sampled_path(
    draw: ImageDraw.ImageDraw,
    points_rc: np.ndarray,
    fill: int,
    width: int,
    closed: bool = False,
) -> None:
    """Rasterize a pre-sampled (row, col) path as a PIL polyline."""
    if points_rc.size == 0:
        return
    xy = [(float(col), float(row)) for row, col in np.asarray(points_rc, dtype=float)]
    if closed and len(xy) >= 2 and xy[0] != xy[-1]:
        xy.append(xy[0])
    draw.line(xy, fill=fill, width=width)


def _sample_closed_polyline(vertices_rc: np.ndarray, n: int) -> np.ndarray:
    """Dense sample of a closed polyline in (row, col) coordinates."""
    verts = np.asarray(vertices_rc, dtype=float)
    if verts.shape[0] < 2:
        return verts.copy()

    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack([verts, verts[0]])

    edge_lengths = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    total_length = float(np.sum(edge_lengths))
    if total_length <= 1e-9:
        return verts[:1].copy()

    samples_remaining = max(int(n), len(edge_lengths) * 8)
    parts: list[np.ndarray] = []
    for idx, edge_len in enumerate(edge_lengths):
        if idx == len(edge_lengths) - 1:
            edge_n = samples_remaining
        else:
            edge_n = max(8, int(round((edge_len / total_length) * n)))
            samples_remaining -= edge_n
        parts.append(
            _sample_line(
                verts[idx][0],
                verts[idx][1],
                verts[idx + 1][0],
                verts[idx + 1][1],
                edge_n,
            )
        )
    return _concat_path_segments(*parts)


def _square_vertices(size: int, inset_frac: float = 0.2) -> np.ndarray:
    top = size * inset_frac
    bottom = size * (1.0 - inset_frac)
    left = size * inset_frac
    right = size * (1.0 - inset_frac)
    return np.array(
        [
            [top, left],
            [top, right],
            [bottom, right],
            [bottom, left],
        ],
        dtype=float,
    )


# ─── Fixture bank ─────────────────────────────────────────────────────────────

FIXTURES: list[FixtureDefinition] = [

    # ── Curves (primary targets for the overhaul) ─────────────────────────────

    FixtureDefinition(
        id="curve-quarter-arc",
        description="Quarter-circle arc, 0°→90° (clockwise in image coords)",
        category="curve",
        draw=lambda draw, size, sw: draw.arc(
            [
                int(size * 0.05), int(size * 0.05),
                int(size * 0.95), int(size * 0.95),
            ],
            start=0, end=90, fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc_from_int_bbox(
            size=size,
            x0_frac=0.05,
            y0_frac=0.05,
            x1_frac=0.95,
            y1_frac=0.95,
            theta0_deg=0.0,
            theta1_deg=90.0,
            n=n,
            inset_px=1.0,
        ),
        tolerance_px=2.5,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-half-arc",
        description="Semicircle arc, 0°→180° (clockwise in image coords)",
        category="curve",
        draw=lambda draw, size, sw: draw.arc(
            [
                int(size * 0.1), int(size * 0.1),
                int(size * 0.9), int(size * 0.9),
            ],
            start=0, end=180, fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc_from_int_bbox(
            size=size,
            x0_frac=0.1,
            y0_frac=0.1,
            x1_frac=0.9,
            y1_frac=0.9,
            theta0_deg=0.0,
            theta1_deg=180.0,
            n=n,
            inset_px=1.0,
        ),
        tolerance_px=2.5,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-true-circle",
        description="True circle loop — closed curve target for cubic fitting",
        category="curve",
        draw=lambda draw, size, sw: draw.arc(
            [
                int(size * 0.12), int(size * 0.12),
                int(size * 0.88), int(size * 0.88),
            ],
            start=0,
            end=359,
            fill=0,
            width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc_from_int_bbox(
            size=size,
            x0_frac=0.12,
            y0_frac=0.12,
            x1_frac=0.88,
            y1_frac=0.88,
            theta0_deg=0.0,
            theta1_deg=360.0,
            n=n,
            inset_px=1.0,
        ),
        tolerance_px=2.8,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-s-cubic",
        description="S-shaped cubic Bézier — classic test for C-continuity",
        category="curve",
        draw=lambda draw, size, sw: _draw_cubic_pil(
            draw,
            p0_xy=(size * 0.1, size * 0.8),   # (col, row)
            p1_xy=(size * 0.3, size * 0.2),
            p2_xy=(size * 0.7, size * 0.8),
            p3_xy=(size * 0.9, size * 0.2),
            fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _offset_points(
            _sample_cubic(
                p0=(size * 0.8, size * 0.1),   # (row, col)
                p1=(size * 0.2, size * 0.3),
                p2=(size * 0.8, size * 0.7),
                p3=(size * 0.2, size * 0.9),
                n=n,
            ),
            # PIL stroke rasterization is pixel-center quantized; this half-pixel
            # shift keeps the analytical curve centered in the rendered stroke.
            dr=-0.5,
            dc=-0.5,
        ),
        tolerance_px=1.5,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-gentle-convex",
        description="Gentle convex curve — shallow bowed cubic",
        category="curve",
        draw=lambda draw, size, sw: _draw_cubic_pil(
            draw,
            p0_xy=(size * 0.1, size * 0.6),
            p1_xy=(size * 0.35, size * 0.45),
            p2_xy=(size * 0.65, size * 0.45),
            p3_xy=(size * 0.9, size * 0.6),
            fill=0,
            width=sw,
        ),
        ideal_sample=lambda size, n: _offset_points(
            _sample_cubic(
                p0=(size * 0.6, size * 0.1),
                p1=(size * 0.45, size * 0.35),
                p2=(size * 0.45, size * 0.65),
                p3=(size * 0.6, size * 0.9),
                n=n,
            ),
            dr=-0.5,
            dc=-0.5,
        ),
        tolerance_px=1.5,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-tight-180",
        description="Tight 180° U-turn — tests multi-segment cubic fitting",
        category="curve",
        draw=lambda draw, size, sw: draw.arc(
            [
                int(size * 0.2), int(size * 0.2),
                int(size * 0.8), int(size * 0.8),
            ],
            start=180, end=360, fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc_from_int_bbox(
            size=size,
            x0_frac=0.2,
            y0_frac=0.2,
            x1_frac=0.8,
            y1_frac=0.8,
            theta0_deg=180.0,
            theta1_deg=360.0,
            n=n,
            inset_px=1.0,
        ),
        tolerance_px=3.0,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-line-curve-line",
        description="Straight into curve into straight — tests line/cubic transitions",
        category="curve",
        draw=lambda draw, size, sw: _draw_sampled_path(
            draw,
            _concat_path_segments(
                _sample_line(size * 0.78, size * 0.12, size * 0.78, size * 0.34, 80),
                _sample_cubic(
                    p0=(size * 0.78, size * 0.34),
                    p1=(size * 0.78, size * 0.52),
                    p2=(size * 0.56, size * 0.62),
                    p3=(size * 0.34, size * 0.62),
                    n=120,
                ),
                _sample_line(size * 0.34, size * 0.62, size * 0.12, size * 0.62, 80),
            ),
            fill=0,
            width=sw,
        ),
        ideal_sample=lambda size, n: _concat_path_segments(
            _sample_line(size * 0.78, size * 0.12, size * 0.78, size * 0.34, max(40, n // 5)),
            _sample_cubic(
                p0=(size * 0.78, size * 0.34),
                p1=(size * 0.78, size * 0.52),
                p2=(size * 0.56, size * 0.62),
                p3=(size * 0.34, size * 0.62),
                n=max(120, (3 * n) // 5),
            ),
            _sample_line(size * 0.34, size * 0.62, size * 0.12, size * 0.62, max(40, n // 5)),
        ),
        tolerance_px=2.0,
        expected_segment_types=["line", "cubic"],
    ),

    FixtureDefinition(
        id="curve-to-straight-to-curve",
        description="Curve to straight to curve — tests cubic-line-cubic continuity",
        category="curve",
        draw=lambda draw, size, sw: _draw_sampled_path(
            draw,
            _concat_path_segments(
                _sample_cubic(
                    p0=(size * 0.78, size * 0.14),
                    p1=(size * 0.50, size * 0.14),
                    p2=(size * 0.38, size * 0.26),
                    p3=(size * 0.38, size * 0.40),
                    n=110,
                ),
                _sample_line(size * 0.38, size * 0.40, size * 0.38, size * 0.60, 70),
                _sample_cubic(
                    p0=(size * 0.38, size * 0.60),
                    p1=(size * 0.38, size * 0.74),
                    p2=(size * 0.50, size * 0.86),
                    p3=(size * 0.78, size * 0.86),
                    n=110,
                ),
            ),
            fill=0,
            width=sw,
        ),
        ideal_sample=lambda size, n: _concat_path_segments(
            _sample_cubic(
                p0=(size * 0.78, size * 0.14),
                p1=(size * 0.50, size * 0.14),
                p2=(size * 0.38, size * 0.26),
                p3=(size * 0.38, size * 0.40),
                n=max(110, (2 * n) // 5),
            ),
            _sample_line(size * 0.38, size * 0.40, size * 0.38, size * 0.60, max(50, n // 5)),
            _sample_cubic(
                p0=(size * 0.38, size * 0.60),
                p1=(size * 0.38, size * 0.74),
                p2=(size * 0.50, size * 0.86),
                p3=(size * 0.78, size * 0.86),
                n=max(110, (2 * n) // 5),
            ),
        ),
        tolerance_px=2.0,
        expected_segment_types=["line", "cubic"],
    ),

    FixtureDefinition(
        id="curve-perfect-circle",
        description="Perfect circle loop — closed circle target",
        category="curve",
        draw=lambda draw, size, sw: draw.ellipse(
            [
                int(size * 0.18), int(size * 0.18),
                int(size * 0.82), int(size * 0.82),
            ],
            outline=0,
            width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc_from_int_bbox(
            size=size,
            x0_frac=0.18,
            y0_frac=0.18,
            x1_frac=0.82,
            y1_frac=0.82,
            theta0_deg=0.0,
            theta1_deg=360.0,
            n=n,
            inset_px=1.0,
        ),
        tolerance_px=2.8,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="square-opposing-curves",
        description="Square loop with outward and inward curved opposite sides",
        category="curve",
        draw=lambda draw, size, sw: _draw_sampled_path(
            draw,
            _concat_path_segments(
                _sample_line(size * 0.18, size * 0.22, size * 0.18, size * 0.78, 80),
                _sample_cubic(
                    p0=(size * 0.18, size * 0.78),
                    p1=(size * 0.30, size * 0.87),
                    p2=(size * 0.70, size * 0.87),
                    p3=(size * 0.82, size * 0.78),
                    n=120,
                ),
                _sample_line(size * 0.82, size * 0.78, size * 0.82, size * 0.22, 80),
                _sample_cubic(
                    p0=(size * 0.82, size * 0.22),
                    p1=(size * 0.70, size * 0.38),
                    p2=(size * 0.30, size * 0.38),
                    p3=(size * 0.18, size * 0.22),
                    n=120,
                ),
            ),
            fill=0,
            width=sw,
            closed=True,
        ),
        ideal_sample=lambda size, n: _concat_path_segments(
            _sample_line(size * 0.18, size * 0.22, size * 0.18, size * 0.78, max(50, n // 6)),
            _sample_cubic(
                p0=(size * 0.18, size * 0.78),
                p1=(size * 0.30, size * 0.87),
                p2=(size * 0.70, size * 0.87),
                p3=(size * 0.82, size * 0.78),
                n=max(120, n // 3),
            ),
            _sample_line(size * 0.82, size * 0.78, size * 0.82, size * 0.22, max(50, n // 6)),
            _sample_cubic(
                p0=(size * 0.82, size * 0.22),
                p1=(size * 0.70, size * 0.38),
                p2=(size * 0.30, size * 0.38),
                p3=(size * 0.18, size * 0.22),
                n=max(120, n // 3),
            ),
        ),
        tolerance_px=2.4,
        expected_segment_types=["line", "cubic"],
    ),

    # ── Lines (sanity checks — fitter must still handle these) ────────────────

    FixtureDefinition(
        id="line-horizontal",
        description="Horizontal line across centre",
        category="line",
        draw=lambda draw, size, sw: draw.line(
            [(int(size * 0.1), size // 2), (int(size * 0.9), size // 2)],
            fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_line(
            size * 0.5, size * 0.1, size * 0.5, size * 0.9, n
        ),
        tolerance_px=1.0,
        expected_segment_types=["line"],
    ),

    FixtureDefinition(
        id="line-diagonal-45",
        description="45° diagonal — classic pixel staircase",
        category="line",
        draw=lambda draw, size, sw: draw.line(
            [(int(size * 0.1), int(size * 0.1)),
             (int(size * 0.9), int(size * 0.9))],
            fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_line(
            size * 0.1, size * 0.1, size * 0.9, size * 0.9, n
        ),
        tolerance_px=1.5,
        expected_segment_types=["line"],
    ),

    FixtureDefinition(
        id="line-horizontal-r3",
        description="Horizontal line rotated 3 degrees clockwise",
        category="line",
        draw=lambda draw, size, sw: _draw_sampled_path(
            draw,
            _rotate_points(
                _sample_line(size * 0.5, size * 0.1, size * 0.5, size * 0.9, 160),
                angle_deg=3.0,
                center_rc=(size * 0.5, size * 0.5),
            ),
            fill=0,
            width=sw,
        ),
        ideal_sample=lambda size, n: _rotate_points(
            _sample_line(size * 0.5, size * 0.1, size * 0.5, size * 0.9, n),
            angle_deg=3.0,
            center_rc=(size * 0.5, size * 0.5),
        ),
        tolerance_px=1.2,
        expected_segment_types=["line"],
    ),

    FixtureDefinition(
        id="square-perfect",
        description="Perfect axis-aligned square loop",
        category="line",
        draw=lambda draw, size, sw: _draw_sampled_path(
            draw,
            _sample_closed_polyline(_square_vertices(size, inset_frac=0.2), 240),
            fill=0,
            width=sw,
            closed=True,
        ),
        ideal_sample=lambda size, n: _sample_closed_polyline(
            _square_vertices(size, inset_frac=0.2),
            n,
        ),
        tolerance_px=1.5,
        expected_segment_types=["line"],
    ),

    FixtureDefinition(
        id="square-perfect-r3",
        description="Perfect square loop rotated 3 degrees clockwise",
        category="line",
        draw=lambda draw, size, sw: _draw_sampled_path(
            draw,
            _sample_closed_polyline(
                _rotate_points(
                    _square_vertices(size, inset_frac=0.22),
                    angle_deg=3.0,
                    center_rc=(size * 0.5, size * 0.5),
                ),
                240,
            ),
            fill=0,
            width=sw,
            closed=True,
        ),
        ideal_sample=lambda size, n: _sample_closed_polyline(
            _rotate_points(
                _square_vertices(size, inset_frac=0.22),
                angle_deg=3.0,
                center_rc=(size * 0.5, size * 0.5),
            ),
            n,
        ),
        tolerance_px=1.6,
        expected_segment_types=["line"],
    ),

    # ── Corners (tests multi-segment output) ──────────────────────────────────

    FixtureDefinition(
        id="corner-l-shape",
        description="L-shaped corner — two perpendicular lines meeting",
        category="corner",
        draw=lambda draw, size, sw: (
            draw.line(
                [(int(size * 0.15), int(size * 0.15)),
                 (int(size * 0.15), int(size * 0.85))],
                fill=0, width=sw,
            ),
            draw.line(
                [(int(size * 0.15), int(size * 0.85)),
                 (int(size * 0.85), int(size * 0.85))],
                fill=0, width=sw,
            ),
        ),
        ideal_sample=lambda size, n: np.vstack([
            _sample_line(size * 0.15, size * 0.15, size * 0.85, size * 0.15, n // 2),
            _sample_line(size * 0.85, size * 0.15, size * 0.85, size * 0.85, n - n // 2),
        ]),
        tolerance_px=1.5,
        expected_segment_types=["line"],
    ),

    FixtureDefinition(
        id="corner-l-shape-r45",
        description="L-shaped corner rotated 45° clockwise — two diagonal lines meeting",
        category="corner",
        # 45° CW rotation of corner-l-shape around image centre.
        # A=(top-centre), B=(middle-left, corner), C=(bottom-centre)
        # PIL coords are (x, y) = (col, row).
        draw=lambda draw, size, sw: (
            draw.line(
                [(int(size * 0.5), int(size * 0.1)),
                 (int(size * 0.1), int(size * 0.5))],
                fill=0, width=sw,
            ),
            draw.line(
                [(int(size * 0.1), int(size * 0.5)),
                 (int(size * 0.5), int(size * 0.9))],
                fill=0, width=sw,
            ),
        ),
        ideal_sample=lambda size, n: np.vstack([
            _sample_line(size * 0.1, size * 0.5, size * 0.5, size * 0.1, n // 2),
            _sample_line(size * 0.5, size * 0.1, size * 0.9, size * 0.5, n - n // 2),
        ]),
        tolerance_px=1.5,
        expected_segment_types=["line"],
    ),
]

FIXTURE_MAP: dict[str, FixtureDefinition] = {f.id: f for f in FIXTURES}


def get_fixture(fixture_id: str) -> FixtureDefinition:
    """Return a FixtureDefinition by id, raising KeyError if unknown."""
    if fixture_id not in FIXTURE_MAP:
        available = sorted(FIXTURE_MAP)
        raise KeyError(
            f"Unknown fixture id {fixture_id!r}. "
            f"Available: {available}"
        )
    return FIXTURE_MAP[fixture_id]
