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
        ideal_sample=lambda size, n: _sample_arc(
            centre_rc=(size * 0.5, size * 0.5),
            radius=size * 0.45,
            theta0_deg=0.0,
            theta1_deg=90.0,
            n=n,
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
                int(size * 0.05), int(size * 0.1),
                int(size * 0.95), int(size * 0.9),
            ],
            start=0, end=180, fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc(
            centre_rc=(size * 0.5, size * 0.5),
            radius=size * 0.40,
            theta0_deg=0.0,
            theta1_deg=180.0,
            n=n,
        ),
        tolerance_px=2.5,
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
        ideal_sample=lambda size, n: _sample_cubic(
            p0=(size * 0.8, size * 0.1),   # (row, col)
            p1=(size * 0.2, size * 0.3),
            p2=(size * 0.8, size * 0.7),
            p3=(size * 0.2, size * 0.9),
            n=n,
        ),
        tolerance_px=1.5,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-gentle-convex",
        description="Gentle convex arc — large radius, ~30° span",
        category="curve",
        draw=lambda draw, size, sw: draw.arc(
            [
                int(-size * 1.0), int(size * 0.1),
                int(size * 2.0),  int(size * 0.9),
            ],
            start=355, end=5, fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc(
            centre_rc=(size * 0.5, size * 0.5),
            radius=size * 1.5,
            theta0_deg=355.0,
            theta1_deg=365.0,
            n=n,
        ),
        tolerance_px=2.0,
        expected_segment_types=["cubic"],
    ),

    FixtureDefinition(
        id="curve-tight-180",
        description="Tight 180° U-turn — tests multi-segment cubic fitting",
        category="curve",
        draw=lambda draw, size, sw: draw.arc(
            [
                int(size * 0.2), int(size * 0.1),
                int(size * 0.8), int(size * 0.9),
            ],
            start=180, end=360, fill=0, width=sw,
        ),
        ideal_sample=lambda size, n: _sample_arc(
            centre_rc=(size * 0.5, size * 0.5),
            radius=size * 0.30,
            theta0_deg=180.0,
            theta1_deg=360.0,
            n=n,
        ),
        tolerance_px=3.0,
        expected_segment_types=["cubic"],
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
