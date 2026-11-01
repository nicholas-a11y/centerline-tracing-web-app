from __future__ import annotations

import numpy as np

import centerline_engine_light as engine


def _rasterised_closed_ellipse(rx, ry, cx=50.0, cy=50.0):
    points = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False):
        p = (int(round(cx + rx * np.cos(angle))), int(round(cy + ry * np.sin(angle))))
        if not points or p != points[-1]:
            points.append(p)
    if points[-1] == points[0]:
        points.pop()
    points.append(points[0])
    return [list(map(float, p)) for p in points]


def _segments_to_polyline(segments):
    vertices = [np.asarray(segments[0]["start_point"], dtype=float)]
    for segment in segments:
        assert segment["type"] == "line"
        vertices.append(np.asarray(segment["end_point"], dtype=float))
    return np.asarray(vertices)


def _max_distance_to_polyline(points, polyline):
    a = polyline[:-1]
    ab = polyline[1:] - a
    lengths = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    worst = 0.0
    for p in np.asarray(points, dtype=float):
        t = np.clip(((p - a) * ab).sum(axis=1) / lengths, 0.0, 1.0)
        worst = max(worst, float(np.sqrt((((a + ab * t[:, None]) - p) ** 2).sum(axis=1)).min()))
    return worst


def test_line_only_fit_follows_small_closed_loops():
    for rx, ry in ((6, 4), (9, 5), (12, 7)):
        loop = _rasterised_closed_ellipse(rx, ry)
        for smoothing in (0.0, 76.0):
            segments = engine.fit_curve_segments(
                [loop],
                source_smoothing=smoothing,
                tolerance_px=0.35,
                enable_curve_fitting=False,
            )[0]
            polyline = _segments_to_polyline(segments)
            assert len(segments) >= 4
            assert _max_distance_to_polyline(loop, polyline) <= 1.6, (rx, ry, smoothing)
