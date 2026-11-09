"""Audit whether legacy fixture ``ideal_sample`` curves are valid ground truth
for centerline placement.

For each ideal sample point we scan along the local normal of the ideal curve
through the rasterised stroke (rendered in memory exactly like
tests/fixtures/generate.py, never written to tests/fixtures/data) and take the
darkness-weighted centroid of the dark run that contains / is nearest to the
point. The signed offset (ideal -> stroke centre) tells us whether the ideal
curve sits in the middle of the drawn stroke.

Usage:
    python -m tests.pipeline.fixture_audit [--size 64] [--stroke-width 3] [--markdown]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.definitions import FIXTURES  # noqa: E402

VALID_P95_PX = 0.5
VALID_OFF_STROKE_FRACTION = 0.02
VALID_ENDPOINT_PX = 1.5


def render_fixture(defn, size: int, stroke_width: int) -> np.ndarray:
    img = Image.new("L", (size, size), color=255)
    defn.draw(ImageDraw.Draw(img), size, stroke_width)
    return np.asarray(img, dtype=np.uint8)


def _normals(pts: np.ndarray) -> np.ndarray:
    tang = np.gradient(pts, axis=0)
    norm = np.hypot(tang[:, 0], tang[:, 1])
    norm[norm < 1e-12] = 1.0
    tang /= norm[:, None]
    return np.stack([-tang[:, 1], tang[:, 0]], axis=1)


def _stroke_endpoints(dark: np.ndarray, ideal: np.ndarray) -> float | None:
    """Distance from each ideal end to the farthest stroke pixel along the ideal tangent (open curves)."""
    if np.hypot(*(ideal[0] - ideal[-1])) < 1.0:
        return None
    rr, cc = np.nonzero(dark)
    stroke = np.stack([rr, cc], axis=1).astype(float)
    worst = 0.0
    for end, prev in ((ideal[0], ideal[min(5, len(ideal) - 1)]), (ideal[-1], ideal[max(-6, -len(ideal))])):
        t = end - prev
        t /= max(np.hypot(*t), 1e-9)
        near = stroke[np.hypot(*(stroke - end).T) <= 6.0]
        if not len(near):
            return float("inf")
        proj = (near - end) @ t
        worst = max(worst, abs(float(proj.max())))
    return worst


def audit_fixture(defn, size: int = 64, stroke_width: int = 3, n_points: int = 600) -> dict:
    pixels = render_fixture(defn, size, stroke_width)
    darkness = 1.0 - pixels.astype(float) / 255.0
    ideal = np.asarray(defn.ideal_sample(size, n_points), dtype=float)
    normals = _normals(ideal)
    half = stroke_width + 3.0
    s = np.arange(-half, half + 1e-9, 0.05)
    offsets = np.full(len(ideal), np.nan)
    for i, (p, n) in enumerate(zip(ideal, normals)):
        rows = p[0] + s * n[0]
        cols = p[1] + s * n[1]
        prof = map_coordinates(darkness, [rows, cols], order=1, mode="constant", cval=0.0)
        dark = prof >= 0.5
        if not dark.any():
            continue
        # contiguous runs; choose the one containing s=0 or nearest to it
        idx = np.flatnonzero(dark)
        splits = np.flatnonzero(np.diff(idx) > 1) + 1
        runs = np.split(idx, splits)
        centre_idx = int(np.argmin(np.abs(s)))
        run = min(runs, key=lambda r: 0 if r[0] <= centre_idx <= r[-1] else min(abs(r[0] - centre_idx),
                                                                                 abs(r[-1] - centre_idx)))
        w = prof[run]
        offsets[i] = float(np.sum(w * s[run]) / np.sum(w))
    valid = ~np.isnan(offsets)
    off = offsets[valid]
    absoff = np.abs(off)
    end_overrun = _stroke_endpoints(pixels < 128, ideal)
    result = {
        "id": defn.id,
        "category": defn.category,
        "size": size,
        "stroke_width": stroke_width,
        "points": int(len(ideal)),
        "off_stroke_fraction": float(1.0 - valid.mean()),
        "bias_px": float(np.mean(off)) if len(off) else None,
        "mean_abs_px": float(np.mean(absoff)) if len(off) else None,
        "p95_abs_px": float(np.percentile(absoff, 95)) if len(off) else None,
        "max_abs_px": float(np.max(absoff)) if len(off) else None,
        "end_mismatch_px": end_overrun,
    }
    reasons = []
    if result["off_stroke_fraction"] > VALID_OFF_STROKE_FRACTION:
        reasons.append(f"{result['off_stroke_fraction']:.1%} of ideal points are off the stroke")
    if result["p95_abs_px"] is None or result["p95_abs_px"] > VALID_P95_PX:
        reasons.append(f"p95 offset {result['p95_abs_px']:.2f}px > {VALID_P95_PX}px")
    if end_overrun is not None and end_overrun > VALID_ENDPOINT_PX:
        reasons.append(f"stroke extends {end_overrun:.2f}px past an ideal endpoint")
    result["valid_ground_truth"] = not reasons
    result["reasons"] = reasons
    return result


def audit_all(size: int = 64, stroke_width: int = 3) -> list[dict]:
    return [audit_fixture(d, size, stroke_width) for d in FIXTURES]


def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.{nd}f}"


def to_markdown(rows: list[dict]) -> str:
    out = ["| fixture | valid | bias px | mean abs px | p95 abs px | max abs px | off-stroke | end mismatch px | reasons |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['id']} | {'yes' if r['valid_ground_truth'] else '**no**'} | {_fmt(r['bias_px'])} | "
            f"{_fmt(r['mean_abs_px'])} | {_fmt(r['p95_abs_px'])} | {_fmt(r['max_abs_px'])} | "
            f"{r['off_stroke_fraction']:.1%} | {_fmt(r['end_mismatch_px'])} | {'; '.join(r['reasons']) or '-'} |"
        )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--stroke-width", type=int, default=3)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)
    rows = audit_all(args.size, args.stroke_width)
    if args.markdown:
        print(to_markdown(rows))
    else:
        for r in rows:
            print(f"{r['id']:28s} valid={r['valid_ground_truth']!s:5s} bias={_fmt(r['bias_px'])} "
                  f"p95={_fmt(r['p95_abs_px'])} off={r['off_stroke_fraction']:.1%} "
                  f"end={_fmt(r['end_mismatch_px'])} {'; '.join(r['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
