"""Generate the built-in lessons under tests/lessons/ (run once; output is committed).

  * crossing lessons: thin line across a thick line, drawn with an exact
    distance-field rasteriser (pixel-centre convention), optional 4x4
    supersampled anti-aliasing.
  * placement lessons converted from tests/fixtures/definitions.py
    (64 px, stroke width 3), rendered in memory exactly like
    tests/fixtures/generate.py. Fixtures whose ideal_sample fails the
    ground-truth audit (fixture_audit.py) are written with status "draft".

Existing lesson.json "status" and "last_run" values are preserved, so
re-running never silently un-baselines a lesson.

Usage:
    python -m tests.pipeline.generate_lessons [--only crossing|placement] [--out tests/lessons]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.definitions import FIXTURES  # noqa: E402
from tests.pipeline.fixture_audit import audit_fixture, render_fixture  # noqa: E402
from tests.pipeline.lesson import DEFAULT_PARAMS, LESSONS_DIR, SCHEMA_VERSION, write_lesson_json  # noqa: E402
from tests.pipeline.scoring import densify  # noqa: E402
from tests.pipeline.svg_reference import write_reference_svg  # noqa: E402

LICENCE = "Generated test data; same licence as this repository."
CROSSING_EXPECTED = (
    "Both centerlines continue through the crossing in their own direction as single "
    "paths (no bridge segment, no kink toward the other line). Lateral error inside the "
    "crossing region stays within tolerance."
)


# ---------------------------------------------------------------- rasteriser

def _clip_to_image(pts: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    m = (pts[:, 0] >= 0) & (pts[:, 0] <= h - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= w - 1)
    idx = np.flatnonzero(m)
    return pts[idx[0]:idx[-1] + 1]


def straight_line(centre_rc, angle_deg, size_wh, step=0.1) -> np.ndarray:
    """Centreline through centre_rc at angle (0 = +col, positive = up/anticlockwise on screen)."""
    w, h = size_wh
    reach = math.hypot(w, h)
    a = math.radians(angle_deg)
    d = np.array([-math.sin(a), math.cos(a)])
    t = np.arange(-reach, reach + step, step)[:, None]
    return _clip_to_image(np.asarray(centre_rc) + t * d, size_wh)


def arc_line(centre_rc, angle_deg, radius, size_wh, step=0.1) -> np.ndarray:
    """Circular arc of given radius passing through centre_rc with tangent angle_deg there."""
    w, h = size_wh
    a = math.radians(angle_deg)
    tangent = np.array([-math.sin(a), math.cos(a)])
    normal = np.array([-tangent[1], tangent[0]])
    c = np.asarray(centre_rc) + radius * normal
    phi0 = math.atan2(*(np.asarray(centre_rc) - c))
    span = math.hypot(w, h) / radius
    phis = np.arange(phi0 - span, phi0 + span, step / radius)
    pts = c + radius * np.stack([np.sin(phis), np.cos(phis)], axis=1)
    return _clip_to_image(pts, size_wh)


def _extend(pts: np.ndarray, by: float) -> np.ndarray:
    """Extend a polyline straight past both ends so strokes run off the canvas."""
    d0 = pts[0] - pts[1]
    d1 = pts[-1] - pts[-2]
    d0 /= np.hypot(*d0)
    d1 /= np.hypot(*d1)
    return np.vstack([pts[0] + d0 * by, pts, pts[-1] + d1 * by])


def rasterise(strokes: list[tuple[np.ndarray, float]], size_wh: tuple[int, int], aa: bool) -> np.ndarray:
    """Return uint8 image (0 = ink). A pixel (or subsample) is ink if within width/2 of a centreline."""
    w, h = size_wh
    sub = 4 if aa else 1
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    ink = np.zeros((h, w), dtype=float)
    for dr in offs:
        for dc in offs:
            q = np.stack([(rr + dr).ravel(), (cc + dc).ravel()], axis=1)
            hit = np.zeros(len(q), dtype=bool)
            for line, width in strokes:
                tree = cKDTree(densify(_extend(line, width + 4), 0.05))
                dist, _ = tree.query(q, distance_upper_bound=width / 2 + 1)
                hit |= dist <= width / 2
            ink += hit.reshape(h, w)
    ink /= sub * sub
    return np.round(255 * (1.0 - ink)).astype(np.uint8)


def _crossing_region(centre_rc, thin_w, thick_w, angle_deg, size_wh):
    a = abs(math.radians(angle_deg))
    half = 0.5 * thick_w / max(math.sin(a), 1e-6) + 0.5 * thin_w / max(math.sin(a), 1e-6) + 2 * thick_w
    w, h = size_wh
    x, y = centre_rc[1] + 0.5, centre_rc[0] + 0.5
    return {"x0": round(max(0.0, x - half), 2), "y0": round(max(0.0, y - half), 2),
            "x1": round(min(float(w), x + half), 2), "y1": round(min(float(h), y + half), 2)}


# ------------------------------------------------------------------ lessons

CROSSINGS = [
    # id, thin, thick, angle, size, aa, curved_radius
    ("cross-t2-w8-a20", 2, 8, 20, 256, False, None),
    ("cross-t2-w8-a35", 2, 8, 35, 256, False, None),
    ("cross-t2-w8-a60", 2, 8, 60, 256, False, None),
    ("cross-t2-w8-a90", 2, 8, 90, 256, False, None),
    ("cross-t3-w12-a35", 3, 12, 35, 256, False, None),
    ("cross-t4-w16-a35", 4, 16, 35, 256, False, None),
    ("cross-t2-w8-a35-curved", 2, 8, 35, 256, False, 300.0),
    ("cross-t2-w8-a35-aa", 2, 8, 35, 256, True, None),
    ("cross-t2-w8-a60-aa", 2, 8, 60, 256, True, None),
    ("cross-t2-w8-a35-640px", 2, 8, 35, 640, False, None),
    ("cross-t8-w8-a60-equal", 8, 8, 60, 256, False, None),
]


def _merge_preserved(folder: Path, meta: dict) -> dict:
    existing = folder / "lesson.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        for key in ("status", "last_run", "notes"):
            if key in old:
                meta[key] = old[key]
    return meta


def build_crossing(spec, out_dir: Path) -> str:
    lid, thin, thick, angle, size, aa, radius = spec
    size_wh = (size, size)
    centre = ((size - 1) / 2.0, (size - 1) / 2.0)
    thick_line = straight_line(centre, 0.0, size_wh)
    thin_line = arc_line(centre, angle, radius, size_wh) if radius else straight_line(centre, angle, size_wh)
    img = rasterise([(thick_line, thick), (thin_line, thin)], size_wh, aa)
    folder = out_dir / lid
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(folder / "image.png")
    refs = [thick_line[[0, -1]], thin_line if radius else thin_line[[0, -1]]]
    if radius:
        refs[1] = refs[1][::5]
    write_reference_svg(folder / "reference.svg", refs, size_wh)
    desc = (f"{thin}px line crossing a {thick}px horizontal line at {angle} degrees on a {size}px canvas"
            f"{', anti-aliased edges' if aa else ''}{f', thin line curved (radius {radius:g}px)' if radius else ''}.")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "id": lid,
        "title": f"Crossing: {thin}px over {thick}px at {angle} deg" + (" (AA)" if aa else "")
                 + (" curved" if radius else "") + (f" {size}px" if size != 256 else ""),
        "category": "crossing",
        "description": desc,
        "notes": "",
        "expected_behaviour": CROSSING_EXPECTED,
        "source": {"kind": "generated", "generator": "tests/pipeline/generate_lessons.py", "fixture_id": None},
        "licence": LICENCE,
        "consent": {"public": True, "by": "repository maintainer"},
        "image": {"file": "image.png", "width": size, "height": size},
        "reference": {"file": "reference.svg"},
        "params": dict(DEFAULT_PARAMS),
        "scoring": {
            "region": _crossing_region(centre, thin, thick, angle, size_wh),
            "tolerance_px": 1.0,
            "min_single_path_coverage": 0.95,
            "require_single_path": True,
            "trim_ends_px": 0.0,
        },
        "status": "active",
    }
    write_lesson_json(folder, _merge_preserved(folder, meta))
    return lid


def build_placement(defn, out_dir: Path, size: int = 64, stroke_width: int = 3) -> str:
    lid = f"fixture-{defn.id}-{size}px-sw{stroke_width}"
    audit = audit_fixture(defn, size, stroke_width)
    folder = out_dir / lid
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_fixture(defn, size, stroke_width), mode="L").save(folder / "image.png")
    ideal = np.asarray(defn.ideal_sample(size, 400), dtype=float)
    closed = bool(np.hypot(*(ideal[0] - ideal[-1])) < 0.5)
    if closed:
        ideal = ideal[:-1]
    write_reference_svg(folder / "reference.svg", [ideal], (size, size), closed=[closed])
    audit_note = (
        f"Ground-truth audit: bias {audit['bias_px']:.2f}px, p95 offset {audit['p95_abs_px']:.2f}px"
        + ("" if audit["valid_ground_truth"] else " - INVALID: " + "; ".join(audit["reasons"]))
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "id": lid,
        "title": f"Placement: {defn.id} ({size}px, stroke {stroke_width})",
        "category": "placement",
        "description": defn.description,
        "notes": audit_note,
        "expected_behaviour": "The centerline follows the middle of the stroke as one path within tolerance.",
        "source": {"kind": "legacy-fixture", "generator": "tests/fixtures/definitions.py", "fixture_id": defn.id,
                   "size": size, "stroke_width": stroke_width},
        "licence": LICENCE,
        "consent": {"public": True, "by": "repository maintainer"},
        "image": {"file": "image.png", "width": size, "height": size},
        "reference": {"file": "reference.svg"},
        "params": dict(DEFAULT_PARAMS),
        "scoring": {
            "region": None,
            "tolerance_px": 1.0,
            "min_single_path_coverage": 0.95,
            "require_single_path": True,
            "trim_ends_px": float(stroke_width + 1),
        },
        "status": "active" if audit["valid_ground_truth"] else "draft",
        "ground_truth_audit": {k: audit[k] for k in ("bias_px", "mean_abs_px", "p95_abs_px", "max_abs_px",
                                                     "off_stroke_fraction", "end_mismatch_px",
                                                     "valid_ground_truth", "reasons")},
    }
    meta = _merge_preserved(folder, meta)
    if not audit["valid_ground_truth"]:
        meta["status"] = "draft"
        meta["notes"] = audit_note
    write_lesson_json(folder, meta)
    return lid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate built-in lessons.")
    ap.add_argument("--only", choices=("crossing", "placement"))
    ap.add_argument("--out", type=Path, default=LESSONS_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "crossing"):
        for spec in CROSSINGS:
            print("wrote", build_crossing(spec, args.out))
    if args.only in (None, "placement"):
        for defn in FIXTURES:
            print("wrote", build_placement(defn, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
