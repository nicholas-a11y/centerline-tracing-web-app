"""Dev-only: report how the reference-SVG reader interprets (or rejects) an SVG.

    python -m tests.pipeline.inspect_svg drawing.svg --bitmap image.png
    python -m tests.pipeline.inspect_svg drawing.svg --size 256x256
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image

from tests.pipeline.svg_reference import SvgReferenceError, read_reference_svg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svg")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--bitmap", help="bitmap the SVG must align with")
    group.add_argument("--size", help="bitmap size as WIDTHxHEIGHT")
    args = ap.parse_args(argv)

    if args.bitmap:
        with Image.open(args.bitmap) as im:
            size = im.size
    else:
        w, h = args.size.lower().split("x")
        size = (int(w), int(h))

    try:
        ref = read_reference_svg(args.svg, size)
    except SvgReferenceError as exc:
        print(f"REJECTED: {exc}")
        return 1

    print(f"accepted: alignment={ref.alignment} svg_size={ref.svg_size} bitmap_size={size}")
    for i, (pts, closed) in enumerate(zip(ref.paths, ref.closed)):
        pts = np.asarray(pts, dtype=float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        outside = int(((pts < -0.5) | (pts > np.array([size[1], size[0]]) - 0.5)).any(axis=1).sum())
        print(f"  path {i}: {len(pts)} pts, closed={closed}, length={length:.1f}px, "
              f"rows {lo[0]:.1f}..{hi[0]:.1f}, cols {lo[1]:.1f}..{hi[1]:.1f}"
              + (f", {outside} pts outside bitmap" if outside else ""))
    for w in ref.warnings:
        print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
