"""
tests/fixtures/generate.py — fixture factory CLI

Usage:
    python -m tests.fixtures.generate                  # generate all fixtures
    python -m tests.fixtures.generate --id curve-quarter-arc
    python -m tests.fixtures.generate --size 128 --stroke-width 4
    python -m tests.fixtures.generate --list

Each fixture is written as two files in tests/fixtures/data/:
    <id>_<size>px_sw<sw>.png   — greyscale rasterized stroke (0=stroke, 255=bg)
    <id>_<size>px_sw<sw>.json  — metadata (tolerance, expected_segment_types, etc.)

The ideal curve is NOT stored here — it is always re-derived analytically at
test time from FixtureDefinition.ideal_sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from .definitions import FIXTURES, get_fixture, FixtureDefinition

DATA_DIR = Path(__file__).parent / "data"


def fixture_stem(fixture_id: str, size: int, stroke_width: int) -> str:
    return f"{fixture_id}_{size}px_sw{stroke_width}"


def generate_fixture(
    defn: FixtureDefinition,
    size: int,
    stroke_width: int,
    data_dir: Path = DATA_DIR,
) -> tuple[Path, Path]:
    """Rasterize one fixture definition to a PNG + sidecar JSON.

    Returns (png_path, json_path).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = fixture_stem(defn.id, size, stroke_width)

    # ── Rasterize ──────────────────────────────────────────────────────────
    img = Image.new("L", (size, size), color=255)   # white background
    draw = ImageDraw.Draw(img)
    defn.draw(draw, size, stroke_width)

    png_path = data_dir / f"{stem}.png"
    img.save(png_path)

    # ── Coverage stats for sanity ──────────────────────────────────────────
    pixels = list(img.getdata())
    stroke_px = sum(1 for p in pixels if p < 128)
    coverage = stroke_px / len(pixels)

    # ── Sidecar JSON ───────────────────────────────────────────────────────
    meta = {
        "id": defn.id,
        "description": defn.description,
        "category": defn.category,
        "size": size,
        "stroke_width": stroke_width,
        "tolerance_px": defn.tolerance_px,
        "expected_segment_types": defn.expected_segment_types,
        "png_file": png_path.name,
        "stats": {
            "total_pixels": len(pixels),
            "stroke_pixels": stroke_px,
            "coverage": round(coverage, 4),
        },
    }

    json_path = data_dir / f"{stem}.json"
    json_path.write_text(json.dumps(meta, indent=2))

    return png_path, json_path


def generate_all(
    size: int,
    stroke_width: int,
    data_dir: Path = DATA_DIR,
    verbose: bool = True,
) -> list[tuple[Path, Path]]:
    results = []
    for defn in FIXTURES:
        png_path, json_path = generate_fixture(defn, size, stroke_width, data_dir)
        results.append((png_path, json_path))
        if verbose:
            print(f"  [{defn.category:6s}] {defn.id:30s}  →  {png_path.name}")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate curve-fitting test fixtures (PNG + JSON)"
    )
    parser.add_argument("--id", help="Generate only this fixture ID")
    parser.add_argument("--size", type=int, default=64, help="Canvas size in pixels (default: 64)")
    parser.add_argument("--stroke-width", type=int, default=3, help="Stroke width (default: 3)")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="Output directory")
    parser.add_argument("--list", action="store_true", help="List fixture IDs and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':35s} {'CATEGORY':8s} DESCRIPTION")
        print("─" * 75)
        for defn in FIXTURES:
            print(f"{defn.id:35s} {defn.category:8s} {defn.description}")
        sys.exit(0)

    print(f"Generating fixtures → {args.out}  (size={args.size}px, sw={args.stroke_width}px)")
    print()

    if args.id:
        defn = get_fixture(args.id)
        png_path, json_path = generate_fixture(defn, args.size, args.stroke_width, args.out)
        print(f"  [{defn.category}] {defn.id}  →  {png_path.name}")
    else:
        generate_all(args.size, args.stroke_width, args.out, verbose=True)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
