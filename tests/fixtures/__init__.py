"""
tests/fixtures/__init__.py

Public API for loading fixtures in tests:

    from tests.fixtures import load_fixture, LoadedFixture
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .definitions import FIXTURES, FIXTURE_MAP, get_fixture, FixtureDefinition
from .generate import DATA_DIR, fixture_stem, generate_fixture

__all__ = [
    "LoadedFixture",
    "load_fixture",
    "FIXTURES",
    "FIXTURE_MAP",
    "fixture_stem",
]


@dataclass
class LoadedFixture:
    """A rasterized fixture ready for use in a test."""

    id: str
    description: str
    category: str

    # Raw pixel data
    pixels: np.ndarray   # shape (height, width), dtype uint8, 0=stroke, 255=bg
    # Float normalised, ready for extract_skeleton_paths (0.0=stroke, 1.0=bg)
    gray: np.ndarray     # shape (height, width), dtype float32

    width: int
    height: int

    # Curve quality contract thresholds (from FixtureDefinition)
    tolerance_px: float
    expected_segment_types: list[str]

    stroke_width: int

    # Back-reference to the definition so tests can call ideal_sample()
    defn: FixtureDefinition


def load_fixture(
    fixture_id: str,
    size: int = 64,
    stroke_width: int = 3,
    data_dir: Path = DATA_DIR,
    auto_generate: bool = True,
) -> LoadedFixture:
    """Load a fixture by ID, auto-generating the PNG/JSON if needed.

    Args:
        fixture_id:     One of the IDs defined in tests/fixtures/definitions.py.
        size:           Canvas size in pixels.
        stroke_width:   Stroke width used when rasterizing.
        data_dir:       Where fixture files live (default: tests/fixtures/data/).
        auto_generate:  Create the fixture file if it doesn't already exist.

    Returns:
        LoadedFixture ready for use in a test.

    Raises:
        KeyError:          If fixture_id is not in the fixture bank.
        FileNotFoundError: If the file is missing and auto_generate=False.
    """
    defn = get_fixture(fixture_id)
    stem = fixture_stem(fixture_id, size, stroke_width)
    png_path = data_dir / f"{stem}.png"
    json_path = data_dir / f"{stem}.json"

    if not png_path.exists():
        if not auto_generate:
            raise FileNotFoundError(
                f"Fixture file not found: {png_path}\n"
                f"Run: python -m tests.fixtures.generate --id {fixture_id}"
            )
        generate_fixture(defn, size, stroke_width, data_dir)

    img = Image.open(png_path).convert("L")
    pixels = np.array(img, dtype=np.uint8)
    meta = json.loads(json_path.read_text())

    return LoadedFixture(
        id=fixture_id,
        description=defn.description,
        category=defn.category,
        pixels=pixels,
        gray=pixels.astype(np.float32) / 255.0,
        width=img.width,
        height=img.height,
        tolerance_px=meta["tolerance_px"],
        expected_segment_types=meta["expected_segment_types"],
        stroke_width=stroke_width,
        defn=defn,
    )
