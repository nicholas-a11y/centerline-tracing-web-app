"""Loader for user-supplied cropped bitmaps in test_images/user_supplied/.

For an image ``name.png`` (or .jpg/.jpeg/.bmp/.tif/.tiff) optional sidecars are:
  * ``name.svg``  - reference centerlines (same rules as lesson reference.svg)
  * ``name.json`` - partial lesson metadata (title, notes, params, scoring,
                    status). Anything missing falls back to lesson defaults.
Images without a reference are reported (metrics/overlays) but not scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .lesson import DEFAULT_PARAMS, DEFAULT_SCORING, REPO_ROOT
from .svg_reference import read_reference_svg

USER_DIR = REPO_ROOT / "test_images" / "user_supplied"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class UserImage:
    id: str
    image_path: Path
    reference_path: Path | None
    meta: dict

    @property
    def status(self) -> str:
        return self.meta.get("status", "baseline" if self.reference_path else "draft")

    @property
    def params(self) -> dict:
        p = dict(DEFAULT_PARAMS)
        p.update(self.meta.get("params") or {})
        return p

    @property
    def scoring(self) -> dict:
        s = dict(DEFAULT_SCORING)
        s.update(self.meta.get("scoring") or {})
        return s

    @property
    def size_wh(self) -> tuple[int, int]:
        with Image.open(self.image_path) as im:
            return im.size

    def load_reference(self) -> list[np.ndarray] | None:
        if not self.reference_path:
            return None
        return read_reference_svg(self.reference_path, self.size_wh).paths

    def load_gray(self) -> np.ndarray:
        with Image.open(self.image_path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def discover_user_images(root: str | Path = USER_DIR) -> list[UserImage]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for img in sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        svg = img.with_suffix(".svg")
        js = img.with_suffix(".json")
        meta = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
        out.append(UserImage(id=f"user-{img.stem}", image_path=img,
                             reference_path=svg if svg.exists() else None, meta=meta))
    return out
