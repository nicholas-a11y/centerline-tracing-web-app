"""Engine-agnostic lesson format (folder per lesson: image.png, reference.svg, lesson.json).

See docs/migration/lesson-format.md for the field reference.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .svg_reference import read_reference_svg

REPO_ROOT = Path(__file__).resolve().parents[1]
LESSONS_DIR = REPO_ROOT / "tests" / "lessons"
SCHEMA_VERSION = 1
STATUSES = ("active", "baseline", "draft")
CATEGORIES = ("crossing", "placement", "user")

DEFAULT_PARAMS = {
    "dark_threshold": 0.20,
    "min_path_length": 2,
    "enable_pruning": False,
    "normalization_mode": "auto",
    "normalization_sensitivity": "medium",
}

# Allowed in lesson.json "params" but never injected into Lesson.params, so existing
# lessons run exactly as before; values match the main UI defaults (templates/index.html).
OPTIONAL_PARAMS = {
    "enable_post_fit_export": False,
    "source_smoothing": 70,
    "enable_curve_fitting": False,
    "cubic_fit_tolerance": 0.35,
    "force_orthogonal_as_lines": False,
}
ALLOWED_PARAMS = (*DEFAULT_PARAMS, *OPTIONAL_PARAMS)

DEFAULT_SCORING = {
    "region": None,                 # {"x0","y0","x1","y1"} in SVG pixel units, or null = whole image
    "tolerance_px": 1.0,            # p95 lateral error limit inside the region
    "min_single_path_coverage": 0.95,
    "require_single_path": True,
    "trim_ends_px": 0.0,            # ignore this much of each reference end
}


class LessonError(ValueError):
    pass


@dataclass
class Lesson:
    folder: Path
    meta: dict

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def status(self) -> str:
        return self.meta.get("status", "draft")

    @property
    def image_path(self) -> Path:
        return self.folder / self.meta["image"]["file"]

    @property
    def reference_path(self) -> Path:
        return self.folder / self.meta["reference"]["file"]

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
        return int(self.meta["image"]["width"]), int(self.meta["image"]["height"])

    def load_reference(self) -> list[np.ndarray]:
        return read_reference_svg(self.reference_path, self.size_wh).paths

    def load_gray(self) -> np.ndarray:
        with Image.open(self.image_path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def validate_meta(meta: dict, folder: Path | None = None) -> list[str]:
    errors = []
    for key in ("schema_version", "id", "title", "category", "image", "reference", "status"):
        if key not in meta:
            errors.append(f"missing field {key!r}")
    if errors:
        return errors
    if meta["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if meta["status"] not in STATUSES:
        errors.append(f"status must be one of {STATUSES}")
    if meta["category"] not in CATEGORIES:
        errors.append(f"category must be one of {CATEGORIES}")
    if folder is not None and meta["id"] != folder.name:
        errors.append(f"id {meta['id']!r} must equal folder name {folder.name!r}")
    unknown = set(meta.get("params") or {}) - set(ALLOWED_PARAMS)
    if unknown:
        errors.append(f"unknown params {sorted(unknown)}; allowed params are {sorted(ALLOWED_PARAMS)}")
    region = (meta.get("scoring") or {}).get("region")
    if region is not None and not all(k in region for k in ("x0", "y0", "x1", "y1")):
        errors.append("scoring.region needs x0, y0, x1, y1")
    if folder is not None:
        img = folder / meta["image"]["file"]
        if not img.exists():
            errors.append(f"image file {img.name} missing")
        else:
            with Image.open(img) as im:
                if im.size != (meta["image"].get("width"), meta["image"].get("height")):
                    errors.append(f"image size {im.size} does not match lesson.json image width/height")
        if not (folder / meta["reference"]["file"]).exists():
            errors.append(f"reference file {meta['reference']['file']} missing")
    return errors


def load_lesson(folder: str | Path) -> Lesson:
    folder = Path(folder)
    meta = json.loads((folder / "lesson.json").read_text(encoding="utf-8"))
    errors = validate_meta(meta, folder)
    if errors:
        raise LessonError(f"{folder.name}: " + "; ".join(errors))
    return Lesson(folder=folder, meta=meta)


def discover_lessons(root: str | Path = LESSONS_DIR) -> list[Lesson]:
    root = Path(root)
    if not root.exists():
        return []
    return [load_lesson(p) for p in sorted(root.iterdir()) if (p / "lesson.json").exists()]


def write_lesson_json(folder: Path, meta: dict) -> None:
    (folder / "lesson.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def engine_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "centerline_engine_light.py",
                                "centerline_engine.py", "centerline_web_app.py", "centerline_core.py"],
                               cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None
