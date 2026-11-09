"""Trace template: prepare a bitmap for hand-tracing in a CAD/vector program.

    python -m tests.pipeline.trace_template <image> [--fade 0.35] [--out <folder>]

Writes, outside the repo by default (``%TEMP%/centerline-trace-templates``):
  <name>-faded.png     the bitmap faded towards white, exact pixel size, 96 dpi
  <name>-template.svg  width/height = bitmap px, viewBox 0 0 W H, the faded image
                       embedded at 0,0 W x H, and no other shapes
  <name>-README.txt    three tracing instructions

The SVG's single <image> is the registration frame the reference reader uses, so
an export that keeps the image lines up with the bitmap whatever units, dpi or
Y-axis direction the program uses. ``build_template_files`` is stateless (bytes
in, bytes out) so a server can return the same files without storing anything.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_FADE = 0.35
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "centerline-trace-templates"
TEMPLATE_DPI = 96
README_TEXT = (
    "Draw one open path per stroke, along the centre of the stroke.\n"
    "Keep the image in the export (do not hide or delete it); it is used to line your paths up with the bitmap.\n"
    "Export as SVG.\n"
)


def fade_pixels(arr: np.ndarray, fade: float) -> np.ndarray:
    """Move every channel value towards white by ``fade`` (0 = unchanged, 1 = white)."""
    a = arr.astype(np.float64)
    return np.clip(np.rint(a + (255.0 - a) * fade), 0, 255).astype(np.uint8)


def _flatten(im: Image.Image) -> Image.Image:
    """Grayscale stays grayscale, anything else becomes RGB; transparency is composited onto white."""
    if im.mode in ("1", "L", "I", "I;16", "F"):
        return im.convert("L")
    if im.mode == "LA" or im.mode == "P" or "A" in im.getbands() or "transparency" in im.info:
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        out = Image.alpha_composite(white, rgba).convert("RGB")
        if im.mode in ("LA", "P") and all(np.ptp(np.asarray(out), axis=2).ravel() == 0):
            return out.convert("L")
        return out
    return im.convert("RGB")


def faded_png_bytes(image: Image.Image, fade: float = DEFAULT_FADE) -> bytes:
    if not 0.0 <= fade < 1.0:
        raise ValueError("fade must be in [0, 1)")
    flat = _flatten(image)
    faded = Image.fromarray(fade_pixels(np.asarray(flat), fade), mode=flat.mode)
    buf = io.BytesIO()
    faded.save(buf, format="PNG", dpi=(TEMPLATE_DPI, TEMPLATE_DPI), optimize=True)
    return buf.getvalue()


def template_svg(png_bytes: bytes, width: int, height: int) -> str:
    data = base64.b64encode(png_bytes).decode("ascii")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">\n'
        f'  <image id="trace-template-image" x="0" y="0" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" style="image-rendering:pixelated" '
        f'xlink:href="data:image/png;base64,{data}"/>\n'
        '</svg>\n'
    )


def build_template_files(image_bytes: bytes, name: str, fade: float = DEFAULT_FADE) -> dict[str, bytes]:
    """Stateless: bitmap bytes -> {filename: bytes} for the faded PNG, template SVG and README."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        im.load()
        width, height = im.size
        png = faded_png_bytes(im, fade)
    return {
        f"{name}-faded.png": png,
        f"{name}-template.svg": template_svg(png, width, height).encode("utf-8"),
        f"{name}-README.txt": README_TEXT.encode("utf-8"),
    }


def write_template(image_path: str | Path, out_dir: str | Path = DEFAULT_OUT_DIR,
                   fade: float = DEFAULT_FADE) -> dict[str, Path]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = build_template_files(image_path.read_bytes(), image_path.stem, fade)
    written = {}
    for fname, data in files.items():
        target = out_dir / fname
        target.write_bytes(data)
        written[fname] = target
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Make a faded bitmap and SVG template for hand-tracing centerlines.")
    ap.add_argument("image")
    ap.add_argument("--fade", type=float, default=DEFAULT_FADE,
                    help="how far to fade towards white, 0..<1 (default %(default)s)")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output folder (default %(default)s)")
    args = ap.parse_args(argv)
    if not 0.0 <= args.fade < 1.0:
        ap.error("--fade must be at least 0 and less than 1")
    written = write_template(args.image, args.out, args.fade)
    for path in written.values():
        print(f"{path}  ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
