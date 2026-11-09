"""Read-only, defensive SVG reader for reference centerlines (stdlib + numpy only).

Output paths use the engine convention: (row, col) floats where integer
coordinates are pixel centres. SVG coordinates are pixel-edge based, so a
registered point (x, y) in bitmap pixel units maps to (row, col) =
(y - 0.5, x - 0.5), matching the +0.5 shift that ``create_svg_output``
applies on export.

Supported:
  * affine ``transform`` lists (matrix, translate, scale, rotate, skewX,
    skewY) on any element, composed through nested groups in document order
  * lengths in px (or unitless), mm, cm, in, pt and pc at 96 px/in
  * registration of SVG coordinates onto the bitmap, in this order:
      1. ``image-frame``: exactly one ``<image>``; its rectangle, after all
         transforms, is the bitmap's full pixel frame in the bitmap's upright
         orientation (LightBurn exports). Root width/height units and the
         viewBox are not used. The embedded pixel data is never decoded; only
         the PNG IHDR / JPEG SOF header is read for its stored width/height. An
         aspect ratio more than 1% off the bitmap's is ``svg_bitmap_mismatch``;
         a different pixel size with the same aspect ratio is a warning.
      2. ``border-rect``: a <rect> marked id/class/data-role="border".
      3. ``size-match``: width/height in CSS px, or the viewBox, equal to the
         bitmap's pixel size.

Rejected with clear errors: DOCTYPE/ENTITY, ``<use>``, nested ``<svg>``,
percentages where a size is needed, rotated or skewed registration frames.
Styles and non-rendered subtrees (defs, clipPath, mask, symbol, ...) are ignored.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_SVG_BYTES = 2 * 1024 * 1024
MAX_GEOMETRY_ELEMENTS = 200
MAX_SAMPLED_POINTS = 200_000
MAX_TREE_DEPTH = 64
SAMPLE_STEP_PX = 0.5
FRAME_ASPECT_TOLERANCE = 0.01
MAX_EMBEDDED_IMAGE_DIM = 16384    # cap on the width/height an embedded image header may declare
EMBEDDED_HEADER_BYTES = 256 * 1024  # bytes of an embedded image decoded from base64 to find its header

_FORBIDDEN_MARKUP = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_LENGTH = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z%]*)\s*$", re.IGNORECASE)
_TRANSFORM_ITEM = re.compile(r"[\s,]*(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^()]*)\)")
_ARGS_ONLY = re.compile(r"^[\s,]*(?:[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?[\s,]*)*$")
_BORDER_MARKERS = {"border", "alignment-border", "bitmap-border"}
_UNIT_PX = {"": 1.0, "px": 1.0, "mm": 96.0 / 25.4, "cm": 96.0 / 2.54, "in": 96.0, "pt": 96.0 / 72.0,
            "pc": 16.0}
_GEOMETRY_TAGS = ("path", "line", "polyline", "polygon", "rect", "circle", "ellipse")
_NON_RENDERED = {"defs", "clipPath", "mask", "pattern", "marker", "symbol", "metadata", "title", "desc",
                 "style", "script", "linearGradient", "radialGradient", "filter", "foreignObject"}


class SvgReferenceError(ValueError):
    """Raised with a precise, user-fixable message when an SVG is rejected.

    ``code`` is a stable machine-readable reason (see docs/teach/design.md §14.4).
    """

    def __init__(self, message: str, code: str = "svg_parse"):
        super().__init__(message)
        self.code = code


@dataclass
class SvgReference:
    paths: list[np.ndarray]           # each (N, 2) [[row, col]] in bitmap pixel-centre coords
    closed: list[bool]
    svg_size: tuple[float, float]     # (width, height) of the registration frame in SVG canvas units
    alignment: str                    # "image-frame" | "border-rect" | "size-match"
    warnings: list[str] = field(default_factory=list)
    frame: dict | None = None         # image-frame details (never includes pixel data)


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _split_length(value: str | None, what: str) -> tuple[float, str] | None:
    if value is None:
        return None
    m = _LENGTH.match(value)
    if not m:
        raise SvgReferenceError(f"Cannot parse {what}={value!r}; use a number with an optional unit.",
                                "svg_units_unsupported")
    unit = m.group(2).lower()
    if unit != "%" and unit not in _UNIT_PX:
        raise SvgReferenceError(
            f"{what}={value!r} uses unit {unit!r}; supported units are px, mm, cm, in, pt and pc.",
            "svg_units_unsupported")
    return float(m.group(1)), unit


def _parse_length(value: str | None, what: str) -> float | None:
    """Length in px (user units). Percentages are rejected where a length is required."""
    parts = _split_length(value, what)
    if parts is None:
        return None
    num, unit = parts
    if unit == "%":
        raise SvgReferenceError(f"{what}={value!r} is a percentage; use an absolute length.",
                                "svg_units_unsupported")
    return num * _UNIT_PX[unit]


def _numbers(text: str | None) -> list[float]:
    return [float(n) for n in _NUMBER.findall(text or "")]


class _Affine:
    """2-D affine map in SVG matrix(a b c d e f) form: x' = a x + c y + e, y' = b x + d y + f."""

    __slots__ = ("m",)

    def __init__(self, m: np.ndarray | None = None):
        self.m = np.eye(3) if m is None else m

    @classmethod
    def svg(cls, a, b, c, d, e, f) -> "_Affine":
        return cls(np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=float))

    @classmethod
    def scale_translate(cls, sx, sy, tx, ty) -> "_Affine":
        return cls.svg(sx, 0.0, 0.0, sy, tx, ty)

    def __matmul__(self, other: "_Affine") -> "_Affine":
        """``self @ other`` applies ``other`` first, then ``self``."""
        return _Affine(self.m @ other.m)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return xy @ self.m[:2, :2].T + self.m[:2, 2]

    @property
    def max_scale(self) -> float:
        return max(float(np.linalg.svd(self.m[:2, :2], compute_uv=False).max()), 1e-9)

    @property
    def axis_aligned(self) -> bool:
        """True when the map only scales/flips/translates each axis (no rotation or skew)."""
        s = self.max_scale
        return abs(self.m[0, 1]) <= 1e-9 * s and abs(self.m[1, 0]) <= 1e-9 * s


def _parse_transform(text: str, where: str) -> _Affine:
    out = _Affine()
    pos = 0
    text = text or ""
    for m in _TRANSFORM_ITEM.finditer(text):
        if m.start() != pos:
            break
        name, arg_text = m.group(1), m.group(2)
        if not _ARGS_ONLY.match(arg_text):
            raise SvgReferenceError(f"{where}: cannot parse transform arguments {arg_text.strip()!r}.",
                                    "svg_transform_invalid")
        a = _numbers(arg_text)
        n = len(a)
        if name == "matrix" and n == 6:
            t = _Affine.svg(*a)
        elif name == "translate" and n in (1, 2):
            t = _Affine.scale_translate(1.0, 1.0, a[0], a[1] if n == 2 else 0.0)
        elif name == "scale" and n in (1, 2):
            t = _Affine.scale_translate(a[0], a[1] if n == 2 else a[0], 0.0, 0.0)
        elif name == "rotate" and n in (1, 3):
            r = math.radians(a[0])
            cr, sr = math.cos(r), math.sin(r)
            t = _Affine.svg(cr, sr, -sr, cr, 0.0, 0.0)
            if n == 3:
                t = (_Affine.scale_translate(1.0, 1.0, a[1], a[2]) @ t
                     @ _Affine.scale_translate(1.0, 1.0, -a[1], -a[2]))
        elif name == "skewX" and n == 1:
            t = _Affine.svg(1.0, 0.0, math.tan(math.radians(a[0])), 1.0, 0.0, 0.0)
        elif name == "skewY" and n == 1:
            t = _Affine.svg(1.0, math.tan(math.radians(a[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            raise SvgReferenceError(f"{where}: {name}() does not take {n} argument(s).", "svg_transform_invalid")
        out = out @ t
        pos = m.end()
    if text[pos:].strip(" \t\r\n,"):
        raise SvgReferenceError(f"{where}: cannot parse transform {text.strip()!r}.", "svg_transform_invalid")
    if not np.all(np.isfinite(out.m)) or abs(np.linalg.det(out.m[:2, :2])) < 1e-12:
        raise SvgReferenceError(f"{where}: transform {text.strip()!r} is degenerate (zero scale).",
                                "svg_transform_invalid")
    return out


class _Budget:
    def __init__(self, max_points: int):
        self.limit = int(max_points)
        self.remaining = int(max_points)

    def take(self, n: int) -> None:
        self.remaining -= int(n)
        if self.remaining < 0:
            raise SvgReferenceError(
                f"SVG geometry is too detailed (more than {self.limit} sampled points). "
                "Simplify the paths or crop to the problem area."
            )


def _segments_for(length_user: float, step_user: float, budget: _Budget) -> int:
    n = max(2, int(math.ceil(max(length_user, 0.0) / max(step_user, 1e-9))))
    budget.take(n)
    return n


def _cubic(p0, p1, p2, p3, step, budget):
    approx = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
    t = np.linspace(0.0, 1.0, _segments_for(approx, step, budget) + 1)[1:, None]
    mt = 1.0 - t
    return mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3


def _quad(p0, p1, p2, step, budget):
    approx = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1)
    t = np.linspace(0.0, 1.0, _segments_for(approx, step, budget) + 1)[1:, None]
    mt = 1.0 - t
    return mt**2 * p0 + 2 * mt * t * p1 + t**2 * p2


def _line(p0, p1, step, budget):
    t = np.linspace(0.0, 1.0, _segments_for(np.linalg.norm(p1 - p0), step, budget) + 1)[1:, None]
    return p0 + (p1 - p0) * t


def _arc(p0, rx, ry, phi_deg, large, sweep, p1, step, budget):
    """SVG elliptical arc, endpoint -> centre parameterisation (SVG 1.1 F.6.5/F.6.6)."""
    if np.allclose(p0, p1):
        return np.zeros((0, 2))
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:
        return _line(p0, p1, step, budget)
    phi = math.radians(phi_deg % 360.0)
    cphi, sphi = math.cos(phi), math.sin(phi)
    dx, dy = (p0 - p1) / 2.0
    x1p = cphi * dx + sphi * dy
    y1p = -sphi * dx + cphi * dy
    lam = (x1p**2) / (rx**2) + (y1p**2) / (ry**2)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    coef = math.sqrt(max(0.0, num / den)) if den > 0 else 0.0
    if bool(large) == bool(sweep):
        coef = -coef
    cxp = coef * (rx * y1p / ry)
    cyp = coef * (-ry * x1p / rx)
    mid = (p0 + p1) / 2.0
    cx = cphi * cxp - sphi * cyp + mid[0]
    cy = sphi * cxp + cphi * cyp + mid[1]

    def _ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    theta1 = _ang(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = _ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi
    n = _segments_for(abs(dtheta) * max(rx, ry), step, budget)
    t = theta1 + dtheta * np.linspace(0.0, 1.0, n + 1)[1:]
    xs = cx + rx * np.cos(t) * cphi - ry * np.sin(t) * sphi
    ys = cy + rx * np.cos(t) * sphi + ry * np.sin(t) * cphi
    out = np.stack([xs, ys], axis=1)
    out[-1] = p1
    return out


def _tokenize_path(d: str) -> list[str]:
    tokens = []
    pos = 0
    d = d or ""
    for m in _PATH_TOKEN.finditer(d):
        gap = d[pos:m.start()]
        if gap.strip(" \t\r\n,"):
            raise SvgReferenceError(f"Unexpected characters {gap.strip()!r} in path data.")
        tokens.append(m.group(0))
        pos = m.end()
    if d[pos:].strip(" \t\r\n,"):
        raise SvgReferenceError(f"Unexpected trailing characters {d[pos:].strip()!r} in path data.")
    return tokens


def _split_arc_flags(tokens: list[str]) -> list[str]:
    """Compact arc flags like 'a5 5 0 011 10 10' tokenise as '011'; split them."""
    out: list[str] = []
    queue = list(tokens)
    cmd = None
    arg_index = 0
    while queue:
        tok = queue.pop(0)
        if tok.isalpha():
            cmd = tok
            arg_index = 0
            out.append(tok)
            continue
        if cmd in ("A", "a") and arg_index % 7 in (3, 4) and len(tok) > 1 and tok[0] in "01" \
                and tok[1] != ".":
            out.append(tok[0])
            queue.insert(0, tok[1:])
            arg_index += 1
            continue
        out.append(tok)
        arg_index += 1
    return out


def _parse_path_d(d: str, step: float, budget: _Budget) -> list[tuple[np.ndarray, bool]]:
    tokens = _split_arc_flags(_tokenize_path(d))
    subpaths: list[tuple[np.ndarray, bool]] = []
    current: list[np.ndarray] = []
    pos = np.zeros(2)
    start = np.zeros(2)
    last_ctrl_c = None
    last_ctrl_q = None
    i = 0
    cmd = None

    def flush(closed=False):
        nonlocal current
        if current:
            pts = np.vstack(current)
            if len(pts) >= 2:
                subpaths.append((pts, closed))
        current = []

    def take(n):
        nonlocal i
        if i + n > len(tokens) or any(t.isalpha() for t in tokens[i:i + n]):
            raise SvgReferenceError(f"Path command {cmd!r} is missing numeric arguments.")
        vals = [float(t) for t in tokens[i:i + n]]
        i += n
        return vals

    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if not subpaths and not current and cmd not in "Mm":
                raise SvgReferenceError("Path data must start with a moveto (M/m) command.")
            if cmd in "Zz":
                if current:
                    if not np.allclose(pos, start):
                        current.append(_line(pos, start, step, budget))
                    pos = start.copy()
                    flush(closed=True)
                last_ctrl_c = last_ctrl_q = None
                continue
        elif cmd is None:
            raise SvgReferenceError("Path data must start with a moveto (M/m) command.")
        elif cmd in "Zz":
            raise SvgReferenceError("Numbers after Z/z without a new command.")
        rel = cmd.islower()
        base = pos if rel else np.zeros(2)
        c = cmd.upper()
        if c == "M":
            x, y = take(2)
            flush()
            pos = base + (x, y)
            start = pos.copy()
            current = [pos.reshape(1, 2)]
            budget.take(1)
            cmd = "l" if rel else "L"  # further coordinate pairs are implicit linetos
            last_ctrl_c = last_ctrl_q = None
            continue
        if not current:
            current = [pos.reshape(1, 2)]
            start = pos.copy()
        if c == "L":
            x, y = take(2)
            nxt = base + (x, y)
            current.append(_line(pos, nxt, step, budget))
            pos = nxt
            last_ctrl_c = last_ctrl_q = None
        elif c == "H":
            (x,) = take(1)
            nxt = np.array([(pos[0] if rel else 0.0) + x, pos[1]])
            current.append(_line(pos, nxt, step, budget))
            pos = nxt
            last_ctrl_c = last_ctrl_q = None
        elif c == "V":
            (y,) = take(1)
            nxt = np.array([pos[0], (pos[1] if rel else 0.0) + y])
            current.append(_line(pos, nxt, step, budget))
            pos = nxt
            last_ctrl_c = last_ctrl_q = None
        elif c == "C":
            a = take(6)
            p1, p2, p3 = base + a[0:2], base + a[2:4], base + a[4:6]
            current.append(_cubic(pos, p1, p2, p3, step, budget))
            last_ctrl_c, last_ctrl_q, pos = p2, None, p3
        elif c == "S":
            a = take(4)
            p1 = 2 * pos - last_ctrl_c if last_ctrl_c is not None else pos.copy()
            p2, p3 = base + a[0:2], base + a[2:4]
            current.append(_cubic(pos, p1, p2, p3, step, budget))
            last_ctrl_c, last_ctrl_q, pos = p2, None, p3
        elif c == "Q":
            a = take(4)
            p1, p2 = base + a[0:2], base + a[2:4]
            current.append(_quad(pos, p1, p2, step, budget))
            last_ctrl_q, last_ctrl_c, pos = p1, None, p2
        elif c == "T":
            a = take(2)
            p1 = 2 * pos - last_ctrl_q if last_ctrl_q is not None else pos.copy()
            p2 = base + a[0:2]
            current.append(_quad(pos, p1, p2, step, budget))
            last_ctrl_q, last_ctrl_c, pos = p1, None, p2
        elif c == "A":
            rx, ry, rot, large, sweep, x, y = take(7)
            nxt = base + (x, y)
            seg = _arc(pos, rx, ry, rot, int(large), int(sweep), nxt, step, budget)
            if len(seg):
                current.append(seg)
            pos = nxt
            last_ctrl_c = last_ctrl_q = None
        else:
            raise SvgReferenceError(f"Unsupported path command {cmd!r}.")
    flush()
    return subpaths


def _points_attr(text: str | None) -> np.ndarray:
    vals = _numbers(text)
    if len(vals) < 4 or len(vals) % 2:
        raise SvgReferenceError("points attribute needs an even number of coordinates (at least 2 points).")
    return np.asarray(vals, dtype=float).reshape(-1, 2)


def _densify_polyline(pts: np.ndarray, step: float, budget: _Budget) -> np.ndarray:
    parts = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        parts.append(_line(a, b, step, budget))
    return np.vstack(parts)


def _is_border(elem: ET.Element) -> bool:
    ids = {
        (elem.get("id") or "").strip().lower(),
        (elem.get("data-role") or "").strip().lower(),
    } | {c.lower() for c in (elem.get("class") or "").split()}
    return bool(ids & _BORDER_MARKERS)


def _f(elem: ET.Element, name: str, default: float = 0.0) -> float:
    v = _parse_length(elem.get(name), f"<{_local(elem.tag)}> {name}")
    return default if v is None else v


def _walk(root: ET.Element) -> list[tuple[ET.Element, _Affine]]:
    """Rendered elements with their current transformation matrix (element user units -> canvas)."""
    out: list[tuple[ET.Element, _Affine]] = []

    def visit(elem: ET.Element, parent: _Affine, depth: int) -> None:
        if depth > MAX_TREE_DEPTH:
            raise SvgReferenceError(f"SVG nesting is deeper than {MAX_TREE_DEPTH} levels.", "svg_parse")
        name = _local(elem.tag)
        if name in _NON_RENDERED:
            return
        ctm = parent
        if elem.get("transform") is not None:
            ctm = parent @ _parse_transform(elem.get("transform"), f"<{name}>")
        out.append((elem, ctm))
        for child in elem:
            visit(child, ctm, depth + 1)

    visit(root, _Affine(), 0)
    return out


def _rect_frame(x: float, y: float, w: float, h: float, ctm: _Affine) -> tuple[np.ndarray, np.ndarray]:
    corners = ctm.apply(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    return corners.min(axis=0), corners.max(axis=0)


def _frame_mapping(lo: np.ndarray, hi: np.ndarray, bw: float, bh: float) -> _Affine:
    ax, ay = bw / (hi[0] - lo[0]), bh / (hi[1] - lo[1])
    return _Affine.scale_translate(ax, ay, -lo[0] * ax, -lo[1] * ay)


def _aspect_off(lo: np.ndarray, hi: np.ndarray, bw: float, bh: float) -> float:
    span = hi - lo
    return abs((span[0] / span[1]) / (bw / bh) - 1.0)


def _href(elem: ET.Element) -> str:
    for k, v in elem.attrib.items():
        if k == "href" or k.endswith("}href"):
            return v or ""
    return ""


def _href_kind(elem: ET.Element) -> str:
    href = _href(elem)
    if href.startswith("data:"):
        return href[5:].split(";", 1)[0].split(",", 1)[0] or "data"
    return "external" if href else "none"


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# JPEG start-of-frame markers (baseline, progressive, lossless, arithmetic); not DHT/JPG/DAC.
_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
_JPEG_NO_LENGTH = {0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7}


def _header_dimensions(head: bytes) -> tuple[str, int, int] | None:
    """(format, width, height) from a PNG IHDR or JPEG SOF header; pixel data is never touched."""
    if head.startswith(_PNG_SIGNATURE):
        if len(head) >= 24 and head[12:16] == b"IHDR":
            return "png", int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        return None
    if head.startswith(b"\xff\xd8"):
        i = 2
        while i + 4 <= len(head):
            if head[i] != 0xFF:
                return None
            marker = head[i + 1]
            if marker == 0xFF:  # fill byte
                i += 1
                continue
            if marker in _JPEG_NO_LENGTH:
                i += 2
                continue
            if marker in (0xD9, 0xDA):  # EOI / start of scan before any SOF
                return None
            if marker in _JPEG_SOF:
                if i + 9 > len(head):
                    return None
                return "jpeg", int.from_bytes(head[i + 7:i + 9], "big"), int.from_bytes(head[i + 5:i + 7], "big")
            i += 2 + int.from_bytes(head[i + 2:i + 4], "big")
    return None


def _embedded_image_size(elem: ET.Element) -> tuple[str, int, int] | str:
    """Stored size of a base64 data-URI image, read from its header only.

    Only the first EMBEDDED_HEADER_BYTES are base64-decoded (enough for PNG IHDR and
    for JPEG SOF after typical EXIF/ICC segments); pixel data is never decoded.
    Returns (format, width, height), or a reason string when the size cannot be read.
    """
    href = _href(elem)
    if not href.startswith("data:"):
        return "the image is linked, not embedded" if href else "the image has no href"
    meta, sep, payload = href.partition(",")
    if not sep or not meta.lower().endswith(";base64"):
        return "the embedded image is not base64-encoded"
    chunk = "".join(payload[:EMBEDDED_HEADER_BYTES * 4 // 3 + 64].split())
    chunk = chunk[:len(chunk) // 4 * 4]
    try:
        head = base64.b64decode(chunk, validate=True)[:EMBEDDED_HEADER_BYTES]
    except (binascii.Error, ValueError):
        return "the embedded image's base64 data is malformed"
    dims = _header_dimensions(head)
    if dims is None:
        return "the embedded image is not a PNG or JPEG with a readable header"
    return dims


def _check_stored_size(img: ET.Element, bw: float, bh: float, warnings: list[str]) -> dict:
    """Compare the embedded image's stored pixel size with the bitmap's (header only)."""
    info = _embedded_image_size(img)
    if isinstance(info, str):
        warnings.append(f"Could not check the embedded image against the bitmap: {info}.")
        return {"stored_format": None, "stored_size": None}
    fmt, sw, sh = info
    if sw <= 0 or sh <= 0 or sw > MAX_EMBEDDED_IMAGE_DIM or sh > MAX_EMBEDDED_IMAGE_DIM:
        raise SvgReferenceError(
            f"The embedded {fmt.upper()} declares {sw}x{sh} px; each side must be 1..{MAX_EMBEDDED_IMAGE_DIM}.",
            "svg_image_dimensions")
    off = abs((sw / sh) / (bw / bh) - 1.0)
    if off > FRAME_ASPECT_TOLERANCE:
        raise SvgReferenceError(
            f"The embedded image is {sw}x{sh} px but the bitmap is {bw:g}x{bh:g} px (aspect ratio differs "
            f"by {off:.1%}); the SVG was traced over a different image.", "svg_bitmap_mismatch")
    if (sw, sh) != (int(bw), int(bh)):
        warnings.append(f"The embedded image is {sw}x{sh} px but the bitmap is {bw:g}x{bh:g} px (same aspect "
                        "ratio); check that the SVG was traced over this bitmap.")
    return {"stored_format": fmt, "stored_size": [sw, sh]}


def _size_match_mapping(root: ET.Element, bw: float, bh: float) -> tuple[_Affine, tuple[float, float]]:
    vb = _numbers(root.get("viewBox"))
    if vb and (len(vb) != 4 or vb[2] <= 0 or vb[3] <= 0):
        raise SvgReferenceError(f"Invalid viewBox {root.get('viewBox')!r}.", "svg_parse")
    no_reg = ("Cannot register the SVG onto the bitmap: there is no <image> placement frame, no "
              "<rect id=\"border\">, and neither width/height nor the viewBox equals the bitmap size "
              f"{bw:g}x{bh:g} px. Keep the bitmap in the export (it is used only for alignment) or "
              "start from the trace template.")
    try:
        w_parts = _split_length(root.get("width"), "width")
        h_parts = _split_length(root.get("height"), "height")
        width = _parse_length(root.get("width"), "width")
        height = _parse_length(root.get("height"), "height")
    except SvgReferenceError:
        if not vb:
            raise
        w_parts = h_parts = None
        width = height = None
    if vb:
        width = vb[2] if width is None else width
        height = vb[3] if height is None else height
    if width is None or height is None:
        raise SvgReferenceError(no_reg, "svg_no_registration")
    # Physical units only equal pixels through an assumed dpi, so a mm/in size is never
    # trusted to match the bitmap on its own; only px/unitless sizes (or the viewBox) are.
    physical = any(p is not None and p[1] not in ("", "px") for p in (w_parts, h_parts))

    if not physical and abs(width - bw) <= 0.5 and abs(height - bh) <= 0.5:
        view = _Affine()
        if vb:
            sx, sy = width / vb[2], height / vb[3]
            par = (root.get("preserveAspectRatio") or "").strip().lower()
            if abs(sx - sy) > 1e-6 * max(sx, sy) and not par.startswith("none"):
                raise SvgReferenceError(
                    "viewBox aspect ratio differs from width/height; set preserveAspectRatio=\"none\" "
                    "or make the viewBox match the bitmap size.", "svg_no_registration")
            view = _Affine.scale_translate(sx, sy, -vb[0] * sx, -vb[1] * sy)
        return _Affine.scale_translate(bw / width, bh / height, 0.0, 0.0) @ view, (width, height)
    if vb and abs(vb[2] - bw) <= 0.5 and abs(vb[3] - bh) <= 0.5:
        return _Affine.scale_translate(bw / vb[2], bh / vb[3], -vb[0] * bw / vb[2], -vb[1] * bh / vb[3]), \
            (vb[2], vb[3])
    raise SvgReferenceError(
        f"SVG size {width:g}x{height:g} px does not match the bitmap size {bw:g}x{bh:g} px. " + no_reg,
        "svg_no_registration")


def read_reference_svg(
    source: str | Path | bytes,
    bitmap_size: tuple[int, int],
    *,
    max_bytes: int = MAX_SVG_BYTES,
    max_elements: int = MAX_GEOMETRY_ELEMENTS,
    max_points: int = MAX_SAMPLED_POINTS,
    sample_step_px: float = SAMPLE_STEP_PX,
) -> SvgReference:
    """Parse reference centerlines from an SVG that is registered to a bitmap.

    ``bitmap_size`` is (width, height) in pixels. Registration order: exactly one
    ``<image>`` (its rendered rectangle is the bitmap frame), a border rect, or a
    size match (see the module docstring).
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        p = Path(source)
        size = p.stat().st_size
        if size > max_bytes:
            raise SvgReferenceError(f"SVG is {size} bytes; the limit is {max_bytes} bytes "
                                    "(embedded images count).", "svg_too_large")
        raw = p.read_bytes()
    if len(raw) > max_bytes:
        raise SvgReferenceError(f"SVG is {len(raw)} bytes; the limit is {max_bytes} bytes "
                                "(embedded images count).", "svg_too_large")
    if _FORBIDDEN_MARKUP.search(raw):
        raise SvgReferenceError("SVG contains a DOCTYPE or ENTITY declaration; remove it and resubmit.",
                                "svg_doctype")

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise SvgReferenceError(f"SVG is not well-formed XML: {exc}", "svg_parse") from None
    if _local(root.tag) != "svg":
        raise SvgReferenceError("Root element must be <svg>.", "svg_parse")

    for elem in root.iter():
        name = _local(elem.tag)
        if name == "use":
            raise SvgReferenceError("<use> elements are not supported; convert clones to paths.",
                                    "svg_use_unsupported")
        if name == "svg" and elem is not root:
            raise SvgReferenceError("Nested <svg> elements are not supported.", "svg_parse")

    bw, bh = float(bitmap_size[0]), float(bitmap_size[1])
    warnings: list[str] = []
    rendered = _walk(root)
    elements = [(e, ctm) for e, ctm in rendered if _local(e.tag) in _GEOMETRY_TAGS]
    if len(elements) > max_elements:
        raise SvgReferenceError(f"SVG has {len(elements)} geometry elements; the limit is {max_elements}.",
                                "svg_too_many_elements")
    images = [(e, ctm) for e, ctm in rendered if _local(e.tag) == "image"]
    border = [(e, ctm) for e, ctm in elements if _local(e.tag) == "rect" and _is_border(e)]

    frame = None
    if len(images) > 1:
        raise SvgReferenceError(f"SVG contains {len(images)} <image> elements; keep exactly one (the bitmap "
                                "being traced) so it can be used for alignment.", "svg_multiple_images")
    if images:
        img, ctm = images[0]
        iw, ih = _parse_length(img.get("width"), "<image> width"), _parse_length(img.get("height"), "<image> height")
        if not iw or not ih or iw <= 0 or ih <= 0:
            raise SvgReferenceError("<image> needs a positive width and height.", "svg_parse")
        if not ctm.axis_aligned:
            raise SvgReferenceError("The <image> is rotated or skewed; place the bitmap upright "
                                    "(flips are fine) and re-export.", "svg_image_rotated")
        lo, hi = _rect_frame(_f(img, "x"), _f(img, "y"), iw, ih, ctm)
        off = _aspect_off(lo, hi, bw, bh)
        if off > FRAME_ASPECT_TOLERANCE:
            raise SvgReferenceError(
                f"The <image> frame aspect ratio differs from the bitmap's {bw:g}x{bh:g} by {off:.1%}; "
                "do not stretch or crop the bitmap in the editor.", "svg_image_aspect")
        mapping = _frame_mapping(lo, hi, bw, bh)
        alignment = "image-frame"
        svg_size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        frame = {"x0": float(lo[0]), "y0": float(lo[1]), "x1": float(hi[0]), "y1": float(hi[1]),
                 "flip_x": bool(ctm.m[0, 0] < 0), "flip_y": bool(ctm.m[1, 1] < 0),
                 "href": _href_kind(img)}
        frame.update(_check_stored_size(img, bw, bh, warnings))
        if border:
            warnings.append("Border rectangle ignored: the <image> placement frame is used for registration.")
    elif border:
        if len(border) > 1:
            raise SvgReferenceError("More than one border rectangle found; keep exactly one.", "svg_parse")
        r, ctm = border[0]
        rw, rh = _f(r, "width"), _f(r, "height")
        if rw <= 0 or rh <= 0:
            raise SvgReferenceError("Border rectangle needs positive width and height.", "svg_parse")
        if not ctm.axis_aligned:
            raise SvgReferenceError("Border rectangle is rotated or skewed; keep it upright.",
                                    "svg_image_rotated")
        lo, hi = _rect_frame(_f(r, "x"), _f(r, "y"), rw, rh, ctm)
        mapping = _frame_mapping(lo, hi, bw, bh)
        alignment = "border-rect"
        svg_size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        if _aspect_off(lo, hi, bw, bh) > FRAME_ASPECT_TOLERANCE:
            warnings.append("Border rectangle aspect ratio differs from the bitmap by more than 1%.")
    else:
        mapping, svg_size = _size_match_mapping(root, bw, bh)
        alignment = "size-match"

    budget = _Budget(max_points)
    paths, closed = [], []
    for e, ctm in elements:
        name = _local(e.tag)
        if name == "rect" and _is_border(e):
            continue
        total = mapping @ ctm
        step_user = sample_step_px / total.max_scale
        found: list[tuple[np.ndarray, bool]] = []
        if name == "path":
            found.extend(_parse_path_d(e.get("d") or "", step_user, budget))
        elif name == "line":
            a = np.array([_f(e, "x1"), _f(e, "y1")])
            b = np.array([_f(e, "x2"), _f(e, "y2")])
            found.append((np.vstack([a[None], _line(a, b, step_user, budget)]), False))
        elif name in ("polyline", "polygon"):
            pts = _points_attr(e.get("points"))
            if name == "polygon" and not np.allclose(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[:1]])
            found.append((_densify_polyline(pts, step_user, budget), name == "polygon"))
        elif name == "rect":
            x, y, w, h = _f(e, "x"), _f(e, "y"), _f(e, "width"), _f(e, "height")
            if w > 0 and h > 0:
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]])
                found.append((_densify_polyline(pts, step_user, budget), True))
        elif name in ("circle", "ellipse"):
            cx, cy = _f(e, "cx"), _f(e, "cy")
            if name == "circle":
                rx = ry = _f(e, "r")
            else:
                rx, ry = _f(e, "rx"), _f(e, "ry")
            if rx > 0 and ry > 0:
                n = _segments_for(2 * math.pi * max(rx, ry), step_user, budget)
                t = np.linspace(0.0, 2 * math.pi, n + 1)
                found.append((np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1), True))
        for pts_xy, is_closed in found:
            px = total.apply(pts_xy)
            paths.append(np.stack([px[:, 1] - 0.5, px[:, 0] - 0.5], axis=1))
            closed.append(bool(is_closed))
    if not paths:
        raise SvgReferenceError("SVG contains no centerline geometry (path/line/polyline/polygon/circle/ellipse).",
                                "svg_no_geometry")
    return SvgReference(paths=paths, closed=closed, svg_size=svg_size, alignment=alignment,
                        warnings=warnings, frame=frame)


def write_reference_svg(path: str | Path, paths_rc: list[np.ndarray], size_wh: tuple[int, int],
                        closed: list[bool] | None = None, decimals: int = 3) -> None:
    """Write (row, col) pixel-centre polylines as a size-matched SVG reference."""
    w, h = int(size_wh[0]), int(size_wh[1])
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
    ]
    closed = closed or [False] * len(paths_rc)
    fmt = f"{{:.{decimals}f}}"
    for idx, (pts, is_closed) in enumerate(zip(paths_rc, closed)):
        pts = np.asarray(pts, dtype=float)
        coords = [f"{fmt.format(c + 0.5)} {fmt.format(r + 0.5)}" for r, c in pts]
        d = "M " + " L ".join(coords) + (" Z" if is_closed else "")
        lines.append(f'  <path id="ref-{idx}" d="{d}" fill="none" stroke="#00a000" stroke-width="0.5"/>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
