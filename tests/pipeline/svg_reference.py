"""Read-only, defensive SVG reader for reference centerlines (stdlib + numpy only).

Output paths use the engine convention: (row, col) floats where integer
coordinates are pixel centres. An SVG point (x, y) in bitmap pixel units
therefore maps to (row, col) = (y - 0.5, x - 0.5), matching the +0.5 shift
that ``create_svg_output`` applies on export.

Deliberately unsupported for now (clear errors are raised):
  * ``transform`` attributes, ``<use>`` elements
  * physical units (mm, cm, in, pt, pc) and percentages on width/height
  * non-uniform viewBox scaling unless preserveAspectRatio="none"
Affine transforms and physical units must be supported before the public
Teach feature, since CAD/Inkscape exports commonly include them.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_SVG_BYTES = 2 * 1024 * 1024
MAX_GEOMETRY_ELEMENTS = 200
MAX_SAMPLED_POINTS = 200_000
SAMPLE_STEP_PX = 0.5

_FORBIDDEN_MARKUP = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_LENGTH = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z%]*)\s*$", re.IGNORECASE)
_BORDER_MARKERS = {"border", "alignment-border", "bitmap-border"}


class SvgReferenceError(ValueError):
    """Raised with a precise, user-fixable message when an SVG is rejected."""


@dataclass
class SvgReference:
    paths: list[np.ndarray]           # each (N, 2) [[row, col]] in bitmap pixel-centre coords
    closed: list[bool]
    svg_size: tuple[float, float]     # (width, height) in SVG user units after viewBox
    alignment: str                    # "size-match" | "border-rect"
    warnings: list[str] = field(default_factory=list)


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _parse_length(value: str | None, what: str) -> float | None:
    if value is None:
        return None
    m = _LENGTH.match(value)
    if not m:
        raise SvgReferenceError(f"Cannot parse {what}={value!r}; use a plain pixel number.")
    unit = m.group(2).lower()
    if unit not in ("", "px"):
        raise SvgReferenceError(
            f"{what}={value!r} uses unit {unit!r}. Only pixel units are supported for now; "
            "export the SVG in px with its size equal to the bitmap's pixel size."
        )
    return float(m.group(1))


def _numbers(text: str | None) -> list[float]:
    return [float(n) for n in _NUMBER.findall(text or "")]


class _Affine:
    """Scale + translate only (what viewBox and border alignment need)."""

    def __init__(self, sx=1.0, sy=1.0, tx=0.0, ty=0.0):
        self.sx, self.sy, self.tx, self.ty = sx, sy, tx, ty

    def then(self, other: "_Affine") -> "_Affine":
        return _Affine(
            self.sx * other.sx,
            self.sy * other.sy,
            self.tx * other.sx + other.tx,
            self.ty * other.sy + other.ty,
        )

    def apply(self, xy: np.ndarray) -> np.ndarray:
        out = np.empty_like(xy, dtype=float)
        out[:, 0] = xy[:, 0] * self.sx + self.tx
        out[:, 1] = xy[:, 1] * self.sy + self.ty
        return out

    @property
    def max_scale(self) -> float:
        return max(abs(self.sx), abs(self.sy), 1e-9)


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


def read_reference_svg(
    source: str | Path | bytes,
    bitmap_size: tuple[int, int],
    *,
    max_bytes: int = MAX_SVG_BYTES,
    max_elements: int = MAX_GEOMETRY_ELEMENTS,
    max_points: int = MAX_SAMPLED_POINTS,
    sample_step_px: float = SAMPLE_STEP_PX,
) -> SvgReference:
    """Parse reference centerlines from an SVG that is aligned to a bitmap.

    ``bitmap_size`` is (width, height) in pixels. Alignment rules:
      * the SVG's effective size (width/height or viewBox) equals the bitmap size, or
      * the SVG contains a <rect> marked id/class/data-role="border" whose
        corners are mapped onto the bitmap's outer edges.
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        p = Path(source)
        size = p.stat().st_size
        if size > max_bytes:
            raise SvgReferenceError(f"SVG is {size} bytes; the limit is {max_bytes} bytes.")
        raw = p.read_bytes()
    if len(raw) > max_bytes:
        raise SvgReferenceError(f"SVG is {len(raw)} bytes; the limit is {max_bytes} bytes.")
    if _FORBIDDEN_MARKUP.search(raw):
        raise SvgReferenceError("SVG contains a DOCTYPE or ENTITY declaration; remove it and resubmit.")

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise SvgReferenceError(f"SVG is not well-formed XML: {exc}") from None
    if _local(root.tag) != "svg":
        raise SvgReferenceError("Root element must be <svg>.")

    bw, bh = float(bitmap_size[0]), float(bitmap_size[1])
    warnings: list[str] = []

    for elem in root.iter():
        name = _local(elem.tag)
        if elem.get("transform") is not None:
            raise SvgReferenceError(
                f"<{name}> has a transform attribute; transforms are not supported yet. "
                "Apply/flatten transforms in your editor (e.g. Inkscape: Path > Object to Path, "
                "then save as plain SVG) and resubmit."
            )
        if name == "use":
            raise SvgReferenceError("<use> elements are not supported; convert clones to paths.")
        if name == "svg" and elem is not root:
            raise SvgReferenceError("Nested <svg> elements are not supported.")

    width = _parse_length(root.get("width"), "width")
    height = _parse_length(root.get("height"), "height")
    vb = _numbers(root.get("viewBox"))
    view = _Affine()
    if vb:
        if len(vb) != 4 or vb[2] <= 0 or vb[3] <= 0:
            raise SvgReferenceError(f"Invalid viewBox {root.get('viewBox')!r}.")
        if width is None:
            width = vb[2]
        if height is None:
            height = vb[3]
        sx, sy = width / vb[2], height / vb[3]
        par = (root.get("preserveAspectRatio") or "").strip().lower()
        if abs(sx - sy) > 1e-6 * max(sx, sy) and not par.startswith("none"):
            raise SvgReferenceError(
                "viewBox aspect ratio differs from width/height; set preserveAspectRatio=\"none\" "
                "or make the viewBox match the bitmap size."
            )
        view = _Affine(sx, sy, -vb[0] * sx, -vb[1] * sy)
    if width is None or height is None:
        raise SvgReferenceError("SVG needs width and height (or a viewBox) equal to the bitmap's pixel size.")

    geometry_tags = ("path", "line", "polyline", "polygon", "rect", "circle", "ellipse")
    elements = [e for e in root.iter() if _local(e.tag) in geometry_tags]
    if len(elements) > max_elements:
        raise SvgReferenceError(f"SVG has {len(elements)} geometry elements; the limit is {max_elements}.")

    border = [e for e in elements if _local(e.tag) == "rect" and _is_border(e)]
    if len(border) > 1:
        raise SvgReferenceError("More than one border rectangle found; keep exactly one.")
    if border:
        r = border[0]
        rx0, ry0, rw, rh = _f(r, "x"), _f(r, "y"), _f(r, "width"), _f(r, "height")
        if rw <= 0 or rh <= 0:
            raise SvgReferenceError("Border rectangle needs positive width and height.")
        corner0 = view.apply(np.array([[rx0, ry0]]))[0]
        corner1 = view.apply(np.array([[rx0 + rw, ry0 + rh]]))[0]
        span = corner1 - corner0
        ax, ay = bw / span[0], bh / span[1]
        mapping = view.then(_Affine(ax, ay, -corner0[0] * ax, -corner0[1] * ay))
        alignment = "border-rect"
        if abs(ax - ay) > 0.01 * max(ax, ay):
            warnings.append("Border rectangle aspect ratio differs from the bitmap by more than 1%.")
    else:
        if abs(width - bw) > 0.5 or abs(height - bh) > 0.5:
            raise SvgReferenceError(
                f"SVG size {width:g}x{height:g} does not match the bitmap size {bw:g}x{bh:g}. "
                "Set the SVG width/height (and viewBox) to the bitmap's pixel size, or add a "
                "<rect id=\"border\"> drawn exactly around the bitmap."
            )
        mapping = view.then(_Affine(bw / width, bh / height))
        alignment = "size-match"

    step_user = sample_step_px / mapping.max_scale
    budget = _Budget(max_points)
    raw_paths: list[tuple[np.ndarray, bool]] = []
    for e in elements:
        name = _local(e.tag)
        if name == "rect" and _is_border(e):
            continue
        if name == "path":
            raw_paths.extend(_parse_path_d(e.get("d") or "", step_user, budget))
        elif name == "line":
            a = np.array([_f(e, "x1"), _f(e, "y1")])
            b = np.array([_f(e, "x2"), _f(e, "y2")])
            raw_paths.append((np.vstack([a[None], _line(a, b, step_user, budget)]), False))
        elif name in ("polyline", "polygon"):
            pts = _points_attr(e.get("points"))
            if name == "polygon" and not np.allclose(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[:1]])
            raw_paths.append((_densify_polyline(pts, step_user, budget), name == "polygon"))
        elif name == "rect":
            x, y, w, h = _f(e, "x"), _f(e, "y"), _f(e, "width"), _f(e, "height")
            if w > 0 and h > 0:
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]])
                raw_paths.append((_densify_polyline(pts, step_user, budget), True))
        elif name in ("circle", "ellipse"):
            cx, cy = _f(e, "cx"), _f(e, "cy")
            if name == "circle":
                rx = ry = _f(e, "r")
            else:
                rx, ry = _f(e, "rx"), _f(e, "ry")
            if rx > 0 and ry > 0:
                n = _segments_for(2 * math.pi * max(rx, ry), step_user, budget)
                t = np.linspace(0.0, 2 * math.pi, n + 1)
                raw_paths.append((np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1), True))

    paths, closed = [], []
    for pts_xy, is_closed in raw_paths:
        px = mapping.apply(pts_xy)
        paths.append(np.stack([px[:, 1] - 0.5, px[:, 0] - 0.5], axis=1))
        closed.append(bool(is_closed))
    if not paths:
        raise SvgReferenceError("SVG contains no centerline geometry (path/line/polyline/polygon/circle/ellipse).")
    return SvgReference(paths=paths, closed=closed, svg_size=(float(width), float(height)),
                        alignment=alignment, warnings=warnings)


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
