"""Score engine centerlines against reference centerlines.

All geometry is (row, col) float in the pixel-centre convention. A scoring
region is given in SVG pixel units {x0, y0, x1, y1} (x = col + 0.5,
y = row + 0.5) or None for the whole image.

Per reference line (only the part inside the region, minus trim_ends_px):
  * best_single_coverage: largest fraction of reference samples within
    tolerance of ONE output path (does one path continue across the crossing?)
  * any_coverage: fraction within tolerance of any output path
  * fragments: output paths that cover >= 10% of the in-region samples
  * lateral error reference -> best path (mean/p95/max), and best path ->
    reference for the best path's own in-region samples
A line passes if coverage (single or any, per require_single_path) is at
least min_single_path_coverage and the reference -> best path p95 is within
tolerance_px.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

DENSIFY_STEP_PX = 0.25
FRAGMENT_MIN_SHARE = 0.10
ENDPOINT_CLUSTER_PX = 1.5
SHORT_PATH_PX = 12.0


def densify(path: np.ndarray, step: float = DENSIFY_STEP_PX) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path.copy()
    seg = np.diff(path, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    parts = [path[:1]]
    for a, d, length in zip(path[:-1], seg, seg_len):
        n = max(1, int(np.ceil(length / step)))
        t = np.arange(1, n + 1, dtype=float)[:, None] / n
        parts.append(a + d * t)
    return np.vstack(parts)


def polyline_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    d = np.diff(path, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def in_region(points: np.ndarray, region: dict | None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if region is None or len(points) == 0:
        return np.ones(len(points), dtype=bool)
    x = points[:, 1] + 0.5
    y = points[:, 0] + 0.5
    return (x >= region["x0"]) & (x <= region["x1"]) & (y >= region["y0"]) & (y <= region["y1"])


def _trim_mask(dense: np.ndarray, trim_px: float, closed: bool = False) -> np.ndarray:
    if trim_px <= 0 or closed or len(dense) < 2:
        return np.ones(len(dense), dtype=bool)
    d = np.diff(dense, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    return (s >= trim_px) & (s <= s[-1] - trim_px)


def _stats(values: np.ndarray) -> dict:
    if values is None or len(values) == 0:
        return {"mean": None, "p95": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


@dataclass
class LineScore:
    index: int
    samples: int
    best_path: int | None
    best_single_coverage: float
    any_coverage: float
    fragments: int
    ref_to_best: dict
    best_to_ref: dict
    ref_to_any: dict
    passed: bool
    reason: str = ""


@dataclass
class LessonScore:
    passed: bool
    lines: list[LineScore]
    graph: dict
    tolerance_px: float
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        worst_cov = min((ln.best_single_coverage for ln in self.lines), default=0.0)
        p95s = [ln.ref_to_best["p95"] for ln in self.lines if ln.ref_to_best["p95"] is not None]
        return {
            "passed": self.passed,
            "min_single_coverage": round(float(worst_cov), 3),
            "max_p95_px": round(float(max(p95s)), 3) if p95s else None,
            "fragments": [ln.fragments for ln in self.lines],
            **{k: self.graph[k] for k in ("path_count", "paths_in_region", "endpoints_in_region",
                                          "junctions_in_region", "short_paths_in_region")},
        }


def graph_metrics(paths: list[np.ndarray], region: dict | None = None,
                  image_shape: tuple[int, int] | None = None,
                  cluster_px: float = ENDPOINT_CLUSTER_PX, short_px: float = SHORT_PATH_PX) -> dict:
    """Endpoint/junction topology from path ends (paths meeting at a point share end coordinates)."""
    ends = []
    for idx, p in enumerate(paths):
        p = np.asarray(p, dtype=float)
        if len(p) == 0:
            continue
        if len(p) >= 2 and np.hypot(*(p[0] - p[-1])) <= 1e-6:
            continue  # closed loop has no free ends
        ends.append((idx, p[0]))
        ends.append((idx, p[-1]))
    nodes: list[dict] = []
    if ends:
        pts = np.array([e[1] for e in ends])
        tree = cKDTree(pts)
        label = -np.ones(len(pts), dtype=int)
        for i in range(len(pts)):
            if label[i] >= 0:
                continue
            stack, label[i] = [i], len(nodes)
            members = []
            while stack:
                j = stack.pop()
                members.append(j)
                for k in tree.query_ball_point(pts[j], cluster_px):
                    if label[k] < 0:
                        label[k] = label[i]
                        stack.append(k)
            nodes.append({"point": pts[members].mean(axis=0), "degree": len(members)})
    border_margin = 2.0

    def _on_border(pt):
        if image_shape is None:
            return False
        h, w = image_shape[:2]
        return pt[0] <= border_margin or pt[1] <= border_margin or pt[0] >= h - 1 - border_margin \
            or pt[1] >= w - 1 - border_margin

    endpoints = [n for n in nodes if n["degree"] == 1 and not _on_border(n["point"])]
    junctions = [n for n in nodes if n["degree"] >= 3]
    region_endpoints = [n for n in endpoints if in_region(n["point"][None], region)[0]]
    region_junctions = [n for n in junctions if in_region(n["point"][None], region)[0]]
    lengths = [polyline_length(p) for p in paths]
    paths_in_region = [i for i, p in enumerate(paths) if np.any(in_region(np.asarray(p, float), region))]
    short_in_region = [i for i in paths_in_region if lengths[i] < short_px
                       and np.all(in_region(np.asarray(paths[i], float)[[0, -1]], region))]
    return {
        "path_count": len(paths),
        "paths_in_region": len(paths_in_region),
        "endpoints": [n["point"].tolist() for n in endpoints],
        "junctions": [{"point": n["point"].tolist(), "degree": n["degree"]} for n in junctions],
        "endpoints_in_region": len(region_endpoints),
        "junctions_in_region": len(region_junctions),
        "short_paths_in_region": len(short_in_region),
        "short_path_lengths": [round(lengths[i], 2) for i in short_in_region],
        "path_lengths": [round(v, 2) for v in lengths],
    }


def score_paths(output_paths: list[np.ndarray], reference_paths: list[np.ndarray], scoring: dict,
                image_shape: tuple[int, int] | None = None,
                reference_closed: list[bool] | None = None) -> LessonScore:
    region = scoring.get("region")
    tol = float(scoring.get("tolerance_px", 1.0))
    min_cov = float(scoring.get("min_single_path_coverage", 0.95))
    single = bool(scoring.get("require_single_path", True))
    trim = float(scoring.get("trim_ends_px", 0.0))
    reference_closed = reference_closed or [False] * len(reference_paths)

    dense_out = [densify(p) for p in output_paths if len(p)]
    trees = [cKDTree(d) for d in dense_out]
    notes: list[str] = []
    lines: list[LineScore] = []
    for li, (ref, closed) in enumerate(zip(reference_paths, reference_closed)):
        dref = densify(ref)
        mask = in_region(dref, region) & _trim_mask(dref, trim, closed)
        R = dref[mask]
        if len(R) == 0:
            notes.append(f"reference line {li} has no samples inside the scoring region")
            continue
        if not trees:
            lines.append(LineScore(li, len(R), None, 0.0, 0.0, 0, _stats(None), _stats(None), _stats(None),
                                   False, "no output paths"))
            continue
        dist = np.stack([t.query(R)[0] for t in trees])  # (paths, samples)
        cover = dist <= tol
        shares = cover.mean(axis=1)
        best = int(np.argmax(shares))
        any_cov = float(cover.any(axis=0).mean())
        fragments = int(np.sum(shares >= FRAGMENT_MIN_SHARE))
        ref_to_best = _stats(dist[best])
        ref_to_any = _stats(dist.min(axis=0))
        bp = dense_out[best]
        bp_in = bp[in_region(bp, region)]
        if len(bp_in):
            ref_tree = cKDTree(dref)
            d_b2r, _ = ref_tree.query(bp_in)
            # Only count output samples that are alongside this reference line.
            d_b2r = d_b2r[d_b2r <= max(4.0 * tol, 4.0)]
        else:
            d_b2r = np.array([])
        cov = float(shares[best]) if single else any_cov
        ok_cov = cov >= min_cov
        ok_err = ref_to_best["p95"] is not None and ref_to_best["p95"] <= tol
        reason = []
        if not ok_cov:
            reason.append(f"{'single-path' if single else 'any-path'} coverage {cov:.2f} < {min_cov:.2f}")
        if not ok_err:
            reason.append(f"p95 lateral {ref_to_best['p95']:.2f}px > {tol:.2f}px")
        lines.append(LineScore(li, len(R), best, float(shares[best]), any_cov, fragments,
                               ref_to_best, _stats(d_b2r), ref_to_any, ok_cov and ok_err, "; ".join(reason)))

    graph = graph_metrics(output_paths, region, image_shape)
    passed = bool(lines) and all(ln.passed for ln in lines)
    return LessonScore(passed=passed, lines=lines, graph=graph, tolerance_px=tol, notes=notes)
