"""Prototype: resolve skeleton junctions so crossing strokes continue straight through.

Standalone post-processor for the paths the live pipeline returns. It does not touch the
engine. Paths are float (row, col) arrays in pixel-index coordinates, as in
lessonkit.runner.PipelineResult.paths.

Algorithm, per junction cluster:
 1. Nodes are clusters of path endpoints; junctions are nodes of degree >= 3.
 2. Junctions joined by a short "bridge" path (shorter than BRIDGE_FACTOR x the sum of
    the inscribed radii at both ends) form one cluster; the bridges are dropped.
 3. Each arm entering the cluster is trimmed back to where it stops bending toward the
    junction: a line is fitted to a window of the arm beyond the junction's influence,
    and the cut is placed at the innermost point from which the arm stays within
    FIT_TOL_PX of that line.
 4. Arms are paired by good continuation (smallest turning angle plus lateral offset),
    choosing the cheapest matching. Pairs above MAX_PAIR_ANGLE_DEG are never joined.
 5. Each pair is joined by a cubic Hermite curve between the two cut points using the
    fitted tangents, so a straight crossing becomes a straight line through the crossing.
    One unpaired arm (a T-junction stem) is extended along its tangent until it meets a
    joined path. If any step fails, the cluster is left exactly as it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CLUSTER_PX = 1.5
BRIDGE_FACTOR = 4.0  # junction spacing grows like width / tan(angle); 4.0 covers crossings down to ~15 deg
FIT_TOL_PX = 0.75
FIT_WINDOW_PX = 24.0
MIN_FIT_PX = 6.0
INFLUENCE_FACTOR = 3.0  # fit window starts this many inscribed radii from the junction
MAX_PAIR_ANGLE_DEG = 35.0
OFFSET_WEIGHT = 4.0  # degrees of pairing cost per px of lateral offset
RESAMPLE_PX = 0.5
MAX_ROUNDS = 16


@dataclass
class ClusterReport:
    junctions: list[list[float]]
    bridges_removed: int
    arms: int
    pairs: list[dict] = field(default_factory=list)
    stems: int = 0
    resolved: bool = False
    reason: str = ""


def dark_mask(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=float)
    if g.max() > 1.0:
        g = g / 255.0
    lo, hi = float(g.min()), float(g.max())
    return g < (lo + hi) / 2.0 if hi > lo else np.zeros(g.shape, bool)


def _length(p: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0


def _resample(p: np.ndarray, step: float = RESAMPLE_PX) -> np.ndarray:
    if len(p) < 2:
        return p.copy()
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p[:1].copy()
    t = np.linspace(0.0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def _nodes(paths: list[np.ndarray]):
    """Return (node_points, node_members); members are (path_index, end) with end 0=start, 1=end."""
    ends, owners = [], []
    for i, p in enumerate(paths):
        if len(p) < 2 or np.hypot(*(p[0] - p[-1])) <= 1e-6:
            continue
        ends += [p[0], p[-1]]
        owners += [(i, 0), (i, 1)]
    if not ends:
        return [], []
    pts = np.array(ends)
    tree = cKDTree(pts)
    label = -np.ones(len(pts), int)
    nodes, members = [], []
    for i in range(len(pts)):
        if label[i] >= 0:
            continue
        label[i] = len(nodes)
        stack, group = [i], []
        while stack:
            j = stack.pop()
            group.append(j)
            for k in tree.query_ball_point(pts[j], CLUSTER_PX):
                if label[k] < 0:
                    label[k] = label[i]
                    stack.append(k)
        nodes.append(pts[group].mean(axis=0))
        members.append([owners[g] for g in group])
    return nodes, members


def _radius(edt: np.ndarray, pt) -> float:
    r = int(np.clip(round(pt[0]), 0, edt.shape[0] - 1))
    c = int(np.clip(round(pt[1]), 0, edt.shape[1] - 1))
    return float(edt[r, c])


def _fit_arm(arm: np.ndarray, influence_px: float):
    """arm is ordered from the junction outward. Returns (cut_index, resampled, cut_point, dir) or None."""
    q = _resample(arm)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    total = s[-1]
    if total < MIN_FIT_PX + 1.0:
        return None
    a = min(max(influence_px, 0.0), total - MIN_FIT_PX)
    for _ in range(4):
        b = min(total, a + FIT_WINDOW_PX)
        sel = (s >= a) & (s <= b)
        win = q[sel]
        if len(win) < 3:
            return None
        centre = win.mean(axis=0)
        _, _, vt = np.linalg.svd(win - centre)
        d = vt[0]
        if np.dot(q[0] - centre, d) < 0:
            d = -d  # point toward the junction
        normal = np.array([-d[1], d[0]])
        dev = np.abs((q - centre) @ normal)
        bad = np.where((dev > FIT_TOL_PX) & (s <= b))[0]
        cut = int(bad.max()) + 1 if len(bad) else 0
        if s[min(cut, len(q) - 1)] <= a + 1e-9:
            break
        # The bend reaches into the window: move the window out and refit.
        a = min(s[min(cut, len(q) - 1)] + 1.0, total - MIN_FIT_PX)
    else:
        return None
    if np.max(dev[sel]) > 2.0 * FIT_TOL_PX:
        return None  # the fitted part is not straight enough to extrapolate
    cut = min(cut, len(q) - 2)
    p = centre + d * np.dot(q[cut] - centre, d)  # start the join on the fitted centre line
    return cut, q, p, d


def _hermite(p0, t0, p1, t1, step: float = RESAMPLE_PX) -> np.ndarray:
    chord = float(np.hypot(*(p1 - p0)))
    n = max(2, int(np.ceil(chord / step)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * chord * t0 + h01 * p1 + h11 * chord * t1


def _ray_hit(p, d, poly: np.ndarray, max_dist: float):
    best = None
    for a, b in zip(poly[:-1], poly[1:]):
        e = b - a
        m = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        u, v = np.linalg.solve(m, a - p)
        if 0 <= v <= 1 and -1e-6 <= u <= max_dist and (best is None or u < best[0]):
            best = (u, p + u * d)
    return None if best is None else best[1]


def _matchings(items: list[int]):
    """All matchings that leave at most one item unpaired."""
    if len(items) <= 1:
        yield []
        return
    first, rest = items[0], items[1:]
    for k in range(len(rest)):
        for m in _matchings(rest[:k] + rest[k + 1:]):
            yield [(first, rest[k])] + m
    if len(items) % 2 == 1:
        yield from _matchings(rest)


def _resolve_one(paths, members, node_pts, cluster, edt):
    radii = [_radius(edt, node_pts[n]) for n in cluster]
    max_r = max(radii + [1.0])
    rep = ClusterReport(junctions=[[round(float(v), 2) for v in node_pts[n]] for n in cluster],
                        bridges_removed=0, arms=0)
    cluster_set = set(cluster)
    end_node = {pe: n for n, mem in enumerate(members) for pe in mem}
    bridges, arms = set(), []
    for n in cluster:
        for (pi, e) in members[n]:
            if end_node.get((pi, 1 - e)) in cluster_set:
                bridges.add(pi)
            else:
                arms.append((pi, e))
    rep.bridges_removed = len(bridges)
    rep.arms = len(arms)
    if len(arms) < 3 or len(arms) > 8:
        rep.reason = f"{len(arms)} arms"
        return None, rep
    fits = []
    for (pi, e) in arms:
        f = _fit_arm(paths[pi] if e == 0 else paths[pi][::-1], INFLUENCE_FACTOR * max_r)
        if f is None:
            rep.reason = f"arm {pi} too short or curved to extrapolate"
            return None, rep
        fits.append(f)

    def cost(i, j):
        _, _, pi_, di = fits[i]
        _, _, pj, dj = fits[j]
        ang = float(np.degrees(np.arccos(np.clip(-np.dot(di, dj), -1.0, 1.0))))
        ni, nj = np.array([-di[1], di[0]]), np.array([-dj[1], dj[0]])
        off = 0.5 * (abs(np.dot(pj - pi_, ni)) + abs(np.dot(pi_ - pj, nj)))
        return ang, float(off)

    idx = list(range(len(arms)))
    best, best_cost = None, None
    for m in _matchings(idx):
        if len(arms) - 2 * len(m) > 1:
            continue
        total, ok = 0.0, True
        for i, j in m:
            ang, off = cost(i, j)
            if arms[i][0] == arms[j][0] or ang > MAX_PAIR_ANGLE_DEG or off > max(3.0, 2.0 * max_r):
                ok = False
                break
            total += ang + OFFSET_WEIGHT * off
        if ok and (best_cost is None or total < best_cost):
            best, best_cost = m, total
    if not best:
        rep.reason = "no arms continue each other"
        return None, rep

    new_paths, used = [], set()
    for i, j in best:
        ci, qi, pi_, di = fits[i]
        cj, qj, pj, dj = fits[j]
        join = _hermite(pi_, di, pj, -dj)
        new_paths.append(np.vstack([qi[ci + 1:][::-1], join, qj[cj + 1:]]))
        used |= {i, j}
        ang, off = cost(i, j)
        rep.pairs.append({"arms": [int(arms[i][0]), int(arms[j][0])], "angle_deg": round(ang, 2),
                          "offset_px": round(off, 2)})
    for k in idx:
        if k in used:
            continue
        ck, qk, pk, dk = fits[k]
        hits = [h for h in (_ray_hit(pk, dk, np_, 4.0 * max_r + 4.0) for np_ in new_paths) if h is not None]
        if not hits:
            rep.reason = "stem does not meet a joined path"
            return None, rep
        hit = min(hits, key=lambda h: float(np.hypot(*(h - pk))))
        new_paths.append(np.vstack([qk[ck + 1:][::-1], _resample(np.vstack([pk, hit]))]))
        rep.stems += 1
    drop = bridges | {pi for pi, _ in arms}
    rep.resolved = True
    return [p for n, p in enumerate(paths) if n not in drop] + new_paths, rep


def resolve_junctions(paths: list[np.ndarray], gray: np.ndarray) -> tuple[list[np.ndarray], list[ClusterReport]]:
    """Return new paths (same frame as the input) and one report per junction cluster examined."""
    paths = [np.asarray(p, float)[:, :2].copy() for p in paths if len(p)]
    edt = ndimage.distance_transform_edt(dark_mask(gray))
    reports: list[ClusterReport] = []
    tried: set = set()
    for _ in range(MAX_ROUNDS):
        node_pts, members = _nodes(paths)
        junctions = [n for n, m in enumerate(members) if len(m) >= 3]
        if not junctions:
            break
        end_node = {pe: n for n, mem in enumerate(members) for pe in mem}
        parent = {n: n for n in junctions}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for pi, p in enumerate(paths):
            a, b = end_node.get((pi, 0)), end_node.get((pi, 1))
            if a in parent and b in parent and a != b:
                limit = BRIDGE_FACTOR * (_radius(edt, node_pts[a]) + _radius(edt, node_pts[b]))
                if _length(p) <= max(limit, 3.0):
                    parent[find(a)] = find(b)
        groups: dict[int, list[int]] = {}
        for n in junctions:
            groups.setdefault(find(n), []).append(n)
        changed = False
        candidates = []
        for cluster in groups.values():
            candidates.append(cluster)
            if len(cluster) > 1:
                candidates += [[n] for n in cluster]  # fallback when the merged cluster cannot be resolved
        for cluster in candidates:
            key = tuple(sorted((round(float(node_pts[n][0]), 1), round(float(node_pts[n][1]), 1)) for n in cluster))
            if key in tried:
                continue
            tried.add(key)
            out, rep = _resolve_one(paths, members, node_pts, cluster, edt)
            reports.append(rep)
            if out is not None:
                paths = out
                changed = True
                break  # node indices are stale; recompute
        if not changed:
            break
    return paths, reports
