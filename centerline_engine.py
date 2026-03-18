#!/usr/bin/env python3
"""
Optimized Circle-Based Centerline Extraction
============================================

This script uses only the circle-based evaluation method for centerline extraction.
All legacy methods (original centerline, KD-tree, red circles) have been removed.
"""

import os
import time
import bisect
import numpy as np
from skimage import io, color, filters, morphology
from skimage.morphology import binary_erosion, footprint_rectangle
from scipy import interpolate
from functools import lru_cache
import svgwrite
import base64
from io import BytesIO
from PIL import Image
import warnings
import json
import pickle
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = False  # Force disable ML functionality
except ImportError:
    print("Warning: scikit-learn not available. ML training features disabled.")
    SKLEARN_AVAILABLE = False
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
#INPUT_PATH = "/Users/nicholas/Downloads/Stage1.JPG"
INPUT_PATH = "/Users/nicholas/Downloads/tractor.png"
OUTPUT_PATH = "/Users/nicholas/Downloads/output_optimized.svg"

# Circle-based evaluation parameters
DARK_THRESHOLD = 0.20  # Increased from 0.15 to capture more of the image
MAX_CIRCLE_RADIUS = 3   # Increased from 1 to provide better coverage
MIN_CIRCLE_RADIUS = 0.2 # Increased from 0.2 for better visibility
CIRCLE_INTERSECTION_BONUS = 2.0

# Path optimization parameters  
RDP_TOLERANCE = 2.0     # Increased from 1.5 to allow longer simplified segments
SMOOTHING_FACTOR = 0.005  # Reduced from 0.01 for less aggressive smoothing
MIN_PATH_LENGTH = 3     # Reduced from 4 to capture even short important segments

# Visualization
SHOW_BITMAP = True
SHOW_CIRCLES = True
SHOW_PRE_OPTIMIZATION_PATHS = True  # Show paths before optimization in magenta
SHOW_ML_PATHS = False  # ML features disabled
SHOW_TRAINING_PATHS = False  # Training paths disabled
CIRCLE_OPACITY = 0.2
BEST_PATH_COLOR = "lime"
BEST_PATH_WIDTH = 3.0
PRE_OPTIMIZATION_PATH_COLOR = "magenta"
PRE_OPTIMIZATION_PATH_WIDTH = 1.5
PRE_OPTIMIZATION_PATH_OPACITY = 0.4

# Multi-path support
OPTIMIZE_TOP_N_PATHS = 10  # Increased from 3 to show more paths

# Machine Learning Training Configuration (DISABLED)
ENABLE_ML_TRAINING = False  # ML training disabled
MANUAL_CENTERLINE_PATH = "/Users/nicholas/Downloads/manual_centerline.json"  # Path to save/load manual annotations
ML_MODEL_PATH = "/Users/nicholas/Downloads/centerline_model.pkl"  # Path to save/load trained model

print("=== OPTIMIZED CIRCLE-BASED CENTERLINE EXTRACTION ===")

class CircleEvaluationSystem:
    """Intensity-field based centerline evaluation.

    Scores are derived directly from the grayscale image so they can be reproduced
    from the raster that is embedded in the generated SVG."""

    def __init__(
        self,
        image,
        dark_threshold,
        max_radius=MAX_CIRCLE_RADIUS,
        min_radius=MIN_CIRCLE_RADIUS,
        intersection_bonus=CIRCLE_INTERSECTION_BONUS,
    ):
        self.image = image.astype(float)
        self.dark_threshold = dark_threshold
        self.height, self.width = image.shape

        # Darkness emphasises pixels below the threshold.
        darkness = np.clip(dark_threshold - self.image, 0.0, dark_threshold)
        if dark_threshold > 0:
            darkness = darkness / dark_threshold

        # Inverted intensity rewards very dark pixels even when below the threshold.
        inverted_intensity = 1.0 - self.image

        # Local gradient magnitude highlights stroke edges and ridges.
        gradient_mag = self._compute_gradient_magnitude(self.image)
        gradient_mag /= (gradient_mag.max() + 1e-6)

        # Blend the components into a single intensity field.
        self.intensity_field = np.clip(
            0.6 * darkness + 0.4 * inverted_intensity,
            0.0,
            1.0,
        )
        self.score_field = np.clip(
            self.intensity_field * (1.0 + 0.5 * gradient_mag),
            0.0,
            None,
        )

        # Legacy attributes retained for compatibility with existing code paths.
        self.circle_positions = []
        self.circle_radii = []
        self.intersection_weights = []

        print(
            "Intensity field prepared: "
            f"mean={self.score_field.mean():.3f}, max={self.score_field.max():.3f}"
        )

    def _compute_gradient_magnitude(self, image):
        """Return gradient magnitude of the grayscale image."""
        gy, gx = np.gradient(image)
        return np.sqrt(gx**2 + gy**2)

    def _bilinear_sample(self, field, y, x):
        """Sample a field at floating-point coordinates with bilinear interpolation."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.clip(y, 0, self.height - 1)
        x = np.clip(x, 0, self.width - 1)

        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = np.clip(y0 + 1, 0, self.height - 1)
        x1 = np.clip(x0 + 1, 0, self.width - 1)

        dy = y - y0
        dx = x - x0

        top = (1.0 - dx) * field[y0, x0] + dx * field[y0, x1]
        bottom = (1.0 - dx) * field[y1, x0] + dx * field[y1, x1]
        return (1.0 - dy) * top + dy * bottom

    def _sample_path(self, path, density=0.4):
        """Return evenly spaced samples along a polyline."""
        pts = np.asarray(path, dtype=float)
        if len(pts) == 0:
            return np.zeros((0, 2))
        if len(pts) == 1:
            return pts.copy()

        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        if total_length <= 0:
            return pts.copy()

        num_samples = max(2, int(total_length * density))
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        normalized = cumulative / total_length

        sample_params = np.linspace(0.0, 1.0, num_samples)
        samples = []
        for param in sample_params:
            idx = np.searchsorted(normalized, param, side="right")
            idx = min(max(1, idx), len(pts) - 1)
            span = normalized[idx] - normalized[idx - 1]
            if span <= 0:
                interpolated = pts[idx].copy()
            else:
                local_t = (param - normalized[idx - 1]) / span
                interpolated = (1.0 - local_t) * pts[idx - 1] + local_t * pts[idx]
            samples.append(interpolated)

        return np.array(samples)

    def evaluate_path(self, path):
        """Evaluate a path using the intensity field."""
        if len(path) < 2:
            return 0.0

        samples = self._sample_path(path, density=0.5)
        if len(samples) == 0:
            return 0.0

        values = self._bilinear_sample(self.score_field, samples[:, 0], samples[:, 1])
        path_length = np.linalg.norm(
            np.diff(np.asarray(path, dtype=float), axis=0), axis=1
        ).sum()

        # Mean field strength scaled by geometric length rewards long, dark paths.
        score = float(values.mean() * (path_length + 1.0))
        return score

    def evaluate_all_paths(self, paths):
        """Evaluate all paths and return scores and statistics."""
        scores = [self.evaluate_path(path) for path in paths]
        if not scores:
            return [], {}

        scores_array = np.array(scores, dtype=float)
        stats = {
            "mean": float(np.mean(scores_array)),
            "max": float(np.max(scores_array)),
            "min": float(np.min(scores_array)),
            "std": float(np.std(scores_array)),
        }
        return scores_array.tolist(), stats

    def point_importance(self, point):
        """Return the field value at a specific (y, x) point."""
        y, x = point
        return float(
            self._bilinear_sample(self.score_field, np.array([y]), np.array([x]))[0]
        )

    def path_intensity_profile(self, path, density=0.4):
        """Return intensity samples along the given path."""
        samples = self._sample_path(path, density=density)
        if len(samples) == 0:
            return np.array([])
        return self._bilinear_sample(
            self.score_field, samples[:, 0], samples[:, 1]
        )

    def get_visualization_data(self):
        """Expose minimal metadata for downstream consumers."""
        return {"mode": "intensity_field"}

def extract_skeleton_paths(image, dark_threshold=0.5, min_object_size=10):
    """Fast skeleton extraction optimized for speed."""
    print("Extracting skeleton paths...")
    
    # Single threshold for speed
    binary = image < dark_threshold
    
    # Remove small objects more efficiently
    if min_object_size > 0:
        binary = morphology.remove_small_objects(binary, min_object_size)
    
    # Skeletonize
    skeleton = morphology.skeletonize(binary)
    
    # Use fast path extraction
    paths = create_fast_paths(skeleton)
    
    print(f"Extracted {len(paths)} skeleton paths")
    return paths

def skeleton_to_paths_improved(skeleton):
    """Convert skeleton to list of paths with better coverage through branch points."""
    
    # Find skeleton pixels
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return []
    
    skeleton_points = set(zip(ys, xs))
    visited = set()
    paths = []
    
    def get_neighbors(y, x):
        """Get 8-connected neighbors that are skeleton pixels."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in skeleton_points:
                    neighbors.append((ny, nx))
        return neighbors
    
    def trace_path_through_branches(start_point, visited_local):
        """Trace a path that can go through branch points for better connectivity."""
        path = [start_point]
        visited_local.add(start_point)
        current = start_point
        
        while True:
            neighbors = get_neighbors(current[0], current[1])
            unvisited_neighbors = [n for n in neighbors if n not in visited_local]
            
            if len(unvisited_neighbors) == 0:
                break
            elif len(unvisited_neighbors) == 1:
                # Continue along the path
                current = unvisited_neighbors[0]
                path.append(current)
                visited_local.add(current)
            else:
                # Multiple unvisited neighbors - choose the one that continues the general direction
                if len(path) >= 2:
                    # Calculate current direction
                    direction = np.array(current) - np.array(path[-2])
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0:
                        direction = direction / direction_norm
                        
                        # Find neighbor that best continues this direction
                        best_neighbor = None
                        best_alignment = -2  # Worst possible dot product
                        
                        for neighbor in unvisited_neighbors:
                            neighbor_direction = np.array(neighbor) - np.array(current)
                            neighbor_norm = np.linalg.norm(neighbor_direction)
                            
                            if neighbor_norm > 0:
                                neighbor_direction = neighbor_direction / neighbor_norm
                                alignment = np.dot(direction, neighbor_direction)
                                
                                if alignment > best_alignment:
                                    best_alignment = alignment
                                    best_neighbor = neighbor
                        
                        if best_neighbor is not None:
                            current = best_neighbor
                            path.append(current)
                            visited_local.add(current)
                            continue
                
                # Fallback: just pick the first unvisited neighbor
                current = unvisited_neighbors[0]
                path.append(current)
                visited_local.add(current)
        
        return path
    
    # First pass: trace main paths starting from endpoints
    for point in skeleton_points:
        if point not in visited:
            neighbors = get_neighbors(point[0], point[1])
            
            # Prioritize endpoints (1 neighbor) and low-degree nodes (2 neighbors)
            if len(neighbors) <= 2:
                local_visited = set()
                path = trace_path_through_branches(point, local_visited)
                
                if len(path) >= MIN_PATH_LENGTH:
                    paths.append(path)
                    visited.update(local_visited)
    
    # Second pass: handle remaining unvisited points (complex branch areas)
    remaining_points = skeleton_points - visited
    
    while remaining_points:
        # Start from any remaining point
        start_point = next(iter(remaining_points))
        local_visited = set()
        path = trace_path_through_branches(start_point, local_visited)
        
        if len(path) >= MIN_PATH_LENGTH:
            paths.append(path)
        
        visited.update(local_visited)
        remaining_points -= local_visited
    
    return paths

def skeleton_to_paths(skeleton):
    """Convert skeleton to list of paths using simple traversal."""
    
    # Find skeleton pixels
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return []
    
    skeleton_points = set(zip(ys, xs))
    visited = set()
    paths = []
    
    def get_neighbors(y, x):
        """Get 8-connected neighbors that are skeleton pixels."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in skeleton_points and (ny, nx) not in visited:
                    neighbors.append((ny, nx))
        return neighbors
    
    def trace_path(start_point):
        """Trace a path from a starting point."""
        path = [start_point]
        visited.add(start_point)
        current = start_point
        
        while True:
            neighbors = get_neighbors(current[0], current[1])
            if len(neighbors) == 0:
                break
            elif len(neighbors) == 1:
                # Continue along the path
                current = neighbors[0]
                path.append(current)
                visited.add(current)
            else:
                # Multiple neighbors - branch point, stop here
                break
        
        return path
    
    # Find all paths
    for point in skeleton_points:
        if point not in visited:
            # Find neighbors to determine if this is a good starting point
            y, x = point
            unvisited_neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in skeleton_points and (ny, nx) not in visited:
                        unvisited_neighbors.append((ny, nx))
            
            # Start tracing from endpoints (1 neighbor) or isolated points
            if len(unvisited_neighbors) <= 1:
                path = trace_path(point)
                if len(path) >= MIN_PATH_LENGTH:
                    paths.append(path)
    
    # Handle any remaining unvisited points (complex structures)
    for point in skeleton_points:
        if point not in visited:
            path = trace_path(point)
            if len(path) >= MIN_PATH_LENGTH:
                paths.append(path)
    
    return paths

def rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker path simplification."""
    if len(points) < 3:
        return points
    
    points = np.array(points)
    
    def rdp_recursive(start, end):
        if end <= start + 1:
            return []
        
        # Find point with maximum distance from line
        line_vec = points[end] - points[start]
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return []
        
        max_dist = 0
        max_idx = start
        
        for i in range(start + 1, end):
            # Distance from point to line
            point_vec = points[i] - points[start]
            cross = np.cross(line_vec, point_vec)
            dist = abs(cross) / line_len
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        if max_dist > epsilon:
            # Recursively simplify
            left = rdp_recursive(start, max_idx)
            right = rdp_recursive(max_idx, end)
            return left + [max_idx] + right
        else:
            return []
    
    indices = [0] + rdp_recursive(0, len(points) - 1) + [len(points) - 1]
    indices = sorted(set(indices))  # Remove duplicates and sort
    
    return [tuple(points[i]) for i in indices]


def _remove_duplicate_points(points):
    """Drop consecutive duplicates so downstream geometry stays stable."""
    if not points:
        return []

    cleaned = [tuple(points[0])]
    for pt in points[1:]:
        current = tuple(pt)
        if current != cleaned[-1]:
            cleaned.append(current)
    return cleaned


def _path_length(points):
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _point_to_segment_distance(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = max(0.0, min(1.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _line_distance_stats(points):
    if len(points) <= 2:
        return 0.0, 0.0

    pts = np.asarray(points, dtype=float)
    start = pts[0]
    end = pts[-1]
    seg = end - start
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= 1e-9:
        distances = [float(np.linalg.norm(p - start)) for p in pts[1:-1]]
        return (float(max(distances)) if distances else 0.0, float(np.mean(distances)) if distances else 0.0)

    distances = [_point_to_segment_distance(p, start, end) for p in pts[1:-1]]
    return (float(max(distances)) if distances else 0.0, float(np.mean(distances)) if distances else 0.0)


def _path_length(points):
    if len(points) < 2:
        return 0.0

    pts = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _max_turn_angle(points):
    if len(points) < 3:
        return 0.0

    pts = np.asarray(points, dtype=float)
    max_angle = 0.0
    for idx in range(1, len(pts) - 1):
        v1 = pts[idx] - pts[idx - 1]
        v2 = pts[idx + 1] - pts[idx]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosang)))
        if angle > max_angle:
            max_angle = angle
    return max_angle


def _project_points_to_line(points, start, end, clamp=True):
    pts = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-9:
        return pts.copy()

    projected = []
    for pt in pts:
        t = float(np.dot(pt - start, seg) / denom)
        if clamp:
            t = max(0.0, min(1.0, t))
        projected.append(start + t * seg)

    projected = np.asarray(projected, dtype=float)
    projected[0] = start
    projected[-1] = end
    return projected


def _is_path_closed(points, close_threshold=1.5):
    if len(points) < 3:
        return False
    pts = np.asarray(points, dtype=float)
    return float(np.linalg.norm(pts[0] - pts[-1])) <= float(close_threshold)


def _find_straight_terminal_run(points, at_start=True, min_points=5, max_points=12, deviation_tol=0.45, turn_tol=18.0):
    if len(points) < min_points + 1:
        return None

    scan = list(points if at_start else reversed(points))
    upper = min(len(scan), max_points)
    best_end = None
    for end_idx in range(min_points - 1, upper):
        subset = scan[: end_idx + 1]
        max_dev, mean_dev = _line_distance_stats(subset)
        if max_dev <= deviation_tol and mean_dev <= deviation_tol * 0.5 and _max_turn_angle(subset) <= turn_tol:
            best_end = end_idx
        elif best_end is not None:
            break

    if best_end is None:
        return None
    return best_end if at_start else (len(points) - 1 - best_end)


def _terminal_turn_angle(points, at_start=True):
    if len(points) < 3:
        return 0.0

    pts = np.asarray(points, dtype=float)
    if at_start:
        first = pts[1] - pts[0]
        second = pts[2] - pts[1]
    else:
        first = pts[-2] - pts[-1]
        second = pts[-3] - pts[-2]

    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return 0.0

    cosang = float(np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _is_terminal_spur(points, at_start=True, deviation_tol=0.45):
    if len(points) < 3:
        return False

    pts = np.asarray(points, dtype=float)
    total_length = _path_length(points)
    if total_length <= 1e-9:
        return False

    segment = pts[1] - pts[0] if at_start else pts[-1] - pts[-2]
    segment_length = float(np.linalg.norm(segment))
    if segment_length > min(8.0, total_length * 0.08):
        return False

    remainder = pts[1:] if at_start else pts[:-1]
    if _path_length(remainder) < max(segment_length * 4.0, 20.0):
        return False

    endpoint = pts[0] if at_start else pts[-1]
    if _point_to_segment_distance(endpoint, remainder[0], remainder[-1]) > max(4.5, deviation_tol * 8.0):
        return False

    return _terminal_turn_angle(points, at_start=at_start) >= 18.0


def _project_via_straight_core(points, deviation_tol=0.45):
    if len(points) < 5 or _is_path_closed(points):
        return None

    trim_start = 1 if _is_terminal_spur(points, at_start=True, deviation_tol=deviation_tol) else 0
    trim_end = 1 if _is_terminal_spur(points, at_start=False, deviation_tol=deviation_tol) else 0
    if trim_start == 0 and trim_end == 0:
        return None

    core_end = len(points) - trim_end
    core = points[trim_start:core_end]
    if len(core) < 2:
        return None

    core_max_dev, core_mean_dev = _line_distance_stats(core)
    if core_max_dev > max(1.6, deviation_tol * 3.6) or core_mean_dev > max(1.0, deviation_tol * 2.2):
        return None

    projected = _project_points_to_line(points, core[0], core[-1], clamp=False)
    return _remove_duplicate_points([tuple(p) for p in projected])


def _is_monotonic_step_line(points, detour_ratio):
    if len(points) < 5:
        return False

    pts = np.asarray(points, dtype=float)
    diffs = np.diff(pts, axis=0)
    if len(diffs) == 0:
        return False

    def _axis_monotonic(values):
        return bool(np.all(values >= -0.1) or np.all(values <= 0.1))

    if not _axis_monotonic(diffs[:, 0]) or not _axis_monotonic(diffs[:, 1]):
        return False

    x_span = float(np.ptp(pts[:, 0]))
    y_span = float(np.ptp(pts[:, 1]))
    dominant_span = max(x_span, y_span)
    minor_span = min(x_span, y_span)
    if dominant_span < 24.0:
        return False
    if minor_span > max(6.0, dominant_span * 0.12):
        return False

    minor_step = float(np.max(np.abs(diffs[:, 0]))) if x_span <= y_span else float(np.max(np.abs(diffs[:, 1])))
    if minor_step > 6.0:
        return False

    return detour_ratio <= 1.12


def _straighten_path_runs(points, deviation_tol=0.45):
    cleaned = _remove_duplicate_points(points)
    if len(cleaned) < 5:
        return cleaned

    projected_core = _project_via_straight_core(cleaned, deviation_tol=deviation_tol)
    if projected_core is not None:
        return projected_core

    pts = np.asarray(cleaned, dtype=float)
    max_dev, mean_dev = _line_distance_stats(pts)
    path_length = _path_length(cleaned)
    chord_length = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
    detour_ratio = path_length / max(chord_length, 1e-6) if chord_length > 0.0 else 1.0
    if _is_monotonic_step_line(cleaned, detour_ratio):
        projected = _project_points_to_line(pts, pts[0], pts[-1])
        return _remove_duplicate_points([tuple(p) for p in projected])

    if (
        max_dev <= max(0.8, deviation_tol * 1.6)
        and mean_dev <= max(0.4, deviation_tol * 0.95)
        and detour_ratio <= 1.6
    ):
        projected = _project_points_to_line(pts, pts[0], pts[-1])
        return _remove_duplicate_points([tuple(p) for p in projected])

    if _is_path_closed(cleaned):
        return cleaned

    adjusted = pts.copy()
    prefix_end = _find_straight_terminal_run(cleaned, at_start=True, deviation_tol=deviation_tol)
    if prefix_end is not None and prefix_end >= 1:
        adjusted[: prefix_end + 1] = _project_points_to_line(
            adjusted[: prefix_end + 1], adjusted[0], adjusted[prefix_end]
        )

    suffix_start = _find_straight_terminal_run(cleaned, at_start=False, deviation_tol=deviation_tol)
    if suffix_start is not None and suffix_start < len(adjusted) - 1:
        adjusted[suffix_start:] = _project_points_to_line(
            adjusted[suffix_start:], adjusted[suffix_start], adjusted[-1]
        )

    return _remove_duplicate_points([tuple(p) for p in adjusted])


def _resample_even_spacing(points):
    if len(points) < 4:
        return list(points)

    pts = np.asarray(points, dtype=float)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 1e-9:
        return list(points)

    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    targets = np.linspace(0.0, total, len(points))
    resampled = []
    j = 1
    for dist in targets:
        while j < len(cumulative) and cumulative[j] < dist:
            j += 1
        if j == len(cumulative):
            interp = pts[-1]
        else:
            span = max(float(cumulative[j] - cumulative[j - 1]), 1e-6)
            ratio = (dist - cumulative[j - 1]) / span
            interp = pts[j - 1] + ratio * (pts[j] - pts[j - 1])
        resampled.append((float(interp[0]), float(interp[1])))
    return _remove_duplicate_points(resampled)


def _orientation(a, b, c):
    return float((b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1]))


def _point_on_segment(point, start, end, tol=1e-6):
    return (
        min(start[0], end[0]) - tol <= point[0] <= max(start[0], end[0]) + tol
        and min(start[1], end[1]) - tol <= point[1] <= max(start[1], end[1]) + tol
    )


def _segments_intersect(a1, a2, b1, b2, tol=1e-6):
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True

    if abs(o1) <= tol and _point_on_segment(b1, a1, a2, tol):
        return True
    if abs(o2) <= tol and _point_on_segment(b2, a1, a2, tol):
        return True
    if abs(o3) <= tol and _point_on_segment(a1, b1, b2, tol):
        return True
    if abs(o4) <= tol and _point_on_segment(a2, b1, b2, tol):
        return True
    return False


def _path_has_self_intersection(points, close_threshold=1.5):
    pts = _remove_duplicate_points(points)
    if len(pts) < 4:
        return False

    closed = _is_path_closed(pts, close_threshold=close_threshold)
    segments = [(np.asarray(pts[idx], dtype=float), np.asarray(pts[idx + 1], dtype=float)) for idx in range(len(pts) - 1)]
    for i, (a1, a2) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if j == i + 1:
                continue
            if closed and i == 0 and j == len(segments) - 1:
                continue
            b1, b2 = segments[j]
            if np.linalg.norm(a1 - b1) <= 1e-6 or np.linalg.norm(a1 - b2) <= 1e-6:
                continue
            if np.linalg.norm(a2 - b1) <= 1e-6 or np.linalg.norm(a2 - b2) <= 1e-6:
                continue
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _has_suspicious_near_closure(points, close_distance=8.0, closure_ratio=3.0):
    pts = _remove_duplicate_points(points)
    if len(pts) < 5:
        return False
    if _is_path_closed(pts, close_threshold=min(1.5, close_distance)):
        return False

    start_end = float(np.linalg.norm(np.asarray(pts[0], dtype=float) - np.asarray(pts[-1], dtype=float)))
    total_length = _path_length(pts)
    return start_end <= close_distance and total_length >= max(24.0, closure_ratio * max(start_end, 1.0))


def _path_geometry_rejection_reason(candidate, reference=None, reject_length_inflation=True, reject_near_closure=True):
    cleaned = _remove_duplicate_points(candidate)
    if len(cleaned) < 2:
        return "degenerate geometry"
    if _path_has_self_intersection(cleaned):
        return "self-intersection"

    if reference is not None:
        ref_cleaned = _remove_duplicate_points(reference)
        ref_length = _path_length(ref_cleaned)
        cand_length = _path_length(cleaned)
        if reject_length_inflation and ref_length > 1e-6 and cand_length > ref_length * 1.12:
            return "length inflation"

        if reject_near_closure and not _is_path_closed(ref_cleaned):
            closure_distance = max(3.0, min(10.0, 0.06 * ref_length + 2.5))
            if _has_suspicious_near_closure(cleaned, close_distance=closure_distance, closure_ratio=3.2):
                return "artificial loop closure"

    return None


@lru_cache(maxsize=2048)
def _precondition_path_cached(path_key, tolerance_key):
    cleaned = _remove_duplicate_points(path_key)
    if len(cleaned) < 4:
        return tuple(cleaned)

    even = _resample_even_spacing(cleaned)
    epsilon = max(0.35, float(tolerance_key))
    simplified = rdp_simplify(even, epsilon)
    if len(simplified) < 2:
        simplified = even

    straightened = _straighten_path_runs(simplified, deviation_tol=max(0.35, epsilon * 0.6))
    if len(straightened) >= 4:
        tightened = rdp_simplify(straightened, max(0.25, epsilon * 0.5))
        if len(tightened) >= 2:
            straightened = tightened

    return tuple(_remove_duplicate_points(straightened))


def precondition_path_for_optimization(path, base_tolerance=0.85):
    """Apply a cheap simplification/straightening pass before expensive scoring."""
    path_key = tuple((float(pt[0]), float(pt[1])) for pt in path)
    tolerance_key = round(float(base_tolerance), 3)
    return list(_precondition_path_cached(path_key, tolerance_key))

def smooth_path_spline(path, smoothing_factor=0.01):
    """Smooth path using spline interpolation."""
    if len(path) < 4:
        return path
    
    points = np.array(path)
    y, x = points[:, 0], points[:, 1]
    
    try:
        # Fit spline
        tck, _ = interpolate.splprep([x, y], s=len(x) * smoothing_factor)
        
        # Generate smooth points
        u_new = np.linspace(0, 1, len(x))
        x_new, y_new = interpolate.splev(u_new, tck)
        
        return list(zip(y_new, x_new))
    except:
        return path

def fit_curve_to_path(path, curve_type='polynomial', degree=3):
    """Fit a curve to the path and return smoothed points."""
    if len(path) < 4:
        return path
    
    path_array = np.array(path)
    y, x = path_array[:, 0], path_array[:, 1]
    
    try:
        if curve_type == 'polynomial':
            # Fit polynomial curve
            t = np.linspace(0, 1, len(path))
            
            # Fit polynomials for x and y separately
            x_poly = np.polyfit(t, x, min(degree, len(path) - 1))
            y_poly = np.polyfit(t, y, min(degree, len(path) - 1))
            
            # Generate smooth curve points
            t_smooth = np.linspace(0, 1, len(path))
            x_smooth = np.polyval(x_poly, t_smooth)
            y_smooth = np.polyval(y_poly, t_smooth)
            
            return list(zip(y_smooth, x_smooth))
            
        elif curve_type == 'spline':
            # Use B-spline fitting (faster than interpolation)
            t = np.linspace(0, 1, len(path))
            
            # Simple spline with reduced smoothing for speed
            tck_x = interpolate.splrep(t, x, s=len(x) * 0.001)
            tck_y = interpolate.splrep(t, y, s=len(y) * 0.001)
            
            x_smooth = interpolate.splev(t, tck_x)
            y_smooth = interpolate.splev(t, tck_y)
            
            return list(zip(y_smooth, x_smooth))
            
    except:
        return path
    
    return path

def optimize_path_with_custom_params(path, circle_system, params, initial_score=None, return_diagnostics=False):
    """
    Optimize a path by simplifying the extracted skeleton path directly.

    This keeps the final blue path close to the original skeleton geometry instead of
    generating spline/arc candidates that substantially reshape it.
    """
    optimize_started_at = time.perf_counter()
    requested_tolerance = float(params.get('rdp_tolerance', 4.0))
    simplification_strength = max(0.0, min(100.0, float(params.get('simplification_strength', 50.0))))
    simplification_ratio = simplification_strength / 100.0
    line_fit_strength = max(0.0, min(100.0, float(params.get('line_fit_strength', 22.0))))
    short_path_protection = max(0.0, min(100.0, float(params.get('short_path_protection', 65.0))))
    line_fit_ratio = line_fit_strength / 100.0
    short_path_protection_ratio = short_path_protection / 100.0
    score_preservation_ratio = max(0.70, min(0.999, float(params.get('score_preservation', 80.0)) / 100.0))
    min_path_length = int(params.get('min_path_length', 3))

    reference_path = _remove_duplicate_points(path)
    current_path = reference_path
    best_score = float(initial_score) if initial_score is not None else circle_system.evaluate_path(current_path)

    diagnostics = None
    if return_diagnostics:
        diagnostics = {
            'input_points': int(len(path)),
            'phase_ms': {},
            'counts': {
                'score_evaluations': 0,
                'candidates_considered': 0,
                'candidates_accepted': 0,
            },
        }

    def _record_phase_elapsed(phase_name, started_at):
        if diagnostics is None:
            return
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        diagnostics['phase_ms'][phase_name] = diagnostics['phase_ms'].get(phase_name, 0.0) + elapsed_ms

    def _evaluate_path_score(candidate):
        score_started_at = time.perf_counter()
        score = circle_system.evaluate_path(candidate)
        if diagnostics is not None:
            diagnostics['counts']['score_evaluations'] += 1
            _record_phase_elapsed('score_evaluation', score_started_at)
        return score

    print(f"    Optimizing path: {len(path)} points, initial score: {best_score:.2f}")

    path_length = max(1.0, float(params.get('path_length', len(reference_path))))
    max_path_length = max(path_length, float(params.get('max_path_length', path_length)))
    min_candidate_points = int(min_path_length)
    if len(reference_path) >= max(10, min_path_length * 4) and path_length >= 18.0 and not _is_path_closed(reference_path):
        min_candidate_points = 2

    relative_length = max(0.0, min(1.0, path_length / max_path_length))
    short_weight = (1.0 - relative_length) ** (0.6 + 0.9 * short_path_protection_ratio)
    min_short_factor = 0.85 - (0.65 * short_path_protection_ratio)
    min_short_factor = max(0.20, min(0.95, min_short_factor))
    length_protection = 1.0 - (1.0 - min_short_factor) * short_weight
    effective_simplification_ratio = simplification_ratio * length_protection

    def _curvature_profile(points):
        """Return mean turning angle (deg) and sharp-turn ratio for adaptive tolerance."""
        if len(points) < 3:
            return 0.0, 0.0
        pts = np.asarray(points, dtype=float)
        v1 = pts[1:-1] - pts[:-2]
        v2 = pts[2:] - pts[1:-1]
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        valid = (norm1 > 1e-6) & (norm2 > 1e-6)
        if not np.any(valid):
            return 0.0, 0.0
        v1 = v1[valid]
        v2 = v2[valid]
        norm1 = norm1[valid]
        norm2 = norm2[valid]
        cosang = np.sum(v1 * v2, axis=1) / (norm1 * norm2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosang))
        mean_angle = float(np.mean(angles)) if len(angles) else 0.0
        sharp_ratio = float(np.mean(angles > 35.0)) if len(angles) else 0.0
        return mean_angle, sharp_ratio

    precondition_started_at = time.perf_counter()
    preconditioned_path = precondition_path_for_optimization(
        reference_path,
        base_tolerance=max(0.45, requested_tolerance * 0.22),
    )
    if len(preconditioned_path) >= min_candidate_points and len(preconditioned_path) <= len(reference_path):
        precondition_reason = _path_geometry_rejection_reason(preconditioned_path, reference_path)
        if precondition_reason is None:
            precondition_score = _evaluate_path_score(preconditioned_path)
            if precondition_score >= best_score * max(0.92, score_preservation_ratio - 0.05):
                current_path = preconditioned_path
                best_score = precondition_score
                print(
                    f"      ✓ Accepted magenta preconditioning: {len(reference_path)} -> {len(preconditioned_path)} points, "
                    f"score: {precondition_score:.2f}"
                )
                if diagnostics is not None:
                    diagnostics['counts']['candidates_accepted'] += 1
    _record_phase_elapsed('precondition', precondition_started_at)

    analysis_started_at = time.perf_counter()
    cleaned_path = _remove_duplicate_points(current_path)
    even_path = _resample_even_spacing(cleaned_path)

    diffs = np.diff(np.asarray(even_path, dtype=float), axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1) if len(diffs) else np.array([0.0])
    total_length = float(seg_lengths.sum())
    mean_angle, sharp_ratio = _curvature_profile(even_path)
    chord_length = 0.0
    if len(even_path) >= 2:
        start_point = np.asarray(even_path[0], dtype=float)
        end_point = np.asarray(even_path[-1], dtype=float)
        chord_length = float(np.linalg.norm(end_point - start_point))
    detour_ratio = total_length / max(chord_length, 1e-6) if chord_length > 0.0 else 1.0

    base_tolerance = requested_tolerance * (0.7 + min(total_length / 220.0, 0.7) + 0.35 * line_fit_ratio)
    if sharp_ratio > 0.25 or mean_angle > 25.0:
        base_tolerance *= 0.65
    adaptive_tolerance = max(0.85, min(base_tolerance, requested_tolerance + 2.5))
    _record_phase_elapsed('path_analysis', analysis_started_at)

    def _consider_simplified_candidate(candidate, label, min_ratio=None):
        nonlocal current_path, best_score

        if diagnostics is not None:
            diagnostics['counts']['candidates_considered'] += 1

        candidate = _remove_duplicate_points(candidate)
        if len(candidate) < min_candidate_points or len(candidate) >= len(current_path):
            return

        rejection_reason = _path_geometry_rejection_reason(candidate, reference_path)
        if rejection_reason is not None:
            print(f"      ✗ Rejected {label.lower()} ({rejection_reason}).")
            return

        candidate_score = _evaluate_path_score(candidate)
        score_ratio = max(score_preservation_ratio, min_ratio if min_ratio is not None else 0.0)

        print(
            f"      {label}: {len(current_path)} -> {len(candidate)} points, "
            f"score: {candidate_score:.2f}"
        )

        if candidate_score >= best_score * score_ratio:
            current_path = candidate
            best_score = candidate_score
            print(f"      ✓ Accepted {label.lower()}.")
            if diagnostics is not None:
                diagnostics['counts']['candidates_accepted'] += 1
        else:
            print(f"      ✗ Rejected {label.lower()} due to score drop.")

    if line_fit_ratio > 0.05:
        rdp_started_at = time.perf_counter()
        try:
            simplified_path = rdp_simplify(even_path, adaptive_tolerance)
            if len(simplified_path) >= min_candidate_points:
                rejection_reason = _path_geometry_rejection_reason(simplified_path, reference_path)
                if rejection_reason is not None:
                    print(f"      ✗ Rejected adaptive pre-simplification ({rejection_reason}).")
                    simplified_path = None
            if simplified_path is not None and len(simplified_path) >= min_candidate_points:
                rdp_score = _evaluate_path_score(simplified_path)
                print(
                    f"      Adaptive RDP: {len(current_path)} -> {len(simplified_path)} points, "
                    f"score: {rdp_score:.2f}, tolerance: {adaptive_tolerance:.2f}, "
                    f"mean angle: {mean_angle:.1f}°, sharp ratio: {sharp_ratio:.2f}"
                )

                if rdp_score >= best_score * max(0.96, score_preservation_ratio + 0.02):
                    current_path = simplified_path
                    best_score = rdp_score
                    print("      ✓ Accepted adaptive pre-simplification.")
                    if diagnostics is not None:
                        diagnostics['counts']['candidates_accepted'] += 1
                else:
                    print("      ✗ Rejected RDP due to score drop.")
        except Exception as e:
            print(f"      RDP failed: {e}")
        finally:
            _record_phase_elapsed('adaptive_rdp', rdp_started_at)

    if simplification_ratio > 0 and len(current_path) >= max(4, min_path_length + 1):
        if line_fit_ratio > 0.05:
            try:
                line_fit_started_at = time.perf_counter()
                line_fit_candidate = rdp_simplify(
                    current_path,
                    adaptive_tolerance * (0.75 + 2.0 * effective_simplification_ratio * line_fit_ratio),
                )
                _consider_simplified_candidate(
                    line_fit_candidate,
                    "Line-fit simplification",
                    min_ratio=max(0.96, score_preservation_ratio + 0.02),
                )
            except Exception as e:
                print(f"      Line-fit simplification failed: {e}")
            finally:
                _record_phase_elapsed('line_fit_simplification', line_fit_started_at)

    print(f"    Final: {len(current_path)} points, score: {best_score:.2f}")
    if diagnostics is not None:
        diagnostics['output_points'] = int(len(current_path))
        diagnostics['total_ms'] = (time.perf_counter() - optimize_started_at) * 1000.0
        diagnostics['score_delta'] = float(best_score) - float(initial_score if initial_score is not None else best_score)
        diagnostics['score_delta_ratio'] = 0.0
        if initial_score not in (None, 0):
            diagnostics['score_delta_ratio'] = diagnostics['score_delta'] / float(initial_score)
    if return_diagnostics:
        return current_path, best_score, diagnostics
    return current_path, best_score

def optimize_path_with_circles(path, circle_system, max_iterations=2):  # Reduced iterations for speed
    """Fast optimize a path using circle evaluation feedback with curve fitting."""
    current_path = path
    best_score = circle_system.evaluate_path(current_path)
    
    print(f"    Initial: {len(path)} points, score: {best_score:.2f}")
    
    for iteration in range(max_iterations):
        # Fast optimization approaches - reduced set for speed
        optimizations = [
            ("rdp_fast", lambda p: rdp_simplify(p, RDP_TOLERANCE)),
            ("curve_poly", lambda p: fit_curve_to_path(p, 'polynomial', degree=2)),
            ("curve_spline", lambda p: fit_curve_to_path(p, 'spline')),
            ("adaptive_fast", lambda p: adaptive_resample_path(p, circle_system, target_reduction=0.7))
        ]
        
        best_optimized = current_path
        improvements = []
        
        for name, optimizer in optimizations:
            try:
                optimized = optimizer(current_path)
                if len(optimized) >= MIN_PATH_LENGTH:
                    score = circle_system.evaluate_path(optimized)
                    if score > best_score:
                        improvements.append((name, score, optimized))
                        if score > best_score:
                            best_score = score
                            best_optimized = optimized
            except:
                continue
        
        if improvements:
            best_method = max(improvements, key=lambda x: x[1])
            print(f"      {best_method[0]}: {len(current_path)} -> {len(best_method[2])} points, score: {best_method[1]:.2f}")
        
        if best_optimized == current_path:
            break  # No improvement
        
        current_path = best_optimized
    
    return current_path, best_score

def adaptive_resample_path(path, circle_system, target_reduction=0.7):  # More aggressive reduction for speed
    """Adaptively resample a path based on intensity-field importance."""
    if len(path) < 4:  # Reduced from 6 for better coverage
        return path

    path_array = np.array(path)
    target_points = max(MIN_PATH_LENGTH, int(len(path) * target_reduction))

    if circle_system:
        importance_scores = np.array(
            [circle_system.point_importance(point) for point in path_array], dtype=float
        )
    else:
        importance_scores = np.ones(len(path_array), dtype=float)

    # Always keep first and last points
    resampled = [path_array[0]]

    if len(importance_scores) > 2 and target_points > 2:
        inner_scores = importance_scores[1:-1]
        if np.max(inner_scores) > 0:
            normalized_scores = inner_scores / np.max(inner_scores)
            intermediate_indices = np.argsort(normalized_scores)[::-1][: target_points - 2]
            intermediate_indices = sorted(intermediate_indices)
            for idx in intermediate_indices:
                resampled.append(path_array[idx + 1])  # Offset for excluded first point
        else:
            # Fall back to uniform sampling when all points look identical.
            linspace_indices = np.linspace(1, len(path_array) - 2, target_points - 2, dtype=int)
            for idx in np.unique(linspace_indices):
                resampled.append(path_array[idx])

    resampled.append(path_array[-1])

    return [tuple(p) for p in resampled]

def merge_nearby_paths(paths, max_gap=30, angle_priority=0.30, verbose=True, should_continue=None):
    """Merge nearby endpoints, prioritizing both distance and directional continuity."""
    if len(paths) <= 1:
        return paths

    angle_priority = float(np.clip(angle_priority, 0.0, 1.0))
    distance_priority = 1.0 - angle_priority
    
    if verbose:
        print(
            f"Merging nearby paths (max gap: {max_gap} pixels, "
            f"angle priority: {angle_priority:.2f})..."
        )
    
    merged_paths = []
    used_indices = set()

    def _normalize(vec):
        norm = np.linalg.norm(vec)
        if norm <= 1e-9:
            return None
        return vec / norm

    def _endpoint_tangent(path, endpoint):
        """Return a robust tangent direction near path start/end."""
        if len(path) < 2:
            return None

        lookahead = min(4, len(path) - 1)
        if endpoint == 'start':
            a = np.array(path[0], dtype=float)
            b = np.array(path[lookahead], dtype=float)
        else:
            a = np.array(path[-(lookahead + 1)], dtype=float)
            b = np.array(path[-1], dtype=float)
        return _normalize(b - a)

    def _outward_direction(path, endpoint):
        """Direction that points outward from the selected endpoint into a potential gap."""
        tangent = _endpoint_tangent(path, endpoint)
        if tangent is None:
            return None
        if endpoint == 'start':
            return -tangent
        return tangent

    def _angle_match_score(path_a, endpoint_a, path_b, endpoint_b):
        """Return [0..1] score, where 1 means highly compatible bridge directions."""
        out_a = _outward_direction(path_a, endpoint_a)
        out_b = _outward_direction(path_b, endpoint_b)
        if out_a is None or out_b is None:
            return 0.5  # Neutral when direction cannot be estimated.

        dot = float(np.clip(np.dot(out_a, out_b), -1.0, 1.0))
        # Best bridge is when outward directions oppose each other (dot ~ -1).
        return (1.0 - dot) * 0.5

    def _build_merged_path(path_a, path_b, connection_type):
        if connection_type == 'end_to_start':
            return list(path_a) + list(path_b)
        if connection_type == 'end_to_end':
            return list(path_a) + list(reversed(path_b))
        if connection_type == 'start_to_start':
            return list(reversed(path_b)) + list(path_a)
        if connection_type == 'start_to_end':
            return list(path_b) + list(path_a)
        return list(path_a)

    def _merge_is_safe(path_a, path_b, connection_type):
        merged = _remove_duplicate_points(_build_merged_path(path_a, path_b, connection_type))
        if _path_has_self_intersection(merged):
            return False
        if _has_suspicious_near_closure(merged, close_distance=max(3.0, max_gap * 0.9), closure_ratio=3.0):
            return False
        return True
    
    for i, path1 in enumerate(paths):
        if i in used_indices or len(path1) < 2:
            continue

        if should_continue is not None and not should_continue():
            break
        
        current_path = list(path1)
        
        # Try to extend this path by connecting to other paths (multiple iterations)
        for iteration in range(3):  # Allow multiple merging passes
            extended = False
            best_merge = None

            for j, path2 in enumerate(paths):
                if j == i or j in used_indices or len(path2) < 2:
                    continue
                if _is_path_closed(current_path) or _is_path_closed(path2):
                    continue

                # Check all possible endpoint combinations.
                candidates = []
                raw_connections = [
                    ('end_to_start', current_path[-1], path2[0], 'end', 'start'),
                    ('end_to_end', current_path[-1], path2[-1], 'end', 'end'),
                    ('start_to_start', current_path[0], path2[0], 'start', 'start'),
                    ('start_to_end', current_path[0], path2[-1], 'start', 'end'),
                ]

                for connection_type, p_a, p_b, endpoint_a, endpoint_b in raw_connections:
                    distance = np.linalg.norm(np.array(p_a) - np.array(p_b))
                    if distance > max_gap:
                        continue

                    angle_score = _angle_match_score(
                        current_path, endpoint_a, path2, endpoint_b
                    )
                    if distance > max_gap * 0.55 and angle_score < 0.55:
                        continue
                    if not _merge_is_safe(current_path, path2, connection_type):
                        continue
                    distance_score = 1.0 - (distance / max(max_gap, 1e-9))
                    combined_score = (
                        distance_priority * distance_score
                        + angle_priority * angle_score
                    )
                    candidates.append(
                        (connection_type, distance, angle_score, combined_score)
                    )

                if candidates:
                    local_best = max(candidates, key=lambda x: x[3])
                    connection_type, distance, angle_score, combined_score = local_best
                    if best_merge is None or combined_score > best_merge['combined_score']:
                        best_merge = {
                            'j': j,
                            'path2': path2,
                            'connection_type': connection_type,
                            'distance': distance,
                            'angle_score': angle_score,
                            'combined_score': combined_score,
                        }

            if best_merge is not None:
                path2 = best_merge['path2']
                connection_type = best_merge['connection_type']
                distance = best_merge['distance']
                angle_score = best_merge['angle_score']
                j = best_merge['j']

                # Merge the paths based on connection type
                if connection_type == 'end_to_start':
                    current_path.extend(path2)
                elif connection_type == 'end_to_end':
                    current_path.extend(reversed(path2))
                elif connection_type == 'start_to_start':
                    current_path = list(reversed(path2)) + current_path
                elif connection_type == 'start_to_end':
                    current_path = path2 + current_path

                used_indices.add(j)
                extended = True

                if verbose:
                    print(
                        f"    Merged paths: {len(path1)} + {len(path2)} = {len(current_path)} "
                        f"points (gap: {distance:.1f}, angle_match: {angle_score:.2f})"
                    )

            if not extended:
                break  # No more paths to merge in this pass
        
        merged_paths.append(current_path)
        used_indices.add(i)
    
    if verbose:
        print(f"  Before merging: {len(paths)} paths")
        print(f"  After merging: {len(merged_paths)} paths")
    
    # Show the lengths of the longest merged paths
    if merged_paths:
        path_lengths = [len(path) for path in merged_paths]
        path_lengths.sort(reverse=True)
        if verbose:
            print(f"  Longest merged paths: {path_lengths[:5]}")
    
    return merged_paths

def remove_overlapping_paths(paths, overlap_threshold=0.5, min_distance=10):
    """Remove paths that significantly overlap with other paths."""
    if len(paths) <= 1:
        return paths
    
    print(f"Removing overlapping paths (overlap threshold: {overlap_threshold}, min distance: {min_distance})...")
    
    def path_overlap_ratio(path1, path2, min_distance):
        """Calculate the overlap ratio between two paths."""
        if len(path1) < 2 or len(path2) < 2:
            return 0.0
        
        # Convert paths to sets of nearby points
        path1_points = set()
        path2_points = set()
        
        for y, x in path1:
            # Add points within min_distance radius
            for dy in range(-min_distance, min_distance + 1):
                for dx in range(-min_distance, min_distance + 1):
                    if dy*dy + dx*dx <= min_distance*min_distance:
                        path1_points.add((y + dy, x + dx))
        
        for y, x in path2:
            # Add points within min_distance radius  
            for dy in range(-min_distance, min_distance + 1):
                for dx in range(-min_distance, min_distance + 1):
                    if dy*dy + dx*dx <= min_distance*min_distance:
                        path2_points.add((y + dy, x + dx))
        
        # Calculate overlap
        intersection = len(path1_points.intersection(path2_points))
        union = len(path1_points.union(path2_points))
        
        return intersection / union if union > 0 else 0.0
    
    # Sort paths by length (keep longer paths)
    indexed_paths = [(i, path) for i, path in enumerate(paths)]
    indexed_paths.sort(key=lambda x: len(x[1]), reverse=True)
    
    filtered_paths = []
    removed_count = 0
    
    for i, (orig_idx, path1) in enumerate(indexed_paths):
        is_duplicate = False
        
        # Check against already accepted paths
        for j, (_, accepted_path) in enumerate(indexed_paths[:i]):
            if j < len(filtered_paths):  # Only check paths we've already accepted
                overlap = path_overlap_ratio(path1, filtered_paths[j], min_distance)
                if overlap > overlap_threshold:
                    print(f"    Removing path {orig_idx} (length {len(path1)}) - {overlap:.2f} overlap with path {j}")
                    is_duplicate = True
                    removed_count += 1
                    break
        
        if not is_duplicate:
            filtered_paths.append(path1)
    
    print(f"  Before overlap removal: {len(paths)} paths")
    print(f"  After overlap removal: {len(filtered_paths)} paths (removed {removed_count})")
    
    return filtered_paths

def create_svg_output(
    image,
    circle_system,
    optimized_paths,
    path_scores,
    pre_optimization_paths=None,
    curve_fit_tolerance=1.0,
    endpoint_tangent_strictness=85.0,
    force_orthogonal_as_lines=False,
    enable_curve_fitting=False,
):
    """Create SVG output with bitmap, pre-optimization paths, and optimized paths."""
    print("Creating SVG output...")
    
    height, width = image.shape
    dwg = svgwrite.Drawing(OUTPUT_PATH, profile="tiny", size=(width, height))
    
    # Add bitmap background
    if SHOW_BITMAP:
        print("Adding bitmap background...")
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        dwg.add(dwg.image(href=f'data:image/png;base64,{img_data}', 
                         insert=(0, 0), size=(width, height), opacity=0.3))
    
    # Add pre-optimization paths (before circle optimization)
    if SHOW_PRE_OPTIMIZATION_PATHS and pre_optimization_paths is not None:
        print(f"Adding {len(pre_optimization_paths)} pre-optimization paths...")
        for i, path in enumerate(pre_optimization_paths):
            if len(path) < 2:
                continue
            
            # Create SVG path with +0.5 pixel offset to align with circle centers
            path_data = []
            for j, point in enumerate(path):
                y, x = point
                if j == 0:
                    path_data.append(f"M {x + 0.5:.2f} {y + 0.5:.2f}")
                else:
                    path_data.append(f"L {x + 0.5:.2f} {y + 0.5:.2f}")
            
            dwg.add(dwg.path(
                d=" ".join(path_data),
                stroke=PRE_OPTIMIZATION_PATH_COLOR,
                fill="none",
                stroke_width=PRE_OPTIMIZATION_PATH_WIDTH,
                opacity=PRE_OPTIMIZATION_PATH_OPACITY
            ))
    
    # Fit optimized paths to SVG-native line/cubic segments.
    fitted_segments = fit_curve_segments(
        optimized_paths,
        tolerance_px=max(0.35, float(curve_fit_tolerance)),
        endpoint_tangent_strictness=float(endpoint_tangent_strictness),
        force_orthogonal_as_lines=bool(force_orthogonal_as_lines),
        enable_curve_fitting=bool(enable_curve_fitting),
    )

    # Add optimized paths
    print(f"Adding {len(optimized_paths)} optimized paths...")
    for i, (path, score, segments) in enumerate(zip(optimized_paths, path_scores, fitted_segments)):
        if len(path) < 2:
            continue
        
        # Color all paths with consistent blue color
        color = "#0066CC"  # Consistent blue hex color
        width = 2.0
        opacity = 1.0  # Keep consistent opacity for all optimized paths
        
        # Emit true SVG cubic segments when available.
        path_data = []
        if segments:
            start_point = segments[0].get("start_point", list(path[0]))
            sy, sx = float(start_point[0]), float(start_point[1])
            path_data.append(f"M {sx + 0.5:.2f} {sy + 0.5:.2f}")

            for seg in segments:
                end = seg.get("end_point")
                if not isinstance(end, (list, tuple)) or len(end) != 2:
                    continue

                ey, ex = float(end[0]), float(end[1])
                if seg.get("type") == "cubic":
                    c1 = seg.get("control1")
                    c2 = seg.get("control2")
                    if (
                        isinstance(c1, (list, tuple))
                        and len(c1) == 2
                        and isinstance(c2, (list, tuple))
                        and len(c2) == 2
                    ):
                        c1y, c1x = float(c1[0]), float(c1[1])
                        c2y, c2x = float(c2[0]), float(c2[1])
                        path_data.append(
                            f"C {c1x + 0.5:.2f} {c1y + 0.5:.2f} "
                            f"{c2x + 0.5:.2f} {c2y + 0.5:.2f} "
                            f"{ex + 0.5:.2f} {ey + 0.5:.2f}"
                        )
                        continue

                path_data.append(f"L {ex + 0.5:.2f} {ey + 0.5:.2f}")
        else:
            for j, point in enumerate(path):
                y, x = point
                if j == 0:
                    path_data.append(f"M {x + 0.5:.2f} {y + 0.5:.2f}")
                else:
                    path_data.append(f"L {x + 0.5:.2f} {y + 0.5:.2f}")
        
        dwg.add(dwg.path(
            d=" ".join(path_data),
            stroke=color,
            fill="none",
            stroke_width=width,
            opacity=opacity
        ))
        
        # Add score annotation for best path
        # if i == 0 and len(path) > 0:
        #     start_y, start_x = path[0]
        #     dwg.add(dwg.text(
        #         f"Best Path (Score: {score:.2f})",
        #         insert=(start_x + 10, start_y - 10),
        #         fill=BEST_PATH_COLOR,
        #         font_size="12px",
        #         font_weight="bold"
        #     ))
    
    dwg.save()
    print(f"SVG saved to: {OUTPUT_PATH}")

def create_fast_paths(skeleton):
    """Extract branch-aware paths from a skeleton while preserving open segments."""
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return []

    print(f"  Starting fast path extraction on {len(ys)} skeleton pixels...")

    skeleton_points = set(zip(ys, xs))
    paths = []

    # Pre-compute 8-connected neighbors.
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    neighbors_dict = {}
    for y, x in skeleton_points:
        neighbors = []
        for dy, dx in offsets:
            candidate = (y + dy, x + dx)
            if candidate in skeleton_points:
                neighbors.append(candidate)
        neighbors_dict[(y, x)] = neighbors

    # Junctions/endpoints (degree != 2) are graph nodes; chains between them are edges.
    node_points = {p for p, n in neighbors_dict.items() if len(n) != 2}

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    visited_edges = set()

    def trace_edge(start, neighbor):
        """Trace one edge from a node to the next node (or dead-end)."""
        path = [start, neighbor]
        visited_edges.add(edge_key(start, neighbor))

        prev = start
        current = neighbor

        while True:
            if current in node_points and current != start:
                return path

            candidates = [n for n in neighbors_dict[current] if n != prev]
            if not candidates:
                return path

            next_point = None
            for cand in candidates:
                if edge_key(current, cand) not in visited_edges:
                    next_point = cand
                    break

            if next_point is None:
                return path

            visited_edges.add(edge_key(current, next_point))
            path.append(next_point)
            prev, current = current, next_point

    # Trace all node-connected edges first so open branches are preserved.
    for node in node_points:
        for neighbor in neighbors_dict[node]:
            key = edge_key(node, neighbor)
            if key in visited_edges:
                continue
            path = trace_edge(node, neighbor)
            if len(path) >= MIN_PATH_LENGTH:
                paths.append(path)

    # Handle pure loops (all points degree==2), which have no explicit nodes.
    for point in skeleton_points:
        for neighbor in neighbors_dict[point]:
            key = edge_key(point, neighbor)
            if key in visited_edges:
                continue

            loop_path = [point, neighbor]
            visited_edges.add(key)
            prev, current = point, neighbor

            while True:
                candidates = [n for n in neighbors_dict[current] if n != prev]
                if not candidates:
                    break

                next_point = None
                for cand in candidates:
                    cand_key = edge_key(current, cand)
                    if cand_key not in visited_edges:
                        next_point = cand
                        visited_edges.add(cand_key)
                        break

                if next_point is None:
                    break

                loop_path.append(next_point)
                prev, current = current, next_point

                if current == point:
                    break

            if len(loop_path) >= MIN_PATH_LENGTH:
                paths.append(loop_path)

    print(f"  Fast path extraction complete: {len(paths)} paths kept (min length: {MIN_PATH_LENGTH})")
    return paths

def create_super_long_paths(skeleton):
    """Create the longest possible continuous paths by very aggressively following skeleton with built-in overlap prevention."""
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return []
    
    skeleton_points = set(zip(ys, xs))
    global_visited = set()
    paths = []
    occupied_regions = []  # Track occupied regions to prevent overlaps
    OVERLAP_DISTANCE = 8  # Minimum distance between path segments
    
    def get_all_neighbors(y, x):
        """Get all skeleton neighbors regardless of visited status."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in skeleton_points:
                    neighbors.append((ny, nx))
        return neighbors
    
    def would_overlap_with_existing_paths(point, existing_paths):
        """Check if a point would create an overlap with existing paths."""
        py, px = point
        for path in existing_paths:
            for path_point in path:
                distance = np.sqrt((py - path_point[0])**2 + (px - path_point[1])**2)
                if distance < OVERLAP_DISTANCE:
                    return True
        return False
    
    def trace_ultra_long_path(start_point):
        """Trace the absolute longest path possible, going through branch points, with overlap prevention."""
        # Check if starting point would overlap with existing paths
        if would_overlap_with_existing_paths(start_point, paths):
            return [], set()
            
        path = [start_point]
        local_visited = {start_point}
        current = start_point
        stuck_count = 0
        max_stuck = 3  # Allow some backtracking
        
        while stuck_count < max_stuck:
            all_neighbors = get_all_neighbors(current[0], current[1])
            
            # Prefer unvisited neighbors, but also check for overlaps
            unvisited_neighbors = [n for n in all_neighbors if n not in local_visited]
            # Filter out neighbors that would create overlaps
            non_overlapping_neighbors = [n for n in unvisited_neighbors 
                                       if not would_overlap_with_existing_paths(n, paths)]
            
            if non_overlapping_neighbors:
                stuck_count = 0  # Reset stuck counter
                
                # Choose next point to maximize path length
                if len(path) >= 2:
                    # Calculate current direction
                    direction = np.array(current) - np.array(path[-2])
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0:
                        direction = direction / direction_norm
                        
                        # Score neighbors by direction alignment and distance from existing path
                        scored_neighbors = []
                        for neighbor in non_overlapping_neighbors:
                            neighbor_direction = np.array(neighbor) - np.array(current)
                            neighbor_norm = np.linalg.norm(neighbor_direction)
                            
                            if neighbor_norm > 0:
                                neighbor_direction = neighbor_direction / neighbor_norm
                                alignment = np.dot(direction, neighbor_direction)
                                
                                # Bonus for continuing in same direction
                                score = alignment
                                
                                # Bonus for being far from path start (encourages long paths)
                                start_distance = np.linalg.norm(np.array(neighbor) - np.array(path[0]))
                                score += start_distance * 0.01  # Small bonus for distance
                                
                                scored_neighbors.append((neighbor, score))
                        
                        if scored_neighbors:
                            # Choose the neighbor with best score
                            best_neighbor = max(scored_neighbors, key=lambda x: x[1])[0]
                            current = best_neighbor
                            path.append(current)
                            local_visited.add(current)
                            continue
                
                # Fallback: choose any non-overlapping unvisited neighbor
                current = non_overlapping_neighbors[0]
                path.append(current)
                local_visited.add(current)
            
            elif all_neighbors:
                # All neighbors visited or would create overlaps - try to find a way to continue
                stuck_count += 1
                
                # Look for neighbors that might lead to unvisited areas without overlaps
                for neighbor in all_neighbors:
                    if not would_overlap_with_existing_paths(neighbor, paths):
                        neighbor_neighbors = get_all_neighbors(neighbor[0], neighbor[1])
                        unvisited_from_neighbor = [n for n in neighbor_neighbors 
                                                 if n not in local_visited and 
                                                 not would_overlap_with_existing_paths(n, paths)]
                        
                        if unvisited_from_neighbor:
                            current = neighbor
                            path.append(current)
                            local_visited.add(current)
                            stuck_count = 0
                            break
            else:
                # No neighbors at all
                break
        
        return path, local_visited
    
    # Sort starting points to maximize path length
    point_scores = []
    for point in skeleton_points:
        neighbors = get_all_neighbors(point[0], point[1])
        degree = len(neighbors)
        
        # Calculate distance to image center to prioritize central starting points
        center_y, center_x = len(skeleton) // 2, len(skeleton[0]) // 2
        center_distance = np.linalg.norm(np.array(point) - np.array([center_y, center_x]))
        
        # Score: prefer endpoints but also consider central locations
        if degree == 1:
            score = 1000 - center_distance  # Endpoints get highest priority
        elif degree == 2:
            score = 500 - center_distance   # Simple points get medium priority
        else:
            score = 100 - center_distance   # Branch points get lower priority but still considered
        
        point_scores.append((point, score))
    
    # Sort by score (highest first)
    point_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Trace paths starting from best points
    for start_point, score in point_scores:
        if start_point not in global_visited:
            path, local_visited = trace_ultra_long_path(start_point)
            
            if len(path) >= MIN_PATH_LENGTH:
                paths.append(path)
                global_visited.update(local_visited)
                
                # If we got a very long path, prioritize it
                if len(path) > 100:
                    print(f"  Found super long path with {len(path)} points!")
    
    # Handle any remaining unvisited areas with a different strategy
    remaining_points = skeleton_points - global_visited
    if remaining_points:
        print(f"  Processing {len(remaining_points)} remaining skeleton points...")
        
        # Group remaining points into connected components and trace each
        while remaining_points:
            start_point = next(iter(remaining_points))
            path, local_visited = trace_ultra_long_path(start_point)
            
            if len(path) >= MIN_PATH_LENGTH:
                paths.append(path)
            
            global_visited.update(local_visited)
            remaining_points -= local_visited
    
    return paths

class CenterlineMLTrainer:
    """Machine learning trainer that learns from manually drawn centerlines."""
    
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.model = None
            self.scaler = None
        self.features = []
        self.scores = []
        self.is_trained = False
        
    def extract_path_features(self, path, image, circle_system):
        """Extract features from a path for machine learning."""
        if len(path) < 2:
            return []
            
        path_array = np.array(path)
        features = []
        
        # Basic path statistics
        path_length = len(path)
        total_distance = np.sum([np.linalg.norm(path_array[i] - path_array[i-1]) 
                                for i in range(1, len(path_array))])
        
        features.extend([
            path_length,
            total_distance,
            total_distance / path_length if path_length > 0 else 0,  # Average segment length
        ])
        
        # Intensity-field features along the path
        if circle_system:
            profile = circle_system.path_intensity_profile(path, density=0.4)
        else:
            profile = np.array([])

        if profile.size > 0:
            features.extend([
                float(np.mean(profile)),
                float(np.max(profile)),
                float(np.min(profile)),
                float(np.std(profile)),
                float(np.mean(profile > 0.5)),  # Fraction of strong field samples
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
            
        # Path smoothness features
        if len(path_array) > 2:
            angles = []
            for i in range(1, len(path_array) - 1):
                v1 = path_array[i] - path_array[i-1]
                v2 = path_array[i+1] - path_array[i]
                
                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
            
            features.extend([
                np.mean(angles) if angles else 0,
                np.std(angles) if angles else 0,
            ])
        else:
            features.extend([0, 0])
            
        return features
    
    def add_manual_centerline(self, path, image, circle_system, quality_score=10.0):
        """Add a manually drawn centerline as training data."""
        features = self.extract_path_features(path, image, circle_system)
        if features:
            self.features.append(features)
            self.scores.append(quality_score)
            print(f"Added manual centerline with {len(path)} points, quality score: {quality_score}")
    
    def add_manual_training_data(self, manual_centerlines, quality_scores=None):
        """Add manual centerlines as training data for ML model."""
        if not self.ml_trainer:
            print("ML trainer not available (sklearn not installed)")
            return
            
        if quality_scores is None:
            quality_scores = [10.0] * len(manual_centerlines)  # Default high quality score
        
        for i, centerline in enumerate(manual_centerlines):
            score = quality_scores[i] if i < len(quality_scores) else 10.0
            self.ml_trainer.add_manual_centerline(centerline, self.image, self, score)
    
    def train_model(self):
        """Train the machine learning model on collected data."""
        if len(self.features) < 2:
            print("Need at least 2 training examples to train model")
            return False
            
        X = np.array(self.features)
        y = np.array(self.scores)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"Trained ML model on {len(self.features)} examples")
        print(f"Feature importance: {self.model.feature_importances_[:5]}...")  # Show first 5
        
        return True
    
    def predict_score(self, path, image, circle_system):
        """Predict the quality score for a path using the trained model."""
        if not self.is_trained:
            return None
            
        features = self.extract_path_features(path, image, circle_system)
        if not features:
            return None
            
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)[0]
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'scores': self.scores
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Saved ML model to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
            self.scores = model_data['scores']
            self.is_trained = True
            
            print(f"Loaded ML model from {filepath} with {len(self.features)} training examples")
            return True
        except FileNotFoundError:
            print(f"No existing model found at {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def save_manual_centerline(path, filepath):
    """Save a manually drawn centerline to JSON file."""
    centerline_data = {
        'path': [(float(point[0]), float(point[1])) for point in path],
        'timestamp': str(np.datetime64('now')),
        'length': len(path)
    }
    
    # Load existing data if file exists
    existing_data = []
    try:
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        pass
    
    # Add new centerline
    existing_data.append(centerline_data)
    
    # Save back to file
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"Saved manual centerline with {len(path)} points to {filepath}")

def load_manual_centerlines(filepath):
    """Load manually drawn centerlines from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        centerlines = []
        for item in data:
            path = [(point[0], point[1]) for point in item['path']]
            centerlines.append(path)
        
        print(f"Loaded {len(centerlines)} manual centerlines from {filepath}")
        return centerlines
    except FileNotFoundError:
        print(f"No manual centerlines found at {filepath}")
        return []
    except Exception as e:
        print(f"Error loading manual centerlines: {e}")
        return []

def create_annotation_svg(image, output_path):
    """Create a simple SVG with just the bitmap for manual centerline annotation."""
    height, width = image.shape
    dwg = svgwrite.Drawing(output_path, profile="tiny", size=(width, height))
    
    # Add bitmap background
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    dwg.add(dwg.image(href=f'data:image/png;base64,{img_data}', 
                     insert=(0, 0), size=(width, height), opacity=0.8))
    
    # Add instructions as text
    dwg.add(dwg.text(
        "Draw centerline manually, then save coordinates to JSON",
        insert=(10, 20),
        fill="red",
        font_size="14px",
        font_weight="bold"
    ))
    
    dwg.save()
    print(f"Annotation SVG saved to: {output_path}")

def parse_svg_path_to_centerline(svg_path_string):
    """Parse an SVG path string to extract centerline coordinates."""
    # This is a simplified parser - in practice you'd want a more robust solution
    import re
    
    # Extract coordinates from SVG path commands (M, L, etc.)
    coords = re.findall(r'[ML]\s*([\d.]+)\s*([\d.]+)', svg_path_string)
    centerline = [(float(y), float(x)) for x, y in coords]  # Note: SVG uses x,y but we use y,x
    
    return centerline

def parse_svg_file_for_paths(svg_filepath):
    """Parse an SVG file to extract manually drawn path coordinates."""
    try:
        import xml.etree.ElementTree as ET
        
        # Parse the SVG file
        tree = ET.parse(svg_filepath)
        root = tree.getroot()
        
        # Define namespace for SVG
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        
        paths = []
        
        # Find all path elements in the SVG
        for path_elem in root.findall('.//svg:path', namespace):
            path_data = path_elem.get('d')
            if path_data:
                coordinates = parse_svg_path_data(path_data)
                if coordinates and len(coordinates) >= 3:  # Minimum path length
                    paths.append(coordinates)
                    print(f"Found path with {len(coordinates)} points")
        
        # Also check for path elements without namespace
        for path_elem in root.findall('.//path'):
            path_data = path_elem.get('d')
            if path_data:
                coordinates = parse_svg_path_data(path_data)
                if coordinates and len(coordinates) >= 3:
                    paths.append(coordinates)
                    print(f"Found path with {len(coordinates)} points")
        
        return paths
        
    except Exception as e:
        print(f"Error parsing SVG file: {e}")
        return []

def parse_svg_path_data(path_data):
    """Parse SVG path data string to extract coordinates."""
    import re
    
    coordinates = []
    
    try:
        # Remove extra whitespace and normalize
        path_data = re.sub(r'\s+', ' ', path_data.strip())
        
        # Split by commands (M, L, C, Q, etc.)
        commands = re.findall(r'[MmLlHhVvCcSsQqTtAaZz][^MmLlHhVvCcSsQqTtAaZz]*', path_data)
        
        current_x, current_y = 0, 0
        
        for command in commands:
            cmd = command[0]
            params = command[1:].strip()
            
            if not params:
                continue
                
            # Extract numbers from parameters
            numbers = re.findall(r'-?\d*\.?\d+', params)
            numbers = [float(n) for n in numbers]
            
            if cmd.upper() == 'M':  # MoveTo
                if len(numbers) >= 2:
                    if cmd.isupper():  # Absolute
                        current_x, current_y = numbers[0], numbers[1]
                    else:  # Relative
                        current_x += numbers[0]
                        current_y += numbers[1]
                    coordinates.append((current_y, current_x))  # Note: we use (y,x) format
                    
                    # Handle additional coordinate pairs as LineTo
                    for i in range(2, len(numbers), 2):
                        if i + 1 < len(numbers):
                            if cmd.isupper():
                                current_x, current_y = numbers[i], numbers[i+1]
                            else:
                                current_x += numbers[i]
                                current_y += numbers[i+1]
                            coordinates.append((current_y, current_x))
            
            elif cmd.upper() == 'L':  # LineTo
                for i in range(0, len(numbers), 2):
                    if i + 1 < len(numbers):
                        if cmd.isupper():  # Absolute
                            current_x, current_y = numbers[i], numbers[i+1]
                        else:  # Relative
                            current_x += numbers[i]
                            current_y += numbers[i+1]
                        coordinates.append((current_y, current_x))
            
            elif cmd.upper() == 'H':  # Horizontal LineTo
                for x in numbers:
                    if cmd.isupper():
                        current_x = x
                    else:
                        current_x += x
                    coordinates.append((current_y, current_x))
            
            elif cmd.upper() == 'V':  # Vertical LineTo
                for y in numbers:
                    if cmd.isupper():
                        current_y = y
                    else:
                        current_y += y
                    coordinates.append((current_y, current_x))
            
            elif cmd.upper() == 'C':  # Cubic Bezier
                # For curves, we'll sample points along the curve
                for i in range(0, len(numbers), 6):
                    if i + 5 < len(numbers):
                        if cmd.isupper():
                            end_x, end_y = numbers[i+4], numbers[i+5]
                        else:
                            end_x = current_x + numbers[i+4]
                            end_y = current_y + numbers[i+5]
                        
                        # Sample a few points between current and end for curves
                        steps = max(3, int(abs(end_x - current_x) + abs(end_y - current_y)) // 10)
                        for step in range(1, steps + 1):
                            t = step / steps
                            interp_x = current_x + t * (end_x - current_x)
                            interp_y = current_y + t * (end_y - current_y)
                            coordinates.append((interp_y, interp_x))
                        
                        current_x, current_y = end_x, end_y
        
        return coordinates
        
    except Exception as e:
        print(f"Error parsing path data: {e}")
        return []

def extract_manual_centerlines_from_svg(svg_filepath, quality_score=10.0):
    """Extract manual centerlines from SVG and add them to training data."""
    print(f"Extracting manual centerlines from: {svg_filepath}")
    
    paths = parse_svg_file_for_paths(svg_filepath)
    
    if not paths:
        print("No valid paths found in SVG file")
        return []
    
    print(f"Found {len(paths)} manual centerline paths")
    
    # Save each path to the manual centerline JSON file
    all_saved_paths = []
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {len(path)} points")
        save_manual_centerline(path, MANUAL_CENTERLINE_PATH)
        all_saved_paths.append(path)
    
    return all_saved_paths

def main():
    """Main processing pipeline."""
    
    # Load and preprocess image
    print(f"Loading image: {INPUT_PATH}")
    img = io.imread(INPUT_PATH)
    
    if len(img.shape) == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(float) / 255.0 if img.dtype != float else img
    
    print(f"Image loaded: {gray.shape}, intensity range: {gray.min():.3f} - {gray.max():.3f}")
    
    # Extract initial skeleton paths with improved coverage
    initial_paths = extract_skeleton_paths(gray, DARK_THRESHOLD, min_object_size=5)
    
    if len(initial_paths) == 0:
        print("ERROR: No skeleton paths found. Try adjusting DARK_THRESHOLD.")
        return
    
    # Merge nearby paths for better connectivity with aggressive settings
    merged_paths = merge_nearby_paths(initial_paths, max_gap=25)
    
    # Filter paths by minimum length
    valid_paths = [path for path in merged_paths if len(path) >= MIN_PATH_LENGTH]
    print(f"Valid paths after length filtering: {len(valid_paths)}")
    
    if len(valid_paths) == 0:
        print("ERROR: No valid paths after filtering. Try reducing MIN_PATH_LENGTH.")
        return
    
    # Initialize circle evaluation system
    print("Initializing circle evaluation system...")
    circle_system = CircleEvaluationSystem(
        gray, 
        DARK_THRESHOLD,
        MAX_CIRCLE_RADIUS,
        MIN_CIRCLE_RADIUS, 
        CIRCLE_INTERSECTION_BONUS
    )
    
    # Evaluate all paths
    print("Evaluating paths with circle system...")
    path_scores, stats = circle_system.evaluate_all_paths(valid_paths)
    
    if len(path_scores) == 0:
        print("ERROR: No path scores computed.")
        return
    
    print(f"Path evaluation complete:")
    print(f"  Score range: {stats['min']:.2f} - {stats['max']:.2f}")
    print(f"  Mean score: {stats['mean']:.2f}")
    print(f"  Score std dev: {stats['std']:.2f}")
    
    # Fast optimization: Only optimize top 5 paths for speed
    FAST_OPTIMIZE_COUNT = min(5, len(valid_paths))
    print(f"Fast optimization: Processing top {FAST_OPTIMIZE_COUNT} paths...")
    sorted_indices = np.argsort(path_scores)[::-1]  # Best first
    top_indices = sorted_indices[:FAST_OPTIMIZE_COUNT]
    
    optimized_paths = []
    optimized_scores = []
    
    for i, idx in enumerate(top_indices):
        path = valid_paths[idx]
        original_score = path_scores[idx]
        
        print(f"  Fast optimizing path {i+1}/{len(top_indices)} (original score: {original_score:.2f})...")
        
        optimized_path, optimized_score = optimize_path_with_circles(path, circle_system)
        
        optimized_paths.append(optimized_path)
        optimized_scores.append(optimized_score)
        
        print(f"    Final: {len(optimized_path)} points, score: {original_score:.2f} -> {optimized_score:.2f}")
    
    # Add remaining unoptimized paths for visualization (up to OPTIMIZE_TOP_N_PATHS total)
    remaining_count = min(OPTIMIZE_TOP_N_PATHS - len(optimized_paths), len(valid_paths) - len(optimized_paths))
    if remaining_count > 0:
        print(f"  Adding {remaining_count} unoptimized paths for visualization...")
        remaining_indices = sorted_indices[len(optimized_paths):len(optimized_paths) + remaining_count]
        for idx in remaining_indices:
            optimized_paths.append(valid_paths[idx])
            optimized_scores.append(path_scores[idx])
    
    # Get corresponding pre-optimization paths for visualization
    all_indices = sorted_indices[:len(optimized_paths)]
    top_pre_optimization_paths = [valid_paths[idx] for idx in all_indices]
    
    # Create SVG output with both pre-optimization and optimized paths
    create_svg_output(gray, circle_system, optimized_paths, optimized_scores, top_pre_optimization_paths)
    
    # Print summary
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best path: {len(optimized_paths[0])} points, score: {optimized_scores[0]:.2f}")
    print(f"Total optimized paths: {len(optimized_paths)}")
    print(f"Output saved to: {OUTPUT_PATH}")

def load_training_paths(image_shape):
    """Load manual training paths from JSON file and transform them to match current image coordinates."""
    if not os.path.exists(MANUAL_CENTERLINE_PATH):
        return []
    
    try:
        with open(MANUAL_CENTERLINE_PATH, 'r') as f:
            training_data = json.load(f)
        
        current_height, current_width = image_shape
        
        # Assume the training data was created on an annotation SVG with standard dimensions
        # Based on typical annotation SVG generation, assume 800x600 or use image aspect ratio
        assumed_training_width = 800
        assumed_training_height = 600
        
        # Calculate scaling factors to map from assumed training size to current image
        width_scale = current_width / assumed_training_width
        height_scale = current_height / assumed_training_height
        
        training_paths = []
        for entry in training_data:
            if 'path' in entry:
                path = entry['path']
                if len(path) == 0:
                    continue
                
                # Transform coordinates from training annotation to current image
                transformed_path = []
                for x, y in path:
                    # Scale the coordinates
                    scaled_x = x * width_scale
                    scaled_y = y * height_scale
                    
                    # Flip Y coordinate (SVG coordinates have Y=0 at top, but we may need to flip)
                    # If the training data appears upside down, flip it
                    flipped_y = current_height - scaled_y  # Flip vertically
                    
                    # Convert to [y, x] format for consistency with skeleton paths
                    transformed_path.append([flipped_y, scaled_x])
                
                training_paths.append(transformed_path)
        
        print(f"Loaded and transformed {len(training_paths)} training paths for visualization")
        if len(training_paths) > 0:
            print(f"  Image dimensions: {current_height}x{current_width}")
            print(f"  Training assumed dimensions: {assumed_training_height}x{assumed_training_width}")
            print(f"  Scale factors: width={width_scale:.3f}, height={height_scale:.3f}")
            
        return training_paths
    except Exception as e:
        print(f"Error loading training paths: {e}")
        return []

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-annotation":
        # Create annotation SVG for manual drawing
        print("Creating annotation SVG...")
        img = io.imread(INPUT_PATH)
        if len(img.shape) == 3:
            gray = color.rgb2gray(img)
        else:
            gray = img.astype(float) / 255.0 if img.dtype != float else img
        
        annotation_path = "/Users/nicholas/Downloads/annotation_template.svg"
        create_annotation_svg(gray, annotation_path)
        print(f"Open {annotation_path} in a vector editor like Inkscape or Adobe Illustrator")
        print("Draw your ideal centerline as a path, then extract coordinates to JSON format")
        print("Example JSON format:")
        print('[{"path": [[y1,x1], [y2,x2], ...], "timestamp": "...", "length": N}]')
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--extract-from-svg":
        # Extract manual centerlines from SVG file
        if len(sys.argv) < 3:
            print("Usage: python script.py --extract-from-svg path/to/file.svg")
            print("This will extract any path elements from the SVG as manual centerlines")
        else:
            svg_file = sys.argv[2]
            if os.path.exists(svg_file):
                paths = extract_manual_centerlines_from_svg(svg_file)
                print(f"Extracted {len(paths)} centerlines from SVG")
                print("Run the main script to retrain the model with this data.")
            else:
                print(f"SVG file not found: {svg_file}")
                
    elif len(sys.argv) > 1 and sys.argv[1] == "--add-manual-centerline":
        # Example of adding a manual centerline
        if len(sys.argv) < 3:
            print("Usage: python script.py --add-manual-centerline 'path_coordinates'")
            print("Example: python script.py --add-manual-centerline '[[100,200],[101,201],[102,202]]'")
        else:
            try:
                import ast
                path_coords = ast.literal_eval(sys.argv[2])
                save_manual_centerline(path_coords, MANUAL_CENTERLINE_PATH)
                print("Manual centerline saved! Run the main script to retrain the model.")
            except Exception as e:
                print(f"Error parsing coordinates: {e}")
    else:
        main()


# ─────────────────────────────────────────────────────────────────────────────
# CURVE FITTING API — stub implementation (replace during overhaul)
# ─────────────────────────────────────────────────────────────────────────────

def fit_curve_segments(
    paths: list,
    tolerance_px: float = 1.0,
    endpoint_tangent_strictness: float = 85.0,
    force_orthogonal_as_lines: bool = False,
    enable_curve_fitting: bool = False,
) -> list:
    """Fit line/cubic segments to paths using Schneider-style iterative fitting."""

    fit_error = max(0.35, float(tolerance_px))
    tangent_blend = max(0.0, min(1.0, float(endpoint_tangent_strictness) / 100.0))

    def _as_point(pt):
        return np.array([float(pt[0]), float(pt[1])], dtype=float)

    def _as_float_pair(pt):
        return [float(pt[0]), float(pt[1])]

    def _clean_points(path_points):
        cleaned = []
        for pt in path_points:
            p = _as_point(pt)
            if not cleaned or np.linalg.norm(p - cleaned[-1]) > 1e-9:
                cleaned.append(p)
        return cleaned

    def _normalize(v):
        n = np.linalg.norm(v)
        if n <= 1e-9:
            return np.array([0.0, 0.0], dtype=float)
        return v / n

    def _line_distance_stats(points):
        if len(points) <= 2:
            return 0.0, 0.0
        p0 = points[0]
        p3 = points[-1]
        seg = p3 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len <= 1e-9:
            d = [float(np.linalg.norm(p - p0)) for p in points[1:-1]]
            return (float(max(d)) if d else 0.0, float(np.mean(d)) if d else 0.0)

        distances = []
        for p in points[1:-1]:
            cross = seg[0] * (p[1] - p0[1]) - seg[1] * (p[0] - p0[0])
            distances.append(abs(float(cross)) / seg_len)
        return (float(max(distances)) if distances else 0.0, float(np.mean(distances)) if distances else 0.0)

    def _max_turn_angle(points):
        if len(points) < 3:
            return 0.0

        max_angle = 0.0
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i - 1]
            v2 = points[i + 1] - points[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 <= 1e-9 or n2 <= 1e-9:
                continue
            cosang = np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cosang)))
            if angle > max_angle:
                max_angle = angle
        return max_angle

    def _axis_step_ratio(points):
        """Return fraction of rounded steps that are axis-aligned."""
        if len(points) < 2:
            return 1.0

        axis_steps = 0
        total_steps = 0
        ints = [
            np.array([int(round(p[0])), int(round(p[1]))], dtype=int)
            for p in points
        ]

        for i in range(1, len(ints)):
            dr = int(ints[i][0] - ints[i - 1][0])
            dc = int(ints[i][1] - ints[i - 1][1])
            if dr == 0 and dc == 0:
                continue
            total_steps += 1
            if dr == 0 or dc == 0:
                axis_steps += 1

        if total_steps == 0:
            return 1.0
        return float(axis_steps) / float(total_steps)

    def _project_point_onto_line(point, line_start, line_end):
        line_vec = line_end - line_start
        denom = float(np.dot(line_vec, line_vec))
        if denom <= 1e-9:
            return point.copy(), 0.0
        t = float(np.dot(point - line_start, line_vec) / denom)
        return line_start + (t * line_vec), t

    def _repair_terminal_line_hook(points, at_start=True):
        if len(points) < 6:
            return points

        subset = points[1:] if at_start else points[:-1]
        if len(subset) < 5:
            return points

        max_dev, mean_dev = _line_distance_stats(subset)
        if max_dev > max(0.55, fit_error * 0.9) or mean_dev > max(0.22, fit_error * 0.35):
            return points
        if _max_turn_angle(subset) > 12.0:
            return points

        terminal_point = points[0] if at_start else points[-1]
        neighbor_point = points[1] if at_start else points[-2]
        trend_start = subset[0]
        trend_end = subset[-1]
        trend_vec = trend_end - trend_start
        terminal_step = neighbor_point - terminal_point if at_start else terminal_point - neighbor_point

        trend_norm = float(np.linalg.norm(trend_vec))
        step_norm = float(np.linalg.norm(terminal_step))
        if trend_norm <= 1e-9 or step_norm <= 1e-9:
            return points

        step_dir = terminal_step / step_norm
        trend_dir = trend_vec / trend_norm
        alignment = float(np.dot(step_dir, trend_dir))
        if alignment >= 0.94:
            return points

        projected, t = _project_point_onto_line(terminal_point, trend_start, trend_end)
        distance_to_line = float(np.linalg.norm(projected - terminal_point))
        if distance_to_line > max(1.5, fit_error * 1.25):
            return points

        if at_start:
            if t < -0.35 or t > 0.08:
                return points
        else:
            if t < 0.92 or t > 1.35:
                return points

        neighbor_projected, _ = _project_point_onto_line(neighbor_point, trend_start, trend_end)
        extension = projected - neighbor_projected
        extension_len = float(np.linalg.norm(extension))
        if extension_len > 1e-6 and extension_len <= max(1.5, fit_error * 1.25):
            projected = projected + extension

        repaired = list(points)
        repaired[0 if at_start else -1] = projected
        return repaired

    def _suppress_terminal_line_hooks(points):
        repaired = _repair_terminal_line_hook(points, at_start=True)
        repaired = _repair_terminal_line_hook(repaired, at_start=False)
        return repaired

    def _extend_uniform_straight_terminals(points, min_points=20):
        if len(points) < min_points:
            return points

        rounded = [
            np.array([int(round(p[0])), int(round(p[1]))], dtype=int)
            for p in points
        ]
        steps = []
        for i in range(1, len(rounded)):
            step = rounded[i] - rounded[i - 1]
            if step[0] == 0 and step[1] == 0:
                continue
            steps.append(step)

        if not steps:
            return points

        first_step = steps[0]
        if np.max(np.abs(first_step)) != 1:
            return points
        if not all(step[0] == first_step[0] and step[1] == first_step[1] for step in steps[1:]):
            return points

        repaired = list(points)
        step_vec = np.array([float(first_step[0]), float(first_step[1])], dtype=float)
        repaired[0] = repaired[0] - step_vec
        repaired[-1] = repaired[-1] + step_vec
        return repaired

    def _fit_axis_lines(points):
        """Return line-only segments for orthogonal/corner-like paths."""
        ints = [
            [int(round(p[0])), int(round(p[1]))]
            for p in points
        ]
        raw_ints = [list(p) for p in ints]
        is_closed_axis_loop = (
            len(ints) >= 4
            and np.linalg.norm(np.array(ints[0], dtype=float) - np.array(ints[-1], dtype=float)) <= 1.5
        )

        def _build_axis_runs(int_points):
            step_axes = []
            for i in range(1, len(int_points)):
                dr = int_points[i][0] - int_points[i - 1][0]
                dc = int_points[i][1] - int_points[i - 1][1]
                if dr == 0 and dc == 0:
                    step_axes.append(None)
                elif abs(dc) >= abs(dr):
                    step_axes.append("h")
                else:
                    step_axes.append("v")

            runs = []
            i = 0
            while i < len(step_axes):
                axis = step_axes[i]
                if axis is None:
                    i += 1
                    continue
                start = i
                i += 1
                while i < len(step_axes) and (step_axes[i] is None or step_axes[i] == axis):
                    i += 1
                end = i - 1
                runs.append({"axis": axis, "start": start, "end": end})
            return runs

        def _merge_closed_loop_runs(runs):
            if not runs:
                return runs

            merged = list(runs)
            if len(merged) >= 2 and merged[0]["axis"] == merged[-1]["axis"]:
                merged = [
                    {
                        "axis": merged[0]["axis"],
                        "start": merged[-1]["start"],
                        "end": merged[0]["end"],
                    }
                ] + merged[1:-1]

            changed = True
            while changed and len(merged) >= 3:
                changed = False
                for i in range(1, len(merged) - 1):
                    curr_len = merged[i]["end"] - merged[i]["start"] + 1
                    if curr_len > 2:
                        continue
                    if merged[i - 1]["axis"] != merged[i + 1]["axis"]:
                        continue

                    merged = (
                        merged[: i - 1]
                        + [{
                            "axis": merged[i - 1]["axis"],
                            "start": merged[i - 1]["start"],
                            "end": merged[i + 1]["end"],
                        }]
                        + merged[i + 2 :]
                    )
                    changed = True
                    break

            if len(merged) >= 2 and merged[0]["axis"] == merged[-1]["axis"]:
                merged = [
                    {
                        "axis": merged[0]["axis"],
                        "start": merged[-1]["start"],
                        "end": merged[0]["end"],
                    }
                ] + merged[1:-1]
            return merged

        def _closed_loop_runs_to_lines(run_defs, int_points):
            if not run_defs:
                return []

            support_lines = []
            for run in run_defs:
                run_points = _cyclic_run_points(run, int_points)
                if len(run_points) < 2:
                    support_lines = []
                    break

                if run["axis"] == "h":
                    row = float(sorted(float(point[0]) for point in run_points)[len(run_points) // 2])
                    support_lines.append({"axis": "h", "value": row})
                else:
                    col = float(sorted(float(point[1]) for point in run_points)[len(run_points) // 2])
                    support_lines.append({"axis": "v", "value": col})

            if len(support_lines) == len(run_defs):
                vertices = []
                for index, current in enumerate(support_lines):
                    previous = support_lines[index - 1]
                    if previous["axis"] == current["axis"]:
                        break

                    if previous["axis"] == "v" and current["axis"] == "h":
                        vertices.append([current["value"], previous["value"]])
                    elif previous["axis"] == "h" and current["axis"] == "v":
                        vertices.append([previous["value"], current["value"]])
                    else:
                        break
                else:
                    segs = []
                    for index, end_point in enumerate(vertices[1:] + vertices[:1]):
                        seg = {"type": "line", "end_point": _as_float_pair(end_point)}
                        if index == 0:
                            seg["start_point"] = _as_float_pair(vertices[0])
                        segs.append(seg)
                    if segs:
                        return segs

            segs = []
            point_count = len(int_points)
            for run in run_defs:
                end_point = int_points[(run["end"] + 1) % point_count]
                segs.append({"type": "line", "end_point": _as_float_pair(end_point)})

            if segs:
                segs[0]["start_point"] = _as_float_pair(int_points[run_defs[0]["start"]])
            return segs

        def _run_step_count(run_def, point_count):
            if point_count <= 0:
                return 0
            count = (int(run_def["end"]) - int(run_def["start"]) + 1) % point_count
            return count if count > 0 else point_count

        def _cyclic_run_points(run_def, int_points):
            point_count = len(int_points)
            if point_count == 0:
                return []

            step_count = _run_step_count(run_def, point_count)

            idx = int(run_def["start"]) % point_count
            points = [np.asarray(int_points[idx], dtype=float)]
            for _ in range(step_count):
                idx = (idx + 1) % point_count
                points.append(np.asarray(int_points[idx], dtype=float))
            return points

        def _line_line_intersection(center_a, direction_a, center_b, direction_b):
            cross = float((direction_a[0] * direction_b[1]) - (direction_a[1] * direction_b[0]))
            if abs(cross) <= 1e-6:
                return None

            delta = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
            t = float((delta[0] * direction_b[1]) - (delta[1] * direction_b[0])) / cross
            return np.asarray(center_a, dtype=float) + (t * np.asarray(direction_a, dtype=float))

        def _merge_short_connector_runs(run_defs, int_points):
            point_count = len(int_points)
            if len(run_defs) < 3 or point_count <= 0:
                return list(run_defs)

            merged = list(run_defs)
            changed = True
            while changed and len(merged) >= 3:
                changed = False
                for index in range(len(merged)):
                    curr = merged[index]
                    if _run_step_count(curr, point_count) > 2:
                        continue

                    rotated = merged[index - 1 :] + merged[: index - 1]
                    prev_run, curr_run, next_run = rotated[0], rotated[1], rotated[2]
                    prev_points = _cyclic_run_points(prev_run, int_points)
                    next_points = _cyclic_run_points(next_run, int_points)
                    if len(prev_points) < 2 or len(next_points) < 2:
                        continue

                    _, prev_direction = _fit_principal_line(prev_points)
                    _, next_direction = _fit_principal_line(next_points)
                    if abs(float(np.dot(prev_direction, next_direction))) < 0.96:
                        continue

                    merged = [
                        {
                            "axis": prev_run["axis"],
                            "start": prev_run["start"],
                            "end": next_run["end"],
                        }
                    ] + rotated[3:]
                    changed = True
                    break

            return merged

        def _closed_loop_best_fit_lines(run_defs, int_points):
            if not run_defs or len(int_points) < 4:
                return []

            run_defs = _merge_short_connector_runs(run_defs, int_points)

            fitted_runs = []
            for run in run_defs:
                run_points = _cyclic_run_points(run, int_points)
                if len(run_points) < 2:
                    return []

                center, direction = _fit_principal_line(run_points)
                if np.linalg.norm(direction) <= 1e-9:
                    return []

                fitted_runs.append({
                    "center": center,
                    "direction": direction,
                    "raw_corner": np.asarray(int_points[(run["end"] + 1) % len(int_points)], dtype=float),
                })

            if all(min(abs(float(run["direction"][0])), abs(float(run["direction"][1]))) <= 0.02 for run in fitted_runs):
                return []

            vertices = []
            for index in range(len(fitted_runs)):
                prev_run = fitted_runs[index - 1]
                curr_run = fitted_runs[index]
                intersection = _line_line_intersection(
                    prev_run["center"],
                    prev_run["direction"],
                    curr_run["center"],
                    curr_run["direction"],
                )
                raw_corner = prev_run["raw_corner"]
                if intersection is None:
                    intersection = raw_corner
                elif float(np.linalg.norm(intersection - raw_corner)) > max(3.0, fit_error * 3.0):
                    intersection = raw_corner
                vertices.append(intersection)

            segs = []
            for index, end_point in enumerate(vertices[1:] + vertices[:1]):
                seg = {"type": "line", "end_point": _as_float_pair(end_point)}
                if index == 0:
                    seg["start_point"] = _as_float_pair(vertices[0])
                segs.append(seg)
            return segs

        if is_closed_axis_loop:
            raw_closed = list(raw_ints)
            if len(raw_closed) >= 2 and raw_closed[0] == raw_closed[-1]:
                raw_closed = raw_closed[:-1]

            closed_runs = _merge_closed_loop_runs(_build_axis_runs(raw_closed + [raw_closed[0]]))
            if len(closed_runs) >= 4:
                best_fit_closed = _closed_loop_best_fit_lines(closed_runs, raw_closed)
                if best_fit_closed:
                    return best_fit_closed
                return _closed_loop_runs_to_lines(closed_runs, raw_closed)

        # For axis-dominant traces, smooth each dominant-axis run by snapping
        # the minor coordinate to the run median. This suppresses one-pixel
        # skeleton artifacts at starts/corners without forcing to raw vertices.
        if len(ints) >= 3:
            step_axes = []
            for i in range(1, len(ints)):
                dr = ints[i][0] - ints[i - 1][0]
                dc = ints[i][1] - ints[i - 1][1]
                if dr == 0 and dc == 0:
                    step_axes.append(None)
                elif abs(dc) >= abs(dr):
                    step_axes.append("h")
                else:
                    step_axes.append("v")

            runs = []
            i = 0
            while i < len(step_axes):
                axis = step_axes[i]
                if axis is None:
                    i += 1
                    continue
                start = i
                i += 1
                while i < len(step_axes) and (step_axes[i] is None or step_axes[i] == axis):
                    i += 1
                end = i - 1
                runs.append((axis, start, end))

            for axis, edge_start, edge_end in runs:
                point_start = edge_start
                point_end = edge_end + 1
                run_points = ints[point_start : point_end + 1]
                if len(run_points) < 2:
                    continue

                if axis == "h":
                    median_row = sorted(p[0] for p in run_points)[len(run_points) // 2]
                    for idx in range(point_start, point_end + 1):
                        ints[idx][0] = median_row
                else:
                    median_col = sorted(p[1] for p in run_points)[len(run_points) // 2]
                    for idx in range(point_start, point_end + 1):
                        ints[idx][1] = median_col

        # For near-1D paths (nearly horizontal or vertical), snap the minor
        # axis to its median value.  This removes staircase noise at skeleton
        # endpoints so that a horizontal/vertical line does not acquire a
        # spurious tilt caused by one or two pixels being off-centre.
        rows = [r for r, _ in ints]
        cols = [c for _, c in ints]
        row_ext = max(rows) - min(rows)
        col_ext = max(cols) - min(cols)
        if col_ext > row_ext * 5:
            # Predominantly horizontal — snap all rows to median row.
            median_row = sorted(rows)[len(rows) // 2]
            ints = [[median_row, c] for _, c in ints]
        elif row_ext > col_ext * 5:
            # Predominantly vertical — snap all cols to median col.
            median_col = sorted(cols)[len(cols) // 2]
            ints = [[r, median_col] for r, _ in ints]

        ints = [(p[0], p[1]) for p in ints]

        simplified = rdp_simplify(ints, max(0.8, fit_error * 1.2)) if len(ints) >= 3 else ints
        if len(simplified) < 2:
            simplified = [ints[0], ints[-1]]

        segs = [
            {"type": "line", "end_point": _as_float_pair(p)}
            for p in simplified[1:]
        ]
        # Expose the (possibly snapped) start so that the SVG M command uses
        # the corrected coordinate rather than the raw skeleton first point.
        if segs:
            segs[0]["start_point"] = _as_float_pair(simplified[0])
        return segs

    def _find_terminal_straight_run(points, at_start=True, min_steps=6, min_length=6.0):
        """Return the index bounding a straight run at an open path terminal."""
        if len(points) < (min_steps + 3):
            return None

        # Do not peel off terminal spans from traces that are globally close to
        # a single slanted line; that would fragment simple rotated lines.
        max_dev, mean_dev = _line_distance_stats(points)
        if max_dev <= max(1.1, fit_error * 1.45) and mean_dev <= max(0.4, fit_error * 0.55):
            return None
        if _max_turn_angle(points) <= 18.0:
            return None

        scan = points if at_start else list(reversed(points))
        ints = [
            np.array([int(round(p[0])), int(round(p[1]))], dtype=int)
            for p in scan
        ]

        direction = None
        run_end = None
        for i in range(1, len(ints)):
            step = ints[i] - ints[i - 1]
            if step[0] == 0 and step[1] == 0:
                continue
            if direction is None:
                direction = step
                run_end = i
                continue
            if step[0] == direction[0] and step[1] == direction[1]:
                run_end = i
                continue
            break

        if direction is None or run_end is None:
            return None

        if run_end < min_steps:
            return None
        if float(np.linalg.norm(scan[run_end] - scan[0])) < float(min_length):
            return None
        if run_end >= len(scan) - 2:
            return None

        return run_end if at_start else (len(points) - 1 - run_end)

    def _curve_is_effectively_line(curve):
        """Return True for axis-aligned cubics whose controls are also axis-aligned."""
        delta = curve[3] - curve[0]
        axis_tol = max(0.15, fit_error * 0.05)
        ctrl_tol = max(0.2, fit_error * 0.08)

        if abs(float(delta[0])) <= axis_tol:
            target_row = 0.5 * float(curve[0][0] + curve[3][0])
            return max(abs(float(curve[1][0] - target_row)), abs(float(curve[2][0] - target_row))) <= ctrl_tol

        if abs(float(delta[1])) <= axis_tol:
            target_col = 0.5 * float(curve[0][1] + curve[3][1])
            return max(abs(float(curve[1][1] - target_col)), abs(float(curve[2][1] - target_col))) <= ctrl_tol

        return False

    def _estimate_endpoint_tangent(points, at_start=True, lookahead=6):
        """Estimate a stable endpoint tangent from multiple forward/backward secants."""
        if len(points) < 2:
            return np.array([0.0, 0.0], dtype=float)

        anchor = points[0] if at_start else points[-1]
        # On longer paths, tiny endpoint hooks can dominate a short-window
        # estimate and flip the handle direction. Expand the secant window so
        # the tangent follows the broader path trend rather than a few noisy
        # terminal samples.
        adaptive_lookahead = max(int(lookahead), len(points) // 4)
        max_hops = min(adaptive_lookahead, len(points) - 1)
        accum = np.array([0.0, 0.0], dtype=float)
        weight_sum = 0.0

        for hop in range(1, max_hops + 1):
            if at_start:
                vec = points[hop] - anchor
            else:
                vec = anchor - points[-1 - hop]

            ln = np.linalg.norm(vec)
            if ln <= 1e-9:
                continue

            # Favor farther secants to suppress staircase noise near endpoints.
            weight = float(hop * hop)
            accum += (vec / ln) * weight
            weight_sum += weight

        if weight_sum > 0:
            tangent = _normalize(accum)
            if np.linalg.norm(tangent) > 1e-9:
                return tangent

        fallback = (points[1] - points[0]) if at_start else (points[-2] - points[-1])
        return _normalize(fallback)

    def _is_closed_loop(points):
        if len(points) < 6:
            return False
        return float(np.linalg.norm(points[0] - points[-1])) <= max(1.25, fit_error * 1.5)

    def _estimate_closed_loop_tangent(points, lookahead=8):
        """Estimate the tangent at a closed-loop seam using secants across it."""
        if len(points) < 4:
            return np.array([0.0, 0.0], dtype=float)

        max_hops = min(max(int(lookahead), len(points) // 8), max(1, (len(points) - 2) // 2))
        accum = np.array([0.0, 0.0], dtype=float)
        weight_sum = 0.0

        for hop in range(1, max_hops + 1):
            prev_pt = points[-1 - hop]
            next_pt = points[hop]
            vec = next_pt - prev_pt
            ln = np.linalg.norm(vec)
            if ln <= 1e-9:
                continue
            weight = float(hop * hop)
            accum += (vec / ln) * weight
            weight_sum += weight

        if weight_sum > 0:
            tangent = _normalize(accum)
            if np.linalg.norm(tangent) > 1e-9:
                return tangent

        return _normalize(points[1] - points[-2])

    def _align_endpoint_handles(curves, start_tangent, end_tangent, closed_loop_tangent=None):
        """Project first/last cubic handles onto stable endpoint tangents."""
        if not curves:
            return curves

        out = [c.copy() for c in curves]
        chord = np.linalg.norm(out[-1][3] - out[0][0])
        min_handle = max(0.35, 0.02 * float(chord))

        if closed_loop_tangent is not None and np.linalg.norm(closed_loop_tangent) > 1e-9:
            seam_tangent = _normalize(closed_loop_tangent)

            first = out[0]
            first_original = first[1].copy()
            first_handle_len = max(min_handle, float(np.linalg.norm(first[1] - first[0])))
            first_aligned = first[0] + seam_tangent * first_handle_len
            first[1] = (1.0 - tangent_blend) * first_original + tangent_blend * first_aligned

            last = out[-1]
            last_original = last[2].copy()
            last_handle_len = max(min_handle, float(np.linalg.norm(last[2] - last[3])))
            last_aligned = last[3] - seam_tangent * last_handle_len
            last[2] = (1.0 - tangent_blend) * last_original + tangent_blend * last_aligned

            return out

        if np.linalg.norm(start_tangent) > 1e-9:
            first = out[0]
            original = first[1].copy()
            handle_len = max(min_handle, float(np.linalg.norm(first[1] - first[0])))
            first_chord_tangent = _normalize(first[3] - first[0])
            stable_start_tangent = _normalize((0.35 * start_tangent) + (0.65 * first_chord_tangent))
            aligned = first[0] + stable_start_tangent * handle_len
            first[1] = (1.0 - tangent_blend) * original + tangent_blend * aligned

        if np.linalg.norm(end_tangent) > 1e-9:
            last = out[-1]
            original = last[2].copy()
            handle_len = max(min_handle, float(np.linalg.norm(last[2] - last[3])))
            last_chord_tangent = _normalize(last[3] - last[0])
            stable_end_tangent = _normalize((0.35 * end_tangent) + (0.65 * last_chord_tangent))
            # For the end handle, project backward from the endpoint so the
            # handle points back along the incoming path direction.
            aligned = last[3] - stable_end_tangent * handle_len
            last[2] = (1.0 - tangent_blend) * original + tangent_blend * aligned

        return out

    def _split_by_corners(points, corner_deg=58.0):
        if len(points) < 3:
            return [points]
        split_indices = [0]
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i - 1]
            v2 = points[i + 1] - points[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 <= 1e-9 or n2 <= 1e-9:
                continue
            cosang = np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
            angle = np.degrees(np.arccos(cosang))
            if angle >= corner_deg:
                split_indices.append(i)
        split_indices.append(len(points) - 1)

        chunks = []
        for a, b in zip(split_indices, split_indices[1:]):
            if b <= a:
                continue
            chunk = points[a : b + 1]
            if len(chunk) >= 2:
                chunks.append(chunk)
        return chunks if chunks else [points]

    def _bezier_eval(curve, t):
        mt = 1.0 - t
        return (
            (mt ** 3) * curve[0]
            + 3.0 * (mt ** 2) * t * curve[1]
            + 3.0 * mt * (t ** 2) * curve[2]
            + (t ** 3) * curve[3]
        )

    def _bezier_first_derivative(curve, t):
        mt = 1.0 - t
        return (
            3.0 * (mt ** 2) * (curve[1] - curve[0])
            + 6.0 * mt * t * (curve[2] - curve[1])
            + 3.0 * (t ** 2) * (curve[3] - curve[2])
        )

    def _bezier_second_derivative(curve, t):
        mt = 1.0 - t
        return (
            6.0 * mt * (curve[2] - 2.0 * curve[1] + curve[0])
            + 6.0 * t * (curve[3] - 2.0 * curve[2] + curve[1])
        )

    def _chord_length_parameterize(points):
        u = [0.0]
        for i in range(1, len(points)):
            u.append(u[-1] + float(np.linalg.norm(points[i] - points[i - 1])))
        total = u[-1]
        if total <= 1e-9:
            return np.linspace(0.0, 1.0, len(points))
        return np.array([v / total for v in u], dtype=float)

    def _generate_bezier(points, u, tan1, tan2):
        p0 = points[0]
        p3 = points[-1]

        C = np.zeros((2, 2), dtype=float)
        X = np.zeros(2, dtype=float)

        for i, ui in enumerate(u):
            b0 = (1.0 - ui) ** 3
            b1 = 3.0 * ui * ((1.0 - ui) ** 2)
            b2 = 3.0 * (ui ** 2) * (1.0 - ui)
            b3 = ui ** 3

            a1 = tan1 * b1
            a2 = tan2 * b2

            C[0, 0] += float(np.dot(a1, a1))
            C[0, 1] += float(np.dot(a1, a2))
            C[1, 0] += float(np.dot(a1, a2))
            C[1, 1] += float(np.dot(a2, a2))

            tmp = points[i] - (p0 * (b0 + b1)) - (p3 * (b2 + b3))
            X[0] += float(np.dot(a1, tmp))
            X[1] += float(np.dot(a2, tmp))

        det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
        seg_len = float(np.linalg.norm(p3 - p0))
        alpha_min = seg_len * 1e-3

        if abs(det) > 1e-10:
            alpha1 = (X[0] * C[1, 1] - X[1] * C[0, 1]) / det
            alpha2 = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
        else:
            alpha1 = alpha2 = seg_len / 3.0

        if alpha1 < alpha_min or alpha2 < alpha_min:
            alpha1 = alpha2 = seg_len / 3.0

        p1 = p0 + tan1 * alpha1
        p2 = p3 + tan2 * alpha2
        return np.array([p0, p1, p2, p3], dtype=float)

    def _reparameterize(points, u, curve):
        out = []
        for i, ui in enumerate(u):
            p = points[i]
            q = _bezier_eval(curve, ui)
            q1 = _bezier_first_derivative(curve, ui)
            q2 = _bezier_second_derivative(curve, ui)

            diff = q - p
            numerator = float(np.dot(diff, q1))
            denominator = float(np.dot(q1, q1) + np.dot(diff, q2))
            if abs(denominator) < 1e-12:
                out.append(ui)
            else:
                t = ui - numerator / denominator
                out.append(max(0.0, min(1.0, t)))
        return np.array(out, dtype=float)

    def _max_error(points, curve, u):
        max_err = -1.0
        split = len(points) // 2
        for i in range(1, len(points) - 1):
            q = _bezier_eval(curve, u[i])
            v = q - points[i]
            err = float(np.dot(v, v))
            if err > max_err:
                max_err = err
                split = i
        return max_err, split

    def _fit_cubic(points, tan1, tan2, error):
        n = len(points)
        if n == 2:
            dist = float(np.linalg.norm(points[1] - points[0])) / 3.0
            return [
                np.array(
                    [
                        points[0],
                        points[0] + tan1 * dist,
                        points[1] + tan2 * dist,
                        points[1],
                    ],
                    dtype=float,
                )
            ]

        u = _chord_length_parameterize(points)
        curve = _generate_bezier(points, u, tan1, tan2)
        max_err, split_idx = _max_error(points, curve, u)

        if max_err <= error * error:
            return [curve]

        if max_err <= (error * error) * 4.0:
            for _ in range(6):
                u = _reparameterize(points, u, curve)
                curve = _generate_bezier(points, u, tan1, tan2)
                max_err, split_idx = _max_error(points, curve, u)
                if max_err <= error * error:
                    return [curve]

        v_prev = points[split_idx - 1] - points[split_idx]
        v_next = points[split_idx] - points[split_idx + 1]
        tan_center = _normalize(v_prev + v_next)
        if np.linalg.norm(tan_center) <= 1e-9:
            tan_center = _normalize(points[split_idx + 1] - points[split_idx - 1])
        if np.linalg.norm(tan_center) <= 1e-9:
            tan_center = _normalize(points[-1] - points[0])

        left = _fit_cubic(points[: split_idx + 1], tan1, tan_center, error)
        right = _fit_cubic(points[split_idx:], -tan_center, tan2, error)
        return left + right

    def _fit_chunk(points):
        if len(points) < 2:
            return []

        is_closed_loop = _is_closed_loop(points)

        if not is_closed_loop:
            prefix_end = _find_terminal_straight_run(points, at_start=True)
            if prefix_end is not None:
                prefix = {
                    "type": "line",
                    "start_point": _as_float_pair(points[0]),
                    "end_point": _as_float_pair(points[prefix_end]),
                }
                suffix = _fit_chunk(points[prefix_end:])
                return [prefix] + suffix if suffix else [prefix]

            suffix_start = _find_terminal_straight_run(points, at_start=False)
            if suffix_start is not None:
                head = _fit_chunk(points[: suffix_start + 1])
                tail = {"type": "line", "end_point": _as_float_pair(points[-1])}
                return head + [tail] if head else [tail]

        # Enforce line-only output for Manhattan/corner-like traces.
        if force_orthogonal_as_lines and _axis_step_ratio(points) >= 0.82:
            return _fit_axis_lines(points)

        max_dev, mean_dev = _line_distance_stats(points)
        if max_dev <= max(1.0, fit_error * 1.35) and mean_dev <= max(0.5, fit_error * 0.65):
            return [{"type": "line", "end_point": _as_float_pair(points[-1])}]

        closed_loop_tangent = None
        if is_closed_loop:
            closed_loop_tangent = _estimate_closed_loop_tangent(points)
            tan1 = closed_loop_tangent
            tan2 = -closed_loop_tangent
        else:
            tan1 = _estimate_endpoint_tangent(points, at_start=True)
            tan2 = _estimate_endpoint_tangent(points, at_start=False)
        if np.linalg.norm(tan1) <= 1e-9:
            tan1 = _normalize(points[-1] - points[0])
        if np.linalg.norm(tan2) <= 1e-9:
            tan2 = -_normalize(points[-1] - points[0])

        curves = _fit_cubic(points, tan1, tan2, fit_error)
        curves = _align_endpoint_handles(curves, tan1, tan2, closed_loop_tangent=closed_loop_tangent)
        out = []
        for curve in curves:
            if _curve_is_effectively_line(curve):
                seg = {
                    "type": "line",
                    "end_point": _as_float_pair(curve[3]),
                }
            else:
                seg = {
                    "type": "cubic",
                    "control1": _as_float_pair(curve[1]),
                    "control2": _as_float_pair(curve[2]),
                    "end_point": _as_float_pair(curve[3]),
                }

            if out and out[-1].get("type") == "line" and seg.get("type") == "line":
                out[-1]["end_point"] = seg["end_point"]
            else:
                out.append(seg)
        return out

    def _segments_as_lines(segments, source_points):
        def _path_span_stats(points, start_idx, end_idx):
            if start_idx < 0 or end_idx >= len(points) or end_idx <= start_idx:
                return 0.0, float("inf")

            seg_pts = np.asarray(points[start_idx : end_idx + 1], dtype=float)
            seg_start = seg_pts[0]
            seg_end = seg_pts[-1]
            seg_delta = seg_end - seg_start
            seg_len = float(np.linalg.norm(seg_delta))
            if seg_len <= 1e-9:
                return 0.0, float("inf")

            max_dev = 0.0
            for sample in seg_pts[1:-1]:
                diff = sample - seg_start
                dist = abs(float((seg_delta[1] * diff[0]) - (seg_delta[0] * diff[1]))) / seg_len
                if dist > max_dev:
                    max_dev = dist
            return seg_len, max_dev

        def _expand_open_spans(span_defs, points):
            if not span_defs or len(points) < 4:
                return list(span_defs)

            expanded = [dict(span) for span in span_defs]
            last_index = len(points) - 1

            for index, span in enumerate(expanded):
                min_start = 0 if index == 0 else (expanded[index - 1]["end"] + 1)
                max_end = last_index if index == len(expanded) - 1 else (expanded[index + 1]["start"] - 1)
                dev_limit = 0.65 if index in (0, len(expanded) - 1) else 0.4

                best_start = int(span["start"])
                best_end = int(span["end"])
                changed = True
                while changed:
                    changed = False
                    candidates = []
                    if best_start > min_start:
                        candidates.append((best_start - 1, best_end))
                    if best_end < max_end:
                        candidates.append((best_start, best_end + 1))
                    if best_start > min_start and best_end < max_end:
                        candidates.append((best_start - 1, best_end + 1))

                    best_candidate = None
                    for cand_start, cand_end in candidates:
                        seg_len, max_dev = _path_span_stats(points, cand_start, cand_end)
                        if seg_len < 3.5 or max_dev > dev_limit:
                            continue

                        coverage = cand_end - cand_start
                        score = (coverage, seg_len - (2.0 * max_dev))
                        if best_candidate is None or score > best_candidate[0]:
                            best_candidate = (score, cand_start, cand_end)

                    if best_candidate is not None:
                        _, cand_start, cand_end = best_candidate
                        if cand_start != best_start or cand_end != best_end:
                            best_start = cand_start
                            best_end = cand_end
                            changed = True

                span["start"] = best_start
                span["end"] = best_end
                span["start_point"] = _as_float_pair(points[best_start])
                span["end_point"] = _as_float_pair(points[best_end])

            return expanded

        def _simplify_open_vertex_indices(vertex_indices, points, deviation_limit=0.6):
            simplified = [int(idx) for idx in vertex_indices]
            if len(simplified) < 3:
                return simplified

            changed = True
            while changed and len(simplified) >= 3:
                changed = False
                cursor = 1
                while cursor < len(simplified) - 1:
                    _, max_dev = _path_span_stats(points, simplified[cursor - 1], simplified[cursor + 1])
                    if max_dev <= deviation_limit:
                        simplified.pop(cursor)
                        changed = True
                    else:
                        cursor += 1
            return simplified

        def _extend_open_polyline_endpoints(vertices, extension_px=2.0):
            if len(vertices) < 2:
                return list(vertices)

            extended = [np.asarray(vertex, dtype=float).copy() for vertex in vertices]

            start_vec = extended[0] - extended[1]
            start_len = float(np.linalg.norm(start_vec))
            if start_len > 1e-9:
                extended[0] = np.round(extended[0] + ((start_vec / start_len) * float(extension_px)))

            end_vec = extended[-1] - extended[-2]
            end_len = float(np.linalg.norm(end_vec))
            if end_len > 1e-9:
                extended[-1] = np.round(extended[-1] + ((end_vec / end_len) * float(extension_px)))

            return [
                np.asarray(vertex, dtype=float)
                for vertex in extended
            ]

        start_point = source_points[0] if len(source_points) else np.asarray([0.0, 0.0], dtype=float)
        if not segments:
            p = _as_float_pair(start_point)
            return [{"type": "line", "start_point": p, "end_point": p}]

        has_non_line_segment = any(seg.get("type") != "line" for seg in segments)
        if has_non_line_segment and len(source_points) >= 4:
            source_closed_loop = float(np.linalg.norm(np.asarray(source_points[0], dtype=float) - np.asarray(source_points[-1], dtype=float))) <= max(1.5, fit_error * 1.5)
            circle_pts = np.asarray(source_points, dtype=float)
            if len(circle_pts) >= 4 and source_closed_loop:
                circle_pts = circle_pts[:-1]

            if len(circle_pts) >= 32:
                circle_center = np.mean(circle_pts, axis=0)
                circle_radii = np.linalg.norm(circle_pts - circle_center, axis=1)
                circle_radius_mean = float(np.mean(circle_radii))
                circle_radius_ratio = (
                    float(np.std(circle_radii)) / circle_radius_mean
                    if circle_radius_mean > 1e-9 else 1.0
                )

                if circle_radius_ratio <= 0.04:
                    circle_angles = np.unwrap(np.arctan2(circle_pts[:, 0] - circle_center[0], circle_pts[:, 1] - circle_center[1]))
                    if circle_angles[-1] < circle_angles[0]:
                        circle_angles = -circle_angles
                        circle_angles = np.unwrap(circle_angles)
                    circle_angles = circle_angles - circle_angles[0]
                    circle_angles_mod = np.mod(circle_angles, 2.0 * np.pi)

                    tangent_candidates = []
                    max_window = min(18, max(8, len(circle_pts) // 4))
                    for start_idx in range(len(circle_pts)):
                        max_end = min(start_idx + max_window, len(circle_pts) - 1)
                        for end_idx in range(start_idx + 4, max_end + 1):
                            seg_pts = circle_pts[start_idx : end_idx + 1]
                            seg_start = seg_pts[0]
                            seg_end = seg_pts[-1]
                            seg_delta = seg_end - seg_start
                            seg_len = float(np.linalg.norm(seg_delta))
                            if seg_len < 4.5:
                                continue

                            max_dev = 0.0
                            for sample in seg_pts[1:-1]:
                                diff = sample - seg_start
                                dist = abs(float((seg_delta[1] * diff[0]) - (seg_delta[0] * diff[1]))) / seg_len
                                if dist > max_dev:
                                    max_dev = dist
                            if max_dev > 0.45:
                                continue

                            mid_idx = (start_idx + end_idx) // 2
                            tangent_candidates.append({
                                "start": start_idx,
                                "end": end_idx,
                                "length": end_idx - start_idx + 1,
                                "deviation": max_dev,
                                "angle": float(circle_angles_mod[mid_idx]),
                                "start_point": _as_float_pair(seg_start),
                                "end_point": _as_float_pair(seg_end),
                                "score": seg_len + (0.35 * float(end_idx - start_idx + 1)) - (4.0 * max_dev),
                            })

                    if tangent_candidates:
                        intervals = sorted(tangent_candidates, key=lambda candidate: (candidate["end"], candidate["start"]))
                        interval_ends = [candidate["end"] for candidate in intervals]
                        previous_non_overlapping = []
                        for candidate in intervals:
                            previous_non_overlapping.append(
                                bisect.bisect_left(interval_ends, candidate["start"] - 1) - 1
                            )

                        best_scores = [0.0] * (len(intervals) + 1)
                        for index, candidate in enumerate(intervals, start=1):
                            take_score = candidate["score"] + best_scores[previous_non_overlapping[index - 1] + 1]
                            skip_score = best_scores[index - 1]
                            best_scores[index] = max(take_score, skip_score)

                        selected_spans = []
                        cursor = len(intervals)
                        while cursor > 0:
                            candidate = intervals[cursor - 1]
                            take_score = candidate["score"] + best_scores[previous_non_overlapping[cursor - 1] + 1]
                            if take_score > best_scores[cursor - 1]:
                                selected_spans.append(candidate)
                                cursor = previous_non_overlapping[cursor - 1] + 1
                            else:
                                cursor -= 1

                        if selected_spans:
                            selected_spans.reverse()
                            selected_spans = sorted(selected_spans, key=lambda candidate: candidate["start"])
                            circular_lines = []
                            for index, span in enumerate(selected_spans):
                                reuse_seg = {
                                    "type": "line",
                                    "end_point": span["end_point"],
                                }
                                if index == 0:
                                    reuse_seg["start_point"] = span["start_point"]
                                circular_lines.append(reuse_seg)

                                next_span = selected_spans[(index + 1) % len(selected_spans)]
                                if next_span["start_point"] != span["end_point"]:
                                    circular_lines.append({
                                        "type": "line",
                                        "end_point": next_span["start_point"],
                                    })

                            if circular_lines:
                                return circular_lines

            if not source_closed_loop and len(source_points) >= 12:
                open_pts = np.asarray(source_points, dtype=float)
                span_candidates = []
                max_window = min(14, max(6, len(open_pts) // 3))
                for start_idx in range(len(open_pts)):
                    max_end = min(start_idx + max_window, len(open_pts) - 1)
                    for end_idx in range(start_idx + 3, max_end + 1):
                        seg_pts = open_pts[start_idx : end_idx + 1]
                        seg_start = seg_pts[0]
                        seg_end = seg_pts[-1]
                        seg_delta = seg_end - seg_start
                        seg_len = float(np.linalg.norm(seg_delta))
                        if seg_len < 3.5:
                            continue

                        max_dev = 0.0
                        for sample in seg_pts[1:-1]:
                            diff = sample - seg_start
                            dist = abs(float((seg_delta[1] * diff[0]) - (seg_delta[0] * diff[1]))) / seg_len
                            if dist > max_dev:
                                max_dev = dist
                        if max_dev > 0.4:
                            continue

                        span_candidates.append({
                            "start": start_idx,
                            "end": end_idx,
                            "start_point": _as_float_pair(seg_start),
                            "end_point": _as_float_pair(seg_end),
                            "score": seg_len + (0.3 * float(end_idx - start_idx + 1)) - (4.0 * max_dev),
                        })

                if span_candidates:
                    intervals = sorted(span_candidates, key=lambda candidate: (candidate["end"], candidate["start"]))
                    interval_ends = [candidate["end"] for candidate in intervals]
                    previous_non_overlapping = []
                    for candidate in intervals:
                        previous_non_overlapping.append(
                            bisect.bisect_left(interval_ends, candidate["start"] - 1) - 1
                        )

                    best_scores = [0.0] * (len(intervals) + 1)
                    for index, candidate in enumerate(intervals, start=1):
                        take_score = candidate["score"] + best_scores[previous_non_overlapping[index - 1] + 1]
                        skip_score = best_scores[index - 1]
                        best_scores[index] = max(take_score, skip_score)

                    selected_spans = []
                    cursor = len(intervals)
                    while cursor > 0:
                        candidate = intervals[cursor - 1]
                        take_score = candidate["score"] + best_scores[previous_non_overlapping[cursor - 1] + 1]
                        if take_score > best_scores[cursor - 1]:
                            selected_spans.append(candidate)
                            cursor = previous_non_overlapping[cursor - 1] + 1
                        else:
                            cursor -= 1

                    if selected_spans:
                        selected_spans.reverse()
                        selected_spans = _expand_open_spans(selected_spans, open_pts)

                        vertex_indices = [0]
                        current_index = 0
                        for span in selected_spans:
                            span_start = int(span["start"])
                            span_end = int(span["end"])
                            if current_index != span_start:
                                vertex_indices.append(span_start)
                            vertex_indices.append(span_end)
                            current_index = span_end

                        if current_index != len(open_pts) - 1:
                            vertex_indices.append(len(open_pts) - 1)

                        vertex_indices = _simplify_open_vertex_indices(vertex_indices, open_pts, deviation_limit=0.6)
                        vertex_points = [np.asarray(open_pts[idx], dtype=float) for idx in vertex_indices]
                        vertex_points = _extend_open_polyline_endpoints(vertex_points, extension_px=2.0)

                        open_lines = []
                        for index, end_point in enumerate(vertex_points[1:]):
                            line_seg = {
                                "type": "line",
                                "end_point": _as_float_pair(end_point),
                            }
                            if index == 0:
                                line_seg["start_point"] = _as_float_pair(vertex_points[0])
                            open_lines.append(line_seg)

                        if open_lines:
                            return open_lines

            source_vertices = [_as_float_pair(p) for p in rdp_simplify(source_points, max(0.75, fit_error))]
            if len(source_vertices) >= 3:
                if source_closed_loop and float(np.linalg.norm(np.asarray(source_vertices[0], dtype=float) - np.asarray(source_vertices[-1], dtype=float))) <= max(1.5, fit_error * 1.5):
                    source_vertices = [source_vertices[-1]] + source_vertices[1:-1]

                if len(source_vertices) >= 3:
                    out = []
                    closed_source_loop = source_closed_loop
                    end_points = (source_vertices[1:] + source_vertices[:1]) if closed_source_loop else source_vertices[1:]
                    for index, end_point in enumerate(end_points):
                        line_seg = {
                            "type": "line",
                            "end_point": _as_float_pair(end_point),
                        }
                        if index == 0:
                            line_seg["start_point"] = _as_float_pair(source_vertices[0])
                        out.append(line_seg)
                    if out:
                        return out

        vertices = [_as_float_pair(start_point)]
        for index, seg in enumerate(segments):
            end_point = seg.get("end_point")
            if end_point is None:
                continue

            if index == 0 and seg.get("start_point") is not None:
                vertices[0] = _as_float_pair(seg.get("start_point", start_point))
            vertices.append(_as_float_pair(end_point))

        is_closed_loop = (
            len(vertices) >= 4
            and float(np.linalg.norm(np.asarray(vertices[0], dtype=float) - np.asarray(vertices[-1], dtype=float))) <= 1.5
        )

        if len(vertices) >= 3 and not is_closed_loop:
            simplified = rdp_simplify(vertices, max(0.35, fit_error * 0.85))
            if len(simplified) >= 2:
                vertices = [_as_float_pair(p) for p in simplified]

        if len(vertices) == 3 and not is_closed_loop and len(source_points) >= 6:
            start = np.asarray(vertices[0], dtype=float)
            corner = np.asarray(vertices[1], dtype=float)
            end = np.asarray(vertices[2], dtype=float)

            start_vec = start - corner
            end_vec = end - corner
            start_len = float(np.linalg.norm(start_vec))
            end_len = float(np.linalg.norm(end_vec))
            if start_len > 1e-9 and end_len > 1e-9:
                cosang = float(np.clip(np.dot(start_vec, end_vec) / (start_len * end_len), -1.0, 1.0))
                corner_angle = float(np.degrees(np.arccos(cosang)))
                if 75.0 <= corner_angle <= 105.0:
                    pts = np.asarray(source_points, dtype=float)
                    split_idx = min(range(len(pts)), key=lambda i: float(np.linalg.norm(pts[i] - corner)))
                    if split_idx >= 2 and (len(pts) - split_idx) >= 3:
                        left_pts = pts[: split_idx + 1]
                        right_pts = pts[split_idx:]
                        left_center, left_dir = _fit_principal_line(left_pts)
                        right_center, right_dir = _fit_principal_line(right_pts)
                        dot_dirs = float(np.clip(abs(np.dot(left_dir, right_dir)), -1.0, 1.0))
                        support_angle = float(np.degrees(np.arccos(dot_dirs)))

                        if 75.0 <= support_angle <= 105.0:
                            theta_left = float(np.arctan2(left_dir[1], left_dir[0]))
                            theta_right = float(np.arctan2(right_dir[1], right_dir[0]))
                            best_dirs = None
                            best_err = None
                            for sign in (1.0, -1.0):
                                alpha = 0.5 * (theta_left + (theta_right - sign * (np.pi / 2.0)))
                                ortho_left = np.array([np.cos(alpha), np.sin(alpha)], dtype=float)
                                ortho_right = np.array([-sign * np.sin(alpha), sign * np.cos(alpha)], dtype=float)
                                if float(np.dot(ortho_left, left_dir)) < 0.0:
                                    ortho_left = -ortho_left
                                if float(np.dot(ortho_right, right_dir)) < 0.0:
                                    ortho_right = -ortho_right

                                err = (
                                    abs(float(np.degrees(np.arctan2(np.sin(theta_left - np.arctan2(ortho_left[1], ortho_left[0])), np.cos(theta_left - np.arctan2(ortho_left[1], ortho_left[0]))))))
                                    + abs(float(np.degrees(np.arctan2(np.sin(theta_right - np.arctan2(ortho_right[1], ortho_right[0])), np.cos(theta_right - np.arctan2(ortho_right[1], ortho_right[0]))))))
                                )
                                if best_err is None or err < best_err:
                                    best_err = err
                                    best_dirs = (ortho_left, ortho_right)

                            if best_dirs is not None:
                                ortho_left, ortho_right = best_dirs
                                cross = float((ortho_left[0] * ortho_right[1]) - (ortho_left[1] * ortho_right[0]))
                                if abs(cross) > 1e-6:
                                    delta = right_center - left_center
                                    t = float((delta[0] * ortho_right[1]) - (delta[1] * ortho_right[0])) / cross
                                    target_corner = left_center + (t * ortho_left)

                                    chord_center = 0.5 * (start + end)
                                    chord_radius = 0.5 * float(np.linalg.norm(end - start))
                                    toward_target = target_corner - chord_center
                                    toward_norm = float(np.linalg.norm(toward_target))
                                    if chord_radius > 1e-9 and toward_norm > 1e-9:
                                        exact_corner = chord_center + (chord_radius * toward_target / toward_norm)
                                        if float(np.linalg.norm(exact_corner - corner)) <= max(2.0, fit_error * 2.0):
                                            vertices[1] = _as_float_pair(exact_corner)

        if len(vertices) < 2:
            p = _as_float_pair(start_point)
            return [{"type": "line", "start_point": p, "end_point": p}]

        out = []
        for index, end_point in enumerate(vertices[1:]):
            line_seg = {
                "type": "line",
                "end_point": _as_float_pair(end_point),
            }
            if index == 0:
                line_seg["start_point"] = _as_float_pair(vertices[0])
            out.append(line_seg)
        return out

    def _fit_principal_line(points):
        pts = np.asarray(points, dtype=float)
        center = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        direction = vh[0]
        endpoint_delta = pts[-1] - pts[0]
        if float(np.dot(direction, endpoint_delta)) < 0.0:
            direction = -direction
        return center, _normalize(direction)

    def _line_projection_t(point, center, direction):
        return float(np.dot(np.asarray(point, dtype=float) - center, direction))

    def _point_line_distance(point, center, direction):
        delta = np.asarray(point, dtype=float) - center
        projection = float(np.dot(delta, direction))
        residual = delta - (projection * direction)
        return float(np.linalg.norm(residual))

    def _recent_diagonal_carry_steps(points, at_start=True, window=8):
        if len(points) < 3:
            return []

        ints = [
            np.array([int(round(p[0])), int(round(p[1]))], dtype=int)
            for p in points
        ]
        step_slice = []
        if at_start:
            seq = zip(ints[:-1], ints[1:])
        else:
            seq = zip(ints[-window - 1 : -1], ints[-window:]) if len(ints) > window else zip(ints[:-1], ints[1:])

        for prev_pt, next_pt in seq:
            step = next_pt - prev_pt
            if step[0] == 0 and step[1] == 0:
                continue
            step_slice.append(step)

        candidates = []
        for step in (step_slice[:window] if at_start else step_slice[-window:]):
            if abs(int(step[0])) != 1 or abs(int(step[1])) != 1:
                continue
            outward = (-step) if at_start else step
            if not any(int(outward[0]) == int(existing[0]) and int(outward[1]) == int(existing[1]) for existing in candidates):
                candidates.append(outward.astype(float))
        return candidates

    def _refine_single_line_endpoints(points, line_segments):
        if len(line_segments) != 1 or line_segments[0].get("type") != "line":
            return line_segments
        if len(points) < 12:
            return line_segments

        max_dev, mean_dev = _line_distance_stats(points)
        if max_dev > max(1.8, fit_error * 1.8) or mean_dev > max(0.7, fit_error * 0.8):
            return line_segments

        center, direction = _fit_principal_line(points)
        if np.linalg.norm(direction) <= 1e-9:
            return line_segments

        refined = [dict(line_segments[0])]
        endpoints = {
            "start_point": np.asarray(refined[0].get("start_point", points[0]), dtype=float),
            "end_point": np.asarray(refined[0].get("end_point", points[-1]), dtype=float),
        }

        for key, at_start in (("start_point", True), ("end_point", False)):
            current = endpoints[key]
            current_dist = _point_line_distance(current, center, direction)
            if current_dist <= 0.45:
                continue

            current_t = _line_projection_t(current, center, direction)
            best = None
            best_dist = current_dist
            for step in _recent_diagonal_carry_steps(points, at_start=at_start, window=8):
                candidate = current + step
                candidate_t = _line_projection_t(candidate, center, direction)
                if at_start and candidate_t >= current_t - 0.25:
                    continue
                if not at_start and candidate_t <= current_t + 0.25:
                    continue

                candidate_dist = _point_line_distance(candidate, center, direction)
                if candidate_dist + 0.15 < best_dist:
                    best = candidate
                    best_dist = candidate_dist

            if best is not None and best_dist <= 0.85:
                endpoints[key] = best

        refined[0]["start_point"] = _as_float_pair(endpoints["start_point"])
        refined[0]["end_point"] = _as_float_pair(endpoints["end_point"])
        return refined

    result = []
    for path in paths:
        if len(path) < 2:
            if len(path) == 1:
                p = _as_point(path[0])
                result.append([{"type": "line", "end_point": _as_float_pair(p), "start_point": _as_float_pair(p)}])
            else:
                result.append([])
            continue

        cleaned = _clean_points(path)
        if len(cleaned) < 2:
            p = cleaned[0] if cleaned else np.array([0.0, 0.0], dtype=float)
            result.append([{"type": "line", "end_point": _as_float_pair(p), "start_point": _as_float_pair(p)}])
            continue

        cleaned = _suppress_terminal_line_hooks(cleaned)
        cleaned = _extend_uniform_straight_terminals(cleaned)

        chunks = _split_by_corners(cleaned)
        path_segments = []
        for chunk in chunks:
            chunk_segments = _fit_chunk(chunk)
            if chunk_segments:
                path_segments.extend(chunk_segments)

        if not path_segments:
            path_segments = [{"type": "line", "end_point": _as_float_pair(cleaned[-1])}]

        if not enable_curve_fitting:
            path_segments = _segments_as_lines(path_segments, cleaned)
            path_segments = _refine_single_line_endpoints(cleaned, path_segments)

        # Use the cleaned first-point as the path start, but only if a chunk
        # fitter (e.g. _fit_axis_lines) has not already set a snapped start.
        path_segments[0].setdefault("start_point", _as_float_pair(cleaned[0]))
        result.append(path_segments)

    return result
