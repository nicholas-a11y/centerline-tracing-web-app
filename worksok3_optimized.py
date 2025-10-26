#!/usr/bin/env python3
"""
Optimized Circle-Based Centerline Extraction
============================================

This script uses only the circle-based evaluation method for centerline extraction.
All legacy methods (original centerline, KD-tree, red circles) have been removed.
"""

import os
import numpy as np
import networkx as nx
from skimage import io, color, filters, morphology
from skimage.morphology import binary_erosion, footprint_rectangle
from scipy import interpolate
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
INPUT_PATH = "/Users/nicholas/Downloads/Stage1.JPG"
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
    """Fast circle-based centerline evaluation system with optional ML enhancement."""
    
    def __init__(self, image, dark_threshold, max_radius=MAX_CIRCLE_RADIUS, min_radius=MIN_CIRCLE_RADIUS, intersection_bonus=CIRCLE_INTERSECTION_BONUS):
        self.image = image
        self.dark_threshold = dark_threshold
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.intersection_bonus = intersection_bonus
        
        # Find dark pixels
        self.dark_mask = image < dark_threshold
        self.dark_pixels = np.argwhere(self.dark_mask)
        
        if len(self.dark_pixels) == 0:
            print("WARNING: No dark pixels found with current threshold")
            self.circle_positions = []
            self.circle_radii = []
            self.intersection_weights = []
            return
        
        # Create circles at dark pixel locations
        self._create_circles()
        print(f"Created {len(self.circle_positions)} evaluation circles")
    
    def _create_circles(self):
        """Create circles at dark pixel locations with radius based on local darkness."""
        self.circle_positions = []
        self.circle_radii = []
        self.intersection_weights = []
        
        # Reduce max circles for faster processing
        max_circles = 2000  # Reduced from 5000 for speed
        if len(self.dark_pixels) > max_circles:
            step = len(self.dark_pixels) // max_circles
            sampled_pixels = self.dark_pixels[::step]
        else:
            sampled_pixels = self.dark_pixels
        
        for pos in sampled_pixels:
            y, x = pos
            intensity = self.image[y, x]
            
            # Map intensity to radius (darker = larger radius)
            normalized_darkness = (self.dark_threshold - intensity) / self.dark_threshold
            radius = self.min_radius + (self.max_radius - self.min_radius) * normalized_darkness
            
            self.circle_positions.append(pos)
            self.circle_radii.append(radius)
        
        # Calculate intersection weights
        self._calculate_intersection_weights()
    
    def _calculate_intersection_weights(self):
        """Calculate how much each circle intersects with others."""
        self.intersection_weights = []
        
        for i, (pos1, r1) in enumerate(zip(self.circle_positions, self.circle_radii)):
            intersections = 0
            for j, (pos2, r2) in enumerate(zip(self.circle_positions, self.circle_radii)):
                if i != j:
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < r1 + r2:  # Circles intersect
                        intersections += 1
            
            weight = 1.0 + intersections * self.intersection_bonus
            self.intersection_weights.append(weight)
    
    def evaluate_path(self, path):
        """Evaluate a path using circle-based scoring."""
        if len(path) < 2 or len(self.circle_positions) == 0:
            return 0.0
        
        # Use circle-based scoring only
        circle_score = self._evaluate_path_circles(path)
        return circle_score
    
    def _evaluate_path_circles(self, path):
        """Original circle-based evaluation method."""
        if len(path) < 2 or len(self.circle_positions) == 0:
            return 0.0
        
        total_score = 0.0
        path_array = np.array(path)
        
        # Sample points along the path (reduced for speed)
        path_length = len(path_array)
        sample_density = 0.3  # Reduced from 0.5 for faster processing
        num_samples = max(5, int(path_length * sample_density))  # Reduced minimum from 10
        
        # Create parameter values for interpolation
        distances = np.zeros(path_length)
        for i in range(1, path_length):
            distances[i] = distances[i-1] + np.linalg.norm(path_array[i] - path_array[i-1])
        
        if distances[-1] == 0:
            return 0.0
        
        # Normalize distances to [0, 1]
        normalized_distances = distances / distances[-1]
        
        # Sample points along the path
        sample_params = np.linspace(0, 1, num_samples)
        sample_points = []
        
        for param in sample_params:
            # Find the segment this parameter falls in
            segment_idx = np.searchsorted(normalized_distances, param)
            if segment_idx >= len(path_array):
                segment_idx = len(path_array) - 1
            elif segment_idx == 0:
                segment_idx = 1
            
            # Interpolate between the two points
            t = (param - normalized_distances[segment_idx-1]) / (normalized_distances[segment_idx] - normalized_distances[segment_idx-1])
            if np.isnan(t):
                t = 0
            
            point = path_array[segment_idx-1] * (1-t) + path_array[segment_idx] * t
            sample_points.append(point)
        
        sample_points = np.array(sample_points)
        
        # Score based on how well the path follows circles
        for sample_point in sample_points:
            for circle_pos, circle_radius, weight in zip(self.circle_positions, self.circle_radii, self.intersection_weights):
                distance = np.linalg.norm(sample_point - circle_pos)
                if distance <= circle_radius:
                    # Score based on how centered the point is in the circle
                    center_score = (circle_radius - distance) / circle_radius
                    total_score += center_score * weight * circle_radius
        
        # Normalize by path length to avoid bias toward longer paths
        return total_score / len(sample_points) if len(sample_points) > 0 else 0.0
    
    def evaluate_all_paths(self, paths):
        """Evaluate all paths and return scores and statistics."""
        scores = []
        for path in paths:
            score = self.evaluate_path(path)
            scores.append(score)
        
        if len(scores) == 0:
            return [], {}
        
        scores = np.array(scores)
        stats = {
            'mean': np.mean(scores),
            'max': np.max(scores),
            'min': np.min(scores),
            'std': np.std(scores)
        }
        
        return scores.tolist(), stats
    
    def get_visualization_data(self):
        """Get data for circle visualization."""
        return {
            'positions': self.circle_positions,
            'radii': self.circle_radii,
            'weights': self.intersection_weights
        }

def extract_skeleton_paths(image, dark_threshold=0.5, min_object_size=10):
    """Fast skeleton extraction for initial path candidates with maximum coverage."""
    print("Extracting skeleton paths...")
    
    # Use multiple threshold approaches for better coverage
    binary1 = image < dark_threshold
    binary2 = image < (dark_threshold * 1.2)  # Slightly more permissive threshold
    
    # Combine both thresholds for maximum coverage
    binary = binary1 | binary2
    
    # Remove only very small objects (reduced significantly)
    binary = morphology.remove_small_objects(binary, min_object_size)
    
    # Skip erosion entirely to preserve maximum structure
    # binary = binary_erosion(binary, footprint_rectangle((1, 1)))
    
    # Skeletonize
    skeleton = morphology.skeletonize(binary)
    
    # Use the most aggressive path extraction for maximum length
    paths = create_super_long_paths(skeleton)
    
    print(f"Extracted {len(paths)} initial skeleton paths")
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
    """Adaptively resample path based on circle importance."""
    if len(path) < 4:  # Reduced from 6 for better coverage
        return path
    
    path_array = np.array(path)
    target_points = max(MIN_PATH_LENGTH, int(len(path) * target_reduction))
    
    # Calculate importance of each point based on nearby circles
    importance_scores = []
    for point in path_array:
        importance = 0
        for circle_pos, circle_radius, weight in zip(circle_system.circle_positions, 
                                                   circle_system.circle_radii, 
                                                   circle_system.intersection_weights):
            distance = np.linalg.norm(point - circle_pos)
            if distance <= circle_radius * 1.5:  # Include nearby circles
                importance += weight * (circle_radius - min(distance, circle_radius)) / circle_radius
        importance_scores.append(importance)
    
    # Always keep first and last points
    resampled = [path_array[0]]
    
    # Select intermediate points based on importance
    if len(importance_scores) > 2:
        # Normalize importance scores
        importance_scores = np.array(importance_scores[1:-1])  # Exclude first/last
        if np.max(importance_scores) > 0:
            importance_scores = importance_scores / np.max(importance_scores)
        
        # Select points with highest importance
        intermediate_indices = np.argsort(importance_scores)[::-1][:target_points-2]
        intermediate_indices = sorted(intermediate_indices)
        
        for idx in intermediate_indices:
            resampled.append(path_array[idx + 1])  # +1 because we excluded first point
    
    resampled.append(path_array[-1])
    
    return [tuple(p) for p in resampled]

def merge_nearby_paths(paths, max_gap=30):
    """Merge paths that have endpoints close to each other for maximum coverage."""
    if len(paths) <= 1:
        return paths
    
    print(f"Merging nearby paths (max gap: {max_gap} pixels)...")
    
    merged_paths = []
    used_indices = set()
    
    for i, path1 in enumerate(paths):
        if i in used_indices or len(path1) < 2:
            continue
        
        current_path = list(path1)
        
        # Try to extend this path by connecting to other paths (multiple iterations)
        for iteration in range(3):  # Allow multiple merging passes
            extended = False
            
            for j, path2 in enumerate(paths):
                if j == i or j in used_indices or len(path2) < 2:
                    continue
                
                # Check all possible connections
                connections = [
                    ('end_to_start', np.linalg.norm(np.array(current_path[-1]) - np.array(path2[0]))),
                    ('end_to_end', np.linalg.norm(np.array(current_path[-1]) - np.array(path2[-1]))),
                    ('start_to_start', np.linalg.norm(np.array(current_path[0]) - np.array(path2[0]))),
                    ('start_to_end', np.linalg.norm(np.array(current_path[0]) - np.array(path2[-1])))
                ]
                
                # Find the best connection within the gap threshold
                best_connection = min(connections, key=lambda x: x[1])
                
                if best_connection[1] <= max_gap:
                    connection_type, distance = best_connection
                    
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
                    print(f"    Merged paths: {len(path1)} + {len(path2)} = {len(current_path)} points (gap: {distance:.1f})")
                    break
            
            if not extended:
                break  # No more paths to merge
        
        merged_paths.append(current_path)
        used_indices.add(i)
    
    print(f"  Before merging: {len(paths)} paths")
    print(f"  After merging: {len(merged_paths)} paths")
    
    # Show the lengths of the longest merged paths
    if merged_paths:
        path_lengths = [len(path) for path in merged_paths]
        path_lengths.sort(reverse=True)
        print(f"  Longest merged paths: {path_lengths[:5]}")
    
    return merged_paths

def create_svg_output(image, circle_system, optimized_paths, path_scores, pre_optimization_paths=None):
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
    
    # Add optimized paths
    print(f"Adding {len(optimized_paths)} optimized paths...")
    for i, (path, score) in enumerate(zip(optimized_paths, path_scores)):
        if len(path) < 2:
            continue
        
        # Color all paths blue with varying opacity based on rank
        color = "blue"
        width = 2.0
        opacity = 1.0 - (i * 0.1)  # Best path has opacity 1.0, others fade
        
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

def create_super_long_paths(skeleton):
    """Create the longest possible continuous paths by very aggressively following skeleton."""
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return []
    
    skeleton_points = set(zip(ys, xs))
    global_visited = set()
    paths = []
    
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
    
    def trace_ultra_long_path(start_point):
        """Trace the absolute longest path possible, going through branch points."""
        path = [start_point]
        local_visited = {start_point}
        current = start_point
        stuck_count = 0
        max_stuck = 3  # Allow some backtracking
        
        while stuck_count < max_stuck:
            all_neighbors = get_all_neighbors(current[0], current[1])
            
            # Prefer unvisited neighbors, but allow revisiting if necessary for longer paths
            unvisited_neighbors = [n for n in all_neighbors if n not in local_visited]
            
            if unvisited_neighbors:
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
                        for neighbor in unvisited_neighbors:
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
                
                # Fallback: choose any unvisited neighbor
                current = unvisited_neighbors[0]
                path.append(current)
                local_visited.add(current)
            
            elif all_neighbors:
                # All neighbors visited - try to find a way to continue by following visited paths
                stuck_count += 1
                
                # Look for neighbors that might lead to unvisited areas
                for neighbor in all_neighbors:
                    neighbor_neighbors = get_all_neighbors(neighbor[0], neighbor[1])
                    unvisited_from_neighbor = [n for n in neighbor_neighbors if n not in local_visited]
                    
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
        
        # Circle alignment features
        circle_hits = 0
        circle_center_distances = []
        circle_weights_sum = 0
        
        for point in path_array[::max(1, len(path_array)//10)]:  # Sample 10 points max
            for circle_pos, circle_radius, weight in zip(circle_system.circle_positions, 
                                                       circle_system.circle_radii, 
                                                       circle_system.intersection_weights):
                distance = np.linalg.norm(point - circle_pos)
                circle_center_distances.append(distance)
                if distance <= circle_radius:
                    circle_hits += 1
                    circle_weights_sum += weight
        
        features.extend([
            circle_hits,
            circle_weights_sum,
            np.mean(circle_center_distances) if circle_center_distances else 0,
            np.min(circle_center_distances) if circle_center_distances else 0,
        ])
        
        # Image intensity features along path
        intensities = []
        for point in path_array[::max(1, len(path_array)//20)]:  # Sample 20 points max
            y, x = int(point[0]), int(point[1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                intensities.append(image[y, x])
        
        if intensities:
            features.extend([
                np.mean(intensities),
                np.min(intensities),
                np.std(intensities),
            ])
        else:
            features.extend([0, 0, 0])
            
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
