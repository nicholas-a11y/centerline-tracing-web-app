#!/usr/bin/env python3
"""
Web-based Centerline Extraction Tool
====================================

A Flask web application for interactive centerline extraction with real-time parameter adjustment.
"""

import threading
import time
from queue import Queue
from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tempfile
import uuid

# Import our existing centerline extraction functions
from worksok3_optimized import (
    CircleEvaluationSystem, extract_skeleton_paths, merge_nearby_paths,
    optimize_path_with_circles, create_svg_output, remove_overlapping_paths,
    optimize_path_with_custom_params
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global storage for session data
sessions = {}

class CenterlineSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.image = None
        self.image_path = None
        self.original_filename = None
        self.results = None
        self.initial_paths = None  # Store raw magenta paths
        self.progress_queue = Queue()  # For progress updates
        self.optimization_thread = None
        self.optimization_complete = False
        self.optimization_stopped = False
        self.partial_optimized_paths = []  # Store partially optimized paths
        self.lock = threading.Lock()  # Lock for thread-safe access to partial_optimized_paths
        self.parameters = {
            'dark_threshold': 0.20,
            'rdp_tolerance': 2.0,      # Increase to 5.0-10.0 for fewer points
            'smoothing_factor': 0.005,  # Increase to 0.02-0.05 for fewer points
            'min_path_length': 3,      # Increase to 8-15 for longer segments
            'enable_optimization': True,   # Enable path optimization and circle evaluation
            'show_pre_optimization': False,  # Show unoptimized paths in SVG
            'include_image': False,    # Include original image in SVG background
        }

def auto_detect_min_path_length(gray_image, dark_threshold, test_lengths=None):
    """
    Automatically detect the best minimum path length by analyzing path distribution.
    
    Args:
        gray_image: Grayscale image (0-1 range)
        dark_threshold: Dark threshold to use for path extraction
        test_lengths: List of path lengths to test (default: [1, 3, 5, 8, 12, 16, 20])
    
    Returns:
        Best min path length and analysis results
    """
    if test_lengths is None:
        test_lengths = [1, 3, 5, 8, 12, 16, 20]
    
    print(f"Auto-detecting min path length using threshold {dark_threshold:.3f}...")
    
    # Extract initial skeleton paths
    try:
        initial_paths = extract_skeleton_paths(gray_image, dark_threshold, min_object_size=3)
        if len(initial_paths) == 0:
            return {
                'best_min_length': 3,
                'best_score': 0,
                'recommendation': 'no paths found - try adjusting threshold first'
            }
        
        # Merge nearby paths
        merged_paths = merge_nearby_paths(initial_paths, max_gap=25)
        
    except Exception as e:
        return {
            'best_min_length': 3,
            'best_score': 0,
            'recommendation': f'error in path extraction: {str(e)}'
        }
    
    # Analyze path length distribution
    path_lengths = [len(path) for path in merged_paths]
    path_lengths.sort(reverse=True)  # Longest first
    
    if len(path_lengths) == 0:
        return {
            'best_min_length': 3,
            'best_score': 0,
            'recommendation': 'no merged paths found'
        }
    
    # Calculate statistics
    total_paths = len(path_lengths)
    median_length = path_lengths[len(path_lengths) // 2] if path_lengths else 0
    
    print(f"  Found {total_paths} merged paths, lengths: {min(path_lengths)}-{max(path_lengths)}, median: {median_length}")
    
    best_min_length = test_lengths[0]
    best_score = 0
    length_results = []
    
    for min_length in test_lengths:
        # Filter paths by this minimum length
        valid_paths = [path for path in merged_paths if len(path) >= min_length]
        
        if len(valid_paths) == 0:
            score = 0
        else:
            # Score based on:
            # 1. Reasonable number of paths (not too few, not too many)
            # 2. Good average length of remaining paths
            # 3. Good coverage (total length of all paths)
            
            path_count = len(valid_paths)
            avg_length = sum(len(path) for path in valid_paths) / len(valid_paths)
            total_length = sum(len(path) for path in valid_paths)
            
            # Normalize scores
            count_score = min(path_count / max(total_paths * 0.3, 1), 1.0)  # Prefer keeping ~30% of paths
            length_score = min(avg_length / max(median_length * 1.5, 10), 1.0)  # Prefer good average length
            coverage_score = min(total_length / max(sum(path_lengths) * 0.7, 100), 1.0)  # Prefer good coverage
            
            # Penalty for being too strict (keeping too few paths)
            if path_count < max(total_paths * 0.1, 2):  # Less than 10% or less than 2 paths
                penalty = 0.5
            else:
                penalty = 1.0
            
            score = (count_score * 0.4 + length_score * 0.3 + coverage_score * 0.3) * penalty
        
        length_results.append({
            'min_length': min_length,
            'valid_paths': len(valid_paths) if 'valid_paths' in locals() else 0,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_min_length = min_length
        
        print(f"    Min length {min_length}: {len(valid_paths) if 'valid_paths' in locals() else 0} paths, score: {score:.3f}")
    
    # Generate recommendation
    if best_score > 0.3:
        recommendation = 'auto-detected'
    elif best_score > 0.1:
        recommendation = 'low confidence - consider manual adjustment'
    else:
        recommendation = 'manual adjustment recommended'
    
    print(f"  Best min path length: {best_min_length} (score: {best_score:.3f})")
    
    return {
        'best_min_length': best_min_length,
        'best_score': best_score,
        'all_results': length_results,
        'recommendation': recommendation,
        'path_stats': {
            'total_paths': total_paths,
            'median_length': median_length,
            'length_range': [min(path_lengths), max(path_lengths)] if path_lengths else [0, 0]
        }
    }

def auto_detect_dark_threshold(gray_image, sample_size=1000, threshold_range=(0.05, 0.8), num_thresholds=15):
    """
    Automatically detect the best dark threshold by sampling the image.
    
    Args:
        gray_image: Grayscale image (0-1 range)
        sample_size: Number of random pixels to sample for analysis
        threshold_range: (min, max) range of thresholds to test
        num_thresholds: Number of threshold values to test
    
    Returns:
        Best threshold value and analysis results
    """
    height, width = gray_image.shape
    
    # Sample random pixels from the image
    sample_indices = np.random.choice(height * width, min(sample_size, height * width), replace=False)
    sample_coords = [(idx // width, idx % width) for idx in sample_indices]
    sample_values = [gray_image[y, x] for y, x in sample_coords]
    
    # Test different threshold values
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    best_threshold = threshold_range[0]
    best_score = 0
    threshold_results = []
    
    print(f"Auto-detecting dark threshold from {len(sample_values)} sample pixels...")
    
    for threshold in thresholds:
        try:
            # Extract skeleton paths with this threshold
            initial_paths = extract_skeleton_paths(gray_image, threshold, min_object_size=3)
            
            if len(initial_paths) == 0:
                score = 0
            else:
                # Merge nearby paths
                merged_paths = merge_nearby_paths(initial_paths, max_gap=25)
                
                # Filter by minimum length
                valid_paths = [path for path in merged_paths if len(path) >= 3]
                
                # Score based on number of valid paths and total path length
                if len(valid_paths) == 0:
                    score = 0
                else:
                    total_length = sum(len(path) for path in valid_paths)
                    # Favor moderate number of paths with good total length
                    # Penalize too few paths (missing details) or too many paths (noise)
                    path_count_score = min(len(valid_paths) / 10.0, 1.0)  # Normalize to 0-1
                    length_score = min(total_length / 1000.0, 1.0)  # Normalize to 0-1
                    score = path_count_score * length_score
            
            threshold_results.append({
                'threshold': threshold,
                'score': score,
                'path_count': len(initial_paths) if 'initial_paths' in locals() else 0,
                'valid_count': len(valid_paths) if 'valid_paths' in locals() else 0
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
            print(f"  Threshold {threshold:.3f}: {len(initial_paths) if 'initial_paths' in locals() else 0} paths, score: {score:.3f}")
            
        except Exception as e:
            print(f"  Threshold {threshold:.3f}: Failed - {e}")
            threshold_results.append({
                'threshold': threshold,
                'score': 0,
                'path_count': 0,
                'valid_count': 0
            })
    
    print(f"Best threshold: {best_threshold:.3f} (score: {best_score:.3f})")
    
    return {
        'best_threshold': best_threshold,
        'best_score': best_score,
        'all_results': threshold_results,
        'recommendation': 'auto-detected' if best_score > 0 else 'manual adjustment needed'
    }

def load_and_process_image(image_path):
    """Load and convert image to grayscale."""
    from skimage import io, color
    
    img = io.imread(image_path)
    
    # Handle different image formats
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA image
            # Convert RGBA to RGB by removing alpha channel
            img = img[:, :, :3]
        elif img.shape[2] == 3:  # RGB image
            pass  # Already in correct format
        else:
            raise ValueError(f"Unsupported image format with {img.shape[2]} channels")
        
        # Convert RGB to grayscale
        gray = color.rgb2gray(img)
    elif len(img.shape) == 2:  # Already grayscale
        gray = img.astype(float) / 255.0 if img.dtype != float else img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    # Ensure values are in 0-1 range
    if gray.max() > 1.0:
        gray = gray / 255.0
    
    return gray

def process_centerlines(session):
    """Process centerlines with current parameters."""
    if session.image is None:
        return None
    
    params = session.parameters
    gray = session.image
    
    # Check if optimization is enabled to determine processing level
    optimization_enabled = params.get('enable_optimization', True)
    
    if optimization_enabled:
        # Full processing mode: use standard extraction
        initial_paths = extract_skeleton_paths(gray, params['dark_threshold'], min_object_size=5)
    else:
        # Fast mode: use the fast extraction function directly
        print("Fast extraction mode for unoptimized processing...")
        # Import the fast function from worksok3_optimized
        from worksok3_optimized import create_fast_paths
        import numpy as np
        from skimage import morphology
        
        # Quick binary conversion and skeletonization with minimal filtering
        binary = gray < params['dark_threshold']
        binary = morphology.remove_small_objects(binary, 2)  # Reduced for speed
        skeleton = morphology.skeletonize(binary)
        initial_paths = create_fast_paths(skeleton)
    
    if len(initial_paths) == 0:
        return {'error': 'No skeleton paths found. Try adjusting the dark threshold.'}
    
    if optimization_enabled:
        # Full processing mode: merge paths and handle overlaps
        
        # Merge nearby paths
        merged_paths = merge_nearby_paths(initial_paths, max_gap=25)
        
        # Filter paths by minimum length
        valid_paths = [path for path in merged_paths if len(path) >= params['min_path_length']]
        
        # Remove overlapping paths if enabled
        if params.get('remove_overlaps', True):
            valid_paths = remove_overlapping_paths(
                valid_paths, 
                overlap_threshold=params.get('overlap_threshold', 0.3),
                min_distance=params.get('min_overlap_distance', 8)
            )
        
        merged_paths_count = len(merged_paths)
    else:
        # Raw mode: no merging, no overlap processing - just filter by length
        print("Raw skeleton mode: skipping path merging and overlap processing...")
        
        # Filter paths by minimum length only
        valid_paths = [path for path in initial_paths if len(path) >= params['min_path_length']]
        merged_paths_count = len(initial_paths)  # Use initial count since no merging
    
    if len(valid_paths) == 0:
        return {'error': 'No valid paths after filtering. Try reducing minimum path length.'}
    
    # Check if optimization is enabled
    if not optimization_enabled:
        # Fast processing mode - skip circle evaluation and optimization
        print(f"Fast processing mode: returning {len(valid_paths)} unoptimized paths...")
        
        # Return unoptimized paths directly
        results = {
            'initial_paths_count': len(initial_paths),
            'merged_paths_count': merged_paths_count,
            'valid_paths_count': len(valid_paths),
            'stats': {'total_paths_evaluated': len(valid_paths)},
            'best_score': 0,  # No scoring in fast mode
            'optimized_paths': valid_paths,  # Use original paths as "optimized"
            'optimized_scores': [1.0] * len(valid_paths),  # Dummy scores
            'pre_optimization_paths': valid_paths,
            'circle_system': None  # No circle system in fast mode
        }
        
        return results

    # Initialize circle evaluation system with fixed parameters
    circle_system = CircleEvaluationSystem(
        gray, 
        params['dark_threshold'],
        3,    # max_circle_radius (fixed)
        0.2,  # min_circle_radius (fixed)
        2.0   # circle_intersection_bonus (fixed)
    )
    
    # Evaluate all paths
    path_scores, stats = circle_system.evaluate_all_paths(valid_paths)
    
    if len(path_scores) == 0:
        return {'error': 'No path scores computed.'}
    
    # Optimize ALL valid paths instead of limiting to top N
    sorted_indices = np.argsort(path_scores)[::-1]  # Best first
    
    optimized_paths = []
    optimized_scores = []
    
    print(f"Optimizing all {len(valid_paths)} valid paths...")
    
    for i, idx in enumerate(sorted_indices):
        path = valid_paths[idx]
        original_score = path_scores[idx]
        
        print(f"  Path {i+1}/{len(valid_paths)}:")
        # Apply custom parameters for optimization
        optimized_path, optimized_score = optimize_path_with_custom_params(
            path, circle_system, params
        )
        
        optimized_paths.append(optimized_path)
        optimized_scores.append(optimized_score)
    
    # Get corresponding pre-optimization paths for visualization (in same order)
    top_pre_optimization_paths = [valid_paths[idx] for idx in sorted_indices]
    
    # Prepare results
    results = {
        'initial_paths_count': len(initial_paths),
        'merged_paths_count': len(merged_paths),
        'valid_paths_count': len(valid_paths),
        'stats': stats,
        'best_score': optimized_scores[0] if optimized_scores else 0,
        'optimized_paths': optimized_paths,
        'optimized_scores': optimized_scores,
        'pre_optimization_paths': top_pre_optimization_paths,
        'circle_system': circle_system
    }
    
    return results

def optimize_path_with_custom_params(path, circle_system, params):
    """
    Optimize path with a conservative RDP pre-simplification followed by a high-quality spline fit.
    This approach prioritizes smoothness and fidelity over aggressive point reduction.
    """
    from worksok3_optimized import rdp_simplify, smooth_path_spline
    
    current_path = path
    best_score = circle_system.evaluate_path(current_path)
    
    print(f"    Optimizing path: {len(path)} points, initial score: {best_score:.2f}")
    
    # 1. Adaptive pre-simplification with curvature awareness
    import numpy as np

    def _remove_duplicate_points(points):
        """Drop consecutive duplicates so downstream math stays stable."""
        if not points:
            return points
        cleaned = [points[0]]
        for pt in points[1:]:
            if pt != cleaned[-1]:
                cleaned.append(pt)
        return cleaned

    def _resample_even_spacing(points):
        """Resample to even arc-length spacing to reduce pixel jitter."""
        if len(points) < 4:
            return points
        pts = np.asarray(points, dtype=float)
        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total = seg_len.sum()
        if total == 0:
            return points
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
                span = max(cumulative[j] - cumulative[j-1], 1e-6)
                ratio = (dist - cumulative[j-1]) / span
                interp = pts[j-1] + ratio * (pts[j] - pts[j-1])
            resampled.append((float(interp[0]), float(interp[1])))
        rounded = [(int(round(p[0])), int(round(p[1]))) for p in resampled]
        return _remove_duplicate_points(rounded)

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

    cleaned_path = _remove_duplicate_points(current_path)
    even_path = _resample_even_spacing(cleaned_path)

    # Estimate geometric complexity to tailor tolerance
    diffs = np.diff(np.asarray(even_path, dtype=float), axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1) if len(diffs) else np.array([0.0])
    total_length = float(seg_lengths.sum())
    mean_angle, sharp_ratio = _curvature_profile(even_path)

    base_tolerance = 1.2 + min(total_length / 150.0, 1.2)  # 1.2 -> 2.4 depending on length
    if sharp_ratio > 0.25 or mean_angle > 25.0:
        base_tolerance *= 0.65  # protect detailed curves
    adaptive_tolerance = max(1.05, min(base_tolerance, 2.4))

    try:
        simplified_path = rdp_simplify(even_path, adaptive_tolerance)
        if len(simplified_path) >= params.get('min_path_length', 3):
            rdp_score = circle_system.evaluate_path(simplified_path)
            print(
                f"      Adaptive RDP: {len(current_path)} -> {len(simplified_path)} points, "
                f"score: {rdp_score:.2f}, tolerance: {adaptive_tolerance:.2f}, "
                f"mean angle: {mean_angle:.1f}°, sharp ratio: {sharp_ratio:.2f}"
            )

            # Only accept if the structural quality is preserved.
            if rdp_score >= best_score * 0.985:
                current_path = simplified_path
                best_score = rdp_score
                print("      ✓ Accepted adaptive pre-simplification.")
            else:
                print("      ✗ Rejected RDP due to score drop.")
    except Exception as e:
        print(f"      RDP failed: {e}")
    
    # 2. Primary Optimization: High-Quality Spline Smoothing
    # This is now the main step for creating a smooth, fitted curve.
    smoothing_factor = params.get('smoothing_factor', 0.01)
    
    try:
        # Ensure there are enough points for spline fitting (k=3, so need at least 4 points)
        if len(current_path) >= 4:
            smoothed_path = smooth_path_spline(current_path, smoothing_factor)
            if len(smoothed_path) >= params.get('min_path_length', 3):
                smooth_score = circle_system.evaluate_path(smoothed_path)
                print(f"      Spline smooth: {len(current_path)} -> {len(smoothed_path)} points, score: {smooth_score:.2f}")
                
                # Accept the spline if its score is very high (at least 95% of the best score),
                # ensuring the smooth curve still accurately follows the centerline.
                if smooth_score > best_score * 0.95:
                    current_path = smoothed_path
                    best_score = smooth_score
                    print(f"      ✓ Accepted high-quality spline fit.")
    except Exception as e:
        print(f"      Spline smoothing failed: {e}")
    
    # 3. Polynomial fitting is removed for consistency and speed.
    
    print(f"    Final: {len(current_path)} points, score: {best_score:.2f}")
    return current_path, best_score

def create_dense_spline(path, target_points, smoothing_factor=0.005):
    """Create a dense spline with specified number of points for better circle alignment."""
    if len(path) < 4:
        return path
    
    from scipy import interpolate
    import numpy as np
    
    points = np.array(path)
    y, x = points[:, 0], points[:, 1]
    
    try:
        # Fit spline with fine smoothing
        tck, _ = interpolate.splprep([x, y], s=len(x) * smoothing_factor)
        
        # Generate many smooth points for better circle alignment
        u_new = np.linspace(0, 1, target_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        
        # Round to pixel coordinates
        result = [(round(y), round(x)) for y, x in zip(y_new, x_new)]
        
        # Remove consecutive duplicates
        filtered_result = [result[0]]
        for point in result[1:]:
            if point != filtered_result[-1]:
                filtered_result.append(point)
        
        return filtered_result
    except:
        return path

@app.route('/auto_detect_min_path_length', methods=['POST'])
def auto_detect_min_path_length_route():
    """Auto-detect the best minimum path length for the current session."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    if session.image is None:
        return jsonify({'error': 'No image loaded'})
    
    try:
        # Use current dark threshold for detection
        current_threshold = session.parameters.get('dark_threshold', 0.20)
        
        # Run auto-detection
        detection_result = auto_detect_min_path_length(session.image, current_threshold)
        
        if detection_result['best_score'] > 0:
            # Update session parameters with detected min path length
            session.parameters['min_path_length'] = detection_result['best_min_length']
            
            return jsonify({
                'success': True,
                'detected_min_length': detection_result['best_min_length'],
                'confidence_score': detection_result['best_score'],
                'recommendation': detection_result['recommendation'],
                'path_stats': detection_result.get('path_stats', {}),
                'updated_parameters': session.parameters
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not detect suitable min path length. Manual adjustment recommended.',
                'detected_min_length': detection_result['best_min_length'],
                'recommendation': detection_result['recommendation']
            })
            
    except Exception as e:
        return jsonify({'error': f'Auto-detection error: {str(e)}'})

@app.route('/auto_detect_threshold', methods=['POST'])
def auto_detect_threshold():
    """Auto-detect the best dark threshold for the current session."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    if session.image is None:
        return jsonify({'error': 'No image loaded'})
    
    try:
        # Run auto-detection
        detection_result = auto_detect_dark_threshold(session.image)
        
        if detection_result['best_score'] > 0:
            # Update session parameters with detected threshold
            session.parameters['dark_threshold'] = detection_result['best_threshold']
            
            return jsonify({
                'success': True,
                'detected_threshold': detection_result['best_threshold'],
                'confidence_score': detection_result['best_score'],
                'recommendation': detection_result['recommendation'],
                'updated_parameters': session.parameters
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not detect suitable threshold. Manual adjustment recommended.',
                'detected_threshold': detection_result['best_threshold'],
                'recommendation': detection_result['recommendation']
            })
            
    except Exception as e:
        return jsonify({'error': f'Auto-detection error: {str(e)}'})

@app.route('/')
def index():
    """Main page with the interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Create session
    session_id = str(uuid.uuid4())
    session = CenterlineSession(session_id)
    
    # Store original filename
    session.original_filename = file.filename
    
    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"centerline_{session_id}_{file.filename}")
    file.save(temp_path)
    
    try:
        # Load and process image
        gray = load_and_process_image(temp_path)
        session.image = gray
        session.image_path = temp_path
        
        # Store session
        sessions[session_id] = session
        
        # Convert image to base64 for display
        pil_img = Image.fromarray((gray * 255).astype(np.uint8))
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image_data': img_data,
            'image_shape': gray.shape,
            'parameters': session.parameters
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/process_immediate', methods=['POST'])
def process_immediate():
    """Immediately extract and display raw magenta paths, then start optimization in background."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    # Update parameters
    if 'parameters' in data:
        session.parameters.update(data['parameters'])
    
    try:
        # Quick raw path extraction
        params = session.parameters
        gray = session.image
        
        # Always use fast extraction for immediate display
        print("Immediate extraction mode...")
        from worksok3_optimized import create_fast_paths
        import numpy as np
        from skimage import morphology
        
        # Quick binary conversion and skeletonization
        binary = gray < params['dark_threshold']
        binary = morphology.remove_small_objects(binary, 2)
        skeleton = morphology.skeletonize(binary)
        initial_paths = create_fast_paths(skeleton)
        
        if len(initial_paths) == 0:
            return jsonify({'error': 'No skeleton paths found. Try adjusting the dark threshold.'})
        
        # Filter by minimum length
        valid_paths = [path for path in initial_paths if len(path) >= params['min_path_length']]
        
        if len(valid_paths) == 0:
            return jsonify({'error': 'No valid paths after filtering. Try reducing minimum path length.'})
        
        # Convert numpy coordinates to regular Python lists for JSON serialization
        json_serializable_paths = []
        for path in valid_paths:
            serializable_path = [[int(point[0]), int(point[1])] for point in path]
            json_serializable_paths.append(serializable_path)
        
        # Store raw paths for immediate display
        session.initial_paths = valid_paths  # Keep original for processing
        session.partial_optimized_paths = []
        session.optimization_complete = False
        session.optimization_stopped = False
        
        # Clear progress queue
        while not session.progress_queue.empty():
            session.progress_queue.get()
        
        # Create immediate results for magenta display
        immediate_results = {
            'initial_paths_count': len(initial_paths),
            'valid_paths_count': len(valid_paths),
            'paths': json_serializable_paths,  # Use JSON-serializable version
            'optimization_started': False
        }
        
        # Start optimization in background if enabled
        if params.get('enable_optimization', True):
            session.optimization_thread = threading.Thread(
                target=background_optimization,
                args=(session,)
            )
            session.optimization_thread.start()
            immediate_results['optimization_started'] = True
        
        return jsonify({
            'success': True,
            'immediate_display': True,
            **immediate_results
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

def background_optimization(session):
    """Run optimization in background thread."""
    try:
        params = session.parameters
        valid_paths = session.initial_paths
        
        session.progress_queue.put("Starting optimization process...")
        
        # Initialize circle evaluation system
        circle_system = CircleEvaluationSystem(
            session.image, 
            params['dark_threshold'],
            3,    # max_circle_radius
            0.2,  # min_circle_radius
            2.0   # circle_intersection_bonus
        )
        
        session.progress_queue.put("Circle evaluation system initialized...")
        
        # Evaluate paths
        session.progress_queue.put("Evaluating path quality...")
        path_scores, stats = circle_system.evaluate_all_paths(valid_paths)
        
        if len(path_scores) == 0:
            session.progress_queue.put("ERROR: No path scores computed")
            return
        
        # Sort paths by score
        sorted_indices = np.argsort(path_scores)[::-1]
        
        session.progress_queue.put(f"Optimizing {len(valid_paths)} paths...")
        
        # Optimize paths one by one with progress updates
        for i, idx in enumerate(sorted_indices):
            if session.optimization_stopped:
                session.progress_queue.put("Optimization stopped by user")
                break
                
            path = valid_paths[idx]
            original_score = path_scores[idx]
            
            session.progress_queue.put(f"Optimizing path {i+1}/{len(valid_paths)} ({len(path)} points, score: {original_score:.3f})")
            
            # Apply optimization with balanced parameters for quality and visible effect
            balanced_params = {
                'rdp_tolerance': 4.0,        # Balanced RDP simplification
                'smoothing_factor': 0.01,    # Moderate smoothing
                'min_path_length': 3         # Keep minimum path length
            }
            
            optimized_path, optimized_score = optimize_path_with_custom_params(
                path, circle_system, balanced_params
            )
            
            # Add to partial results - always add, even if not dramatically different
            with session.lock:
                session.partial_optimized_paths.append(optimized_path)
            
            # Show optimization results with dramatic reduction emphasis
            if len(optimized_path) != len(path):
                reduction_percent = ((len(path) - len(optimized_path)) / len(path)) * 100
                session.progress_queue.put(f"  → OPTIMIZED: {len(path)} → {len(optimized_path)} points ({reduction_percent:.1f}% reduction, score: {optimized_score:.3f})")
            else:
                session.progress_queue.put(f"  → Refined: {len(path)} points (score: {optimized_score:.3f})")
            
            # Update progress
            progress_percent = ((i + 1) / len(valid_paths)) * 100
            session.progress_queue.put(f"Progress: {progress_percent:.1f}% - {i+1}/{len(valid_paths)} paths completed")
            
            # Add small delay to make progress visible (optional)
            import time
            time.sleep(0.1)  # 100ms delay to see progress
        
        session.optimization_complete = True
        session.progress_queue.put("Optimization complete!")
        
    except Exception as e:
        session.progress_queue.put(f"Optimization error: {str(e)}")

@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Get optimization progress updates."""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    messages = []
    
    # Get all available progress messages
    while not session.progress_queue.empty():
        messages.append(session.progress_queue.get())
    
    return jsonify({
        'messages': messages,
        'optimization_complete': session.optimization_complete,
        'optimized_count': len(session.partial_optimized_paths),
        'total_paths': len(session.initial_paths) if session.initial_paths else 0
    })

@app.route('/stop_optimization', methods=['POST'])
def stop_optimization():
    """Stop optimization process early."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    session.optimization_stopped = True
    
    return jsonify({'success': True, 'message': 'Optimization stopping...'})

@app.route('/process', methods=['POST'])
def process():
    """Process centerlines with given parameters."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    
    # Update parameters
    if 'parameters' in data:
        session.parameters.update(data['parameters'])
        
        # Force show_pre-optimization to True when optimization is disabled
        if not session.parameters.get('enable_optimization', True):
            session.parameters['show_pre_optimization'] = True
    
    try:
        # Process centerlines
        results = process_centerlines(session)
        
        if results and 'error' not in results:
            session.results = results
            
            # Calculate point reduction statistics
            original_points = sum(len(path) for path in results.get('pre_optimization_paths', []))
            optimized_points = sum(len(path) for path in results.get('optimized_paths', []))
            reduction_percentage = ((original_points - optimized_points) / max(original_points, 1)) * 100 if original_points > 0 else 0
            
            # Prepare response data
            response = {
                'success': True,
                'stats': results['stats'],
                'initial_paths_count': results['initial_paths_count'],
                'merged_paths_count': results['merged_paths_count'],
                'valid_paths_count': results['valid_paths_count'],
                'best_score': results['best_score'],
                'paths_count': len(results['optimized_paths']),
                'original_points': original_points,
                'optimized_points': optimized_points,
                'point_reduction_percentage': round(reduction_percentage, 1)
            }
            
            return jsonify(response)
        else:
            return jsonify(results or {'error': 'Unknown processing error'})
            
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/test_svg/<session_id>')
def test_svg(session_id):
    """Test SVG generation route."""
    print(f"Test SVG route called for session: {session_id}")
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})
    
    session = sessions[session_id]
    print(f"Session found. Has initial_paths: {bool(session.initial_paths)}")
    if session.initial_paths:
        print(f"Number of initial paths: {len(session.initial_paths)}")
    
    return jsonify({'status': 'test successful', 'has_paths': bool(session.initial_paths)})

@app.route('/generate_svg', methods=['POST'])
def generate_svg():
    """Generate and return SVG visualization with support for progressive display."""
    data = request.json
    session_id = data.get('session_id')
    display_mode = data.get('mode', 'final')  # 'immediate', 'progressive', or 'final'
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'})

    session = sessions[session_id]
    
    # Check what data we have available
    if display_mode == 'immediate' and session.initial_paths:
        # Show only magenta paths immediately
        print(f"Generating immediate SVG with {len(session.initial_paths)} paths")
        paths_to_show = session.initial_paths
        optimized_paths = []
        circle_system = None
        pre_optimization_paths = paths_to_show
        
    elif display_mode == 'progressive':
        # Show magenta + current blue progress
        with session.lock:
            # Make a thread-safe copy of the paths to avoid issues during iteration
            optimized_paths = list(session.partial_optimized_paths)
        
        print(f"Generating progressive SVG with {len(optimized_paths)} optimized paths")
        paths_to_show = session.initial_paths
        circle_system = None  # Could add this later
        pre_optimization_paths = paths_to_show
        
    elif session.results:
        # Show final results
        print("Generating final SVG from session.results")
        results = session.results
        paths_to_show = results.get('pre_optimization_paths', [])
        optimized_paths = results.get('optimized_paths', [])
        circle_system = results.get('circle_system')
        pre_optimization_paths = paths_to_show
        
    else:
        return jsonify({'error': 'No results available yet'})

    try:
        temp_svg = f"/tmp/centerline_result_{session_id}.svg"
        
        # Set global variables for SVG generation
        import worksok3_optimized as wo
        original_output = wo.OUTPUT_PATH
        original_show_bitmap = wo.SHOW_BITMAP
        original_show_pre = wo.SHOW_PRE_OPTIMIZATION_PATHS
        
        wo.OUTPUT_PATH = temp_svg
        wo.SHOW_BITMAP = session.parameters.get('include_image', False)
        
        # Always show pre-optimization paths for immediate/progressive modes
        if display_mode in ['immediate', 'progressive']:
            wo.SHOW_PRE_OPTIMIZATION_PATHS = True
        else:
            wo.SHOW_PRE_OPTIMIZATION_PATHS = session.parameters.get('show_pre_optimization', False)
        
        # Generate SVG based on mode
        if display_mode == 'immediate':
            # Only magenta paths, no blue
            print("Creating immediate SVG with magenta paths only")
            create_svg_output(
                session.image,
                None,  # No circle system
                [],    # No optimized paths (no blue lines)
                [],    # No scores
                pre_optimization_paths  # Show magenta paths
            )
        else:
            # Progressive or final - show both magenta and blue
            print(f"Creating {display_mode} SVG with {len(optimized_paths)} blue paths and {len(pre_optimization_paths)} magenta paths")
            create_svg_output(
                session.image,
                circle_system,
                optimized_paths,  # Blue paths
                [1.0] * len(optimized_paths),  # Dummy scores
                pre_optimization_paths  # Magenta paths
            )
        
        # Restore original settings
        wo.OUTPUT_PATH = original_output
        wo.SHOW_BITMAP = original_show_bitmap
        wo.SHOW_PRE_OPTIMIZATION_PATHS = original_show_pre
        
        # Read and return SVG content
        if os.path.exists(temp_svg):
            with open(temp_svg, 'r') as f:
                svg_content = f.read()
            print(f"SVG file created successfully, size: {len(svg_content)} characters")
            os.remove(temp_svg)
            return jsonify({'svg': svg_content})
        else:
            print(f"SVG file not found at: {temp_svg}")
            return jsonify({'error': 'SVG generation failed - file not created'})
            
    except Exception as e:
        return jsonify({'error': f'SVG generation error: {str(e)}'})

@app.route('/download_svg/<session_id>')
def download_svg(session_id):
    """Download SVG file."""
    if session_id not in sessions:
        return "Invalid session", 404
    
    session = sessions[session_id]
    
    # Check for progressive data or traditional results
    has_progressive_data = session.initial_paths and len(session.initial_paths) > 0
    has_traditional_results = session.results and 'error' not in session.results
    
    if not has_progressive_data and not has_traditional_results:
        return "No valid results", 404
    
    try:
        # Create filename based on original upload
        original_filename = session.original_filename or 'centerline_extraction'
        # Remove extension and add _centerline.svg
        name_without_ext = os.path.splitext(original_filename)[0]
        download_filename = f"{name_without_ext}_centerline.svg"
        
        # Create temporary SVG file
        temp_svg = os.path.join(tempfile.gettempdir(), f"centerline_download_{session_id}.svg")
        
        # Generate SVG with user settings
        import worksok3_optimized as wo
        original_output = wo.OUTPUT_PATH
        original_show_bitmap = wo.SHOW_BITMAP
        original_show_pre = wo.SHOW_PRE_OPTIMIZATION_PATHS
        
        wo.OUTPUT_PATH = temp_svg
        wo.SHOW_BITMAP = session.parameters.get('include_image', False)  # User-controlled
        
        # Determine what to show based on available data
        if has_progressive_data:
            # Use progressive data
            print(f"Generating download SVG with progressive data: {len(session.initial_paths)} initial paths, {len(session.partial_optimized_paths)} optimized paths")
            wo.SHOW_PRE_OPTIMIZATION_PATHS = True  # Always show magenta paths
            
            create_svg_output(
                session.image,
                None,  # No circle system in progressive mode
                session.partial_optimized_paths,  # Blue optimized paths (may be empty if just started)
                [1.0] * len(session.partial_optimized_paths),  # Dummy scores
                session.initial_paths  # Magenta initial paths
            )
        else:
            # Use traditional results
            results = session.results
            show_pre = session.parameters.get('show_pre_optimization', False)
            optimization_enabled = session.parameters.get('enable_optimization', True)
            if not optimization_enabled:
                show_pre = True
            wo.SHOW_PRE_OPTIMIZATION_PATHS = show_pre
            
            if results['circle_system'] is not None:
                create_svg_output(
                    session.image,
                    results['circle_system'],
                    results['optimized_paths'],
                    results['optimized_scores'],
                    results['pre_optimization_paths']
                )
            else:
                create_svg_output(
                    session.image,
                    None,
                    results['optimized_paths'],
                    results['optimized_scores'],
                    results['pre_optimization_paths']
                )
        
        # Restore original settings
        wo.OUTPUT_PATH = original_output
        wo.SHOW_BITMAP = original_show_bitmap
        wo.SHOW_PRE_OPTIMIZATION_PATHS = original_show_pre
        
        return send_file(temp_svg, as_attachment=True, download_name=download_filename)
        
    except Exception as e:
        return f"Error generating download: {str(e)}", 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', 5002))
    
    print("Starting Centerline Extraction Web App...")
    print(f"Access the app at: http://localhost:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
